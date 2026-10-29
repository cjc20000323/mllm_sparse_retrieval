from __future__ import annotations

import ast
import copy
import inspect
import io
import json
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import qwen_vl_prompts as prompts
from qwen_vl_encoding import NativeQwenEncoder, SparseEncoder, forward_views, isolate_views, prepare_views
import qwen_vl_pipeline as pipeline


@dataclass
class Data:
    dataset_name: str = "flickr"
    dataset_split: str = "test"
    prompt_type: str = "prompt_5"
    tbpr_type: str = "origin_type"
    cir_type: str = "type"
    random_view_seed: int = 42
    reps_loc: str = "after_pad"
    sparse_length: int = 8
    text_sparse_length: int = 8
    image_sparse_length: int = 8
    sparse_value_type: str = "sum"
    sparse_lower_or_upper: str = "lower"
    sparse_value_mean: bool = False
    sparse_manual: bool = True
    is_filtered: bool = False
    num_expended_tokens: int = 0
    per_device_batch_size: int = 2
    use_generated_validation_data: bool = False
    generated_validation_data_root: str = "unused"
    encode_is_query: bool = False
    dense_output_dir: str = "unused"
    sparse_output_dir: str = "unused"


CHAT_TEMPLATE = """{% if messages[0]['role'] != 'system' %}<|im_start|>system
You are a helpful assistant.<|im_end|>
{% endif %}{% for message in messages %}<|im_start|>{{ message['role'] }}
{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for item in message['content'] %}{% if item['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{% else %}{{ item['text'] }}{% endif %}{% endfor %}{% endif %}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""


def processor_fixture():
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import Qwen2TokenizerFast, Qwen2VLImageProcessor, Qwen2_5_VLProcessor
    backend = Tokenizer(models.BPE(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    specials = ["<unk>", "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"]
    trainer = trainers.BpeTrainer(vocab_size=520, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    backend.train_from_iterator([
        "Summary above sentence in one word: Summary the people or objects in above image in one word: " * 20,
        "user assistant system You are a helpful assistant. Yes No A B Image Sentence Person Candidate Query " * 20,
        "short caption a longer caption about people running in a park red blue green What is the caption? " * 20,
    ], trainer=trainer)
    tokenizer = Qwen2TokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<|endoftext|>",
                                  eos_token="<|im_end|>", additional_special_tokens=specials[2:], chat_template=CHAT_TEMPLATE)
    tokenizer.add_tokens(["Yes", "No", "A", "B"])
    kwargs = dict(tokenizer=tokenizer, chat_template=CHAT_TEMPLATE,
                  image_processor=Qwen2VLImageProcessor(patch_size=2, temporal_patch_size=1, merge_size=2,
                                                        min_pixels=64, max_pixels=144))
    if "video_processor" in inspect.signature(Qwen2_5_VLProcessor.__init__).parameters:
        from transformers.video_processing_utils import BaseVideoProcessor
        kwargs["video_processor"] = BaseVideoProcessor()
    return Qwen2_5_VLProcessor(**kwargs)


def model_fixture(processor):
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    tokenizer = processor.tokenizer
    config = Qwen2_5_VLConfig(
        text_config=dict(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2,
                         rope_scaling=dict(rope_type="default", mrope_section=[1, 1, 2])),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4, out_hidden_size=32,
                           patch_size=2, temporal_patch_size=1, spatial_merge_size=2, fullatt_block_indexes=[0], window_size=4),
        image_token_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        video_token_id=tokenizer.convert_tokens_to_ids("<|video_pad|>"),
        vision_start_token_id=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        vision_end_token_id=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(23)
    return Qwen2_5_VLForConditionalGeneration(config).eval()


class QwenNativeTests(unittest.TestCase):
    processor_factory = staticmethod(processor_fixture)
    model_factory = staticmethod(model_fixture)

    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(redirect_stderr(io.StringIO()))

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.processor = cls.processor_factory()
        cls.model = cls.model_factory(cls.processor)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def encoder(self, task="ir", concat=True, **kwargs):
        data = Data(**kwargs)
        args = SimpleNamespace(eol_type="disassembleeol_separate" if concat else "prompteol",
                               calculate_type="concat" if concat else "separate", model_name_or_path="renamed-checkpoint")
        return NativeQwenEncoder(self.model, self.processor, args, data, task)

    def test_packed_views_equal_independent_native_forward(self):
        for context, images in [("a longer caption\n", []), ("<image>\n", [Image.new("RGB", (8, 12), "red")]),
                                ("<image>\na question about people\n", [Image.new("RGB", (12, 8), "blue")])]:
            views = ["Summary above image in one word: ", "Summary the people in one word: ", "Summary the environment in one word: "]
            packed = prepare_views(self.processor, [context], [views], [images])
            actual, actual_hidden = forward_views(self.model, packed, isolated=True)
            for index, view in enumerate(views):
                separate = prepare_views(self.processor, [context], [[view]], [images])
                expected, expected_hidden = forward_views(self.model, separate, isolated=False)
                torch.testing.assert_close(actual[0][index], expected[0][0], atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(actual_hidden[0][index], expected_hidden[0][0], atol=2e-5, rtol=2e-5)

    def test_batch_padding_and_image_sizes_match_single_inputs(self):
        records = [dict(text="short caption"), dict(text="a much longer caption about people running in a park")]
        image_records = [dict(text="", image=Image.new("RGB", (8, 8), "red")),
                         dict(text="", image=Image.new("RGB", (12, 8), "blue"))]
        for concat in (False, True):
            for side in ("before_pad", "after_pad"):
                encoder = self.encoder(concat=concat, reps_loc=side)
                for batch in (records, image_records):
                    actual, dense = encoder.encode(batch)
                    for index, record in enumerate(batch):
                        expected, single_dense = encoder.encode([record])
                        torch.testing.assert_close(actual[index], expected[0], atol=3e-5, rtol=3e-5)
                        torch.testing.assert_close(dense[index], single_dense[0], atol=3e-5, rtol=3e-5)

    def test_changing_one_view_does_not_leak_into_other_views(self):
        views = ["Summary above sentence in one word: ", "Summary the people in one word: ", "Summary the environment in one word: "]
        batch = prepare_views(self.processor, ["a caption\n"], [views], [[]])
        expected, _ = forward_views(self.model, batch, isolated=True)
        changed = [views[0], "A completely different and much longer instruction about appearance: ", views[2]]
        actual, _ = forward_views(self.model, prepare_views(self.processor, ["a caption\n"], [changed], [[]]), isolated=True)
        torch.testing.assert_close(actual[0][[0, 2]], expected[0][[0, 2]], atol=2e-5, rtol=2e-5)
        mask = isolate_views(self.model, batch)["attention_mask"]
        first, second = batch.spans[0][:2]
        self.assertTrue((mask[0, 0, first[0]:first[1], first[1]:] < -1e20).all())
        self.assertTrue((mask[0, 0, second[0]:second[1], first[0]:first[1]] < -1e20).all())

    def test_all_task_prompt_paths_and_view_counts(self):
        for task, name, record in [
            ("ir", "flickr", dict(text="short caption")),
            ("tbpr", "CUHK-PEDES", dict(text="short caption")),
            ("t2it", "webqa", dict(text="question")),
            ("t2it", "webqa", dict(text="caption", image=Image.new("RGB", (8, 8)))),
            ("it2t", "remuq", dict(text="question", image=Image.new("RGB", (8, 8)))),
            ("it2t", "llava", dict(text="answer")),
            ("cir", "fashion-iq", dict(text="make it blue", image=Image.new("RGB", (8, 8)), composed=True, dress_type="shirt")),
        ]:
            encoder = self.encoder(task, dataset_name=name)
            sparse, dense = encoder.encode([record])
            self.assertEqual(sparse[0].shape, (5, len(self.processor.tokenizer)))
            self.assertEqual(dense.shape, (1, 32))
            self.assertNotIn("<tool_call>", encoder.last_prompts[0])
            self.assertTrue(torch.isfinite(sparse[0]).all())

    def test_required_mm_token_api_matches_native_encoding(self):
        """Exercise the new required arguments against real tiny native models.

        Inherited by Qwen2 and Qwen3 tests. The wrapper checks the new contract,
        then executes the installed model with explicit native position IDs.
        """
        native = self.model
        owner = native if hasattr(native, "get_rope_index") else native.model
        testcase = self

        class RequiredMmModel:
            config = native.config
            forward_calls = 0
            rope_calls = 0

            def get_input_embeddings(self):
                return native.get_input_embeddings()

            def check_types(self, input_ids, types):
                expected = torch.zeros_like(input_ids)
                expected[input_ids == self.config.image_token_id] = 1
                expected[input_ids == self.config.video_token_id] = 2
                torch.testing.assert_close(types, expected)

            def get_rope_index(self, input_ids, mm_token_type_ids, image_grid_thw=None,
                               video_grid_thw=None, attention_mask=None):
                self.rope_calls += 1
                self.check_types(input_ids, mm_token_type_ids)
                testcase.assertEqual(attention_mask.ndim, 2)
                kwargs = dict(input_ids=input_ids, image_grid_thw=image_grid_thw,
                              video_grid_thw=video_grid_thw, attention_mask=attention_mask)
                if "mm_token_type_ids" in inspect.signature(owner.get_rope_index).parameters:
                    kwargs["mm_token_type_ids"] = mm_token_type_ids
                return owner.get_rope_index(**kwargs)

            def forward(self, input_ids, mm_token_type_ids, **kwargs):
                self.forward_calls += 1
                self.check_types(input_ids, mm_token_type_ids)
                if kwargs.get("position_ids") is None:
                    kwargs["position_ids"] = self.get_rope_index(
                        input_ids, mm_token_type_ids, image_grid_thw=kwargs.get("image_grid_thw"),
                        video_grid_thw=kwargs.get("video_grid_thw"), attention_mask=kwargs["attention_mask"])[0]
                if "mm_token_type_ids" in inspect.signature(native.forward).parameters:
                    kwargs["mm_token_type_ids"] = mm_token_type_ids
                return native(input_ids=input_ids, **kwargs)

            __call__ = forward

        records = [dict(text="short caption"),
                   dict(text="", image=Image.new("RGB", (12, 8), "red")),
                   dict(text="a longer caption", image=Image.new("RGB", (8, 8), "blue"))]
        for concat in (False, True):
            for padding in ("before_pad", "after_pad"):
                for batch_records in (records[:1], records[1:], records):
                    with self.subTest(concat=concat, padding=padding, count=len(batch_records)):
                        baseline = self.encoder(concat=concat, reps_loc=padding)
                        expected, expected_dense = baseline.encode(batch_records)
                        adapted = self.encoder(concat=concat, reps_loc=padding)
                        adapted.model = required = RequiredMmModel()
                        # Both processor-provided IDs and the missing-ID fallback
                        # must work after expansion, including mixed padded rows.
                        for provided in (False, True):
                            prepared = adapted.prepare(batch_records)
                            if provided:
                                types = torch.zeros_like(prepared.inputs["input_ids"])
                                types[prepared.inputs["input_ids"] == native.config.image_token_id] = 1
                                prepared.inputs["mm_token_type_ids"] = types
                            else:
                                prepared.inputs.pop("mm_token_type_ids", None)
                            with patch.object(adapted, "prepare", return_value=prepared):
                                actual, dense = adapted.encode(batch_records)
                            for left, right in zip(actual, expected):
                                torch.testing.assert_close(left, right, atol=3e-5, rtol=3e-5)
                            torch.testing.assert_close(dense, expected_dense, atol=3e-5, rtol=3e-5)
                        self.assertEqual(required.forward_calls, 2)
                        self.assertEqual(required.rope_calls, 2)

    def test_prompt_counts_and_random_views(self):
        for number in range(1, 8):
            for kind in (dict(text="caption"), dict(text="", image=Image.new("RGB", (8, 8)))):
                encoder = self.encoder(prompt_type=f"prompt_{number}")
                self.assertEqual(len(encoder.prepare([kind]).spans[0]), number + 1)
        first = self.encoder(prompt_type="prompt_random_5").prepare([dict(text="caption")]).prompts
        second = self.encoder(prompt_type="prompt_random_5").prepare([dict(text="caption")]).prompts
        self.assertEqual(first, second)

    def test_native_rerank_matrix_body_matches_mistral(self):
        for direction in ("image", "text"):
            for mode in list(prompts.IR_POINTWISE) + list(prompts.DIRECTED):
                value = prompts.rerank_template(self.processor, "ir", direction, mode)
                self.assertIn("<|im_start|>user", value)
                self.assertEqual(value.count(prompts.IMAGE), 1)
                self.assertNotIn("[INST]", value)
        for task in ("ir", "tbpr", "t2it", "it2t"):
            value = prompts.rerank_template(self.processor, task, "text", "what_caption_generation", caption=True)
            self.assertIn("What is", value)
            self.assertIn("person image" if task == "tbpr" else "query question" if task == "t2it" else "answer" if task == "it2t" else "caption", value)
        for mode in ("caption_generation_6", "caption_generation_7"):
            with self.assertRaisesRegex(ValueError, "BOS"):
                prompts.rerank_template(self.processor, "ir", "image", mode, caption=True)
        with self.assertRaisesRegex(ValueError, "No Mistral/Qwen counterpart"):
            prompts.rerank_template(self.processor, "ir", "image", "typo")
        with self.assertRaisesRegex(ValueError, "no implemented scoring"):
            prompts.rerank_template(self.processor, "cir", "composed", "caption_generation", caption=True)

    def test_reverse_order_and_labels_preserve_baseline(self):
        normal = prompts.rerank_template(self.processor, "ir", "text", "query_relevant")
        reverse = prompts.rerank_template(self.processor, "ir", "text", "reverse_query_relevant")
        self.assertLess(normal.index("<sent>"), normal.index(prompts.IMAGE))
        self.assertLess(reverse.index(prompts.IMAGE), reverse.index("<sent>"))

    def test_sparse_invalid_ids_do_not_shift_weights(self):
        tokenizer = SimpleNamespace(get_vocab=lambda: {"good": 0, "Ġword": 2, "outside": 100, "<special>": 4})
        encoder = SparseEncoder(tokenizer, Data(is_filtered=True), concat=True)
        vector = encoder.vector(torch.tensor([[1.0, 100.0, 2.0, 100.0, 100.0], [3.0, 100.0, 1.0, 100.0, 100.0]]), dict(text="", image="image"))
        self.assertEqual(vector, {"good": 400, "word": 300})

    def test_metadata_rejects_old_index_or_changed_prompt(self):
        args = self.encoder().model_args
        training = SimpleNamespace(task_type="ir")
        expected = pipeline.prompt_metadata(args, Data(), training, self.processor, "text")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "rebuild"):
                pipeline.validate_manifest(directory, expected)
            (Path(directory) / pipeline.manifest_name(expected)).write_text(json.dumps(expected), encoding="utf-8")
            self.assertEqual(pipeline.validate_manifest(directory, expected), expected)
            changed = pipeline.prompt_metadata(args, Data(prompt_type="prompt_3"), training, self.processor, "text")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                pipeline.validate_manifest(directory, changed)
            self.processor.tokenizer.padding_side = "right"
            self.assertEqual(pipeline.prompt_metadata(args, Data(), training, self.processor, "text")["fingerprint"], expected["fingerprint"])
            with patch.object(self.processor, "chat_template", self.processor.chat_template + " "):
                changed = pipeline.prompt_metadata(args, Data(), training, self.processor, "text")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                pipeline.validate_manifest(directory, changed)

    def test_config_detects_renamed_checkpoints_without_name_matching(self):
        with tempfile.TemporaryDirectory(prefix="custom-weights-") as directory:
            (Path(directory) / "config.json").write_text('{"model_type":"qwen2_5_vl"}', encoding="utf-8")
            self.assertTrue(prompts.is_qwen25_checkpoint(directory))
            (Path(directory) / "config.json").write_text('{"model_type":"llava_next"}', encoding="utf-8")
            self.assertFalse(prompts.is_qwen25_checkpoint(directory))

    def test_alpha_is_selected_on_validation_not_test_queries(self):
        calls = []
        class Dataset:
            data_name = "flickr"
            img_id_list = ["i0", "i1"]
            images = [Image.new("RGB", (8, 8)), Image.new("RGB", (12, 8))]
            def __getitem__(self, index):
                return "", self.images[index], f"t{index}", self.img_id_list[index]
            def get_target(self, identifier, direction):
                return {"i0": ["t0"], "i1": ["t1"]}[identifier]
        class Index:
            def __init__(self, dense, sparse, expected, search, rank):
                calls.append((dense, sparse, expected["config"]["split"]))
            def close(self):
                pass
            def retrieve(self, encoder, records, data):
                dense, sparse = {}, {}
                for record in records:
                    correct = "t0" if record["id"] == "i0" else "t1"
                    wrong = "t1" if correct == "t0" else "t0"
                    dense[record["id"]] = pipeline._run({wrong: 2.0, correct: 1.0})
                    sparse[record["id"]] = pipeline._run({correct: 2.0, wrong: 1.0})
                return dense, sparse
        with tempfile.TemporaryDirectory() as root:
            args = self.encoder().model_args
            training = SimpleNamespace(task_type="ir", output_dir=root)
            data = Data(use_generated_validation_data=True)
            search = SimpleNamespace(query_type="image", use_fixed_alpha=False, alpha=0.7, depth=2,
                                     passage_reps="test-dense", sparse_index="test-sparse", val_passage_reps="val-dense",
                                     val_sparse_index="val-sparse", validation_dataset_name=None, use_fixed_validation_dataset=False)
            with patch.object(pipeline, "load_runtime", return_value=(self.model, self.processor, 0, 1)), patch.object(pipeline, "build_dataset", return_value=Dataset()) as build, patch.object(pipeline, "CorpusSearch", Index), patch("pandas.DataFrame.to_excel"):
                pipeline.run_search(args, data, search, training)
            self.assertEqual(calls, [("test-dense", "test-sparse", "test"), ("val-dense", "val-sparse", "val")])
            self.assertTrue(build.call_args.kwargs["generated"])
            self.assertEqual(json.loads((Path(root) / "recall_metrics.json").read_text())["alpha"], 0.1)

    def test_missing_validation_selects_on_target_with_one_native_forward(self):
        calls = []
        class Dataset:
            data_name = "flickr"
            img_id_list = ["i0", "i1"]
            images = [Image.new("RGB", (8, 8)), Image.new("RGB", (12, 8))]
            def __getitem__(self, index):
                return "", self.images[index], f"t{index}", self.img_id_list[index]
            def get_target(self, identifier, direction):
                return ["t0" if identifier == "i0" else "t1"]
        class Index:
            def __init__(self, dense, sparse, expected, search, rank):
                calls.append(expected["config"]["split"])
            def close(self):
                pass
            def retrieve(self, encoder, records, data):
                encoder.encode(records)  # A real batched visual Qwen forward.
                dense, sparse = {}, {}
                for record in records:
                    correct = "t0" if record["id"] == "i0" else "t1"
                    wrong = "t1" if correct == "t0" else "t0"
                    dense[record["id"]] = pipeline._run({wrong: 2.0, correct: 1.0})
                    sparse[record["id"]] = pipeline._run({correct: 2.0, wrong: 1.0})
                return dense, sparse
        with tempfile.TemporaryDirectory() as root:
            training = SimpleNamespace(task_type="ir", output_dir=root)
            search = SimpleNamespace(query_type="image", use_fixed_alpha=False, alpha=0.9, depth=2,
                                     passage_reps="test-dense", sparse_index="test-sparse",
                                     val_passage_reps=None, val_sparse_index=None)
            with patch.object(pipeline, "load_runtime", return_value=(self.model, self.processor, 0, 1)), \
                    patch.object(pipeline, "build_dataset", return_value=Dataset()) as build, \
                    patch.object(pipeline, "CorpusSearch", Index), patch("pandas.DataFrame.to_excel"), \
                    patch.object(self.model, "forward", wraps=self.model.forward) as forward:
                pipeline.run_search(self.encoder().model_args, Data(use_generated_validation_data=True), search, training)
            self.assertEqual(forward.call_count, 1)
            self.assertEqual(build.call_count, 1)
            self.assertNotIn("generated", build.call_args.kwargs)
            self.assertEqual(calls, ["test"])
            report = json.loads((Path(root) / "recall_metrics.json").read_text())
            self.assertEqual(report["alpha"], 0.1)
            self.assertEqual(report["alpha_selection"], dict(source="target", dataset="flickr", split="test"))

    def test_equal_cross_modal_ids_are_kept_but_cir_reference_is_removed(self):
        index = pipeline.CorpusSearch.__new__(pipeline.CorpusSearch)
        index.lookup = ["0", "1"]
        index.dense = SimpleNamespace(search=lambda reps, depth: (np.array([[1.0, 0.5]]), np.array([[0, 1]])))
        index.sparse = None
        index.search = SimpleNamespace(depth=2, remove_query=True)
        for reference in (False, True):
            record = dict(id="0", native_id=0, text="caption")
            if reference:
                record["reference_id"] = "0"
            with patch.object(pipeline, "encode_batches", return_value=iter([([record], np.ones((1, 32)), [{}])])):
                dense, _ = index.retrieve(None, [record], Data())
            self.assertEqual("0" in dense["0"]["docs"], not reference)

    def test_entrypoints_route_qwen_before_legacy_model_loading(self):
        for filename, target in [("encode.py", "run_encode"), ("search.py", "run_search"), ("rerank.py", "run_search")]:
            tree = ast.parse((SRC / filename).read_text(encoding="utf-8"))
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == ("_run_rerank" if filename == "rerank.py" else "main"))
            branch = next(node for node in function.body if isinstance(node, ast.If) and "is_native_qwen_checkpoint" in ast.unparse(node.test))
            self.assertIn(f"return {target}(", ast.unparse(branch))

    def test_supported_task_mode_matrix_runs_native_scoring(self):
        image = Image.new("RGB", (8, 8), "red")
        query = [dict(id="q", native_id="q", text="query text", image=image,
                      dress_type="shirt", composed=True)]
        candidates = [dict(id="c", text="caption", image=image)]
        ir = list(prompts.IR_POINTWISE) + list(prompts.DIRECTED)
        pointwise = {
            "ir": ir,
            "tbpr": list(ir),
            "t2it": ["relevant", "old_relevant", "query_relevant", "origin_old_relevant"],
            "it2t": ["relevant", "old_relevant", "query_relevant", "origin_old_relevant", "answer_relevant"],
            "cir": ["relevant", "old_relevant", "query_relevant", "origin_old_relevant", "please_relevant",
                    "in_one_word_relevant", "role_relevant", "role_old_relevant"],
        }
        captions = {
            "ir": list(prompts.CAPTION_SUFFIXES) + ["detailed_caption_generation", "detailed_caption_generation_1"],
            "tbpr": ["caption_generation", "what_caption_generation", "describe_caption_generation"],
            "t2it": ["caption_generation", "what_caption_generation"],
            "it2t": ["caption_generation", "what_caption_generation"],
            "cir": [],
        }
        for task, direction in [("ir", "image"), ("ir", "text"), ("tbpr", "text"),
                                ("t2it", "text"), ("it2t", "composed"), ("cir", "composed")]:
            for mode in pointwise[task] + captions[task]:
                search = SimpleNamespace(query_type=direction, rerank_type="pointwise", rerank_template=mode,
                                         rerank_num=1, rerank_batch_size=1, modify_type="no", use_rerank_prior_cache=False)
                with self.subTest(task=task, direction=direction, mode=mode), \
                        patch.object(pipeline, "dataset_records", return_value=candidates):
                    result = pipeline.rerank_records(
                        self.model, self.processor, SimpleNamespace(data_name="flickr"), query,
                        {"q": {"c": 1.0}}, task, self.encoder().model_args, Data(), search,
                        SimpleNamespace(task_type=task, fp16=False, bf16=False), 0, 1)
                    self.assertTrue(torch.isfinite(torch.tensor(result["q"]["c"])))

    def test_real_qwen_reranking_all_tasks_and_padding_sides(self):
        image = Image.new("RGB", (8, 8), "red")
        args = SimpleNamespace(model_name_or_path="renamed-checkpoint")
        for task, direction in [("ir", "image"), ("ir", "text"), ("tbpr", "text"), ("t2it", "text"), ("it2t", "image"), ("cir", "composed")]:
            queries = [dict(id="q", native_id="q", text="query text", image=image, dress_type="shirt", composed=True)]
            candidates = [dict(id="c0", text="short caption", image=image),
                          dict(id="c1", text="a longer caption about people", image=Image.new("RGB", (12, 8), "blue"))]
            dataset = SimpleNamespace(data_name="flickr")
            training = SimpleNamespace(task_type=task, fp16=False, bf16=False)
            for side in ("before_pad", "after_pad"):
                for mode in ("relevant", "what_caption_generation"):
                    if task == "cir" and "caption_generation" in mode:
                        continue
                    for modify in ("no", "modify_single"):
                        search = SimpleNamespace(query_type=direction, rerank_type="pointwise", rerank_template=mode,
                                                 rerank_num=2, rerank_batch_size=1, modify_type=modify,
                                                 use_rerank_prior_cache=False)
                        with self.subTest(task=task, side=side, mode=mode, modify=modify), patch.object(pipeline, "dataset_records", return_value=candidates):
                            expected = pipeline.rerank_records(self.model, self.processor, dataset, queries,
                                                              {"q": {"c0": 1.0, "c1": 0.5}}, task, args, Data(reps_loc=side), search, training, 0, 1)
                            search.rerank_batch_size = 2
                            actual = pipeline.rerank_records(self.model, self.processor, dataset, queries,
                                                            {"q": {"c0": 1.0, "c1": 0.5}}, task, args, Data(reps_loc=side), search, training, 0, 1)
                            for key in candidates:
                                self.assertAlmostEqual(expected["q"][key["id"]], actual["q"][key["id"]], places=5)

    def test_qwen_prior_cache_equals_online_correction(self):
        image = Image.new("RGB", (8, 8), "red")
        candidates = [dict(id="c0", text="short caption"), dict(id="c1", text="a longer caption")]
        queries = [dict(id="q", native_id="q", text="query text", image=image)]
        args = SimpleNamespace(model_name_or_path="renamed-checkpoint")
        for task in ("ir", "it2t"):
            training = SimpleNamespace(task_type=task, fp16=False, bf16=False)
            with tempfile.TemporaryDirectory() as root, patch.object(pipeline, "dataset_records", return_value=candidates):
                search = SimpleNamespace(query_type="image", rerank_type="pointwise", rerank_template="what_caption_generation",
                                         rerank_num=2, rerank_batch_size=2, modify_type="modify_single", use_rerank_prior_cache=False,
                                         rerank_prior_cache_dir=root)
                run = {"q": {"c0": 1.0, "c1": 0.5}}
                expected = pipeline.rerank_records(self.model, self.processor, SimpleNamespace(data_name="flickr"), queries, run, task, args, Data(), search, training, 0, 1)
                search.use_rerank_prior_cache = True
                actual = pipeline.rerank_records(self.model, self.processor, SimpleNamespace(data_name="flickr"), queries, run, task, args, Data(), search, training, 0, 1)
                self.assertEqual(expected, actual)

    def test_artifact_roundtrip_search_and_rerank_for_both_methods(self):
        class Dataset:
            data_name = "flickr"
            img_id_list = ["i0", "i1"]
            text_id_list = ["t0", "t1", "t2"]
            text_dict = {"t0": "short caption", "t1": "a longer caption about people", "t2": "caption blue"}
            images = [Image.new("RGB", (8, 8), "red"), Image.new("RGB", (12, 8), "blue")]
            def __getitem__(self, index):
                return "", self.images[index], self.text_id_list[index], self.img_id_list[index]
            def get_target(self, identifier, direction):
                return {"i0": ["t0", "t2"], "i1": ["t1"]}[identifier] if direction == "image" else {"t0": "i0", "t1": "i1", "t2": "i0"}[identifier]

        class Dense:
            def __init__(self, reps):
                self.reps = np.empty((0, reps.shape[-1]), dtype="float32")
            def add(self, reps):
                self.reps = np.concatenate([self.reps, reps])
            def search(self, queries, depth):
                scores = queries @ self.reps.T
                indices = np.argsort(-scores, axis=1)[:, :depth]
                return np.take_along_axis(scores, indices, axis=1), indices

        class Sparse:
            def __init__(self, index, _):
                self.docs = [json.loads(line) for path in (pipeline.artifact_root(index) / "collection").glob("*.jsonl") for line in path.read_text(encoding="utf-8").splitlines()]
            def set_analyzer(self, _):
                pass
            def batch_search(self, topics, ids, depth, threads):
                from collections import Counter
                return {identifier: sorted([SimpleNamespace(docid=doc["id"], score=sum(value * Counter(topic.split())[term] for term, value in doc["vector"].items())) for doc in self.docs], key=lambda item: item.score, reverse=True)[:depth] for identifier, topic in zip(ids, topics)}
            def close(self):
                pass

        modules = {name: types.ModuleType(name) for name in ("tevatron.retriever.searcher", "pyserini.search.lucene", "pyserini.analysis")}
        modules["tevatron.retriever.searcher"].FaissFlatSearcher = Dense
        modules["pyserini.search.lucene"].LuceneImpactSearcher = Sparse
        modules["pyserini.analysis"].JWhiteSpaceAnalyzer = object
        for concat in (False, True):
            for direction in ("image", "text"):
                with self.subTest(concat=concat, direction=direction), tempfile.TemporaryDirectory() as root:
                    data = Data(dense_output_dir=str(Path(root) / "dense"), sparse_output_dir=str(Path(root) / "sparse"))
                    args = self.encoder(concat=concat).model_args
                    training = SimpleNamespace(task_type="ir", encode_type="text" if direction == "image" else "image", fp16=False, bf16=False, output_dir=str(Path(root) / "result"))
                    search = SimpleNamespace(query_type=direction, rerank_type="pointwise", rerank_template="what_caption_generation",
                                             rerank_num=2, rerank_batch_size=2, modify_type="no", use_rerank_prior_cache=False,
                                             use_fixed_alpha=True, alpha=0.5, depth=2, remove_query=False,
                                             use_gpu=False, threads=1, val_passage_reps=None, val_sparse_index=None)
                    with patch.object(pipeline, "load_runtime", return_value=(self.model, self.processor, 0, 1)), patch.object(pipeline, "build_dataset", side_effect=lambda *a, **k: Dataset()), patch.dict(sys.modules, modules):
                        pipeline.run_encode(args, data, training)
                        dense, sparse = pipeline.artifact_dirs(args, data, training)
                        (sparse / "collection/index").mkdir()
                        (sparse / "collection/index/segments_1").touch()
                        search.passage_reps, search.sparse_index = str(dense), str(sparse)
                        with patch.object(pipeline, "_write_metrics_csv", wraps=pipeline._write_metrics_csv) as csv_report:
                            pipeline.run_search(args, data, search, training, rerank=True)
                            self.assertEqual(csv_report.call_args.args[0], Path(training.output_dir) / "best.csv")
                        report = json.loads((Path(training.output_dir) / "recall_metrics.json").read_text())
                        self.assertEqual(set(report["metrics"]), {"dense", "sparse", "hybrid", "rerank"})
                        self.assertTrue((Path(training.output_dir) / "rerank_run.trec").is_file())
                        self.assertEqual(len(list((sparse / "collection").glob("*.jsonl"))), 1)
                        pipeline.run_encode(args, data, training)
                        self.assertFalse((sparse / "index").exists())
                        self.assertFalse((sparse / "collection/index").exists())
                        self.assertEqual(len(list(sparse.glob("index.stale_*"))), 1)
                        self.assertEqual(len(list(dense.glob("corpus*.pkl"))), 1)
                        self.assertEqual(len(list((sparse / "collection").glob("corpus*.jsonl"))), 1)
                        pipeline.validate_manifest(dense, pipeline.prompt_metadata(
                            args, data, training, self.processor, training.encode_type,
                            family=prompts.model_family(self.model.config)))

    def test_sparse_aliases_use_old_or_nested_index_and_leaf_manifest(self):
        expected = pipeline.prompt_metadata(self.encoder().model_args, Data(), SimpleNamespace(task_type="ir"),
                                            self.processor, "text", family=prompts.model_family(self.model.config))
        loaded = []
        class Sparse:
            def __init__(self, index, _):
                loaded.append(Path(index))
            def set_analyzer(self, analyzer):
                pass
            def close(self):
                pass
        modules = {name: types.ModuleType(name) for name in ("pyserini.search.lucene", "pyserini.analysis")}
        modules["pyserini.search.lucene"].LuceneImpactSearcher = Sparse
        modules["pyserini.analysis"].JWhiteSpaceAnalyzer = object
        with tempfile.TemporaryDirectory(prefix="qwen sparse aliases ") as temporary, patch.dict(sys.modules, modules):
            root = Path(temporary)
            index, collection = root / "index", root / "collection"
            index.mkdir()
            (index / "segments_1").touch()
            collection.mkdir()
            manifest = root / pipeline.manifest_name(expected)
            manifest.write_text(json.dumps(expected), encoding="utf-8")
            # An unrelated JSON inside collection must never become the manifest
            # or an input document for the searcher's Lucene loader.
            (collection / pipeline.manifest_name(expected)).write_text('{"fingerprint":"wrong"}', encoding="utf-8")
            for path in (root, index, collection, str(collection) + "/"):
                with self.subTest(path=str(path)):
                    self.assertEqual(pipeline.validate_manifest(path, expected), expected)
                    searcher = pipeline.CorpusSearch(None, path, expected, SimpleNamespace(), 0)
                    self.assertEqual(loaded[-1], index)
                    searcher.close()
            self.assertEqual(len(loaded), 4)
            nested = collection / "index"
            nested.mkdir()
            for path in (root, collection, nested):
                self.assertEqual(pipeline.validate_manifest(path, expected), expected)
                with self.assertRaisesRegex(ValueError, "Missing Qwen Lucene index"):
                    pipeline.CorpusSearch(None, path, expected, SimpleNamespace(), 0)
            (nested / "segments_2").touch()
            for path in (root, collection, nested, str(nested) + "/"):
                self.assertEqual(pipeline.validate_manifest(path, expected), expected)
                searcher = pipeline.CorpusSearch(None, path, expected, SimpleNamespace(), 0)
                self.assertEqual(loaded[-1], nested)
                searcher.close()
            # Explicit legacy paths remain valid with both layouts present.
            searcher = pipeline.CorpusSearch(None, index, expected, SimpleNamespace(), 0)
            self.assertEqual(loaded[-1], index)
            searcher.close()
            (nested / "segments_2").unlink()
            nested.rmdir()
            (index / "segments_1").unlink()
            index.rmdir()
            with self.assertRaisesRegex(ValueError, "Missing Qwen Lucene index.*index"):
                pipeline.CorpusSearch(None, collection, expected, SimpleNamespace(), 0)
            self.assertEqual(len(loaded), 9)
            manifest.write_text(json.dumps(dict(expected, fingerprint="mismatched")), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Qwen prompt/configuration mismatch"):
                pipeline.CorpusSearch(None, collection, expected, SimpleNamespace(), 0)


if __name__ == "__main__":
    unittest.main()
