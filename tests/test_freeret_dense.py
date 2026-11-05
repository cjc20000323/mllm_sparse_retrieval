"""CPU checks for FreeRet dense states, sparse invariance and artifact routing."""
import ast
from contextlib import redirect_stdout
import copy
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import dense_representation as dense
import adaptive_view_encoding as adaptive
import adaptive_views as views
from qwen_vl_encoding import NativeQwenEncoder, SparseEncoder


def data_args(**changes):
    return NS(**dict(dict(dataset_name="flickr", dataset_split="test", prompt_type="prompt_5", random_view_seed=42,
        reps_loc="after_pad", tbpr_type="origin_type", cir_type="type", sparse_value_type="sum",
        sparse_lower_or_upper="lower", sparse_value_mean=False, sparse_manual=False, is_filtered=False,
        num_expended_tokens=0, sparse_length=20, text_sparse_length=20, image_sparse_length=20,
        llava_image_mode="anyres", per_device_batch_size=3, dense_output_dir="dense", sparse_output_dir="sparse"),
        **changes))


def model_args(use=False, name="fixture"):
    return NS(use_freeret_dense=use, model_name_or_path=name, eol_type="disassembleeol_separate", calculate_type="concat")


def temporary_directory():
    result = tempfile.TemporaryDirectory()
    if os.name == "nt":
        result.name = "\\\\?\\" + result.name
    return result


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(8, 8)
        self.post_attention_layernorm = nn.LayerNorm(8)
        self.mlp = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8))

    def forward(self, value):
        value = value + self.attn(value)
        return value + self.mlp(self.post_attention_layernorm(value))


class ToyModel(nn.Module):
    def __init__(self, family="qwen2_5_vl", nesting="model.layers"):
        super().__init__()
        self.config = NS(model_type=family, text_config=NS(model_type="llama", vocab_size=128256))
        self.embedding = nn.Embedding(32, 8)
        self.norm = nn.LayerNorm(8)
        self.lm_head = nn.Linear(8, 32, bias=False)
        owner = self
        for part in nesting.split(".")[:-1]:
            child = nn.Module()
            setattr(owner, part, child)
            owner = child
        owner.layers = nn.ModuleList([DecoderLayer(), DecoderLayer()])
        self.nesting = nesting

    def decoder_layers(self):
        result = self
        for part in self.nesting.split("."):
            result = getattr(result, part)
        return result

    def forward(self, input_ids, output_hidden_states=True, fail=False, **kwargs):
        first = value = self.embedding(input_ids)
        for layer in self.decoder_layers():
            value = layer(value)
        if fail:
            raise RuntimeError("fixture failure")
        value = self.norm(value)
        return CausalLMOutputWithPast(logits=self.lm_head(value), hidden_states=(first, value) if output_hidden_states else None)


class CaptureTests(unittest.TestCase):
    def test_five_families_match_freeret_node_and_keep_logits_and_weights(self):
        families = (("qwen2_vl", "model.layers"), ("qwen2_5_vl", "model.language_model.layers"),
                    ("qwen3_vl", "model.language_model.layers"), ("llava_next", "language_model.layers"),
                    ("llava_next", "language_model.model.layers"))
        for family, nesting in families:
            with self.subTest(family=family, nesting=nesting):
                model = ToyModel(family, nesting).eval()
                if "model.layers" in nesting and family == "llava_next":
                    model.config.text_config.model_type = "mistral"
                inputs = torch.tensor([[0, 0, 1, 2, 3], [1, 2, 3, 0, 0]])
                weights = copy.deepcopy(model.state_dict())
                captured = []
                handle = model.decoder_layers()[-1].post_attention_layernorm.register_forward_pre_hook(
                    lambda _m, values: captured.append(values[0].detach().clone()))
                old = model(inputs, output_hidden_states=True)
                handle.remove()
                self.assertFalse(torch.equal(old.hidden_states[-1], captured[0]))
                with redirect_stdout(io.StringIO()):
                    dense.configure_model(model, True)
                handles = model._freeret_dense_hooks
                dense.configure_model(model, True)
                self.assertIs(handles, model._freeret_dense_hooks)
                new = model(inputs, output_hidden_states=True)
                self.assertTrue(torch.equal(captured[0], new.hidden_states[-1]))
                self.assertTrue(torch.equal(old.logits, new.logits))
                for positions in ([4, 4], [4, 2]):
                    self.assertTrue(torch.equal(new.hidden_states[-1][torch.arange(2), positions],
                                                captured[0][torch.arange(2), positions]))
                for key, value in weights.items():
                    self.assertTrue(torch.equal(value, model.state_dict()[key]))
                dense.configure_model(model, False)
                self.assertFalse(hasattr(model, "_freeret_dense_hooks"))
                self.assertTrue(torch.equal(old.hidden_states[-1], model(inputs, output_hidden_states=True).hidden_states[-1]))

    def test_low_memory_capture_does_not_change_rerank_or_selection_logits_and_cleans_errors(self):
        model = ToyModel().eval()
        inputs = dict(input_ids=torch.tensor([[1, 2, 3]]), output_hidden_states=False)
        old = model(**inputs)
        with redirect_stdout(io.StringIO()):
            dense.configure_model(model, True)
        normal = model(**inputs)
        self.assertFalse(hasattr(normal, "_freeret_dense_hidden"))
        self.assertTrue(torch.equal(old.logits, normal.logits))
        forced = dense.forward(model, **inputs)
        self.assertIsNone(forced.hidden_states)
        self.assertTrue(torch.equal(forced.logits, old.logits))
        self.assertEqual(forced._freeret_dense_hidden.shape, (1, 3, 8))
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            dense.forward(model, **inputs, fail=True)
        self.assertFalse(model._freeret_dense_force_capture)
        self.assertTrue(torch.equal(forced._freeret_dense_hidden, dense.forward(model, **inputs)._freeret_dense_hidden))

    def test_unsupported_models_only_fail_when_enabled_and_missing_node_never_falls_back(self):
        model = ToyModel("internvl_chat")
        self.assertIs(dense.configure_model(model, False), model)
        self.assertFalse(model._forward_hooks)
        with self.assertRaisesRegex(ValueError, "Unsupported model"):
            dense.configure_model(model, True)
        model.config.model_type = "qwen3_vl"
        del model.model.layers[-1].post_attention_layernorm
        with self.assertRaisesRegex(ValueError, "Cannot locate"):
            dense.configure_model(model, True)


def qwen_processor():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    words = ["[UNK]", "[PAD]", "<|im_start|>", "<|im_end|>", "user", "assistant", "a", "b"]
    words += [f"v{i}" for i in range(60 - len(words))]
    words += ["<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"]
    backend = Tokenizer(WordLevel({word: index for index, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
        additional_special_tokens=words[2:4] + words[60:])

    class Processor:
        def __init__(self):
            self.tokenizer = tokenizer

        def apply_chat_template(self, messages, **kwargs):
            body = "".join("<|vision_start|><|image_pad|><|vision_end|>" if part["type"] == "image" else part["text"]
                           for message in messages for part in message["content"])
            return "<|im_start|>user\n" + body + "<|im_end|>\n<|im_start|>assistant\n"

        def __call__(self, images=None, **kwargs):
            result = dict(self.tokenizer(**kwargs))
            result.pop("token_type_ids", None)
            if images:
                result["pixel_values"] = torch.zeros(len(images) * 4, 3 * 2 * 14 * 14)
                result["image_grid_thw"] = torch.tensor([[1, 2, 2]] * len(images))
            return result

    return Processor()


class RealModelTests(unittest.TestCase):
    def compare(self, model, processor, records, family, adapter="adaptive"):
        for location in ("after_pad", "before_pad"):
            for count in (1, 5):
                with self.subTest(family=family, adapter=adapter, location=location, views=count):
                    data = data_args(reps_loc=location)
                    selected = {"selected": views.view_pool()[:count]}
                    args = model_args(False, family)
                    make = (lambda: adaptive.AdaptiveEncoder(model, processor, args, data, "ir", selected)) if adapter == "adaptive" else (
                        lambda: NativeQwenEncoder(model, processor, args, data, "ir"))
                    encoder = make()
                    captured = []
                    norm = [module for name, module in model.named_modules() if name.endswith("post_attention_layernorm")][-1]
                    handle = norm.register_forward_pre_hook(lambda _m, inputs: captured.append(inputs[0].detach().clone()))
                    baseline, old_dense = encoder.encode(records)
                    handle.remove()
                    prompt = list(encoder.last_prompts)
                    batch = encoder.prepare(records)
                    offset = 2 if family.startswith("llava") else 1
                    expected = torch.nn.functional.normalize(torch.stack([
                        captured[0][row, blocks[0][1] - offset] for row, blocks in enumerate(batch.spans)]), dim=-1)
                    args.use_freeret_dense = True
                    with redirect_stdout(io.StringIO()):
                        replacement = make()
                    sparse, new_dense = replacement.encode(records)
                    self.assertEqual(prompt, replacement.last_prompts)
                    self.assertTrue(torch.equal(new_dense, expected))
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(baseline, sparse)))
                    if family.startswith("qwen"):
                        converter = SparseEncoder(processor.tokenizer, data, concat=True)
                        self.assertEqual([converter.vector(row, record) for row, record in zip(baseline, records)],
                                         [converter.vector(row, record) for row, record in zip(sparse, records)])
                    dense.configure_model(model, False)

    def test_native_and_adaptive_qwen_three_families_with_text_image_and_fused_inputs(self):
        from transformers import (Qwen2VLConfig, Qwen2VLForConditionalGeneration,
            Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen3VLConfig, Qwen3VLForConditionalGeneration)
        from PIL import Image
        for config_class, model_class, family in ((Qwen2VLConfig, Qwen2VLForConditionalGeneration, "qwen2"),
                (Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, "qwen25"),
                (Qwen3VLConfig, Qwen3VLForConditionalGeneration, "qwen3")):
            text = dict(vocab_size=64, hidden_size=24, intermediate_size=32, num_hidden_layers=1,
                num_attention_heads=2, num_key_value_heads=2, head_dim=12, max_position_embeddings=4096,
                rope_scaling={"rope_type": "default", "mrope_section": [2, 2, 2]})
            vision = dict(depth=1, embed_dim=16, hidden_size=16, intermediate_size=32, num_heads=2,
                out_hidden_size=24, patch_size=14, temporal_patch_size=2, spatial_merge_size=2,
                fullatt_block_indexes=[0], deepstack_visual_indexes=[])
            if config_class is Qwen2VLConfig:
                vision["hidden_size"] = 24
            config = config_class(text_config=text, vision_config=vision, image_token_id=60,
                video_token_id=61, vision_start_token_id=62, vision_end_token_id=63)
            config._attn_implementation = "eager"
            model = model_class(config).eval()
            image = Image.new("RGB", (28, 28))
            records = [{"id": "t", "text": "a b"}, {"id": "i", "image": image}, {"id": "f", "text": "a", "image": image}]
            for adapter in ("native", "adaptive"):
                self.compare(model, qwen_processor(), records, family, adapter)

    def test_llava_llama_and_mistral_with_native_image_processor(self):
        from PIL import Image
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import (CLIPImageProcessor, LlavaNextConfig, LlavaNextForConditionalGeneration,
            LlavaNextProcessor, LlamaConfig, MistralConfig, PreTrainedTokenizerFast)
        import llava_resolution
        words = ["[UNK]", "[PAD]", "<s>", "</s>", "<|begin_of_text|>", "<|end_of_text|>", "<|eot_id|>",
                 "<|start_header_id|>", "<|end_header_id|>", "a", "b", "user", "assistant"]
        words += [f"v{i}" for i in range(63-len(words))] + ["<image>"]
        for text_class, family in ((LlamaConfig, "llava_llama3"), (MistralConfig, "llava_mistral")):
            backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
                additional_special_tokens=words[2:9] + ["<image>"])
            text = text_class(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                num_attention_heads=2, num_key_value_heads=2, pad_token_id=1)
            config = LlavaNextConfig(text_config=text.to_dict(), vision_config=dict(hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, image_size=28, patch_size=14, projection_dim=16),
                image_token_index=63, vision_feature_layer=-1, image_grid_pinpoints=[[28, 28]])
            config._attn_implementation = "eager"
            model = LlavaNextForConditionalGeneration(config).eval()
            processor = LlavaNextProcessor(CLIPImageProcessor(size={"shortest_edge":28}, crop_size={"height":28,"width":28},
                image_grid_pinpoints=[[28,28]]), tokenizer, patch_size=14, num_additional_image_tokens=1,
                vision_feature_select_strategy="default", image_token="<image>")
            processor = llava_resolution.configure(model, processor, "base")
            image = Image.new("RGB", (28,28))
            self.compare(model, processor, [{"id":"t","text":"a b"},{"id":"i","image":image},
                                          {"id":"f","image":image,"text":"a"}], family)


class ArtifactTests(unittest.TestCase):
    def test_false_fingerprints_and_paths_are_unchanged_and_true_paths_are_idempotent(self):
        original = dict(version=1, prompt="unchanged", sparse_length=40)
        self.assertIs(dense.configuration(original, False), original)
        for path, expected in (("out/leaf", "out/leaf"+dense.SUFFIX),
            ("out/leaf/collection/index", "out/leaf"+dense.SUFFIX+"/collection/index"),
            ("out/leaf/corpus_0.pkl", "out/leaf"+dense.SUFFIX+"/corpus_0.pkl")):
            self.assertEqual(str(dense.mode_path(path, False)), path)
            marked = dense.mode_path(path, True)
            self.assertEqual(marked.as_posix(), expected)
            self.assertEqual(dense.mode_path(marked, True), marked)

    def test_mode_mismatch_and_missing_sidecar_are_rejected(self):
        with temporary_directory() as directory:
            old = Path(directory)/"old"
            new = dense.mode_path(old, True)
            dense.validate_artifact(old, False)
            with self.assertRaisesRegex(ValueError, "Missing dense representation"):
                dense.validate_artifact(new, True)
            dense.write_manifest(new, True)
            dense.validate_artifact(new/"collection/index", True)
            with self.assertRaisesRegex(ValueError, "mode mismatch"):
                dense.validate_artifact(new, False)
            dense.sidecar(new).write_text('{"source":"wrong","version":1}')
            with self.assertRaisesRegex(ValueError, "mode mismatch"):
                dense.validate_artifact(new, True)

    def test_sparse_copy_is_independent_and_different_documents_fail(self):
        for collection in (True, False):
            with self.subTest(collection=collection), temporary_directory() as directory:
                old = Path(directory)/"sparse"
                new = dense.mode_path(old, True)
                suffix = "collection" if collection else ""
                for root in (old,new):
                    (root/suffix).mkdir(parents=True)
                    (root/suffix/"corpus_0.jsonl").write_text('{"id":"one","vector":{"term":30}}\n')
                source = old/suffix/"index"
                source.mkdir()
                (source/"segments_1").write_bytes(b"index")
                (source/"write.lock").touch()
                with redirect_stdout(io.StringIO()):
                    self.assertTrue(dense.reuse_sparse_index(new))
                target = new/suffix/"index"
                self.assertFalse((target/"write.lock").exists())
                (target/"segments_1").write_bytes(b"changed independently")
                self.assertEqual((source/"segments_1").read_bytes(),b"index")
                (new/suffix/"corpus_0.jsonl").write_text('{"id":"one","vector":{"term":31}}\n')
                with self.assertRaisesRegex(ValueError,"Sparse documents differ"):
                    dense.reuse_sparse_index(new)

    def test_completed_manifest_configurations_and_hashes_are_validated_before_reuse(self):
        for key in ("configuration","config"):
            with self.subTest(key=key), temporary_directory() as directory:
                old=Path(directory)/"sparse"; new=dense.mode_path(old,True)
                config=dict(version=1,prompt="same",sparse_length=40)
                def manifest(root, configuration):
                    options=dict(ensure_ascii=False,sort_keys=True)
                    if key=="configuration": options["separators"]=(",",":")
                    value={key:configuration,"fingerprint":hashlib.sha256(json.dumps(configuration,**options).encode()).hexdigest()}
                    (root/"manifest.json").write_text(json.dumps(value))
                for root, configuration in ((old,config),(new,dense.configuration(config,True))):
                    (root/"collection").mkdir(parents=True)
                    (root/"collection/corpus_0.jsonl").write_text('{"id":"a","vector":{"a":1}}\n')
                    manifest(root,configuration)
                (old/"collection/index").mkdir()
                (old/"collection/index/segments_1").touch()
                with redirect_stdout(io.StringIO()):
                    self.assertTrue(dense.reuse_sparse_index(new,manifest_name="manifest.json"))
                manifest(old,dict(config,sparse_length=30))
                with self.assertRaisesRegex(ValueError,"different encoding configuration"):
                    dense.reuse_sparse_index(new,manifest_name="manifest.json")

    def test_adaptive_metadata_and_paths_keep_selection_fingerprint_separate(self):
        import adaptive_view_overlap as overlap
        data=data_args(); selection=dict(version=views.VERSION,selection_modality="both",selected=views.view_pool()[:5],
            views_fingerprint="a"*64)
        for task,kind in (("ir","image"),("tbpr","image"),("t2it","corpus"),("it2t","corpus")):
            for mode in ("anyres","base"):
                with self.subTest(task=task,mode=mode):
                    data.llava_image_mode=mode
                    old=adaptive.metadata(model_args(False),data,task,kind,{"model":"fixture"},selection)
                    new=adaptive.metadata(model_args(True),data,task,kind,{"model":"fixture"},selection)
                    stripped=dict(new["configuration"]); stripped.pop("dense_representation")
                    self.assertEqual(stripped,old["configuration"])
                    self.assertNotEqual(old["fingerprint"],new["fingerprint"])
                    self.assertEqual(old["configuration"]["views_fingerprint"],new["configuration"]["views_fingerprint"])
                    training=NS(task_type=task,encode_type=kind)
                    old_paths=adaptive.artifact_dirs(model_args(False),data,training,selection)
                    new_paths=adaptive.artifact_dirs(model_args(True),data,training,selection)
                    self.assertEqual(new_paths,[dense.mode_path(path,True) for path in old_paths])
                    self.assertTrue(all(str(path).count(dense.SUFFIX)==1 for path in new_paths))
                    old_strategy=overlap.OverlapSelectionStrategy(overlap.OverlapViewArguments(),model_args(False),data,task)
                    new_strategy=overlap.OverlapSelectionStrategy(overlap.OverlapViewArguments(),model_args(True),data,task)
                    self.assertEqual(old_strategy.configuration,new_strategy.configuration)


class EntrypointTests(unittest.TestCase):
    def test_all_seven_entrypoints_accept_true_false_and_default_false(self):
        from transformers import HfArgumentParser
        import arguments
        from adaptive_view_overlap import OverlapViewArguments
        from adaptive_views import AdaptiveViewArguments
        from time_estimation import RerankTimeEstimationArguments
        namespace = dict(vars(arguments), HfArgumentParser=HfArgumentParser,
            AdaptiveViewArguments=AdaptiveViewArguments, OverlapViewArguments=OverlapViewArguments,
            RerankTimeEstimationArguments=RerankTimeEstimationArguments)
        for name in ("encode", "search", "rerank", "encode_adaptive_views", "encode_adaptive_views_overlap"):
            tree = ast.parse((ROOT/"src"/(name+".py")).read_text(encoding="utf-8"))
            # Exercise each entry's actual parser declaration without loading models,
            # NLTK data or Java during the module's top-level imports.
            declaration = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "parser" for target in node.targets)
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "HfArgumentParser")
            parser = eval(compile(ast.Expression(declaration), str(ROOT/"src"/(name+".py")), "eval"), namespace)
            for value in (None, "false", "true"):
                with self.subTest(entry=name, value=value):
                    argv = ["--model_name_or_path", "fixture", "--output_dir", "unused",
                            "--report_to", "none", "--use_cpu", "true"]
                    if value is not None: argv += ["--use_freeret_dense", value]
                    parsed = parser.parse_args_into_dataclasses(argv)
                    self.assertIs(parsed[0].use_freeret_dense, value == "true")
        for name in ("three_stage_retrieval_pipeline", "three_stage_retrieval_pipeline_adaptive_views"):
            module = importlib.import_module(name)
            for value in (None, "false", "true"):
                for flag in ("--use_freeret_dense", "--use-freeret-dense"):
                    with self.subTest(entry=name, value=value, flag=flag):
                        args = module._parser().parse_args(["--dataset-name", "flickr"] +
                            ([] if value is None else [flag, value]))
                        self.assertIs(args.use_freeret_dense, value == "true")
                        model, _ = module._retrieval_namespaces(args,
                            eol_type="disassembleeol_separate", calculate_type="concat", sparse_manual=False)
                        self.assertIs(dense.enabled(model), value == "true")
                        if value != "true": self.assertNotIn("use_freeret_dense", vars(model))

    def test_entry_and_pipeline_isolate_explicit_outputs_and_validate_input_modes(self):
        with temporary_directory() as directory:
            old = Path(directory)/"corpus"
            new = dense.mode_path(old, True)
            dense.write_manifest(new, True)
            for prepare in ("entry", "pipeline"):
                for use in (False, True):
                    with self.subTest(prepare=prepare, use=use):
                        args = NS(use_freeret_dense=use, output_dir=str(Path(directory)/"custom"),
                            retrieval_model_name_or_path="fixture", stage2_dense_index=str(old),
                            stage2_sparse_index=str(old/"collection/index"))
                        search = NS(passage_reps=str(old), sparse_index=str(old/"collection/index"))
                        with patch.object(dense, "validate_checkpoint"):
                            if prepare == "entry": dense.prepare_entry(model_args(use), data_args(), args, search)
                            else: dense.prepare_pipeline(args)
                        self.assertEqual(args.output_dir, str(dense.mode_path(Path(directory)/"custom", use)))
                        self.assertEqual(search.passage_reps if prepare=="entry" else args.stage2_dense_index,
                                         str(new if use else old))
                        if use:
                            dense.sidecar(new).unlink()
                            with patch.object(dense, "validate_checkpoint"), self.assertRaisesRegex(ValueError, "Missing dense"):
                                if prepare=="entry": dense.prepare_entry(model_args(use), data_args(), args, search)
                                else: dense.prepare_pipeline(args)
                            dense.write_manifest(new, True)

    def test_batch_switch_preserves_selection_cache_for_every_task_and_pmi_overlap(self):
        import adaptive_encode_batch as encode_batch
        import adaptive_retrieval_batch as retrieval_batch
        with temporary_directory() as directory:
            root = Path(directory)
            for family in ("qwen25", "llava_llama"):
                for group in ("ir", "tbpr", "fused"):
                    for overlap in (False, True):
                        for pmi in (False, True):
                            with self.subTest(family=family, group=group, overlap=overlap, pmi=pmi):
                                env = dict(VIEW_EXAMPLE_MODE="fixed", MODEL_FAMILY=family, CHECKPOINT_ROOT=str(root/"checkpoints"),
                                    DATA_ROOT=str(root), DENSE_OUTPUT_ROOT=str(root/"dense"),
                                    SPARSE_OUTPUT_ROOT=str(root/"sparse"), USE_VIEW_PMI=str(pmi).lower(),
                                    USE_VIEW_OVERLAP_SELECTION=str(overlap).lower())
                                old = retrieval_batch.Settings.from_environment(group, env, project=root)
                                new = retrieval_batch.Settings.from_environment(group, dict(env, USE_FREERET_DENSE="true"), project=root)
                                for dataset in old.datasets:
                                    for task, query, kind in retrieval_batch.job_types(old, dataset):
                                        before = retrieval_batch.namespaces(old, family, dataset, task, kind)
                                        after = retrieval_batch.namespaces(new, family, dataset, task, kind)
                                        self.assertFalse(dense.enabled(before[0])); self.assertTrue(dense.enabled(after[0]))
                                        identity = {"model":"fixture"}
                                        self.assertEqual(retrieval_batch.selection_location(old, *before, identity)[:2],
                                                         retrieval_batch.selection_location(new, *after, identity)[:2])
                                        model_path = old.models[family]/"config.json"
                                        model_path.parent.mkdir(parents=True, exist_ok=True); model_path.touch()
                                        for annotation in encode_batch.annotation_files(root, dataset, "test"):
                                            annotation.parent.mkdir(parents=True, exist_ok=True); annotation.touch()
                                old_jobs, missing = encode_batch.prepare_jobs(encode_batch.Settings.from_environment(group, env, project=root))
                                new_jobs, missing_new = encode_batch.prepare_jobs(encode_batch.Settings.from_environment(group,
                                    dict(env, USE_FREERET_DENSE="true"), project=root))
                                self.assertFalse(missing or missing_new)
                                self.assertEqual(len(old_jobs), len(new_jobs))
                                for before, after in zip(old_jobs, new_jobs):
                                    self.assertEqual(before.key, after.key)
                                    self.assertEqual(after.variables["USE_FREERET_DENSE"], "true")
                                    self.assertEqual(after.variables["OUTPUT_DIR"],
                                        str(dense.mode_path(before.variables["OUTPUT_DIR"], True)))

    def test_bash_helpers_and_six_encoding_launchers_forward_the_switch(self):
        bash = shutil.which("bash") if os.name != "nt" else "D:/Git/bin/bash.exe"
        if not bash or not Path(bash).is_file(): self.skipTest("Bash unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root/"capture.sh"
            capture.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@"\n', encoding="utf-8")
            capture.chmod(0o755)
            for use in ("false", "true"):
                environment = dict(os.environ, USE_FREERET_DENSE=use, DEEPSPEED_BIN=capture.as_posix(),
                    MODEL_FAMILY="qwen25", MODEL_PATH="fixture", OUTPUT_DIR="custom", DRY_RUN="false",
                    LLAVA_IMAGE_MODE="anyres")
                for group, dataset, kind in (("ir","flickr","image"),("tbpr","CUHK-PEDES","image"),
                                             ("fused","remuq","corpus")):
                    for overlap in ("", "_overlap"):
                        name = f"encode_adaptive_views{overlap}_{group}.sh"
                        with self.subTest(script=name, use=use):
                            result = subprocess.run([bash, (ROOT/"scripts"/name).as_posix(), "--run-one"], cwd=ROOT,
                                env=dict(environment, DATASET=dataset, ENCODE_TYPE=kind), capture_output=True, text=True, timeout=30)
                            self.assertEqual(result.returncode, 0, result.stderr)
                            command = result.stdout.splitlines()
                            self.assertEqual(command[command.index("--use_freeret_dense")+1], use)
                result = subprocess.run([bash], cwd=ROOT, env=environment, input='source scripts/freeret_dense_mode.sh\n'
                    'dense_mode_path "out/a b/collection/index"\n', capture_output=True, text=True, timeout=30)
                expected = "out/a b" + (dense.SUFFIX if use=="true" else "") + "/collection/index"
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, expected)


if __name__ == "__main__":
    unittest.main()
