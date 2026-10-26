"""Offline VLM2Vec query/retrieval tests with real loaders, tiny weights and FAISS."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tevatron/src"))

import numpy as np
from PIL import Image
import torch

import dense_encoding as common
import dense_search as search
import vlm2vec_dense as vlm
import test_dense_search as search_fixtures
import test_mmret_dense as data_fixtures
import test_vlm2vec_dense as vlm_fixtures


def arguments(name="flickr", query_type="text"):
    model, data, query, train = search_fixtures.arguments(name, query_type)
    model.model_name_or_path = vlm.MODEL_ID
    return model, data, query, train


class VLM2VecSearchUtilityTests(unittest.TestCase):
    def test_task_instructions_and_exact_query_forms(self):
        expected = {
            "ir:text": "Retrieve an image matching the following description:",
            "ir:image": "Find an image caption describing the given image.",
            "tbpr": "Retrieve a person image matching the following description:",
            "t2it": "Retrieve an image and text pair answering the following question:",
            "it2t": "Represent the given image with the following question:",
        }
        self.assertEqual(vlm.QUERY_INSTRUCTIONS, expected)
        self.assertEqual(vlm.query_prompt("caption", False, expected["ir:text"]),
                         expected["ir:text"] + " caption")
        self.assertEqual(vlm.query_prompt(None, True, expected["ir:image"]),
                         "<image> " + expected["ir:image"])
        self.assertEqual(vlm.query_prompt("question", True, expected["it2t"]),
                         "<image> Represent the given image with the following question: question")
        self.assertEqual(vlm.query_prompt("words", True, "Custom task:"), "<image> Custom task: words")
        self.assertEqual(vlm.candidate_prompt("words", False), "words")
        self.assertEqual(vlm.candidate_prompt("words", True), "<image>\nwords")

    def test_invalid_prompt_inputs(self):
        for instruction in (None, "", "  ", "<image> Task:"):
            with self.assertRaises(ValueError):
                vlm.query_prompt("text", False, instruction)
        with self.assertRaises(TypeError):
            vlm.query_prompt(1, False, "Task:")
        with self.assertRaisesRegex(ValueError, "must contain"):
            vlm.query_prompt(None, False, "Task:")

    def test_entry_dispatch_and_custom_options(self):
        import search_dense
        for model_path, kind in ((vlm.MODEL_ID, "auto"), ("/custom/checkpoint", "vlm2vec")):
            argv = ["search_dense.py", "--model_name_or_path", model_path, "--dense_model_type", kind,
                    "--data_root", "/data-root", "--passage_reps", "/index", "--dataset_name", "webqa",
                    "--task_type", "t2it", "--query_type", "text", "--vlm2vec_image_resolution", "low",
                    "--vlm2vec_query_instruction", "Custom task:", "--report_to", "none", "--output_dir", "unused"]
            with patch.object(sys, "argv", argv), patch.object(search_dense, "run_dense_search") as run:
                search_dense.main()
            self.assertEqual(run.call_args.args[-1], "vlm2vec")
            self.assertEqual(run.call_args.args[4].vlm2vec_image_resolution, "low")
            self.assertEqual(run.call_args.args[5].vlm2vec_query_instruction, "Custom task:")

    def test_native_instruction_flags_cannot_leak_to_other_models(self):
        model, data, query, train = arguments()
        query.passage_reps = "/index"
        cases = [("vlm2vec", search.DenseSearchArguments(mmret_query_instruction="Task:")),
                 ("mmret", search.DenseSearchArguments(vlm2vec_query_instruction="Task:")),
                 ("unime", search.DenseSearchArguments(vlm2vec_query_instruction="Task:"))]
        for kind, native in cases:
            with patch.object(search.VLM2VecQueryEncoder, "from_pretrained") as load:
                with self.assertRaisesRegex(ValueError, "only supported"):
                    search.run_dense_search(model, data, query, train, common.DenseEncoderArguments(), native, kind)
            load.assert_not_called()
        import search_dense
        argv = ["search_dense.py", "--model_name_or_path", "example/clip-model",
                "--vlm2vec_query_instruction", "Task:", "--output_dir", "unused", "--report_to", "none"]
        with patch.object(sys, "argv", argv), patch.object(search_dense.CLIPModel, "from_pretrained") as load:
            with self.assertRaisesRegex(ValueError, "native"):
                search_dense.main()
        load.assert_not_called()

    def test_invalid_settings_fail_before_context_or_model_load(self):
        model, data, query, train = arguments()
        query.passage_reps = "/index"
        with patch.object(search.VLM2VecQueryEncoder, "from_pretrained") as load:
            for instruction in ("", "   ", "Task <image>"):
                context = unittest.mock.Mock()
                with self.assertRaisesRegex(ValueError, "instruction"):
                    search.run_dense_search(model, data, query, train, common.DenseEncoderArguments(),
                        search.DenseSearchArguments(vlm2vec_query_instruction=instruction), "vlm2vec",
                        context_factory=context)
                context.assert_not_called()
            with self.assertRaisesRegex(ValueError, "image_resolution"):
                search.run_dense_search(model, data, query, train,
                    common.DenseEncoderArguments(vlm2vec_image_resolution="invalid"), search.DenseSearchArguments(), "vlm2vec")
        load.assert_not_called()

    def test_resolution_result_paths_and_explicit_override(self):
        model, data, query, _ = arguments()
        for resolution in vlm.IMAGE_RESOLUTIONS:
            dense = common.DenseEncoderArguments(vlm2vec_image_resolution=resolution)
            self.assertEqual(search.result_directory(model, data, query, model_type="vlm2vec", dense_args=dense),
                             Path("search_results/VLM2Vec-LLaVa-Next/flickr/text/test") / f"vlm2vec_{resolution}")
        query.save_dir = "/custom results"
        self.assertEqual(search.result_directory(model, data, query, model_type="vlm2vec", dense_args=dense),
                         Path(query.save_dir))

    def test_resolution_custom_instruction_and_empty_rank_reach_native_encoder(self):
        model, data, query, train = arguments("webqa")
        data.per_device_batch_size = 1
        query.passage_reps = "/index"
        queries = SimpleNamespace(source=None, name="webqa")

        class EmptyLocalShard:
            source, name = queries.source, queries.name

            def __len__(self):
                return 1

        encoder = SimpleNamespace(dimension=16)
        with patch.object(search.VLM2VecQueryEncoder, "from_pretrained", return_value=encoder) as load, \
             patch.object(search, "load_dense_index", return_value=(None, [101], None)), \
             patch.object(search, "write_recall") as write, \
             patch.object(search.dist, "is_initialized", return_value=False):
            counts = search.run_dense_search(model, data, query, train,
                common.DenseEncoderArguments(vlm2vec_image_resolution="mid"),
                search.DenseSearchArguments(vlm2vec_query_instruction="Custom task:"), "vlm2vec",
                context_factory=lambda: common.EncodingContext(2, 3, torch.device("cpu")),
                query_builder=lambda *unused: EmptyLocalShard())
        self.assertEqual(load.call_args.kwargs["image_resolution"], "mid")
        self.assertEqual(encoder.instruction, "Custom task:")
        np.testing.assert_array_equal(counts, np.zeros(8))
        write.assert_not_called()


class VLM2VecSearchDatasetTests(unittest.TestCase):
    make_source = data_fixtures.MMRetDatasetTests.make_source
    parquet_frame = data_fixtures.MMRetDatasetTests.parquet_frame
    setUp = data_fixtures.MMRetDatasetTests.setUp
    setUpClass = classmethod(data_fixtures.MMRetDatasetTests.setUpClass.__func__)
    build = search_fixtures.DenseSearchDatasetTests.build

    def test_eight_datasets_query_ids_modalities_and_instructions(self):
        for name, (task, _, _) in common.DATASET_PROFILES.items():
            self.make_source(name)
            directions = ("text", "image") if task == "ir" else (("image", "composed") if task == "it2t" else ("text",))
            for direction in directions:
                with self.subTest(dataset=name, direction=direction):
                    queries = self.build(name, direction)
                    expected = ["caption-1", "caption-2"] if task == "ir" and direction == "text" else (
                        ["image-A"] if task == "ir" else ["0", "1"] if task == "tbpr" else ["q"])
                    self.assertEqual(queries.ids, expected)
                    key = f"ir:{direction}" if task == "ir" else task
                    for index in range(len(queries)):
                        item = queries[index]
                        prompt = vlm.query_prompt(item.text, item.image is not None, vlm.QUERY_INSTRUCTIONS[key])
                        if task == "it2t":
                            self.assertEqual(item.text, "query")
                            self.assertIsNotNone(item.image)
                            queries.source.id2query[item.identifier]["image"] = {"path": "picture.png", "bytes": None}
                            with vlm.load_image(queries[index].image, "low") as image:
                                self.assertEqual(image.size, (128, 128))
                        self.assertIn(vlm.QUERY_INSTRUCTIONS[key], prompt)
                        self.assertEqual("<image>" in prompt, item.image is not None)

    def test_generated_validation_uses_original_image_fallback(self):
        self.make_source("flickr")
        directory = self.root / "generated/data/flickr"
        directory.mkdir(parents=True)
        (directory / "flickr_val.csv").write_bytes((self.root / "data/flickr/flickr_test.csv").read_bytes())
        queries = self.build("flickr", "image", split="val", use_generated_validation_data=True,
                             generated_validation_data_root=str(self.root / "generated"))
        self.assertEqual(queries.data_dir, directory)
        with vlm.load_image(queries[0].image, "original") as image:
            self.assertEqual(image.size, (10, 20))


class VLM2VecSearchNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vlm_fixtures.VLM2VecNativeModelTests.setUpClass.__func__(cls)
        cls.query_encoder = vlm.VLM2VecQueryEncoder.from_pretrained(str(cls.current), "cpu", torch.float32,
                                                                   image_resolution="original")

    def test_native_query_modalities_match_individual_and_final_hidden_layer(self):
        images = [Image.new("RGB", (28, 28), "red"), Image.new("RGB", (56, 28), "blue")]
        for mode, key in (("text", "ir:text"), ("image", "ir:image"), ("joint", "it2t")):
            self.query_encoder.instruction = vlm.QUERY_INSTRUCTIONS[key]
            records = [common.Candidate(index, text=("short" if index == 0 else "long text") if mode != "image" else None,
                                        image=images[index] if mode != "text" else None) for index in range(2)]
            together = self.query_encoder.encode_batch(records)
            individual = np.concatenate([self.query_encoder.encode_batch([item]) for item in records])
            np.testing.assert_allclose(together, individual, atol=4e-6)
            np.testing.assert_allclose(np.linalg.norm(together, axis=1), [1, 1], atol=1e-6)
            prompts = [self.query_encoder.format_prompt(item) for item in records]
            inputs = self.processor(text=prompts, images=images if mode != "text" else None, padding=True, return_tensors="pt")
            inputs["position_ids"] = (inputs["attention_mask"].cumsum(-1) - 1).clamp_min(0)
            with torch.inference_mode():
                outputs = self.model(**inputs, output_hidden_states=True, use_cache=False, logits_to_keep=1)
                expected = torch.nn.functional.normalize(
                    common.last_token_pool(outputs.hidden_states[-1], inputs["attention_mask"]).float(), dim=-1)
            np.testing.assert_allclose(together, expected.numpy(), atol=4e-6)
            self.assertEqual(together.dtype, np.float32)

    def test_query_does_not_use_chat_generation_cache_or_all_layer_outputs(self):
        self.query_encoder.instruction = "Custom task:"
        with patch.object(self.query_encoder.backbone, "forward", wraps=self.query_encoder.backbone.forward) as forward, \
             patch.object(self.query_encoder.processor, "apply_chat_template", side_effect=AssertionError("chat")) as chat:
            self.query_encoder.encode_batch([common.Candidate("q", text="short")])
        self.assertFalse(forward.call_args.kwargs["use_cache"])
        self.assertFalse(forward.call_args.kwargs["output_hidden_states"])
        self.assertFalse(hasattr(self.query_encoder.backbone, "lm_head"))
        chat.assert_not_called()

    def helper(self):
        helper = data_fixtures.MMRetDatasetTests()
        helper.setUpClass()
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        return helper

    def test_all_eight_datasets_run_real_candidate_to_query_retrieval(self):
        helper = self.helper()
        for name, (task, _, _) in common.DATASET_PROFILES.items():
            helper.make_source(name)
            directions = ("text", "image") if task == "ir" else (("image", "composed") if task == "it2t" else ("text",))
            for direction in directions:
                with self.subTest(dataset=name, direction=direction):
                    model, data, query, train = arguments(name, direction)
                    model.model_name_or_path = str(self.current)
                    train.encode_type = ("image" if direction == "text" else "text") if task == "ir" else (
                        "image" if task == "tbpr" else "corpus")
                    with patch.object(helper.pd, "read_parquet", side_effect=helper.parquet_frame):
                        candidates = common.build_candidates(data, train, str(helper.root), helper.dataset, model_label="VLM2Vec")
                        matrix = self.encoder.encode_batch([candidates[index] for index in range(len(candidates))])
                        index_dir = helper.root / "index" / name / direction
                        common.save_shard(index_dir, 0, matrix, candidates.ids)
                        query.passage_reps = str(index_dir)
                        with patch.object(search, "write_recall") as write, \
                             patch.object(search.dist, "is_initialized", return_value=False):
                            counts = search.run_dense_search(model, data, query, train,
                                common.DenseEncoderArguments(data_root=str(helper.root), vlm2vec_image_resolution="original"),
                                search.DenseSearchArguments(), "vlm2vec",
                                context_factory=lambda: common.EncodingContext(0, 1, torch.device("cpu")))
                    self.assertGreater(counts[-1], 0)
                    np.testing.assert_array_equal(counts, np.full(8, counts[-1]))
                    write.assert_called_once()
                    self.assertEqual(write.call_args.args[1].name, "vlm2vec_original")

    def test_real_encode_and_search_entrypoints_share_the_saved_index(self):
        import encode_dense
        import search_dense
        helper = self.helper()
        helper.make_source("flickr")
        output = helper.root / "dense-output"
        base = ["--model_name_or_path", str(self.current), "--dense_model_type", "vlm2vec",
                "--dataset_name", "flickr", "--dataset_split", "test", "--task_type", "ir", "--encode_type", "text",
                "--data_root", str(helper.root), "--dense_output_dir", str(output), "--vlm2vec_image_resolution", "low",
                "--output_dir", str(helper.root / "trainer"), "--report_to", "none"]
        distributed_keys = {"WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}
        environment = {key: value for key, value in os.environ.items() if key not in distributed_keys}
        with patch.dict(os.environ, environment, clear=True), patch.object(sys, "argv", ["encode_dense.py", *base]):
            encode_dense.main()
        index = next(output.rglob("corpus_0.pkl")).parent
        results = helper.root / "results"
        argv = ["search_dense.py", *base, "--query_type", "image", "--passage_reps", str(index),
                "--save_dir", str(results), "--use_gpu", "true"]
        with patch.dict(os.environ, environment, clear=True), patch.object(sys, "argv", argv), \
             patch.object(helper.pd.DataFrame, "to_excel", autospec=True) as write:
            search_dense.main()
        frame, target = write.call_args.args
        self.assertEqual(target, results / "dense.xlsx")
        np.testing.assert_allclose(frame.iloc[0].to_numpy(), np.ones(7))

    def test_cpu_reduced_precision_fails_before_loading(self):
        model, data, query, train = arguments()
        query.passage_reps = "/index"
        train.bf16 = True
        with patch.object(search.VLM2VecQueryEncoder, "from_pretrained") as load:
            with self.assertRaisesRegex(ValueError, "CPU.*FP32"):
                search.run_dense_search(model, data, query, train, common.DenseEncoderArguments(),
                    search.DenseSearchArguments(), "vlm2vec",
                    context_factory=lambda: common.EncodingContext(0, 1, torch.device("cpu")),
                    query_builder=lambda *unused: [common.Candidate("q", text="short")])
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
