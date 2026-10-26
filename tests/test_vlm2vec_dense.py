"""Offline VLM2Vec tests: real loaders and native tiny HF weights, no VLM2Vec wrapper."""

from io import BytesIO
import csv
import os
from pathlib import Path
import pickle
import sys
import tempfile
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
import vlm2vec_dense as vlm
import test_mmret_dense as fixtures


def arguments(name="flickr", encode_type=None):
    model, data, train = fixtures.arguments(name, encode_type)
    model.model_name_or_path = vlm.MODEL_ID
    return model, data, train


class VLM2VecUtilityTests(unittest.TestCase):
    def test_model_detection_and_explicit_override(self):
        for path in (vlm.MODEL_ID, "/weights/TIGER-Lab-VLM2Vec-LLaVa-Next/", r"C:\weights\VLM2Vec-LLaVa-Next"):
            self.assertTrue(vlm.is_vlm2vec_model(path))
        for path in ("TIGER-Lab/VLM2Vec-Qwen2VL-7B", "VLM2Vec-V2.0", "VLM2Vec-Full", "clip", "UniME"):
            self.assertFalse(vlm.is_vlm2vec_model(path))
        self.assertTrue(vlm.is_vlm2vec_model("/renamed/checkpoint", "vlm2vec"))
        self.assertFalse(vlm.is_vlm2vec_model(vlm.MODEL_ID, "mmret"))
        with self.assertRaisesRegex(ValueError, "only"):
            vlm.is_vlm2vec_model("VLM2Vec-V2.0", "vlm2vec")

    def test_raw_candidate_templates_do_not_add_chat_or_summary(self):
        self.assertEqual(vlm.candidate_prompt(None, True), "<image> Represent the given image.")
        self.assertEqual(vlm.candidate_prompt("caption", False), "caption")
        self.assertEqual(vlm.candidate_prompt("caption", True), "<image>\ncaption")
        with self.assertRaisesRegex(ValueError, "must contain"):
            vlm.candidate_prompt(None, False)
        with self.assertRaises(TypeError):
            vlm.candidate_prompt(123, False)

    def test_resolution_resize_and_image_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Image.new("L", (19, 53), 100)
            filename = Path(directory) / "person.bmp"
            image.save(filename)
            raw = BytesIO()
            image.save(raw, format="BMP")
            for value in (image, str(filename), raw.getvalue(), {"bytes": raw.getvalue(), "path": "unused"},
                          {"bytes": None, "path": str(filename)}):
                for mode, size in (("high", (1344, 1344)), ("mid", (672, 672)), ("low", (128, 128)),
                                   ("original", (19, 53))):
                    with vlm.load_image(value, mode) as result:
                        self.assertEqual(result.size, size)
                        self.assertEqual(result.mode, "RGB")
            self.assertEqual(image.size, (19, 53))
            with self.assertRaisesRegex(ValueError, "image_resolution"):
                vlm.load_image(image, "invalid")

    def test_resolution_is_part_of_index_path(self):
        for name in common.DATASET_PROFILES:
            model, data, train = arguments(name)
            base = common.output_directory(model, data, train)
            paths = [vlm.vlm2vec_output_directory(model, data, train, resolution) for resolution in vlm.IMAGE_RESOLUTIONS]
            self.assertEqual(len(set(paths)), 4)
            self.assertTrue(all(path.parent == base.parent for path in paths))
            self.assertTrue(paths[0].name.endswith("_vlm2vec_high"))

    def test_parser_routes_before_optional_legacy_backends(self):
        import encode_dense
        for model_path, kind in ((vlm.MODEL_ID, "auto"), ("/renamed/full-weights", "vlm2vec")):
            argv = ["encode_dense.py", "--model_name_or_path", model_path, "--dense_model_type", kind,
                    "--dataset_name", "webqa", "--task_type", "t2it", "--encode_type", "corpus",
                    "--data_root", "/data-root", "--vlm2vec_image_resolution", "low",
                    "--output_dir", "unused", "--report_to", "none"]
            with patch.object(sys, "argv", argv), patch.object(encode_dense, "run_vlm2vec_encoding") as run, \
                 patch.object(encode_dense, "run_unime_encoding") as unime, \
                 patch.object(encode_dense, "run_mmret_encoding") as mmret:
                encode_dense.main()
            run.assert_called_once()
            unime.assert_not_called()
            mmret.assert_not_called()
            self.assertEqual(run.call_args.args[-1].vlm2vec_image_resolution, "low")
        import ast
        tree = ast.parse((PROJECT_ROOT / "src/vlm2vec_dense.py").read_text(encoding="utf-8"))
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertFalse(any(module.startswith(("VLM2Vec", "peft", "mmret_dense", "unime_dense", "template"))
                             for module in imports))

    def test_new_checkpoint_routes_to_native_search_before_legacy(self):
        import search_dense
        for model_path, kind in ((vlm.MODEL_ID, "auto"), ("/local/full", "vlm2vec")):
            argv = ["search_dense.py", "--model_name_or_path", model_path, "--dense_model_type", kind,
                    "--output_dir", "unused", "--report_to", "none"]
            with patch.object(sys, "argv", argv), patch.object(search_dense, "run_dense_search") as run:
                search_dense.main()
            run.assert_called_once()
            self.assertEqual(run.call_args.args[-1], "vlm2vec")

    def test_invalid_query_task_and_resolution_rejected_before_loading(self):
        model, data, train = arguments()
        with patch.object(vlm.VLM2VecEncoder, "from_pretrained") as load:
            data.encode_is_query = True
            with self.assertRaisesRegex(ValueError, "candidates only"):
                vlm.run_vlm2vec_encoding(model, data, train, common.DenseEncoderArguments())
            data.encode_is_query = False
            train.task_type = "it2t"
            with self.assertRaisesRegex(ValueError, "requires --task_type"):
                vlm.run_vlm2vec_encoding(model, data, train, common.DenseEncoderArguments())
            with self.assertRaisesRegex(ValueError, "image_resolution"):
                vlm.run_vlm2vec_encoding(model, data, train,
                    common.DenseEncoderArguments(vlm2vec_image_resolution="invalid"))
        load.assert_not_called()

    def test_disjoint_output_shards_and_empty_worker(self):
        model, data, train = arguments()
        records = [common.Candidate(str(index), text="caption") for index in range(2)]
        encoder = SimpleNamespace(dimension=2,
            encode_batch=lambda batch: np.array([[int(item.identifier), 1] for item in batch], dtype=np.float32))
        with tempfile.TemporaryDirectory() as directory:
            data.dense_output_dir = directory
            for rank in range(3):
                context = common.EncodingContext(rank, 3, torch.device("cpu"))
                with patch.object(vlm, "setup_context", return_value=context), \
                     patch.object(vlm, "build_candidates", return_value=records), \
                     patch.object(vlm.VLM2VecEncoder, "from_pretrained", return_value=encoder) as load, \
                     patch.object(common.dist, "is_initialized", return_value=False):
                    vlm.run_vlm2vec_encoding(model, data, train,
                        common.DenseEncoderArguments(vlm2vec_image_resolution="low"))
                self.assertEqual(load.call_args.kwargs["image_resolution"], "low")
            root = vlm.vlm2vec_output_directory(model, data, train, "low")
            all_ids = []
            for rank in range(3):
                with (root / f"corpus_{rank}.pkl").open("rb") as handle:
                    matrix, ids = pickle.load(handle)
                self.assertEqual(matrix.shape, (len(ids), 2))
                self.assertEqual(matrix.dtype, np.float32)
                all_ids.extend(ids)
            self.assertEqual(sorted(all_ids), ["0", "1"])


class VLM2VecDatasetTests(unittest.TestCase):
    make_source = fixtures.MMRetDatasetTests.make_source
    parquet_frame = fixtures.MMRetDatasetTests.parquet_frame
    setUp = fixtures.MMRetDatasetTests.setUp

    @classmethod
    def setUpClass(cls):
        import dataset
        import pandas as pd
        cls.dataset, cls.pd = dataset, pd

    def build(self, name, encode_type=None, split="test", **kwargs):
        _, data, train = arguments(name, encode_type)
        data.dataset_split = split
        for key, value in kwargs.items():
            setattr(data, key, value)
        old = self.dataset.dataset_path_prefix
        with patch.object(self.pd, "read_parquet", side_effect=self.parquet_frame):
            candidates = common.build_candidates(data, train, str(self.root), self.dataset, model_label="VLM2Vec")
        self.assertEqual(self.dataset.dataset_path_prefix, old)
        return candidates

    def test_all_eight_dataset_modalities_ids_and_images(self):
        for name, (task, _, _) in common.DATASET_PROFILES.items():
            self.make_source(name)
            modes = ("image", "text") if task == "ir" else ("image",) if task == "tbpr" else ("corpus",)
            for mode in modes:
                with self.subTest(dataset=name, mode=mode):
                    candidates = self.build(name, mode)
                    expected = ["caption-1", "caption-2"] if mode == "text" else (
                        ["image-A"] if task == "ir" else ["0"] if task == "tbpr" else [101, 202])
                    self.assertEqual(candidates.ids, expected)
                    for index in range(len(candidates)):
                        sample = candidates[index]
                        prompt = vlm.candidate_prompt(sample.text, sample.image is not None)
                        if sample.image is not None:
                            with vlm.load_image(sample.image, "low") as image:
                                self.assertEqual(image.size, (128, 128))
                            self.assertIn("<image>", prompt)
                        else:
                            self.assertEqual(prompt, sample.text)
                    if name == "webqa":
                        self.assertEqual(candidates[1].image["path"], str(candidates.data_dir / "picture.png"))
                        self.assertNotIn(999, candidates.ids)

    def test_generated_validation_split_and_original_image_fallback(self):
        self.make_source("flickr")
        generated = self.root / "generated/data/flickr"
        generated.mkdir(parents=True)
        (generated / "flickr_val.csv").write_bytes((self.root / "data/flickr/flickr_test.csv").read_bytes())
        candidates = self.build("flickr", "image", "val", use_generated_validation_data=True,
                                generated_validation_data_root=str(self.root / "generated"))
        self.assertEqual(candidates.data_dir, generated)
        self.assertTrue(Path(candidates[0].image).is_file())


class VLM2VecNativeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse only miniature checkpoint construction; VLM2Vec inference is independent.
        fixtures.MMRetNativeModelTests.setUpClass.__func__(cls)
        grids = [[28, 28], [28, 56], [56, 28], [56, 56]]
        cls.model.config.image_grid_pinpoints = grids
        cls.processor.image_processor.image_grid_pinpoints = grids
        cls.model.config.save_pretrained(cls.current)
        cls.processor.save_pretrained(cls.current)
        cls.encoder = vlm.VLM2VecEncoder.from_pretrained(str(cls.current), "cpu", torch.float32,
                                                       image_resolution="original")

    def test_direct_hf_loading_uses_same_repo_and_no_base_or_adapter(self):
        from transformers import AutoConfig, LlavaNextForConditionalGeneration, LlavaNextProcessor
        with patch.object(AutoConfig, "from_pretrained", return_value=self.model.config) as config, \
             patch.object(LlavaNextForConditionalGeneration, "from_pretrained", return_value=(self.model, {})) as load, \
             patch.object(LlavaNextProcessor, "from_pretrained", return_value=self.processor) as processor:
            encoder = vlm.VLM2VecEncoder.from_pretrained(vlm.MODEL_ID, "cpu", torch.float32,
                                                       cache_dir="/hf-cache", image_resolution="mid")
        for call in (config, load, processor):
            self.assertEqual(call.call_args.args, (vlm.MODEL_ID,))
            self.assertEqual(call.call_args.kwargs["cache_dir"], "/hf-cache")
            self.assertFalse(call.call_args.kwargs["trust_remote_code"])
        self.assertTrue(load.call_args.kwargs["use_safetensors"])
        self.assertEqual(encoder.image_resolution, "mid")

    def test_text_image_and_joint_batches_match_individual_and_native_final_layer(self):
        images = [Image.new("RGB", (28, 28), "red"), Image.new("RGB", (56, 28), "blue")]
        for mode in ("text", "image", "joint"):
            batch = [common.Candidate(index, text=("short" if index == 0 else "long text long text")
                                      if mode != "image" else None,
                                      image=images[index] if mode != "text" else None) for index in range(2)]
            together = self.encoder.encode_batch(batch)
            individual = np.concatenate([self.encoder.encode_batch([item]) for item in batch])
            np.testing.assert_allclose(together, individual, atol=4e-6)
            np.testing.assert_allclose(np.linalg.norm(together, axis=1), [1, 1], atol=1e-6)
            prompts = [vlm.candidate_prompt(item.text, item.image is not None) for item in batch]
            inputs = self.processor(text=prompts, images=images if mode != "text" else None,
                                    padding=True, return_tensors="pt")
            inputs["position_ids"] = (inputs["attention_mask"].cumsum(-1) - 1).clamp_min(0)
            with torch.inference_mode():
                result = self.model(**inputs, output_hidden_states=True, use_cache=False, logits_to_keep=1)
                expected = torch.nn.functional.normalize(
                    common.last_token_pool(result.hidden_states[-1], inputs["attention_mask"]).float(), dim=-1)
            np.testing.assert_allclose(together, expected.numpy(), atol=4e-6)
            self.assertEqual(together.dtype, np.float32)

    def test_no_generation_chat_cache_or_all_layer_language_outputs(self):
        with patch.object(self.encoder.backbone, "forward", wraps=self.encoder.backbone.forward) as forward, \
             patch.object(self.encoder.processor, "apply_chat_template", side_effect=AssertionError("chat wrapper")) as chat:
            self.encoder.encode_batch([common.Candidate("a", text="short")])
        self.assertFalse(forward.call_args.kwargs["use_cache"])
        self.assertFalse(forward.call_args.kwargs["output_hidden_states"])
        self.assertFalse(hasattr(self.encoder.backbone, "lm_head"))
        chat.assert_not_called()

    def test_wrong_architecture_and_incomplete_weights_fail(self):
        from transformers import AutoConfig, LlavaNextForConditionalGeneration
        with patch.object(AutoConfig, "from_pretrained", return_value=SimpleNamespace(model_type="qwen2_vl")), \
             patch.object(LlavaNextForConditionalGeneration, "from_pretrained") as load:
            with self.assertRaisesRegex(ValueError, "full LLaVA-NeXT"):
                vlm.VLM2VecEncoder.from_pretrained("/wrong-model", "cpu", torch.float32)
        load.assert_not_called()
        for info in ({"missing_keys": ["model.image_newline"]}, {"mismatched_keys": ["vision_tower"]},
                     {"unexpected_keys": ["adapter.weight"]}):
            with patch.object(LlavaNextForConditionalGeneration, "from_pretrained", return_value=(self.model, info)):
                with self.assertRaisesRegex(ValueError, "did not load completely"):
                    vlm.VLM2VecEncoder.from_pretrained(str(self.current), "cpu", torch.float32)

    def test_original_checkpoint_key_conversion(self):
        from safetensors.torch import save_file
        state = {}
        for key, tensor in self.model.state_dict().items():
            if key.startswith("model.language_model."):
                key = key.replace("model.language_model.", "language_model.model.", 1)
            elif key.startswith("model."):
                key = key.removeprefix("model.")
            elif key == "lm_head.weight":
                key = "language_model.lm_head.weight"
            state[key] = tensor
        directory = self.root / "vlm2vec-old-keys"
        self.model.config.save_pretrained(directory)
        save_file(state, str(directory / "model.safetensors"), metadata={"format": "pt"})
        self.processor.save_pretrained(directory)
        encoder = vlm.VLM2VecEncoder.from_pretrained(str(directory), "cpu", torch.float32, image_resolution="original")
        batch = [common.Candidate("a", text="short", image=Image.new("RGB", (28, 56)))]
        np.testing.assert_allclose(encoder.encode_batch(batch), self.encoder.encode_batch(batch), atol=3e-6)

    def test_bad_input_and_context_overflow_keep_candidate_ids(self):
        self.assertEqual(self.encoder.encode_batch([]).shape, (0, 16))
        with self.assertRaisesRegex(ValueError, "missing-id"):
            self.encoder.encode_batch([common.Candidate("missing-id")])
        with self.assertRaisesRegex(ValueError, "consistent image"):
            self.encoder.encode_batch([common.Candidate("a", text="short"),
                                       common.Candidate("b", image=Image.new("RGB", (28, 28)))])
        with self.assertRaisesRegex(ValueError, "Candidate IDs.*overflow"):
            self.encoder.encode_batch([common.Candidate("overflow", text="long " * 300)])

    def test_real_cli_with_tiny_weights_and_flickr(self):
        import encode_dense
        directory = self.root / "entry-data/data/flickr"
        directory.mkdir(parents=True)
        with (directory / "flickr_test.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["imgid"])
            writer.writerow(["img", "not-needed.png", "short", "t1"])
            writer.writerow(["img", "not-needed.png", "long text", "t2"])
        output = self.root / "entry-output"
        argv = ["encode_dense.py", "--model_name_or_path", str(self.current), "--dense_model_type", "vlm2vec",
                "--dataset_name", "flickr", "--dataset_split", "test", "--task_type", "ir", "--encode_type", "text",
                "--data_root", str(self.root / "entry-data"), "--dense_output_dir", str(output),
                "--vlm2vec_image_resolution", "low", "--output_dir", str(self.root / "trainer"), "--report_to", "none"]
        distributed_keys = {"WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}
        environment = {key: value for key, value in os.environ.items() if key not in distributed_keys}
        with patch.object(sys, "argv", argv), patch.dict(os.environ, environment, clear=True):
            encode_dense.main()
        files = list(output.rglob("corpus_0.pkl"))
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].parent.name.endswith("_vlm2vec_low"))
        with files[0].open("rb") as handle:
            matrix, ids = pickle.load(handle)
        self.assertEqual(ids, ["t1", "t2"])
        self.assertEqual(matrix.shape, (2, 16))
        np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), [1, 1], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
