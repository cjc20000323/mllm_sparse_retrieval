"""Offline UniME integration tests, sharing tiny dataset/checkpoint fixtures only."""

from io import BytesIO
import csv
import os
from pathlib import Path
import pickle
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
import mmret_dense as mmret
import unime_dense as unime
import test_mmret_dense as fixtures


class UniMEUtilityTests(unittest.TestCase):
    def test_model_detection_and_explicit_arbitrary_directory(self):
        for path in (unime.MODEL_ID, "/models/DeepGlint-AI-UniME-LLaVA-1.6-7B/",
                     r"C:\weights\UniME-LLaVA-1.6-7B", "UniME_LLaVA_1.6_7B"):
            self.assertTrue(unime.is_unime_model(path), path)
        self.assertTrue(unime.is_unime_model("/custom/weights", "unime"))
        for name in ("clip", "gme", "LamRA", "VLM2Vec", "BAAI/BGE-VL-MLLM-S1"):
            self.assertFalse(unime.is_unime_model(name), name)
        self.assertFalse(unime.is_unime_model(unime.MODEL_ID, "mmret"))
        with self.assertRaisesRegex(ValueError, "auto, mmret or unime"):
            unime.is_unime_model("weights", "bad")

    def test_unsupported_families_do_not_fall_through_to_clip(self):
        for name in ("UniME-V2_qwen2VL_2B", "UniME-LLaVA-OV-7B", "UniME-LLaVA-OneVision-7B",
                     "UniME-Phi3.5-V-4.2B", "UniME-R1-4B"):
            for mode in ("auto", "unime"):
                with self.subTest(name=name, mode=mode), self.assertRaisesRegex(ValueError, "Only DeepGlint-AI"):
                    unime.is_unime_model(name, mode)

    def test_official_and_joint_extension_templates(self):
        self.assertEqual(unime.candidate_prompt(None, True),
                         "[INST] <image>\nSummary above image in one word: [/INST]")
        self.assertEqual(unime.candidate_prompt("caption", False),
                         "[INST] caption\nSummary above sentence in one word: [/INST]")
        joint = unime.candidate_prompt("caption", True)
        self.assertEqual(joint, "[INST] <image>\ncaption\nSummary above image and text in one word: [/INST]")
        self.assertNotEqual(joint, mmret.candidate_prompt("caption", True))
        self.assertEqual(joint.count("<image>"), 1)
        with self.assertRaises(ValueError):
            unime.candidate_prompt(None, False)
        with self.assertRaises(TypeError):
            unime.candidate_prompt(123, False)

    def test_image_decode_preserves_dimensions_and_mmret_still_resizes(self):
        image = Image.new("L", (19, 53), 100)
        raw = BytesIO()
        image.save(raw, format="BMP")
        for value in (image, raw.getvalue(), {"bytes": raw.getvalue(), "path": "unused"}):
            with unime.load_image(value) as decoded:
                self.assertEqual(decoded.size, (19, 53))
                self.assertEqual(decoded.mode, "RGB")
            with mmret.load_image(value) as decoded:
                self.assertEqual(decoded.size, (512, 512))
        self.assertEqual(image.size, (19, 53))
        self.assertEqual(image.mode, "L")

    def test_query_flag_and_bad_configuration_rejected_before_loading(self):
        model, data, train = fixtures.arguments()
        model.model_name_or_path = unime.MODEL_ID
        data.encode_is_query = True
        with patch.object(unime.UniMEEncoder, "from_pretrained") as load:
            with self.assertRaisesRegex(ValueError, "UniME encode_dense supports candidates only"):
                unime.run_unime_encoding(model, data, train, common.DenseEncoderArguments())
        load.assert_not_called()

    def test_parser_auto_and_explicit_dispatch_without_mmret(self):
        import encode_dense
        for model_path, mode in ((unime.MODEL_ID, "auto"), ("/arbitrary/model", "unime")):
            argv = ["encode_dense.py", "--model_name_or_path", model_path, "--dense_model_type", mode,
                    "--dataset_name", "webqa", "--task_type", "t2it", "--encode_type", "corpus",
                    "--data_root", "/data-root", "--output_dir", "unused", "--report_to", "none"]
            with patch.object(sys, "argv", argv), patch.object(encode_dense, "run_unime_encoding") as run, \
                 patch.object(encode_dense, "run_mmret_encoding") as old_run:
                encode_dense.main()
            old_run.assert_not_called()
            model, data, train, dense = run.call_args.args
            self.assertEqual((data.dataset_name, train.task_type, train.encode_type), ("webqa", "t2it", "corpus"))
            self.assertEqual(dense.data_root, "/data-root")

    def test_empty_shard_and_misaligned_output_use_shared_guard(self):
        encoder = SimpleNamespace(dimension=16)
        matrix, ids = common.encode_shard(encoder, [], [], 1, "unime empty")
        self.assertEqual(matrix.shape, (0, 16))
        self.assertEqual(matrix.dtype, np.float32)
        self.assertEqual(ids, [])
        encoder.encode_batch = lambda batch: np.zeros((len(batch) + 1, 16))
        with self.assertRaisesRegex(ValueError, "output shape"):
            common.encode_shard(encoder, [common.Candidate("x", text="a")], [0], 1, "bad")


class UniMEDatasetTests(unittest.TestCase):
    setUp = fixtures.MMRetDatasetTests.setUp
    make_source = fixtures.MMRetDatasetTests.make_source
    parquet_frame = fixtures.MMRetDatasetTests.parquet_frame

    @classmethod
    def setUpClass(cls):
        fixtures.MMRetDatasetTests.setUpClass.__func__(cls)

    def build(self, name, encode_type=None):
        _, data, train = fixtures.arguments(name, encode_type)
        old = self.dataset.dataset_path_prefix
        with patch.object(self.pd, "read_parquet", side_effect=self.parquet_frame):
            candidates = common.build_candidates(data, train, str(self.root), self.dataset, model_label="UniME")
        self.assertEqual(self.dataset.dataset_path_prefix, old)
        return candidates

    def test_all_eight_dataset_ids_and_modalities(self):
        for name in common.DATASET_PROFILES:
            with self.subTest(dataset=name):
                self.make_source(name)
                candidates = self.build(name)
                self.assertGreater(len(candidates), 0)
                sample = candidates[0]
                if name in ("coco", "flickr", "CUHK-PEDES", "ICFG-PEDES", "RSTPReid"):
                    self.assertIsNone(sample.text)
                    with unime.load_image(sample.image) as image:
                        self.assertEqual(image.size, (10, 20))
                    expected_id = "image-A" if name in ("coco", "flickr") else "0"
                    self.assertEqual(sample.identifier, expected_id)
                else:
                    self.assertEqual(candidates.ids, [101, 202])
                    self.assertIsInstance(sample.text, str)
                    self.assertEqual(sample.image is not None, name == "webqa")
                self.assertEqual(len(set(candidates.ids)), len(candidates))

    def test_ir_text_candidates_and_webqa_joint_extension(self):
        for name in ("coco", "flickr"):
            self.make_source(name)
            candidates = self.build(name, "text")
            self.assertEqual(candidates.ids, ["caption-1", "caption-2"])
            self.assertIsNone(candidates[0].image)
            self.assertIn("Summary above sentence", unime.candidate_prompt(candidates[0].text, False))
        self.make_source("webqa")
        candidates = self.build("webqa")
        for index in range(len(candidates)):
            sample = candidates[index]
            with unime.load_image(sample.image) as image:
                self.assertEqual(image.size, (16, 32))
            prompt = unime.candidate_prompt(sample.text, True)
            self.assertIn(sample.text, prompt)
            self.assertIn("<image>", prompt)
        self.assertNotIn(999, candidates.ids)  # Retain the existing WebQA qrels filtering.


class UniMENativeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Only reuse fixture construction, not the MMRet test methods/encoder.
        fixtures.MMRetNativeModelTests.setUpClass.__func__(cls)
        grids = [[28, 28], [28, 56], [56, 28], [56, 56]]
        cls.model.config.image_grid_pinpoints = grids
        cls.processor.image_processor.image_grid_pinpoints = grids
        cls.model.config.save_pretrained(cls.current)
        cls.processor.save_pretrained(cls.current)
        cls.encoder = unime.UniMEEncoder.from_pretrained(str(cls.current), "cpu", torch.float32)

    def test_native_text_batches_match_individual_and_final_layer(self):
        batch = [common.Candidate("a", text="short"), common.Candidate("b", text="long text long text")]
        together = self.encoder.encode_batch(batch)
        single = np.concatenate([self.encoder.encode_batch([item]) for item in batch])
        np.testing.assert_allclose(together, single, atol=3e-6)
        inputs = self.encoder.processor(text=[unime.candidate_prompt(item.text, False) for item in batch],
                                        padding=True, return_tensors="pt")
        inputs["position_ids"] = (inputs["attention_mask"].cumsum(-1) - 1).clamp_min(0)
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True, use_cache=False, logits_to_keep=1)
            expected = torch.nn.functional.normalize(common.last_token_pool(outputs.hidden_states[-1], inputs["attention_mask"]), dim=-1)
        np.testing.assert_allclose(together, expected.numpy(), atol=3e-6)
        np.testing.assert_allclose(np.linalg.norm(together, axis=1), np.ones(2), atol=1e-6)
        self.assertEqual(together.dtype, np.float32)

    def test_image_and_joint_anyres_batches_with_different_crop_counts(self):
        images = [Image.new("RGB", (28, 28), "red"), Image.new("RGB", (56, 28), "blue")]
        for texts in ((None, None), ("short", "long text long text")):
            batch = [common.Candidate(i, text=text, image=image) for i, (text, image) in enumerate(zip(texts, images))]
            with patch.object(self.encoder, "processor", wraps=self.encoder.processor) as processor:
                together = self.encoder.encode_batch(batch)
                seen = processor.call_args.kwargs["images"]
                self.assertEqual([image.size for image in seen], [(28, 28), (56, 28)])
            single = np.concatenate([self.encoder.encode_batch([item]) for item in batch])
            self.assertEqual(together.shape, (2, 16))
            np.testing.assert_allclose(together, single, atol=3e-6)

    def test_legacy_weight_conversion(self):
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
        directory = self.root / "unime-legacy"
        self.model.config.save_pretrained(directory)
        save_file(state, str(directory / "model.safetensors"), metadata={"format": "pt"})
        self.processor.save_pretrained(directory)
        encoder = unime.UniMEEncoder.from_pretrained(str(directory), "cpu", torch.float32)
        batch = [common.Candidate("a", text="short", image=Image.new("RGB", (28, 56)))]
        np.testing.assert_allclose(encoder.encode_batch(batch), self.encoder.encode_batch(batch), atol=2e-6)

    def test_no_cache_all_layer_outputs_or_lm_head(self):
        with patch.object(self.encoder.backbone, "forward", wraps=self.encoder.backbone.forward) as forward:
            self.encoder.encode_batch([common.Candidate("a", text="short")])
        self.assertFalse(forward.call_args.kwargs["use_cache"])
        self.assertFalse(forward.call_args.kwargs["output_hidden_states"])
        self.assertFalse(hasattr(self.encoder.backbone, "lm_head"))

    def test_context_error_and_bad_inputs_report_unime_ids(self):
        with self.assertRaisesRegex(ValueError, "UniME input exceeds context"):
            self.encoder.encode_batch([common.Candidate("long", text="long " * 300)])
        with self.assertRaisesRegex(ValueError, "UniME preprocessing failed.*bad-image"):
            self.encoder.encode_batch([common.Candidate("bad-image", image=b"not an image")])

    def test_empty_rank_shards_and_id_coverage(self):
        records = [common.Candidate("a", text="short"), common.Candidate("b", text="long text")]
        full = self.encoder.encode_batch(records)
        found = {}
        for rank in range(3):
            embeddings, ids = common.encode_shard(self.encoder, records, common.shard_indices(2, rank, 3), 2, "UniME split")
            target = common.save_shard(self.root / "shards", rank, embeddings, ids)
            with target.open("rb") as handle:
                saved, saved_ids = pickle.load(handle)
            self.assertEqual(saved_ids, ids)
            self.assertEqual(saved.shape, (len(ids), 16))
            for identifier, row in zip(ids, saved):
                self.assertNotIn(identifier, found)
                found[identifier] = row
        np.testing.assert_allclose(np.stack([found[item.identifier] for item in records]), full, atol=3e-6)

    def test_complete_entrypoint_with_local_weights_and_real_flickr_loader(self):
        import encode_dense
        directory = self.root / "data-root/data/flickr"
        directory.mkdir(parents=True)
        with (directory / "flickr_test.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["imgid"])
            writer.writerow(["image", "unused.png", "short", "caption-A"])
            writer.writerow(["image", "unused.png", "long text", "caption-B"])
        output = self.root / "unime-output"
        argv = ["encode_dense.py", "--model_name_or_path", str(self.current), "--dense_model_type", "unime",
                "--dataset_name", "flickr", "--dataset_split", "test", "--task_type", "ir", "--encode_type", "text",
                "--data_root", str(self.root / "data-root"), "--dense_output_dir", str(output),
                "--per_device_batch_size", "2", "--report_to", "none", "--output_dir", str(self.root / "trainer")]
        worker_keys = {"RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}
        plain_env = {key: value for key, value in os.environ.items() if key not in worker_keys}
        with patch.object(sys, "argv", argv), patch.dict(os.environ, plain_env, clear=True):
            encode_dense.main()
        targets = list(output.rglob("corpus_0.pkl"))
        self.assertEqual(len(targets), 1)
        with targets[0].open("rb") as handle:
            embeddings, ids = pickle.load(handle)
        self.assertEqual(ids, ["caption-A", "caption-B"])
        self.assertEqual(embeddings.shape, (2, 16))


if __name__ == "__main__":
    unittest.main()
