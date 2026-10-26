"""Offline CPU tests: real dataset loaders, native tiny LLaVA, and fake encoding."""

from __future__ import annotations

import ast
import csv
from io import BytesIO
import json
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

import mmret_dense as mmret


def arguments(name="flickr", encode_type=None):
    from arguments import ModelArguments, PromptRepsLLMDataArguments
    task = mmret.DATASET_PROFILES[name][0]
    encode_type = encode_type or ("image" if task in ("ir", "tbpr") else "corpus")
    return (ModelArguments(model_name_or_path="BAAI/BGE-VL-MLLM-S1"),
            PromptRepsLLMDataArguments(dataset_name=name, dataset_split="test", per_device_batch_size=2),
            SimpleNamespace(task_type=task, encode_type=encode_type, fp16=False, bf16=False))


class MMRetUtilityTests(unittest.TestCase):
    def test_detection_and_explicit_override(self):
        for name in ("JUNJIE99/MMRet-MLLM-S1", "/models/BAAI-BGE-VL-MLLM-S2/", r"C:\models\MMRet-MLLM-S1"):
            self.assertTrue(mmret.is_mmret_model(name))
        for name in ("clip", "MMRet-base", "BAAI/BGE-VL-large", "gme", "LamRA", "VLM2Vec"):
            self.assertFalse(mmret.is_mmret_model(name))
        self.assertTrue(mmret.is_mmret_model("/arbitrary/checkpoint", "mmret"))
        with self.assertRaisesRegex(ValueError, "auto or mmret"):
            mmret.is_mmret_model("x", "other")

    def test_validate_all_profiles_and_invalid_task_modality(self):
        for name in mmret.DATASET_PROFILES:
            _, data, train = arguments(name)
            mmret.validate_task(name, train.task_type, train.encode_type)
            with self.assertRaisesRegex(ValueError, "candidates only"):
                mmret.validate_task(name, train.task_type, train.encode_type, True)
            with self.assertRaisesRegex(ValueError, "requires --task_type"):
                mmret.validate_task(name, "cir", train.encode_type)
            with self.assertRaisesRegex(ValueError, "requires --encode_type"):
                mmret.validate_task(name, train.task_type, "wrong")
        mmret.validate_task("coco", "ir", "text")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            mmret.validate_task("other", "ir", "image")

    def test_candidate_template_matches_official_whitespace(self):
        self.assertEqual(mmret.candidate_prompt(None, True), "[INST]  <image> [/INST]")
        self.assertEqual(mmret.candidate_prompt("caption", False), "[INST]  caption \n [/INST]")
        self.assertEqual(mmret.candidate_prompt("caption", True), "[INST]  caption \n <image> [/INST]")
        self.assertNotIn("<query>", mmret.candidate_prompt("caption", True))

    def test_last_valid_token_for_both_padding_sides(self):
        hidden = torch.arange(24).reshape(2, 4, 3).float()
        for mask, positions in (([[0, 0, 1, 1], [0, 1, 1, 1]], [3, 3]),
                                ([[1, 1, 0, 0], [1, 1, 1, 0]], [1, 2])):
            pooled = mmret.last_token_pool(hidden, torch.tensor(mask))
            torch.testing.assert_close(pooled, hidden[torch.arange(2), positions])
        with self.assertRaises(ValueError):
            mmret.last_token_pool(hidden, torch.zeros(2, 4))

    def test_image_formats_and_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            bmp = Path(directory) / "pedestrian.bmp"
            image = Image.new("RGB", (18, 35), "blue")
            image.save(bmp)
            raw = BytesIO()
            image.save(raw, format="PNG")
            for source in (bmp, str(bmp), raw.getvalue(), memoryview(raw.getvalue()), image,
                           {"bytes": raw.getvalue(), "path": "missing.png"}, {"bytes": None, "path": str(bmp)}):
                with self.subTest(source=type(source)):
                    decoded = mmret.load_image(source)
                    self.assertEqual(decoded.size, (512, 512))
                    self.assertEqual(decoded.mode, "RGB")
                    decoded.close()
            self.assertEqual(image.size, (18, 35))
            with self.assertRaisesRegex(ValueError, "Missing"):
                mmret.load_image({"bytes": None, "path": None})

    def test_disjoint_shards_including_empty_workers(self):
        for length in (0, 1, 5, 8):
            for world in (1, 2, 7):
                shards = [list(mmret.shard_indices(length, rank, world)) for rank in range(world)]
                flattened = [item for shard in shards for item in shard]
                self.assertEqual(sorted(flattened), list(range(length)))
                self.assertEqual(len(flattened), len(set(flattened)))

    def test_single_process_never_requires_default_process_group(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"}), \
             patch.object(mmret.dist, "is_initialized", return_value=False), \
             patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(mmret.dist, "get_rank") as rank, \
             patch.object(mmret.dist, "get_world_size") as world, \
             patch.object(mmret.dist, "init_process_group") as initialize:
            context = mmret.setup_context()
        self.assertEqual((context.rank, context.world_size, context.device.type), (0, 1, "cpu"))
        rank.assert_not_called()
        world.assert_not_called()
        initialize.assert_not_called()

    def test_multi_process_selects_local_gpu_and_nccl(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "1"}), \
             patch.object(mmret.dist, "is_initialized", return_value=False), \
             patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "device_count", return_value=2), \
             patch.object(torch.cuda, "set_device") as set_device, \
             patch.object(mmret.dist, "init_process_group") as initialize:
            context = mmret.setup_context()
        self.assertEqual(str(context.device), "cuda:1")
        self.assertTrue(context.owns_process_group)
        set_device.assert_called_once_with(torch.device("cuda:1"))
        initialize.assert_called_once_with("nccl")

    def test_reuses_initialized_process_group(self):
        with patch.object(mmret.dist, "is_initialized", return_value=True), \
             patch.object(mmret.dist, "get_rank", return_value=1), \
             patch.object(mmret.dist, "get_world_size", return_value=2), \
             patch.dict(os.environ, {"LOCAL_RANK": "0"}), \
             patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(mmret.dist, "init_process_group") as initialize:
            context = mmret.setup_context()
        self.assertEqual((context.rank, context.world_size), (1, 2))
        self.assertFalse(context.owns_process_group)
        initialize.assert_not_called()

    def test_owned_group_is_cleaned_up_on_dataset_error(self):
        model, data, train = arguments()
        context = mmret.EncodingContext(1, 2, torch.device("cpu"), owns_process_group=True)
        with patch.object(mmret, "setup_context", return_value=context), \
             patch.object(mmret, "build_candidates", side_effect=ValueError("invalid dataset")), \
             patch.object(mmret.dist, "is_initialized", return_value=True), \
             patch.object(mmret.dist, "destroy_process_group") as destroy, \
             patch.object(mmret.dist, "barrier") as barrier:
            with self.assertRaisesRegex(ValueError, "invalid dataset"):
                mmret.run_mmret_encoding(model, data, train, mmret.DenseEncoderArguments())
        destroy.assert_called_once()
        barrier.assert_not_called()

    def test_output_path_keeps_hierarchy_without_fixed_slice(self):
        for name in ("coco", "CUHK-PEDES", "webqa", "remuq"):
            model, data, train = arguments(name)
            data.dense_output_dir = "dense"
            model.model_name_or_path = "/short/S1"
            directory = mmret.output_directory(model, data, train)
            self.assertEqual(directory.parts[:3], ("dense", "S1", name))
            self.assertIn("test", directory.parts)
            if train.task_type == "tbpr":
                self.assertIn("origin_type", directory.parts)
            self.assertTrue(directory.name.endswith("prompteol_lower_no_mean"))

    def test_pickle_alignment_atomic_write_and_stale_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            matrix = np.array([[1, 0], [0, 1]], dtype=np.float16)
            target = mmret.save_shard(directory, 0, matrix, [17, "text-2"])
            with target.open("rb") as handle:
                embeddings, ids = pickle.load(handle)
            self.assertEqual(embeddings.dtype, np.float32)
            self.assertEqual(ids, [17, "text-2"])
            self.assertEqual(list(directory.iterdir()), [target])
            mmret.check_existing_shards(directory, 1)
            mmret.save_shard(directory, 1, np.empty((0, 2)), [])
            with self.assertRaisesRegex(ValueError, "Stale shard"):
                mmret.check_existing_shards(directory, 1)
            mmret.check_existing_shards(directory, 2)
            with self.assertRaisesRegex(ValueError, "misaligned"):
                mmret.save_shard(directory, 0, matrix, [17])

    def test_encode_shards_batches_and_empty_output(self):
        class Encoder:
            dimension = 2

            def encode_batch(self, batch):
                return np.array([[float(item.identifier), 1.0] for item in batch], dtype=np.float32)

        candidates = [mmret.Candidate(index, text=str(index)) for index in range(5)]
        all_ids = []
        for rank in range(3):
            reps, ids = mmret.encode_shard(Encoder(), candidates, mmret.shard_indices(5, rank, 3), 2, "test")
            np.testing.assert_array_equal(reps[:, 0], ids)
            all_ids.extend(ids)
        self.assertEqual(sorted(all_ids), list(range(5)))
        reps, ids = mmret.encode_shard(Encoder(), candidates, [], 2, "empty")
        self.assertEqual(reps.shape, (0, 2))
        self.assertEqual(ids, [])

    def test_entry_routes_mmret_before_optional_legacy_imports(self):
        source = (PROJECT_ROOT / "src/encode_dense.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        route = next(node for node in main.body if isinstance(node, ast.If) and "is_mmret_model" in ast.unparse(node.test))
        legacy = [node for node in main.body if isinstance(node, ast.ImportFrom) and node.module in
                  ("eva_clip", "models.blip_itm", "sentence_transformers", "peft")]
        self.assertEqual(len(legacy), 4)
        self.assertTrue(all(node.lineno > route.end_lineno for node in legacy))
        self.assertIsInstance(route.body[-1], ast.Return)

    def test_entry_parser_accepts_mmret_options_and_dispatches(self):
        import encode_dense
        argv = ["encode_dense.py", "--model_name_or_path", "/custom/model", "--dense_model_type", "mmret",
                "--dataset_name", "webqa", "--task_type", "t2it", "--encode_type", "corpus",
                "--data_root", "/custom/data", "--report_to", "none", "--output_dir", "unused"]
        with patch.object(sys, "argv", argv), patch.object(encode_dense, "run_mmret_encoding") as run:
            encode_dense.main()
        model, data, train, dense = run.call_args.args
        self.assertEqual(model.model_name_or_path, "/custom/model")
        self.assertEqual((data.dataset_name, train.task_type, train.encode_type), ("webqa", "t2it", "corpus"))
        self.assertEqual(dense.data_root, "/custom/data")

    def test_legacy_clip_loading_stays_on_original_branch(self):
        import encode_dense
        stub_modules = {"models.blip_itm": SimpleNamespace(blip_itm=None),
                        "eva_clip": SimpleNamespace(create_model_and_transforms=None, get_tokenizer=None),
                        "sentence_transformers": SimpleNamespace(SentenceTransformer=None),
                        "peft": SimpleNamespace(PeftModel=None)}
        argv = ["encode_dense.py", "--model_name_or_path", "example/clip-model", "--report_to", "none",
                "--output_dir", "unused"]
        with patch.object(sys, "argv", argv), patch.dict(sys.modules, stub_modules), \
             patch.dict(os.environ, {"WORLD_SIZE": "1"}), \
             patch.object(encode_dense.CLIPModel, "from_pretrained", side_effect=RuntimeError("legacy loader reached")) as load:
            with self.assertRaisesRegex(RuntimeError, "legacy loader reached"):
                encode_dense.main()
        self.assertEqual(load.call_args.args, ("example/clip-model",))


class MMRetDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import dataset
        import pandas as pd
        cls.dataset = dataset
        cls.pd = pd

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        raw = BytesIO()
        Image.new("RGB", (16, 32), "green").save(raw, format="PNG")
        self.image_bytes = raw.getvalue()

    def make_source(self, name):
        directory = self.root / "data" / mmret.DATASET_PROFILES[name][1]
        directory.mkdir(parents=True)
        if name in ("coco", "flickr"):
            image_dir = directory / ("val2014" if name == "coco" else "flickr30k-images")
            image_dir.mkdir()
            Image.new("RGB", (10, 20)).save(image_dir / "sample.png")
            with (directory / f"{name}_test.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["imgid"])
                for text_id in ("caption-1", "caption-2"):
                    row = (["image-A", "val2014", "sample.png", text_id, text_id] if name == "coco" else
                           ["image-A", "sample.png", text_id, text_id])
                    writer.writerow(row)
        elif mmret.DATASET_PROFILES[name][0] == "tbpr":
            (directory / "imgs").mkdir()
            Image.new("RGB", (10, 20)).save(directory / "imgs/person.bmp")
            key = "img_path" if name == "RSTPReid" else "file_path"
            records = [{"split": "test", "id": 19, key: "person.bmp", "captions": ["person caption", "other caption"]},
                       {"split": "train", "id": 99, key: "excluded.bmp", "captions": ["exclude"]}]
            filename = {"CUHK-PEDES": "reid_raw.json", "ICFG-PEDES": "ICFG-PEDES.json",
                        "RSTPReid": "data_captions.json"}[name]
            (directory / filename).write_text(json.dumps(records), encoding="utf-8")
        else:
            (directory / "picture.png").write_bytes(self.image_bytes)
        return directory

    def parquet_frame(self, path):
        filename = Path(path).name
        if filename.startswith("query"):
            return self.pd.DataFrame([{"id": "q", "text": "query", "image": {"bytes": self.image_bytes}}])
        if filename.startswith("qrels"):
            return self.pd.DataFrame([{"query-id": "q", "corpus-id": 101}, {"query-id": "q", "corpus-id": 202}])
        if filename == "corpus-00000-of-00002.parquet":
            return self.pd.DataFrame([{"id": 101, "text": "paired text", "image": {"bytes": self.image_bytes}},
                                      {"id": 999, "text": "existing loader excludes this", "image": {"bytes": self.image_bytes}}])
        if filename == "corpus-00001-of-00002.parquet":
            return self.pd.DataFrame([{"id": 202, "text": "second pair", "image": {"bytes": None, "path": "picture.png"}}])
        return self.pd.DataFrame([{"id": 101, "text": "candidate text"}, {"id": 202, "text": "longer candidate text"}])

    def build(self, name, encode_type=None):
        _, data, train = arguments(name, encode_type)
        old_root = self.dataset.dataset_path_prefix
        old_path = getattr(self.dataset, mmret.DATASET_PROFILES[name][2])
        with patch.object(self.pd, "read_parquet", side_effect=self.parquet_frame):
            candidates = mmret.build_candidates(data, train, str(self.root), self.dataset)
        self.assertEqual(self.dataset.dataset_path_prefix, old_root)
        self.assertEqual(getattr(self.dataset, mmret.DATASET_PROFILES[name][2]), old_path)
        return candidates

    def test_coco_flickr_both_candidate_modalities(self):
        for name in ("coco", "flickr"):
            self.make_source(name)
            images = self.build(name, "image")
            self.assertEqual(len(images), 1)
            self.assertEqual(images[0].identifier, "image-A")
            self.assertTrue(Path(images[0].image).is_file())
            self.assertIsNone(images[0].text)
            texts = self.build(name, "text")
            self.assertEqual([texts[i].identifier for i in range(len(texts))], ["caption-1", "caption-2"])
            self.assertEqual(texts[0].text, "caption-1")
            self.assertIsNone(texts[0].image)

    def test_all_three_tbpr_datasets_keep_native_ids_and_split(self):
        for name in ("CUHK-PEDES", "ICFG-PEDES", "RSTPReid"):
            self.make_source(name)
            candidates = self.build(name)
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0].identifier, "0")
            self.assertIsNone(candidates[0].text)
            image = mmret.load_image(candidates[0].image)
            self.assertEqual(image.size, (512, 512))
            image.close()

    def test_webqa_joint_records_bytes_paths_and_existing_qrels_selection(self):
        self.make_source("webqa")
        candidates = self.build("webqa")
        self.assertEqual(candidates.ids, [101, 202])
        for index in range(2):
            candidate = candidates[index]
            self.assertIsInstance(candidate.text, str)
            image = mmret.load_image(candidate.image)
            self.assertEqual(image.size, (512, 512))
            image.close()
        self.assertEqual(candidates[0].text, "paired text")
        self.assertEqual(candidates[1].image["path"], str(candidates.data_dir / "picture.png"))

    def test_remuq_llava_text_corpora(self):
        for name in ("remuq", "llava"):
            self.make_source(name)
            candidates = self.build(name)
            self.assertEqual(candidates.ids, [101, 202])
            self.assertEqual(candidates[0].text, "candidate text")
            self.assertIsNone(candidates[0].image)

    def test_constructor_failure_restores_paths(self):
        _, data, train = arguments()
        old = self.dataset.flickr_file_path
        with self.assertRaises(FileNotFoundError):
            mmret.build_candidates(data, train, str(self.root), self.dataset)
        self.assertEqual(self.dataset.flickr_file_path, old)


class MMRetNativeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from safetensors.torch import save_file
        from transformers import (LlavaNextConfig, LlavaNextForConditionalGeneration, LlavaNextImageProcessor,
                                  LlavaNextProcessor, PreTrainedTokenizerFast)

        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        tokenizer = Tokenizer(WordLevel({"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3, "<image>": 4,
                                        "[INST]": 5, "[/INST]": 6, "short": 7, "long": 8, "text": 9}, unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", bos_token="<s>",
                                                eos_token="</s>", pad_token="<pad>",
                                                additional_special_tokens=["<image>", "[INST]", "[/INST]"],
                                                padding_side="left")
        image_processor = LlavaNextImageProcessor(size={"shortest_edge": 28},
                                                  crop_size={"height": 28, "width": 28},
                                                  image_grid_pinpoints=[[28, 28]])
        cls.processor = LlavaNextProcessor(image_processor=image_processor, tokenizer=fast_tokenizer,
                                           patch_size=14, num_additional_image_tokens=1,
                                           vision_feature_select_strategy="default")
        config = LlavaNextConfig(
            vision_config={"model_type": "clip_vision_model", "hidden_size": 16, "intermediate_size": 32,
                           "num_hidden_layers": 2, "num_attention_heads": 2, "image_size": 28, "patch_size": 14},
            text_config={"model_type": "mistral", "vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
                         "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2,
                         "max_position_embeddings": 256, "pad_token_id": 3, "sliding_window": None},
            image_token_index=4, image_seq_length=4, image_grid_pinpoints=[[28, 28]], pad_token_id=3,
            vision_feature_layer=-2)
        torch.manual_seed(12)
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        cls.model = LlavaNextForConditionalGeneration(config).eval()
        cls.current = cls.root / "native"
        cls.model.config.save_pretrained(cls.current)
        save_file(cls.model.state_dict(), str(cls.current / "model.safetensors"), metadata={"format": "pt"})
        cls.processor.save_pretrained(cls.current)
        cls.encoder = mmret.MMRetEncoder.from_pretrained(str(cls.current), "cpu", torch.float32)

    def test_text_batch_matches_individual_and_native_last_layer(self):
        batch = [mmret.Candidate("a", text="short"), mmret.Candidate("b", text="long text long text")]
        together = self.encoder.encode_batch(batch)
        individual = np.concatenate([self.encoder.encode_batch([item]) for item in batch])
        np.testing.assert_allclose(together, individual, atol=2e-6)
        self.assertEqual(together.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(together, axis=1), [1, 1], atol=1e-6)
        inputs = self.encoder.processor(text=[mmret.candidate_prompt(item.text, False) for item in batch],
                                        padding=True, return_tensors="pt")
        inputs["position_ids"] = (inputs["attention_mask"].cumsum(-1) - 1).clamp_min(0)
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True, use_cache=False, logits_to_keep=1)
        expected = torch.nn.functional.normalize(mmret.last_token_pool(outputs.hidden_states[-1], inputs["attention_mask"]), dim=-1)
        np.testing.assert_allclose(together, expected.numpy(), atol=2e-6)

    def test_native_image_and_joint_batches_match_individual(self):
        image = Image.new("RGB", (19, 31), "red")
        for texts in ((None, None), ("short", "long text long text")):
            with self.subTest(texts=texts):
                records = [mmret.Candidate(index, text=text, image=image) for index, text in enumerate(texts)]
                together = self.encoder.encode_batch(records)
                individual = np.concatenate([self.encoder.encode_batch([item]) for item in records])
                self.assertEqual(together.shape, (2, 16))
                np.testing.assert_allclose(together, individual, atol=3e-6)
        self.assertEqual(image.size, (19, 31))

    def test_old_checkpoint_keys_are_converted_without_remote_code(self):
        from safetensors.torch import save_file
        legacy = {}
        for key, value in self.model.state_dict().items():
            if key.startswith("model.language_model."):
                key = key.replace("model.language_model.", "language_model.model.", 1)
            elif key.startswith("model."):
                key = key.removeprefix("model.")
            elif key == "lm_head.weight":
                key = "language_model.lm_head.weight"
            legacy[key] = value
        directory = self.root / "legacy"
        # 4.57 save_pretrained inversely renames keys even for an explicitly
        # supplied state_dict. Write the fixture directly to avoid doing that
        # conversion a second time on our already-legacy key names.
        self.model.config.save_pretrained(directory)
        save_file(legacy, str(directory / "model.safetensors"), metadata={"format": "pt"})
        self.processor.save_pretrained(directory)
        encoder = mmret.MMRetEncoder.from_pretrained(str(directory), "cpu", torch.float32)
        records = [mmret.Candidate("x", text="short text", image=Image.new("RGB", (20, 20)))]
        np.testing.assert_allclose(encoder.encode_batch(records), self.encoder.encode_batch(records), atol=1e-6)

    def test_missing_core_weights_are_not_silently_randomized(self):
        from transformers import LlavaNextForConditionalGeneration
        with patch.object(LlavaNextForConditionalGeneration, "from_pretrained",
                          return_value=(self.model, {"missing_keys": ["model.image_newline"]})):
            with self.assertRaisesRegex(ValueError, "did not load completely"):
                mmret.MMRetEncoder.from_pretrained(str(self.current), "cpu", torch.float32)

    def test_bad_inputs_and_context_overflow(self):
        with self.assertRaisesRegex(ValueError, "no input"):
            self.encoder.encode_batch([mmret.Candidate("missing")])
        with self.assertRaisesRegex(ValueError, "consistent image"):
            self.encoder.encode_batch([mmret.Candidate("a", text="short"),
                                       mmret.Candidate("b", image=Image.new("RGB", (10, 10)))])
        with self.assertRaisesRegex(ValueError, "exceeds context"):
            self.encoder.encode_batch([mmret.Candidate("long-id", text="long " * 300)])
        self.assertEqual(self.encoder.encode_batch([]).shape, (0, 16))

    def test_complete_single_process_run_saves_pickle(self):
        model, data, train = arguments()
        model.model_name_or_path = str(self.current)
        train.encode_type = "text"
        data.dense_output_dir = str(self.root / "output")
        records = [mmret.Candidate("caption-1", text="short"), mmret.Candidate("caption-2", text="long text")]
        with patch.object(mmret, "build_candidates", return_value=records), \
             patch.object(mmret, "setup_context", return_value=mmret.EncodingContext(0, 1, torch.device("cpu"))), \
             patch.object(mmret.dist, "is_initialized", return_value=False), \
             patch.object(mmret.dist, "barrier") as barrier:
            mmret.run_mmret_encoding(model, data, train, mmret.DenseEncoderArguments(dense_model_type="mmret"))
        target = mmret.output_directory(model, data, train) / "corpus_0.pkl"
        with target.open("rb") as handle:
            embeddings, ids = pickle.load(handle)
        self.assertEqual(ids, ["caption-1", "caption-2"])
        self.assertEqual(embeddings.shape, (2, 16))
        barrier.assert_not_called()

    def test_real_entrypoint_with_tiny_local_weights_and_dataset(self):
        import encode_dense
        data_root = self.root / "entry-data"
        directory = data_root / "data/flickr"
        directory.mkdir(parents=True)
        with (directory / "flickr_test.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["imgid"])
            writer.writerow(["image", "unused.png", "short", "native-caption-1"])
            writer.writerow(["image", "unused.png", "long text", "native-caption-2"])
        output = self.root / "entry-output"
        argv = ["encode_dense.py", "--model_name_or_path", str(self.current), "--dense_model_type", "mmret",
                "--data_root", str(data_root), "--dataset_name", "flickr", "--dataset_split", "test",
                "--task_type", "ir", "--encode_type", "text", "--dense_output_dir", str(output),
                "--per_device_batch_size", "2", "--report_to", "none", "--output_dir", str(self.root / "trainer")]
        # A plain Python launch has no LOCAL_RANK. Setting it to 0 would tell
        # Accelerate to initialize distributed communication even on one CPU.
        worker_keys = {"WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}
        plain_env = {key: value for key, value in os.environ.items() if key not in worker_keys}
        with patch.object(sys, "argv", argv), patch.dict(os.environ, plain_env, clear=True):
            encode_dense.main()
        targets = list(output.rglob("corpus_0.pkl"))
        self.assertEqual(len(targets), 1)
        with targets[0].open("rb") as handle:
            matrix, ids = pickle.load(handle)
        self.assertEqual(ids, ["native-caption-1", "native-caption-2"])
        self.assertEqual(matrix.shape, (2, 16))


if __name__ == "__main__":
    unittest.main()
