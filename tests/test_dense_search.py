"""Offline search tests using real dataset loaders, FAISS and tiny LLaVA weights."""

from io import BytesIO
import importlib.util
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tevatron/src"))

import numpy as np
from PIL import Image
import torch

import dense_search as search
from dense_encoding import Candidate, DenseEncoderArguments, EncodingContext, save_shard
import mmret_dense as mmret
import unime_dense as unime
import test_mmret_dense as encoding_tests


def arguments(name="flickr", query_type="text"):
    from arguments import ModelArguments, PromptRepsLLMDataArguments, PromptRepsLLMSearchArguments
    return (ModelArguments(model_name_or_path="BAAI/BGE-VL-MLLM-S1"),
            PromptRepsLLMDataArguments(dataset_name=name, dataset_split="test", per_device_batch_size=2),
            PromptRepsLLMSearchArguments(query_type=query_type, depth=5),
            SimpleNamespace(task_type=search.DATASET_PROFILES[name][0], fp16=False, bf16=False))


class DenseQueryTests(unittest.TestCase):
    def test_mmret_query_templates_and_candidate_compatibility(self):
        prefix = "[INST] \n <instruct> task <query> "
        self.assertEqual(mmret.query_prompt("words", False, "task"), prefix + " words \n [/INST]")
        self.assertEqual(mmret.query_prompt(None, True, "task"), prefix + " <image> [/INST]")
        self.assertEqual(mmret.query_prompt("words", True, "task"), prefix + " words \n <image> [/INST]")
        self.assertEqual(mmret.candidate_prompt("words", True), "[INST]  words \n <image> [/INST]")
        for instruction in (None, "", "   "):
            with self.assertRaises(ValueError):
                mmret.query_prompt("words", False, instruction)
        with self.assertRaises(TypeError):
            mmret.query_prompt(1, False, "task")

    def test_validation_directions(self):
        for name, (task, _, _) in search.DATASET_PROFILES.items():
            types = ("text", "image") if task == "ir" else (("image", "composed") if task == "it2t" else ("text",))
            for query_type in types:
                search.validate_query_task(name, task, query_type)
            with self.assertRaisesRegex(ValueError, "query_type"):
                search.validate_query_task(name, task, "invalid")
            with self.assertRaisesRegex(ValueError, "task_type"):
                search.validate_query_task(name, "invalid", types[0])
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            search.validate_query_task("other", "ir", "text")

    def test_entry_parser_dispatches_both_models(self):
        import search_dense
        for model_type in ("mmret", "unime"):
            argv = ["search_dense.py", "--model_name_or_path", "/custom/checkpoint",
                    "--dense_model_type", model_type, "--data_root", "/data root",
                    "--passage_reps", "/corpus", "--output_dir", "unused", "--report_to", "none"]
            if model_type == "mmret":
                argv.extend(["--mmret_query_instruction", "custom task"])
            with patch.object(sys, "argv", argv), patch.object(search_dense, "run_dense_search") as run:
                search_dense.main()
            self.assertEqual(run.call_args.args[4].data_root, "/data root")
            self.assertEqual(run.call_args.args[-1], model_type)
            self.assertEqual(run.call_args.args[2].passage_reps, "/corpus")
        # Importing the entrypoint must not import optional legacy backends.
        import ast
        tree = ast.parse((PROJECT_ROOT / "src/search_dense.py").read_text(encoding="utf-8"))
        legacy = {"dataset", "models.blip_itm", "eva_clip", "sentence_transformers", "peft", "encode_dense"}
        self.assertFalse(any(isinstance(node, ast.ImportFrom) and node.module in legacy for node in tree.body))

    def test_legacy_clip_dispatch_is_preserved(self):
        import search_dense
        # Load modules with process-wide registration before patch.dict restores
        # sys.modules; otherwise pyarrow extension types get registered twice.
        import dataset
        stubs = {"models.blip_itm": SimpleNamespace(blip_itm=None),
                 "eva_clip": SimpleNamespace(create_model_and_transforms=None, get_tokenizer=None),
                 "sentence_transformers": SimpleNamespace(SentenceTransformer=None),
                 "peft": SimpleNamespace(PeftModel=None)}
        argv = ["search_dense.py", "--model_name_or_path", "example/clip-model",
                "--report_to", "none", "--output_dir", "unused"]
        with patch.object(sys, "argv", argv), patch.dict(sys.modules, stubs), \
             patch.object(search_dense.CLIPModel, "from_pretrained", side_effect=RuntimeError("legacy reached")) as load:
            with self.assertRaisesRegex(RuntimeError, "legacy reached"):
                search_dense.main()
        self.assertEqual(load.call_args.args, ("example/clip-model",))


class DenseSearchDatasetTests(unittest.TestCase):
    make_source = encoding_tests.MMRetDatasetTests.make_source
    parquet_frame = encoding_tests.MMRetDatasetTests.parquet_frame

    @classmethod
    def setUpClass(cls):
        import dataset
        import pandas as pd
        cls.dataset, cls.pd = dataset, pd

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        raw = BytesIO()
        Image.new("RGB", (18, 35)).save(raw, format="PNG")
        self.image_bytes = raw.getvalue()

    def build(self, name, query_type="text", split="test", **options):
        _, data, query, train = arguments(name, query_type)
        data.dataset_split = split
        for key, value in options.items():
            setattr(data, key, value)
        old_root = self.dataset.dataset_path_prefix
        with patch.object(self.pd, "read_parquet", side_effect=self.parquet_frame):
            queries = search.build_queries(data, train, query, str(self.root), self.dataset)
        self.assertEqual(self.dataset.dataset_path_prefix, old_root)
        return queries

    def test_ir_both_directions_and_split(self):
        for name in ("coco", "flickr"):
            self.make_source(name)
            for query_type in ("text", "image"):
                queries = self.build(name, query_type)
                self.assertEqual(len(queries), 2 if query_type == "text" else 1)
                item = queries[0]
                if query_type == "text":
                    self.assertEqual(item.identifier, "caption-1")
                    self.assertIsNone(item.image)
                else:
                    self.assertEqual(item.identifier, "image-A")
                    self.assertIsNone(item.text)
                    self.assertTrue(Path(item.image).is_file())
            directory = self.root / "data" / name
            (directory / f"{name}_val.csv").write_bytes((directory / f"{name}_test.csv").read_bytes())
            self.assertEqual(self.build(name, split="val").source.split, "val")

    def test_pedestrian_text_queries_and_identity_relevance(self):
        for name in ("CUHK-PEDES", "ICFG-PEDES", "RSTPReid"):
            self.make_source(name)
            queries = self.build(name)
            self.assertEqual(len(queries), 2)
            self.assertIsNone(queries[0].image)
            counts = search.recall_counts(queries.source, name, "text", {queries.ids[0]: {"0": 1.0}})
            np.testing.assert_array_equal(counts, np.ones(8))

    def test_multimodal_queries_and_relative_image_paths(self):
        self.make_source("webqa")
        queries = self.build("webqa")
        self.assertEqual(queries[0], Candidate("q", text="query"))
        for name in ("remuq", "llava"):
            self.make_source(name)
            for query_type in ("image", "composed"):
                queries = self.build(name, query_type)
                self.assertEqual(queries[0].text, "query")
                self.assertEqual(queries[0].image["bytes"], self.image_bytes)
                queries.source.id2query["q"]["image"] = {"bytes": None, "path": "picture.png"}
                self.assertEqual(queries[0].image["path"], str(queries.data_dir / "picture.png"))
                queries.source.id2query["q"]["image"] = "picture.png"
                with unime.load_image(queries[0].image) as image:
                    self.assertEqual(image.size, (18, 35))

    def test_generated_validation_selects_csv_and_original_images(self):
        self.make_source("flickr")
        generated = self.root / "generated"
        directory = generated / "data/flickr"
        directory.mkdir(parents=True)
        (directory / "flickr_val.csv").write_bytes((self.root / "data/flickr/flickr_test.csv").read_bytes())
        queries = self.build("flickr", "image", split="val", use_generated_validation_data=True,
                             generated_validation_data_root=str(generated))
        self.assertEqual(queries.data_dir, directory)
        self.assertTrue(Path(queries[0].image).is_file())


class DenseIndexTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_shards_sorting_depth_self_removal_and_gpu_fallback(self):
        save_shard(self.root, 0, np.array([[1, 0], [0, 1]]), ["a", "b"])
        save_shard(self.root, 1, np.empty((0, 2)), [])
        save_shard(self.root, 2, np.array([[-1, 0]]), ["c"])
        index, ids, _ = search.load_dense_index(self.root, 2, use_gpu=True, device="cpu")
        run = search.search_batch(index, [[1, 0], [0, 1]], ids, ["a", "q"], 10, retrieval_batch_size=1)
        self.assertEqual(list(run["a"]), ["a", "b", "c"])
        self.assertEqual(list(run["q"])[0], "b")
        run = search.search_batch(index, [[1, 0]], ids, ["a"], 2, remove_query=True)
        self.assertEqual(list(run["a"]), ["b", "c"])
        single, ids, _ = search.load_dense_index(self.root / "corpus_0.pkl", 2)
        self.assertEqual(single.ntotal, 2)
        self.assertEqual(ids, ["a", "b"])

    def test_index_validation(self):
        with self.assertRaisesRegex(ValueError, "No corpus"):
            search.load_dense_index(self.root, 2)
        path = save_shard(self.root, 0, np.empty((0, 2)), [])
        with self.assertRaisesRegex(ValueError, "empty"):
            search.load_dense_index(path, 2)
        for reps, ids, message in (([[1, 0]], [], "misaligned"), ([[1, 0, 0]], [1], "dimension"),
                                   ([[float("nan"), 0]], [1], "nonfinite"),
                                   ([[1, 0], [0, 1]], [1, "1"], "Duplicate")):
            with path.open("wb") as handle:
                pickle.dump((reps, ids), handle)
            with self.assertRaisesRegex(ValueError, message):
                search.load_dense_index(path, 2)
        save_shard(self.root, 0, np.array([[1, 0]]), ["a"])
        save_shard(self.root, 1, np.array([[0, 1]]), ["a"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            search.load_dense_index(self.root, 2)

    def test_recall_aggregation_uses_actual_counts_and_handles_empty_workers(self):
        source = SimpleNamespace(get_target=lambda qid, direction: {"q1": ["a", "b"], "q2": "c"}[qid])
        first = search.recall_counts(source, "flickr", "image", {"q1": {"z": 2.0, "b": 1.0}})
        second = search.recall_counts(source, "flickr", "image", {"q2": {"c": 1.0}})
        np.testing.assert_array_equal(first + second, [1, 2, 2, 2, 2, 2, 2, 2])
        context = EncodingContext(0, 1, torch.device("cpu"))
        with patch.object(search.dist, "is_initialized", return_value=False), \
             patch.object(search.dist, "all_reduce") as reduce:
            np.testing.assert_array_equal(search.reduce_recall_counts(first, context), first)
            reduce.assert_not_called()
        total = first + second
        with patch.object(search.dist, "is_initialized", return_value=True), \
             patch.object(search.dist, "all_reduce", side_effect=lambda tensor, op: tensor.copy_(torch.tensor(total))):
            np.testing.assert_array_equal(search.reduce_recall_counts(np.zeros(8), context), total)
        self.assertEqual(search.recall_counts(source, "flickr", "image", {}).sum(), 0)

    def test_result_paths_and_excel(self):
        model, data, args, _ = arguments()
        self.assertEqual(search.result_directory(model, data, args),
                         Path("search_results/BGE-VL-MLLM-S1/flickr/text/test"))
        args.save_dir = str(self.root)
        self.assertEqual(search.result_directory(model, data, args), self.root)
        if not any(importlib.util.find_spec(name) for name in ("xlsxwriter", "openpyxl")):
            self.skipTest("An Excel writer is needed for the real spreadsheet output test")
        target = search.write_recall(np.array([1, 2, 2, 2, 2, 2, 2, 2]), self.root)
        with zipfile.ZipFile(target) as workbook:
            sheet = workbook.read("xl/worksheets/sheet1.xml").decode()
            self.assertIn("0.5", sheet)

    def test_recall_spreadsheet_layout_without_optional_engine(self):
        import pandas as pd
        with patch.object(pd.DataFrame, "to_excel", autospec=True) as write:
            target = search.write_recall(np.array([1, 2, 2, 2, 2, 2, 2, 2]), self.root)
        frame, path = write.call_args.args
        self.assertEqual(path, target)
        self.assertEqual(list(frame.columns), list(search.RECALL_K))
        np.testing.assert_allclose(frame.iloc[0].to_numpy(), [0.5, 1, 1, 1, 1, 1, 1])
        self.assertFalse(write.call_args.kwargs["index"])


class DenseNativeSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the miniature checkpoint fixture, not its candidate-encoding tests.
        fixture = encoding_tests.MMRetNativeModelTests
        fixture.setUpClass()
        cls.addClassCleanup(fixture.doClassCleanups)
        cls.fixture = fixture

    def test_native_query_modalities_match_individual_batches(self):
        for encoder_type in (search.MMRetQueryEncoder, unime.UniMEEncoder):
            encoder = encoder_type.from_pretrained(str(self.fixture.current), "cpu", torch.float32)
            if isinstance(encoder, search.MMRetQueryEncoder):
                encoder.instruction = "Retrieve text."
            for texts, image in ((("short", "long text"), None),
                                 ((None, None), Image.new("RGB", (19, 31))),
                                 (("short", "long text"), Image.new("RGB", (19, 31)))):
                records = [Candidate(i, text=text, image=image) for i, text in enumerate(texts)]
                together = encoder.encode_batch(records)
                individual = np.concatenate([encoder.encode_batch([record]) for record in records])
                np.testing.assert_allclose(together, individual, atol=4e-6)
                np.testing.assert_allclose(np.linalg.norm(together, axis=1), [1, 1], atol=1e-6)

    def test_single_process_full_search_and_empty_worker(self):
        model, data, args, train = arguments()
        model.model_name_or_path = str(self.fixture.current)
        data.dataset_name = "webqa"
        train.task_type = "t2it"
        records = [Candidate("q", text="short")]
        queries = SimpleNamespace(source=SimpleNamespace(get_target=lambda qid: "a"), name="webqa")

        class Queries:
            source, name = queries.source, queries.name

            def __len__(self):
                return len(records)

            def __getitem__(self, index):
                return records[index]

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            args.passage_reps = str(save_shard(directory, 0, np.eye(1, 16, dtype=np.float32), ["a"]))
            for model_type in ("mmret", "unime"):
                context = EncodingContext(0, 1, torch.device("cpu"))
                with patch.object(search, "write_recall") as write, \
                     patch.object(search.dist, "is_initialized", return_value=False):
                    counts = search.run_dense_search(model, data, args, train, DenseEncoderArguments(),
                        search.DenseSearchArguments(), model_type, context_factory=lambda: context,
                        query_builder=lambda *unused: Queries())
                np.testing.assert_array_equal(counts, np.ones(8))
                write.assert_called_once()
            context = EncodingContext(1, 2, torch.device("cpu"))
            with patch.object(search, "write_recall") as write, \
                 patch.object(search.dist, "is_initialized", return_value=False):
                counts = search.run_dense_search(model, data, args, train, DenseEncoderArguments(),
                    search.DenseSearchArguments(), "unime", context_factory=lambda: context,
                    query_builder=lambda *unused: Queries())
            np.testing.assert_array_equal(counts, np.zeros(8))
            write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
