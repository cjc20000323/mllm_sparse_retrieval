"""Native Qwen three-stage integration; no checkpoints, CUDA or JVM needed."""
from collections import Counter
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from dataclasses import replace
import copy
import io
import json
import pickle
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import test_qwen25_retrieval as q25
import test_qwen2_retrieval as q2
import test_qwen3_retrieval as q3
import test_three_stage_retrieval_pipeline as fixtures
import test_three_stage_dataset_adapters as adapters
import test_three_stage_time_estimation as timing_fixtures
import estimate_three_stage_retrieval_time as estimate

pipeline = fixtures.pipeline
shared = q25.pipeline


class Flickr:
    data_name = "flickr"
    img_id_list = ["i0", "i1"]
    text_id_list = ["t0", "t1", "t2"]
    text_dict = {"t0": "short caption", "t1": "a longer caption about people", "t2": "caption blue"}
    img_dict = {"i0": "0.png", "i1": "1.png"}
    img2text = {"i0": ["t0", "t2"], "i1": ["t1"]}

    def __init__(self, root):
        self.root = Path(root)
        directory = self.root / "data/flickr/flickr30k-images"
        directory.mkdir(parents=True, exist_ok=True)
        for name, size, color in (("0.png", (8, 8), "red"), ("1.png", (12, 8), "blue")):
            Image.new("RGB", size, color).save(directory / name)

    def __getitem__(self, index):
        identifier = self.img_id_list[index]
        return "", self.root / "data/flickr/flickr30k-images" / self.img_dict[identifier], "", identifier

    def get_target(self, identifier, direction):
        return self.img2text[identifier] if direction == "image" else {"t0": "i0", "t1": "i1", "t2": "i0"}[identifier]


def arguments(root, **values):
    args = pipeline._parser().parse_args([
        "--dataset-name", "flickr", "--query-type", "image", "--data-root", str(root),
        "--retrieval-model-name-or-path", "renamed-qwen", "--ranker-model-name-or-path", "renamed-qwen",
        "--retrieval-device", "cpu", "--retrieval-device-map", "none", "--ranker-device-map", "none",
        "--retrieval-dtype", "float32", "--ranker-dtype", "float32", "--retrieval-batch-size", "2",
        "--sparse-length", "8", "--text-sparse-length", "8", "--image-sparse-length", "8",
        "--sparse-value-type", "sum", "--stage1-depth", "2", "--stage2-depth", "2",
        "--rerank-num", "2", "--rerank-batch-size", "2", "--output-dir", str(Path(root) / "result"),
    ])
    for key, value in values.items():
        setattr(args, key, value)
    return args


def sparse_search(index_path, topics, qids, depth, *unused, **kwargs):
    index_path = Path(index_path)
    root = shared.artifact_root(index_path)
    documents = [json.loads(line) for path in (root / "collection").glob("*.jsonl")
                 for line in path.read_text(encoding="utf-8").splitlines()]
    return {qid: dict(sorted(((doc["id"], float(sum(value * Counter(topics[qid].split())[term]
                                                  for term, value in doc["vector"].items())))
                             for doc in documents), key=lambda item: item[1], reverse=True)[:depth]) for qid in qids}


class QwenThreeStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.models = {}
        for family, module in (("qwen2_vl", q2), ("qwen2_5_vl", q25), ("qwen3_vl", q3)):
            processor = module.processor_fixture()
            cls.models[family] = (module.model_fixture(processor), processor)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(redirect_stderr(io.StringIO()))

    def runtime(self, family, args):
        model, processor = self.models[family]
        return SimpleNamespace(encoder=model, model=model, processor=processor, qwen_family=family,
                               qwen_load_settings=pipeline._qwen_load_settings(args), query_cache={},
                               device=torch.device("cpu"), dtype=torch.float32)

    def corpus(self, root, family, args, dataset, *, manual=False):
        model, processor = self.models[family]
        data = q25.Data(dataset_name=args.dataset_name, dataset_split=args.dataset_split,
                        sparse_manual=manual, per_device_batch_size=2,
                        dense_output_dir=str(Path(root) / "dense"), sparse_output_dir=str(Path(root) / "sparse"))
        model_args, _ = pipeline._retrieval_namespaces(
            args, eol_type="prompteol" if manual else pipeline.STAGE2_EOL_TYPE,
            calculate_type="separate" if manual else "concat", sparse_manual=manual,
        )
        training = SimpleNamespace(task_type=args.task_type, encode_type=shared.corpus_kind(args.task_type, args.query_type))
        with patch.object(shared, "load_runtime", return_value=(model, processor, 0, 1)), \
                patch.object(shared, "build_dataset", return_value=dataset):
            shared.run_encode(model_args, data, training)
        dense, sparse = shared.artifact_dirs(model_args, data, training)
        (sparse / "collection/index").mkdir()
        (sparse / "collection/index/segments_1").touch()
        self.assertEqual(pipeline._resolve_lucene_index(sparse), sparse / "collection/index")
        return str(dense), str(sparse)

    def test_three_native_families_real_artifacts_reuse_and_rerank(self):
        for family in self.models:
            for direction in ("image", "text"):
                with self.subTest(family=family, direction=direction), tempfile.TemporaryDirectory() as root:
                    args = arguments(root, query_type=direction)
                    dataset = Flickr(root)
                    args.stage2_dense_index, args.stage2_sparse_index = self.corpus(root, family, args, dataset)
                    args.stage1_sparse_index = args.stage2_sparse_index
                    runtime = self.runtime(family, args)
                    model, processor = self.models[family]
                    with patch.object(pipeline, "build_dataset", return_value=dataset), \
                            patch.object(pipeline, "_load_retrieval_runtime", return_value=runtime), \
                            patch.object(pipeline, "retrieve_sparse_topics_from_lucene", side_effect=sparse_search), \
                            patch.object(model, "forward", wraps=model.forward) as forward:
                        context = pipeline.DistributedContext()
                        stage1, encoded, global_run, counts = pipeline._run_sparse_concat_stage1(args, context, pipeline.StageTimer())
                        calls = forward.call_count
                        self.assertEqual(calls, (len(encoded.query_ids) + 1) // 2)
                        self.assertIsNone(runtime.encoder)
                        self.assertFalse(runtime.query_cache)
                        stage2_timer = pipeline.StageTimer()
                        dense, sparse, hybrid = pipeline.run_stage2(args, stage1, None, stage2_timer, encoded_queries=encoded)
                        self.assertEqual(forward.call_count, calls)
                        self.assertEqual(stage2_timer.components["query_encoding"], 0.0)
                        self.assertEqual(set(hybrid), set(stage1))
                        self.assertEqual(counts, [len(encoded.query_ids)])
                        for qid in hybrid:
                            self.assertLessEqual(set(hybrid[qid]), set(stage1[qid]))
                        selected = pipeline._select_encoded_queries(encoded, list(reversed(encoded.query_ids)))
                        self.assertEqual(selected.qwen_metadata, encoded.qwen_metadata)
                        self.assertIsNot(selected.qwen_metadata, encoded.qwen_metadata)
                        for mode in ("relevant", "what_caption_generation"):
                            args.rerank_template = mode
                            with patch.object(pipeline, "_load_ranker_model", return_value=(model, processor)):
                                reranked = pipeline.run_stage3(args, hybrid)
                            self.assertEqual(set(reranked), set(hybrid))
                            self.assertTrue(all(np.isfinite(score) for scores in reranked.values() for score in scores.values()))
                            self.assertTrue(all(set(reranked[qid]) <= set(hybrid[qid]) for qid in hybrid))
                        for settings in (("random_view_seed", 43), ("is_filtered", True), ("retrieval_qwen_max_pixels", 100)):
                            with patch.object(args, *settings), self.assertRaisesRegex(ValueError, "reuse configuration mismatch"):
                                pipeline.run_stage2(args, stage1, None, encoded_queries=encoded)
                        manifest = Path(args.stage2_dense_index) / shared.manifest_name(encoded.qwen_metadata)
                        state = json.loads(manifest.read_text())
                        state["fingerprint"] = "wrong-family-or-settings"
                        manifest.write_text(json.dumps(state))
                        with self.assertRaisesRegex(ValueError, "mismatch"):
                            pipeline.run_stage2(args, stage1, None, encoded_queries=encoded)

    def test_manual_stage1_and_concat_stage2_encode_separately(self):
        for family in self.models:
            with self.subTest(family=family), tempfile.TemporaryDirectory() as root:
                args, dataset = arguments(root, stage1_method="sparse_manual"), Flickr(root)
                args.stage2_dense_index, args.stage2_sparse_index = self.corpus(root, family, args, dataset)
                _, args.stage1_sparse_index = self.corpus(root, family, args, dataset, manual=True)
                runtime = self.runtime(family, args)
                with patch.object(pipeline, "build_dataset", return_value=dataset), \
                        patch.object(pipeline, "retrieve_sparse_topics_from_lucene", side_effect=sparse_search), \
                        patch.object(runtime.encoder, "forward", wraps=runtime.encoder.forward) as forward:
                    stage1 = pipeline.run_stage1(args, runtime)
                    calls = forward.call_count
                    pipeline.run_stage2(args, stage1, runtime)
                    self.assertEqual(forward.call_count, calls * 2)

    def test_all_native_task_adapters_and_prompts_match_standalone(self):
        for family in self.models:
            for name, (task, direction, _, _) in adapters.PROFILES.items():
                with self.subTest(family=family, dataset=name), tempfile.TemporaryDirectory() as root:
                    adapters.write_dataset(Path(root), name, image_size=(8, 8))
                    args = arguments(root, dataset_name=name, task_type=task, query_type=direction)
                    runtime = self.runtime(family, args)
                    dataset = pipeline.build_dataset(args, runtime.processor)
                    records = pipeline._qwen_query_records(args, pipeline._query_records(args, dataset))
                    for eol, calc, manual in (("prompteol", "separate", True), (pipeline.STAGE2_EOL_TYPE, "concat", False)):
                        with patch.object(pipeline, "build_dataset", return_value=dataset):
                            actual = pipeline.encode_queries_online(args, runtime, eol_type=eol, calculate_type=calc,
                                                                    sparse_manual=manual, progress_prefix="test")
                        model_args, data = pipeline._retrieval_namespaces(args, eol_type=eol, calculate_type=calc, sparse_manual=manual)
                        encoder = q25.NativeQwenEncoder(runtime.encoder, runtime.processor, model_args, data, task)
                        dense = np.concatenate([vectors for _, vectors, _ in shared.encode_batches(encoder, records, data)])
                        np.testing.assert_allclose(actual.dense_representations, dense, atol=3e-5, rtol=3e-5)
                    for mode in ("relevant", "what_caption_generation"):
                        args.rerank_template = mode
                        with patch.object(pipeline, "_load_ranker_model", return_value=self.models[family]):
                            result = pipeline.run_stage3(args, adapters.relevant_run(name))
                        self.assertEqual(set(result), set(adapters.relevant_run(name)))

    def test_ir_caption_leak_and_cir_composition_are_handled(self):
        image = Image.new("RGB", (8, 8), "red")
        for family in self.models:
            with self.subTest(family=family), tempfile.TemporaryDirectory() as root:
                args = arguments(root)
                runtime = self.runtime(family, args)
                records = [dict(id="q", text="THIS REFERENCE CAPTION MUST NOT ENTER THE PROMPT", image=image)]
                with patch.object(pipeline, "build_dataset", return_value=object()), patch.object(pipeline, "_query_records", return_value=records):
                    encoded = pipeline.encode_queries_online(args, runtime, eol_type=pipeline.STAGE2_EOL_TYPE,
                                                             calculate_type="concat", sparse_manual=False, progress_prefix="test")
                self.assertNotIn(records[0]["text"], encoded.qwen_metadata["query_examples"][0])
                args.task_type, args.dataset_name, args.query_type = "cir", "fashion-iq", "composed"
                records[0].update(text="make it blue", dress_type="shirt")
                with patch.object(pipeline, "build_dataset", return_value=object()), patch.object(pipeline, "_query_records", return_value=records):
                    encoded = pipeline.encode_queries_online(args, runtime, eol_type=pipeline.STAGE2_EOL_TYPE,
                                                             calculate_type="concat", sparse_manual=False, progress_prefix="test")
                self.assertIn("change the style of this shirt to make it blue", encoded.qwen_metadata["query_examples"][0])
                directory = Path(root) / "data/fashion-iq/images/images"
                directory.mkdir(parents=True)
                image.save(directory / "ref.png")
                image.save(directory / "candidate.png")
                dataset = SimpleNamespace(img_dict={"ref": "ref.png", "candidate": "candidate.png"},
                                          text_dict={"cap": "make it blue"}, get_text=lambda key: "make it blue",
                                          get_dress_type=lambda key: "shirt")
                with patch.object(pipeline, "build_dataset", return_value=dataset), \
                        patch.object(pipeline, "_load_ranker_model", return_value=self.models[family]):
                    result = pipeline.run_stage3(args, {"ref_cap": {"candidate": 1.0}})
                self.assertTrue(np.isfinite(result["ref_cap"]["candidate"]))

    def test_native_model_loading_config_route_and_device_ownership(self):
        for family in self.models:
            with self.subTest(family=family), tempfile.TemporaryDirectory() as root:
                model, processor = self.models[family]
                model.save_pretrained(root)
                processor.save_pretrained(root)
                args = arguments(root, retrieval_model_name_or_path=root, ranker_model_name_or_path=root)
                args.retrieval_qwen_min_pixels, args.retrieval_qwen_max_pixels = 64, 144
                args.ranker_qwen_min_pixels, args.ranker_qwen_max_pixels = 64, 256
                with patch.object(shared, "init_process", side_effect=AssertionError("caller owns the process group")):
                    runtime = pipeline._load_retrieval_runtime(args)
                    self.assertEqual(runtime.qwen_family, family)
                    self.assertIsInstance(runtime.encoder, type(model))
                    pipeline._release_retrieval_runtime(runtime)
                    ranker, ranker_processor = pipeline._load_ranker_model(args)
                    self.assertIsInstance(ranker, type(model))
                    self.assertEqual(ranker_processor.image_processor.max_pixels, 256)
                    del ranker
                args.rerank_template = "caption_generation_6"
                with patch.object(type(model), "from_pretrained") as load:
                    with self.assertRaisesRegex(ValueError, "BOS"):
                        pipeline._load_ranker_model(args)
                    load.assert_not_called()

    def test_qwen_parser_defaults_and_pixel_validation_in_both_entrypoints(self):
        with tempfile.TemporaryDirectory() as root:
            args = arguments(root)
            for field in ("retrieval_qwen_min_pixels", "retrieval_qwen_max_pixels", "ranker_qwen_min_pixels", "ranker_qwen_max_pixels"):
                self.assertEqual(getattr(args, field), 0)
                with patch.object(args, field, -1), self.assertRaisesRegex(ValueError, "pixel limits"):
                    pipeline._validate_args(args)
            parsed = estimate.parser().parse_args(["--dataset-name", "webqa", "--retrieval-qwen-max-pixels", "1234", "--ranker-qwen-max-pixels", "5678"])
            self.assertEqual((parsed.retrieval_qwen_max_pixels, parsed.ranker_qwen_max_pixels), (1234, 5678))

    def test_native_estimation_reuses_queries_and_keeps_setup_outside_observer(self):
        for family in self.models:
            with self.subTest(family=family), tempfile.TemporaryDirectory() as root:
                args = arguments(root, retrieval_device="auto", rerank_template="what_caption_generation")
                dataset = Flickr(root)
                args.stage2_dense_index, args.stage2_sparse_index = self.corpus(root, family, args, dataset)
                args.stage1_sparse_index = args.stage2_sparse_index
                runtime = self.runtime(family, args)
                model, processor = self.models[family]
                clock = timing_fixtures.Clock()
                collective = timing_fixtures.Collectives(clock)
                def load_ranker(*unused):
                    self.assertIsNone(runtime.encoder)
                    self.assertFalse(runtime.query_cache)
                    return model, processor
                with patch.object(pipeline, "build_dataset", return_value=dataset), \
                        patch.object(pipeline, "_load_retrieval_runtime", return_value=runtime) as load_queries, \
                        patch.object(pipeline, "_load_ranker_model", side_effect=load_ranker), \
                        patch.object(pipeline, "retrieve_sparse_topics_from_lucene", side_effect=sparse_search), \
                        patch.object(pipeline, "_synchronize_cuda_devices"), \
                        patch.object(pipeline, "RecallReporter", side_effect=AssertionError("no estimation recall")):
                    report = estimate.run_estimation(args, context=pipeline.DistributedContext(), collectives=collective)
                self.assertEqual(load_queries.call_count, 1)
                self.assertEqual(report["status"], "completed")
                self.assertEqual(report["stages"]["stage2"]["rank_reports"][0]["components_seconds"]["query_encoding"], 0.0)
                self.assertEqual([p.name for p in Path(args.output_dir).iterdir()], ["time_estimate.json"])

    def test_stage3_native_prior_cache_matches_online_correction(self):
        for family in self.models:
            with self.subTest(family=family), tempfile.TemporaryDirectory() as root:
                args = arguments(root, rerank_template="what_caption_generation", modify_type="modify_single",
                                 rerank_prior_cache_dir=str(Path(root) / "prior"))
                dataset = Flickr(root)
                run = {"i0": {"t0": 1.0, "t1": 0.5}}
                with patch.object(pipeline, "build_dataset", return_value=dataset), \
                        patch.object(pipeline, "_load_ranker_model", return_value=self.models[family]):
                    expected = pipeline.run_stage3(args, run)
                    args.use_rerank_prior_cache = True
                    actual = pipeline.run_stage3(args, run)
                    self.assertEqual(expected, actual)
                    self.assertEqual(actual, pipeline.run_stage3(args, run))


if __name__ == "__main__":
    unittest.main()
