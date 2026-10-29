from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import pickle
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_three_stage_retrieval_pipeline as fixtures

pipeline = fixtures.pipeline
ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash") or next((str(p) for p in (Path("D:/Git/bin/bash.exe"), Path("C:/Program Files/Git/bin/bash.exe")) if p.is_file()), None)


def runs():
    wrong = {f"wrong-{i}": 1.0 for i in range(12)}
    stage1 = {"q": {**wrong, "correct": 0.0}}
    dense = {"q": {**{key: 0.0 for key in wrong}, "correct": 1.0}}
    sparse = {"q": {**wrong, "correct": 0.0}}
    return stage1, dense, sparse


def evaluation(*args):
    return pipeline.RecallEvaluation([1, 5, 10, 100], {"q": {"correct"}}, {}, None, 1)


class AlphaSelectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="alpha selection ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.args = fixtures.ThreeStageRecallTests.args(
            stage2_run=None, stage2_dense_index="dense", stage2_sparse_index="sparse",
            retrieval_model_name_or_path="fake-model", select_hybrid_alpha=True,
            output_dir=str(self.root / "output"),
        )

    def validation(self):
        dense, sparse = self.root / "val dense", self.root / "val sparse"
        dense.mkdir(); (dense / "corpus_0.pkl").touch()
        (sparse / "index").mkdir(parents=True); (sparse / "index/segments_1").touch()
        first = self.root / "val stage1.json"
        first.write_text(json.dumps(runs()[0]), encoding="utf-8")
        self.args.val_passage_reps = str(dense)
        self.args.val_sparse_index = str(sparse)
        self.args.val_stage1_run = str(first)

    def execute(self, *, fail_ranker=False):
        stage1, dense, sparse = runs()
        events = []

        def stage2(args, local, runtime, **kwargs):
            events.append((args.dataset_name, args.dataset_split))
            d, s = (sparse, dense) if args.dataset_split == "val" else (dense, sparse)
            return d, s, pipeline.fuse_restricted_runs(local, d, s, args.hybrid_alpha, args.stage1_depth)

        ranker_inputs = []

        def ranker(args, run, *extra, **kwargs):
            ranker_inputs.append((args.hybrid_alpha, run))
            if fail_ranker:
                raise RuntimeError("ranker failure")
            return run

        with (
            patch.object(pipeline, "_parser", return_value=SimpleNamespace(parse_args=lambda: self.args)),
            patch.object(pipeline, "_initialise_distributed", return_value=pipeline.DistributedContext()),
            patch.object(pipeline, "run_stage1", return_value=stage1) as first,
            patch.object(pipeline, "run_stage2", side_effect=stage2) as second,
            patch.object(pipeline, "run_stage3", side_effect=ranker) as third,
            patch.object(pipeline, "_build_recall_evaluation", side_effect=evaluation),
            patch.object(pipeline, "_load_retrieval_runtime", return_value=SimpleNamespace()) as loader,
            patch.object(pipeline, "_release_retrieval_runtime"),
            patch.object(pipeline.RecallReporter, "_output_dependencies", side_effect=fixtures.fake_excel_dependencies),
            patch.object(pipeline, "_progress", side_effect=fixtures.ProgressRecorder()),
            patch.object(pipeline.tqdm, "write"),
        ):
            if fail_ranker:
                with self.assertRaisesRegex(RuntimeError, "ranker failure"):
                    pipeline.main()
            else:
                pipeline.main()
        return events, ranker_inputs, (first.call_count, second.call_count, third.call_count, loader.call_count)

    def report(self):
        directory = Path(self.args.output_dir)
        payload = json.loads((directory / "alpha_selection.json").read_text(encoding="utf-8"))
        with (directory / "stage2_alpha_sweep.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue((directory / "stage2_alpha_sweep.csv").read_bytes().startswith(b"\xef\xbb\xbf"))
        return payload, rows

    def test_target_fallback_selects_best_and_reranks_once(self):
        events, inputs, counts = self.execute()
        self.assertEqual(events, [("flickr", "test")])
        self.assertEqual(counts, (1, 1, 1, 1))
        payload, rows = self.report()
        self.assertEqual(payload["selection"]["alpha"], 0.6)
        self.assertEqual(payload["selection"]["source"], "target")
        self.assertEqual(payload["selection"]["dataset_split"], "test")
        self.assertEqual(len(rows), 27)
        self.assertEqual({float(row["alpha"]) for row in rows}, set(pipeline.ALPHA_CANDIDATES))
        self.assertEqual([float(row["alpha"]) for row in rows if row["selected"] == "True" and row["variant"] == "hybrid"], [0.6])
        alpha, actual = inputs[0]
        self.assertEqual(alpha, 0.6)
        self.assertEqual(next(iter(actual["q"])), "correct")
        saved = json.loads((Path(self.args.output_dir) / "stage2_hybrid_run.json").read_text())
        self.assertEqual(actual, saved)
        metadata = json.loads((Path(self.args.output_dir) / "pipeline_metadata.json").read_text())
        self.assertEqual(metadata["stage2"]["alpha"], alpha)

    def test_qwen_validation_uses_nested_index_and_rejects_incomplete_new_index(self):
        self.validation()
        sparse = Path(self.args.val_sparse_index)
        (sparse / "qwen25_prompt_manifest.json").write_text("{}")
        # The valid legacy index remains present throughout this test.
        nested = sparse / "collection/index"
        nested.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "Missing Qwen Lucene index"):
            pipeline._validate_alpha_options(self.args)
        (nested / "segments_2").touch()
        for path in (sparse, sparse / "collection", nested):
            self.args.val_sparse_index = str(path)
            pipeline._validate_alpha_options(self.args)
        self.args.val_sparse_index = str(sparse / "index")
        pipeline._validate_alpha_options(self.args)

    def test_cross_dataset_validation_wins_even_when_target_prefers_another_alpha(self):
        self.validation()
        self.args.dataset_name = "coco"
        self.args.validation_dataset_name = "flickr"
        events, inputs, counts = self.execute()
        self.assertEqual(events, [("flickr", "val"), ("coco", "test")])
        self.assertEqual(counts, (2, 2, 1, 2))
        payload, rows = self.report()
        self.assertEqual(payload["selection"]["source"], "validation")
        self.assertEqual(payload["selection"]["dataset_name"], "flickr")
        self.assertEqual(payload["selection"]["alpha"], 0.1)
        self.assertEqual(len(rows), 54)
        target = [row for row in rows if row["evaluation_role"] == "target" and row["variant"] == "hybrid"]
        selected, = [row for row in target if row["selected"] == "True"]
        self.assertEqual(float(selected["alpha"]), 0.1)
        self.assertEqual(float(selected["R@1"]), 0.0)
        self.assertTrue(any(float(row["R@1"]) == 1.0 for row in target))
        self.assertEqual(inputs[0][0], 0.1)
        self.assertNotEqual(next(iter(inputs[0][1]["q"])), "correct")
        self.assertIn("stage1", payload["validation_timing"])
        self.assertTrue((Path(self.args.output_dir) / "alpha_validation/stage2_hybrid_run.json").is_file())

    def test_reports_survive_reranking_failure(self):
        self.execute(fail_ranker=True)
        payload, rows = self.report()
        self.assertEqual(payload["selection"]["alpha"], 0.6)
        self.assertEqual(len(rows), 27)
        timing = json.loads((Path(self.args.output_dir) / "timing_metrics.json").read_text())
        self.assertEqual(timing["status"], "failed")

    def test_disabled_mode_keeps_fixed_weight_without_sweep_files(self):
        self.args.select_hybrid_alpha = False
        _, inputs, counts = self.execute()
        self.assertEqual(counts, (1, 1, 1, 1))
        self.assertEqual(inputs[0][0], 0.5)
        self.assertFalse((Path(self.args.output_dir) / "alpha_selection.json").exists())

    def test_invalid_validation_inputs_and_precomputed_runs(self):
        self.args.val_passage_reps = "missing"
        with self.assertRaisesRegex(ValueError, "both"):
            pipeline._validate_alpha_options(self.args)
        self.args.val_sparse_index = "missing"
        with self.assertRaisesRegex(ValueError, "corpus"):
            pipeline._validate_alpha_options(self.args)
        self.args.val_passage_reps = self.args.val_sparse_index = None
        for name in ("stage2_run", "ranker_run"):
            setattr(self.args, name, "precomputed")
            with self.assertRaisesRegex(ValueError, "unsupported"):
                pipeline._validate_alpha_options(self.args)
            setattr(self.args, name, None)
        self.validation()
        self.args.validation_dataset_name = "webqa"
        with self.assertRaisesRegex(ValueError, "same task"):
            pipeline._validate_alpha_options(self.args)
        self.args.validation_dataset_name = "flickr"
        self.args.select_hybrid_alpha = False
        with self.assertRaisesRegex(ValueError, "require"):
            pipeline._validate_alpha_options(self.args)

    def test_validation_coarse_configuration_does_not_inherit_target_run_or_command(self):
        self.validation()
        self.args.stage1_method = "sparse_concat"
        self.args.val_stage1_run = None
        self.args.stage1_command = "must not run on validation"
        validation = pipeline._alpha_validation_args(self.args)
        self.assertIsNone(validation.stage1_run)
        self.assertIsNone(validation.stage1_command)
        self.assertEqual(validation.stage1_sparse_index, self.args.val_sparse_index)
        pipeline._validate_alpha_options(self.args)
        self.args.stage1_method = "sparse_manual"
        with self.assertRaisesRegex(ValueError, "val-stage1-sparse-index"):
            pipeline._validate_alpha_options(self.args)

    def test_all_zero_scores_choose_smallest_alpha_and_fashion_objective(self):
        reporter = pipeline.AlphaSweepReporter(self.root)
        stage1, dense, sparse = runs()
        empty = pipeline.RecallEvaluation([1, 5, 10], {"q": {"absent"}}, {}, None, 1)
        with patch.object(pipeline, "_build_recall_evaluation", return_value=empty), patch.object(pipeline.tqdm, "write"), patch.object(pipeline, "_progress", side_effect=fixtures.ProgressRecorder()):
            self.assertEqual(reporter.sweep(self.args, stage1, dense, sparse, role="target", timing=pipeline.StageTimer()), 0.1)
        rows = [{"category": category, "recall": {"10": 0.5, "50": 1.0}} for category in ("dress", "shirt", "toptee")]
        self.assertEqual(pipeline._alpha_objective(rows, "fashion-iq"), 0.75)
        with self.assertRaisesRegex(ValueError, "missing"):
            pipeline._alpha_objective(rows[:2], "fashion-iq")

    def test_validation_sparse_concat_reuses_local_queries(self):
        import numpy as np
        self.args.stage1_method, self.args.stage1_run = "sparse_concat", None
        stage1, dense, sparse = runs()
        encoded = pipeline.EncodedQueries(["q"], np.ones((1, 2), dtype="float32"), {"q": "topic"})
        with (
            patch.object(pipeline, "_run_sparse_concat_stage1", return_value=(stage1, encoded, stage1, [1])) as first,
            patch.object(pipeline, "run_stage2", return_value=(dense, sparse, {})) as second,
            patch.object(pipeline, "_load_retrieval_runtime", side_effect=AssertionError("reloaded")),
            patch.object(pipeline, "_release_retrieval_runtime"),
        ):
            actual = pipeline._run_alpha_validation(self.args, pipeline.DistributedContext())
        first.assert_called_once()
        self.assertIs(second.call_args.kwargs["encoded_queries"], encoded)
        self.assertEqual(actual[:3], (stage1, dense, sparse))

    def test_sparse_concat_encodes_once_for_all_nine_weights(self):
        import numpy as np
        self.args.stage1_method, self.args.stage1_run = "sparse_concat", None
        self.args.stage1_sparse_index = "sparse"
        _, dense, sparse = runs()
        encoded = pipeline.EncodedQueries(["q"], np.ones((1, 2), dtype="float32"), {"q": "topic"})
        with (
            patch.object(pipeline, "_parser", return_value=SimpleNamespace(parse_args=lambda: self.args)),
            patch.object(pipeline, "_initialise_distributed", return_value=pipeline.DistributedContext()),
            patch.object(pipeline, "_stage1_query_shard", return_value=(["q"], ["q"], [1])),
            patch.object(pipeline, "_load_retrieval_runtime", return_value=SimpleNamespace()) as loader,
            patch.object(pipeline, "_release_retrieval_runtime"),
            patch.object(pipeline, "encode_queries_online", return_value=encoded) as encoder,
            patch.object(pipeline, "retrieve_dense_from_corpus", return_value=dense) as dense_search,
            patch.object(pipeline, "retrieve_sparse_topics_from_lucene", return_value=sparse) as sparse_search,
            patch.object(pipeline, "_build_recall_evaluation", side_effect=evaluation),
            patch.object(pipeline, "run_stage3", return_value=sparse) as ranker,
            patch.object(pipeline.RecallReporter, "_output_dependencies", side_effect=fixtures.fake_excel_dependencies),
            patch.object(pipeline, "_progress", side_effect=fixtures.ProgressRecorder()),
            patch.object(pipeline.tqdm, "write"),
        ):
            pipeline.main()
        loader.assert_called_once()
        encoder.assert_called_once()
        dense_search.assert_called_once()
        self.assertEqual(sparse_search.call_count, 2)  # Stage 1 and Stage 2, independent of alpha count.
        ranker.assert_called_once()
        self.assertEqual(self.args.hybrid_alpha, 0.6)

    def test_scalar_and_root_errors_broadcast_to_all_ranks(self):
        import torch.distributed as dist
        packet = []
        def broadcast(value, src):
            if value[0] is not None:
                packet[:] = pickle.loads(pickle.dumps(value))
            value[:] = packet
        with patch.object(dist, "broadcast_object_list", side_effect=broadcast):
            for rank in range(3):
                action = (lambda: 0.6) if rank == 0 else lambda: self.fail("worker evaluated")
                self.assertEqual(pipeline._alpha_main_result(pipeline.DistributedContext(rank, rank, 3), action), 0.6)
            for rank in range(3):
                def action():
                    raise OSError("CSV write failed")
                with self.assertRaisesRegex(ValueError, "CSV write failed"):
                    pipeline._alpha_main_result(pipeline.DistributedContext(rank, rank, 3), action)

    def test_empty_worker_keeps_query_ownership_and_reranks_only_its_empty_shard(self):
        import numpy as np
        self.args.stage1_method, self.args.stage1_run = "sparse_concat", None
        encoded = pipeline.EncodedQueries([], np.empty((0, 0), dtype="float32"), {})
        context = pipeline.DistributedContext(2, 2, 3)
        with (
            patch.object(pipeline, "_parser", return_value=SimpleNamespace(parse_args=lambda: self.args)),
            patch.object(pipeline, "_initialise_distributed", return_value=context),
            patch.object(pipeline, "_shutdown_distributed"),
            patch.object(pipeline, "_run_sparse_concat_stage1", return_value=({}, encoded, {}, [1, 1, 0])),
            patch.object(pipeline, "_gather_run", return_value={}),
            patch.object(pipeline, "_record_stage_timing"),
            patch.object(pipeline, "_alpha_main_result", return_value=0.6),
            patch.object(pipeline, "_scatter_run", side_effect=AssertionError("repartitioned")),
            patch.object(pipeline, "_load_retrieval_runtime", side_effect=AssertionError("loaded")),
            patch.object(pipeline, "_release_retrieval_runtime"),
            patch.object(pipeline, "run_stage3", return_value={}) as ranker,
        ):
            pipeline.main()
        self.assertEqual(ranker.call_args.args[1], {})
        self.assertEqual(self.args.hybrid_alpha, 0.6)

    def test_time_estimator_rejects_selection(self):
        with patch.object(sys, "path", [str(ROOT / "src"), *sys.path]):
            import estimate_three_stage_retrieval_time as estimator
            with self.assertRaisesRegex(ValueError, "fixed alpha"):
                estimator.validate_args(self.args)


@unittest.skipUnless(BASH, "Bash required")
class AlphaLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="alpha launcher ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "scripts").mkdir()
        self.names = ["three_stage_retrieval_pipeline" + suffix + ".sh" for suffix in ("", "_tbpr", "_multimodal", "_qwen")]
        for name in self.names:
            shutil.copyfile(ROOT / "scripts" / name, self.root / "scripts" / name)
        (self.root / "src").mkdir()
        (self.root / "src/three_stage_retrieval_pipeline.py").touch()
        (self.root / "model").mkdir()
        (self.root / "dense").mkdir(); (self.root / "dense/corpus_0.pkl").touch()
        (self.root / "sparse/index").mkdir(parents=True); (self.root / "sparse/index/segments_1").touch()
        files = {"flickr": ["flickr_test.csv"], "CUHK-PEDES": ["reid_raw.json"],
                 "WebQA": ["query-00000-of-00001.parquet", "qrels-00000-of-00001.parquet", "corpus-00000-of-00002.parquet", "corpus-00001-of-00002.parquet"]}
        for directory, names in files.items():
            directory = self.root / "data" / directory
            directory.mkdir(parents=True)
            for name in names:
                (directory / name).touch()
        self.record = self.root / "args.bin"
        tool = self.root / "record tool"
        tool.write_bytes(b'#!/usr/bin/env bash\nprintf "%s\\0" "$@" >> "$RECORD"\nprintf "\\n" >> "$RECORD"\n')
        tool.chmod(0o755)
        source = "".join((ROOT / "scripts" / name).read_text(encoding="utf-8") for name in self.names)
        variables = set(re.findall(r"\$\{([A-Z_][A-Z_0-9]*)(?::-|-)", source))
        self.env = {key: val for key, val in os.environ.items() if key not in variables}
        self.env.update(DATA_ROOT=self.root.as_posix(), BASE_MODEL="model", RETRIEVAL_MODEL_PATH=(self.root / "model").as_posix(),
                        RANKER_MODEL_PATH=(self.root / "model").as_posix(), STAGE1_SPARSE_INDEX=(self.root / "sparse").as_posix(),
                        STAGE2_SPARSE_INDEX=(self.root / "sparse").as_posix(), STAGE2_DENSE_INDEX=(self.root / "dense").as_posix(),
                        DEEPSPEED_BIN=tool.as_posix(), RECORD=self.record.as_posix(), GPUS="1",
                        QWEN_OUTPUT_ROOT=(self.root / "results").as_posix(),
                        PATH=str(Path(BASH).parent) + os.pathsep + os.environ.get("PATH", ""))

    def launch(self, number, *args, **env):
        self.record.unlink(missing_ok=True)
        result = subprocess.run([BASH, (self.root / "scripts" / self.names[number]).as_posix(), *args],
                                cwd=self.root.parent, env={**self.env, **env}, capture_output=True,
                                encoding="utf-8", errors="replace", timeout=30)
        calls = [line.decode().rstrip("\0").split("\0") for line in self.record.read_bytes().splitlines()] if self.record.exists() else []
        return result, calls

    def test_all_launchers_pass_switch_validation_paths_and_preserve_explicit_output(self):
        for number in range(3):
            with self.subTest(number=number):
                result, calls = self.launch(number, SELECT_HYBRID_ALPHA="true", VAL_PASSAGE_REPS="/val with spaces/dense", VAL_SPARSE_INDEX="/val/sparse")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                call, = calls
                self.assertIn("--select-hybrid-alpha", call)
                self.assertEqual(call[call.index("--val-passage-reps") + 1], "/val with spaces/dense")
                self.assertTrue(call[call.index("--output-dir") + 1].endswith("/alpha_select"))
                result, calls = self.launch(number, "--select-hybrid-alpha", "--val-passage-reps=/cli dense",
                                            "--val-sparse-index", "/cli sparse", "--output-dir", "/explicit result")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                call, = calls
                self.assertEqual(call[call.index("--output-dir") + 1], "/explicit result")
                self.assertEqual(call[call.index("--val-passage-reps") + 1], "/cli dense")
                result, calls = self.launch(number)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotIn("--select-hybrid-alpha", calls[0])

    def batch_recorder(self):
        fields = ("DATASET", "SELECT_HYBRID_ALPHA", "VAL_PASSAGE_REPS", "VAL_SPARSE_INDEX", "VALIDATION_DATASET_NAME", "OUTPUT_DIR")
        values = " ".join('"${' + name + '-}"' for name in fields)
        (self.root / "scripts" / self.names[0]).write_bytes(
            ('#!/usr/bin/env bash\nprintf "%s\\0" ' + values + ' >> "$RECORD"\nprintf "\\n" >> "$RECORD"\n').encode())

    def test_qwen_batch_validation_tables_isolate_jobs(self):
        self.batch_recorder()
        script = self.root / "scripts" / self.names[3]
        text = script.read_text(encoding="utf-8")
        for table, values in (("VAL_DENSE_INDEXES", {"flickr": "/flickr dense", "coco": "/coco dense"}),
                              ("VAL_SPARSE_INDEXES", {"flickr": "/flickr sparse", "coco": "/coco sparse"}),
                              ("VALIDATION_DATASETS", {"flickr": "flickr", "coco": "coco"})):
            entries = " ".join("[" + key + "]=" + shlex.quote(val) for key, val in values.items())
            text = text.replace("declare -A " + table + "=()", "declare -A " + table + "=(" + entries + ")")
        script.write_bytes(text.encode())
        result, calls = self.launch(3, SELECT_HYBRID_ALPHA="true", VAL_PASSAGE_REPS="stale", VAL_SPARSE_INDEX="stale", VALIDATION_DATASET_NAME="flickr")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([call[:5] for call in calls], [["flickr", "true", "/flickr dense", "/flickr sparse", "flickr"],
                                                     ["coco", "true", "/coco dense", "/coco sparse", "coco"]])
        self.assertTrue(all(call[5].endswith("/alpha_select") for call in calls))

    def test_qwen_shared_validation_requires_explicit_dataset_and_same_task(self):
        self.batch_recorder()
        settings = dict(SELECT_HYBRID_ALPHA="true", VAL_PASSAGE_REPS="/shared dense", VAL_SPARSE_INDEX="/shared sparse")
        result, calls = self.launch(3, **settings)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(calls)
        result, calls = self.launch(3, **settings, VALIDATION_DATASET_NAME="flickr")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([call[4] for call in calls], ["flickr", "flickr"])
        result, calls = self.launch(3, **settings, VALIDATION_DATASET_NAME="flickr", DATASETS="flickr webqa")
        self.assertEqual(result.returncode, 2)
        self.assertFalse(calls)
        result, calls = self.launch(3, SELECT_HYBRID_ALPHA="true")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([call[2:5] for call in calls], [["", "", ""], ["", "", ""]])


if __name__ == "__main__":
    unittest.main()
