"""CSV reports retain metrics, partial progress and metadata without Excel dependencies."""
from contextlib import nullcontext, redirect_stderr, redirect_stdout
import csv
import importlib
import io
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MODULES = ("three_stage_retrieval_pipeline", "three_stage_retrieval_pipeline_adaptive_views")

from test_freeret_dense import temporary_directory


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def evaluation(api, categories=None):
    return api.RecallEvaluation(k_values=[1, 5, 10], targets={"q1": {"c1"}, "q2": {"c2"}},
                                query_categories=categories or {}, candidate_identities=None, dataset_query_count=2)


class CsvReportTests(unittest.TestCase):
    def test_recall_round_trip_preserves_every_stage_metric_and_unicode_category(self):
        for module in MODULES:
            with self.subTest(module=module), temporary_directory() as directory:
                api = importlib.import_module(module)
                output = Path(directory)
                args = SimpleNamespace(dataset_name="flickr", dataset_split="test", task_type="ir", query_type="text")
                labels = {"q1": '上衣, "红色"\n室外', "q2": "鞋子"}
                with patch.dict(sys.modules, {"openpyxl": None}), redirect_stdout(io.StringIO()), \
                     patch.object(api, "_build_recall_evaluation", return_value=evaluation(api, labels)):
                    reporter = api.RecallReporter(args, ["q1", "q2"], output)
                    reporter._save()
                    columns, rows = read_csv(output / "recall_metrics.csv")
                    self.assertEqual(columns, ["stage", "variant", "category", "query_count", "min_candidates", "max_candidates", "R@1", "R@5", "R@10"])
                    self.assertFalse(rows)
                    for stage, variants in (("stage1", ("coarse",)), ("stage2", ("dense", "sparse", "hybrid")), ("stage3", ("ranker",))):
                        run = {"q1": {"c1": 2.0, "c0": 1.0}, "q2": {"c0": 2.0, "c2": 1.0}}
                        reporter.record(stage, {variant: run for variant in variants})
                        _, rows = read_csv(output / "recall_metrics.csv")
                        payload = json.loads((output / "recall_metrics.json").read_text(encoding="utf-8"))
                        self.assertEqual(len(rows), len(payload["results"]))
                        for actual, expected in zip(rows, payload["results"]):
                            for key in ("stage", "variant", "category"):
                                self.assertEqual(actual[key], expected[key])
                            for key in ("query_count", "min_candidates", "max_candidates"):
                                self.assertEqual(int(actual[key]), expected[key])
                            for k in (1, 5, 10):
                                self.assertEqual(float(actual[f"R@{k}"]), expected["recall"][str(k)])
                    self.assertEqual({row["stage"] for row in rows}, {"stage1", "stage2", "stage3"})
                self.assertFalse(list(output.glob("*.xlsx")))
                self.assertFalse(list(output.glob("*.tmp.*")))

    def test_timing_csv_preserves_components_totals_and_incomplete_values(self):
        for module in MODULES:
            for completed in (False, True):
                with self.subTest(module=module, completed=completed), temporary_directory() as directory:
                    api = importlib.import_module(module)
                    output = Path(directory)
                    reporter = api.TimingReporter(SimpleNamespace(dataset_name="flickr"), output, time.perf_counter())
                    measurement = dict(source="computed", compute_seconds=1.25, critical_rank=1,
                                       components_seconds={'图文, 编码\n': 0.75, "search": 0.5},
                                       auxiliary_seconds={"output": 0.125})
                    with patch.dict(sys.modules, {"openpyxl": None}), redirect_stdout(io.StringIO()):
                        reporter.record("stage1", measurement)
                        if completed:
                            reporter.record("stage2", dict(measurement, compute_seconds=2.0))
                            reporter.record("stage3", dict(source="precomputed", compute_seconds=0.0, critical_rank=0,
                                                            components_seconds={}, auxiliary_seconds={}))
                        reporter.finish(completed=completed, failed_stage=None if completed else "stage2")
                    columns, rows = read_csv(output / "timing_metrics.csv")
                    self.assertEqual(columns, ["record_type", "stage", "source", "status", "seconds", "duration", "critical_rank", "component"])
                    payload = json.loads((output / "timing_metrics.json").read_text(encoding="utf-8"))
                    stage1 = next(row for row in rows if row["record_type"] == "stage" and row["stage"] == "stage1")
                    self.assertEqual(float(stage1["seconds"]), 1.25)
                    self.assertEqual(int(stage1["critical_rank"]), 1)
                    self.assertTrue(any(row["component"] == '图文, 编码\n' and float(row["seconds"]) == 0.75 for row in rows))
                    total = next(row for row in rows if row["component"] == "core_compute_total")
                    self.assertEqual(total["status"], "completed" if completed else "failed")
                    self.assertEqual(float(total["seconds"]) if total["seconds"] else None, payload["totals"]["compute_seconds"])
                    if completed:
                        self.assertEqual(float(total["seconds"]), 3.25)
                    else:
                        failed = next(row for row in rows if row["record_type"] == "stage" and row["stage"] == "stage2")
                        self.assertEqual(failed["status"], "failed")
                        self.assertEqual(failed["seconds"], "")
                        self.assertEqual(failed["duration"], "N/A")
                    self.assertFalse(list(output.glob("*.xlsx")))

    def test_failed_csv_write_preserves_the_previous_complete_table(self):
        for module in MODULES:
            with self.subTest(module=module), temporary_directory() as directory:
                api = importlib.import_module(module)
                path = Path(directory) / "report.csv"
                api._write_metrics_csv(path, [{"value": "previous"}], ["value"])
                previous = path.read_bytes()
                def broken_rows():
                    yield {"value": "new partial row"}
                    raise OSError("interrupted write")
                with self.assertRaisesRegex(OSError, "interrupted write"):
                    api._write_metrics_csv(path, broken_rows(), ["value"])
                self.assertEqual(path.read_bytes(), previous)

    def test_precomputed_pipeline_publishes_csv_metadata_and_retains_completed_stages_on_failure(self):
        for module in MODULES:
            for failed in (False, True):
                with self.subTest(module=module, failed=failed), temporary_directory() as directory:
                    api = importlib.import_module(module)
                    root = Path(directory)
                    source, output = root / "input.json", root / "results"
                    source.write_text(json.dumps({"q1": {"c1": 2.0}, "q2": {"c2": 1.0}}), encoding="utf-8")
                    argv = [module, "--dataset-name", "flickr", "--query-type", "text", "--output-dir", str(output),
                            "--stage1-run", str(source), "--stage2-run", str(source), "--ranker-run", str(source)]
                    failure = patch.object(api, "run_stage3", side_effect=RuntimeError("ranker failed")) if failed else nullcontext()
                    with patch.object(sys, "argv", argv), patch.object(api, "_initialise_distributed", return_value=api.DistributedContext()), \
                         patch.object(api, "_shutdown_distributed"), patch.object(api, "_build_recall_evaluation", return_value=evaluation(api)), \
                         patch.dict(sys.modules, {"openpyxl": None}), failure, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        if failed:
                            with self.assertRaisesRegex(RuntimeError, "ranker failed"):
                                api.main()
                        else:
                            api.main()
                    _, recalls = read_csv(output / "recall_metrics.csv")
                    _, timings = read_csv(output / "timing_metrics.csv")
                    self.assertEqual({row["stage"] for row in recalls}, {"stage1", "stage2"} if failed else {"stage1", "stage2", "stage3"})
                    self.assertTrue(timings)
                    if not failed:
                        metadata = json.loads((output / "pipeline_metadata.json").read_text(encoding="utf-8"))
                        self.assertIn("recall_metrics.csv", metadata["files"])
                        self.assertIn("timing_metrics.csv", metadata["files"])
                        self.assertEqual(metadata["timing"]["files"], ["timing_metrics.json", "timing_metrics.csv"])
                        self.assertFalse(any(name.endswith(".xlsx") for name in metadata["files"]))
                    self.assertFalse(list(output.glob("*.xlsx")))


if __name__ == "__main__":
    unittest.main()
