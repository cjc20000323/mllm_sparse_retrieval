from __future__ import annotations

import io
import json
import sys
import unittest
from contextlib import ExitStack, redirect_stderr
from pathlib import Path
from typing import Any
from unittest.mock import patch

import test_freeret_time_estimation as fixtures


freeret = fixtures.freeret


class FreeRetActualTimingTests(unittest.TestCase):
    # Reuse only setup/helpers, not the ETA test methods themselves.
    setUp = fixtures.FreeRetTimeEstimationIntegrationTests.setUp
    make_progress = fixtures.FreeRetTimeEstimationIntegrationTests.make_progress
    make_task = fixtures.FreeRetTimeEstimationIntegrationTests.make_task
    make_args = fixtures.FreeRetTimeEstimationIntegrationTests.make_args

    def timed_call(self, function: Any, seconds: float) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            self.clock.advance(seconds)
            return function(*args, **kwargs)
        return call

    def read_report(self, directory: Path | None = None) -> dict[str, Any]:
        return json.loads(((directory or self.root) / "timing_metrics.json").read_text())

    def operation_timers(self) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(patch.object(
            freeret, "topk_similarity", self.timed_call(freeret.topk_similarity, 3),
        ))
        stack.enter_context(patch.object(
            freeret, "prepare_similarity_embeddings",
            self.timed_call(freeret.prepare_similarity_embeddings, 5),
        ))
        stack.enter_context(patch.object(
            freeret, "evaluate_records", self.timed_call(freeret.evaluate_records, 2),
        ))
        return stack

    def main_args(self, datasets: str = "flickr") -> list[str]:
        return [
            "freeret_retrieval.py", "--datasets", datasets,
            "--model_path", "fake-model", "--output_dir", str(self.root),
            "--data_root", str(self.root), "--target_batch_size", "3",
            "--query_batch_size", "2", "--rerank_batch_size", "1",
            "--rerank_num", "2", "--stage1_topk", "2", "--metrics_ks", "1", "2",
        ]

    def test_stage_formulas_outputs_and_wall_time_exclude_timing_write(self) -> None:
        task = self.make_task(queries=3, candidates=5)
        args = self.make_args(enabled=False)
        with self.operation_timers(), patch.object(
            freeret, "_write_timing_json", self.timed_call(freeret._write_timing_json, 11),
        ), patch.object(freeret, "StableRateEstimator", side_effect=AssertionError("no ETA")):
            metrics = freeret.run_task(self.model, task, args, self.root)
        report = self.read_report()
        expected = {
            "candidate_encoding_seconds": 4, "query_encoding_seconds": 4,
            "similarity_search_seconds": 6, "reranking_seconds": 6,
            "online_retrieval_seconds": 10, "retrieval_seconds": 14,
            "total_compute_seconds": 20, "online_retrieval_and_rerank_seconds": 16,
            "target_transfer_seconds": 5, "cache_read_seconds": 0,
            "cache_write_seconds": 0, "task_end_to_end_seconds": 29,
        }
        for key, value in expected.items():
            self.assertEqual(report[key], value, key)
        self.assertEqual(self.clock.now, 40)  # The 11-second timing write is excluded.
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["time_unit"], "seconds")
        self.assertEqual(report["num_queries"], 3)
        self.assertEqual(report["processed_queries"], 3)
        self.assertEqual(report["failed_components"], [])
        self.assertIsNone(report["error"])
        self.assertEqual(json.loads((self.root / "metrics.json").read_text()), metrics)
        self.assertNotIn("retrieval_seconds", metrics)  # Existing result contract unchanged.
        self.assertEqual(len((self.root / "results.jsonl").read_text().splitlines()), 3)
        self.assertTrue((self.root / "config.json").is_file())
        self.assertFalse((self.root / "time_estimate.json").exists())
        self.assertIn("[timing]", self.stdout.getvalue())

    def test_cache_io_and_target_placement_are_not_compute(self) -> None:
        task = self.make_task(queries=3, candidates=5)
        args = self.make_args(enabled=False)
        args.cache_targets = True
        with self.operation_timers(), patch.object(
            freeret.torch, "save", self.timed_call(freeret.torch.save, 4),
        ), patch.object(freeret.torch, "load", self.timed_call(freeret.torch.load, 3)):
            freeret.run_task(self.model, task, args, self.root)
            cold = self.read_report()
            self.model.encode_batch_sizes.clear()
            freeret.run_task(self.model, task, args, self.root)
            warm = self.read_report()
        self.assertFalse(cold["target_cache_hit"])
        self.assertEqual(cold["cache_write_seconds"], 4)
        self.assertEqual(cold["cache_read_seconds"], 0)
        self.assertEqual(cold["total_compute_seconds"], 20)
        self.assertEqual(cold["task_end_to_end_seconds"], 33)
        self.assertTrue(warm["target_cache_hit"])
        self.assertEqual(warm["candidate_encoding_seconds"], 0)
        self.assertEqual(warm["cache_read_seconds"], 3)
        self.assertEqual(warm["cache_write_seconds"], 0)
        self.assertEqual(warm["target_transfer_seconds"], 5)
        self.assertEqual(warm["retrieval_seconds"], 10)
        self.assertEqual(warm["total_compute_seconds"], 16)
        self.assertEqual(warm["task_end_to_end_seconds"], 28)
        self.assertEqual(self.model.encode_batch_sizes, [2, 1])

    def test_rejected_cache_records_read_then_full_encoding_and_rewrite(self) -> None:
        args = self.make_args(enabled=False)
        args.cache_targets = True
        freeret.torch.save({}, self.root / "target_embeddings_fake-model.pt")
        with patch.object(freeret.torch, "load", self.timed_call(freeret.torch.load, 3)):
            freeret.run_task(self.model, self.make_task(queries=1, candidates=5), args, self.root)
        report = self.read_report()
        self.assertFalse(report["target_cache_hit"])
        self.assertEqual(report["cache_read_seconds"], 3)
        self.assertEqual(report["candidate_encoding_seconds"], 4)
        self.assertEqual(report["processed_queries"], 1)

    def test_prompt_preparation_and_final_concat_are_timed(self) -> None:
        # Both one-word prompt preparation and rerank prompt preparation belong
        # to the respective stages; final target concatenation does too.
        with patch.object(freeret, "one_word_prompt", self.timed_call(freeret.one_word_prompt, 0.5)), \
                patch.object(freeret, "rerank_prompt", self.timed_call(freeret.rerank_prompt, 0.25)), \
                patch.object(freeret.torch, "cat", self.timed_call(freeret.torch.cat, 0.75)):
            freeret.run_task(
                self.model, self.make_task(queries=3, candidates=5),
                self.make_args(enabled=False), self.root,
            )
        report = self.read_report()
        self.assertEqual(report["candidate_encoding_seconds"], 4 + 2.5 + 0.75)
        self.assertEqual(report["query_encoding_seconds"], 4 + 1.5)
        self.assertEqual(report["reranking_seconds"], 6 + 1.5)

    def test_filters_and_query_limit_apply_to_actual_counts(self) -> None:
        task = self.make_task(queries=6, candidates=2)
        task.queries[0].relevant_ids.clear()
        args = self.make_args(enabled=False)
        args.max_queries = 3
        freeret.run_task(self.model, task, args, self.root)
        report = self.read_report()
        self.assertEqual(report["num_queries"], 3)
        self.assertEqual(report["processed_queries"], 3)
        self.assertEqual(report["skipped_unjudged_queries"], 1)

    def test_query_failure_saves_partial_durations_and_original_error(self) -> None:
        encode = self.model.encode_one_word
        def fail_on_second_query(texts: Any, paths: Any) -> Any:
            if len(self.model.encode_batch_sizes) == 2:
                self.clock.advance(7)
                raise RuntimeError("query encoding failed")
            return encode(texts, paths)
        with patch.object(self.model, "encode_one_word", side_effect=fail_on_second_query), \
                self.assertRaisesRegex(RuntimeError, "query encoding failed"):
            freeret.run_task(
                self.model, self.make_task(queries=3, candidates=2),
                self.make_args(enabled=False), self.root,
            )
        report = self.read_report()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["processed_queries"], 2)
        self.assertEqual(report["candidate_encoding_seconds"], 2)
        self.assertEqual(report["query_encoding_seconds"], 9)
        self.assertEqual(report["reranking_seconds"], 4)
        self.assertEqual(report["failed_components"], ["query_encoding_seconds"])
        self.assertEqual(report["error"]["message"], "query encoding failed")
        self.assertFalse((self.root / "metrics.json").exists())
        self.assertTrue(all(bar.closed for bar in self.progress))

    def test_task_timing_write_failure_does_not_mask_model_error(self) -> None:
        with patch.object(self.model, "rerank", side_effect=RuntimeError("model failed")), \
                patch.object(freeret, "_write_timing_json", side_effect=OSError("disk unavailable")), \
                redirect_stderr(io.StringIO()) as errors, \
                self.assertRaisesRegex(RuntimeError, "model failed"):
            freeret.run_task(
                self.model, self.make_task(queries=1, candidates=2),
                self.make_args(enabled=False), self.root,
            )
        self.assertIn("disk unavailable", errors.getvalue())

    def test_main_aggregates_four_tasks_loads_model_once_and_includes_data_load(self) -> None:
        snapshots: list[dict[str, Any]] = []
        original_writer = freeret._write_timing_json
        def timing_write(path: Path, report: dict[str, Any]) -> None:
            if path.name == "timing_summary.json":
                snapshots.append(report)
            self.clock.advance(1)
            original_writer(path, report)
        def load_task(root: Any, dataset: str, split: str, direction: str, materializer: Any) -> Any:
            self.clock.advance(7)
            task = self.make_task(queries=3, candidates=5, dataset=dataset, direction=direction)
            task.split = "actual_test"  # E.g. fallback to a real split in a data adapter.
            return task
        with patch.object(sys, "argv", self.main_args("flickr,coco")), \
                patch.object(freeret, "FreeRetQwen", side_effect=self.timed_call(lambda **_: self.model, 10)) as model_load, \
                patch.object(freeret, "load_task", side_effect=load_task), \
                patch.object(freeret, "_write_timing_json", side_effect=timing_write), \
                self.operation_timers():
            freeret.main()
        output_root = self.root / "fake-model"
        report = json.loads((output_root / "timing_summary.json").read_text())
        self.assertEqual(model_load.call_count, 1)
        self.assertEqual(report["model_loading_seconds"], 10)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["requested_tasks"], 4)
        self.assertEqual(report["completed_tasks"], 4)
        self.assertEqual(report["failed_tasks"], 0)
        self.assertEqual(report["measured_totals"]["retrieval_seconds"], 56)
        self.assertEqual(report["measured_totals"]["reranking_seconds"], 24)
        self.assertEqual(report["measured_totals"]["total_compute_seconds"], 80)
        self.assertEqual(report["task_end_to_end_seconds_sum"], 144)  # 4 * (29+7)
        # Shared model load once, all task work, and 9 intermediate timing writes.
        self.assertEqual(report["end_to_end_seconds"], 10 + 144 + 9)
        self.assertEqual(self.clock.now, report["end_to_end_seconds"] + 1)
        self.assertEqual([r["completed_tasks"] for r in snapshots], [0, 1, 2, 3, 4, 4])
        normal_metrics = json.loads((output_root / "summary.json").read_text())
        self.assertEqual(len(normal_metrics), 4)
        self.assertEqual(
            [(task["dataset"], task["direction"]) for task in report["tasks"]],
            [("flickr", "t2i"), ("flickr", "i2t"), ("coco", "t2i"), ("coco", "i2t")],
        )
        for task, metrics in zip(report["tasks"], normal_metrics):
            path = output_root / task["dataset"] / f"actual_test_{task['direction']}"
            self.assertEqual(task["split"], "actual_test")
            self.assertEqual(task["task_end_to_end_seconds"], 36)
            self.assertEqual(task["processed_queries"], 3)
            self.assertEqual(self.read_report(path), task)
            self.assertEqual(json.loads((path / "metrics.json").read_text()), metrics)
            self.assertTrue((path / "results.jsonl").is_file())
        self.assertFalse((output_root / "time_estimate_summary.json").exists())

    def test_main_records_load_failure_without_losing_completed_task(self) -> None:
        def load_task(root: Any, dataset: str, split: str, direction: str, materializer: Any) -> Any:
            self.clock.advance(7)
            if direction == "i2t":
                raise RuntimeError("dataset load failed")
            return self.make_task(queries=1, candidates=2)
        with patch.object(sys, "argv", self.main_args()), \
                patch.object(freeret, "FreeRetQwen", return_value=self.model), \
                patch.object(freeret, "load_task", side_effect=load_task), \
                redirect_stderr(io.StringIO()), \
                self.assertRaisesRegex(RuntimeError, "dataset load failed"):
            freeret.main()
        report = json.loads((self.root / "fake-model" / "timing_summary.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["completed_tasks"], 1)
        self.assertEqual(report["failed_tasks"], 1)
        self.assertEqual(report["tasks"][0]["status"], "completed")
        self.assertEqual(report["tasks"][1]["status"], "failed")
        self.assertEqual(report["tasks"][1]["task_end_to_end_seconds"], 7)
        self.assertEqual(report["tasks"][1]["processed_queries"], 0)
        self.assertEqual(report["tasks"][1]["total_compute_seconds"], 0)
        self.assertEqual(report["tasks"][1]["error"]["message"], "dataset load failed")
        self.assertTrue((self.root / "fake-model" / "flickr" / "test_t2i" / "timing_metrics.json").exists())

    def test_main_records_model_load_failure_and_its_elapsed_time(self) -> None:
        def model_load(**kwargs: Any) -> Any:
            self.clock.advance(10)
            raise RuntimeError("model load failed")
        with patch.object(sys, "argv", self.main_args()), \
                patch.object(freeret, "FreeRetQwen", side_effect=model_load), \
                redirect_stderr(io.StringIO()), \
                self.assertRaisesRegex(RuntimeError, "model load failed"):
            freeret.main()
        report = json.loads((self.root / "fake-model" / "timing_summary.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["model_loading_seconds"], 10)
        self.assertEqual(report["end_to_end_seconds"], 10)
        self.assertEqual(report["started_tasks"], 0)
        self.assertEqual(report["completed_tasks"], 0)
        self.assertEqual(report["error"]["message"], "model load failed")

    def test_main_timing_save_failure_keeps_original_failure(self) -> None:
        writer = freeret._write_timing_json
        def write(path: Path, report: Any) -> None:
            if report["status"] == "failed":
                raise OSError("timing disk failure")
            writer(path, report)
        with patch.object(sys, "argv", self.main_args()), \
                patch.object(freeret, "FreeRetQwen", side_effect=RuntimeError("original load error")), \
                patch.object(freeret, "_write_timing_json", side_effect=write), \
                redirect_stderr(io.StringIO()) as errors, \
                self.assertRaisesRegex(RuntimeError, "original load error"):
            freeret.main()
        self.assertIn("timing disk failure", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
