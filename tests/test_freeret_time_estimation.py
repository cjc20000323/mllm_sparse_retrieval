from __future__ import annotations

import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "freeret_time_estimation_test_module", PROJECT_ROOT / "src" / "freeret_retrieval.py"
)
assert SPEC is not None and SPEC.loader is not None
freeret: Any = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = freeret
SPEC.loader.exec_module(freeret)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeProgress:
    def __init__(self, **kwargs: Any) -> None:
        self.total = kwargs["total"]
        self.n = 0
        self.closed = False
        self.messages: list[str] = []

    def __enter__(self) -> "FakeProgress":
        return self

    def __exit__(self, *_: Any) -> None:
        self.closed = True

    def update(self, count: int) -> None:
        self.n += count

    def write(self, message: str) -> None:
        self.messages.append(message)


class FakeModel:
    device = torch.device("cpu")
    model_family = "fake"
    model_class_name = "FakeModel"
    processor_class_name = "FakeProcessor"
    transformers_version = "test"
    attn_implementation = "sdpa"
    _decoder_norm_path = "fake.norm"
    max_pixels_applied = True

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.encode_batch_sizes: list[int] = []
        self.rerank_batch_sizes: list[int] = []

    def set_max_pixels(self, pixels: int) -> None:
        self.max_pixels = pixels

    def encode_one_word(self, texts: list[str], paths: Any) -> torch.Tensor:
        self.encode_batch_sizes.append(len(texts))
        self.clock.advance(2.0)
        return torch.ones((len(texts), 2))

    def rerank(self, texts: list[str], paths: Any) -> torch.Tensor:
        self.rerank_batch_sizes.append(len(texts))
        self.clock.advance(1.0)
        return torch.ones(len(texts))


class StableRateEstimatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()

    def test_warmup_counts_batches_not_individual_items(self) -> None:
        estimator = freeret.StableRateEstimator(1000, clock=self.clock)
        for batch in range(1, 21):
            self.clock.advance(10.0)
            self.assertFalse(estimator.observe_batch(batch * 4))
        self.assertEqual(estimator.num_rate_samples, 0)
        for batch in range(21, 26):
            self.clock.advance(5.0)
            self.assertEqual(estimator.observe_batch(batch * 4), batch == 25)
        report = estimator.result(completed=False)
        self.assertEqual(report["rate_samples"], [0.8] * 5)
        self.assertEqual(report["completed_batches"], 25)
        self.assertEqual(report["elapsed_seconds"], 225.0)
        self.assertEqual(report["remaining_seconds"], 1125.0)
        self.assertEqual(report["estimated_total_seconds"], 1350.0)

    def test_first_sample_waits_five_seconds_after_warmup(self) -> None:
        estimator = freeret.StableRateEstimator(100, clock=self.clock, warmup_batches=1)
        self.clock.advance(100)
        estimator.observe_batch(1)
        self.clock.advance(4.99)
        estimator.observe_batch(2)
        self.assertEqual(estimator.num_rate_samples, 0)
        self.clock.advance(0.02)
        estimator.observe_batch(3)
        self.assertEqual(estimator.num_rate_samples, 1)
        self.assertAlmostEqual(estimator.rates[0], 2 / 5.01)

    def test_slow_batch_produces_one_sample_not_multiple(self) -> None:
        estimator = freeret.StableRateEstimator(100, clock=self.clock, warmup_batches=0)
        self.clock.advance(100)
        self.assertFalse(estimator.observe_batch(2))
        self.assertEqual(estimator.num_rate_samples, 1)
        self.assertEqual(list(estimator.rates), [0.02])

    def test_unstable_window_slides_until_five_stable_rates(self) -> None:
        estimator = freeret.StableRateEstimator(100, clock=self.clock, warmup_batches=0)
        for count, interval in enumerate([5, 25, 5, 5, 5, 5, 5], 1):
            self.clock.advance(interval)
            self.assertEqual(estimator.observe_batch(count), count == 7)
        self.assertEqual(estimator.num_rate_samples, 7)
        self.assertEqual(list(estimator.rates), [0.2] * 5)

    def test_cv_threshold_is_strict_not_inclusive(self) -> None:
        rates = [1.0, 1.0, 1.0, 1.0, 1.1]
        mean = sum(rate / 5 for rate in rates)
        boundary = math.sqrt(sum((rate / mean - 1) ** 2 for rate in rates) / 5)
        estimator = freeret.StableRateEstimator(100, clock=self.clock, max_cv=boundary)
        # Supply rates directly to test the exact floating-point boundary.
        estimator.rates.extend(rates[:4])
        estimator.completed_batches = 20
        self.clock.advance(10)
        self.assertFalse(estimator.observe_batch(11))
        self.assertAlmostEqual(estimator._window_stats()[1], boundary)
        self.assertIsNone(estimator.stable_estimate)

    def test_nonfinite_observation_resets_consecutive_window(self) -> None:
        estimator = freeret.StableRateEstimator(100, clock=self.clock, warmup_batches=0)
        for count in range(1, 5):
            self.clock.advance(5)
            estimator.observe_batch(count)
        self.clock.now = math.inf
        self.assertFalse(estimator.observe_batch(5))
        self.assertEqual(list(estimator.rates), [])
        self.assertIsNone(estimator.stable_estimate)

    def test_small_completed_stage_saves_actual_duration(self) -> None:
        estimator = freeret.StableRateEstimator(2, clock=self.clock)
        for count in (1, 2):
            self.clock.advance(3)
            estimator.observe_batch(count)
        report = estimator.result(completed=True)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["estimated_total_seconds"], 6)
        self.assertEqual(report["remaining_seconds"], 0)
        self.assertFalse(report["stabilized"])
        self.assertIsNone(report["rate_per_second"])

    def test_unstable_completed_stage_does_not_claim_estimated_status(self) -> None:
        estimator = freeret.StableRateEstimator(8, clock=self.clock, warmup_batches=0)
        for count in range(1, 9):
            self.clock.advance(5 if count % 2 else 25)
            estimator.observe_batch(count)
        report = estimator.result(completed=True)
        self.assertEqual(report["status"], "completed")
        self.assertFalse(report["stabilized"])
        self.assertEqual(report["estimated_total_seconds"], 120)

    def test_target_can_continue_after_stability_and_keep_original_estimate(self) -> None:
        estimator = freeret.StableRateEstimator(6, clock=self.clock, warmup_batches=0)
        for count in range(1, 6):
            self.clock.advance(5)
            estimator.observe_batch(count)
        self.clock.advance(20)
        self.assertFalse(estimator.observe_batch(6))
        report = estimator.result(completed=True)
        self.assertEqual(report["estimated_total_seconds"], 45)
        self.assertEqual(report["stable_estimate"]["estimated_total_seconds"], 30)
        self.assertEqual(report["completed_batches"], 6)

    def test_incomplete_unstable_stage_cannot_be_reported_as_success(self) -> None:
        estimator = freeret.StableRateEstimator(10, clock=self.clock)
        with self.assertRaises(ValueError):
            estimator.result(completed=False)
        with self.assertRaises(ValueError):
            estimator.result(completed=True)


class FreeRetTimeEstimationIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image = self.root / "tiny.png"
        Image.new("RGB", (2, 2)).save(self.image)
        self.clock = FakeClock()
        self.model = FakeModel(self.clock)
        self.progress: list[FakeProgress] = []
        self.stdout = io.StringIO()
        # Keep real CPU tensor operations and prompts; no checkpoint/API/GPU.
        self.redirect = redirect_stdout(self.stdout)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)
        self.timer_patch = patch.object(freeret.time, "perf_counter", self.clock)
        self.timer_patch.start()
        self.addCleanup(self.timer_patch.stop)
        self.progress_patch = patch.object(freeret, "tqdm", self.make_progress)
        self.progress_patch.start()
        self.addCleanup(self.progress_patch.stop)

    def make_progress(self, **kwargs: Any) -> FakeProgress:
        bar = FakeProgress(**kwargs)
        self.progress.append(bar)
        return bar

    def make_task(
        self, queries: int = 200, candidates: int = 120,
        dataset: str = "flickr", direction: str = "t2i",
    ) -> Any:
        return freeret.RetrievalTask(
            dataset=dataset, split="test", direction=direction,
            prompt_key=freeret.prompt_key_for(dataset, direction),
            queries=[
                freeret.QueryItem(
                    query_id=f"q-{index}", text=f"query {index}",
                    image_path=str(self.image) if direction == "i2t" else None,
                    relevant_ids={"c-0"},
                )
                for index in range(queries)
            ],
            candidates=[
                freeret.CandidateItem(
                    candidate_id=f"c-{index}", text=f"candidate {index}",
                    image_path=str(self.image) if direction == "t2i" else None,
                )
                for index in range(candidates)
            ],
        )

    def make_args(self, enabled: bool = True) -> Any:
        args = freeret.build_parser().parse_args(
            ["--estimate_time_only"] if enabled else []
        )
        args.model_path = "fake-model"
        args.target_batch_size = 3
        args.query_batch_size = 2
        args.rerank_batch_size = 1
        args.rerank_num = 2
        args.stage1_topk = 2
        args.metrics_ks = [1, 2]
        return args

    def test_full_targets_then_early_query_exit_and_seconds_formula(self) -> None:
        report = freeret.run_task(self.model, self.make_task(), self.make_args(), self.root)
        self.assertEqual(self.model.encode_batch_sizes[:40], [3] * 40)
        self.assertEqual(self.model.encode_batch_sizes[40:], [2] * 25)
        self.assertEqual(len(self.model.rerank_batch_sizes), 100)
        self.assertEqual(report["processed_queries"], 50)
        self.assertEqual(report["status"], "estimated")
        self.assertEqual(report["target_elapsed_seconds"], 80.0)
        self.assertEqual(report["query_elapsed_seconds"], 150.0)
        self.assertEqual(report["query_estimated_total_seconds"], 600.0)
        self.assertEqual(report["estimated_total_seconds"], 680.0)
        target = report["stages"]["target_encoding"]
        self.assertEqual(target["completed_items"], 120)
        self.assertEqual(target["completed_batches"], 40)
        self.assertTrue(target["stabilized"])
        self.assertEqual(report["stages"]["query_retrieval_rerank"]["completed_batches"], 25)
        self.assertEqual([bar.n for bar in self.progress], [120, 50])
        self.assertTrue(all(bar.closed for bar in self.progress))
        self.assertEqual(json.loads((self.root / "time_estimate.json").read_text()), report)
        for filename in ("metrics.json", "results.jsonl", "config.json", "timing_metrics.json", "timing_summary.json"):
            self.assertFalse((self.root / filename).exists())

    def test_existing_normal_outputs_are_preserved(self) -> None:
        for name in ("metrics.json", "results.jsonl", "config.json"):
            (self.root / name).write_text("original", encoding="utf-8")
        freeret.run_task(self.model, self.make_task(queries=1, candidates=2), self.make_args(), self.root)
        for name in ("metrics.json", "results.jsonl", "config.json"):
            self.assertEqual((self.root / name).read_text(), "original")

    def test_small_dataset_reports_actual_total_not_early_exit(self) -> None:
        report = freeret.run_task(self.model, self.make_task(queries=3, candidates=5), self.make_args(), self.root)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["processed_queries"], 3)
        self.assertEqual(report["query_remaining_seconds"], 0)
        self.assertEqual(report["estimated_total_seconds"], 14)
        self.assertFalse(report["stages"]["query_retrieval_rerank"]["stabilized"])

    def test_stability_on_last_query_batch_is_still_completed(self) -> None:
        report = freeret.run_task(self.model, self.make_task(queries=50, candidates=2), self.make_args(), self.root)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["processed_queries"], 50)
        self.assertTrue(report["stages"]["query_retrieval_rerank"]["stabilized"])

    def test_filtered_queries_and_max_queries_define_estimation_total(self) -> None:
        task = self.make_task(queries=200, candidates=2)
        task.queries[0].relevant_ids.clear()
        args = self.make_args()
        args.max_queries = 60
        report = freeret.run_task(self.model, task, args, self.root)
        self.assertEqual(report["num_queries"], 60)
        self.assertEqual(report["skipped_unjudged_queries"], 1)
        self.assertEqual(report["query_estimated_total_seconds"], 180)

    def test_full_cache_is_reused_and_read_time_is_recorded(self) -> None:
        task = self.make_task()
        args = self.make_args()
        args.cache_targets = True
        cache_path = self.root / "target_embeddings_fake-model.pt"
        torch.save(
            {"prompt_version": freeret.PROMPT_VERSION, "prompt_key": task.prompt_key,
             "max_pixels": freeret.get_prompt_spec(task.prompt_key).max_pixels,
             "candidate_ids": [item.candidate_id for item in task.candidates],
             "embeddings": torch.ones((120, 2))},
            cache_path,
        )
        real_load = freeret.torch.load

        def timed_load(*args: Any, **kwargs: Any) -> Any:
            self.clock.advance(3)
            return real_load(*args, **kwargs)

        with patch.object(freeret.torch, "load", timed_load):
            report = freeret.run_task(self.model, task, args, self.root)
        self.assertTrue(report["target_cache_hit"])
        self.assertEqual(report["stages"]["target_encoding"]["status"], "cached")
        self.assertEqual(report["target_elapsed_seconds"], 3)
        self.assertEqual(report["estimated_total_seconds"], 603)
        self.assertEqual(self.model.encode_batch_sizes, [2] * 25)

    def test_cache_miss_still_writes_all_target_embeddings(self) -> None:
        args = self.make_args()
        args.cache_targets = True
        report = freeret.run_task(self.model, self.make_task(), args, self.root)
        cache = torch.load(self.root / "target_embeddings_fake-model.pt", map_location="cpu")
        self.assertFalse(report["target_cache_hit"])
        self.assertEqual(tuple(cache["embeddings"].shape), (120, 2))

    def test_default_mode_runs_all_queries_and_never_calls_estimator(self) -> None:
        args = self.make_args(enabled=False)
        self.assertFalse(args.estimate_time_only)
        with patch.object(freeret, "StableRateEstimator", side_effect=AssertionError("disabled")):
            report = freeret.run_task(self.model, self.make_task(queries=60, candidates=2), args, self.root)
        self.assertEqual(report["num_queries"], 60)
        self.assertGreater(self.clock.calls, 0)
        self.assertTrue((self.root / "metrics.json").is_file())
        self.assertTrue((self.root / "timing_metrics.json").is_file())
        self.assertEqual(len((self.root / "results.jsonl").read_text().splitlines()), 60)
        self.assertFalse((self.root / "time_estimate.json").exists())

    def test_real_failure_closes_bar_and_does_not_save_false_estimate(self) -> None:
        with patch.object(self.model, "rerank", side_effect=RuntimeError("real model failure")):
            with self.assertRaisesRegex(RuntimeError, "real model failure"):
                freeret.run_task(self.model, self.make_task(queries=2, candidates=2), self.make_args(), self.root)
        self.assertTrue(all(bar.closed for bar in self.progress))
        self.assertFalse((self.root / "time_estimate.json").exists())

    def test_main_profiles_both_directions_for_all_datasets_and_preserves_summary(self) -> None:
        output_root = self.root / "fake-model"
        output_root.mkdir()
        normal_summary = output_root / "summary.json"
        normal_summary.write_text("original", encoding="utf-8")

        def load_task(root: Any, dataset: str, split: str, direction: str, materializer: Any) -> Any:
            return self.make_task(dataset=dataset, direction=direction)

        args = ["freeret_retrieval.py", "--datasets", "flickr,coco", "--estimate_time_only",
                "--model_path", "fake-model", "--output_dir", str(self.root),
                "--data_root", str(self.root), "--rerank_num", "2",
                "--target_batch_size", "3", "--query_batch_size", "2",
                "--rerank_batch_size", "1"]
        with patch.object(sys, "argv", args), patch.object(freeret, "FreeRetQwen", return_value=self.model), \
                patch.object(freeret, "load_task", side_effect=load_task):
            freeret.main()
        summary = json.loads((output_root / "time_estimate_summary.json").read_text())
        self.assertEqual(summary["completed_tasks"], 4)
        self.assertTrue(summary["all_tasks_processed"])
        self.assertEqual(summary["status"], "estimated")
        self.assertEqual(
            [(task["dataset"], task["direction"]) for task in summary["tasks"]],
            [("flickr", "t2i"), ("flickr", "i2t"), ("coco", "t2i"), ("coco", "i2t")],
        )
        self.assertEqual(summary["estimated_total_seconds"], sum(t["estimated_total_seconds"] for t in summary["tasks"]))
        self.assertEqual(normal_summary.read_text(), "original")
        self.assertFalse((output_root / "timing_summary.json").exists())
        for task in summary["tasks"]:
            self.assertTrue(Path(task["report_path"]).is_file())
            self.assertEqual(task["processed_queries"], 50)
            self.assertEqual(task["stages"]["target_encoding"]["completed_items"], 120)
            self.assertFalse(Path(task["report_path"]).with_name("timing_metrics.json").exists())

    def test_main_persists_finished_tasks_before_later_task_failure(self) -> None:
        def load_task(root: Any, dataset: str, split: str, direction: str, materializer: Any) -> Any:
            if direction == "i2t":
                raise RuntimeError("second task failed")
            return self.make_task(queries=1, candidates=2)

        args = ["freeret_retrieval.py", "--datasets", "flickr", "--estimate_time_only",
                "--model_path", "fake-model", "--output_dir", str(self.root)]
        with patch.object(sys, "argv", args), patch.object(freeret, "FreeRetQwen", return_value=self.model), \
                patch.object(freeret, "load_task", side_effect=load_task), \
                self.assertRaisesRegex(RuntimeError, "second task failed"):
            freeret.main()
        summary = json.loads((self.root / "fake-model" / "time_estimate_summary.json").read_text())
        self.assertEqual(summary["status"], "in_progress")
        self.assertEqual(summary["completed_tasks"], 1)
        self.assertFalse(summary["all_tasks_processed"])


if __name__ == "__main__":
    unittest.main()
