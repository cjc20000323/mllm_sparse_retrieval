"""Offline three-stage estimation/report tests; no checkpoints or CUDA required."""

from contextlib import ExitStack
import gc
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
# Register the same pipeline module instance used by its existing regression tests.
import test_three_stage_retrieval_pipeline as fixtures
import estimate_three_stage_retrieval_time as estimate
from time_estimation import StageTimeObserver, merge_stage_time_reports

pipeline = estimate.pipeline


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class Collectives:
    rank = 0
    world_size = 1

    def __init__(self, clock):
        self.clock = clock
        self.gathers = 0

    def barrier(self):
        pass

    def all_done(self, done):
        self.clock.advance(2)  # Timing protocol overhead must not enter the compute estimate.
        return done

    def gather(self, report):
        self.gathers += 1
        return [report]


def arguments(directory, name="webqa", query_type=None):
    task = pipeline.DATASET_TASK_TYPES[name]
    query_type = query_type or ("image" if task in ("ir", "it2t") else "text")
    return estimate.parser().parse_args([
        "--dataset-name", name, "--task-type", task, "--query-type", query_type,
        "--stage1-sparse-index", "sparse", "--stage2-sparse-index", "sparse", "--stage2-dense-index", "dense",
        "--retrieval-model-name-or-path", "query-model", "--ranker-model-name-or-path", "ranker-model",
        "--output-dir", str(directory),
    ])


class EstimationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def execution(self, *, total=100, failure=None, own_context=False):
        args = arguments(self.root / "output")
        clock = Clock()
        collective = Collectives(clock)
        context = pipeline.DistributedContext()
        events, encoded_ref, processed = [], [], []

        def stage1(args, context, timer):
            events.append("stage1")
            timer.components.update(query_encoding=10.0, sparse_search=3.0)
            clock.advance(300)  # Most of this simulated stage is setup, not measured core.
            ids = [str(index) for index in range(total)]
            encoded = pipeline.EncodedQueries(ids, np.zeros((total, 2), dtype=np.float32), {qid: "topic" for qid in ids})
            encoded_ref.append(weakref.ref(encoded))
            run = {qid: {"candidate": 1.0} for qid in ids}
            events.append("retrieval_model_released")
            return run, encoded, run.copy(), [total]

        def stage2(args, local, runtime, *, timing, encoded_queries):
            events.append("stage2_reuse")
            self.assertIsNone(runtime)
            self.assertEqual(encoded_queries.query_ids, list(local))
            if failure == "stage2":
                raise RuntimeError("original stage2 failure")
            timing.components.update(dense_search=4.0, sparse_search=2.0, fusion=1.0)
            return {}, {}, local.copy()

        def stage3(args, hybrid, context, *, timing, progress_observer):
            events.append("ranker_loading")
            gc.collect()
            self.assertIsNone(encoded_ref[0](), "CPU query vectors retained into reranking")
            clock.advance(1000)
            timing.auxiliary["prior_cache_prepare"] = 7.0
            if failure == "stage3":
                raise RuntimeError("original stage3 failure")
            result = {}
            for qid, scores in progress_observer.iterate(hybrid.items(), total_items=len(hybrid), item_count=lambda _: 1):
                clock.advance(5)
                processed.append(qid)
                result[qid] = scores
            return result

        observer = lambda *a, **kw: StageTimeObserver(*a, clock=clock, **kw)
        with ExitStack() as stack:
            stack.enter_context(patch.object(pipeline, "_run_sparse_concat_stage1", new=stage1))
            stack.enter_context(patch.object(pipeline, "run_stage2", new=stage2))
            stack.enter_context(patch.object(pipeline, "run_stage3", new=stage3))
            stack.enter_context(patch.object(pipeline, "_synchronize_cuda_devices"))
            forbidden = [stack.enter_context(patch.object(pipeline, name, side_effect=AssertionError(name))) for name in
                         ("write_json_run", "write_trec_run", "RecallReporter", "TimingReporter", "_load_retrieval_runtime")]
            stack.enter_context(patch.object(estimate, "initialize_context", return_value=context))
            shutdown = stack.enter_context(patch.object(pipeline, "_shutdown_distributed"))
            if failure:
                with self.assertRaisesRegex(RuntimeError, "original " + failure):
                    estimate.run_estimation(args, context=None if own_context else context, collectives=collective,
                                            observer_factory=observer, clock=clock)
                result = None
            else:
                result = estimate.run_estimation(args, context=None if own_context else context, collectives=collective,
                                                 observer_factory=observer, clock=clock)
            for mock in forbidden:
                mock.assert_not_called()
            if own_context:
                shutdown.assert_called_once_with(context, synchronize=not bool(failure))
            else:
                shutdown.assert_not_called()
        saved = json.loads((self.root / "output/time_estimate.json").read_text(encoding="utf-8"))
        return result, saved, processed, collective, events

    def test_stable_estimate_reuses_vectors_releases_models_and_writes_no_metrics(self):
        result, saved, processed, collective, events = self.execution(own_context=True)
        self.assertEqual(saved["status"], "estimated")
        self.assertEqual(processed, [str(index) for index in range(25)])
        self.assertEqual(saved["estimated_total_seconds"], 520.0)
        self.assertEqual(saved["stages"]["stage1"]["estimated_total_seconds"], 13.0)
        self.assertEqual(saved["stages"]["stage2"]["estimated_total_seconds"], 7.0)
        self.assertEqual(saved["stages"]["stage3"]["estimated_total_seconds"], 500.0)
        self.assertGreater(saved["profiling_job_wall_seconds"], 1300)
        self.assertEqual(saved["stages"]["stage3"]["rank_reports"][0]["estimation_control_seconds"], 50.0)
        self.assertEqual(saved["distributed"]["num_queries"], 100)
        self.assertEqual(collective.gathers, 3)
        self.assertEqual(events, ["stage1", "retrieval_model_released", "stage2_reuse", "ranker_loading"])
        self.assertEqual([p.name for p in (self.root / "output").iterdir()], ["time_estimate.json"])
        self.assertEqual(result["estimated_total_seconds"], saved["estimated_total_seconds"])

    def test_short_shard_reports_completed_actual_instead_of_fake_stability(self):
        _, saved, processed, _, _ = self.execution(total=3)
        self.assertEqual(saved["status"], "completed")
        self.assertEqual(len(processed), 3)
        self.assertFalse(saved["stages"]["stage3"]["rank_reports"][0]["stabilized"])
        self.assertEqual(saved["estimated_total_seconds"], 35.0)

    def test_stage2_failure_retains_stage1_without_exception_path_collective(self):
        _, saved, _, collective, _ = self.execution(failure="stage2", own_context=True)
        self.assertEqual(saved["status"], "failed")
        self.assertIsNone(saved["estimated_total_seconds"])
        self.assertEqual(set(saved["stages"]), {"stage1"})
        self.assertEqual(saved["error"]["stage"], "stage2")
        self.assertEqual(collective.gathers, 1)
        self.assertTrue((self.root / "output/failed_rank_0.json").exists())

    def test_rerank_failure_preserves_both_measured_stages(self):
        _, saved, _, collective, _ = self.execution(failure="stage3", own_context=True)
        self.assertEqual(saved["status"], "failed")
        self.assertIsNone(saved["estimated_total_seconds"])
        self.assertEqual(set(saved["stages"]), {"stage1", "stage2"})
        self.assertEqual(collective.gathers, 2)

    def test_all_eight_dataset_directions_validate_without_allocating_models(self):
        for name in estimate.DATASETS:
            task = pipeline.DATASET_TASK_TYPES[name]
            directions = ("text", "image") if task == "ir" else (("image", "composed") if task == "it2t" else ("text",))
            for direction in directions:
                estimate.validate_args(arguments(self.root / "output", name, direction))

    def test_invalid_modes_are_rejected_before_context_initialization(self):
        settings = (("stage1_method", "clip"), ("stage1_run", "run.json"), ("stage1_command", "command"),
                    ("stage2_run", "run.json"), ("ranker_run", "run.json"), ("rerank_type", "listwise"),
                    ("dataset_name", "edis"), ("stage2_dense_index", None), ("retrieval_device_map", "balanced"),
                    ("hybrid_alpha", float("nan")), ("sparse_threads", 0), ("ranker_model_name_or_path", None))
        for field, value in settings:
            args = arguments(self.root / "output")
            setattr(args, field, value)
            with self.subTest(field=field), patch.object(estimate, "initialize_context") as init:
                with self.assertRaises(ValueError):
                    estimate.run_estimation(args)
                init.assert_not_called()

    def test_initialize_pins_even_single_worker_to_its_local_gpu(self):
        import torch
        args = arguments(self.root / "output")
        context = pipeline.DistributedContext()
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "set_device") as set_device, \
             patch.object(pipeline, "_initialise_distributed", return_value=context):
            self.assertIs(estimate.initialize_context(args), context)
        set_device.assert_called_once_with(0)
        self.assertEqual(args.retrieval_device_map, "cuda:0")
        self.assertEqual(args.ranker_device_map, "cuda:0")
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                estimate.initialize_context(args)

    def test_default_stage3_call_does_not_receive_observer_but_profiled_call_does(self):
        args = arguments(self.root / "output")
        ranker = SimpleNamespace(caption_generation_rerank=Mock(return_value={}), rerank=Mock(return_value={}))
        marker = object()
        for template, method in (("caption_generation", ranker.caption_generation_rerank), ("relevant", ranker.rerank)):
            args.rerank_template = template
            pipeline._dispatch_stage3_rerank(ranker, {}, args, None, None, None)
            self.assertNotIn("progress_observer", method.call_args.kwargs)
            pipeline._dispatch_stage3_rerank(ranker, {}, args, None, None, None, progress_observer=marker)
            self.assertIs(method.call_args.kwargs["progress_observer"], marker)

    def test_report_uses_maximum_rank_per_stage_not_gpu_sum(self):
        args = arguments(self.root / "output")
        writer = estimate.TimeEstimateReport(args, pipeline.DistributedContext(0, 0, 3))
        writer.query_counts = [100, 5, 0]
        samples = [{"rank": rank, "source": "computed", "compute_seconds": seconds,
                    "components_seconds": {"compute": seconds}, "auxiliary_seconds": {}} for rank, seconds in enumerate((2, 7, 0))]
        for stage in ("stage1", "stage2"):
            writer.record_compute(stage, pipeline.StageTimer(), SimpleNamespace(gather=lambda _: samples))
        rows = [{"rank": rank, "status": "estimated" if rank == 0 else "completed", "elapsed_seconds": elapsed,
                 "estimated_total_seconds": total, "total_items": count, "completed_items": processed}
                for rank, (elapsed, total, count, processed) in enumerate(((125, 500, 100, 25), (25, 25, 5, 5), (0, 0, 0, 0)))]
        writer.finish(merge_stage_time_reports(rows), pipeline.StageTimer())
        saved = json.loads(writer.path.read_text())
        self.assertEqual(saved["estimated_total_seconds"], 514)
        self.assertEqual(saved["stages"]["stage1"]["critical_rank"], 1)
        self.assertEqual(saved["stages"]["stage3"]["completed_items_including_sampler_padding"], 30)
        self.assertEqual(len(saved["stages"]["stage1"]["rank_reports"]), 3)

    def test_worker_failure_writes_only_its_diagnostic(self):
        writer = estimate.TimeEstimateReport(arguments(self.root / "output"), pipeline.DistributedContext(1, 1, 3))
        writer.save()
        self.assertFalse(writer.path.exists())
        writer.fail("stage3", RuntimeError("worker failed"))
        self.assertFalse(writer.path.exists())
        self.assertTrue((writer.path.parent / "failed_rank_1.json").exists())

    def test_empty_stage3_worker_joins_observer_without_model_or_dataset_loading(self):
        args = arguments(self.root / "output")
        clock = Clock()
        collective = Collectives(clock)
        observer = StageTimeObserver("stage3", early_stop=True, collectives=collective, clock=clock)
        with patch.object(pipeline, "build_dataset", side_effect=AssertionError("dataset loaded")), \
             patch.object(pipeline, "_load_ranker_model", side_effect=AssertionError("model loaded")):
            result = pipeline.run_stage3(args, {}, progress_observer=observer)
        self.assertEqual(result, {})
        self.assertEqual(observer.report["status"], "completed")
        self.assertEqual(observer.report["rank_reports"][0]["completed_items"], 0)
        self.assertEqual(collective.gathers, 1)

    def test_existing_output_requires_overwrite_without_deleting_other_files(self):
        args = arguments(self.root)
        with self.assertRaises(FileExistsError):
            estimate.validate_args(args)
        other = self.root / "normal_results.trec"
        other.write_text("keep")
        args.overwrite = True
        estimate.validate_args(args)
        writer = estimate.TimeEstimateReport(args, pipeline.DistributedContext())
        writer.save()
        self.assertEqual(other.read_text(), "keep")
        self.assertFalse(list(self.root.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
