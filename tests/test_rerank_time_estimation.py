from __future__ import annotations

import ast
import io
import json
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from time_estimation import (
    RerankTimeEstimationArguments, RerankTimeEstimateWriter,
    StageTimeObserver, TorchStageCollectives, merge_stage_time_reports,
)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class SingleRank:
    rank = 0
    world_size = 1

    def __init__(self) -> None:
        self.control_rounds = 0
        self.gathers = 0

    def barrier(self) -> None:
        pass

    def all_done(self, done: bool) -> bool:
        self.control_rounds += 1
        return done

    def gather(self, report: Any) -> list[Any]:
        self.gathers += 1
        return [report]


class ThreadRanks:
    """Deterministic collective transport, without network/GPU dependencies."""

    def __init__(self, size: int) -> None:
        self.world_size = size
        self.sync = threading.Barrier(size, timeout=5)
        self.done = [False] * size
        self.reports: list[Any] = [None] * size

    def rank_transport(self, rank: int) -> Any:
        group = self

        class Transport:
            def barrier(self) -> None:
                group.sync.wait()

            def all_done(self, done: bool) -> bool:
                group.done[rank] = done
                group.sync.wait()
                result = all(group.done)
                group.sync.wait()
                return result

            def gather(self, report: Any) -> list[Any]:
                group.reports[rank] = report
                group.sync.wait()
                return list(group.reports)

        transport = Transport()
        transport.rank = rank
        transport.world_size = self.world_size
        return transport


class Progress:
    def __init__(self, **kwargs: Any) -> None:
        self.total = kwargs["total"]
        self.n = 0
        self.closed = False
        self.messages: list[str] = []

    def __enter__(self) -> "Progress":
        return self

    def __exit__(self, *_: Any) -> None:
        self.closed = True

    def update(self, value: int) -> None:
        self.n += value

    def write(self, value: str) -> None:
        self.messages.append(value)


class StageObserverTests(unittest.TestCase):
    def test_retrieval_keeps_running_after_stability(self) -> None:
        clock, transport = Clock(), SingleRank()
        observer = StageTimeObserver("retrieval", early_stop=False, collectives=transport, clock=clock)
        processed = []
        for batch in observer.iterate(range(40), total_items=80, item_count=lambda _: 2):
            clock.advance(5)
            processed.append(batch)
        self.assertEqual(len(processed), 40)
        self.assertEqual(transport.control_rounds, 0)
        self.assertEqual(observer.report["status"], "completed")
        self.assertEqual(observer.report["elapsed_seconds"], 200)
        local = observer.report["rank_reports"][0]
        self.assertEqual(local["stable_estimate"]["completed_batches"], 25)
        self.assertEqual(local["completed_batches"], 40)

    def test_rerank_stops_after_twenty_warmup_queries_and_five_samples(self) -> None:
        clock, transport = Clock(), SingleRank()
        synchronizations = []
        observer = StageTimeObserver(
            "rerank", early_stop=True, collectives=transport, clock=clock,
            synchronize=lambda: synchronizations.append(True),
        )
        processed = []
        for query in observer.iterate(range(100), total_items=100, item_count=lambda _: 1):
            clock.advance(5)
            processed.append(query)
        self.assertEqual(len(processed), 25)
        self.assertEqual(len(synchronizations), 26)  # phase boundary + completed queries
        self.assertEqual(observer.report["status"], "estimated")
        self.assertEqual(observer.report["estimated_total_seconds"], 500)
        self.assertEqual(observer.report["elapsed_seconds"], 125)

    def test_progress_updates_after_work_and_closes_when_stopping(self) -> None:
        clock = Clock()
        bars = []

        def factory(**kwargs: Any) -> Progress:
            bars.append(Progress(**kwargs))
            return bars[-1]

        observer = StageTimeObserver("rerank", early_stop=True, collectives=SingleRank(), clock=clock)
        for index in observer.progress(
            range(100), total=100, total_items=100, item_count=lambda _: 1,
            progress_factory=factory, desc="test", unit="query",
        ):
            self.assertEqual(bars[0].n, index)
            clock.advance(5)
        self.assertEqual(bars[0].n, 25)
        self.assertTrue(bars[0].closed)
        self.assertEqual(len(bars[0].messages), 1)

    def test_query_count_and_batch_count_are_distinct(self) -> None:
        clock = Clock()
        observer = StageTimeObserver("retrieval", early_stop=True, collectives=SingleRank(), clock=clock)
        for _ in observer.iterate(range(100), total_items=400, item_count=lambda _: 4):
            clock.advance(10)
        local = observer.report["rank_reports"][0]
        self.assertEqual(local["completed_batches"], 25)
        self.assertEqual(local["completed_items"], 100)
        self.assertEqual(local["rate_samples"], [0.4] * 5)

    def test_control_waits_are_not_mistaken_for_model_compute_time(self) -> None:
        clock = Clock()

        class SlowControl(SingleRank):
            def all_done(self, done: bool) -> bool:
                clock.advance(100)
                return done

        observer = StageTimeObserver("rerank", early_stop=True, collectives=SlowControl(), clock=clock)
        for _ in observer.iterate(range(100), total_items=100, item_count=lambda _: 1):
            clock.advance(5)
        local = observer.report["rank_reports"][0]
        self.assertEqual(local["completed_items"], 25)
        self.assertEqual(local["rate_samples"], [0.2] * 5)
        self.assertEqual(local["elapsed_seconds"], 125)
        self.assertEqual(local["estimated_total_seconds"], 500)
        self.assertEqual(local["estimation_control_seconds"], 2500)
        self.assertEqual(local["profiling_wall_seconds"], 2625)

    def test_small_and_empty_stages_report_actual_completion(self) -> None:
        for size in (0, 3, 25):
            with self.subTest(size=size):
                clock = Clock()
                observer = StageTimeObserver("rerank", early_stop=True, collectives=SingleRank(), clock=clock)
                for _ in observer.iterate(range(size), total_items=size, item_count=lambda _: 1):
                    clock.advance(5)
                self.assertEqual(observer.report["status"], "completed")
                self.assertEqual(observer.report["estimated_total_seconds"], size * 5)

    def test_input_mismatch_and_cancellation_do_not_gather_false_success(self) -> None:
        transport = SingleRank()
        observer = StageTimeObserver("rerank", early_stop=True, collectives=transport, clock=Clock())
        with self.assertRaisesRegex(ValueError, "declared item count"):
            list(observer.iterate([], total_items=1, item_count=lambda _: 1))
        self.assertIsNone(observer.report)
        self.assertEqual(transport.gathers, 0)
        observer = StageTimeObserver("rerank", early_stop=True, collectives=transport, clock=Clock())
        iterator = observer.iterate(range(10), total_items=10, item_count=lambda _: 1)
        next(iterator)
        iterator.close()
        self.assertIsNone(observer.report)
        self.assertEqual(transport.gathers, 0)

    def run_threaded(self, sizes: list[int], *, early_stop: bool, uneven_rates: bool = False) -> list[Any]:
        group = ThreadRanks(len(sizes))
        results: list[Any] = [None] * len(sizes)
        errors = []

        def worker(rank: int) -> None:
            try:
                clock = Clock()
                observer = StageTimeObserver(
                    "test", early_stop=early_stop,
                    collectives=group.rank_transport(rank), clock=clock,
                )
                processed = []
                for item in observer.iterate(range(sizes[rank]), total_items=sizes[rank], item_count=lambda _: 1):
                    interval = 5
                    if uneven_rates and rank == 1 and item < 40:
                        interval = 25 if item % 2 else 5
                    clock.advance(interval)
                    processed.append(item)
                results[rank] = (observer.report, processed)
            except BaseException as exc:
                errors.append(exc)
                group.sync.abort()

        threads = [threading.Thread(target=worker, args=(rank,), daemon=True) for rank in range(len(sizes))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(7)
        self.assertFalse(any(t.is_alive() for t in threads), "Collective protocol deadlocked")
        self.assertEqual(errors, [])
        self.assertTrue(all(result[0] == results[0][0] for result in results))
        return results

    def test_unequal_ranks_empty_shard_and_early_completion_do_not_deadlock(self) -> None:
        results = self.run_threaded([100, 3, 0], early_stop=True)
        self.assertEqual([len(result[1]) for result in results], [25, 3, 0])
        self.assertEqual(results[0][0]["status"], "estimated")
        self.assertEqual(results[0][0]["estimated_total_seconds"], 500)

    def test_ranks_can_stabilize_on_different_iterations(self) -> None:
        results = self.run_threaded([100, 100], early_stop=True, uneven_rates=True)
        self.assertEqual(len(results[0][1]), 25)
        self.assertGreater(len(results[1][1]), 40)
        self.assertLess(len(results[1][1]), 100)
        report = results[0][0]
        self.assertEqual(report["estimated_total_seconds"], max(r["estimated_total_seconds"] for r in report["rank_reports"]))

    def test_full_retrieval_waits_for_all_unequal_shards(self) -> None:
        results = self.run_threaded([40, 5, 0], early_stop=False)
        self.assertEqual([len(result[1]) for result in results], [40, 5, 0])
        self.assertEqual(results[0][0]["status"], "completed")
        self.assertEqual(results[0][0]["elapsed_seconds"], 200)

    def test_torch_adapter_does_not_require_an_initialized_group(self) -> None:
        import torch.distributed as dist

        with patch.object(dist, "is_initialized", return_value=False), \
                patch.object(dist, "get_rank", side_effect=AssertionError("must not call")):
            transport = TorchStageCollectives()
            transport.barrier()
            self.assertEqual(transport.rank, 0)
            self.assertTrue(transport.all_done(True))
            self.assertFalse(transport.all_done(False))
            self.assertEqual(transport.gather({"test": 1}), [{"test": 1}])


class TimingReportAndIntegrationTests(unittest.TestCase):
    def test_writer_saves_incremental_seconds_without_touching_normal_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            (root / "best.xlsx").write_bytes(b"original")
            writer = RerankTimeEstimateWriter(temporary, {"world_size": 2}, rank=0)
            writer.save()
            self.assertEqual(json.loads(writer.path.read_text())["status"], "in_progress")
            writer.record("retrieval", {"status": "completed", "elapsed_seconds": 80, "estimated_total_seconds": 80})
            partial = json.loads(writer.path.read_text())
            self.assertIsNone(partial["estimated_total_seconds"])
            writer.record("rerank", {"status": "estimated", "elapsed_seconds": 150, "estimated_total_seconds": 600})
            report = json.loads(writer.path.read_text())
            self.assertEqual(report["time_unit"], "seconds")
            self.assertEqual(report["status"], "estimated")
            self.assertEqual(report["estimated_total_seconds"], 680)
            self.assertEqual((root / "best.xlsx").read_bytes(), b"original")
            self.assertFalse(writer.path.with_suffix(".json.tmp").exists())

    def test_nonzero_rank_does_not_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            writer = RerankTimeEstimateWriter(temporary, {}, rank=1)
            writer.save()
            self.assertFalse(writer.path.exists())

    def test_flag_defaults_and_hf_argument_parser(self) -> None:
        from transformers import HfArgumentParser

        parser = HfArgumentParser((RerankTimeEstimationArguments,))
        self.assertFalse(parser.parse_args_into_dataclasses([])[0].estimate_time_only)
        args = parser.parse_args_into_dataclasses(["--estimate_time_only"])[0]
        self.assertTrue(args.estimate_time_only)
        args.validate(SimpleNamespace(eol_type="disassembleeol_separate", calculate_type="concat"))
        with self.assertRaisesRegex(ValueError, "requires"):
            args.validate(SimpleNamespace(eol_type="prompteol", calculate_type="concat"))
        with self.assertRaisesRegex(ValueError, "placeholder"):
            args.validate(
                SimpleNamespace(eol_type="disassembleeol_separate", calculate_type="concat"),
                SimpleNamespace(rerank_type="listwise", rerank_template="relevant"),
            )

    def test_only_formal_retrieval_loops_are_observed(self) -> None:
        tree = ast.parse((PROJECT_ROOT / "src" / "rerank.py").read_text(encoding="utf-8"))
        formal = [node for node in ast.walk(tree) if isinstance(node, ast.For) and "test_dataloader" in ast.unparse(node.iter)]
        validation = [node for node in ast.walk(tree) if isinstance(node, ast.For) and "val_dataloader" in ast.unparse(node.iter)]
        self.assertEqual(len(formal), 5)
        self.assertTrue(all(node.iter.func.id == "test_progress" for node in formal))
        self.assertEqual(len(validation), 1)
        self.assertEqual(validation[0].iter.func.id, "tqdm")

    def test_ranker_apis_keep_optional_observer_and_all_query_loops_forward_it(self) -> None:
        import inspect
        import reranker

        for method in (reranker.Reranker.rerank, reranker.Reranker.caption_generation_rerank):
            self.assertIsNone(inspect.signature(method).parameters["progress_observer"].default)
        tree = ast.parse((PROJECT_ROOT / "src" / "reranker.py").read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_stage3_query_progress"]
        # All task-specific paths now share one batched query loop.
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(len(call.args) == 2 and call.args[1].id == "progress_observer" for call in calls))

    def test_main_destroys_only_owned_group_including_on_exception(self) -> None:
        tree = ast.parse((PROJECT_ROOT / "src" / "rerank.py").read_text(encoding="utf-8"))
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        for existing in (False, True):
            with self.subTest(existing=existing):
                states = [existing]
                destroyed = []
                dist = SimpleNamespace(
                    is_available=lambda: True, is_initialized=lambda: states[0],
                    destroy_process_group=lambda: destroyed.append(True),
                )

                def run() -> None:
                    states[0] = True
                    raise RuntimeError("actual failure")

                namespace = {"dist": dist, "_run_rerank": run}
                exec(compile(ast.Module(body=[node], type_ignores=[]), "main-test", "exec"), namespace)
                with self.assertRaisesRegex(RuntimeError, "actual failure"):
                    namespace["main"]()
                self.assertEqual(len(destroyed), 0 if existing else 1)


if __name__ == "__main__":
    unittest.main()
