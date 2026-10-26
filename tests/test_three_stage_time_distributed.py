"""Real CPU/Gloo processes exercising the new orchestration and empty ranks."""

from datetime import timedelta
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch.distributed as dist

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import estimate_three_stage_retrieval_time as estimate
from time_estimation import StageTimeObserver, TorchStageCollectives


class ProcessCollectives:
    """Non-network process transport for protocol tests when Windows Gloo is absent."""

    world_size = 3

    def __init__(self, rank, transport):
        self.rank = rank
        self.gate, self.flags, self.incoming, self.outgoing = transport

    def barrier(self):
        self.gate.wait(timeout=15)

    def all_done(self, done):
        self.flags[self.rank] = int(done)
        self.barrier()
        result = all(self.flags)
        self.barrier()
        return result

    def gather(self, report):
        self.incoming.put((self.rank, report))
        self.barrier()
        if self.rank == 0:
            rows = [row for _, row in sorted(self.incoming.get(timeout=15) for _ in range(3))]
            for queue in self.outgoing:
                queue.put(rows)
        rows = self.outgoing[self.rank].get(timeout=15)
        self.barrier()
        return rows


def timing_worker(rank, directory, transport=None):
    root = Path(directory)
    result = {}
    try:
        if transport is None:
            dist.init_process_group("gloo", init_method=(root / "store").as_uri(), rank=rank, world_size=3,
                                    timeout=timedelta(seconds=15))
        args = estimate.parser().parse_args([
            "--dataset-name", "webqa", "--task-type", "t2it", "--query-type", "text",
            "--stage1-sparse-index", "sparse", "--stage2-sparse-index", "sparse", "--stage2-dense-index", "dense",
            "--retrieval-model-name-or-path", "model", "--ranker-model-name-or-path", "model",
            "--output-dir", str(root / "estimate"), "--overwrite",
        ])
        pipeline = estimate.pipeline
        context = pipeline.DistributedContext(rank, rank, 3)
        class Clock:
            now = 0.0

            def __call__(self):
                return self.now

        clock = Clock()
        counts = [100, 60, 0]

        def stage1(args, context, timer):
            timer.components["query_encoding"] = [10.0, 7.0, 0.0][rank]
            run = {f"{rank}-{i}": {"candidate": 1.0} for i in range(counts[rank])}
            return run, object(), {}, counts

        def stage2(args, run, runtime, *, timing, encoded_queries):
            timing.components["search"] = [3.0, 2.0, 0.0][rank]
            return {}, {}, run

        processed = []
        def stage3(args, run, context, *, timing, progress_observer):
            for index, item in enumerate(progress_observer.iterate(run.items(), total_items=len(run), item_count=lambda _: 1)):
                clock.now += 25 if rank == 1 and index < 40 and index % 2 else 5
                processed.append(item[0])
            return {}

        with patch.object(pipeline, "_run_sparse_concat_stage1", new=stage1), \
             patch.object(pipeline, "run_stage2", new=stage2), patch.object(pipeline, "run_stage3", new=stage3), \
             patch.object(pipeline, "_synchronize_cuda_devices"):
            collective = TorchStageCollectives("cpu") if transport is None else ProcessCollectives(rank, transport)
            payload = estimate.run_estimation(args, context=context, collectives=collective, clock=clock,
                observer_factory=lambda *a, **kw: StageTimeObserver(*a, clock=clock, **kw))
        result = {"processed": processed, "total": payload["estimated_total_seconds"], "status": payload["status"]}
    except BaseException as exc:
        result = {"error": repr(exc)}
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        (root / f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")


class DistributedEstimationTests(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo is unavailable")
    def test_uneven_and_empty_workers_stop_at_different_times_without_deadlock(self):
        self.exercise_workers(gloo=True)

    def test_three_spawned_processes_with_non_network_transport(self):
        self.exercise_workers(gloo=False)

    def exercise_workers(self, *, gloo):
        with tempfile.TemporaryDirectory() as directory:
            context = multiprocessing.get_context("spawn")
            queues = []
            transport = None
            if not gloo:
                queues = [context.Queue() for _ in range(4)]
                transport = (context.Barrier(3), context.Array("i", 3), queues[0], queues[1:])
            processes = [context.Process(target=timing_worker, args=(rank, directory, transport)) for rank in range(3)]
            try:
                for process in processes:
                    process.start()
                deadline = time.monotonic() + 55
                for process in processes:
                    process.join(max(0, deadline - time.monotonic()))
                self.assertFalse(any(p.is_alive() for p in processes), "Distributed estimation hung")
                self.assertEqual([p.exitcode for p in processes], [0, 0, 0])
                rows = [json.loads((Path(directory) / f"rank-{rank}.json").read_text()) for rank in range(3)]
                errors = [row["error"] for row in rows if "error" in row]
                if errors and all(any(marker in error for marker in ("10013", "Permission denied", "unsupported gloo device")) for error in errors):
                    self.skipTest("Local Gloo transport is unavailable: " + errors[0])
                self.assertEqual(errors, [])
                self.assertEqual(len(rows[0]["processed"]), 25)
                self.assertGreater(len(rows[1]["processed"]), 40)
                self.assertLess(len(rows[1]["processed"]), 60)
                self.assertEqual(rows[2]["processed"], [])
                all_ids = [qid for row in rows for qid in row["processed"]]
                self.assertEqual(len(set(all_ids)), len(all_ids))
                self.assertEqual([row["total"] for row in rows], [713.0] * 3)
                output = Path(directory) / "estimate"
                self.assertEqual([path.name for path in output.iterdir()], ["time_estimate.json"])
                saved = json.loads((output / "time_estimate.json").read_text())
                self.assertEqual(saved["distributed"]["per_rank_query_counts"], [100, 60, 0])
                self.assertEqual(saved["status"], "estimated")
                self.assertEqual(len(saved["stages"]["stage3"]["rank_reports"]), 3)
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    if process.pid is not None:
                        process.join(5)
                for queue in queues:
                    queue.close()
                    queue.join_thread()


if __name__ == "__main__":
    unittest.main()
