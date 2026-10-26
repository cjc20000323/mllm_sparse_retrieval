"""CPU/Gloo process test for the same control protocol used by GPU/NCCL."""

from __future__ import annotations

import json
import multiprocessing
import sys
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path

import torch.distributed as dist


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from time_estimation import StageTimeObserver, TorchStageCollectives


class ProcessClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def gloo_timing_worker(rank: int, root: str) -> None:
    root_path = Path(root)
    result = {}
    try:
        dist.init_process_group(
            "gloo", init_method=(root_path / "store").as_uri(),
            rank=rank, world_size=3, timeout=timedelta(seconds=15),
        )
        collective = TorchStageCollectives("cpu")
        clock = ProcessClock()
        retrieval = StageTimeObserver("retrieval", early_stop=False, collectives=collective, clock=clock)
        for _ in retrieval.iterate(range([80, 5, 0][rank]), total_items=[80, 5, 0][rank], item_count=lambda _: 1):
            clock.now += 5
        rerank = StageTimeObserver("rerank", early_stop=True, collectives=collective, clock=clock)
        processed = 0
        for item in rerank.iterate(range([100, 100, 0][rank]), total_items=[100, 100, 0][rank], item_count=lambda _: 1):
            clock.now += 25 if rank == 1 and item < 40 and item % 2 else 5
            processed += 1
        result = {"retrieval": retrieval.report, "rerank": rerank.report, "processed": processed}
    except BaseException as exc:
        result = {"error": repr(exc)}
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        (root_path / f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")


class GlooTimeEstimationTests(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo is unavailable")
    def test_three_real_processes_with_uneven_and_empty_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = multiprocessing.get_context("spawn")
            processes = [context.Process(target=gloo_timing_worker, args=(rank, temporary)) for rank in range(3)]
            try:
                for process in processes:
                    process.start()
                deadline = time.monotonic() + 55
                for process in processes:
                    process.join(max(0, deadline - time.monotonic()))
                self.assertFalse(any(p.is_alive() for p in processes), "Distributed timing protocol hung")
                self.assertEqual([p.exitcode for p in processes], [0, 0, 0])
                reports = [json.loads((Path(temporary) / f"rank-{rank}.json").read_text()) for rank in range(3)]
                errors = [r["error"] for r in reports if "error" in r]
                unavailable = ("10013", "Permission denied", "unsupported gloo device")
                if errors and all(any(marker in error for marker in unavailable) for error in errors):
                    self.skipTest("Local Gloo transport is unavailable: " + errors[0])
                self.assertEqual(errors, [])
                self.assertEqual(reports[0]["processed"], 25)
                self.assertGreater(reports[1]["processed"], 40)
                self.assertLess(reports[1]["processed"], 100)
                self.assertEqual(reports[2]["processed"], 0)
                for report in reports:
                    self.assertEqual(report["retrieval"]["elapsed_seconds"], 400)
                    self.assertEqual(report["rerank"], reports[0]["rerank"])
                    self.assertEqual(report["rerank"]["status"], "estimated")
                    self.assertEqual(report["rerank"]["estimated_total_seconds"], 900)
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    if process.pid is not None:
                        process.join(5)


if __name__ == "__main__":
    unittest.main()
