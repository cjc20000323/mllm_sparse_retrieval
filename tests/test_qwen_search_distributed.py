"""Two real Gloo workers must evaluate alpha only once, globally, on rank zero."""
from __future__ import annotations

import csv
import io
import json
import multiprocessing
import tempfile
import time
import unittest
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch.distributed as dist

import test_qwen25_retrieval as shared
from test_qwen_search_alpha_selection import Dataset

pipeline = shared.pipeline


def search_worker(rank, root, rerank=False, fault=None, pipe=None):
    from hybrid import fuse
    root = Path(root)
    counts = dict(fuse=0, recall=0, csv=0, queries=[], reranked=[], alpha=None)
    transport = ExitStack()
    try:
        if pipe is None:
            dist.init_process_group("gloo", init_method=(root / "store").as_uri(), rank=rank,
                                    world_size=2, timeout=timedelta(seconds=25))
        else:
            # The same production control flow can also be checked on Windows
            # builds without a usable Gloo transport, using two real processes.
            def all_gather(parts, obj):
                if rank == 0:
                    parts[:] = [obj, pipe.recv()]
                    pipe.send(parts)
                else:
                    pipe.send(obj)
                    parts[:] = pipe.recv()
            def broadcast(packet, src):
                if rank == src:
                    pipe.send(packet[0])
                else:
                    packet[0] = pipe.recv()
            def scatter(received, parts, src):
                if rank == src:
                    received[0] = parts[0]
                    pipe.send(parts[1])
                else:
                    received[0] = pipe.recv()
            transport.enter_context(patch.object(dist, "is_initialized", return_value=True))
            transport.enter_context(patch.object(dist, "get_world_size", return_value=2))
            transport.enter_context(patch.object(dist, "all_gather_object", side_effect=all_gather))
            transport.enter_context(patch.object(dist, "broadcast_object_list", side_effect=broadcast))
            transport.enter_context(patch.object(dist, "scatter_object_list", side_effect=scatter))
        data = shared.Data()
        records = [dict(id=f"q{i}", native_id=f"q{i}") for i in range(3)]
        model = SimpleNamespace(config=SimpleNamespace(model_type="qwen2_5_vl"))
        training = SimpleNamespace(task_type="ir", output_dir=str(root / "output"))
        search = SimpleNamespace(query_type="image", use_fixed_alpha=False, alpha=0.9, depth=2,
                                 passage_reps="dense", sparse_index="sparse", val_passage_reps=None,
                                 val_sparse_index=None, rerank_type="pointwise", rerank_template="relevant",
                                 modify_type="no", use_rerank_prior_cache=False)

        class Index:
            def __init__(self, *args):
                pass
            def close(self):
                pass
            def retrieve(self, encoder, batch, data):
                counts["queries"].extend(record["id"] for record in batch)
                dense, sparse = {}, {}
                for record in batch:
                    correct, wrong = "good-" + record["id"], "wrong-" + record["id"]
                    dense[record["id"]] = pipeline._run({wrong: 0.0, correct: 1.0})
                    sparse[record["id"]] = pipeline._run({wrong: 1.0, correct: 0.0})
                return dense, sparse

        original_recall, original_csv, original_status = pipeline.recall, pipeline._write_metrics_csv, pipeline._search_status
        def counted_fuse(*args, **kwargs):
            counts["fuse"] += 1
            return fuse(*args, **kwargs)
        def counted_recall(*args, **kwargs):
            counts["recall"] += 1
            if fault == "selection" and rank == 0 and counts["recall"] == 3:
                raise ValueError("injected root selection failure")
            return original_recall(*args, **kwargs)
        def counted_csv(path, rows):
            counts["csv"] += 1
            if fault == "output" and Path(path).name == "best.csv":
                raise PermissionError("injected root output failure")
            return original_csv(path, rows)
        def counted_status(status):
            result = original_status(status)
            if "alpha" in result:
                counts["alpha"] = result["alpha"]
            return result
        def local_rerank(*args, already_sharded=False):
            if not already_sharded:
                raise AssertionError("Cached candidates must not be sharded twice")
            local = args[4]
            expected = {record["id"] for record in records[rank::2]}
            if set(local) != expected:
                raise AssertionError(f"Incorrect candidate shard: {set(local)} != {expected}")
            counts["reranked"] = sorted(local)
            return local

        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            stack.enter_context(patch.object(pipeline, "load_runtime", return_value=(model, None, rank, 2)))
            stack.enter_context(patch.object(pipeline, "build_dataset", return_value=Dataset()))
            stack.enter_context(patch.object(pipeline, "dataset_records", return_value=records))
            stack.enter_context(patch.object(pipeline, "prompt_metadata", return_value=dict(
                fingerprint="fixture", kind="text", config=dict(version=pipeline.PROMPT_VERSION))))
            stack.enter_context(patch.object(pipeline, "NativeQwenEncoder", return_value=SimpleNamespace(last_prompts=[])))
            stack.enter_context(patch.object(pipeline, "CorpusSearch", Index))
            stack.enter_context(patch("hybrid.fuse", side_effect=counted_fuse))
            stack.enter_context(patch.object(pipeline, "recall", side_effect=counted_recall))
            stack.enter_context(patch.object(pipeline, "_write_metrics_csv", side_effect=counted_csv))
            stack.enter_context(patch.object(pipeline, "_search_status", side_effect=counted_status))
            stack.enter_context(patch.object(pipeline, "rerank_records", side_effect=local_rerank))
            stack.enter_context(patch("qwen_vl_prompts.rerank_template", return_value="template"))
            pipeline.run_search(SimpleNamespace(), data, search, training, rerank=rerank)
    except BaseException as exc:
        counts["error"] = repr(exc)
    finally:
        transport.close()
        if dist.is_initialized():
            dist.destroy_process_group()
        if pipe is not None:
            pipe.close()
        (root / f"rank-{rank}.json").write_text(json.dumps(counts), encoding="utf-8")


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo is unavailable")
class QwenSearchDistributedTests(unittest.TestCase):
    def run_workers(self, *, rerank=False, fault=None, transport="gloo"):
        temporary = tempfile.TemporaryDirectory(prefix="qwen alpha gloo ")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        context = multiprocessing.get_context("spawn")
        pipes = context.Pipe() if transport == "pipe" else (None, None)
        processes = [context.Process(target=search_worker, args=(rank, str(root), rerank, fault, pipes[rank])) for rank in range(2)]
        try:
            for process in processes:
                process.start()
            for pipe in pipes:
                if pipe is not None:
                    pipe.close()
            deadline = time.monotonic() + 55
            for process in processes:
                process.join(max(0, deadline - time.monotonic()))
            self.assertFalse(any(process.is_alive() for process in processes), "Qwen search collectives hung")
            self.assertEqual([process.exitcode for process in processes], [0, 0])
            reports = [json.loads((root / f"rank-{rank}.json").read_text()) for rank in range(2)]
            errors = [report["error"] for report in reports if "error" in report]
            unavailable = ("10013", "Permission denied", "unsupported gloo device")
            if errors and all(any(marker in error for marker in unavailable) for error in errors):
                self.skipTest("Local Gloo transport is unavailable: " + errors[0])
            return root, reports
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                if process.pid is not None:
                    process.join(5)
            for pipe in pipes:
                if pipe is not None:
                    pipe.close()

    def test_only_root_scans_nine_alphas_and_writes_cached_best(self):
        root, reports = self.run_workers()
        self.assertFalse([report.get("error") for report in reports if "error" in report])
        self.assertEqual([report["fuse"] for report in reports], [9, 0])
        self.assertEqual([report["recall"] for report in reports], [11, 0])
        self.assertEqual([report["csv"] for report in reports], [2, 0])
        self.assertEqual([report["alpha"] for report in reports], [0.6, 0.6])
        self.assertEqual(sorted(key for report in reports for key in report["queries"]), ["q0", "q1", "q2"])
        with (root / "output/alpha_sweep.csv").open(encoding="utf-8-sig", newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 27)
        with (root / "output/best.csv").open(encoding="utf-8-sig", newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 3)

    def test_rerank_receives_disjoint_cached_candidates_without_extra_fusion(self):
        root, reports = self.run_workers(rerank=True)
        self.assertFalse([report.get("error") for report in reports if "error" in report])
        self.assertEqual([report["fuse"] for report in reports], [9, 0])
        self.assertEqual([report["recall"] for report in reports], [12, 0])
        self.assertEqual(sorted(key for report in reports for key in report["reranked"]), ["q0", "q1", "q2"])
        report = json.loads((root / "output/recall_metrics.json").read_text())
        self.assertEqual(report["metrics"]["rerank"]["Recall@1"], 100)

    def test_selection_error_reaches_both_ranks_without_hanging(self):
        _, reports = self.run_workers(fault="selection")
        self.assertTrue(all("injected root selection failure" in report.get("error", "") for report in reports))
        self.assertEqual([report["fuse"] for report in reports], [1, 0])

    def test_output_error_reaches_both_ranks_without_hanging(self):
        _, reports = self.run_workers(fault="output")
        self.assertTrue(all("injected root output failure" in report.get("error", "") for report in reports))
        self.assertEqual([report["fuse"] for report in reports], [9, 0])


class QwenSearchPipeTests(QwenSearchDistributedTests):
    __unittest_skip__ = False

    def run_workers(self, *, rerank=False, fault=None):
        return super().run_workers(rerank=rerank, fault=fault, transport="pipe")


if __name__ == "__main__":
    unittest.main()
