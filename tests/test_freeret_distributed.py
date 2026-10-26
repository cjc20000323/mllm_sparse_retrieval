from __future__ import annotations

import copy
import io
import json
import multiprocessing
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from PIL import Image

import test_freeret_time_estimation as fixtures


freeret = fixtures.freeret


class RankClock:
    def __init__(self) -> None:
        self.local = threading.local()

    def __call__(self) -> float:
        return getattr(self.local, "now", 0.0)

    def advance(self, seconds: float) -> None:
        self.local.now = self() + seconds


class ThreadGroup:
    def __init__(self, size: int) -> None:
        self.size = size
        self.sync = threading.Barrier(size, timeout=12)
        self.values: list[Any] = [None] * size

    def exchange(self, rank: int, value: Any, operation: str) -> Any:
        self.values[rank] = pickle.loads(pickle.dumps(value))
        self.sync.wait()
        if operation == "broadcast":
            result = self.values[0]
        elif operation == "all_done":
            result = all(self.values)
        else:
            result = list(self.values)
        result = pickle.loads(pickle.dumps(result))
        self.sync.wait()
        return result


class ThreadCollectives:
    device = torch.device("cpu")

    def __init__(self, group: ThreadGroup, rank: int) -> None:
        self.group = group
        self.rank = self.local_rank = rank
        self.world_size = group.size

    def broadcast(self, value: Any) -> Any:
        return self.group.exchange(self.rank, value, "broadcast")

    def gather(self, value: Any) -> Any:
        return self.group.exchange(self.rank, value, "gather")

    def all_done(self, value: bool) -> bool:
        return self.group.exchange(self.rank, value, "all_done")

    def barrier(self) -> None:
        self.group.sync.wait()


class IndexedModel(fixtures.FakeModel):
    def __init__(self, clock: RankClock, rank: int = 0, *, fail: bool = False) -> None:
        super().__init__(clock)
        self.rank = rank
        self.fail = fail
        self.target_ids: list[int] = []

    def encode_one_word(self, texts: list[str], paths: Any) -> torch.Tensor:
        if self.fail:
            raise RuntimeError("worker model failure")
        self.encode_batch_sizes.append(len(texts))
        self.clock.advance(2 + self.rank)
        values = []
        for text in texts:
            target = re.search(r"candidate (\d+)", text)
            number = int(target.group(1)) if target else 0
            if target:
                self.target_ids.append(number)
            values.append([float(number + 1), 1.0])
        return torch.tensor(values)

    def rerank(self, texts: list[str], paths: Any) -> torch.Tensor:
        self.rerank_batch_sizes.append(len(texts))
        self.clock.advance(1 + self.rank)
        return torch.tensor([
            float(re.search(r"candidate (\d+)", text).group(1))
            if re.search(r"candidate (\d+)", text) else 0.0 for text in texts
        ])


class FreeRetDistributedTests(unittest.TestCase):
    make_task = fixtures.FreeRetTimeEstimationIntegrationTests.make_task
    make_args = fixtures.FreeRetTimeEstimationIntegrationTests.make_args

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.image = self.root / "tiny.png"
        Image.new("RGB", (2, 2)).save(self.image)
        self.clock = RankClock()
        self.output = io.StringIO()
        for manager in (
            redirect_stdout(self.output), redirect_stderr(self.output),
            patch.object(freeret.time, "perf_counter", self.clock),
            patch.object(freeret, "tqdm", fixtures.FakeProgress),
        ):
            manager.__enter__()
            self.addCleanup(manager.__exit__, None, None, None)
        self.models: dict[int, IndexedModel] = {}
        self.loads: list[tuple[str, str]] = []

    def args(self, *, estimate: bool = False) -> Any:
        args = self.make_args(enabled=estimate)
        args.datasets = "flickr"
        args.direction = "i2t"
        args.target_batch_size = 2
        args.data_root = self.root
        args.output_dir = self.root / "output"
        return args

    def execute(
        self, args: Any, *, size: int = 2, queries: int = 7, candidates: int = 5,
        fail_rank: int | None = None, load_failure: bool = False, filtered: bool = False,
    ) -> dict[int, BaseException]:
        group = ThreadGroup(size)
        errors: dict[int, BaseException] = {}
        def factory(**kwargs: Any) -> IndexedModel:
            rank = self.clock.local.rank
            model = IndexedModel(self.clock, rank, fail=rank == fail_rank)
            self.models[rank] = model
            return model
        def load(root: Any, dataset: str, split: str, direction: str, materializer: Any) -> Any:
            self.assertEqual(self.clock.local.rank, 0)
            if load_failure:
                raise FileNotFoundError("missing test data")
            self.loads.append((dataset, direction))
            task = self.make_task(queries=queries, candidates=candidates, dataset=dataset, direction=direction)
            if filtered:
                task.queries[0].relevant_ids.clear()
            return task
        def worker(rank: int) -> None:
            self.clock.local.rank = rank
            try:
                context = freeret.FreeRetDistributed(ThreadCollectives(group, rank))
                freeret._distributed_main(copy.copy(args), context)
            except BaseException as exc:
                errors[rank] = exc
                group.sync.abort()  # Mimic torchrun terminating peers on a failed rank.
        with patch.object(freeret, "FreeRetQwen", side_effect=factory), \
                patch.object(freeret, "load_task", side_effect=load):
            threads = [threading.Thread(target=worker, args=(rank,), daemon=True) for rank in range(size)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(20)
            group.sync.abort()
            for thread in threads:
                thread.join(2)
            self.assertFalse(any(thread.is_alive() for thread in threads), "distributed protocol hung")
        return errors

    def run_path(self, args: Any, dataset: str = "flickr", direction: str = "i2t") -> Path:
        return Path(args.output_dir) / "fake-model" / dataset / f"test_{direction}"

    def test_uneven_shards_match_serial_results_and_full_candidate_pool(self) -> None:
        args = self.args()
        task = self.make_task(queries=7, candidates=5, direction="i2t")
        serial = self.root / "serial"
        expected = freeret.run_task(IndexedModel(self.clock), task, args, serial)
        self.assertEqual(self.execute(args), {})
        actual = self.run_path(args)
        self.assertEqual(json.loads((actual / "metrics.json").read_text()), expected)
        self.assertEqual((actual / "results.jsonl").read_text(), (serial / "results.jsonl").read_text())
        self.assertEqual(sorted(index for model in self.models.values() for index in model.target_ids), list(range(5)))
        rows = [json.loads(line) for line in (actual / "results.jsonl").read_text().splitlines()]
        self.assertEqual([row["query_id"] for row in rows], [f"q-{index}" for index in range(7)])
        self.assertTrue(all(row["stage1"][0]["candidate_id"] == "c-4" for row in rows))

    def test_timing_uses_max_not_sum_and_preserves_global_counts(self) -> None:
        args = self.args()
        self.assertEqual(self.execute(args), {})
        report = json.loads((self.run_path(args) / "timing_metrics.json").read_text())
        ranks = report["rank_reports"]
        for component in freeret.ACTUAL_TIMING_COMPONENTS:
            self.assertEqual(report[component], max(rank[component] for rank in ranks))
        self.assertLess(report["candidate_encoding_seconds"], sum(rank["candidate_encoding_seconds"] for rank in ranks))
        self.assertEqual([rank["processed_queries"] for rank in ranks], [4, 3])
        self.assertEqual(report["processed_queries"], 7)
        self.assertEqual(report["num_queries"], 7)
        self.assertEqual(report["world_size"], 2)
        self.assertEqual(report["total_compute_seconds"], report["retrieval_seconds"] + report["reranking_seconds"])
        summary = json.loads((Path(args.output_dir) / "fake-model" / "timing_summary.json").read_text())
        self.assertEqual(summary["completed_tasks"], 1)
        self.assertEqual(summary["world_size"], 2)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(list((Path(args.output_dir) / "fake-model" / ".distributed").iterdir()), [])

    def test_empty_query_and_candidate_shards_complete_without_padding(self) -> None:
        args = self.args()
        self.assertEqual(self.execute(args, size=3, queries=1, candidates=1), {})
        report = json.loads((self.run_path(args) / "timing_metrics.json").read_text())
        self.assertEqual([r["processed_queries"] for r in report["rank_reports"]], [1, 0, 0])
        self.assertEqual(sum(len(m.target_ids) for m in self.models.values()), 1)
        self.assertEqual(len((self.run_path(args) / "results.jsonl").read_text().splitlines()), 1)

    def test_single_worker_uses_distributed_results_cache_and_timing(self) -> None:
        args = self.args()
        args.cache_targets = True
        self.assertEqual(self.execute(args, size=1), {})
        path = self.run_path(args)
        report = json.loads((path / "timing_metrics.json").read_text())
        self.assertEqual(report["world_size"], 1)
        self.assertEqual(report["processed_queries"], 7)
        self.assertEqual(len((path / "results.jsonl").read_text().splitlines()), 7)
        self.assertEqual(self.execute(args, size=1), {})
        self.assertEqual(self.models[0].target_ids, [])

    def test_single_worker_eta_still_estimates_and_preserves_global_counts(self) -> None:
        args = self.args(estimate=True)
        self.assertEqual(self.execute(args, size=1, queries=400, candidates=121), {})
        report = json.loads((self.run_path(args) / "time_estimate.json").read_text())
        self.assertEqual(report["num_queries"], 400)
        self.assertLess(report["processed_queries"], 400)
        self.assertEqual(report["stages"]["query_retrieval_rerank"]["world_size"], 1)
        self.assertFalse((self.run_path(args) / "metrics.json").exists())

    def test_filter_and_limit_are_applied_before_sharding(self) -> None:
        args = self.args()
        args.max_queries = 3
        self.assertEqual(self.execute(args, queries=8, filtered=True), {})
        path = self.run_path(args)
        records = [json.loads(line) for line in (path / "results.jsonl").read_text().splitlines()]
        self.assertEqual([row["query_id"] for row in records], ["q-1", "q-2", "q-3"])
        self.assertEqual(json.loads((path / "metrics.json").read_text())["skipped_unjudged_queries"], 1)

    def test_cache_written_once_and_reused_without_candidate_encoding(self) -> None:
        args = self.args()
        args.cache_targets = True
        self.assertEqual(self.execute(args), {})
        payload = torch.load(self.run_path(args) / "target_embeddings_fake-model.pt", map_location="cpu")
        self.assertEqual(payload["embeddings"][:, 0].tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(self.execute(args), {})
        self.assertTrue(all(not model.target_ids for model in self.models.values()))
        report = json.loads((self.run_path(args) / "timing_metrics.json").read_text())
        self.assertTrue(report["target_cache_hit"])
        self.assertEqual(report["candidate_encoding_seconds"], 0)

    def test_multiple_datasets_and_directions_reuse_one_model_per_rank(self) -> None:
        args = self.args()
        args.datasets = "coco,flickr"
        args.direction = "both"
        self.assertEqual(self.execute(args, queries=1, candidates=2), {})
        self.assertEqual(len(self.models), 2)
        self.assertEqual(self.loads, [("coco", "t2i"), ("coco", "i2t"), ("flickr", "t2i"), ("flickr", "i2t")])
        report = json.loads((Path(args.output_dir) / "fake-model" / "timing_summary.json").read_text())
        self.assertEqual(report["completed_tasks"], 4)
        self.assertEqual(len(report["model_loading_rank_reports"]), 2)
        self.assertEqual(len(json.loads((Path(args.output_dir) / "fake-model" / "summary.json").read_text())), 4)

    def test_eta_stabilizes_each_shard_and_aggregates_max_duration(self) -> None:
        args = self.args(estimate=True)
        self.assertEqual(self.execute(args, queries=400, candidates=121), {})
        path = self.run_path(args)
        report = json.loads((path / "time_estimate.json").read_text())
        self.assertEqual(report["status"], "estimated")
        self.assertEqual(report["num_queries"], 400)
        self.assertLess(report["processed_queries"], 400)
        self.assertEqual(sum(len(m.target_ids) for m in self.models.values()), 121)
        for stage in report["stages"].values():
            self.assertEqual(stage["estimated_total_seconds"], max(r["estimated_total_seconds"] for r in stage["rank_reports"]))
        self.assertEqual(report["estimated_total_seconds"], report["target_elapsed_seconds"] + report["query_estimated_total_seconds"])
        self.assertEqual(report["processed_queries"], sum(r["completed_items"] for r in report["stages"]["query_retrieval_rerank"]["rank_reports"]))
        for name in ("timing_metrics.json", "results.jsonl", "metrics.json", "config.json"):
            self.assertFalse((path / name).exists())
        self.assertFalse((Path(args.output_dir) / "fake-model" / "timing_summary.json").exists())

    def test_eta_empty_shards_and_cache_hit_complete(self) -> None:
        args = self.args(estimate=True)
        args.cache_targets = True
        self.assertEqual(self.execute(args, size=3, queries=1, candidates=1), {})
        self.assertEqual(self.execute(args, size=3, queries=1, candidates=1), {})
        report = json.loads((self.run_path(args) / "time_estimate.json").read_text())
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["processed_queries"], 1)
        self.assertTrue(report["target_cache_hit"])

    def test_worker_failure_keeps_diagnostics_and_no_completed_result(self) -> None:
        args = self.args()
        errors = self.execute(args, fail_rank=1)
        self.assertIn(1, errors)
        self.assertIn("worker model failure", str(errors[1]))
        root = Path(args.output_dir) / "fake-model"
        self.assertTrue(list((root / ".distributed").glob("run-*/failed-rank-1.json")))
        self.assertNotEqual(json.loads((root / "timing_summary.json").read_text())["status"], "completed")
        self.assertFalse((self.run_path(args) / "metrics.json").exists())

    def test_primary_load_failure_reaches_peers_without_hanging(self) -> None:
        errors = self.execute(self.args(), load_failure=True)
        self.assertEqual(len(errors), 2)
        self.assertIn("missing test data", str(errors[0]))

    def test_record_merge_rejects_duplicate_or_missing_indices(self) -> None:
        freeret._write_timing_json(self.root / "results-0.json", {"records": [{"index": 0, "record": {}}]})
        freeret._write_timing_json(self.root / "results-1.json", {"records": [{"index": 0, "record": {}}]})
        with self.assertRaisesRegex(RuntimeError, "index"):
            freeret._merge_indexed_records(self.root, 2, 2, partial=False)
        freeret._write_timing_json(self.root / "results-1.json", {"records": []})
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            freeret._merge_indexed_records(self.root, 2, 2, partial=False)
        self.assertEqual(freeret._merge_indexed_records(self.root, 2, 2, partial=True), [{}])

    def test_initialization_binds_local_gpu_even_with_gloo_controls(self) -> None:
        with patch.dict(os.environ, {"WORLD_SIZE": "2", "LOCAL_WORLD_SIZE": "2", "LOCAL_RANK": "1"}), \
                patch.object(torch.cuda, "device_count", return_value=2), \
                patch.object(torch.cuda, "set_device") as bind, \
                patch.object(torch.distributed, "init_process_group") as initialize, \
                patch.object(torch.distributed, "is_nccl_available", return_value=False), \
                patch.object(freeret, "FreeRetCollectives") as collective:
            collective.return_value.rank = 1
            collective.return_value.world_size = 2
            context = freeret.FreeRetDistributed.initialize()
        bind.assert_called_once_with(1)
        self.assertEqual(initialize.call_args.args[0], "gloo")
        self.assertEqual(str(context.model_device), "cuda:1")


class DistributedInitializationTests(unittest.TestCase):
    def initialize(self, environment: dict[str, str], *, cli_rank: int | None = None, gpus: int = 2) -> Any:
        with patch.dict(os.environ, environment, clear=True), \
                patch.object(torch.cuda, "device_count", return_value=gpus), \
                patch.object(torch.cuda, "set_device") as bind, \
                patch.object(torch.distributed, "init_process_group") as initialize, \
                patch.object(torch.distributed, "is_nccl_available", return_value=True), \
                patch.object(freeret, "FreeRetCollectives") as collective:
            self.bind = bind
            self.initialize_group = initialize
            collective.return_value.rank = int(environment.get("RANK", "0"))
            collective.return_value.world_size = int(environment["WORLD_SIZE"])
            context = freeret.FreeRetDistributed.initialize(cli_rank)
        bind.assert_called_once_with(context.local_rank)
        initialize.assert_called_once()
        self.assertEqual(initialize.call_args.args[0], "nccl")
        return context

    def test_deepspeed_uses_remapped_local_rank_not_physical_gpu_id(self) -> None:
        context = self.initialize({
            "WORLD_SIZE": "2", "LOCAL_SIZE": "2", "LOCAL_RANK": "1", "RANK": "1",
            "CUDA_VISIBLE_DEVICES": "2,3",
        }, cli_rank=1)
        self.assertEqual(str(context.model_device), "cuda:1")
        self.assertEqual(context.rank, 1)

    def test_deepspeed_single_worker_initializes_and_binds_gpu(self) -> None:
        context = self.initialize({
            "WORLD_SIZE": "1", "LOCAL_SIZE": "1", "LOCAL_RANK": "0", "RANK": "0",
            "CUDA_VISIBLE_DEVICES": "3",
        }, cli_rank=0, gpus=1)
        self.assertEqual(str(context.model_device), "cuda:0")
        self.assertEqual(context.world_size, 1)

    def test_torchrun_environment_remains_supported(self) -> None:
        context = self.initialize({"WORLD_SIZE": "2", "LOCAL_WORLD_SIZE": "2", "LOCAL_RANK": "1"})
        self.assertEqual(context.local_rank, 1)

    def test_cli_local_rank_is_used_when_environment_rank_is_absent(self) -> None:
        context = self.initialize({"WORLD_SIZE": "2", "LOCAL_SIZE": "2", "RANK": "1"}, cli_rank=1)
        self.assertEqual(str(context.model_device), "cuda:1")

    def test_invalid_local_rank_and_multi_node_fail_before_initialization(self) -> None:
        for overrides, cli_rank, message in [
            ({"LOCAL_RANK": "0"}, 1, "disagrees"),
            ({"LOCAL_RANK": "-1"}, None, "one visible GPU"),
            ({"LOCAL_RANK": "2"}, None, "one visible GPU"),
            ({"LOCAL_SIZE": "1"}, None, "single node"),
            ({"LOCAL_WORLD_SIZE": "1"}, None, "single node"),
            ({"WORLD_SIZE": "3"}, None, "one visible GPU"),
            ({"WORLD_SIZE": "0"}, None, "one visible GPU"),
        ]:
            environment = {"WORLD_SIZE": "2", "LOCAL_SIZE": "2", "LOCAL_RANK": "1", **overrides}
            with self.subTest(environment=environment, cli_rank=cli_rank), \
                    self.assertRaisesRegex(ValueError, message):
                self.initialize(environment, cli_rank=cli_rank)
            self.initialize_group.assert_not_called()
            self.bind.assert_not_called()

    def test_no_visible_gpu_rejected_before_binding_or_process_group_creation(self) -> None:
        with self.assertRaisesRegex(ValueError, "one visible GPU"):
            self.initialize({"WORLD_SIZE": "1", "LOCAL_RANK": "0", "LOCAL_SIZE": "1"}, gpus=0)
        self.initialize_group.assert_not_called()
        self.bind.assert_not_called()

    def test_missing_local_rank_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "LOCAL_RANK or --local_rank"):
            self.initialize({"WORLD_SIZE": "1", "LOCAL_SIZE": "1"}, gpus=1)

    def test_main_dispatches_single_and_multi_worker_launcher_environments(self) -> None:
        for size, cli_argument in [(1, []), (1, ["--local_rank=0"]), (2, ["--local-rank=0"])]:
            environment = {"WORLD_SIZE": str(size), "LOCAL_SIZE": str(size), "LOCAL_RANK": "0", "RANK": "0"}
            with self.subTest(size=size, cli_argument=cli_argument), \
                    patch.dict(os.environ, environment, clear=True), \
                    patch.object(sys, "argv", ["freeret_retrieval.py", *cli_argument]), \
                    patch.object(freeret, "_distributed_main") as distributed, \
                    patch.object(freeret, "FreeRetQwen") as model:
                freeret.main()
            distributed.assert_called_once()
            model.assert_not_called()

    def test_cli_rank_alone_selects_distributed_path(self) -> None:
        with patch.dict(os.environ, {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_SIZE": "1"}, clear=True), \
                patch.object(sys, "argv", ["freeret_retrieval.py", "--local_rank=0"]), \
                patch.object(freeret, "_distributed_main") as distributed:
            freeret.main()
        self.assertEqual(distributed.call_args.args[0].local_rank, 0)


class LauncherContractTests(unittest.TestCase):
    def test_shell_only_launches_deepspeed_in_selected_python_environment(self) -> None:
        script = (fixtures.PROJECT_ROOT / "scripts" / "freeret_retrieval.sh").read_text(encoding="utf-8")
        self.assertIn('env -u CUDA_VISIBLE_DEVICES "${PYTHON_BIN}" -m deepspeed.launcher.runner', script)
        self.assertIn('--hostfile "" --include "localhost:${SELECTED_GPU_LIST}"', script)
        self.assertIn('--master_addr 127.0.0.1 --master_port "${MASTER_PORT}"', script)
        self.assertIn('exec "${COMMAND[@]}"', script)
        self.assertNotIn("torch.distributed.run", script)
        self.assertNotIn("--nproc-per-node", script)
        self.assertNotIn('--num_gpus "', script)


@unittest.skipUnless(shutil.which("bash"), "Bash is unavailable locally")
class LauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="freeret launch ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.stub = self.root / "python stub"
        self.capture = self.root / "arguments.txt"
        self.capture_env = self.root / "environment.txt"
        self.stub.write_text(
            '#!/usr/bin/env bash\n'
            'if [[ "${1:-}" == "-c" ]]; then\n'
            '  case "$2" in\n'
            '    *torch.cuda.device_count*) echo "${STUB_GPU_COUNT}"; exit "${STUB_CUDA_ERROR:-0}" ;;\n'
            '    *deepspeed.launcher.runner*) exit "${STUB_DS_ERROR:-0}" ;;\n'
            '    *) exit 9 ;;\n'
            '  esac\n'
            'else\n'
            '  printf "%s\\n" "$@" > "${STUB_CAPTURE}"\n'
            '  printf "%s\\n" "${CUDA_VISIBLE_DEVICES-unset}" > "${STUB_CAPTURE_ENV}"\n'
            '  exit "${STUB_EXIT_CODE:-0}"\n'
            'fi\n', encoding="utf-8",
        )
        self.stub.chmod(0o755)

    def run_shell(self, *, count: str = "4", requested: str = "auto", ids: str | None = None,
                  **overrides: str) -> subprocess.CompletedProcess[str]:
        environment = dict(os.environ)
        for name in ("CUDA_VISIBLE_DEVICES", "MASTER_PORT", "ESTIMATE_TIME_ONLY", "CACHE_TARGETS"):
            environment.pop(name, None)
        environment.update(
            PYTHON_BIN=str(self.stub), STUB_GPU_COUNT=count, STUB_CAPTURE=str(self.capture),
            STUB_CAPTURE_ENV=str(self.capture_env), NUM_PROCESSES=requested, **overrides,
        )
        if ids is not None:
            environment["CUDA_VISIBLE_DEVICES"] = ids
        return subprocess.run(
            [shutil.which("bash"), str(fixtures.PROJECT_ROOT / "scripts" / "freeret_retrieval.sh")],
            env=environment, capture_output=True, text=True, timeout=15,
        )

    def test_auto_manual_and_single_all_use_deepspeed(self) -> None:
        for count, requested, ids, selected in [
            ("4", "auto", None, "0,1,2,3"), ("4", "2", None, "0,1"),
            ("4", "1", None, "0"), ("1", "auto", None, "0"),
            ("2", "auto", "2,3", "2,3"), ("2", "1", "3,1", "3"),
            ("2", "2", "3,1", "3,1"),
        ]:
            with self.subTest(count=count, requested=requested, ids=ids):
                result = self.run_shell(count=count, requested=requested, ids=ids)
                self.assertEqual(result.returncode, 0, result.stderr)
                arguments = self.capture.read_text().splitlines()
                self.assertEqual(arguments[:2], ["-m", "deepspeed.launcher.runner"])
                self.assertEqual(arguments[arguments.index("--include") + 1], f"localhost:{selected}")
                self.assertEqual(arguments[arguments.index("--hostfile") + 1], "")
                self.assertEqual(arguments[arguments.index("--master_port") + 1], "29500")
                self.assertEqual(self.capture_env.read_text().strip(), "unset")
                self.assertNotIn("--num_gpus", arguments)

    def test_port_and_retrieval_options_are_forwarded_unchanged(self) -> None:
        result = self.run_shell(
            ids="2,3", count="2", MASTER_PORT="29517", DATASETS="coco,flickr", DIRECTION="i2t",
            ESTIMATE_TIME_ONLY="true", CACHE_TARGETS="1", RERANK_BATCH_SIZE="4", METRICS_KS="1 5 10",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self.capture.read_text().splitlines()
        for option, value in [("--master_port", "29517"), ("--datasets", "coco,flickr"),
                              ("--direction", "i2t"), ("--rerank_batch_size", "4")]:
            self.assertEqual(arguments[arguments.index(option) + 1], value)
        index = arguments.index("--metrics_ks")
        self.assertEqual(arguments[index + 1:index + 4], ["1", "5", "10"])
        self.assertIn("--estimate_time_only", arguments)
        self.assertIn("--cache_targets", arguments)
        self.assertIn("physical GPU IDs=2,3", result.stdout)
        self.assertIn("DEVICE_MAP is ignored", result.stdout)

    def test_invalid_process_counts_ports_and_gpu_lists_fail_without_launch(self) -> None:
        cases = [
            ({"requested": value}, "NUM_PROCESSES") for value in ("0", "-1", "text", "5", "999999999999999999999")
        ] + [({"MASTER_PORT": value}, "MASTER_PORT") for value in ("0", "65536", "text", "-1")]
        cases += [({"ids": value}, "CUDA_VISIBLE_DEVICES") for value in (
            "", "1,1", "0,,1", "0,", "01,2", "GPU-123", "MIG-123", "2,3",
        )]
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                result = self.run_shell(**overrides)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(message, result.stderr)
                self.assertFalse(self.capture.exists())

    def test_no_gpu_and_failed_preflight_have_no_python_fallback(self) -> None:
        for overrides, message in [
            ({"count": "0"}, "at least one visible CUDA GPU"),
            ({"STUB_CUDA_ERROR": "1"}, "Cannot query GPUs"),
            ({"count": "not-a-number"}, "Invalid visible GPU count"),
            ({"STUB_DS_ERROR": "1"}, "DeepSpeed could not be imported"),
        ]:
            with self.subTest(overrides=overrides):
                result = self.run_shell(**overrides)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(message, result.stderr)
                self.assertFalse(self.capture.exists())

    def test_launcher_exit_status_is_propagated(self) -> None:
        result = self.run_shell(STUB_EXIT_CODE="7")
        self.assertEqual(result.returncode, 7)


def cpu_process_worker(rank: int, root_value: str) -> None:
    root = Path(root_value)
    status: dict[str, Any] = {}
    try:
        torch.distributed.init_process_group(
            "gloo", init_method=(root / "store").as_uri(), rank=rank,
            world_size=3, timeout=timedelta(seconds=15),
        )
        context = freeret.FreeRetDistributed(freeret.FreeRetCollectives("cpu"))
        clock = RankClock()
        for estimate in (False, True):
            args = freeret.build_parser().parse_args([
                "--datasets", "flickr", "--direction", "i2t",
                "--model_path", "fake-model", "--data_root", str(root),
                "--output_dir", str(root / ("eta" if estimate else "actual")),
                "--target_batch_size", "2", "--query_batch_size", "2",
                "--rerank_num", "2", "--rerank_batch_size", "1",
            ] + (["--estimate_time_only"] if estimate else []))
            def load(*_: Any) -> Any:
                return freeret.RetrievalTask(
                    dataset="flickr", split="test", direction="i2t", prompt_key="flickr_i2t",
                    queries=[freeret.QueryItem(
                        query_id=f"q-{index}", image_path=str(root / "tiny.png"), relevant_ids={"c-0"},
                    ) for index in range(1 if estimate else 7)],
                    candidates=[freeret.CandidateItem(candidate_id=f"c-{index}", text=f"candidate {index}")
                                for index in range(1 if estimate else 5)],
                )
            with patch.object(freeret.time, "perf_counter", clock), \
                    patch.object(freeret, "tqdm", fixtures.FakeProgress), \
                    patch.object(freeret, "load_task", side_effect=load), \
                    patch.object(freeret, "FreeRetQwen", side_effect=lambda **_: IndexedModel(clock, rank)), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                freeret._distributed_main(args, context)
        status = {"completed": True}
    except BaseException as exc:
        status = {"error": repr(exc)}
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        (root / f"status-{rank}.json").write_text(json.dumps(status), encoding="utf-8")


class RealProcessTests(unittest.TestCase):
    @unittest.skipUnless(torch.distributed.is_available() and torch.distributed.is_gloo_available(), "Gloo unavailable")
    def test_three_cpu_processes_normal_and_eta_with_empty_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (2, 2)).save(root / "tiny.png")
            spawn = multiprocessing.get_context("spawn")
            processes = [spawn.Process(target=cpu_process_worker, args=(rank, temporary)) for rank in range(3)]
            try:
                for process in processes:
                    process.start()
                deadline = time.monotonic() + 60
                for process in processes:
                    process.join(max(0, deadline - time.monotonic()))
                self.assertFalse(any(process.is_alive() for process in processes), "real distributed run hung")
                self.assertEqual([process.exitcode for process in processes], [0, 0, 0])
                statuses = [json.loads((root / f"status-{rank}.json").read_text()) for rank in range(3)]
                errors = [value["error"] for value in statuses if "error" in value]
                unavailable = ("10013", "Permission denied", "unsupported gloo device")
                if errors and all(any(marker in error for marker in unavailable) for error in errors):
                    self.skipTest("Local Gloo transport is unavailable: " + errors[0])
                self.assertEqual(errors, [])
                actual = root / "actual" / "fake-model" / "flickr" / "test_i2t"
                records = [json.loads(line) for line in (actual / "results.jsonl").read_text().splitlines()]
                self.assertEqual([record["query_id"] for record in records], [f"q-{index}" for index in range(7)])
                eta = json.loads((root / "eta" / "fake-model" / "flickr" / "test_i2t" / "time_estimate.json").read_text())
                self.assertEqual(eta["processed_queries"], 1)
                self.assertEqual(eta["status"], "completed")
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    if process.pid is not None:
                        process.join(5)


if __name__ == "__main__":
    unittest.main()
