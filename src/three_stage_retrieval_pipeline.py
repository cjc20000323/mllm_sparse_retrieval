"""A standalone three-stage retrieval and reranking pipeline.

The existing search and reranking programs are intentionally left untouched.
This file follows the loading flow used by ``search.py`` / ``rerank.py`` and
calls the existing :class:`reranker.Reranker` for the final stage.

Pipeline
--------
1. Stage 1 is a coarse candidate generator.  It can be one of:

   * ``clip`` or ``blip``: direct image/text feature retrieval;
   * ``sparse_manual``: online query encoding with
     ``prompteol + separate + sparse_manual`` against a persisted Lucene
     corpus index;
   * ``sparse_concat``: online query encoding with
     ``disassembleeol_separate + concat`` against a persisted Lucene corpus
     index. Queries are sharded across all workers; each worker retains its
     own CPU query representations for Stage 2.

   A precomputed TREC/JSON run can also be supplied for any stage.  This is
   useful when a baseline model is maintained by a separate script.

2. Stage 2 is always the current-program configuration
   ``disassembleeol_separate + concat``.  Just like ``search.py``, it encodes
   queries online (or reuses Stage-1 ``sparse_concat`` query representations),
   loads dense ``corpus*.pkl`` shards and a sparse Lucene
   ``index/``, normalizes both scores in the same per-query min-max fashion as
   ``hybrid.py``, and fuses them inside the Stage-1 candidate pool.

3. Stage 3 calls the existing pointwise reranker on the Stage-2 candidate
   list.  Caption-generation templates use ``caption_generation_rerank``;
   other templates use ``rerank``.

``encode.py`` is only expected to persist corpus-side artifacts.  No
``query.pkl`` or ``query.tsv`` is required.  Every intermediate run is still
saved for inspection or resume.

``--select-hybrid-alpha`` scans weights 0.1..0.9 from the same Stage-2 scores.
Paired validation indexes select the weight on validation; otherwise the target
split selects it. All weights are reported in ``stage2_alpha_sweep.csv`` before
Stage 3 reranks only the selected candidates.

Dataset adapters follow the existing retrieval tasks: COCO/Flickr use ``ir``,
CUHK-PEDES/ICFG-PEDES/RSTPReid use ``tbpr`` (text -> person images), WebQA
uses ``t2it`` (text -> image+text), and ReMuQ/LLaVA use ``it2t`` (image+text
-> text). The launcher selects these task/query/corpus settings from DATASET.

Each completed stage also reports query-level Recall@K using the same cutoffs
as ``metrices.RecallMetrics``.  ``recall_metrics.json`` and
``recall_metrics.xlsx`` are updated after every stage, including precomputed
runs, so completed measurements survive a later-stage failure.

Core-compute timings exclude model/index loading, evaluation and output I/O.
``timing_metrics.json`` / ``timing_metrics.xlsx`` distinguish their sum from
the end-to-end wall time (which includes those costs).
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import os
import pickle
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Optional, Sequence

from tqdm import tqdm

try:
    from rerank_prior_cache import validate_prior_cache_options
except ModuleNotFoundError as exc:
    if exc.name != "rerank_prior_cache":
        raise
    from src.rerank_prior_cache import validate_prior_cache_options


Run = dict[str, dict[str, float]]

STAGE1_METHODS = ("clip", "blip", "sparse_manual", "sparse_concat")
STAGE2_EOL_TYPE = "disassembleeol_separate"
STAGE2_CALCULATE_TYPE = "concat"
ALPHA_CANDIDATES = tuple(i / 10 for i in range(1, 10))

DATASET_TASK_TYPES = {
    "coco": "ir", "flickr": "ir", "fashion-iq": "cir",
    "CUHK-PEDES": "tbpr", "ICFG-PEDES": "tbpr", "RSTPReid": "tbpr",
    "webqa": "t2it", "edis": "t2it", "remuq": "it2t", "llava": "it2t",
}


def _progress(
    iterable: Any = None,
    *,
    desc: str,
    unit: str,
    total: Optional[int] = None,
    leave: bool = True,
    disable: Optional[bool] = None,
):
    """Create consistently formatted CLI progress bars for the pipeline."""

    return tqdm(
        iterable,
        desc=desc,
        unit=unit,
        total=total,
        leave=leave,
        dynamic_ncols=True,
        disable=(int(os.environ.get("RANK", "0")) != 0 if disable is None else disable),
    )


@dataclass
class EncodedQueries:
    """Dense vectors and Lucene topics produced from the same online pass."""

    query_ids: list[str]
    dense_representations: Any
    sparse_topics: dict[str, str]
    # Corpus-compatible fingerprint and pipeline settings survive model release.
    qwen_metadata: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class DistributedContext:
    """Process identity for single-process or NCCL data-parallel execution."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "N/A"
    milliseconds = round(max(seconds, 0.0) * 1000)
    hours, remaining = divmod(milliseconds, 3600000)
    minutes, remaining = divmod(remaining, 60000)
    return f"{hours:02d}:{minutes:02d}:{remaining / 1000:06.3f}"


def _model_cuda_devices(model: Any = None, device: Any = None) -> tuple[str, ...]:
    """Find only devices actually used, without initializing CUDA on CPU runs."""
    values = [device]
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, Mapping):
        values.extend(device_map.values())
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        values.extend(parameter.device for parameter in parameters())
    devices = set()
    for value in values:
        if isinstance(value, int):
            devices.add(f"cuda:{value}")
        elif value is not None and str(value).startswith("cuda"):
            normalized = str(value)
            if normalized == "cuda":
                import torch
                normalized = f"cuda:{torch.cuda.current_device()}"
            devices.add(normalized)
    return tuple(sorted(devices))


def _synchronize_cuda_devices(devices: Sequence[str]) -> None:
    if devices:
        import torch
        for device in devices:
            torch.cuda.synchronize(device)


class StageTimer:
    """Accumulate non-overlapping compute spans; never time stage setup/I/O."""

    def __init__(self, source: str = "computed") -> None:
        self.source = source
        self.components: dict[str, float] = {}
        self.auxiliary: dict[str, float] = {}
        self._devices: dict[tuple[int, str], tuple[str, ...]] = {}

    @contextmanager
    def measure(
        self, component: str, *, model: Any = None, device: Any = None,
        core: bool = True,
    ):
        key = (id(model), str(device))
        if key not in self._devices:
            self._devices[key] = _model_cuda_devices(model, device)
        devices = self._devices[key]
        _synchronize_cuda_devices(devices)
        started = time.perf_counter()
        try:
            yield
        except BaseException:
            # Failed CUDA work cannot be synchronized/timed reliably. The
            # pipeline records this stage as failed without claiming a total.
            raise
        else:
            _synchronize_cuda_devices(devices)
            elapsed = time.perf_counter() - started
            destination = self.components if core else self.auxiliary
            destination[component] = destination.get(component, 0.0) + elapsed

    def snapshot(self, rank: int) -> dict[str, Any]:
        seconds: Optional[float] = sum(self.components.values())
        if self.source == "external_command":
            seconds = None
        elif self.source == "precomputed":
            seconds = 0.0
        report = {
            "rank": rank,
            "source": self.source,
            "compute_seconds": seconds,
            "components_seconds": dict(self.components),
            "auxiliary_seconds": dict(self.auxiliary),
        }
        if hasattr(self, "prior_cache"):
            report["prior_cache"] = self.prior_cache
        return report


def _compute_scope(timer: Optional[StageTimer], component: str, **kwargs: Any):
    return timer.measure(component, **kwargs) if timer is not None else nullcontext()


def _merge_stage_timings(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Use the slowest rank's entire compute path, not summed GPU times."""
    measurable = all(sample["compute_seconds"] is not None for sample in samples)
    critical = max(samples, key=lambda sample: sample["compute_seconds"] or 0.0)
    report = {
        "source": samples[0]["source"],
        "compute_seconds": critical["compute_seconds"] if measurable else None,
        "critical_rank": critical["rank"] if measurable else None,
        "components_seconds": dict(critical["components_seconds"]) if measurable else {},
        "per_rank_compute_seconds": {str(sample["rank"]): sample["compute_seconds"] for sample in samples},
        "auxiliary_seconds": {
            name: max(sample["auxiliary_seconds"].get(name, 0.0) for sample in samples)
            for name in {name for sample in samples for name in sample["auxiliary_seconds"]}
        },
    }
    if "prior_cache" in critical:
        report["prior_cache"] = critical["prior_cache"]
    return report


def _collect_stage_timing(timer: StageTimer, context: DistributedContext) -> Optional[dict[str, Any]]:
    sample = timer.snapshot(context.rank)
    if not context.is_distributed:
        return _merge_stage_timings([sample])
    import torch.distributed as dist

    gathered: Optional[list[Any]] = [None] * context.world_size if context.is_main else None
    dist.gather_object(sample, gathered, dst=0)
    if not context.is_main:
        return None
    assert gathered is not None
    return _merge_stage_timings(gathered)


class TimingReporter:
    """Incremental rank-0 reports; failure finalization uses no collectives."""

    def __init__(self, args: Any, output_dir: Path, started: float) -> None:
        self.args = args
        self.output_dir = output_dir
        self.started = started
        self.status = "running"
        self.stages: dict[str, dict[str, Any]] = {
            stage: {"stage": stage, "status": "not_started", "compute_seconds": None,
                    "components_seconds": {}, "auxiliary_seconds": {}}
            for stage in ("stage1", "stage2", "stage3")
        }

    def record(self, stage: str, measurement: dict[str, Any]) -> None:
        self.stages[stage] = {"stage": stage, "status": "completed", **measurement}
        self._save(time.perf_counter() - self.started)
        seconds = measurement["compute_seconds"]
        display = f"{seconds:.3f}s ({_format_duration(seconds)})" if seconds is not None else "N/A"
        tqdm.write(f"[timing][{stage}] core_compute={display}, source={measurement['source']}")

    def finish(self, *, completed: bool, failed_stage: Optional[str] = None) -> None:
        wall_seconds = time.perf_counter() - self.started
        self.status = "completed" if completed else "failed"
        if failed_stage is not None:
            self.stages[failed_stage]["status"] = "failed"
            self.stages[failed_stage]["compute_seconds"] = None
        payload = self._save(wall_seconds)
        totals = payload["totals"]
        seconds = totals["compute_seconds"]
        display = (
            f"{seconds:.3f}s ({_format_duration(seconds)})" if seconds is not None
            else f"N/A (incomplete; measured completed stages={totals['measured_compute_seconds']:.3f}s)"
        )
        tqdm.write(f"[timing][total] core_compute={display}")
        tqdm.write(f"[timing][total] end_to_end={wall_seconds:.3f}s ({_format_duration(wall_seconds)}), status={self.status}")

    def _save(self, wall_seconds: float) -> dict[str, Any]:
        records = list(self.stages.values())
        complete = self.status == "completed" and all(
            row["status"] == "completed" and row["compute_seconds"] is not None
            for row in records
        )
        measured = sum(row["compute_seconds"] or 0.0 for row in records if row["status"] == "completed")
        payload = {
            **{name: getattr(self.args, name, None) for name in ("dataset_name", "dataset_split", "task_type", "query_type")},
            "status": self.status,
            "unit": "seconds",
            "compute_scope": "encoding/batch preprocessing, retrieval, sorting, fusion and reranking; excludes model/index/dataset setup, evaluation, output I/O and cleanup",
            "distributed_reduction": "max rank-local accumulated compute time per stage; components are from that same critical rank",
            "wall_scope": "after argument validation through result/recall/metadata output; excludes imports, final timing report output and distributed teardown",
            "stages": records,
            "totals": {
                "compute_complete": complete,
                "compute_seconds": measured if complete else None,
                "measured_compute_seconds": measured,
                "wall_seconds": wall_seconds,
            },
        }
        temporary = self.output_dir / "timing_metrics.tmp.json"
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        temporary.replace(self.output_dir / "timing_metrics.json")

        table = []
        for row in records:
            table.append({
                "record_type": "stage", "stage": row["stage"],
                "source": row.get("source"), "status": row["status"],
                "seconds": row["compute_seconds"], "duration": _format_duration(row["compute_seconds"]),
                "critical_rank": row.get("critical_rank"),
            })
            for component, seconds in row["components_seconds"].items():
                table.append({"record_type": "component", "stage": row["stage"],
                              "component": component, "seconds": seconds, "duration": _format_duration(seconds)})
            for component, seconds in row["auxiliary_seconds"].items():
                table.append({"record_type": "auxiliary_wall", "stage": row["stage"],
                              "component": component, "seconds": seconds, "duration": _format_duration(seconds)})
        for scope, seconds in (("core_compute_total", payload["totals"]["compute_seconds"]), ("end_to_end_total", wall_seconds)):
            table.append({"record_type": "total", "component": scope, "status": self.status,
                          "seconds": seconds, "duration": _format_duration(seconds)})
        pandas, engine = RecallReporter._output_dependencies()
        excel_temp = self.output_dir / "timing_metrics.tmp.xlsx"
        pandas.DataFrame(table).to_excel(excel_temp, index=False, engine=engine)
        excel_temp.replace(self.output_dir / "timing_metrics.xlsx")
        return payload


def _record_stage_timing(
    stage: str, timer: StageTimer, context: DistributedContext,
    reporter: Optional[TimingReporter],
) -> None:
    measurement = _collect_stage_timing(timer, context)
    if context.is_main:
        assert reporter is not None and measurement is not None
        reporter.record(stage, measurement)


def _initialise_distributed(local_rank_arg: int = -1) -> DistributedContext:
    """Initialise one NCCL process per GPU when launched by DeepSpeed/torchrun."""

    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            str(local_rank_arg if local_rank_arg >= 0 else 0),
        )
    )
    if world_size <= 1:
        return DistributedContext()
    if not torch.cuda.is_available():
        raise RuntimeError("多进程三阶段流水线需要 CUDA 和 NCCL。")

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(seconds=3600))
    return DistributedContext(
        rank=dist.get_rank(),
        local_rank=local_rank,
        world_size=dist.get_world_size(),
    )


def _shutdown_distributed(context: DistributedContext, *, synchronize: bool) -> None:
    if not context.is_distributed:
        return
    import torch.distributed as dist

    if not dist.is_initialized():
        return
    if synchronize:
        dist.barrier()
    dist.destroy_process_group()


def _partition_run(run: Run, world_size: int) -> list[Run]:
    """Split query entries deterministically without padding or duplication."""

    if world_size <= 0:
        raise ValueError(f"world_size 必须为正整数，当前为 {world_size}。")
    shards: list[Run] = [{} for _ in range(world_size)]
    for index, (query_id, scores) in enumerate(run.items()):
        shards[index % world_size][query_id] = scores
    return shards


def _merge_run_shards(shards: Sequence[Run]) -> Run:
    merged: Run = {}
    for shard in shards:
        duplicate_ids = set(merged).intersection(shard)
        if duplicate_ids:
            duplicate = sorted(duplicate_ids)[0]
            raise ValueError(f"分布式结果包含重复 query ID: {duplicate}")
        merged.update(shard)
    return merged


def _validate_query_coverage(expected: Run, actual: Run, stage: str) -> None:
    expected_ids = set(expected)
    actual_ids = set(actual)
    if expected_ids == actual_ids:
        return
    missing = sorted(expected_ids - actual_ids)
    unexpected = sorted(actual_ids - expected_ids)
    details = []
    if missing:
        details.append(f"missing={missing[0]}")
    if unexpected:
        details.append(f"unexpected={unexpected[0]}")
    raise ValueError(f"{stage} query 汇总不完整: {', '.join(details)}")


def _scatter_run(run: Run, context: DistributedContext) -> Run:
    if not context.is_distributed:
        return run
    import torch.distributed as dist

    received: list[Any] = [None]
    scatter_list = _partition_run(run, context.world_size) if context.is_main else None
    dist.scatter_object_list(received, scatter_list, src=0)
    shard = received[0]
    if not isinstance(shard, dict):
        raise TypeError(f"收到无效的 run 分片: {type(shard).__name__}")
    return shard


def _stage1_query_shard(
    args: argparse.Namespace, context: DistributedContext,
) -> tuple[list[str], list[str], list[int]]:
    """Partition query IDs before any rank allocates a query encoder."""
    global_ids: list[str] = []
    packets = None
    if context.is_main:
        try:
            dataset = build_dataset(args)
            global_ids = [_normalise_id(query_id) for query_id in _dataset_query_ids(args, dataset)]
            del dataset
            if not global_ids:
                raise ValueError("Stage 1 has no queries to encode.")
            if any(not query_id for query_id in global_ids) or len(global_ids) != len(set(global_ids)):
                raise ValueError("Stage-1 query IDs are empty or duplicated after normalization.")
            shards = [global_ids[rank::context.world_size] for rank in range(context.world_size)]
            counts = [len(shard) for shard in shards]
            packets = [{"ids": shard, "counts": counts, "error": None} for shard in shards]
        except Exception as exc:
            # All ranks receive metadata failures through the normal scatter;
            # no rank loads weights while peers wait for a successful packet.
            packets = [{"error": f"{type(exc).__name__}: {exc}"} for _ in range(context.world_size)]
    if context.is_distributed:
        import torch.distributed as dist

        received: list[Any] = [None]
        dist.scatter_object_list(received, packets, src=0)
        packet = received[0]
    else:
        packet = packets[0]
    if packet["error"] is not None:
        raise ValueError(f"Cannot partition Stage-1 queries: {packet['error']}")
    return packet["ids"], global_ids, packet["counts"]


def _validate_encoded_queries(
    encoded: EncodedQueries, expected_ids: Optional[Sequence[str]] = None,
) -> None:
    """Reject incomplete/misaligned CPU representations before reuse."""
    import numpy as np

    if not isinstance(encoded, EncodedQueries):
        raise TypeError("Expected EncodedQueries for Stage-1 query reuse.")
    ids = encoded.query_ids
    if any(not isinstance(query_id, str) or query_id != _normalise_id(query_id) for query_id in ids):
        raise ValueError("Encoded query IDs must be normalized strings.")
    if len(ids) != len(set(ids)):
        raise ValueError("Encoded queries contain duplicate query IDs.")
    dense = encoded.dense_representations
    if not isinstance(dense, np.ndarray) or dense.ndim != 2 or dense.shape[0] != len(ids):
        raise ValueError("Encoded dense query rows do not match the query IDs.")
    if (ids and dense.shape[1] == 0) or dense.dtype.kind != "f" or not np.isfinite(dense).all():
        raise ValueError("Encoded dense query vectors must be finite floating-point CPU arrays.")
    if set(encoded.sparse_topics) != set(ids) or any(
        not isinstance(topic, str) for topic in encoded.sparse_topics.values()
    ):
        raise ValueError("Encoded sparse topics do not match the query IDs.")
    if expected_ids is not None and ids != [_normalise_id(query_id) for query_id in expected_ids]:
        raise ValueError("Encoded query IDs/order do not match the Stage-2 query shard.")


def _select_encoded_queries(encoded: EncodedQueries, query_ids: Sequence[str]) -> EncodedQueries:
    """Slice both representations together, including reordered/empty shards."""
    import numpy as np

    _validate_encoded_queries(encoded)
    requested = [_normalise_id(query_id) for query_id in query_ids]
    if len(requested) != len(set(requested)):
        raise ValueError("Requested encoded query shard contains duplicate query IDs.")
    positions = {query_id: index for index, query_id in enumerate(encoded.query_ids)}
    missing = [query_id for query_id in requested if query_id not in positions]
    if missing:
        raise ValueError(f"Stage-1 encoding is missing query ID: {missing[0]}")
    return EncodedQueries(
        requested,
        np.ascontiguousarray(encoded.dense_representations[[positions[query_id] for query_id in requested]]),
        {query_id: encoded.sparse_topics[query_id] for query_id in requested},
        qwen_metadata=copy.deepcopy(encoded.qwen_metadata),
    )


def _scatter_encoded_queries(
    encoded: Optional[EncodedQueries], run: Run, local_run: Run, context: DistributedContext,
) -> EncodedQueries:
    """Send only each rank's CPU vectors/topics, using the run's partition order.

    Validation failures are sent in the same collective, so no worker silently
    falls back to model loading/encoding or waits for a successful root packet.
    """
    packets = None
    if context.is_main:
        try:
            _validate_encoded_queries(encoded)
            if set(encoded.query_ids) != set(run):
                raise ValueError("Stage-1 run and query encoding coverage differ.")
            packets = [
                {"encoded": _select_encoded_queries(encoded, list(shard)), "error": None}
                for shard in _partition_run(run, context.world_size)
            ]
        except (TypeError, ValueError) as exc:
            packets = [{"error": f"{type(exc).__name__}: {exc}"} for _ in range(context.world_size)]
    if context.is_distributed:
        import torch.distributed as dist

        received: list[Any] = [None]
        dist.scatter_object_list(received, packets, src=0)
        packet = received[0]
    else:
        packet = packets[0]
    if packet["error"] is not None:
        raise ValueError(f"Cannot reuse Stage-1 query encodings: {packet['error']}")
    shard = packet["encoded"]
    _validate_encoded_queries(shard, list(local_run))
    return shard


def _gather_run(local_run: Run, context: DistributedContext) -> Run:
    if not context.is_distributed:
        return local_run
    import torch.distributed as dist

    gathered: Optional[list[Any]] = (
        [None for _ in range(context.world_size)] if context.is_main else None
    )
    dist.gather_object(local_run, gathered, dst=0)
    if not context.is_main:
        return {}
    assert gathered is not None
    if any(not isinstance(shard, dict) for shard in gathered):
        raise TypeError("分布式 run 汇总收到非字典分片。")
    return _merge_run_shards(gathered)


def _device_map_for_process(
    configured: str,
    context: DistributedContext,
) -> Any:
    normalized = configured.lower()
    if normalized == "none":
        return "none"
    if context.is_distributed and normalized == "auto":
        return {"": context.local_rank}
    return configured


def _normalise_id(value: Any) -> str:
    """Convert IDs from parquet/pickle/numpy scalar types to stable strings."""

    if value is None:
        return ""
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].lstrip("+-").isdigit():
        return text[:-2]
    return text


def _score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"无法把检索分数转换为浮点数: {value!r}") from exc


def _plain_scores(value: Any) -> dict[str, float]:
    """Read the run shapes used by both ``hybrid.py`` and TREC/JSON files."""

    if isinstance(value, Mapping) and "docs" in value:
        value = value["docs"]

    if isinstance(value, Mapping):
        return {
            _normalise_id(doc_id): _score(doc_score)
            for doc_id, doc_score in value.items()
            if _normalise_id(doc_id)
        }

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result: dict[str, float] = {}
        for item in value:
            if isinstance(item, Mapping):
                doc_id = item.get("docid", item.get("doc_id", item.get("id")))
                doc_score = item.get("score", item.get("value", 0.0))
            elif isinstance(item, Sequence) and len(item) >= 2:
                doc_id, doc_score = item[0], item[1]
            else:
                raise ValueError(f"无法解析 run 中的候选项: {item!r}")
            result[_normalise_id(doc_id)] = _score(doc_score)
        return result

    raise ValueError(f"无法解析 run 中的候选分数字典: {value!r}")


def _run_from_json_payload(payload: Any) -> Run:
    if isinstance(payload, Mapping) and "run" in payload:
        payload = payload["run"]

    if isinstance(payload, Mapping):
        return {
            _normalise_id(query_id): _plain_scores(scores)
            for query_id, scores in payload.items()
            if _normalise_id(query_id)
        }

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        run: Run = {}
        for row in payload:
            if not isinstance(row, Mapping):
                raise ValueError(f"JSON run 列表中的元素必须是对象: {row!r}")
            query_id = _normalise_id(row.get("qid", row.get("query_id", row.get("query-id"))))
            doc_id = _normalise_id(row.get("docid", row.get("doc_id", row.get("corpus_id", row.get("corpus-id")))))
            if not query_id or not doc_id:
                raise ValueError(f"JSON run 记录缺少 qid/docid: {row!r}")
            run.setdefault(query_id, {})[doc_id] = _score(row.get("score", 0.0))
        return run

    raise ValueError("JSON run 必须是 query -> candidates 映射或记录列表。")


def read_run(path: Path) -> Run:
    """Read a TREC run, JSON run, JSONL run, or pickle run."""

    if not path.is_file():
        raise FileNotFoundError(f"找不到 run 文件: {path}")

    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        if suffix == ".json":
            with path.open("r", encoding="utf-8") as handle:
                return _run_from_json_payload(json.load(handle))

        run: Run = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    line_run = _run_from_json_payload(payload)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"无法解析 JSONL run 第 {line_number} 行: {path}") from exc
                for query_id, scores in line_run.items():
                    run.setdefault(query_id, {}).update(scores)
        return run

    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            return _run_from_json_payload(pickle.load(handle))

    run: Run = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) < 5:
                raise ValueError(f"TREC run 第 {line_number} 行字段不足: {line!r}")
            query_id = _normalise_id(fields[0])
            doc_id = _normalise_id(fields[2])
            if query_id and doc_id:
                run.setdefault(query_id, {})[doc_id] = _score(fields[4])
    return run


def write_trec_run(run: Run, path: Path, tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for query_id, scores in run.items():
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            for rank, (doc_id, value) in enumerate(ranked, start=1):
                handle.write(f"{query_id} Q0 {doc_id} {rank} {value:.12g} {tag}\n")


def write_json_run(run: Run, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(run, handle, ensure_ascii=False, indent=2)


def _top_ids(scores: Mapping[str, float], depth: int) -> list[str]:
    if depth <= 0:
        return []
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:depth]]


def _minmax(scores: Mapping[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    minimum = min(scores.values())
    maximum = max(scores.values())
    denominator = max(maximum - minimum, 1e-9)
    return {doc_id: (value - minimum) / denominator for doc_id, value in scores.items()}


def fuse_restricted_runs(
    candidate_run: Run,
    dense_run: Run,
    sparse_run: Run,
    alpha: float,
    candidate_depth: int,
    progress_desc: str = "[stage2] fuse runs",
    timing: Optional[StageTimer] = None,
) -> Run:
    """Fuse Stage-2 runs only over the candidate pool from Stage 1."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"hybrid alpha 必须位于 [0, 1]，当前为 {alpha}。")

    result: Run = {}
    with _compute_scope(timing, "fusion"):
        for query_id, stage1_scores in _progress(
            candidate_run.items(),
            desc=progress_desc,
            unit="query",
            total=len(candidate_run),
        ):
            candidate_ids = _top_ids(stage1_scores, candidate_depth)
            dense_scores = _minmax(dense_run.get(query_id, {}))
            sparse_scores = _minmax(sparse_run.get(query_id, {}))
            fused: dict[str, float] = {}
            for doc_id in candidate_ids:
                fused[doc_id] = (
                    alpha * dense_scores.get(doc_id, 0.0)
                    + (1.0 - alpha) * sparse_scores.get(doc_id, 0.0)
                )
            result[query_id] = dict(sorted(fused.items(), key=lambda item: item[1], reverse=True))
    return result


def _to_numpy(value: Any):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _load_pickle_pair(path: Path) -> tuple[Any, list[Any]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (tuple, list)) or len(payload) != 2:
        raise ValueError(f"密集表示文件必须保存为 (representations, lookup): {path}")
    reps, lookup = payload
    return reps, list(lookup)


def _find_corpus_pickles(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    matches = sorted(path.glob("corpus*.pkl"))
    if not matches:
        raise FileNotFoundError(f"在 {path} 下找不到 corpus*.pkl。")
    return matches


def load_dense_corpus(path: Path) -> tuple[Any, list[str]]:
    import numpy as np

    rep_parts = []
    lookup: list[str] = []
    for corpus_path in _find_corpus_pickles(path):
        reps, part_lookup = _load_pickle_pair(corpus_path)
        array = _to_numpy(reps).astype("float32", copy=False)
        if array.ndim != 2:
            raise ValueError(f"密集表示必须是二维数组: {corpus_path}, shape={array.shape}")
        rep_parts.append(array)
        lookup.extend(_normalise_id(item) for item in part_lookup)
    if not rep_parts:
        raise ValueError(f"没有读取到任何密集表示: {path}")
    representations = np.concatenate(rep_parts, axis=0)
    if len(lookup) != len(representations):
        raise ValueError(
            f"密集表示数量和 lookup 数量不一致: {len(representations)} != {len(lookup)}"
        )
    return representations, lookup


def _l2_normalize(array: Any):
    import numpy as np

    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def retrieve_dense_from_corpus(
    query_representations: Any,
    query_lookup: Sequence[str],
    corpus_path: Path,
    qids: Iterable[str],
    depth: int,
    candidate_filter: Optional[Mapping[str, set[str]]] = None,
    normalize: bool = True,
    progress_desc: str = "[stage2] dense retrieval",
    timing: Optional[StageTimer] = None,
) -> Run:
    import numpy as np

    query_reps = _to_numpy(query_representations).astype("float32", copy=False)
    query_lookup = [_normalise_id(item) for item in query_lookup]
    corpus_reps, corpus_lookup = load_dense_corpus(corpus_path)
    if query_reps.ndim != 2 or corpus_reps.ndim != 2:
        raise ValueError("在线 query 表征和 corpus*.pkl 表征必须是二维数组。")
    if query_reps.shape[1] != corpus_reps.shape[1]:
        raise ValueError(
            f"query/corpus 表征维度不一致: {query_reps.shape[1]} != {corpus_reps.shape[1]}"
        )
    if normalize:
        corpus_reps = _l2_normalize(corpus_reps)

    corpus_to_row: dict[str, int] = {}
    for index, corpus_id in enumerate(corpus_lookup):
        corpus_to_row.setdefault(corpus_id, index)

    requested_qids = list(qids)
    result: Run = {}
    with _compute_scope(timing, "dense_search"):
        if normalize:
            query_reps = _l2_normalize(query_reps)
        query_to_row = {query_id: index for index, query_id in enumerate(query_lookup)}
        for query_id in _progress(
            requested_qids,
            desc=progress_desc,
            unit="query",
            total=len(requested_qids),
        ):
            query_id = _normalise_id(query_id)
            row_index = query_to_row.get(query_id)
            if row_index is None:
                continue
            allowed = candidate_filter.get(query_id) if candidate_filter is not None else None
            if allowed is None:
                indices = np.arange(len(corpus_lookup), dtype=np.int64)
            else:
                indices = np.asarray(
                    [corpus_to_row[doc_id] for doc_id in allowed if doc_id in corpus_to_row],
                    dtype=np.int64,
                )
            if len(indices) == 0:
                result[query_id] = {}
                continue
            scores = corpus_reps[indices] @ query_reps[row_index]
            order = np.argsort(-scores, kind="stable")[:depth]
            result[query_id] = {
                corpus_lookup[int(indices[position])]: float(scores[position])
                for position in order
            }
    return result


def _resolve_lucene_index(path: Path) -> Path:
    from qwen_vl_paths import has_qwen_manifest, resolve_sparse_index
    if has_qwen_manifest(path):
        return resolve_sparse_index(path)
    if (path / "index").is_dir():
        return path / "index"
    return path


def _topic_token_frequencies(topic: str) -> dict[str, int]:
    """Match Pyserini's local ``TokFreqQueryEncoder`` without importing it."""

    frequencies: dict[str, int] = {}
    for token in topic.strip().split():
        frequencies[token] = frequencies.get(token, 0) + 1
    return frequencies


def _positive_idf_terms(
    index_reader: Any,
    *,
    progress_desc: str,
) -> set[str]:
    """Return terms accepted by ``LuceneImpactSearcher(min_idf=0)``."""

    stats = index_reader.stats()
    document_count = int(stats["documents"])
    unique_terms = stats.get("unique_terms")
    term_total = int(unique_terms) if unique_terms is not None else None
    allowed_terms: set[str] = set()
    for term in _progress(
        index_reader.terms(),
        desc=progress_desc,
        unit="term",
        total=term_total,
    ):
        if 0 < int(term.df) < document_count:
            allowed_terms.add(str(term.term))
    return allowed_terms


def retrieve_sparse_topics_from_lucene(
    index_path: Path,
    topics_by_id: Mapping[str, str],
    qids: Iterable[str],
    depth: int,
    threads: int,
    progress_prefix: str = "[sparse]",
    timing: Optional[StageTimer] = None,
) -> Run:
    requested_ids = [
        _normalise_id(qid)
        for qid in qids
        if _normalise_id(qid) in topics_by_id
    ]
    result: Run = {query_id: {} for query_id in requested_ids}
    selected_ids = [query_id for query_id in requested_ids if topics_by_id[query_id].strip()]
    if not selected_ids:
        return result

    # Import only Pyserini's local Lucene/JVM bindings. Importing
    # LuceneImpactSearcher also imports pyserini.encode, whose optional OpenAI
    # encoder can require credentials even though these topics are fully local.
    try:
        from pyserini.index.lucene import LuceneIndexReader
        from pyserini.pyclass import JArrayList, JHashMap, JInt, autoclass
    except ImportError as exc:
        raise RuntimeError(
            "稀疏阶段需要 pyserini。请使用当前项目用于 sparse search 的环境运行。"
        ) from exc

    index = _resolve_lucene_index(index_path)
    index_reader = LuceneIndexReader(str(index))
    try:
        allowed_terms = _positive_idf_terms(
            index_reader,
            progress_desc=f"{progress_prefix} scan vocabulary",
        )
    finally:
        reader_close: Any = getattr(index_reader, "close", None)
        if callable(reader_close):
            reader_close()

    simple_impact_searcher = autoclass("io.anserini.search.SimpleImpactSearcher")
    searcher = simple_impact_searcher(str(index))
    try:
        with _compute_scope(timing, "sparse_search"):
            query_list = JArrayList()
            qid_list = JArrayList()
            for query_id in _progress(
                selected_ids,
                desc=f"{progress_prefix} prepare sparse queries",
                unit="query",
                total=len(selected_ids),
            ):
                java_query = JHashMap()
                for token, weight in _topic_token_frequencies(
                    topics_by_id[query_id]
                ).items():
                    if token in allowed_terms:
                        java_query.put(token, JInt(weight))
                query_list.add(java_query)
                qid_list.add(query_id)

            with _progress(
                desc=f"{progress_prefix} Lucene search",
                unit="batch",
                total=1,
            ) as search_progress:
                java_results = searcher.batch_search(
                    query_list,
                    qid_list,
                    int(depth),
                    max(int(threads), 1),
                )
                search_progress.update(1)
            raw_results = {
                _normalise_id(entry.getKey()): entry.getValue()
                for entry in java_results.entrySet().toArray()
            }
            for query_id in _progress(
                selected_ids,
                desc=f"{progress_prefix} collect sparse hits",
                unit="query",
                total=len(selected_ids),
            ):
                hits = raw_results.get(query_id, [])
                result[query_id] = {
                    _normalise_id(hit.docid): float(hit.score)
                    for hit in hits
                }
        return result
    finally:
        close_fn: Any = getattr(searcher, "close", None)
        if callable(close_fn):
            close_fn()


def _retrieval_dtype(name: str):
    import torch

    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float16


def _load_retrieval_runtime(
    args: argparse.Namespace,
    distributed_context: Optional[DistributedContext] = None,
) -> Any:
    """Load the online query encoder in the same way as search.py/rerank.py."""

    import torch
    from transformers import (
        AutoModel,
        AutoProcessor,
        LlavaForConditionalGeneration,
        LlavaNextForConditionalGeneration,
        LlavaNextProcessor,
        LlavaProcessor,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VLProcessor,
    )

    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    model_path = args.retrieval_model_name_or_path
    if not model_path:
        raise ValueError("在线大模型查询编码需要 --retrieval-model-name-or-path。")

    context = distributed_context or DistributedContext()
    dtype = _retrieval_dtype(args.retrieval_dtype)
    if context.is_distributed and args.retrieval_device in {"auto", "cuda"}:
        device = torch.device("cuda", context.local_rank)
    else:
        device = torch.device(
            args.retrieval_device
            if args.retrieval_device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
    load_kwargs: dict[str, Any] = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    device_map = _device_map_for_process(args.retrieval_device_map, context)
    if device_map != "none":
        load_kwargs["device_map"] = device_map

    from qwen_vl_prompts import is_native_qwen_checkpoint, model_family
    if is_native_qwen_checkpoint(model_path):
        from qwen_vl_pipeline import load_native_qwen
        if args.sparse_logits_multiply_view_count:
            raise ValueError("Qwen does not support the LLaVA-only sparse x5 ablation")
        encoder, processor = load_native_qwen(
            model_path, dtype=dtype, device_map=None if device_map == "none" else device_map,
            device=device, min_pixels=getattr(args, "retrieval_qwen_min_pixels", 0),
            max_pixels=getattr(args, "retrieval_qwen_max_pixels", 0),
            lora_model_path=args.retrieval_lora_model_path,
        )
        return SimpleNamespace(
            model=encoder, encoder=encoder, processor=processor, tokenizer=processor.tokenizer,
            model_path=model_path, device=device, dtype=dtype, query_cache={},
            qwen_family=model_family(encoder.config), qwen_load_settings=_qwen_load_settings(args),
        )

    from encode import get_filtered_ids
    import model as model_module

    def load(model_class: Any):
        try:
            return model_class.from_pretrained(model_path, **load_kwargs)
        except TypeError:
            fallback = dict(load_kwargs)
            fallback.pop("torch_dtype", None)
            fallback["dtype"] = dtype
            return model_class.from_pretrained(model_path, **fallback)

    if "llava-hf-llava-1.5-7b-hf" in model_path:
        encoder = load(LlavaForConditionalGeneration)
        processor = LlavaProcessor.from_pretrained(model_path)
    elif "Qwen2.5-VL-7B-Instruct" in model_path or "Qwen2.5-VL-3B-Instruct" in model_path:
        encoder = load(Qwen2_5_VLForConditionalGeneration)
        processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
    elif "InternVL2_5-8B" in model_path or "InternVL2_5-4B" in model_path:
        encoder = AutoModel.from_pretrained(
            model_path,
            **load_kwargs,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    else:
        encoder = load(LlavaNextForConditionalGeneration)
        processor = LlavaNextProcessor.from_pretrained(model_path)
        if "royokong-e5-v" in model_path:
            setattr(processor, "patch_size", 14)

    if args.retrieval_device_map.lower() == "none":
        encoder = encoder.to(device)
    if args.retrieval_lora_model_path:
        from peft import PeftModel

        encoder = PeftModel.from_pretrained(encoder, args.retrieval_lora_model_path)
        encoder = encoder.merge_and_unload()

    tokenizer = getattr(processor, "tokenizer", processor)
    if args.retrieval_reps_loc == "after_pad" and hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
        tokenizer.padding = True
    raw_vocab = tokenizer.get_vocab()
    vocab_dict = {token_id: token for token, token_id in raw_vocab.items()}
    filtered_ids = get_filtered_ids(tokenizer)
    original_get_rank = None
    if not model_module.dist.is_initialized():
        original_get_rank = model_module.dist.get_rank
        model_module.dist.get_rank = lambda: 0
    model = model_module.MLLMRetrievalModel(encoder, vocab_dict=raw_vocab).eval()
    return SimpleNamespace(
        model=model,
        encoder=encoder,
        processor=processor,
        tokenizer=tokenizer,
        image_tokenizer=processor if "InternVL2_5" in model_path else tokenizer,
        vocab_dict=vocab_dict,
        filtered_ids=filtered_ids,
        model_path=model_path,
        device=device,
        dtype=dtype,
        query_cache={},
        model_module=model_module,
        original_get_rank=original_get_rank,
    )


def _release_retrieval_runtime(runtime: Any) -> None:
    if runtime is None:
        return
    import torch

    if getattr(runtime, "original_get_rank", None) is not None:
        runtime.model_module.dist.get_rank = runtime.original_get_rank
    for name in ("model", "encoder", "processor", "tokenizer", "image_tokenizer", "qwen_adapter"):
        if hasattr(runtime, name):
            setattr(runtime, name, None)
    if hasattr(runtime, "query_cache"):
        runtime.query_cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _retrieval_namespaces(
    args: argparse.Namespace,
    *,
    eol_type: str,
    calculate_type: str,
    sparse_manual: bool,
) -> tuple[Any, Any]:
    model_args = SimpleNamespace(
        model_name_or_path=args.retrieval_model_name_or_path,
        eol_type=eol_type,
        calculate_type=calculate_type,
        use_output_embedding_cluster=False,
        lora=bool(getattr(args, "retrieval_lora_model_path", None)),
        lora_model_path=getattr(args, "retrieval_lora_model_path", None),
    )
    data_args = SimpleNamespace(
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        qwen_min_pixels=getattr(args, "retrieval_qwen_min_pixels", 0),
        qwen_max_pixels=getattr(args, "retrieval_qwen_max_pixels", 0),
        per_device_batch_size=args.retrieval_batch_size,
        reps_loc=args.retrieval_reps_loc,
        sparse_manual=sparse_manual,
        sparse_length=args.sparse_length,
        text_sparse_length=args.text_sparse_length,
        image_sparse_length=args.image_sparse_length,
        num_expended_tokens=args.num_expended_tokens,
        is_filtered=args.is_filtered,
        sparse_value_type=args.sparse_value_type,
        sparse_lower_or_upper=args.sparse_lower_or_upper,
        prompt_type=args.prompt_type,
        random_view_seed=args.random_view_seed,
        prompt_generation=False,
        prompt_generation_model="llama",
        sparse_value_mean=args.sparse_value_mean,
        sparse_logits_multiply_view_count=args.sparse_logits_multiply_view_count,
        sparse_type="single",
        tbpr_type=args.tbpr_type,
        cir_type=args.cir_type,
        print_sparse=args.print_sparse,
        composed_top_type="text",
        composed_top_k=3,
    )
    return model_args, data_args


def _qwen_load_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {key: getattr(args, key, None) for key in (
        "retrieval_model_name_or_path", "retrieval_lora_model_path", "retrieval_dtype",
        "retrieval_qwen_min_pixels", "retrieval_qwen_max_pixels",
    )}


def _is_qwen_checkpoint(path):
    src_dir = str(Path(__file__).resolve().parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from qwen_vl_prompts import is_native_qwen_checkpoint
    return bool(path) and is_native_qwen_checkpoint(path)


def _qwen_encoding_settings(args, eol_type, calculate_type, sparse_manual):
    model_args, data_args = _retrieval_namespaces(
        args, eol_type=eol_type, calculate_type=calculate_type, sparse_manual=sparse_manual,
    )
    return dict(model=vars(model_args), data=vars(data_args), task=args.task_type,
                query_type=args.query_type, dtype=args.retrieval_dtype)


def _qwen_query_records(args, records):
    """Preserve retrieval modalities, never attach reference captions to IR images."""
    result = []
    for record in records:
        value = dict(record)
        value["text"] = _plain_text(value.get("text", ""))
        if args.task_type == "ir" and args.query_type == "image":
            value["text"] = ""
        elif args.task_type == "cir":
            value["composed"] = True
        result.append(value)
    return result


def _validate_qwen_artifacts(args, encoded, paths, *, eol_type, calculate_type, sparse_manual):
    """Validate even the rank-local reuse path, without reloading GPU weights."""
    if not encoded.query_ids:
        return
    expected = encoded.qwen_metadata
    if expected is None:
        if _is_qwen_checkpoint(getattr(args, "retrieval_model_name_or_path", None)):
            raise ValueError("Qwen query encodings are missing their configuration fingerprint; encode again")
        return
    from qwen_vl_pipeline import validate_manifest
    settings = _qwen_encoding_settings(args, eol_type, calculate_type, sparse_manual)
    if expected.get("pipeline_settings") != settings:
        raise ValueError("Qwen query reuse configuration mismatch; encoding settings changed between stages")
    for path, dense in paths:
        actual = validate_manifest(path, expected)
        if dense and len(_find_corpus_pickles(Path(path))) != actual["num_shards"]:
            raise ValueError("Qwen dense corpus shard count differs from its manifest")


def _dataset_query_ids(args: argparse.Namespace, dataset: Any) -> Sequence[Any]:
    if args.task_type in {"t2it", "it2t"}:
        return dataset.query_id_list
    if args.task_type == "cir":
        return dataset.composed_id_list
    if args.task_type == "tbpr" or args.query_type == "text":
        return dataset.text_id_list
    return dataset.img_id_list


def _query_records(args: argparse.Namespace, dataset: Any) -> list[dict[str, Any]]:
    """Return the query side only; corpus representations come from encode.py."""

    data_root = Path(args.data_root).expanduser().resolve()
    records: list[dict[str, Any]] = []
    if args.task_type == "t2it":
        for query_id in dataset.query_id_list:
            records.append({"id": _normalise_id(query_id), "text": dataset.id2query[query_id]})
        return records

    if args.task_type == "it2t":
        for query_id in dataset.query_id_list:
            query = dataset.id2query[query_id]
            records.append(
                {
                    "id": _normalise_id(query_id),
                    "text": query["text"],
                    "image": query["image"],
                }
            )
        return records

    if args.task_type == "cir":
        for composed_id in dataset.composed_id_list:
            separator = composed_id.index("_")
            image_id = composed_id[:separator]
            text_id = composed_id[separator + 1:]
            dress_type = dataset.dress_type_dict[image_id]
            if dress_type == "toptee":
                dress_type = "shirt"
            records.append(
                {
                    "id": _normalise_id(composed_id),
                    "text": dataset.get_text(text_id),
                    "image": data_root / "data" / dataset.data_name / "images" / "images" / dataset.img_dict[image_id],
                    "dress_type": dress_type,
                }
            )
        return records

    if args.query_type == "text" or args.task_type == "tbpr":
        for text_id in dataset.text_id_list:
            records.append({"id": _normalise_id(text_id), "text": dataset.text_dict[text_id]})
        return records

    for image_id in dataset.img_id_list:
        companion_text = ""
        text_ids = getattr(dataset, "img2text", {}).get(image_id, [])
        if text_ids:
            companion_text = dataset.text_dict[text_ids[0]]
        records.append(
            {
                "id": _normalise_id(image_id),
                "text": companion_text,
                "image": _standard_image_path(dataset, image_id, data_root),
            }
        )
    return records


def _select_query_records(
    records: Sequence[dict[str, Any]],
    requested_query_ids: Optional[Sequence[str]],
) -> list[dict[str, Any]]:
    if requested_query_ids is None:
        return list(records)
    requested = [_normalise_id(query_id) for query_id in requested_query_ids]
    if len(requested) != len(set(requested)):
        raise ValueError("query 分片中包含重复 ID。")
    records_by_id = {
        _normalise_id(record["id"]): record
        for record in records
    }
    missing = [query_id for query_id in requested if query_id not in records_by_id]
    if missing:
        raise KeyError(f"数据集中找不到分片 query ID: {missing[0]}")
    return [records_by_id[query_id] for query_id in requested]


def _plain_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return " ".join(str(item) for item in value)
    return str(value)


def _base_image_prompt(runtime: Any, args: argparse.Namespace, task_type: str) -> Any:
    import template as templates

    model_path = runtime.model_path
    if task_type == "it2t":
        if "llava-hf-llava-1.5-7b-hf" in model_path:
            prompt = templates.img_prompt_no_special_llava_v1_5
        elif "Qwen2.5-VL" in model_path:
            prompt = templates.qwen2_5_img_prompt
        elif "InternVL2_5" in model_path:
            prompt = templates.img_prompt_intern_vl_v2_5
        elif "llava-hf-llava-v1.6-mistral-7b-hf" in model_path:
            prompt = templates.mistral_it2t_query_prompt
        elif "llava-hf-llava-v1.6-vicuna" in model_path:
            prompt = templates.vicuna_img_prompt
        else:
            prompt = templates.it2t_query_prompt
    elif "llava-hf-llava-1.5-7b-hf" in model_path:
        prompt = templates.img_prompt_no_special_llava_v1_5
    elif "Qwen2.5-VL" in model_path:
        prompt = runtime.processor.apply_chat_template(
            templates.img_prompt_qwen_v2_5, tokenize=False, add_generation_prompt=True
        )
    elif "InternVL2_5" in model_path:
        prompt = runtime.processor.apply_chat_template(
            templates.img_prompt_intern_vl_v2_5, tokenize=False, add_generation_prompt=True
        )
    elif "llava-hf-llava-v1.6-mistral-7b-hf" in model_path:
        prompt = templates.mistral_img_prompt
    elif "llava-hf-llava-v1.6-vicuna" in model_path:
        prompt = templates.vicuna_img_prompt
    else:
        prompt = templates.img_prompt
    return prompt


def _concat_aspect_prompts(templates: Any, prompt_type: str) -> Sequence[str]:
    names = {
        "prompt_1": "retrieval_disassemble_image_prompts_1_for_concat",
        "prompt_2": "retrieval_disassemble_image_prompts_2_for_concat",
        "prompt_3": "retrieval_disassemble_image_prompts_3_for_concat",
        "prompt_4": "retrieval_disassemble_image_prompts_4_for_concat",
        "prompt_5": "retrieval_disassemble_image_prompts_for_concat",
        "prompt_6": "retrieval_disassemble_image_prompts_6_for_concat",
        "prompt_7": "retrieval_disassemble_image_prompts_7_for_concat",
    }
    if prompt_type not in names:
        raise ValueError(f"当前在线 concat 编码不支持 prompt_type={prompt_type!r}。")
    return getattr(templates, names[prompt_type])


def _concat_image_prompt(runtime: Any, args: argparse.Namespace, task_type: str) -> str:
    import template as templates

    model_path = runtime.model_path
    if task_type == "it2t":
        if "llava-hf-llava-v1.6-mistral-7b-hf" in model_path:
            prefix = templates.llava_mistral_template_fusion_prefix
            element = templates.llava_mistral_template_content_element
            original_prompt = templates.fusion_prompt_for_concat
        else:
            prefix = templates.llama3_template_fusion_prefix
            element = templates.llama3_template_content_element
            original_prompt = templates.img_prompt_for_concat
        aspects = (
            templates.retrieval_disassemble_query_prompts_llava_it2t_retrieval_for_concat
            if args.dataset_name == "llava"
            else templates.retrieval_disassemble_query_prompts_it2t_retrieval_for_concat
        )
    else:
        if "llava-hf-llava-v1.6-mistral-7b-hf" in model_path:
            prefix = templates.llava_mistral_template_image_prefix
            element = templates.llava_mistral_template_content_element
        elif "llava-hf-llava-v1.6-vicuna" in model_path:
            prefix = templates.llava_vicuna_template_image_prefix
            element = templates.llava_vicuna_template_content_element
        elif "llava-hf-llava-v1.6-34b-hf" in model_path:
            prefix = templates.llava_34b_template_image_prefix
            element = templates.llava_34b_template_content_element
        elif "Qwen2.5-VL-7B-Instruct" in model_path:
            prefix = templates.qwen2_5_template_image_prefix
            element = templates.qwen2_5_template_content_element
        else:
            prefix = templates.llama3_template_image_prefix
            element = templates.llama3_template_content_element
        original_prompt = templates.img_prompt_for_concat
        aspects = _concat_aspect_prompts(templates, args.prompt_type)

    prompt = prefix + element.format(original_prompt)
    for aspect in aspects:
        prompt += element.format(aspect)
    return prompt


def _prepare_image_inputs(
    runtime: Any,
    args: argparse.Namespace,
    records: Sequence[Mapping[str, Any]],
    *,
    calculate_type: str,
) -> Any:
    raw_images = [_image_from_value(record["image"]) for record in records]
    if "InternVL2_5" in runtime.model_path and calculate_type == "separate":
        import torch
        from utils import build_transform, dynamic_preprocess

        transform = build_transform(input_size=448)
        pixel_batches = []
        for image in raw_images:
            patches = dynamic_preprocess(
                image, image_size=448, use_thumbnail=True, max_num=12
            )
            pixel_batches.append(
                torch.stack([transform(patch) for patch in patches])
                .to(dtype=runtime.dtype, device=runtime.device)
            )
        return pixel_batches
    texts = [_plain_text(record.get("text", "")) for record in records]
    if calculate_type == "concat":
        prompt = _concat_image_prompt(runtime, args, args.task_type)
    else:
        prompt = _base_image_prompt(runtime, args, args.task_type)
    prompt_texts = [str(prompt).replace("<sent>", text) for text in texts]
    inputs = runtime.processor(
        images=raw_images,
        text=prompt_texts,
        return_tensors="pt",
        padding=True,
    )
    if hasattr(inputs, "to"):
        return inputs.to(runtime.device)
    return {
        key: value.to(runtime.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def _encode_query_batch(
    runtime: Any,
    args: argparse.Namespace,
    records: Sequence[Mapping[str, Any]],
    model_args: Any,
    data_args: Any,
) -> tuple[Any, Any]:
    texts = [_plain_text(record.get("text", "")) for record in records]
    calculate_type = model_args.calculate_type
    if args.task_type == "cir":
        raw_images = [_image_from_value(record["image"]) for record in records]
        dress_types = [record["dress_type"] for record in records]
        method = (
            runtime.model.encode_data_for_cir
            if calculate_type == "separate"
            else runtime.model.encode_data_concat_for_cir
        )
        return method(
            texts,
            raw_images,
            dress_types,
            "composed",
            runtime.processor,
            runtime.device,
            model_args,
            data_args,
        )

    if args.task_type == "t2it":
        method = (
            runtime.model.encode_data_for_t2it
            if calculate_type == "separate"
            else runtime.model.encode_data_concat_for_t2it
        )
        return method(texts, "query", runtime.processor, runtime.device, model_args, data_args)

    if args.task_type == "tbpr":
        method = (
            runtime.model.encode_data_for_tbpr
            if calculate_type == "separate"
            else runtime.model.encode_data_concat_for_tbpr
        )
        return method(texts, "text", runtime.processor, runtime.device, model_args, data_args)

    if args.task_type == "it2t" or args.query_type == "image":
        image_inputs = _prepare_image_inputs(
            runtime, args, records, calculate_type=calculate_type
        )
        if args.task_type == "it2t":
            method = (
                runtime.model.encode_data_for_it2t
                if calculate_type == "separate"
                else runtime.model.encode_data_concat_for_it2t
            )
            return method(
                image_inputs, "query", runtime.processor, runtime.device, model_args, data_args
            )
        method = (
            runtime.model.encode_data
            if calculate_type == "separate"
            else runtime.model.encode_data_concat
        )
        return method(
            image_inputs, "image", runtime.processor, runtime.device, model_args, data_args
        )

    method = (
        runtime.model.encode_data
        if calculate_type == "separate"
        else runtime.model.encode_data_concat
    )
    return method(texts, "text", runtime.processor, runtime.device, model_args, data_args)


def _merge_sparse_values(
    tokens: Sequence[str], values: Sequence[Any], value_type: str
) -> dict[str, int]:
    vector: dict[str, int] = {}
    for token, value in zip(tokens, values):
        weight = int(value)
        if not token or weight <= 0:
            continue
        if token not in vector:
            vector[token] = weight
        elif value_type == "replace":
            vector[token] = weight
        elif value_type == "sum":
            vector[token] += weight
        else:
            vector[token] = max(vector[token], weight)
    return vector


def _vector_to_topic(vector: Mapping[str, int]) -> str:
    return " ".join(
        " ".join([str(token)] * int(weight))
        for token, weight in vector.items()
        if int(weight) > 0
    )


def _batch_sparse_topics(
    runtime: Any,
    records: Sequence[Mapping[str, Any]],
    logits: Any,
    model_args: Any,
    data_args: Any,
    args: argparse.Namespace,
) -> dict[str, str]:
    import torch
    from encode import (
        get_img_valid_disassemble_tokens_values,
        get_img_valid_tokens_values,
        get_text_valid_disassemble_tokens_values,
        get_text_valid_tokens_values,
    )

    if isinstance(logits, list):
        logits = torch.cat(list(logits), dim=0)
    if not torch.is_tensor(logits) or logits.ndim != 2:
        raise ValueError(f"在线查询 logits 必须是二维张量，实际为 {type(logits).__name__}。")

    batch_size = len(records)
    if logits.shape[0] % batch_size != 0:
        raise ValueError(
            f"query logits 数量无法按 batch 分组: {logits.shape[0]} vs batch={batch_size}"
        )
    view_count = logits.shape[0] // batch_size
    if model_args.eol_type == "prompteol" and view_count != 1:
        raise ValueError(
            f"prompteol 查询每条应产生一个 logits，实际每条产生 {view_count} 个。"
        )
    text_query = args.task_type in {"t2it", "tbpr"} or (
        args.task_type == "ir" and args.query_type == "text"
    )
    topics: dict[str, str] = {}
    for index, record in enumerate(records):
        text = _plain_text(record.get("text", ""))
        if model_args.eol_type == "prompteol":
            current_logits = logits[index]
            if text_query:
                tokens, values = get_text_valid_tokens_values(
                    text,
                    runtime.tokenizer,
                    current_logits,
                    runtime.vocab_dict,
                    data_args,
                    runtime.filtered_ids,
                    model_args,
                )
            else:
                tokens, values = get_img_valid_tokens_values(
                    runtime.image_tokenizer,
                    current_logits,
                    runtime.vocab_dict,
                    data_args,
                    runtime.filtered_ids,
                    model_args,
                )
        else:
            current_logits = logits[index * view_count:(index + 1) * view_count]
            if text_query:
                tokens, values = get_text_valid_disassemble_tokens_values(
                    text,
                    runtime.tokenizer,
                    current_logits,
                    runtime.vocab_dict,
                    data_args,
                    runtime.filtered_ids,
                    None,
                    model_args,
                )
            else:
                tokens, values = get_img_valid_disassemble_tokens_values(
                    runtime.image_tokenizer,
                    current_logits,
                    runtime.vocab_dict,
                    data_args,
                    runtime.filtered_ids,
                    None,
                    model_args,
                )
        vector = _merge_sparse_values(tokens, values, data_args.sparse_value_type)
        if data_args.sparse_value_mean and view_count > 1:
            vector = {token: weight // view_count for token, weight in vector.items()}
        topics[_normalise_id(record["id"])] = _vector_to_topic(vector)
    return topics


def _dense_batch_to_numpy(representations: Any, expected_rows: int):
    import numpy as np
    import torch

    if isinstance(representations, list):
        rows = []
        for item in representations:
            if isinstance(item, list):
                item = torch.stack(item, dim=0).mean(dim=0)
            if torch.is_tensor(item) and item.ndim == 2:
                item = item.mean(dim=0)
            rows.append(item)
        representations = torch.stack(rows, dim=0)
    array = _to_numpy(representations).astype("float32", copy=False)
    if array.ndim == 3:
        array = array.mean(axis=1)
    if array.ndim != 2 or array.shape[0] != expected_rows:
        raise ValueError(
            f"在线 dense query 表征形状错误: shape={array.shape}, expected_rows={expected_rows}"
        )
    return np.asarray(array)


def encode_queries_online(
    args: argparse.Namespace,
    runtime: Any,
    *,
    eol_type: str,
    calculate_type: str,
    sparse_manual: bool,
    progress_prefix: str,
    requested_query_ids: Optional[Sequence[str]] = None,
    timing: Optional[StageTimer] = None,
) -> EncodedQueries:
    import numpy as np
    import torch

    requested_ids_key = (
        None
        if requested_query_ids is None
        else tuple(_normalise_id(query_id) for query_id in requested_query_ids)
    )
    cache_key = (
        args.dataset_name,
        args.dataset_split,
        args.task_type,
        args.query_type,
        eol_type,
        calculate_type,
        sparse_manual,
        args.prompt_type,
        args.sparse_length,
        args.text_sparse_length,
        args.image_sparse_length,
        args.sparse_value_type,
        args.sparse_lower_or_upper,
        args.sparse_value_mean,
        args.num_expended_tokens,
        requested_ids_key,
    )
    native = getattr(runtime, "qwen_family", None) in {"qwen2_vl", "qwen2_5_vl", "qwen3_vl"}
    qwen_metadata = None
    qwen_encoder = sparse_encoder = None
    if native:
        from qwen_vl_encoding import NativeQwenEncoder, SparseEncoder
        from qwen_vl_pipeline import corpus_kind, prompt_metadata
        if args.sparse_logits_multiply_view_count:
            raise ValueError("Qwen does not support the LLaVA-only sparse x5 ablation")
        if getattr(runtime, "qwen_load_settings", _qwen_load_settings(args)) != _qwen_load_settings(args):
            raise ValueError("Qwen model, dtype, adapter or pixel limits changed; reload the retrieval runtime")
        model_args, data_args = _retrieval_namespaces(
            args, eol_type=eol_type, calculate_type=calculate_type, sparse_manual=sparse_manual,
        )
        qwen_metadata = prompt_metadata(
            model_args, data_args, SimpleNamespace(task_type=args.task_type), runtime.processor,
            corpus_kind(args.task_type, args.query_type), family=runtime.qwen_family,
        )
        qwen_metadata["pipeline_settings"] = _qwen_encoding_settings(args, eol_type, calculate_type, sparse_manual)
        cache_key = ("qwen", qwen_metadata["fingerprint"],
                     json.dumps(qwen_metadata["pipeline_settings"], sort_keys=True), requested_ids_key)
        qwen_encoder = NativeQwenEncoder(runtime.encoder, runtime.processor, model_args, data_args, args.task_type)
        sparse_encoder = SparseEncoder(runtime.processor.tokenizer, data_args, concat=calculate_type == "concat")
    cached = runtime.query_cache.get(cache_key)
    if cached is not None:
        return cached

    dataset = build_dataset(args, runtime.processor)
    records = _select_query_records(
        _query_records(args, dataset),
        requested_query_ids,
    )
    if native:
        records = _qwen_query_records(args, records)
    if not native:
        model_args, data_args = _retrieval_namespaces(
            args,
            eol_type=eol_type,
            calculate_type=calculate_type,
            sparse_manual=sparse_manual,
        )
    query_ids: list[str] = []
    dense_parts = []
    topics: dict[str, str] = {}
    batch_starts = range(0, len(records), args.retrieval_batch_size)
    with _compute_scope(timing, "query_encoding", model=getattr(runtime, "encoder", None), device=runtime.device):
        for start in _progress(
            batch_starts,
            desc=f"{progress_prefix} encode queries",
            unit="batch",
            total=len(batch_starts),
        ):
            batch = records[start:start + args.retrieval_batch_size]
            data_args.per_device_batch_size = len(batch)
            amp = (
                torch.autocast(device_type="cuda", dtype=runtime.dtype)
                if runtime.device.type == "cuda" and runtime.dtype != torch.float32
                else nullcontext()
            )
            with torch.inference_mode(), amp:
                if native:
                    logits, dense_representations = qwen_encoder.encode(batch)
                else:
                    logits, dense_representations = _encode_query_batch(
                        runtime, args, batch, model_args, data_args
                    )
            query_ids.extend(_normalise_id(record["id"]) for record in batch)
            dense_parts.append(_dense_batch_to_numpy(dense_representations, len(batch)))
            if native:
                topics.update({record["id"]: _vector_to_topic(sparse_encoder.vector(row, record))
                               for row, record in zip(logits, batch)})
            else:
                topics.update(
                    _batch_sparse_topics(runtime, batch, logits, model_args, data_args, args)
                )
        if not dense_parts:
            raise ValueError("当前数据集没有可编码的查询。")
        encoded = EncodedQueries(
            query_ids=query_ids,
            dense_representations=np.concatenate(dense_parts, axis=0),
            sparse_topics=topics,
            qwen_metadata=qwen_metadata,
        )
        if native:
            encoded.qwen_metadata["query_examples"] = qwen_encoder.last_prompts[:2]
    runtime.query_cache[cache_key] = encoded
    return encoded


def _patch_dataset_paths(data_root: Path) -> Any:
    """Point dataset.py's imported path constants at the requested data root."""

    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    import dataset as dataset_module

    prefix = str(data_root.expanduser().resolve()).rstrip("/\\") + "/"
    dataset_module.dataset_path_prefix = prefix
    directory_names = {
        "coco_file_path": "coco",
        "flickr_file_path": "flickr",
        "fashion_iq_file_path": "fashion-iq",
        "cuhk_pedes_file_path": "CUHK-PEDES",
        "icfg_pedes_flie_path": "ICFG-PEDES",
        "rstpreid_file_path": "RSTPReid",
        "webqa_file_path": "WebQA",
        "remuq_file_path": "UMRB-ReMuQ",
        "llava_file_path": "UMRB-LLaVA",
        "edis_file_path": "EDIS",
    }
    for variable, directory in directory_names.items():
        setattr(dataset_module, variable, prefix + "data/" + directory + "/")
    return dataset_module


def build_dataset(args: argparse.Namespace, processor: Any = None) -> Any:
    dataset_module = _patch_dataset_paths(Path(args.data_root))
    split = args.dataset_split
    task_type = args.task_type
    if task_type == "cir":
        return dataset_module.ComposedTextImageRetrievalDataset(
            args.dataset_name, processor, split, "composed"
        )
    if task_type == "tbpr":
        return dataset_module.TextPersonRetrievalDataset(
            args.dataset_name, processor, split, "full"
        )
    if task_type == "t2it":
        dataset = dataset_module.Text2ImagetextRetrievalDataset(
            args.dataset_name, processor, split, "query"
        )
        _resolve_multimodal_image_paths(dataset, task_type)
        return dataset
    if task_type == "it2t":
        dataset = dataset_module.Imagetext2TextRetrievalDataset(
            args.dataset_name, processor, split, "query"
        )
        _resolve_multimodal_image_paths(dataset, task_type)
        return dataset
    mode = "full" if args.query_type == "text" else "single"
    if task_type not in {"ir", "tbpr"}:
        raise ValueError(f"不支持的 task_type: {task_type}")
    if task_type == "ir":
        return dataset_module.CrossModalRetrievalDataset(
            args.dataset_name, processor, split, mode
        )
    return dataset_module.TextPersonRetrievalDataset(
        args.dataset_name, processor, split, mode
    )


@dataclass(frozen=True)
class RecallEvaluation:
    """CPU-only relevance data, shared by every stage on the main process."""

    k_values: list[int]
    targets: dict[str, set[str]]
    query_categories: dict[str, str]
    candidate_identities: Optional[dict[str, str]]
    dataset_query_count: int


def _target_ids(value: Any) -> set[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        value = [value]
    return {normalized for item in value if (normalized := _normalise_id(item))}


def _build_recall_evaluation(
    args: argparse.Namespace,
    query_ids: Iterable[str],
) -> RecallEvaluation:
    # Also supports importing this entry point through importlib in CPU tests.
    src_dir = str(Path(__file__).resolve().parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from metrices import get_recall_k_settings

    dataset = build_dataset(args)
    raw_query_ids = _dataset_query_ids(args, dataset)
    query_keys = {_normalise_id(query_id): query_id for query_id in raw_query_ids}
    selected_ids = list(dict.fromkeys(_normalise_id(qid) for qid in query_ids))
    if not selected_ids:
        raise ValueError("无法计算召回率：第一阶段没有查询。")

    targets: dict[str, set[str]] = {}
    categories: dict[str, str] = {}
    for query_id in selected_ids:
        if query_id not in query_keys:
            raise ValueError(
                f"无法评估 query={query_id}: 在 {args.dataset_name}/"
                f"{args.dataset_split} 中找不到该查询的真实标签。"
            )
        raw_query_id = query_keys[query_id]
        try:
            if args.task_type == "ir":
                target = dataset.get_target(raw_query_id, args.query_type)
            elif args.task_type == "tbpr":
                target = dataset.get_target_from_text(raw_query_id)
            else:
                target = dataset.get_target(raw_query_id)
            targets[query_id] = _target_ids(target)
            if not targets[query_id]:
                raise ValueError("目标为空")
            if args.dataset_name == "fashion-iq":
                categories[query_id] = dataset.get_dress_type(target)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"无法评估 query={query_id}: 真实标签缺失或无效。") from exc

    identities = None
    if args.task_type == "tbpr":
        identities = {
            _normalise_id(image_id): _normalise_id(dataset.get_target_from_image(image_id))
            for image_id in dataset.img_id_list
        }
        if any(not identity for identity in identities.values()):
            raise ValueError("行人检索图像缺少 person ID，无法计算召回率。")
    # Retain only labels, not the dataset's text/image corpus or processor.
    return RecallEvaluation(
        k_values=get_recall_k_settings(args.dataset_name),
        targets=targets,
        query_categories=categories,
        candidate_identities=identities,
        dataset_query_count=len(query_keys),
    )


def evaluate_recall(
    run: Run,
    evaluation: RecallEvaluation,
    *,
    stage: str,
    variant: str,
) -> list[dict[str, Any]]:
    """Count queries with any relevant top-K hit, including empty/missing runs."""

    unexpected = set(run) - set(evaluation.targets)
    if unexpected:
        raise ValueError(f"{stage}/{variant} 包含未纳入评估的 query: {sorted(unexpected)[0]}")
    groups: dict[str, list[str]] = {"all": list(evaluation.targets)}
    for query_id, category in evaluation.query_categories.items():
        groups.setdefault(category, []).append(query_id)

    first_hit_ranks: dict[str, Optional[int]] = {}
    for query_id, targets in evaluation.targets.items():
        first_hit_ranks[query_id] = None
        for rank, candidate_id in enumerate(
            _top_ids(run.get(query_id, {}), max(evaluation.k_values)), start=1
        ):
            target_id = candidate_id
            if evaluation.candidate_identities is not None:
                try:
                    target_id = evaluation.candidate_identities[candidate_id]
                except KeyError as exc:
                    raise ValueError(
                        f"{stage}/{variant}: 候选图像 {candidate_id} 缺少行人身份标签。"
                    ) from exc
            if target_id in targets:
                first_hit_ranks[query_id] = rank
                break

    rows: list[dict[str, Any]] = []
    for category, query_ids in groups.items():
        counts = [len(run.get(query_id, {})) for query_id in query_ids]
        hits = {
            str(k): sum(
                first_hit_ranks[query_id] is not None and first_hit_ranks[query_id] <= k
                for query_id in query_ids
            )
            for k in evaluation.k_values
        }
        rows.append({
            "stage": stage,
            "variant": variant,
            "category": category,
            "query_count": len(query_ids),
            "min_candidates": min(counts),
            "max_candidates": max(counts),
            "hits": hits,
            "recall": {k: count / len(query_ids) for k, count in hits.items()},
        })
    return rows


class RecallReporter:
    """Print and incrementally persist all completed stages on rank 0 only."""

    def __init__(
        self,
        args: argparse.Namespace,
        query_ids: Iterable[str],
        output_dir: Path,
    ) -> None:
        self.args = args
        self.output_dir = output_dir
        self.evaluation = _build_recall_evaluation(args, query_ids)
        self.rows: list[dict[str, Any]] = []
        self._pandas, self._excel_engine = self._output_dependencies()
        query_count = len(self.evaluation.targets)
        tqdm.write(
            f"[recall] evaluation queries={query_count}/"
            f"{self.evaluation.dataset_query_count} (stage1 query set); "
            "R@K uses all available candidates when K exceeds the returned count."
        )

    @staticmethod
    def _output_dependencies() -> tuple[Any, str]:
        import pandas as pd
        try:
            import openpyxl  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "召回率 Excel 输出需要 openpyxl，请在当前运行环境中安装 openpyxl。"
            ) from exc
        return pd, "openpyxl"

    def record(self, stage: str, runs: Mapping[str, Run]) -> None:
        new_rows = []
        for variant, run in runs.items():
            new_rows.extend(evaluate_recall(
                run, self.evaluation, stage=stage, variant=variant,
            ))
        self.rows.extend(new_rows)
        self._save()
        for row in new_rows:
            metrics = " ".join(
                f"R@{k}={row['recall'][str(k)]:.2%}"
                for k in self.evaluation.k_values
            )
            tqdm.write(
                f"[{stage}][{row['variant']}][{row['category']}] "
                f"queries={row['query_count']} "
                f"candidates={row['min_candidates']}..{row['max_candidates']} | {metrics}"
            )

    def _save(self) -> None:
        payload = {
            "dataset_name": self.args.dataset_name,
            "dataset_split": self.args.dataset_split,
            "task_type": self.args.task_type,
            "query_type": self.args.query_type,
            "recall_k_values": self.evaluation.k_values,
            "recall_scale": "0-1",
            "recall_definition": "fraction of queries with any relevant target in top K",
            "evaluation_query_source": "stage1_run",
            "query_count": len(self.evaluation.targets),
            "dataset_query_count": self.evaluation.dataset_query_count,
            "candidate_policy": "use all available candidates when fewer than K are returned",
            "results": self.rows,
        }
        json_path = self.output_dir / "recall_metrics.json"
        json_temp = self.output_dir / "recall_metrics.tmp.json"
        with json_temp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        json_temp.replace(json_path)

        table = [{
            **{key: row[key] for key in (
                "stage", "variant", "category", "query_count",
                "min_candidates", "max_candidates",
            )},
            **{f"R@{k}": row["recall"][str(k)] for k in self.evaluation.k_values},
        } for row in self.rows]
        excel_temp = self.output_dir / "recall_metrics.tmp.xlsx"
        self._pandas.DataFrame(table).to_excel(
            excel_temp, index=False, engine=self._excel_engine,
        )
        excel_temp.replace(self.output_dir / "recall_metrics.xlsx")


def _resolve_multimodal_image_paths(dataset: Any, task_type: str) -> None:
    """Resolve path-only Parquet images without decoding or copying byte payloads."""

    records = dataset.id2query if task_type == "it2t" else dataset.id2candidate
    base = Path(dataset.data_path)
    for record in records.values():
        source = record["image"]
        if isinstance(source, Mapping):
            if source.get("bytes"):
                continue
            path = source.get("path")
            if not path:
                continue  # The decoder supplies a clear error for invalid images.
            source = dict(source)
            # An empty byte payload must not shadow a usable path in Reranker.
            source["bytes"] = None
            path = Path(path).expanduser()
            source["path"] = str(path if path.is_absolute() else base / path)
            record["image"] = source
        elif isinstance(source, (str, Path)):
            path = Path(source).expanduser()
            record["image"] = str(path if path.is_absolute() else base / path)


def _image_from_value(value: Any):
    from PIL import Image

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, Mapping):
        for key in ("bytes", "data", "contents", "content"):
            payload = value.get(key)
            if payload is not None and len(payload):
                value = BytesIO(bytes(payload))
                break
        else:
            if not value.get("path"):
                raise ValueError("Parquet 图像必须提供非空 bytes 或有效 path。")
            value = Path(value["path"])
    elif isinstance(value, (bytes, bytearray, memoryview)):
        value = BytesIO(bytes(value))
    elif isinstance(value, (str, Path)):
        value = Path(value)
    else:
        raise ValueError(f"不支持的图像输入: {type(value).__name__}；需要 bytes、path 或 PIL 图像。")
    with Image.open(value) as image:
        return image.convert("RGB").copy()


def _blip_preprocess_image(raw_image: Any, image_size: int, device: Any):
    """Match the image preprocessing used by ``encode_dense.py`` for BLIP."""

    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    if not isinstance(raw_image, Image.Image):
        raw_image = _image_from_value(raw_image)
    raw_image = raw_image.convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    return transform(raw_image).unsqueeze(0).to(device)


def _standard_image_path(dataset: Any, image_id: str, data_root: Path) -> Path:
    image_id = _normalise_id(image_id)
    image_name = dataset.img_dict[image_id]
    if dataset.data_name == "coco":
        return data_root / "data" / dataset.data_name / dataset.img2filepath[image_id] / image_name
    if dataset.data_name == "flickr":
        return data_root / "data" / dataset.data_name / "flickr30k-images" / image_name
    if dataset.data_name in {"CUHK-PEDES", "ICFG-PEDES", "RSTPReid"}:
        return data_root / "data" / dataset.data_name / "imgs" / image_name
    if dataset.data_name == "fashion-iq":
        return data_root / "data" / dataset.data_name / "images" / "images" / image_name
    raise ValueError(f"无法推断数据集 {dataset.data_name} 的图像路径。")


def _direct_stage1_items(args: argparse.Namespace, dataset: Any) -> tuple[list[tuple[str, str]], list[tuple[str, Any]]]:
    """Return (query text/image items, corpus text/image items) for CLIP/BLIP."""

    data_root = Path(args.data_root)
    query_items: list[tuple[str, Any]] = []
    corpus_items: list[tuple[str, Any]] = []

    if args.task_type == "cir":
        raise ValueError("CLIP/BLIP 第一阶段暂不直接支持 CIR；请提供 --stage1-run。")

    if args.task_type == "t2it":
        query_items = [(_normalise_id(qid), dataset.id2query[qid]) for qid in dataset.query_id_list]
        corpus_items = [
            (
                _normalise_id(cid),
                _image_from_value(dataset.id2candidate[cid]["image"]),
            )
            for cid in dataset.candidate_id_list
        ]
        return query_items, corpus_items

    if args.task_type == "it2t":
        query_items = [
            (_normalise_id(qid), _image_from_value(dataset.id2query[qid]["image"]))
            for qid in dataset.query_id_list
        ]
        corpus_items = [(_normalise_id(cid), dataset.id2candidate[cid]) for cid in dataset.candidate_id_list]
        return query_items, corpus_items

    if args.task_type == "tbpr":
        query_items = [(_normalise_id(tid), dataset.text_dict[tid]) for tid in dataset.text_id_list]
        corpus_items = [
            (
                _normalise_id(iid),
                _image_from_value(_standard_image_path(dataset, iid, data_root)),
            )
            for iid in dataset.img_id_list
        ]
        return query_items, corpus_items

    if args.query_type == "text":
        query_items = [(_normalise_id(tid), dataset.text_dict[tid]) for tid in dataset.text_id_list]
        corpus_items = [
            (
                _normalise_id(iid),
                _image_from_value(_standard_image_path(dataset, iid, data_root)),
            )
            for iid in dataset.img_id_list
        ]
    else:
        query_items = [
            (
                _normalise_id(iid),
                _image_from_value(_standard_image_path(dataset, iid, data_root)),
            )
            for iid in dataset.img_id_list
        ]
        corpus_items = [(_normalise_id(tid), dataset.text_dict[tid]) for tid in dataset.text_id_list]
    return query_items, corpus_items


def _load_feature_model(
    method: str,
    model_path: str,
    device_name: str,
    dtype_name: str = "float32",
):
    import torch

    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = _retrieval_dtype(dtype_name)
    use_device_map = False
    if method == "clip":
        from transformers import CLIPModel, CLIPProcessor

        # Match encode_dense.py.  Its non-DDP CUDA path loads CLIP with
        # device_map="cuda" and the selected torch dtype.
        load_kwargs = {"torch_dtype": dtype}
        use_device_map = device.type == "cuda" and device_name in {"auto", "cuda"}
        if use_device_map:
            load_kwargs["device_map"] = "cuda"
        model = CLIPModel.from_pretrained(model_path, **load_kwargs)
        processor = CLIPProcessor.from_pretrained(model_path)
    elif method == "blip":
        src_dir = Path(__file__).resolve().parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        try:
            from models.blip_itm import blip_itm
        except ImportError as exc:
            raise RuntimeError(
                "无法导入项目内的 models.blip_itm，无法执行与 encode_dense.py 一致的 BLIP 第一阶段。"
            ) from exc

        checkpoint_path = Path(model_path).expanduser() / "model_large.pth"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "BLIP 模型目录中未找到 model_large.pth："
                f"{checkpoint_path}。three_stage_retrieval_pipeline 与 encode_dense.py "
                "要求 --stage1-model-name-or-path 指向包含该文件的目录。"
            )
        model = blip_itm(pretrained=str(checkpoint_path), vit="large")
        processor = None
    else:
        raise ValueError(f"不支持的第一阶段特征模型：{method}")
    # A model loaded with device_map="cuda" is already dispatched and must
    # not be moved again.  BLIP and explicitly selected devices use .to().
    if not use_device_map:
        model = model.to(device)
    model = model.eval()
    return model, processor, device


def _extract_feature(output: Any, modality: str):
    import torch

    if torch.is_tensor(output):
        feature = output
    else:
        names = (
            ("image_embeds", "pooler_output", "last_hidden_state")
            if modality == "image"
            else ("text_embeds", "pooler_output", "last_hidden_state")
        )
        feature = None
        for name in names:
            candidate = getattr(output, name, None)
            if torch.is_tensor(candidate):
                feature = candidate
                break
        if feature is None and isinstance(output, (tuple, list)):
            feature = next((item for item in output if torch.is_tensor(item)), None)
        if feature is None:
            raise RuntimeError(f"无法从 {type(output).__name__} 中提取 {modality} 表征。")
    if feature.ndim == 3:
        feature = feature.mean(dim=1)
    if feature.ndim != 2:
        raise RuntimeError(f"{modality} 表征应为二维张量，实际 shape={tuple(feature.shape)}")
    return feature


def _clip_text_max_length(model: Any, processor: Any) -> int:
    text_config = getattr(getattr(model, "config", None), "text_config", None)
    model_max_length = getattr(text_config, "max_position_embeddings", None)
    if isinstance(model_max_length, int) and model_max_length > 0:
        return model_max_length

    tokenizer = getattr(processor, "tokenizer", None)
    tokenizer_max_length = getattr(tokenizer, "model_max_length", None)
    if (
        isinstance(tokenizer_max_length, int)
        and 0 < tokenizer_max_length < 1_000_000
    ):
        return tokenizer_max_length
    return 77


def _encode_feature_batch(model: Any, processor: Any, items: Sequence[Any], modality: str, device: Any):
    import torch

    if processor is None:
        if modality == "image":
            with torch.inference_mode():
                raw_images = [
                    _blip_preprocess_image(item, 384, device)
                    for item in items
                ]
                images = torch.cat(raw_images, dim=0)
                image_feat = model.visual_encoder(images)
                return model.vision_proj(image_feat[:, 0, :])
        if modality == "text":
            with torch.inference_mode():
                text_input = model.tokenizer(
                    list(items),
                    padding="max_length",
                    truncation=True,
                    max_length=35,
                    return_tensors="pt",
                ).to(device)
                text_output = model.text_encoder(
                    text_input.input_ids,
                    attention_mask=text_input.attention_mask,
                    mode="text",
                )
                return model.text_proj(text_output.last_hidden_state[:, 0, :])
        raise ValueError(f"不支持的 BLIP 特征模态：{modality}")

    if modality == "image":
        inputs = processor(
            images=list(items),
            return_tensors="pt",
            padding=True,
        )
        feature_method = getattr(model, "get_image_features", None)
        if feature_method is None:
            raise RuntimeError("当前 CLIP/BLIP 模型没有 get_image_features，无法执行直接特征检索。")
    else:
        inputs = processor(
            text=list(items),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=_clip_text_max_length(model, processor),
        )
        feature_method = getattr(model, "get_text_features", None)
        if feature_method is None:
            raise RuntimeError("当前 CLIP/BLIP 模型没有 get_text_features，无法执行直接特征检索。")
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    with torch.inference_mode():
        if modality == "image":
            output = feature_method(inputs["pixel_values"])
        else:
            output = feature_method(
                inputs["input_ids"],
                inputs["attention_mask"],
            )
    return _extract_feature(output, modality).detach().float().cpu()


def _encode_blip_t2it_batch(
    model: Any,
    images: Sequence[Any],
    texts: Sequence[str],
    device: Any,
):
    """Encode t2it corpus items exactly as the BLIP branch in encode_dense.py."""

    import torch
    import torch.nn.functional as F

    with torch.inference_mode():
        image_inputs = [
            _blip_preprocess_image(image, 384, device)
            for image in images
        ]
        image_inputs = torch.cat(image_inputs, dim=0)
        image_feat = model.visual_encoder(image_inputs)
        image_reps = model.vision_proj(image_feat[:, 0, :])

        text_input = model.tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(device)
        text_output = model.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            mode="text",
        )
        text_reps = model.text_proj(text_output.last_hidden_state[:, 0, :])

        image_reps = F.normalize(image_reps, dim=-1)
        text_reps = F.normalize(text_reps, dim=-1)
        return image_reps + text_reps


def _encode_clip_t2it_batch(
    model: Any,
    processor: Any,
    images: Sequence[Any],
    texts: Sequence[str],
    device: Any,
):
    """Encode a CLIP t2it candidate as normalized image + text features."""

    import torch
    import torch.nn.functional as F

    image_inputs = processor(
        images=list(images),
        return_tensors="pt",
        padding=True,
    )
    image_inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in image_inputs.items()
    }

    text_inputs = processor(
        text=list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=_clip_text_max_length(model, processor),
    )
    text_inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in text_inputs.items()
    }

    with torch.inference_mode():
        image_features = model.get_image_features(image_inputs["pixel_values"])
        text_features = model.get_text_features(
            text_inputs["input_ids"],
            text_inputs["attention_mask"],
        )

    return (
        F.normalize(image_features, dim=-1)
        + F.normalize(text_features, dim=-1)
    ).detach().float().cpu()


def retrieve_stage1_feature_model(
    args: argparse.Namespace, method: str, timing: Optional[StageTimer] = None,
) -> Run:
    import numpy as np

    dataset = build_dataset(args)
    query_items, corpus_items = _direct_stage1_items(args, dataset)
    model, processor, device = _load_feature_model(
        method,
        args.stage1_model_name_or_path,
        args.stage1_device,
        args.stage1_dtype,
    )

    # The item type is unambiguous for our dataset adapters: text is str and
    # image is a PIL image.  Keep this explicit for readability and safety.
    if args.task_type == "t2it" or (args.task_type == "ir" and args.query_type == "text") or args.task_type == "tbpr":
        query_modality = "text"
        corpus_modality = "image"
    elif args.task_type == "it2t" or (args.task_type == "ir" and args.query_type == "image"):
        query_modality = "image"
        corpus_modality = "text"
    else:
        raise ValueError("无法根据 task_type/query_type 判断第一阶段的图文方向。")

    query_features = []
    query_ids = []
    query_batch_starts = range(0, len(query_items), args.stage1_batch_size)
    with _compute_scope(timing, "query_encoding", model=model, device=device):
        for start in _progress(
            query_batch_starts,
            desc="[stage1] encode queries",
            unit="batch",
            total=len(query_batch_starts),
        ):
            batch = query_items[start:start + args.stage1_batch_size]
            query_ids.extend(query_id for query_id, _ in batch)
            query_features.append(
                _encode_feature_batch(model, processor, [value for _, value in batch], query_modality, device)
                .detach()
                .cpu()
            )
    corpus_features = []
    corpus_ids = []
    corpus_batch_starts = range(0, len(corpus_items), args.stage1_batch_size)
    with _compute_scope(timing, "corpus_encoding", model=model, device=device):
        for start in _progress(
            corpus_batch_starts,
            desc="[stage1] encode corpus",
            unit="batch",
            total=len(corpus_batch_starts),
        ):
            batch = corpus_items[start:start + args.stage1_batch_size]
            corpus_ids.extend(doc_id for doc_id, _ in batch)
            if method == "blip" and args.task_type == "t2it":
                corpus_features.append(
                    _encode_blip_t2it_batch(
                        model,
                        [value for _, value in batch],
                        [dataset.id2candidate[doc_id]["text"] for doc_id, _ in batch],
                        device,
                    )
                    .detach()
                    .cpu()
                )
            elif method == "clip" and args.task_type == "t2it":
                corpus_features.append(
                    _encode_clip_t2it_batch(
                        model,
                        processor,
                        [value for _, value in batch],
                        [dataset.id2candidate[doc_id]["text"] for doc_id, _ in batch],
                        device,
                    )
                    .detach()
                    .cpu()
                )
            else:
                corpus_features.append(
                    _encode_feature_batch(model, processor, [value for _, value in batch], corpus_modality, device)
                    .detach()
                    .cpu()
                )

    with _compute_scope(timing, "feature_search"):
        query_matrix = _l2_normalize(torch_cat(query_features))
        corpus_matrix = _l2_normalize(torch_cat(corpus_features))
        if query_matrix.shape[1] != corpus_matrix.shape[1]:
            raise ValueError(
                f"第一阶段模型的图像/文本特征维度不一致: {query_matrix.shape[1]} != {corpus_matrix.shape[1]}"
            )

        scores = query_matrix @ corpus_matrix.T
        result: Run = {}
        top_depth = min(args.stage1_depth, len(corpus_ids))
        for row, query_id in enumerate(
            _progress(
                query_ids,
                desc="[stage1] rank queries",
                unit="query",
                total=len(query_ids),
            )
        ):
            order = np.argsort(-scores[row].numpy(), kind="stable")[:top_depth]
            result[query_id] = {corpus_ids[int(index)]: float(scores[row, int(index)]) for index in order}
    del model, processor, query_matrix, corpus_matrix, scores
    gc.collect()
    return result


def torch_cat(tensors: Sequence[Any]):
    import torch

    if not tensors:
        raise ValueError("没有可拼接的特征批次。")
    return torch.cat(list(tensors), dim=0)


def _run_external_command(command: str, args: argparse.Namespace, output_run: Path) -> None:
    substitutions = {
        "output_run": str(output_run),
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "data_root": str(Path(args.data_root).resolve()),
    }
    try:
        formatted = command.format(**substitutions)
    except KeyError as exc:
        raise ValueError(
            f"stage1 command 使用了未知占位符 {exc!s}；可用占位符为 "
            "{output_run}, {dataset_name}, {dataset_split}, {data_root}。"
        ) from exc
    print(f"[stage1] execute: {formatted}")
    subprocess.run(formatted, shell=True, check=True, cwd=str(Path(__file__).resolve().parents[1]))


def _run_sparse_concat_stage1(
    args: argparse.Namespace, context: DistributedContext, timing: StageTimer,
) -> tuple[Run, EncodedQueries, Run, list[int]]:
    """Encode/search each query once and retain its representations locally."""
    import numpy as np

    with timing.measure("query_partition", core=False):
        query_ids, global_ids, counts = _stage1_query_shard(args, context)
    prefix = f"[stage1][rank {context.rank}/{context.world_size}]" if context.is_distributed else "[stage1]"
    tqdm.write(f"{prefix} method=sparse_concat; local_queries={len(query_ids)}; per_rank_queries={counts}")
    runtime = None
    encoded = EncodedQueries([], np.empty((0, 0), dtype="float32"), {})
    timing.components.setdefault("query_encoding", 0.0)
    timing.components.setdefault("sparse_search", 0.0)
    try:
        if query_ids:
            tqdm.write(f"{prefix} loading local query encoder")
            runtime = _load_retrieval_runtime(args, context)
            encoded = encode_queries_online(
                args, runtime, eol_type=STAGE2_EOL_TYPE,
                calculate_type=STAGE2_CALCULATE_TYPE, sparse_manual=False,
                progress_prefix=prefix, requested_query_ids=query_ids, timing=timing,
            )
        _validate_encoded_queries(encoded, query_ids)
        local_run = run_stage1(
            args, runtime, timing=timing, encoded_queries=encoded, progress_prefix=prefix,
        )
        _validate_query_coverage({query_id: {} for query_id in query_ids}, local_run, "Stage 1 local")
        local_run = {query_id: local_run[query_id] for query_id in query_ids}
    finally:
        # Retain only the local CPU EncodedQueries. Clearing the runtime cache
        # must not clear this object, which is handed directly to Stage 2.
        _release_retrieval_runtime(runtime)
    stage1_run = _gather_run(local_run, context)
    if context.is_main:
        _validate_query_coverage({query_id: {} for query_id in global_ids}, stage1_run, "Stage 1")
        stage1_run = {query_id: stage1_run[query_id] for query_id in global_ids}
    return local_run, encoded, stage1_run, counts


def run_stage1(
    args: argparse.Namespace, retrieval_runtime: Any = None,
    timing: Optional[StageTimer] = None,
    *, encoded_queries: Optional[EncodedQueries] = None, progress_prefix: str = "[stage1]",
) -> Run:
    if args.stage1_run:
        run_path = Path(args.stage1_run)
        if timing is not None:
            timing.source = "external_command" if args.stage1_command else "precomputed"
        if args.stage1_command:
            with _compute_scope(timing, "external_command_wall", core=False):
                _run_external_command(args.stage1_command, args, run_path)
        return read_run(run_path)

    if args.stage1_command:
        raise ValueError("提供 --stage1-command 时必须同时提供 --stage1-run 作为输出文件。")

    if args.stage1_method in {"clip", "blip"}:
        if not args.stage1_model_name_or_path:
            raise ValueError("CLIP/BLIP 第一阶段需要 --stage1-model-name-or-path。")
        return retrieve_stage1_feature_model(args, args.stage1_method, timing=timing)

    if args.stage1_sparse_index:
        if encoded_queries is not None and args.stage1_method != "sparse_concat":
            raise ValueError("Preencoded Stage-1 queries require sparse_concat.")
        if retrieval_runtime is None and encoded_queries is None:
            raise ValueError("第一阶段稀疏检索需要在线查询编码模型。")
        if encoded_queries is not None:
            _validate_encoded_queries(encoded_queries)
            encoded = encoded_queries
        elif args.stage1_method == "sparse_manual":
            encoded = encode_queries_online(
                args,
                retrieval_runtime,
                eol_type="prompteol",
                calculate_type="separate",
                sparse_manual=True,
                progress_prefix=progress_prefix,
                timing=timing,
            )
        else:
            encoded = encode_queries_online(
                args,
                retrieval_runtime,
                eol_type=STAGE2_EOL_TYPE,
                calculate_type=STAGE2_CALCULATE_TYPE,
                sparse_manual=False,
                progress_prefix=progress_prefix,
                timing=timing,
            )
        if not encoded.query_ids:
            return {}
        _validate_qwen_artifacts(
            args, encoded, [(args.stage1_sparse_index, False)],
            eol_type="prompteol" if args.stage1_method == "sparse_manual" else STAGE2_EOL_TYPE,
            calculate_type="separate" if args.stage1_method == "sparse_manual" else STAGE2_CALCULATE_TYPE,
            sparse_manual=args.stage1_method == "sparse_manual",
        )
        return retrieve_sparse_topics_from_lucene(
            Path(args.stage1_sparse_index),
            encoded.sparse_topics,
            qids=encoded.query_ids,
            depth=args.stage1_depth,
            threads=args.sparse_threads,
            progress_prefix=progress_prefix,
            timing=timing,
        )
    raise ValueError(
        f"{args.stage1_method} 第一阶段需要 --stage1-run，或提供 --stage1-sparse-index。"
    )


def run_stage2(
    args: argparse.Namespace,
    stage1_run: Run,
    retrieval_runtime: Any = None,
    timing: Optional[StageTimer] = None,
    *, encoded_queries: Optional[EncodedQueries] = None,
) -> tuple[Run, Run, Run]:
    if timing is not None and args.stage2_run:
        timing.source = "precomputed"
    if encoded_queries is not None and not args.stage2_run:
        if args.stage1_method != "sparse_concat":
            raise ValueError("Stage-2 query reuse requires sparse_concat Stage 1.")
        _validate_encoded_queries(encoded_queries, list(stage1_run))
        if timing is not None:
            timing.components.setdefault("query_encoding", 0.0)
    if not stage1_run:
        return {}, {}, {}
    qids = list(stage1_run)
    candidate_filter = {
        query_id: set(_top_ids(scores, args.stage1_depth))
        for query_id, scores in stage1_run.items()
    }

    if args.stage2_run:
        hybrid_run = read_run(Path(args.stage2_run))
        restricted: Run = {}
        for query_id, candidate_ids in _progress(
            candidate_filter.items(),
            desc="[stage2] restrict precomputed run",
            unit="query",
            total=len(candidate_filter),
        ):
            scores = hybrid_run.get(query_id, {})
            restricted[query_id] = {
                doc_id: scores[doc_id]
                for doc_id in _top_ids(scores, args.stage2_depth)
                if doc_id in candidate_ids
            }
        return {}, {}, restricted

    required = {
        "stage2_dense_index": args.stage2_dense_index,
        "stage2_sparse_index": args.stage2_sparse_index,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(
            "第二阶段必须提供当前 disassembleeol_separate+concat 的 dense/sparse artifacts；"
            f"缺少: {', '.join(missing)}"
        )
    if retrieval_runtime is None and encoded_queries is None:
        raise ValueError("第二阶段需要在线查询编码模型。")

    encoded = encoded_queries if encoded_queries is not None else encode_queries_online(
        args,
        retrieval_runtime,
        eol_type=STAGE2_EOL_TYPE,
        calculate_type=STAGE2_CALCULATE_TYPE,
        sparse_manual=False,
        progress_prefix="[stage2]",
        requested_query_ids=qids,
        timing=timing,
    )

    _validate_qwen_artifacts(
        args, encoded, [(args.stage2_dense_index, True), (args.stage2_sparse_index, False)],
        eol_type=STAGE2_EOL_TYPE, calculate_type=STAGE2_CALCULATE_TYPE, sparse_manual=False,
    )

    dense_run = retrieve_dense_from_corpus(
        encoded.dense_representations,
        encoded.query_ids,
        Path(args.stage2_dense_index),
        qids,
        depth=args.stage2_depth,
        candidate_filter=candidate_filter,
        normalize=not args.no_dense_normalize,
        progress_desc="[stage2] dense retrieval",
        timing=timing,
    )
    sparse_run = retrieve_sparse_topics_from_lucene(
        Path(args.stage2_sparse_index),
        encoded.sparse_topics,
        qids,
        depth=max(args.stage2_depth, args.stage1_depth),
        threads=args.sparse_threads,
        progress_prefix="[stage2]",
        timing=timing,
    )
    hybrid_run = fuse_restricted_runs(
        stage1_run,
        dense_run,
        sparse_run,
        alpha=args.hybrid_alpha,
        candidate_depth=args.stage1_depth,
        progress_desc="[stage2] fuse runs",
        timing=timing,
    )
    return dense_run, sparse_run, hybrid_run


def _ranker_dtype(name: str):
    import torch

    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float16


def _load_ranker_model(
    args: argparse.Namespace,
    distributed_context: Optional[DistributedContext] = None,
):
    import torch

    from transformers import (
        AutoModel,
        AutoProcessor,
        LlavaForConditionalGeneration,
        LlavaNextForConditionalGeneration,
        LlavaNextProcessor,
        LlavaProcessor,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VLProcessor,
    )

    context = distributed_context or DistributedContext()
    model_path = args.ranker_model_name_or_path
    dtype = _ranker_dtype(args.ranker_dtype)
    load_kwargs: dict[str, Any] = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    device_map = _device_map_for_process(args.ranker_device_map, context)
    if device_map != "none":
        load_kwargs["device_map"] = device_map

    if _is_qwen_checkpoint(model_path):
        from qwen_vl_pipeline import load_native_qwen, load_qwen_processor
        from qwen_vl_prompts import rerank_template
        config, processor = load_qwen_processor(
            model_path, min_pixels=getattr(args, "ranker_qwen_min_pixels", 0),
            max_pixels=getattr(args, "ranker_qwen_max_pixels", 0),
        )
        if args.rerank_type != "pointwise":
            raise ValueError("Qwen supports pointwise reranking only")
        rerank_template(processor, args.task_type, args.query_type, args.rerank_template,
                        caption="caption_generation" in args.rerank_template)
        device = torch.device("cuda", context.local_rank) if torch.cuda.is_available() else torch.device("cpu")
        model, processor = load_native_qwen(
            model_path, dtype=dtype, device_map=None if device_map == "none" else device_map,
            device=device, config=config, processor=processor,
        )
        processor.tokenizer.padding_side = "left" if args.rerank_reps_loc == "after_pad" else "right"
        return model, processor

    def load(model_class: Any):
        try:
            return model_class.from_pretrained(model_path, **load_kwargs)
        except TypeError:
            fallback = dict(load_kwargs)
            fallback.pop("torch_dtype", None)
            fallback["dtype"] = dtype
            return model_class.from_pretrained(model_path, **fallback)

    if "llava-hf-llava-1.5-7b-hf" in model_path:
        model = load(LlavaForConditionalGeneration)
        processor = LlavaProcessor.from_pretrained(model_path)
    elif "Qwen2.5-VL-7B-Instruct" in model_path or "Qwen2.5-VL-3B-Instruct" in model_path:
        model = load(Qwen2_5_VLForConditionalGeneration)
        processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
    elif "InternVL2_5-8B" in model_path:
        model = AutoModel.from_pretrained(
            model_path,
            **load_kwargs,
            trust_remote_code=True,
            use_flash_attn=True,
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    else:
        model = load(LlavaNextForConditionalGeneration)
        processor = LlavaNextProcessor.from_pretrained(model_path)

    if context.is_distributed and device_map == "none":
        model = model.to(torch.device("cuda", context.local_rank))
    if args.rerank_reps_loc == "after_pad" and hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
        processor.tokenizer.padding = True
    return model.eval(), processor


def _ranker_dataset_maps(dataset: Any, args: argparse.Namespace) -> tuple[Any, Any, Any]:
    if args.task_type == "t2it":
        return dataset.id2query, None, dataset.id2candidate
    if args.task_type == "it2t":
        return dataset.id2candidate, None, dataset.id2query
    if args.task_type == "cir":
        return dataset.text_dict, None, dataset.img_dict
    if args.task_type == "tbpr":
        return dataset.text_dict, None, dataset.img_dict
    img_filepath_map = dataset.img2filepath if args.dataset_name == "coco" else None
    return dataset.text_dict, img_filepath_map, dataset.img_dict


def _native_ranker_run(args: argparse.Namespace, dataset: Any, run: Run) -> Any:
    """Translate persisted string IDs back to native dataset keys for Reranker."""

    if args.task_type not in {"tbpr", "t2it", "it2t"} or not run:
        return run

    def lookup(ids: Iterable[Any], label: str) -> dict[str, Any]:
        result = {}
        for native_id in ids:
            key = _normalise_id(native_id)
            if not key or key in result:
                raise ValueError(f"Stage 3 {args.dataset_name}: {label} ID 缺失或存在歧义: {native_id!r}")
            result[key] = native_id
        return result

    query_keys = lookup(_dataset_query_ids(args, dataset), "query")
    candidate_keys = lookup(
        dataset.img_id_list if args.task_type == "tbpr" else dataset.candidate_id_list,
        "candidate",
    )

    def native_id(keys: Mapping[str, Any], value: Any, label: str) -> Any:
        key = _normalise_id(value)
        if key not in keys:
            raise ValueError(
                f"Stage 3 {args.dataset_name}: 找不到 {label} ID {value!r}；"
                "请检查数据集、split 和候选索引是否匹配。"
            )
        return keys[key]

    return {
        native_id(query_keys, query_id, "query"): {
            native_id(candidate_keys, candidate_id, "candidate"): score
            for candidate_id, score in scores.items()
        }
        for query_id, scores in run.items()
    }


def _dispatch_stage3_rerank(
    ranker: Any,
    hybrid_run: Run,
    args: argparse.Namespace,
    data_args: Any,
    training_args: Any,
    model_args: Any,
    *, progress_observer: Any = None,
) -> Any:
    common_kwargs = {
        "rerank_batch_size": args.rerank_batch_size,
        "rerank_prompt_type": args.rerank_template,
    }
    if progress_observer is not None:
        common_kwargs["progress_observer"] = progress_observer
    if "caption_generation" in args.rerank_template:
        search_args = SimpleNamespace(
            modify_type=args.modify_type,
            use_rerank_prior_cache=getattr(args, "use_rerank_prior_cache", False),
            rerank_prior_cache_dir=getattr(args, "rerank_prior_cache_dir", None),
        )
        if search_args.use_rerank_prior_cache:
            common_kwargs["prior_cache_prepared"] = True
        return ranker.caption_generation_rerank(
            hybrid_run,
            args.rerank_type,
            args.rerank_num,
            data_args,
            training_args,
            model_args,
            search_args,
            **common_kwargs,
        )
    return ranker.rerank(
        hybrid_run,
        args.rerank_type,
        args.rerank_num,
        data_args,
        training_args,
        model_args,
        **common_kwargs,
    )


def run_stage3(
    args: argparse.Namespace,
    hybrid_run: Run,
    distributed_context: Optional[DistributedContext] = None,
    timing: Optional[StageTimer] = None,
    *, progress_observer: Any = None,
) -> Run:
    import torch

    if args.ranker_run:
        if timing is not None:
            timing.source = "precomputed"
        if getattr(args, "use_rerank_prior_cache", False) and (
            distributed_context is None or distributed_context.is_main
        ):
            tqdm.write("[prior-cache] skipped: Stage 3 uses a precomputed ranker run.")
        return read_run(Path(args.ranker_run))
    if not args.ranker_model_name_or_path:
        raise ValueError("第三阶段需要 --ranker-model-name-or-path，或提供 --ranker-run。")
    if args.rerank_type != "pointwise":
        raise ValueError("当前新流水线使用现有 Reranker 的 pointwise 实现，请设置 --rerank-type pointwise。")

    if not hybrid_run and progress_observer is not None:
        # An empty profiling worker still joins control rounds, but needs no model.
        for _ in progress_observer.iterate((), total_items=0, item_count=lambda item: 1):
            pass
        return {}

    # Check native IDs before allocating GPU weights. Keep original dataset keys
    # (also used by prior-cache candidate_key), rather than copying string maps.
    dataset = build_dataset(args)
    native_run = _native_ranker_run(args, dataset, hybrid_run)
    model, processor = _load_ranker_model(args, distributed_context)
    dataset.processor = processor
    text_map, img_filepath_map, img_path_map = _ranker_dataset_maps(dataset, args)

    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    import reranker as reranker_module

    original_path_prefix = reranker_module.path_prefix
    reranker_module.path_prefix = str(Path(args.data_root).expanduser().resolve()).rstrip("/\\") + "/"
    ranker = reranker_module.Reranker(
        model,
        processor,
        args.dataset_name,
        args.query_type,
        text_map,
        img_filepath_map,
        img_path_map,
        getattr(processor, "tokenizer").get_vocab(),
        dataset,
    )
    model_args = SimpleNamespace(model_name_or_path=args.ranker_model_name_or_path)
    data_args = SimpleNamespace(reps_loc=args.rerank_reps_loc, dataset_split=args.dataset_split)
    training_args = SimpleNamespace(
        fp16=args.ranker_dtype == "float16",
        task_type=args.task_type,
    )

    reranked: Any = {}
    try:
        if getattr(args, "use_rerank_prior_cache", False):
            search_args = SimpleNamespace(
                modify_type=args.modify_type, use_rerank_prior_cache=True,
                rerank_prior_cache_dir=args.rerank_prior_cache_dir,
            )
            with timing.measure("prior_cache_prepare", model=model, core=False) if timing is not None else nullcontext():
                prior_report = ranker.prepare_caption_prior_cache(
                    data_args, training_args, model_args, search_args,
                    args.rerank_batch_size, args.rerank_template, args.rerank_type,
                )
            if timing is not None:
                timing.prior_cache = prior_report
        with _compute_scope(timing if hybrid_run else None, "reranking", model=model):
            observer_kwargs = {"progress_observer": progress_observer} if progress_observer is not None else {}
            reranked = _dispatch_stage3_rerank(
                ranker,
                native_run,
                args,
                data_args,
                training_args,
                model_args,
                **observer_kwargs,
            )
    finally:
        reranker_module.path_prefix = original_path_prefix
        del ranker, model, processor, dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        _normalise_id(query_id): _plain_scores(scores)
        for query_id, scores in reranked.items()
    }


def _alpha_validation_args(args: argparse.Namespace) -> Optional[argparse.Namespace]:
    if not getattr(args, "val_passage_reps", None):
        return None
    validation = copy.deepcopy(args)
    validation.dataset_name = args.validation_dataset_name or args.dataset_name
    validation.dataset_split = args.validation_split
    validation.stage1_run = args.val_stage1_run
    validation.stage1_command = None
    validation.stage2_run = None
    validation.ranker_run = None
    validation.stage2_dense_index = args.val_passage_reps
    validation.stage2_sparse_index = args.val_sparse_index
    validation.stage1_sparse_index = args.val_stage1_sparse_index or (
        args.val_sparse_index if args.stage1_method == "sparse_concat" else None
    )
    for name in ("stage1_run", "stage1_sparse_index", "stage2_dense_index", "stage2_sparse_index"):
        value = getattr(validation, name)
        if value:
            setattr(validation, name, str(Path(value).expanduser()))
    return validation


def _validate_alpha_options(args: argparse.Namespace) -> None:
    inputs = [getattr(args, name, None) for name in (
        "val_passage_reps", "val_sparse_index", "val_stage1_run",
        "val_stage1_sparse_index", "validation_dataset_name",
    )]
    if not getattr(args, "select_hybrid_alpha", False):
        if any(inputs):
            raise ValueError("Validation inputs require --select-hybrid-alpha.")
        return
    if args.stage2_run or args.ranker_run:
        raise ValueError("Alpha selection requires computed Stage 2 and reranking; --stage2-run/--ranker-run are unsupported.")
    if not args.stage2_dense_index or not args.stage2_sparse_index:
        raise ValueError("Alpha selection requires both Stage-2 dense and sparse indexes.")
    if bool(args.val_passage_reps) != bool(args.val_sparse_index):
        raise ValueError("Provide both --val-passage-reps and --val-sparse-index, or neither to select on the target split.")
    validation = _alpha_validation_args(args)
    if validation is None:
        if any(inputs):
            raise ValueError("Validation dataset/Stage-1 inputs require both validation Stage-2 indexes.")
        return
    if DATASET_TASK_TYPES.get(validation.dataset_name) != args.task_type:
        raise ValueError("Validation dataset must have the same task type as the target dataset.")
    dense = Path(validation.stage2_dense_index).expanduser()
    if not (dense.is_file() and dense.match("corpus*.pkl")) and not any(path.is_file() for path in dense.glob("corpus*.pkl")):
        raise ValueError(f"No validation corpus*.pkl found: {dense}")
    indexes = [validation.stage2_sparse_index]
    if validation.stage1_run:
        if not Path(validation.stage1_run).expanduser().is_file():
            raise ValueError(f"Validation Stage-1 run does not exist: {validation.stage1_run}")
    elif validation.stage1_method.startswith("sparse"):
        if not validation.stage1_sparse_index:
            raise ValueError("Validation sparse_manual requires --val-stage1-sparse-index.")
        indexes.append(validation.stage1_sparse_index)
    elif not validation.stage1_model_name_or_path:
        raise ValueError("Validation CLIP/BLIP requires a Stage-1 model or --val-stage1-run.")
    for name in indexes:
        index = _resolve_lucene_index(Path(name).expanduser())
        if not any(path.is_file() for path in index.glob("segments_*")):
            raise ValueError(f"No validation Lucene segments_* found: {index}")


def _alpha_main_result(context: DistributedContext, action: Any) -> Any:
    """Broadcast the scalar selection or root evaluation/I/O error, never vectors."""
    packet = [None]
    if context.is_main:
        try:
            packet[0] = {"value": action(), "error": None}
        except Exception as exc:
            packet[0] = {"error": f"{type(exc).__name__}: {exc}"}
    if context.is_distributed:
        import torch.distributed as dist
        dist.broadcast_object_list(packet, src=0)
    if packet[0]["error"] is not None:
        raise ValueError(f"Alpha selection failed: {packet[0]['error']}")
    return packet[0]["value"]


def _alpha_objective(rows: Sequence[dict[str, Any]], dataset_name: str) -> float:
    by_category = {row["category"]: row["recall"] for row in rows}
    try:
        values = (
            [by_category[category][str(k)] for category in ("dress", "shirt", "toptee") for k in (10, 50)]
            if dataset_name == "fashion-iq" else [by_category["all"][str(k)] for k in (1, 5, 10)]
        )
    except KeyError as exc:
        raise ValueError("Alpha selection is missing a required recall cutoff/category.") from exc
    return sum(values) / len(values)


class AlphaSweepReporter:
    """Rank-0 CSV/selection output, incrementally saved before Stage 3."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.rows: list[dict[str, Any]] = []
        self.scans: list[dict[str, Any]] = []
        self.selection: Optional[dict[str, Any]] = None
        self.validation_timing: dict[str, Any] = {}
        self.last_run: Run = {}

    def sweep(
        self, args: argparse.Namespace, stage1: Run, dense: Run, sparse: Run,
        *, role: str, timing: StageTimer,
    ) -> float:
        _validate_query_coverage(stage1, dense, f"{role} dense")
        _validate_query_coverage(stage1, sparse, f"{role} sparse")
        with timing.measure("alpha_evaluation", core=False):
            evaluation = _build_recall_evaluation(args, stage1)
            baselines = [row for name, run in (("dense", dense), ("sparse", sparse))
                         for row in evaluate_recall(run, evaluation, stage="stage2", variant=name)]
        scores = []
        best_score, best_alpha = -1.0, ALPHA_CANDIDATES[0]
        chosen_alpha = self.selection["alpha"] if self.selection else None
        best_run: Run = {}
        for alpha in ALPHA_CANDIDATES:
            run = fuse_restricted_runs(stage1, dense, sparse, alpha, args.stage1_depth,
                                       progress_desc=f"[alpha][{role}] {alpha:.1f}", timing=timing)
            with timing.measure("alpha_evaluation", core=False):
                hybrid_rows = evaluate_recall(run, evaluation, stage="stage2", variant="hybrid")
                score = _alpha_objective(hybrid_rows, args.dataset_name)
                scores.append({"alpha": alpha, "score": score})
                if score > best_score:
                    best_score, best_alpha = score, alpha
                    if chosen_alpha is None:
                        best_run = run
                if chosen_alpha == alpha:
                    best_run = run
                for row in [*baselines, *hybrid_rows]:
                    self.rows.append({
                        "dataset_name": args.dataset_name, "dataset_split": args.dataset_split,
                        "evaluation_role": role, "alpha": alpha, "beta": round(1 - alpha, 1),
                        "variant": row["variant"], "category": row["category"],
                        "query_count": row["query_count"],
                        "selection_score": score if row["variant"] == "hybrid" else "",
                        **{f"R@{k}": recall for k, recall in row["recall"].items()},
                    })
                tqdm.write(f"[alpha][{role}] alpha={alpha:.1f} mean recall={score:.6f}")
        self.scans.append({"role": role, "dataset_name": args.dataset_name,
                           "dataset_split": args.dataset_split, "scores": scores})
        if self.selection is None:
            self.selection = {"alpha": best_alpha, "beta": round(1 - best_alpha, 1), "score": best_score,
                              "source": role, "dataset_name": args.dataset_name,
                              "dataset_split": args.dataset_split}
        self.last_run = best_run
        if role == "validation":
            with timing.measure("alpha_output", core=False):
                directory = self.output_dir / "alpha_validation"
                for name, run in (("stage1", stage1), ("stage2_dense", dense),
                                  ("stage2_sparse", sparse), ("stage2_hybrid", best_run)):
                    write_trec_run(run, directory / f"{name}_run.trec", name)
                    write_json_run(run, directory / f"{name}_run.json")
        with timing.measure("alpha_output", core=False):
            self.save()
        tqdm.write(f"[alpha] selected={self.selection['alpha']:.1f} "
                   f"source={self.selection['source']} {self.selection['dataset_name']}/"
                   f"{self.selection['dataset_split']} mean recall={self.selection['score']:.6f}")
        return self.selection["alpha"]

    def save(self) -> None:
        payload = {"selection": self.selection, "candidates": ALPHA_CANDIDATES,
                   "objective": "mean R@1,R@5,R@10; Fashion-IQ: mean category R@10,R@50",
                   "tie_break": "smallest alpha", "recall_scale": "0-1", "scans": self.scans,
                   "validation_timing": self.validation_timing}
        temporary = self.output_dir / "alpha_selection.tmp.json"
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.output_dir / "alpha_selection.json")
        columns = ["dataset_name", "dataset_split", "evaluation_role", "alpha", "beta", "variant",
                   "category", "query_count", "selection_score", "selected"]
        cutoffs = sorted({int(key[2:]) for row in self.rows for key in row if key.startswith("R@")})
        temporary = self.output_dir / "stage2_alpha_sweep.tmp.csv"
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns + [f"R@{k}" for k in cutoffs])
            writer.writeheader()
            for row in self.rows:
                writer.writerow({**row, "selected": row["alpha"] == self.selection["alpha"]})
        temporary.replace(self.output_dir / "stage2_alpha_sweep.csv")


def _run_alpha_validation(
    args: argparse.Namespace, context: DistributedContext,
) -> tuple[Run, Run, Run, StageTimer, StageTimer]:
    """Execute validation Stage 1/2 once, retaining rank-local query reuse."""
    first, second = StageTimer("precomputed" if args.stage1_run else "computed"), StageTimer()
    runtime, encoded = None, None
    stage1: Run = {}
    try:
        if args.stage1_method == "sparse_concat" and not args.stage1_run:
            local_stage1, encoded, stage1, _ = _run_sparse_concat_stage1(args, context, first)
        else:
            if context.is_main:
                if args.stage1_method == "sparse_manual" and not args.stage1_run:
                    runtime = _load_retrieval_runtime(args, context)
                stage1 = run_stage1(args, runtime, timing=first)
            local_stage1 = _scatter_run(stage1, context)
        if encoded is None and runtime is None and local_stage1:
            runtime = _load_retrieval_runtime(args, context)
        dense, sparse, _ = run_stage2(args, local_stage1, runtime, timing=second, encoded_queries=encoded)
        dense, sparse = _gather_run(dense, context), _gather_run(sparse, context)
        return stage1, dense, sparse, first, second
    finally:
        encoded = None
        _release_retrieval_runtime(runtime)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Stage-1 coarse retrieval -> Stage-2 hybrid retrieval -> Stage-3 ranker."
    )
    parser.add_argument(
        "--dataset-name", required=True,
        help="Dataset key: coco, flickr, CUHK-PEDES, ICFG-PEDES, RSTPReid, webqa, remuq or llava.",
    )
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--task-type", choices=("ir", "tbpr", "t2it", "it2t", "cir"), default="ir")
    parser.add_argument(
        "--query-type", choices=("text", "image", "composed"), default="text",
        help="TBPR/WebQA use text; ReMuQ/LLaVA use image (or composed), always including query text.",
    )
    parser.add_argument("--data-root", default="/root/autodl-tmp")
    parser.add_argument(
        "--local_rank",
        "--local-rank",
        dest="local_rank",
        type=int,
        default=-1,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--stage1-method",
        choices=STAGE1_METHODS,
        default="sparse_concat",
        help="Stage-1 coarse retriever. sparse_manual=prompteol+separate+sparse_manual; "
             "sparse_concat=disassembleeol_separate+concat.",
    )
    parser.add_argument("--stage1-run", help="Existing TREC/JSON/JSONL/pickle run; skips Stage-1 computation.")
    parser.add_argument(
        "--stage1-command",
        help="Optional command that writes --stage1-run. Supports "
             "{output_run}, {dataset_name}, {dataset_split}, {data_root}.",
    )
    parser.add_argument(
        "--stage1-model-name-or-path",
        help="CLIP checkpoint or BLIP directory containing model_large.pth for direct Stage 1.",
    )
    parser.add_argument("--stage1-device", default="auto", help="Stage-1 device, e.g. auto, cuda, cuda:0, cpu.")
    parser.add_argument(
        "--stage1-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float32",
        help=(
            "CLIP loading dtype, corresponding to encode_dense.py's "
            "--fp16/--bf16 settings. BLIP remains float32 like encode_dense.py."
        ),
    )
    parser.add_argument("--stage1-batch-size", type=int, default=32)
    parser.add_argument(
        "--stage1-sparse-index",
        help="Corpus-side sparse output directory produced by encode/indexing, or its Lucene index directory.",
    )
    parser.add_argument("--stage1-depth", type=int, default=1000)

    parser.add_argument(
        "--stage2-run",
        help="Optional precomputed Stage-2 hybrid run. If provided, dense/sparse Stage-2 artifacts are not loaded.",
    )
    parser.add_argument("--stage2-dense-index", help="Stage-2 passage_reps directory containing corpus*.pkl.")
    parser.add_argument("--stage2-sparse-index", help="Stage-2 corpus-side sparse directory containing index/.")
    parser.add_argument("--stage2-depth", type=int, default=1000)
    parser.add_argument("--hybrid-alpha", type=float, default=0.5, help="Dense weight in Stage-2 hybrid fusion.")
    parser.add_argument("--select-hybrid-alpha", action="store_true", help="Scan 0.1..0.9; select on validation indexes, or the target split when neither is supplied.")
    parser.add_argument("--val-passage-reps", help="Validation dense corpus*.pkl directory (paired with --val-sparse-index).")
    parser.add_argument("--val-sparse-index", help="Validation sparse corpus directory or Lucene index/.")
    parser.add_argument("--validation-dataset-name", help="Defaults to the target dataset; must have the same task type.")
    parser.add_argument("--validation-split", default="val")
    parser.add_argument("--val-stage1-sparse-index", help="Required for validation sparse_manual; sparse_concat defaults to --val-sparse-index.")
    parser.add_argument("--val-stage1-run", help="Optional validation Stage-1 precomputed run, separate from the target run.")
    parser.add_argument("--sparse-threads", type=int, default=16)
    parser.add_argument("--no-dense-normalize", action="store_true")

    parser.add_argument(
        "--retrieval-model-name-or-path",
        help="VLM checkpoint used to encode sparse/Stage-2 queries online, matching search.py.",
    )
    parser.add_argument("--retrieval-lora-model-path")
    parser.add_argument("--retrieval-device", default="auto")
    parser.add_argument("--retrieval-device-map", default="auto")
    parser.add_argument(
        "--retrieval-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--retrieval-batch-size", type=int, default=1)
    parser.add_argument(
        "--retrieval-reps-loc",
        choices=("before_pad", "after_pad"),
        default="after_pad",
    )
    parser.add_argument("--prompt-type", default="prompt_5")
    parser.add_argument("--retrieval-qwen-min-pixels", type=int, default=0, help="Qwen query image minimum pixel area; 0 uses checkpoint defaults.")
    parser.add_argument("--retrieval-qwen-max-pixels", type=int, default=0, help="Qwen query image maximum pixel area; 0 uses checkpoint defaults.")
    parser.add_argument("--random-view-seed", type=int, default=42)
    parser.add_argument("--sparse-length", type=int, default=128)
    parser.add_argument("--text-sparse-length", type=int, default=128)
    parser.add_argument("--image-sparse-length", type=int, default=128)
    parser.add_argument("--num-expended-tokens", type=int, default=0)
    parser.add_argument("--sparse-value-type", choices=("replace", "sum", "max"), default="replace")
    parser.add_argument("--sparse-lower-or-upper", choices=("lower", "upper"), default="lower")
    parser.add_argument("--sparse-value-mean", action="store_true")
    parser.add_argument("--sparse-logits-multiply-view-count", action="store_true")
    parser.add_argument("--is-filtered", action="store_true")
    parser.add_argument("--print-sparse", action="store_true")
    parser.add_argument("--tbpr-type", default="origin_type", help="TBPR views: origin_type or type; must match the candidate index.")
    parser.add_argument("--cir-type", default="type")

    parser.add_argument("--ranker-run", help="Optional precomputed Stage-3 run; skips model loading.")
    parser.add_argument("--ranker-model-name-or-path", help="Current VLM checkpoint used by Reranker.")
    parser.add_argument("--ranker-device-map", default="auto")
    parser.add_argument("--ranker-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--ranker-qwen-min-pixels", type=int, default=0, help="Qwen reranker image minimum pixel area; 0 uses checkpoint defaults.")
    parser.add_argument("--ranker-qwen-max-pixels", type=int, default=0, help="Qwen reranker image maximum pixel area; 0 uses checkpoint defaults.")
    parser.add_argument("--rerank-type", choices=("pointwise", "listwise"), default="pointwise")
    parser.add_argument("--rerank-num", type=int, default=20)
    parser.add_argument("--rerank-batch-size", type=int, default=1)
    parser.add_argument("--rerank-template", default="relevant")
    parser.add_argument("--use-rerank-prior-cache", "--use_rerank_prior_cache", action="store_true")
    parser.add_argument("--rerank-prior-cache-dir", "--rerank_prior_cache_dir", default=None)
    parser.add_argument(
        "--modify_type",
        "--modify-type",
        dest="modify_type",
        choices=("no", "modify_single"),
        default="no",
        help="Caption-generation rerank score modification, matching rerank.py.",
    )
    parser.add_argument("--rerank-reps-loc", choices=("before_pad", "after_pad"), default="after_pad")

    parser.add_argument("--output-dir", default="./three_stage_results")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    _validate_alpha_options(args)
    for prefix in ("retrieval", "ranker"):
        low, high = (getattr(args, f"{prefix}_qwen_{edge}_pixels", 0) for edge in ("min", "max"))
        if min(low, high) < 0 or (low and high and low > high):
            raise ValueError(f"Invalid {prefix} Qwen pixel limits: require nonnegative values and min <= max")
    expected_task = DATASET_TASK_TYPES.get(args.dataset_name)
    if expected_task is not None and args.task_type != expected_task:
        raise ValueError(
            f"{args.dataset_name} 需要 --task-type {expected_task}，实际为 {args.task_type}。"
        )
    query_types = {
        "ir": {"image", "text"}, "tbpr": {"text"}, "t2it": {"text"},
        "it2t": {"image", "composed"}, "cir": {"composed"},
    }
    if args.query_type not in query_types[args.task_type]:
        allowed = "/".join(sorted(query_types[args.task_type]))
        raise ValueError(f"{args.task_type} 请设置 --query-type {allowed}。")
    if args.task_type == "tbpr" and args.tbpr_type not in {"origin_type", "type"}:
        raise ValueError("TBPR 请设置 --tbpr-type origin_type 或 type，并与候选编码设置一致。")
    if getattr(args, "use_rerank_prior_cache", False) and not args.ranker_run:
        validate_prior_cache_options(True, args.modify_type, args.rerank_template,
                                     args.task_type, args.query_type, args.rerank_type)
    positive = {
        "stage1_depth": args.stage1_depth,
        "stage2_depth": args.stage2_depth,
        "stage1_batch_size": args.stage1_batch_size,
        "retrieval_batch_size": args.retrieval_batch_size,
        "sparse_length": args.sparse_length,
        "text_sparse_length": args.text_sparse_length,
        "image_sparse_length": args.image_sparse_length,
        "rerank_num": args.rerank_num,
        "rerank_batch_size": args.rerank_batch_size,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} 必须为正整数。")
    needs_online_stage1 = (
        args.stage1_method in {"sparse_manual", "sparse_concat"}
        and not args.stage1_run
    )
    needs_online_stage2 = not args.stage2_run
    if (needs_online_stage1 or needs_online_stage2) and not args.retrieval_model_name_or_path:
        raise ValueError(
            "在线稀疏/混合查询编码需要 --retrieval-model-name-or-path；"
            "encode.py 只提供 corpus 侧索引。"
        )
    if args.task_type == "cir" and args.query_type != "composed":
        raise ValueError("CIR 请设置 --query-type composed。")
    if args.modify_type != "no" and "caption_generation" not in args.rerank_template:
        raise ValueError(
            "--modify_type=modify_single 仅适用于 caption_generation rerank template。"
        )
    is_main_process = int(os.environ.get("RANK", "0")) == 0
    if is_main_process and args.stage1_method in {"sparse_manual", "sparse_concat"}:
        expected = (
            "prompteol + separate + sparse_manual"
            if args.stage1_method == "sparse_manual"
            else "disassembleeol_separate + concat"
        )
        print(f"Stage 1 sparse mode: {expected}")
    if is_main_process:
        print(f"Stage 2 fixed mode: {STAGE2_EOL_TYPE} + {STAGE2_CALCULATE_TYPE}")
    if args.output_dir and Path(args.output_dir).exists() and not args.overwrite:
        raise FileExistsError(
            f"输出目录已存在: {args.output_dir}。请更换目录或添加 --overwrite。"
        )


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        _validate_args(args)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))

    wall_started = time.perf_counter()
    distributed_context = _initialise_distributed(
        getattr(args, "local_rank", -1)
    )
    pipeline_progress = (
        _progress(
            desc="[pipeline] stages",
            unit="stage",
            total=3,
        )
        if distributed_context.is_main
        else None
    )
    completed = False
    timing_reporter = None
    active_stage = None
    stage1_timer = StageTimer(
        "external_command" if getattr(args, "stage1_command", None)
        else "precomputed" if getattr(args, "stage1_run", None) else "computed"
    )
    stage2_timer = StageTimer("precomputed" if getattr(args, "stage2_run", None) else "computed")
    stage3_timer = StageTimer("precomputed" if getattr(args, "ranker_run", None) else "computed")
    try:
        output_dir = Path(args.output_dir).expanduser().resolve()
        if distributed_context.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            timing_reporter = TimingReporter(args, output_dir, wall_started)

        select_alpha = getattr(args, "select_hybrid_alpha", False)
        alpha_reporter = AlphaSweepReporter(output_dir) if select_alpha and distributed_context.is_main else None
        if select_alpha:
            # The requested fixed weight is irrelevant in selection mode.
            args.hybrid_alpha = 0.5
            validation_args = _alpha_validation_args(args)
            if validation_args is not None:
                active_stage = "stage2"
                validation_stage1, validation_dense, validation_sparse, validation_first, validation_second = _run_alpha_validation(
                    validation_args, distributed_context,
                )
                args.hybrid_alpha = _alpha_main_result(distributed_context, lambda: alpha_reporter.sweep(
                    validation_args, validation_stage1, validation_dense, validation_sparse,
                    role="validation", timing=validation_second,
                ))
                measurements = {"stage1": _collect_stage_timing(validation_first, distributed_context),
                                "stage2": _collect_stage_timing(validation_second, distributed_context)}
                def save_validation_timing():
                    alpha_reporter.validation_timing = measurements
                    alpha_reporter.save()
                    alpha_reporter.last_run = {}
                _alpha_main_result(distributed_context, save_validation_timing)
                del validation_stage1, validation_dense, validation_sparse

        retrieval_runtime = None
        local_encoded = None
        parallel_stage1 = args.stage1_method == "sparse_concat" and not args.stage1_run
        stage1_rank_counts: list[int] = []
        reuse_stage1_queries = (
            parallel_stage1 and not args.stage2_run
        )
        recall_reporter = None
        stage1_run: Run = {}
        try:
            active_stage = "stage1"
            if parallel_stage1:
                local_stage1_run, local_encoded, stage1_run, stage1_rank_counts = _run_sparse_concat_stage1(
                    args, distributed_context, stage1_timer,
                )
                if not reuse_stage1_queries:
                    local_encoded = None
            else:
                if distributed_context.is_main:
                    if args.stage1_method == "sparse_manual" and not args.stage1_run:
                        tqdm.write("[query encoder] loading retrieval VLM for stage 1")
                        retrieval_runtime = _load_retrieval_runtime(args, distributed_context)
                    tqdm.write(f"[stage1] method={args.stage1_method}")
                    stage1_run = run_stage1(args, retrieval_runtime, timing=stage1_timer)
                local_stage1_run = _scatter_run(stage1_run, distributed_context)
            if distributed_context.is_main:
                write_trec_run(
                    stage1_run,
                    output_dir / "stage1_run.trec",
                    args.stage1_method,
                )
                write_json_run(stage1_run, output_dir / "stage1_run.json")
                recall_reporter = RecallReporter(args, stage1_run, output_dir)
                recall_reporter.record("stage1", {args.stage1_method: stage1_run})
            _record_stage_timing("stage1", stage1_timer, distributed_context, timing_reporter)
            active_stage = None
            if pipeline_progress is not None:
                pipeline_progress.update(1)

            active_stage = "stage2"
            if reuse_stage1_queries:
                if distributed_context.is_main:
                    tqdm.write(
                        f"[stage2] reused Stage-1 query encodings: {len(stage1_run)} queries; "
                        "rank-local reuse, no vector transfer or query model loading/encoding"
                    )
            if not args.stage2_run and local_encoded is None and retrieval_runtime is None and local_stage1_run:
                if distributed_context.is_main:
                    tqdm.write("[query encoder] loading retrieval VLM for stage 2")
                retrieval_runtime = _load_retrieval_runtime(
                    args,
                    distributed_context,
                )

            if distributed_context.is_main:
                tqdm.write("[stage2] distributed restricted hybrid retrieval")
            local_dense_run, local_sparse_run, local_stage2_run = run_stage2(
                args,
                local_stage1_run,
                retrieval_runtime,
                timing=stage2_timer,
                encoded_queries=local_encoded,
            )
            dense_run = _gather_run(local_dense_run, distributed_context)
            sparse_run = _gather_run(local_sparse_run, distributed_context)
            stage2_run = _gather_run(local_stage2_run, distributed_context) if not select_alpha else {}
            if select_alpha:
                args.hybrid_alpha = _alpha_main_result(distributed_context, lambda: alpha_reporter.sweep(
                    args, stage1_run, dense_run, sparse_run, role="target", timing=stage2_timer,
                ))
                # Keep the original query ownership; no vector/run repartition is needed.
                local_stage2_run = fuse_restricted_runs(
                    local_stage1_run, local_dense_run, local_sparse_run, args.hybrid_alpha,
                    args.stage1_depth, timing=stage2_timer,
                )
                if distributed_context.is_main:
                    stage2_run = alpha_reporter.last_run
            if distributed_context.is_main:
                _validate_query_coverage(stage1_run, stage2_run, "Stage 2")
                if dense_run:
                    write_trec_run(
                        dense_run,
                        output_dir / "stage2_dense_run.trec",
                        "stage2_dense",
                    )
                if sparse_run:
                    write_trec_run(
                        sparse_run,
                        output_dir / "stage2_sparse_run.trec",
                        "stage2_sparse",
                    )
                write_trec_run(
                    stage2_run,
                    output_dir / "stage2_hybrid_run.trec",
                    "stage2_hybrid",
                )
                write_json_run(stage2_run, output_dir / "stage2_hybrid_run.json")
                stage2_metrics_runs = {}
                if dense_run:
                    stage2_metrics_runs["dense"] = dense_run
                if sparse_run:
                    stage2_metrics_runs["sparse"] = sparse_run
                stage2_metrics_runs["hybrid"] = stage2_run
                assert recall_reporter is not None
                recall_reporter.record("stage2", stage2_metrics_runs)
            if pipeline_progress is not None:
                pipeline_progress.update(1)
        finally:
            local_encoded = None
            _release_retrieval_runtime(retrieval_runtime)

        _record_stage_timing("stage2", stage2_timer, distributed_context, timing_reporter)
        active_stage = "stage3"
        if distributed_context.is_main:
            tqdm.write("[stage3] distributed ranker reranking")
        if args.ranker_run and distributed_context.is_distributed:
            precomputed_stage3 = (
                read_run(Path(args.ranker_run))
                if distributed_context.is_main
                else {}
            )
            local_stage3_run = _scatter_run(
                precomputed_stage3,
                distributed_context,
            )
        else:
            local_stage3_run = run_stage3(
                args,
                local_stage2_run,
                distributed_context,
                timing=stage3_timer,
            )
        stage3_run = _gather_run(local_stage3_run, distributed_context)
        if distributed_context.is_main:
            _validate_query_coverage(stage1_run, stage3_run, "Stage 3")
            write_trec_run(
                stage3_run,
                output_dir / "stage3_ranker_run.trec",
                "stage3_ranker",
            )
            write_json_run(stage3_run, output_dir / "stage3_ranker_run.json")
            assert recall_reporter is not None
            recall_reporter.record("stage3", {"ranker": stage3_run})
        _record_stage_timing("stage3", stage3_timer, distributed_context, timing_reporter)
        active_stage = None
        if pipeline_progress is not None:
            pipeline_progress.update(1)

        if distributed_context.is_main:
            metadata = {
                "dataset_name": args.dataset_name,
                "dataset_split": args.dataset_split,
                "task_type": args.task_type,
                "query_type": args.query_type,
                "timing": {
                    "unit": "seconds",
                    "stage_scope": "core_compute_excluding_setup_evaluation_output",
                    "distributed_reduction": "max_rank_accumulated_compute_per_stage",
                    "files": ["timing_metrics.json", "timing_metrics.xlsx"],
                },
                "recall": {
                    "k_values": recall_reporter.evaluation.k_values,
                    "scale": "0-1",
                    "evaluation_query_source": "stage1_run",
                    "query_count": len(recall_reporter.evaluation.targets),
                    "dataset_query_count": recall_reporter.evaluation.dataset_query_count,
                },
                "distributed": {
                    "world_size": distributed_context.world_size,
                    "stage1_query_sharded": parallel_stage1 and distributed_context.is_distributed,
                    "stage2_query_sharded": distributed_context.is_distributed,
                    "stage3_query_sharded": distributed_context.is_distributed,
                },
                "stage1": {
                    "method": args.stage1_method,
                    "per_rank_query_counts": {str(rank): count for rank, count in enumerate(stage1_rank_counts)},
                    "query_encoding": "online" if args.stage1_method.startswith("sparse") and not args.stage1_run else "precomputed_or_direct",
                    "retrieval_model_name_or_path": args.retrieval_model_name_or_path,
                    "sparse_index": args.stage1_sparse_index,
                    "eol_type": "prompteol" if args.stage1_method == "sparse_manual" else (
                        STAGE2_EOL_TYPE if args.stage1_method == "sparse_concat" else None
                    ),
                    "calculate_type": "separate" if args.stage1_method == "sparse_manual" else (
                        STAGE2_CALCULATE_TYPE if args.stage1_method == "sparse_concat" else None
                    ),
                    "sparse_manual": args.stage1_method == "sparse_manual",
                    "query_count": len(stage1_run),
                },
                "stage2": {
                    "query_encoding": (
                        "reused_stage1" if reuse_stage1_queries else
                        "online" if not args.stage2_run else "precomputed_run"
                    ),
                    "reused_query_count": len(stage1_run) if reuse_stage1_queries else 0,
                    "retrieval_model_name_or_path": args.retrieval_model_name_or_path,
                    "dense_corpus_index": args.stage2_dense_index,
                    "sparse_corpus_index": args.stage2_sparse_index,
                    "eol_type": STAGE2_EOL_TYPE,
                    "calculate_type": STAGE2_CALCULATE_TYPE,
                    "alpha": args.hybrid_alpha,
                    "candidate_depth": args.stage1_depth,
                    "query_count": len(stage2_run),
                },
                "stage3": {
                    "ranker_model_name_or_path": args.ranker_model_name_or_path,
                    "rerank_type": args.rerank_type,
                    "rerank_num": args.rerank_num,
                    "rerank_template": args.rerank_template,
                    "modify_type": args.modify_type,
                    "query_count": len(stage3_run),
                },
                "files": [
                    "stage1_run.trec",
                    "stage1_run.json",
                    "stage2_hybrid_run.trec",
                    "stage2_hybrid_run.json",
                    "stage3_ranker_run.trec",
                    "stage3_ranker_run.json",
                    "recall_metrics.json",
                    "recall_metrics.xlsx",
                    "timing_metrics.json",
                    "timing_metrics.xlsx",
                ],
            }
            if select_alpha:
                metadata["stage2"]["alpha_selection"] = alpha_reporter.selection
                metadata["files"].extend(["stage2_alpha_sweep.csv", "alpha_selection.json"])
            with (output_dir / "pipeline_metadata.json").open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, ensure_ascii=False, indent=2)
            assert timing_reporter is not None
            timing_reporter.finish(completed=True)
            tqdm.write(f"Pipeline completed. Results: {output_dir}")
        completed = True
    except BaseException:
        if timing_reporter is not None:
            try:
                timing_reporter.finish(completed=False, failed_stage=active_stage)
            except Exception as timing_error:
                # In particular, do not mask a retrieval error with an Excel
                # dependency/I/O error. JSON is always attempted first.
                tqdm.write(f"[timing] Could not finalize timing report: {timing_error}")
        # Preserve the original traceback, including interrupted runs.
        raise
    finally:
        if pipeline_progress is not None:
            pipeline_progress.close()
        _shutdown_distributed(distributed_context, synchronize=completed)


if __name__ == "__main__":
    main()
