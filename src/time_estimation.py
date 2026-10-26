"""Shared, lightweight throughput estimation; no model or GPU imports at import time."""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


class StableRateEstimator:
    """Batch-boundary throughput sampling, independent of tqdm's EMA refreshes.

    The first 20 batches still execute; only their rates are excluded. A slow
    batch yields one sample over its actual duration, never synthetic samples
    at elapsed five-second boundaries. Callers must report completed work:
    FreeRet returns CPU tensors; the stage observer synchronizes its GPU.
    """

    def __init__(
        self,
        total_items: int,
        *,
        clock: Optional[Callable[[], float]] = None,
        warmup_batches: int = 20,
        sample_interval_seconds: float = 5.0,
        stable_samples: int = 5,
        max_cv: float = 0.05,
    ) -> None:
        if total_items < 0 or warmup_batches < 0 or stable_samples < 2:
            raise ValueError("Invalid item, warmup, or stable-sample count")
        if not math.isfinite(sample_interval_seconds) or sample_interval_seconds <= 0:
            raise ValueError("Sampling interval must be finite and positive")
        if not math.isfinite(max_cv) or max_cv <= 0:
            raise ValueError("CV threshold must be finite and positive")
        self.total_items = total_items
        self.warmup_batches = warmup_batches
        self.sample_interval_seconds = sample_interval_seconds
        self.stable_samples = stable_samples
        self.max_cv = max_cv
        self.clock = clock if clock is not None else time.perf_counter
        self.started_at = self.clock()
        self.last_sample_at = self.started_at
        self.last_sample_items = 0
        self.completed_items = 0
        self.completed_batches = 0
        self.num_rate_samples = 0
        self.rates: deque[float] = deque(maxlen=stable_samples)
        self.stable_estimate: Optional[dict[str, Any]] = None

    def _window_stats(self) -> tuple[Optional[float], Optional[float]]:
        if not self.rates:
            return None, None
        mean = sum(rate / len(self.rates) for rate in self.rates)
        # Scale before squaring so very large finite rates cannot overflow.
        cv = math.sqrt(sum((rate / mean - 1.0) ** 2 for rate in self.rates) / len(self.rates))
        return mean, cv

    def observe_batch(self, completed_items: int) -> bool:
        """Return True only when this batch first establishes stability."""
        if not self.completed_items < completed_items <= self.total_items:
            raise ValueError("Each completed batch must advance the item count")
        self.completed_items = completed_items
        self.completed_batches += 1
        if self.stable_estimate is not None:
            return False
        now = self.clock()
        if not math.isfinite(now):
            self.rates.clear()
            return False
        if self.completed_batches <= self.warmup_batches:
            self.last_sample_at = now
            self.last_sample_items = completed_items
            return False
        interval = now - self.last_sample_at
        if interval < self.sample_interval_seconds:
            return False
        rate = (completed_items - self.last_sample_items) / interval
        self.last_sample_at = now
        self.last_sample_items = completed_items
        if not math.isfinite(rate) or rate <= 0:
            self.rates.clear()
            return False
        self.rates.append(rate)
        self.num_rate_samples += 1
        mean, cv = self._window_stats()
        assert mean is not None and cv is not None
        if len(self.rates) == self.stable_samples and cv < self.max_cv:
            elapsed = now - self.started_at
            remaining = (self.total_items - completed_items) / mean
            self.stable_estimate = {
                "elapsed_seconds": elapsed,
                "remaining_seconds": remaining,
                "estimated_total_seconds": elapsed + remaining,
                "completed_items": completed_items,
                "completed_batches": self.completed_batches,
                "rate_per_second": mean,
                "rate_samples": list(self.rates),
                "rate_cv": cv,
            }
            return True
        return False

    def result(self, *, completed: bool) -> dict[str, Any]:
        if completed and self.completed_items != self.total_items:
            raise ValueError("A completed stage must have processed all items")
        if not completed and self.stable_estimate is None:
            raise ValueError("Cannot estimate an incomplete, unstable stage")
        mean, cv = self._window_stats()
        elapsed = (
            self.clock() - self.started_at
            if completed
            else self.stable_estimate["elapsed_seconds"]
        )
        remaining = 0.0 if completed else self.stable_estimate["remaining_seconds"]
        return {
            "status": "completed" if completed else "estimated",
            "elapsed_seconds": elapsed,
            "remaining_seconds": remaining,
            "estimated_total_seconds": elapsed + remaining,
            "total_items": self.total_items,
            "completed_items": self.completed_items,
            "completed_batches": self.completed_batches,
            "stabilized": self.stable_estimate is not None,
            "rate_per_second": mean,
            "rate_samples": list(self.rates),
            "rate_cv": cv if len(self.rates) == self.stable_samples else None,
            "num_rate_samples": self.num_rate_samples,
            "stable_estimate": self.stable_estimate,
        }


@dataclass
class RerankTimeEstimationArguments:
    estimate_time_only: bool = False

    def validate(self, model_args: Any, search_args: Any = None) -> None:
        if self.estimate_time_only and (
            model_args.eol_type != "disassembleeol_separate"
            or model_args.calculate_type != "concat"
        ):
            raise ValueError(
                "--estimate_time_only currently requires "
                "--eol_type disassembleeol_separate --calculate_type concat"
            )
        if (self.estimate_time_only and search_args is not None
                and "caption_generation" not in search_args.rerank_template
                and search_args.rerank_type != "pointwise"):
            raise ValueError(
                "--estimate_time_only requires an implemented query-level reranker; "
                "the relevant-template listwise reranker is currently a placeholder"
            )


class TorchStageCollectives:
    """Small control messages only; never gather embeddings or ranked results."""

    def __init__(self, device: Any = None) -> None:
        import torch
        import torch.distributed as dist

        self.torch = torch
        self.dist = dist
        self.distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.device = torch.device(device or "cpu")
        if not self.distributed or "nccl" not in str(dist.get_backend()).lower():
            self.device = torch.device("cpu")
        elif self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())

    def barrier(self) -> None:
        if self.distributed:
            if self.device.type == "cuda":
                self.dist.barrier(device_ids=[self.device.index])
            else:
                self.dist.barrier()

    def all_done(self, done: bool) -> bool:
        if not self.distributed:
            return done
        flag = self.torch.tensor(int(done), device=self.device, dtype=self.torch.int32)
        self.dist.all_reduce(flag, op=self.dist.ReduceOp.MIN)
        return bool(flag.item())

    def gather(self, report: dict[str, Any]) -> list[dict[str, Any]]:
        if not self.distributed:
            return [report]
        reports: list[Any] = [None] * self.world_size
        self.dist.all_gather_object(reports, report)
        return reports


def merge_stage_time_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Parallel wall time is the slowest rank, not the sum of GPU times."""
    if not reports:
        raise ValueError("No rank timing reports were supplied")
    return {
        "status": "estimated" if any(r["status"] == "estimated" for r in reports) else "completed",
        "elapsed_seconds": max(r["elapsed_seconds"] for r in reports),
        "estimated_total_seconds": max(r["estimated_total_seconds"] for r in reports),
        "total_items_including_sampler_padding": sum(r["total_items"] for r in reports),
        "completed_items_including_sampler_padding": sum(r["completed_items"] for r in reports),
        "world_size": len(reports),
        "aggregation": "maximum_rank_duration",
        "rank_reports": reports,
    }


class StageTimeObserver:
    """Observe complete batches/queries, optionally stopping each rank early.

    Finished ranks keep participating in control rounds without yielding more
    work. This is essential when shards are unequal, empty, or stabilize at
    different times. Collectives run only on successful iteration, never in an
    exception/finally handler where a failed peer could no longer participate.
    """

    def __init__(
        self, stage: str, *, early_stop: bool, collectives: Any,
        synchronize: Optional[Callable[[], None]] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.stage = stage
        self.early_stop = early_stop
        self.collectives = collectives
        self.synchronize = synchronize or (lambda: None)
        self.clock = clock if clock is not None else time.perf_counter
        self.control_seconds = 0.0
        self.estimator: Optional[StableRateEstimator] = None
        self.report: Optional[dict[str, Any]] = None
        self._write: Optional[Callable[[str], None]] = None

    def iterate(self, items: Any, *, total_items: int, item_count: Callable[[Any], int]):
        if self.estimator is not None:
            raise RuntimeError("A stage observer cannot be reused for a second loop")
        self.synchronize()
        self.collectives.barrier()
        wall_started = self.clock()
        # Profiling adds per-query control rounds which normal reranking does
        # not have. Exclude their overhead/waits instead of measuring an
        # artificially lock-stepped multi-GPU workload.
        estimator = self.estimator = StableRateEstimator(
            total_items, clock=lambda: self.clock() - self.control_seconds,
        )
        iterator = iter(items)
        local_report = None
        done = False
        while True:
            if not done:
                if estimator.completed_items == total_items:
                    done = True
                    local_report = estimator.result(completed=True)
                else:
                    try:
                        item = next(iterator)
                    except StopIteration:
                        raise ValueError("Stage input ended before its declared item count") from None
                    yield item
                    self.synchronize()
                    count = item_count(item)
                    if count <= 0:
                        raise ValueError("A completed work unit must contain at least one item")
                    newly_stable = estimator.observe_batch(estimator.completed_items + count)
                    if newly_stable and self._write is not None and self.collectives.rank == 0:
                        stable = estimator.stable_estimate
                        action = "waiting for other ranks" if self.early_stop else "continuing the full retrieval"
                        self._write(
                            f"[time-estimate][{self.stage}] stable CV={stable['rate_cv']:.2%}; "
                            f"estimated_total_seconds={stable['estimated_total_seconds']:.3f}; {action}."
                        )
                    done = estimator.completed_items == total_items or (
                        self.early_stop and estimator.stable_estimate is not None
                    )
                    if done:
                        local_report = estimator.result(completed=estimator.completed_items == total_items)
            if self.early_stop:
                control_started = self.clock()
                all_done = self.collectives.all_done(done)
                self.control_seconds += max(0.0, self.clock() - control_started)
                if all_done:
                    break
            elif done:
                break
        local_report["rank"] = self.collectives.rank
        local_report["rate_unit"] = "query/second"
        local_report["estimation_control_seconds"] = self.control_seconds
        local_report["profiling_wall_seconds"] = self.clock() - wall_started
        self.report = merge_stage_time_reports(self.collectives.gather(local_report))

    def progress(
        self, items: Any, *, total: int, total_items: int,
        item_count: Callable[[Any], int], progress_factory: Callable[..., Any],
        desc: str, unit: str,
    ):
        with progress_factory(total=total, desc=desc, unit=unit, dynamic_ncols=True) as progress:
            self._write = progress.write
            for item in self.iterate(items, total_items=total_items, item_count=item_count):
                yield item
                progress.update(1)


class RerankTimeEstimateWriter:
    """Rank-zero, incremental reporting, separate from recall/result artifacts."""

    def __init__(self, output_dir: str, metadata: dict[str, Any], *, rank: int) -> None:
        self.path = Path(output_dir).expanduser().resolve() / "time_estimate.json"
        self.metadata = dict(metadata)
        self.rank = rank
        self.stages: dict[str, Any] = {}

    def record(self, stage: str, report: dict[str, Any]) -> None:
        self.stages[stage] = report
        self.save()
        if self.rank == 0:
            print(
                f"[time-estimate][{stage}] status={report['status']}; "
                f"estimated_total_seconds={report['estimated_total_seconds']:.3f}; "
                f"elapsed_seconds={report['elapsed_seconds']:.3f}"
            )

    def save(self) -> None:
        if self.rank != 0:
            return
        complete = "retrieval" in self.stages and "rerank" in self.stages
        report = {
            "mode": "estimate_time_only",
            "time_unit": "seconds",
            "status": self.stages["rerank"]["status"] if complete else "in_progress",
            "estimated_total_seconds": (
                self.stages["retrieval"]["elapsed_seconds"]
                + self.stages["rerank"]["estimated_total_seconds"]
                if complete else None
            ),
            "timing_scope": "formal query encoding/dense+sparse search + reranking",
            "excluded_time": [
                "model_loading", "dataset_preparation", "index_loading",
                "validation_retrieval", "alpha_selection", "recall_evaluation",
                "result_writes", "resource_cleanup", "estimation_control_collectives",
            ],
            "estimation_policy": {
                "warmup_batches": 20, "sample_interval_seconds": 5.0,
                "stable_samples": 5, "max_cv_exclusive": 0.05,
                "cv_definition": "population_standard_deviation / mean",
                "retrieval_work_unit": "dataloader_batch",
                "rerank_work_unit": "complete_query",
            },
            "config": self.metadata,
            "stages": self.stages,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(".json.tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        temporary_path.replace(self.path)
        if complete:
            print(
                f"[time-estimate][total] estimated_total_seconds={report['estimated_total_seconds']:.3f}; "
                f"saved={self.path}"
            )
