"""Profile the existing pipeline: complete retrieval, FreeRet-style rerank ETA.

Use scripts/estimate_three_stage_retrieval_time.sh for one GPU per process.
This entrypoint never exports partial retrieval runs or evaluates Recall.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Callable

import three_stage_retrieval_pipeline as pipeline
from time_estimation import StageTimeObserver, TorchStageCollectives


DATASETS = ("coco", "flickr", "CUHK-PEDES", "ICFG-PEDES", "RSTPReid", "webqa", "remuq", "llava")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class TimeEstimateReport:
    """Rank-zero incremental report; worker failures use independent diagnostics."""

    def __init__(self, args, context, *, clock: Callable[[], float] = time.perf_counter):
        self.path = Path(args.output_dir).expanduser().resolve() / "time_estimate.json"
        self.context = context
        self.clock = clock
        self.started_at = clock()
        self.config = vars(args).copy()
        self.status = "in_progress"
        self.stages: dict[str, Any] = {}
        self.query_counts: list[int] = []
        self.error = None

    def payload(self) -> dict[str, Any]:
        complete = self.status in ("estimated", "completed") and len(self.stages) == 3
        return {
            "mode": "estimate_time_only",
            "status": self.status,
            "time_unit": "seconds",
            "estimated_total_seconds": (
                sum(self.stages[stage]["estimated_total_seconds"] for stage in ("stage1", "stage2", "stage3"))
                if complete else None
            ),
            "profiling_job_wall_seconds": self.clock() - self.started_at,
            "profiling_wall_scope": "rank-zero wall after distributed initialization; includes setup/incremental report I/O; excludes imports, final report write and distributed teardown",
            "timing_scope": "Stage 1/2 measured core compute + Stage 3 projected complete-query reranking",
            "total_formula": "max_rank(stage1_compute) + max_rank(stage2_compute) + max_rank(stage3_estimated_total)",
            "excluded_time": ["model_loading", "dataset_preparation", "index_loading", "corpus_index_creation",
                              "recall_evaluation", "result_writes", "resource_cleanup", "prior_cache_preparation",
                              "distributed_query_partition_and_result_transfer", "estimation_control_collectives"],
            "estimation_policy": {
                "warmup_batches": 20, "sample_interval_seconds": 5.0,
                "stable_samples": 5, "max_cv_exclusive": 0.05,
                "cv_definition": "population_standard_deviation / mean",
                "work_unit": "complete_query_including_all_rerank_candidate_microbatches",
                "stage1_and_stage2": "complete_actual_retrieval_no_early_stop",
                "stage3": "stop_each_rank_when_stable_or_exhausted",
            },
            "distributed": {
                "world_size": self.context.world_size,
                "device_policy": "one_complete_model_per_local_GPU; retrieval_released_before_ranker_loading",
                "per_rank_query_counts": self.query_counts,
                "num_queries": sum(self.query_counts),
                "sampler_padding": False,
            },
            "config": self.config,
            "stages": self.stages,
            "error": self.error,
        }

    def save(self) -> None:
        if self.context.is_main:
            atomic_json(self.path, self.payload())

    def record_compute(self, stage, timer, collectives) -> None:
        samples = collectives.gather(timer.snapshot(self.context.rank))
        merged = pipeline._merge_stage_timings(samples)
        seconds = merged["compute_seconds"]
        if seconds is None:
            raise ValueError("The estimation entrypoint requires computed, measurable stages.")
        self.stages[stage] = {
            **merged, "status": "completed", "measurement": "actual_core_compute",
            "elapsed_seconds": seconds, "estimated_total_seconds": seconds, "rank_reports": samples,
        }
        self.save()
        if self.context.is_main:
            print(f"[time-estimate][{stage}] actual_compute_seconds={seconds:.3f}")

    def finish(self, rerank_report, setup_timer) -> None:
        if rerank_report["total_items_including_sampler_padding"] != sum(self.query_counts):
            raise ValueError("Rerank observer query count does not match the disjoint retrieval shards.")
        self.stages["stage3"] = {
            **rerank_report,
            "measurement": "stable_throughput_projection_or_completed_actual",
            "setup_seconds_rank_local": dict(setup_timer.auxiliary),
        }
        if hasattr(setup_timer, "prior_cache"):
            self.stages["stage3"]["prior_cache"] = setup_timer.prior_cache
        self.status = rerank_report["status"]
        self.save()
        if self.context.is_main:
            seconds = self.payload()["estimated_total_seconds"]
            print(f"[time-estimate][total] {seconds:.3f}s ({pipeline._format_duration(seconds)}); saved={self.path}")

    def fail(self, stage, exc: BaseException) -> None:
        self.status = "failed"
        self.error = {"stage": stage, "rank": self.context.rank, "type": type(exc).__name__, "message": str(exc)}
        # No collective in the exception path: peers may have already exited.
        atomic_json(self.path.parent / f"failed_rank_{self.context.rank}.json", self.error)
        self.save()


def parser() -> argparse.ArgumentParser:
    result = pipeline._parser()
    result.description = "Estimate three-stage retrieval time: complete Stage 1/2, early-stop Stage 3 once throughput stabilizes."
    result.set_defaults(output_dir="./three_stage_time_estimates", rerank_template="caption_generation", rerank_num=5)
    return result


def validate_args(args) -> None:
    if args.dataset_name not in DATASETS:
        raise ValueError(f"Estimation supports exactly these eight datasets: {', '.join(DATASETS)}")
    if args.stage1_method != "sparse_concat" or any(
        (args.stage1_run, args.stage1_command, args.stage2_run, args.ranker_run)
    ):
        raise ValueError("Time estimation requires online sparse_concat -> hybrid -> pointwise; precomputed runs/external commands are not supported.")
    if args.rerank_type != "pointwise":
        raise ValueError("Time estimation requires --rerank-type pointwise.")
    for name in ("stage1_sparse_index", "stage2_dense_index", "stage2_sparse_index", "ranker_model_name_or_path"):
        if not getattr(args, name):
            raise ValueError(f"Time estimation requires --{name.replace('_', '-')}")
    if args.retrieval_device not in ("auto", "cuda") or any(
        value not in ("auto", "none") for value in (args.retrieval_device_map, args.ranker_device_map)
    ):
        raise ValueError("Time estimation pins complete models to each local GPU; use device=auto/cuda and device-map=auto/none.")
    if args.sparse_threads < 1 or not 0 <= args.hybrid_alpha <= 1:
        raise ValueError("Sparse threads must be positive and hybrid alpha must be in [0, 1].")
    pipeline._validate_args(args)


def initialize_context(args):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Three-stage time estimation requires CUDA; use the DeepSpeed launcher.")
    context = pipeline._initialise_distributed(args.local_rank)
    torch.cuda.set_device(context.local_rank)
    device = f"cuda:{context.local_rank}"
    # Also pin a one-worker launch; auto must not shard one model over other GPUs.
    args.retrieval_device = device
    args.retrieval_device_map = device
    args.ranker_device_map = device
    return context


def run_estimation(args, *, context=None, collectives=None, observer_factory=StageTimeObserver,
                   clock: Callable[[], float] = time.perf_counter):
    """Shared orchestration; injected CPU contexts/models are used by offline tests."""
    validate_args(args)
    owns_context = context is None
    context = initialize_context(args) if context is None else context
    writer = None
    succeeded = False
    active_stage = "setup"
    local_encoded = None
    try:
        writer = TimeEstimateReport(args, context, clock=clock)
        writer.save()
        collectives = collectives or TorchStageCollectives(f"cuda:{context.local_rank}")
        stage1_timer, stage2_timer, stage3_timer = (pipeline.StageTimer() for _ in range(3))
        active_stage = "stage1"
        local_stage1, local_encoded, global_stage1, counts = pipeline._run_sparse_concat_stage1(args, context, stage1_timer)
        writer.query_counts = counts
        del global_stage1  # No output/evaluation of retrieval runs in this entrypoint.
        writer.record_compute("stage1", stage1_timer, collectives)
        active_stage = "stage2"
        try:
            dense, sparse, hybrid = pipeline.run_stage2(args, local_stage1, None, timing=stage2_timer,
                                                       encoded_queries=local_encoded)
            del dense, sparse
            pipeline._validate_query_coverage(local_stage1, hybrid, "Stage 2 local")
        finally:
            # Stage 1 already released its runtime; retain no CPU query vectors for Stage 3.
            local_encoded = None
        writer.record_compute("stage2", stage2_timer, collectives)
        del local_stage1
        active_stage = "stage3"
        observer = observer_factory(
            "stage3", early_stop=True, collectives=collectives,
            synchronize=lambda: pipeline._synchronize_cuda_devices((f"cuda:{context.local_rank}",)),
        )
        # The observer starts only inside Reranker's query loop, after model/cache setup.
        partial_run = pipeline.run_stage3(args, hybrid, context, timing=stage3_timer, progress_observer=observer)
        del partial_run, hybrid  # Never gather, persist, or evaluate partially reranked queries.
        if observer.report is None:
            raise RuntimeError("Selected reranker did not complete its query-level timing observer.")
        writer.finish(observer.report, stage3_timer)
        succeeded = True
        return writer.payload()
    except BaseException as exc:
        if writer is not None:
            try:
                writer.fail(active_stage, exc)
            except Exception as reporting_error:
                print(f"[time-estimate][rank {context.rank}] Could not persist failure: {reporting_error}")
        raise
    finally:
        local_encoded = None
        if owns_context:
            pipeline._shutdown_distributed(context, synchronize=succeeded)


def main(argv=None) -> None:
    cli = parser()
    args = cli.parse_args(argv)
    try:
        run_estimation(args)
    except (ValueError, FileExistsError) as exc:
        cli.error(str(exc))


if __name__ == "__main__":
    main()
