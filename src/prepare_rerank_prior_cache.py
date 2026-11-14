"""Prepare caption priors with the same model, candidates and scoring as Stage 3."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

from rerank_prior_cache import DEFAULT_CACHE_DIR, validate_prior_cache_options


def encoding_prior_task(dataset, kind):
    """Return the supported retrieval direction for a corpus encoding job."""
    if dataset in {"flickr", "coco"} and kind == "text":
        return "ir", "image"
    if dataset in {"remuq", "llava"} and kind == "corpus":
        return "it2t", "image"
    return None


def encoding_prior_controls(env, project):
    value = env.get("USE_RERANK_PRIOR_CACHE", "false").lower()
    if value not in {"true", "1", "yes", "false", "0", "no"}:
        raise ValueError("USE_RERANK_PRIOR_CACHE must be true/false, 1/0 or yes/no")
    enabled = value in {"true", "1", "yes"}
    directory = Path(env.get("RERANK_PRIOR_CACHE_DIR") or Path(project) / "cache" / "rerank_prior").expanduser()
    directory = (directory if directory.is_absolute() else Path(project) / directory).resolve()
    controls = {
        "USE_RERANK_PRIOR_CACHE": str(enabled).lower(),
        "RERANK_PRIOR_CACHE_DIR": str(directory),
        "RERANK_TEMPLATE": env.get("RERANK_TEMPLATE", "what_caption_generation"),
        "RERANK_BATCH_SIZE": env.get("RERANK_BATCH_SIZE", "1"),
        "RERANK_REPS_LOC": env.get("RERANK_REPS_LOC", "after_pad"),
    }
    if enabled:
        try:
            size = int(controls["RERANK_BATCH_SIZE"])
        except ValueError as exc:
            raise ValueError("RERANK_BATCH_SIZE must be a positive integer") from exc
        if size < 1:
            raise ValueError("RERANK_BATCH_SIZE must be a positive integer")
        controls["RERANK_BATCH_SIZE"] = str(size)
        if controls["RERANK_REPS_LOC"] not in {"before_pad", "after_pad"}:
            raise ValueError("RERANK_REPS_LOC must be before_pad or after_pad")
        validate_prior_cache_options(True, "modify_single", controls["RERANK_TEMPLATE"], "ir", "image")
        if directory.exists() and not directory.is_dir():
            raise ValueError(f"Expected a prior cache directory: {directory}")
    return controls


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset-name", required=True)
    result.add_argument("--dataset-split", default="test")
    result.add_argument("--task-type", choices=("ir", "it2t"), required=True)
    result.add_argument("--query-type", choices=("image", "text", "composed"), default="image")
    result.add_argument("--data-root", default="/root/autodl-tmp")
    result.add_argument("--ranker-model-name-or-path", required=True)
    result.add_argument("--ranker-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    result.add_argument("--ranker-device-map", default="auto")
    result.add_argument("--ranker-llava-image-mode", choices=("anyres", "base"), default="anyres")
    result.add_argument("--ranker-qwen-min-pixels", type=int, default=0)
    result.add_argument("--ranker-qwen-max-pixels", type=int, default=0)
    result.add_argument("--rerank-template", default="what_caption_generation")
    result.add_argument("--rerank-batch-size", type=int, default=1)
    result.add_argument("--rerank-reps-loc", choices=("before_pad", "after_pad"), default="after_pad")
    result.add_argument("--rerank-prior-cache-dir", "--rerank_prior_cache_dir", default=str(DEFAULT_CACHE_DIR))
    result.add_argument("--local_rank", "--local-rank", type=int, default=-1, help=argparse.SUPPRESS)
    result.set_defaults(modify_type="modify_single", rerank_type="pointwise", use_rerank_prior_cache=True)
    return result


def validate_args(args):
    validate_prior_cache_options(True, args.modify_type, args.rerank_template,
                                 args.task_type, args.query_type, args.rerank_type)
    if args.rerank_batch_size < 1:
        raise ValueError("rerank-batch-size must be positive")
    low, high = args.ranker_qwen_min_pixels, args.ranker_qwen_max_pixels
    if min(low, high) < 0 or (low and high and low > high):
        raise ValueError("Invalid reranker Qwen pixel limits")
    directory = Path(args.rerank_prior_cache_dir).expanduser()
    if directory.exists() and not directory.is_dir():
        raise ValueError(f"Expected a prior cache directory: {directory}")


def prepare(args, *, api=None, reranker_module=None):
    """All ranks enter, including ranks with no local candidates."""
    validate_args(args)
    api = api if api is not None else importlib.import_module("three_stage_retrieval_pipeline_adaptive_views")
    context = api._initialise_distributed(args.local_rank)
    success = False
    original_prefix = None
    try:
        dataset = api.build_dataset(args)
        model, processor = api._load_ranker_model(args, context)
        dataset.processor = processor
        text_map, image_files, image_paths = api._ranker_dataset_maps(dataset, args)
        reranker_module = reranker_module if reranker_module is not None else importlib.import_module("reranker")
        original_prefix = reranker_module.path_prefix
        reranker_module.path_prefix = str(Path(args.data_root).expanduser().resolve()).rstrip("/\\") + "/"
        ranker = reranker_module.Reranker(
            model, processor, args.dataset_name, args.query_type,
            text_map, image_files, image_paths, processor.tokenizer.get_vocab(), dataset,
        )
        # Match Stage 3 exactly: even autocast flags participate in the cache key.
        model_args = SimpleNamespace(model_name_or_path=args.ranker_model_name_or_path)
        data_args = SimpleNamespace(reps_loc=args.rerank_reps_loc, dataset_split=args.dataset_split)
        training_args = SimpleNamespace(fp16=args.ranker_dtype == "float16", task_type=args.task_type)
        search_args = SimpleNamespace(modify_type=args.modify_type, use_rerank_prior_cache=True,
                                      rerank_prior_cache_dir=args.rerank_prior_cache_dir)
        report = ranker.prepare_caption_prior_cache(
            data_args, training_args, model_args, search_args,
            args.rerank_batch_size, args.rerank_template, args.rerank_type,
        )
        success = True
        return report
    finally:
        if original_prefix is not None:
            reranker_module.path_prefix = original_prefix
        api._shutdown_distributed(context, synchronize=success)


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    try:
        prepare(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Prior cache preparation failed: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
