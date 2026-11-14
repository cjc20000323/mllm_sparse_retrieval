"""Compare example-sampling seeds using the original adaptive-view probabilities.

Run scripts/adaptive_views_seed_experiment.sh, or use --help for the Python CLI.
Only example selection and candidate scoring run: no corpus encoding or retrieval.
Each invocation uses fresh results, one model load, and one shared candidate mapping.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
from uuid import uuid4

from adaptive_encode_batch import CHECKPOINTS, annotation_files


PROJECT = Path(__file__).resolve().parents[1]
TASKS = {"flickr": "ir", "coco": "ir", "CUHK-PEDES": "tbpr", "ICFG-PEDES": "tbpr",
         "RSTPReid": "tbpr", "webqa": "t2it", "remuq": "it2t", "llava": "it2t"}
REPORTS = ("seed_view_probabilities", "round_probabilities", "view_stability", "seed_comparison")


def boolean(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"true", "1", "yes"}:
        return True
    if str(value).lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value!r}")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model-family", choices=CHECKPOINTS, default="qwen25")
    result.add_argument("--checkpoint-root", type=Path, default=Path("/root/gpufree-share/checkpoints"))
    result.add_argument("--model-path", type=Path)
    result.add_argument("--lora-model-path", type=Path)
    result.add_argument("--dataset", choices=TASKS, default="flickr")
    result.add_argument("--dataset-split", default="test")
    result.add_argument("--view-selection-split", default="")
    result.add_argument("--data-root", type=Path, default=Path("/root/gpufree-share"))
    result.add_argument("--output-root", type=Path, default=PROJECT / "adaptive_view_seed_results")
    result.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 42],
                        help="Example-sampling seeds; order is preserved and the first is the comparison baseline")
    result.add_argument("--inference-seed", type=int, default=42,
                        help="Fixed for every trial; independent of example sampling")
    result.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N; one process, no DeepSpeed")
    result.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    result.add_argument("--reps-loc", choices=("after_pad", "before_pad"), default="after_pad")
    result.add_argument("--llava-image-mode", choices=("anyres", "base"), default="anyres")
    result.add_argument("--qwen-min-pixels", type=int, default=0)
    result.add_argument("--qwen-max-pixels", type=int, default=0)
    result.add_argument("--view-selection-rounds", type=int, default=20)
    result.add_argument("--view-selection-modality", choices=("text", "image", "both"), default="both")
    result.add_argument("--view-example-mode", choices=("fixed", "auto"), default="fixed")
    result.add_argument("--view-example-max-tokens", type=int, default=0,
                        help="Auto-mode total budget including one answer token; 0 uses the model context limit")
    result.add_argument("--view-texts-per-round", type=int, default=8, help="Fixed mode only")
    result.add_argument("--view-images-per-round", type=int, default=2, help="Fixed mode only")
    result.add_argument("--use-view-pmi", type=boolean, default=False)
    result.add_argument("--view-pmi-prior-prompt-mode", choices=("false", "instruction_only"))
    result.add_argument("--use-view-token-variants", "--view-selection-token-variants", "--view_selection_token_variants",
                        type=boolean, nargs="?", const=True, default=False)
    result.add_argument("--use-hierarchical-views", type=boolean, default=False)
    result.add_argument("--shuffle-view-options", type=boolean, default=False)
    result.add_argument("--view-selection-run-id", default="",
                        help="Optional 16-character mapping ID; one mapping is shared by all sampling seeds")
    result.add_argument("--dry-run", type=boolean, nargs="?", const=True, default=False,
                        help="Check configuration/local paths and print the plan without loading weights or writing output")
    return result


def validate_args(args):
    from adaptive_views import resolve_pmi_prior_prompt_mode
    resolve_pmi_prior_prompt_mode(getattr(args, "view_pmi_prior_prompt_mode", None), pmi=args.use_view_pmi)
    args.model_path = (args.model_path or args.checkpoint_root / CHECKPOINTS[args.model_family]).expanduser().resolve()
    for name in ("data_root", "output_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.lora_model_path:
        args.lora_model_path = args.lora_model_path.expanduser().resolve()
    args.seeds = list(dict.fromkeys(args.seeds))
    if not args.seeds or args.view_selection_rounds < 1:
        raise ValueError("At least one seed and a positive round count are required")
    if not 0 <= args.inference_seed < 2 ** 32:
        raise ValueError("INFERENCE_SEED must be in [0, 2**32)")
    if min(args.qwen_min_pixels, args.qwen_max_pixels, args.view_example_max_tokens) < 0:
        raise ValueError("Pixel limits and the auto-mode token budget must be nonnegative")
    if args.qwen_min_pixels and args.qwen_max_pixels and args.qwen_min_pixels > args.qwen_max_pixels:
        raise ValueError("QWEN_MIN_PIXELS must not exceed QWEN_MAX_PIXELS")
    if args.llava_image_mode == "base" and not args.model_family.startswith("llava_"):
        raise ValueError("LLAVA_IMAGE_MODE=base requires LLaVA-NeXT families; use anyres for Qwen")
    if args.view_example_mode == "fixed":
        for name in ("text", "image"):
            count = getattr(args, "view_texts_per_round" if name == "text" else "view_images_per_round")
            if args.view_selection_modality in {name, "both"} and count < 1:
                raise ValueError(f"Fixed-mode {name} examples per round must be positive")
    if args.view_selection_run_id:
        import re
        if not args.shuffle_view_options or not re.fullmatch(r"[0-9a-f]{16}", args.view_selection_run_id):
            raise ValueError("VIEW_SELECTION_RUN_ID requires shuffling and 16 lowercase hexadecimal characters")
    for name in ("dataset_split", "view_selection_split"):
        value = getattr(args, name)
        if value and (value in {".", ".."} or "/" in value or "\\" in value):
            raise ValueError(f"Invalid {name}: {value!r}")
    return args


def source_split(args):
    return args.view_selection_split or args.dataset_split


def preflight(args):
    required = [args.model_path / "config.json", *annotation_files(args.data_root, args.dataset, source_split(args))]
    if args.lora_model_path:
        required.append(args.lora_model_path / "adapter_config.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("Missing input file(s): " + "; ".join(missing))


def configuration(args):
    return {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()
            if name != "use_view_token_variants" or value}


def pipeline_args(args, api):
    task = TASKS[args.dataset]
    # Dataset-level pools are independent of retrieval direction. IR uses full captions.
    result = api._parser().parse_args(["--dataset-name", args.dataset, "--task-type", task,
                                       "--query-type", "image" if task == "it2t" else "text"])
    result.dataset_split = source_split(args)
    result.data_root = str(args.data_root)
    result.retrieval_model_name_or_path = str(args.model_path)
    result.retrieval_lora_model_path = str(args.lora_model_path) if args.lora_model_path else None
    result.adaptive_views = True
    result.view_selection_token_variants = args.use_view_token_variants
    result.retrieval_device_map = "none"
    for name in ("device", "dtype", "reps_loc", "llava_image_mode", "qwen_min_pixels", "qwen_max_pixels"):
        setattr(result, "retrieval_" + name, getattr(args, name))
    return result


def selection_options(args, views):
    return views.AdaptiveViewArguments(
        adaptive_views=True, view_selection_hierarchical=args.use_hierarchical_views,
        view_selection_token_variants=args.use_view_token_variants, view_selection_prefix_tokens=getattr(args, "use_view_prefix_tokens", False),
        view_selection_shuffle=args.shuffle_view_options,
        view_selection_shuffle_run_id=args.view_selection_run_id or None,
        view_selection_modality=args.view_selection_modality, view_selection_pmi=args.use_view_pmi,
        view_pmi_prior_prompt_mode=getattr(args, "view_pmi_prior_prompt_mode", None),
        view_selection_rounds=args.view_selection_rounds, view_example_mode=args.view_example_mode,
        view_example_max_tokens=args.view_example_max_tokens, view_texts_per_round=args.view_texts_per_round,
        view_images_per_round=args.view_images_per_round)


def write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def ranking(selection, values=None):
    scores = selection["scores"] if values is None else values
    return sorted(range(len(selection["pool"])), key=lambda index: (-scores[index], index))


def reports(selections):
    """Summarize complete seed trials, with population std and first-seed comparisons."""
    final_rows, round_rows, stability, comparisons = [], [], [], []
    baseline = selections[0][1]
    baseline_order = ranking(baseline)
    top_k = len(baseline["selected"])
    pool = baseline["pool"]
    orders = [ranking(selection) for _, selection in selections]
    for (seed, selection), order in zip(selections, orders):
        if selection["pool"] != pool:
            raise ValueError("Cannot compare different candidate pools")
        positions = {index: rank for rank, index in enumerate(order, 1)}
        for index, view in enumerate(pool):
            final_rows.append(dict(seed=seed, **{key: view[key] for key in ("label", "aspect")},
                probability=selection["scores"][index], rank=positions[index],
                text_mean=None if selection["text_mean"] is None else selection["text_mean"][index],
                image_mean=None if selection["image_mean"] is None else selection["image_mean"][index],
                selected=positions[index] <= top_k))
        mapping = selection.get("option_shuffle")
        labels = {row["canonical_label"]: row["display_label"] for row in mapping["mapping"]} if mapping else {}
        for row in selection["rounds"]:
            for modality in ("text", "image"):
                values = row[modality + "_probabilities"]
                if values is None:
                    continue
                positions = {index: rank for rank, index in enumerate(ranking(selection, values), 1)}
                raw = row.get(modality + "_raw_probabilities", values)
                pmi = row.get(modality + "_pmi_scores")
                for index, view in enumerate(pool):
                    round_rows.append(dict(seed=seed, round=row["round"] + 1, modality=modality,
                        label=view["label"], display_label=labels.get(view["label"], view["label"]), aspect=view["aspect"],
                        probability=values[index], rank=positions[index], raw_probability=raw[index],
                        prior_probability=selection["calibration"]["prior_probabilities"][index] if pmi is not None else None,
                        pmi_score=pmi[index] if pmi is not None else None,
                        sample_count=len(row[modality + "_ids"]), input_tokens=row[modality + "_tokens"]))
        delta = [abs(first - second) for first, second in zip(selection["scores"], baseline["scores"], strict=True)]
        comparisons.append(dict(seed=seed, baseline_seed=selections[0][0],
            top1=pool[order[0]]["label"], top5_labels=" ".join(pool[index]["label"] for index in order[:top_k]),
            top1_changed=order[0] != baseline_order[0],
            top5_set_changed=set(order[:top_k]) != set(baseline_order[:top_k]),
            ranking_changed=order != baseline_order,
            max_probability_delta=max(delta), mean_absolute_probability_delta=statistics.mean(delta)))
    for index, view in enumerate(pool):
        values = [selection["scores"][index] for _, selection in selections]
        stability.append(dict(label=view["label"], aspect=view["aspect"], seed_count=len(values),
            probability_mean=statistics.mean(values), probability_std=statistics.pstdev(values),
            probability_min=min(values), probability_max=max(values),
            top1_frequency=sum(order[0] == index for order in orders) / len(orders),
            top5_frequency=sum(index in order[:top_k] for order in orders) / len(orders)))
    return dict(zip(REPORTS, (final_rows, round_rows, stability, comparisons)))


def image_reference(value):
    if isinstance(value, (str, Path)):
        return dict(kind="path", path=str(value))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return dict(kind="embedded_image", byte_count=len(value), sha256=hashlib.sha256(value).hexdigest())
    if isinstance(value, dict):
        result = dict(kind="dataset_image", path=value.get("path"))
        if value.get("bytes") is not None:
            result.update(byte_count=len(value["bytes"]), sha256=hashlib.sha256(value["bytes"]).hexdigest())
        return result
    return dict(kind="in_memory_image", type=type(value).__name__,
                size=list(value.size) if hasattr(value, "size") else None, mode=getattr(value, "mode", None))


def sampled_examples(selection, runtime, pools, views):
    import adaptive_view_shuffle as shuffle
    displayed = shuffle.display_pool(selection["pool"], selection.get(shuffle.FIELD))
    family = views.family_of(runtime.encoder.config)
    rows = []
    references = {}
    for row in selection["rounds"]:
        for modality in views.enabled_modalities(selection["selection_modality"]):
            ids = row[modality + "_ids"]
            items = [pools[modality][identifier] for identifier in ids]
            examples = []
            for item in items:
                if modality == "text":
                    examples.append(dict(id=item["id"], text=item["text"]))
                else:
                    if item["id"] not in references:
                        references[item["id"]] = image_reference(item["image"])
                    examples.append(dict(id=item["id"], image_reference=references[item["id"]]))
            prompt = views.render_choice(runtime.processor, family, displayed,
                texts=[item["text"] for item in items] if modality == "text" else None,
                image_count=len(items) if modality == "image" else 0)
            rows.append(dict(round=row["round"] + 1, modality=modality, sample_ids=ids, examples=examples,
                prompt=prompt, input_tokens=row[modality + "_tokens"], packing=row.get(modality + "_examples")))
    return rows


def run_experiment(args, runtime, identity, texts, images, *, views, seed_inference):
    import adaptive_view_shuffle as shuffle
    options = selection_options(args, views)
    # for_options records a fresh run ID on options once; all seed copies inherit it.
    mapping = shuffle.for_options(options, identity, args.dataset, views.view_pool(views.options_pool_mode(options)))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    output = args.output_root / args.model_path.name / args.dataset / source_split(args) / run_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(version=1, configuration=configuration(args), model=identity, task=TASKS[args.dataset],
        selection_split=source_split(args), pool_sizes=dict(text=len(texts), image=len(images)),
        baseline_seed=args.seeds[0], option_mapping=mapping, completed_seeds=[], status="running")
    views.atomic_json(output / "experiment.json", manifest)
    views.atomic_json(output / "view_pool.json", views.view_pool(views.options_pool_mode(options)))
    if mapping is not None:
        views.atomic_json(output / "option_mapping.json", mapping)
    print(f"[seed experiment] output: {output}", flush=True)
    pools = dict(text={item["id"]: item for item in texts}, image={item["id"]: item for item in images})
    complete = []
    seed = None
    try:
        for seed in args.seeds:
            seed_options = copy.deepcopy(options)
            seed_options.view_selection_seed = seed
            seed_inference(args.inference_seed)
            print(f"[seed experiment] sampling_seed={seed}; inference_seed={args.inference_seed}", flush=True)
            selection = views.choose_views(runtime.encoder, runtime.processor, identity, args.dataset,
                source_split(args), texts, images, seed_options)
            if selection.get(shuffle.FIELD) != mapping:
                raise ValueError("Candidate mapping changed between sampling seeds")
            seed_root = output / f"seed_{seed}"
            views.atomic_json(seed_root / "view_selection.json", selection)
            views.atomic_json(seed_root / "sampled_examples.json", sampled_examples(selection, runtime, pools, views))
            complete.append((seed, selection))
            current = reports(complete)
            for name, rows in current.items():
                write_csv(output / f"{name}.csv", rows)
            manifest["completed_seeds"].append(seed)
            views.atomic_json(output / "experiment.json", manifest)
            comparison = current["seed_comparison"][-1]
            print(f"[seed experiment] seed={seed} top1={comparison['top1']} top5={comparison['top5_labels']} "
                  f"max_delta={comparison['max_probability_delta']:.8f} "
                  f"ranking_changed={comparison['ranking_changed']}", flush=True)
        manifest["status"] = "complete"
        views.atomic_json(output / "experiment.json", manifest)
    except Exception as exc:
        manifest.update(status="failed", failed_seed=seed, error=f"{type(exc).__name__}: {exc}")
        views.atomic_json(output / "experiment.json", manifest)
        raise
    return output


def run(args):
    import adaptive_view_encoding as adaptive
    import adaptive_views as views
    from transformers import set_seed
    api = adaptive.pipeline_module()
    runtime = None
    try:
        set_seed(args.inference_seed)
        runtime = api._load_retrieval_runtime(pipeline_args(args, api))
        runtime.encoder.eval()
        runtime.processor = adaptive.configure_processor(runtime.encoder.config, runtime.processor,
                                                         args.reps_loc, args.llava_image_mode)
        identity = views.model_identity(str(args.model_path), runtime.encoder.config, runtime.processor,
                                        dtype=args.dtype, lora=str(args.lora_model_path) if args.lora_model_path else None)
        texts, images = adaptive.source_pools(pipeline_args(args, api), api, args.view_selection_modality)
        return run_experiment(args, runtime, identity, texts, images, views=views, seed_inference=set_seed)
    finally:
        api._release_retrieval_runtime(runtime)


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        validate_args(args)
        preflight(args)
        if args.dry_run:
            print(json.dumps(dict(configuration=configuration(args), task=TASKS[args.dataset],
                selection_split=source_split(args), baseline_seed=args.seeds[0],
                stages=["sample_examples", "score_candidates", "compare_seeds"]), ensure_ascii=False, indent=2))
            return 0
        output = run(args)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        return 2
    print(f"[seed experiment] complete: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
