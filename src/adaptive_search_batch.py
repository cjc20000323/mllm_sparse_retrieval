"""Batch full-corpus adaptive search with exact encoding lookup or explicit corpus paths."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from uuid import uuid4

import adaptive_views as views
from adaptive_encode_batch import (CHECKPOINTS, GROUPS, PROJECT, annotation_files, boolean,
    configured_list, encoding_controls, resolved_path, split_name)
from adaptive_retrieval_batch import (FAMILIES, namespaces, selection_options, selection_location,
    recover_selection_cache, MissingInput)
import search_adaptive_views as search
import stc_sparse as stc
import lens_sparse as lens

SEARCH_DEFAULTS = dict(DEPTH="1000", HYBRID_ALPHA="0.5", SELECT_HYBRID_ALPHA="false",
    RETRIEVAL_BATCH_SIZE="1", THREADS="16", USE_GPU="false", OVERWRITE="false", DRY_RUN="false")
ENCODE_ARGUMENTS = ("REPS_LOC", "SPARSE_LENGTH", "TEXT_SPARSE_LENGTH", "IMAGE_SPARSE_LENGTH",
    "SPARSE_VALUE_TYPE", "SPARSE_LOWER_OR_UPPER", "SPARSE_VALUE_MEAN", "IS_FILTERED",
    "NUM_EXPENDED_TOKENS", "TBPR_TYPE", "LLAVA_IMAGE_MODE", "QWEN_MIN_PIXELS", "QWEN_MAX_PIXELS",
    "USE_FREERET_DENSE", "USE_FREERET_PROMPT_CONSTRAINTS", "USE_FREERET_FULL_PROMPT")


@dataclass
class Settings:
    group: str
    families: tuple
    datasets: tuple
    queries: tuple
    models: dict
    roots: dict
    controls: dict
    options: object
    split: str
    selection_split: str
    environment: dict
    project: Path
    selection_root: Path | None

    @classmethod
    def from_environment(cls, group, env=None, *, project=PROJECT):
        env = dict(os.environ if env is None else env)
        project = Path(project).resolve()
        if boolean(env.get("USE_VIEW_OVERLAP_SELECTION", "false"), "USE_VIEW_OVERLAP_SELECTION"):
            raise ValueError("Adaptive search does not support overlap")
        families = configured_list(env, "MODEL_FAMILY", "MODEL_FAMILIES", CHECKPOINTS, CHECKPOINTS)
        datasets = configured_list(env, "DATASET", "DATASETS", GROUPS[group]["datasets"], GROUPS[group]["datasets"])
        queries = configured_list(env, "IR_QUERY_TYPE", "IR_QUERY_TYPES", ("text", "image"), ("text", "image")) if group == "ir" else ()
        total = len(families) * len(datasets) * (len(queries) or 1)
        single_options = ("DENSE_CORPUS_DIR", "SPARSE_CORPUS_DIR", "VIEW_SELECTION_FILE", "OUTPUT_DIR",
                          "VAL_PASSAGE_REPS", "VAL_SPARSE_INDEX", "VAL_VIEW_SELECTION_FILE", "VALIDATION_DATASET_NAME")
        if total > 1 and any(env.get(name) for name in single_options):
            raise ValueError("Explicit corpus/selection/output/validation paths require one search job")
        if len(families) > 1 and env.get("MODEL_PATH"):
            raise ValueError("MODEL_PATH requires one model; use MODEL_PATH_<FAMILY> for batches")
        def path(name, default):
            return resolved_path(env.get(name) or default, project, name)
        checkpoint_root = path("CHECKPOINT_ROOT", "/root/autodl-fs/checkpoints")
        models = {family: resolved_path(env.get("MODEL_PATH") or env.get(f"MODEL_PATH_{family.upper()}") or
                    str(checkpoint_root / CHECKPOINTS[family]), project, "model path") for family in families}
        if len({model.name.casefold() for model in models.values()}) != len(models):
            raise ValueError("Model directory names must be distinct")
        c = encoding_controls(env)
        c.update({name: env.get(name) or default for name, default in SEARCH_DEFAULTS.items()})
        c["VIEW_SELECTION_RUN_ID"] = env.get("VIEW_SELECTION_RUN_ID", "")
        c["RETRIEVAL_LLAVA_IMAGE_MODE"], c["RETRIEVAL_DTYPE"] = c["LLAVA_IMAGE_MODE"], c["DTYPE"]
        if c["LLAVA_IMAGE_MODE"] == "base" and any(not family.startswith("llava_") for family in families):
            raise ValueError("LLAVA_IMAGE_MODE=base requires LLaVA-NeXT families only; use anyres for Qwen")
        if c["SHUFFLE_VIEW_OPTIONS"] == "true" and not c["VIEW_SELECTION_RUN_ID"]:
            raise ValueError("Shuffled search needs VIEW_SELECTION_RUN_ID from encoding; it never generates a new ID")
        for name in ("DEPTH", "RETRIEVAL_BATCH_SIZE", "THREADS"):
            if int(c[name]) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("SELECT_HYBRID_ALPHA", "USE_GPU", "OVERWRITE", "DRY_RUN"):
            c[name] = str(boolean(c[name], name)).lower()
        if not 0 <= float(c["HYBRID_ALPHA"]) <= 1:
            raise ValueError("HYBRID_ALPHA must be finite and in [0,1]")
        if int(c["QWEN_MIN_PIXELS"]) and int(c["QWEN_MAX_PIXELS"]) and int(c["QWEN_MIN_PIXELS"]) > int(c["QWEN_MAX_PIXELS"]):
            raise ValueError("QWEN_MIN_PIXELS must not exceed QWEN_MAX_PIXELS")
        options = selection_options(env, c)
        split = split_name(env.get("DATASET_SPLIT", "test"), "DATASET_SPLIT")
        return cls(group, families, datasets, queries, models,
            dict(data=path("DATA_ROOT", "/root/autodl-tmp"), dense=path("DENSE_OUTPUT_ROOT", "./dense_output"),
                 sparse=path("SPARSE_OUTPUT_ROOT", "./sparse_output"), results=path("SEARCH_OUTPUT_ROOT", "./adaptive_search_results")),
            c, options, split, split_name(env.get("VIEW_SELECTION_SPLIT") or split, "VIEW_SELECTION_SPLIT"),
            env, project, path("VIEW_SELECTION_ROOT", "") if env.get("VIEW_SELECTION_ROOT") else None)


@dataclass
class Job:
    key: str
    command: list
    output: Path
    configuration: dict


def prepare_jobs(settings, *, identity_loader=None, backend=None):
    import adaptive_view_encoding as default_backend
    backend = backend or default_backend
    identity_loader = identity_loader or backend.load_model_identity
    c, env = settings.controls, settings.environment
    resolver = views.SelectionFileResolver(settings.selection_root, sparse_root=settings.roots["sparse"])
    jobs, missing = [], []
    for family in settings.families:
        identity = None
        for dataset in settings.datasets:
            task = search.TASKS[dataset]
            queries = settings.queries if task == "ir" else ("composed",) if task == "it2t" else ("text",)
            for query in queries:
                key = f"{family}|{dataset}|{query}|{settings.split}"
                try:
                    required = [settings.models[family] / "config.json", *annotation_files(settings.roots["data"], dataset, settings.split)]
                    absent = [str(path) for path in required if not path.is_file()]
                    if absent:
                        raise MissingInput("Missing inputs: " + ", ".join(absent))
                    if identity is None:
                        identity = identity_loader(str(settings.models[family]), dtype=c["DTYPE"], reps_loc=c["REPS_LOC"],
                            min_pixels=int(c["QWEN_MIN_PIXELS"]), max_pixels=int(c["QWEN_MAX_PIXELS"]), image_mode=c["LLAVA_IMAGE_MODE"])
                        if identity["family"] != FAMILIES[family]:
                            raise ValueError(f"Model family mismatch for {family}: {identity['family']}")
                    explicit = bool(env.get("DENSE_CORPUS_DIR") or env.get("SPARSE_CORPUS_DIR"))
                    selection_file = env.get("VIEW_SELECTION_FILE")
                    if explicit:
                        dense_path, sparse_path = env.get("DENSE_CORPUS_DIR"), env.get("SPARSE_CORPUS_DIR")
                    else:
                        kind = "text" if task == "ir" and query == "image" else "image" if task in {"ir", "tbpr"} else "corpus"
                        model, data, training = namespaces(settings, family, dataset, task, kind)
                        if selection_file:
                            locked = views.load_selection(selection_file, identity=identity, dataset=dataset)
                        else:
                            location, cache_key, strategy = selection_location(settings, model, data, training, identity)
                            if location.is_file():
                                locked = views.load_selection(location, identity=identity, dataset=dataset,
                                    selection_modality=settings.options.view_selection_modality,
                                    view_selection_pmi=settings.options.view_selection_pmi,
                                    view_selection_hierarchical=settings.options.view_selection_hierarchical,
                                    view_example_mode=settings.options.view_example_mode,
                                    view_example_max_tokens=settings.options.view_example_max_tokens)
                                views.require_equal(dict(split=locked["selection_split"]), dict(split=settings.selection_split), "Selection source split")
                                views.require_equal(locked["sampling"], cache_key["sampling"], "Selection sampling")
                                views.require_equal(dict(shuffle=locked.get(views.option_shuffle.FIELD)),
                                    dict(shuffle=cache_key.get(views.option_shuffle.FIELD)), "Selection option shuffle")
                                if strategy is not None:
                                    strategy.validate(locked)
                            else:
                                location, locked = recover_selection_cache(settings, model, data, training, identity,
                                    location, backend, resolver)
                            selection_file = str(location)
                        if views.is_overlap_selection(locked):
                            raise ValueError("Overlap artifacts are not supported")
                        dense_path, sparse_path = backend.artifact_dirs(model, data, training, locked)
                        if not all((Path(path) / views.MANIFEST).is_file() for path in (dense_path, sparse_path)):
                            raise MissingInput(f"Missing exact encoded corpora: dense={dense_path}; sparse={sparse_path}")
                    argv = ["--model_name_or_path", str(settings.models[family]), "--dataset_name", dataset,
                        "--task_type", task, "--query_type", query, "--dataset_split", settings.split,
                        "--adaptive_data_root", str(settings.roots["data"]), "--search_output_root", str(settings.roots["results"]),
                        "--dtype", c["DTYPE"], "--depth", c["DEPTH"], "--alpha", c["HYBRID_ALPHA"],
                        "--use_fixed_alpha", str(c["SELECT_HYBRID_ALPHA"] != "true").lower(),
                        "--retrieval_batch_size", c["RETRIEVAL_BATCH_SIZE"], "--threads", c["THREADS"],
                        "--use_gpu", c["USE_GPU"], "--overwrite", c["OVERWRITE"],
                        "--view_probability_mass", c["USE_VIEW_PROBABILITY_MASS"],
                        "--view_selection_hierarchical", c["USE_HIERARCHICAL_VIEWS"]]
                    if c["USE_VIEW_PROBABILITY_MASS"] == "true":
                        argv += ["--view_probability_mass_threshold", c["VIEW_PROBABILITY_MASS_THRESHOLD"]]
                    for name in ENCODE_ARGUMENTS:
                        argv += ["--" + name.lower(), c[name]]
                    # The shell's false default is explicit; only direct Python
                    # calls with an omitted flag inherit the corpus mode.
                    argv += ["--use_stc_sparse", c["USE_STC_SPARSE"]]
                    argv += ["--use_lens_sparse", c["USE_LENS_SPARSE"]]
                    if c["USE_LENS_SPARSE"] == "true":
                        for name in lens.NAMES:
                            if name != "USE_LENS_SPARSE":
                                argv += ["--" + name.lower(), c[name]]
                    if c["USE_STC_SPARSE"] == "true":
                        for name in stc.NAMES:
                            if name != "USE_STC_SPARSE":
                                argv += ["--" + name.lower(), c[name]]
                    for flag, value in (("passage_reps", dense_path), ("sparse_index", sparse_path),
                        ("view_selection_file", selection_file), ("view_selection_root", settings.selection_root),
                        ("view_selection_run_id", c["VIEW_SELECTION_RUN_ID"]),
                        ("view_selection_pmi", c.get("USE_VIEW_PMI")), ("view_selection_modality", c.get("VIEW_SELECTION_MODALITY"))):
                        if value is not None and str(value):
                            argv += ["--" + flag, str(value)]
                    for name in ("OUTPUT_DIR", "VAL_PASSAGE_REPS", "VAL_SPARSE_INDEX", "VAL_VIEW_SELECTION_FILE",
                                 "VALIDATION_DATASET_NAME", "VALIDATION_SPLIT"):
                        if env.get(name):
                            argv += ["--" + name.lower(), env[name]]
                    args = search.parser().parse_args(argv)
                    experiment = search.prepare(args, identity_loader=lambda *a, **k: identity, backend=backend)
                    validation = search.prepare_validation(args, experiment, identity_loader=lambda *a, **k: identity, backend=backend)
                    if not args.use_fixed_alpha and (not experiment.dense_root or not experiment.sparse_root):
                        raise ValueError("Automatic alpha requires both dense and sparse corpora")
                    config = search.run_configuration(experiment, validation)
                    output = search.output_location(experiment, config)
                    argv += ["--output_dir", str(output)]
                    command = [env.get("DEEPSPEED_BIN") or "deepspeed", f"--num_gpus={c['GPUS']}",
                        f"--num_nodes={c['NUM_NODES']}", "src/search_adaptive_views.py", *argv]
                    jobs.append(Job(key, command, output, config))
                except (FileNotFoundError, MissingInput) as exc:
                    missing.append((key, str(exc)))
    return jobs, missing


def run(settings, *, runner=None, identity_loader=None, backend=None):
    jobs, missing = prepare_jobs(settings, identity_loader=identity_loader, backend=backend)
    rows = [dict(job=key, status="skipped", error=error) for key, error in missing]
    for key, error in missing:
        print(f"[adaptive search] skipped {key}: {error}", flush=True)
    dry = settings.controls["DRY_RUN"] == "true"
    summary = settings.roots["results"] / ".batches" / uuid4().hex / "batch_summary.csv"
    if not dry:
        search.write_csv(summary, rows)
        print(f"[adaptive search] batch summary: {summary}", flush=True)
    for job in jobs:
        print(f"[adaptive search] {job.key}\nOutput: {job.output}\nCommand: {shlex.join(job.command)}", flush=True)
        if dry:
            continue
        try:
            result = (runner or subprocess.run)(job.command, cwd=settings.project, env=settings.environment)
            if result.returncode:
                rows.append(dict(job=job.key, status="failed", output_dir=str(job.output), error=f"exit {result.returncode}"))
                search.write_csv(summary, rows)
                return result.returncode
            report = json.loads((job.output / "recall_metrics.json").read_text(encoding="utf-8"))
            metadata = json.loads((job.output / "run_metadata.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            rows.append(dict(job=job.key, status="failed", output_dir=str(job.output), error=str(exc)))
            search.write_csv(summary, rows)
            raise
        for method, metrics in report["metrics"].items():
            rows.append(dict(job=job.key, status="complete", method=method, output_dir=str(job.output),
                alpha=report["alpha"], selection_source=report["alpha_selection_source"],
                selected_views=json.dumps(metadata["selected_views"], ensure_ascii=False), **metrics))
        search.write_csv(summary, rows)
    print(f"[adaptive search] {'planned' if dry else 'completed'}={len(jobs)} skipped={len(missing)}", flush=True)
    return 0 if jobs else 2


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, epilog="Configure MODEL_FAMILIES, DATASETS, IR_QUERY_TYPES; "
        "SELECT_HYBRID_ALPHA=false uses HYBRID_ALPHA (default .5). DENSE_CORPUS_DIR/SPARSE_CORPUS_DIR override one job. "
        "DRY_RUN=true validates and prints without loading model weights or writing files. No overlap or reranking.")
    p.add_argument("--task-group", choices=tuple(GROUPS), required=True)
    args = p.parse_args(argv)
    try:
        return run(Settings.from_environment(args.task_group))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
