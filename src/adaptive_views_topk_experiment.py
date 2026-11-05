"""Independent Top-1..7 adaptive sparse-view experiment.

Run scripts/adaptive_views_topk_experiment.sh once. Configuration is taken from
its environment, not from the old encode/retrieval launchers. DRY_RUN is read-only.
The original five-view selection remains immutable: its complete probability
ranking supplies seven prefixes, stored in this experiment's own manifests.

STOP_AFTER_STAGE2=true skips all ranker setup and execution. Each model/dataset
shares one probability selection across IR directions; query and candidate
prefixes always match. Whole-input dense vectors are shared across prefixes.
Results use the existing retrieval/evaluation algorithms and 0..1 Recall scale.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from uuid import uuid4


PROJECT = Path(__file__).resolve().parents[1]
COUNTS = tuple(range(1, 8))
VERSION = 1
CHECKPOINTS = {
    "qwen2": "Qwen-Qwen2-VL-7B-Instruct",
    "qwen25": "Qwen-Qwen2.5-VL-7B-Instruct",
    "qwen3": "Qwen-Qwen3-VL-8B-Instruct",
    "llava_llama": "llava-hf-llama3-llava-next-8b-hf",
    "llava_mistral": "llava-hf-llava-v1.6-mistral-7b-hf",
}
TASKS = {"flickr": "ir", "coco": "ir", "CUHK-PEDES": "tbpr",
         "ICFG-PEDES": "tbpr", "RSTPReid": "tbpr", "webqa": "t2it",
         "remuq": "it2t", "llava": "it2t"}
DEFAULTS = {
    "DATASET_SPLIT": "test", "VIEW_SELECTION_SPLIT": "", "VIEW_SELECTION_FILE": "",
    "DTYPE": "float16", "PER_DEVICE_BATCH_SIZE": 1, "RETRIEVAL_BATCH_SIZE": 1,
    "REPS_LOC": "after_pad", "SPARSE_LENGTH": 40, "SPARSE_VALUE_TYPE": "sum",
    "SPARSE_LOWER_OR_UPPER": "lower", "SPARSE_VALUE_MEAN": False, "IS_FILTERED": False,
    "NUM_EXPENDED_TOKENS": 0, "TBPR_TYPE": "origin_type", "LLAVA_IMAGE_MODE": "anyres",
    "QWEN_MIN_PIXELS": 0, "QWEN_MAX_PIXELS": 0, "VIEW_SELECTION_ROUNDS": 20,
    "VIEW_TEXTS_PER_ROUND": 8, "VIEW_IMAGES_PER_ROUND": 2, "VIEW_SELECTION_SEED": 42,
    "VIEW_EXAMPLE_MODE": "auto", "VIEW_EXAMPLE_MAX_TOKENS": 0,
    "SHUFFLE_VIEW_OPTIONS": False, "VIEW_SELECTION_RUN_ID": "",
    "USE_FREERET_DENSE": False, "USE_FREERET_PROMPT_CONSTRAINTS": False,
    "USE_FREERET_FULL_PROMPT": False, "STOP_AFTER_STAGE2": False,
    "STAGE1_DEPTH": 1000, "STAGE2_DEPTH": 1000, "SPARSE_THREADS": 16,
    "HYBRID_ALPHA": 0.5, "RERANK_NUM": 5, "RERANK_BATCH_SIZE": 1,
    "GPUS": 1, "NUM_NODES": 1,
}


def boolean(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError(f"Expected true/false, got {value!r}")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path, rows, columns=None):
    rows = list(rows)
    columns = columns or list(dict.fromkeys(key for row in rows for key in row))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def rank_action(action):
    """Propagate a worker-local failure before peers enter the next collective."""
    from adaptive_view_encoding import worker_errors
    result, error = None, None
    try:
        result = action()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    worker_errors(error)
    return result


def stamp(root, configuration, files, name="complete.json"):
    root = Path(root)
    atomic_json(root / name, dict(configuration=configuration, fingerprint=digest(configuration),
        files={str(Path(file).relative_to(root)): file_hash(file) for file in files}))


def completed(root, configuration, name="complete.json"):
    root = Path(root)
    path = root / name
    if not path.is_file():
        return False
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("configuration") != configuration or value.get("fingerprint") != digest(configuration):
        raise ValueError(f"Experiment configuration mismatch: {path}")
    if not value.get("files"):
        raise ValueError(f"Empty completion manifest: {path}")
    for relative, expected in value["files"].items():
        item = root / relative
        if not item.resolve().is_relative_to(root.resolve()) or not item.is_file() or file_hash(item) != expected:
            raise ValueError(f"Incomplete/corrupt experiment artifact: {item}")
    return True


def _list(env, singular, plural, default, allowed):
    values = (env.get(singular) or env.get(plural) or default).split()
    if not values or len(values) != len(set(values)) or any(item not in allowed for item in values):
        raise ValueError(f"Invalid {plural}: {values}; allowed={list(allowed)}")
    return values


def build_jobs(env, project=PROJECT):
    """Resolve the complete matrix before creating outputs or starting workers."""
    project = Path(project).resolve()
    def path(value):
        item = Path(value).expanduser()
        return str((item if item.is_absolute() else project / item).resolve())
    families = _list(env, "MODEL_FAMILY", "MODEL_FAMILIES", "qwen25", CHECKPOINTS)
    datasets = _list(env, "DATASET", "DATASETS", "flickr", TASKS)
    directions = _list(env, "IR_QUERY_TYPE", "IR_QUERY_TYPES", "text", ("text", "image"))
    if env.get("MODEL_PATH") and len(families) != 1:
        raise ValueError("MODEL_PATH requires one model family; use per-family MODEL_PATH_* for a batch")
    if env.get("VIEW_SELECTION_FILE") and len(families) * len(datasets) != 1:
        raise ValueError("VIEW_SELECTION_FILE requires one model/dataset pair (both IR directions are allowed)")
    controls = {}
    stop_after_stage2 = boolean(env.get("STOP_AFTER_STAGE2") or False)
    for key, default in DEFAULTS.items():
        raw = default if stop_after_stage2 and key in {"RERANK_NUM", "RERANK_BATCH_SIZE"} else env.get(key, "") or default
        controls[key] = boolean(raw) if isinstance(default, bool) else type(default)(raw)
    for key in ("USE_VIEW_PMI", "USE_HIERARCHICAL_VIEWS"):
        controls[key] = boolean(env[key]) if env.get(key) else None
    controls["VIEW_SELECTION_MODALITY"] = env.get("VIEW_SELECTION_MODALITY") or None
    for key in ("TEXT_SPARSE_LENGTH", "IMAGE_SPARSE_LENGTH"):
        controls[key] = int(env.get(key) or controls["SPARSE_LENGTH"])
    for key in ("GPUS", "NUM_NODES", "PER_DEVICE_BATCH_SIZE", "RETRIEVAL_BATCH_SIZE", "SPARSE_LENGTH",
                "TEXT_SPARSE_LENGTH", "IMAGE_SPARSE_LENGTH", "VIEW_SELECTION_ROUNDS",
                "STAGE1_DEPTH", "STAGE2_DEPTH", "SPARSE_THREADS"):
        if controls[key] < 1:
            raise ValueError(f"{key} must be positive")
    for key in ("NUM_EXPENDED_TOKENS", "QWEN_MIN_PIXELS", "QWEN_MAX_PIXELS", "VIEW_EXAMPLE_MAX_TOKENS"):
        if controls[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    choices = {"DTYPE": ("float16", "bfloat16", "float32"), "REPS_LOC": ("before_pad", "after_pad"),
               "SPARSE_VALUE_TYPE": ("sum", "max", "replace"), "SPARSE_LOWER_OR_UPPER": ("lower", "upper"),
               "TBPR_TYPE": ("origin_type", "type"), "LLAVA_IMAGE_MODE": ("anyres", "base"),
               "VIEW_EXAMPLE_MODE": ("auto", "fixed"), "VIEW_SELECTION_MODALITY": (None, "both", "text", "image")}
    for key, allowed in choices.items():
        if controls[key] not in allowed:
            raise ValueError(f"Invalid {key}: {controls[key]}")
    if controls["VIEW_EXAMPLE_MODE"] == "fixed" and not controls["VIEW_SELECTION_FILE"]:
        mode = controls["VIEW_SELECTION_MODALITY"] or "both"
        for modality in ("text", "image"):
            key = "VIEW_TEXTS_PER_ROUND" if modality == "text" else "VIEW_IMAGES_PER_ROUND"
            if mode in {"both", modality} and controls[key] < 1:
                raise ValueError(f"{key} must be positive for enabled fixed-count examples")
    if not math.isfinite(controls["HYBRID_ALPHA"]) or not 0 <= controls["HYBRID_ALPHA"] <= 1:
        raise ValueError("HYBRID_ALPHA must be in [0,1]")
    if controls["QWEN_MIN_PIXELS"] and controls["QWEN_MAX_PIXELS"] and controls["QWEN_MIN_PIXELS"] > controls["QWEN_MAX_PIXELS"]:
        raise ValueError("QWEN_MIN_PIXELS must not exceed QWEN_MAX_PIXELS")
    if (controls["USE_FREERET_FULL_PROMPT"] or controls["USE_FREERET_PROMPT_CONSTRAINTS"]) and not controls["USE_FREERET_DENSE"]:
        raise ValueError("FreeRet prompt modes require USE_FREERET_DENSE=true")
    if controls["SHUFFLE_VIEW_OPTIONS"] and not controls["VIEW_SELECTION_FILE"] and not controls["VIEW_SELECTION_RUN_ID"]:
        controls["VIEW_SELECTION_RUN_ID"] = uuid4().hex[:16]
    for key in ("DATASET_SPLIT", "VIEW_SELECTION_SPLIT"):
        value = controls[key] or controls["DATASET_SPLIT"]
        if value in {".", ".."} or any(char in value for char in "/\\"):
            raise ValueError(f"Invalid {key}: {value!r}")
    root = path(env.get("OUTPUT_ROOT") or "./adaptive_topk_results")
    data_root = path(env.get("DATA_ROOT") or "/root/autodl-tmp")
    selection_root = path(env.get("SPARSE_OUTPUT_ROOT") or "./sparse_output")
    checkpoint_root = Path(path(env.get("CHECKPOINT_ROOT") or "/root/autodl-fs/checkpoints"))
    jobs = []
    for family in families:
        model = path(env.get("MODEL_PATH") or env.get("MODEL_PATH_" + family.upper()) or str(checkpoint_root / CHECKPOINTS[family]))
        if controls["LLAVA_IMAGE_MODE"] == "base" and not family.startswith("llava_"):
            raise ValueError("LLAVA_IMAGE_MODE=base requires LLaVA-NeXT families only; use anyres for Qwen")
        for dataset in datasets:
            task = TASKS[dataset]
            for query in directions if task == "ir" else ("image" if task == "it2t" else "text",):
                job = dict(controls, model_family=family, model_path=model, dataset=dataset, task=task, query_type=query,
                           project=str(project), output_root=root, data_root=data_root, selection_root=selection_root)
                if job["VIEW_SELECTION_FILE"]:
                    job["VIEW_SELECTION_FILE"] = path(job["VIEW_SELECTION_FILE"])
                if not controls["STOP_AFTER_STAGE2"]:
                    job.update(RANKER_MODEL_PATH=path(env.get("RANKER_MODEL_PATH") or env.get("RANKER_MODEL_PATH_" + family.upper()) or model),
                        RANKER_DTYPE=env.get("RANKER_DTYPE") or controls["DTYPE"],
                        RANKER_LLAVA_IMAGE_MODE=env.get("RANKER_LLAVA_IMAGE_MODE") or controls["LLAVA_IMAGE_MODE"],
                        RANKER_QWEN_MIN_PIXELS=int(env.get("RANKER_QWEN_MIN_PIXELS") or controls["QWEN_MIN_PIXELS"]),
                        RANKER_QWEN_MAX_PIXELS=int(env.get("RANKER_QWEN_MAX_PIXELS") or controls["QWEN_MAX_PIXELS"]))
                    if (job["RANKER_DTYPE"] not in choices["DTYPE"] or job["RANKER_LLAVA_IMAGE_MODE"] not in choices["LLAVA_IMAGE_MODE"]
                            or min(job["RANKER_QWEN_MIN_PIXELS"], job["RANKER_QWEN_MAX_PIXELS"]) < 0
                            or min(job["RERANK_NUM"], job["RERANK_BATCH_SIZE"]) < 1):
                        raise ValueError("Invalid ranker configuration")
                    if job["RANKER_QWEN_MIN_PIXELS"] and job["RANKER_QWEN_MAX_PIXELS"] and job["RANKER_QWEN_MIN_PIXELS"] > job["RANKER_QWEN_MAX_PIXELS"]:
                        raise ValueError("RANKER_QWEN_MIN_PIXELS must not exceed RANKER_QWEN_MAX_PIXELS")
                jobs.append(job)
    return jobs


def pipeline_args(job, api):
    args = api._parser().parse_args(["--dataset-name", job["dataset"], "--task-type", job["task"], "--query-type", job["query_type"]])
    mapping = {"DATASET_SPLIT": "dataset_split", "REPS_LOC": "retrieval_reps_loc", "DTYPE": "retrieval_dtype",
               "LLAVA_IMAGE_MODE": "retrieval_llava_image_mode", "QWEN_MIN_PIXELS": "retrieval_qwen_min_pixels",
               "QWEN_MAX_PIXELS": "retrieval_qwen_max_pixels", "RETRIEVAL_BATCH_SIZE": "retrieval_batch_size"}
    for key in ("SPARSE_LENGTH", "TEXT_SPARSE_LENGTH", "IMAGE_SPARSE_LENGTH", "SPARSE_VALUE_TYPE", "SPARSE_LOWER_OR_UPPER",
                "SPARSE_VALUE_MEAN", "IS_FILTERED", "NUM_EXPENDED_TOKENS", "TBPR_TYPE", "STAGE1_DEPTH", "STAGE2_DEPTH",
                "SPARSE_THREADS", "HYBRID_ALPHA", "USE_FREERET_DENSE", "USE_FREERET_PROMPT_CONSTRAINTS", "USE_FREERET_FULL_PROMPT"):
        mapping[key] = key.lower()
    for key, target in mapping.items():
        setattr(args, target, job[key])
    args.data_root, args.retrieval_model_name_or_path = job["data_root"], job["model_path"]
    args.retrieval_device_map = "none"
    args.adaptive_views = True  # Enables the same SDPA loading path; old preflight/main are never called.
    args.rerank_template = "what_caption_generation"
    args.modify_type = "modify_single" if job["task"] == "it2t" or (job["task"] == "ir" and job["query_type"] == "image") else "no"
    if not job["STOP_AFTER_STAGE2"]:
        for key, target in {"RANKER_MODEL_PATH": "ranker_model_name_or_path", "RANKER_DTYPE": "ranker_dtype",
                            "RANKER_LLAVA_IMAGE_MODE": "ranker_llava_image_mode", "RANKER_QWEN_MIN_PIXELS": "ranker_qwen_min_pixels",
                            "RANKER_QWEN_MAX_PIXELS": "ranker_qwen_max_pixels", "RERANK_NUM": "rerank_num",
                            "RERANK_BATCH_SIZE": "rerank_batch_size"}.items():
            setattr(args, target, job[key])
        args.ranker_device_map = "none"
    return args


def selection_options(job, views):
    options = views.AdaptiveViewArguments()
    mapping = {"USE_VIEW_PMI": "view_selection_pmi", "USE_HIERARCHICAL_VIEWS": "view_selection_hierarchical",
               "SHUFFLE_VIEW_OPTIONS": "view_selection_shuffle", "VIEW_SELECTION_RUN_ID": "view_selection_shuffle_run_id"}
    for key in ("VIEW_SELECTION_MODALITY", "VIEW_SELECTION_ROUNDS", "VIEW_TEXTS_PER_ROUND", "VIEW_IMAGES_PER_ROUND",
                "VIEW_SELECTION_SEED", "VIEW_EXAMPLE_MODE", "VIEW_EXAMPLE_MAX_TOKENS"):
        mapping[key] = key.lower()
    for key, target in mapping.items():
        if job[key] is not None:
            setattr(options, target, job[key])
    return options


def resolve_selection(job, args, identity, config, get_runtime, api):
    import adaptive_views as views
    import adaptive_view_encoding as adaptive
    import adaptive_view_shuffle as shuffle
    options = selection_options(job, views)
    explicit = job["VIEW_SELECTION_FILE"]
    if explicit:
        selection = views.load_selection(explicit, identity=identity, dataset=job["dataset"],
            selection_modality=job["VIEW_SELECTION_MODALITY"], view_selection_pmi=job["USE_VIEW_PMI"],
            view_selection_hierarchical=job["USE_HIERARCHICAL_VIEWS"])
        mapping = selection.get(shuffle.FIELD)
        if job["VIEW_SELECTION_RUN_ID"] and (mapping is None or mapping["run_id"] != shuffle.validate_run_id(job["VIEW_SELECTION_RUN_ID"])):
            raise ValueError("VIEW_SELECTION_FILE does not match VIEW_SELECTION_RUN_ID")
        if job["SHUFFLE_VIEW_OPTIONS"] and mapping is None:
            raise ValueError("SHUFFLE_VIEW_OPTIONS=true requires a source file with a recorded option mapping")
        # Selection controls belong to the immutable source file. All seven groups inherit them.
        return selection, str(Path(explicit).resolve())
    mode = views.resolve_modality(options.view_selection_modality)
    sampling = views.selection_sampling(options, mode, model_config=config)
    candidate_mode = views.options_pool_mode(options)
    permutation = shuffle.for_options(options, identity, job["dataset"], views.view_pool(candidate_mode))
    source_split = job["VIEW_SELECTION_SPLIT"] or job["DATASET_SPLIT"]
    key = views.selection_cache_key(identity, job["dataset"], source_split, mode, sampling, job["data_root"],
        pmi=bool(options.view_selection_pmi), shuffle=permutation, pool_mode=candidate_mode)
    location = views.selection_cache_path(job["selection_root"], job["model_path"], job["dataset"], key)
    if location.is_file():
        selection = views.load_selection(location, identity=identity, dataset=job["dataset"], selection_modality=mode,
            view_selection_pmi=bool(options.view_selection_pmi), view_selection_hierarchical=bool(options.view_selection_hierarchical),
            view_example_mode=options.view_example_mode, view_example_max_tokens=options.view_example_max_tokens)
        views.require_equal(selection["sampling"], sampling, "Top-K selection sampling")
        views.require_equal(dict(mapping=selection.get(shuffle.FIELD)), dict(mapping=permutation), "Top-K option mapping")
        return selection, str(location.resolve())
    source_args = copy.copy(args)
    source_args.dataset_split = source_split
    texts, images = adaptive.source_pools(source_args, api, mode)
    runtime = get_runtime()
    selection = views.choose_views(runtime.encoder, runtime.processor, identity, job["dataset"], source_split, texts, images, options)
    views.atomic_json(location, selection)
    views.atomic_json(location.with_name("view_pool.json"), selection["pool"])
    if permutation:
        views.atomic_json(location.with_name("option_mapping.json"), permutation)
    return selection, str(location.resolve())


def ranked_views(selection):
    """Recheck full canonical scores, then deterministically retain all candidates."""
    import numpy as np
    import adaptive_views as views
    mode = selection["selection_modality"]
    active = views.enabled_modalities(mode)
    matrices = {name: [row[name + "_probabilities"] for row in selection["rounds"]] if name in active else None
                for name in ("text", "image")}
    ranking = views.rank_probabilities(matrices["text"], matrices["image"], selection["pool"], selection_modality=mode)
    if not np.allclose(selection["scores"], ranking["scores"], atol=1e-7):
        raise ValueError("Saved candidate probabilities differ from their per-round aggregate")
    order = sorted(range(len(selection["pool"])), key=lambda index: (-ranking["scores"][index], index))
    if len(order) < max(COUNTS):
        raise ValueError("Top-1..7 experiment requires at least seven candidate perspectives")
    return [dict(rank=rank + 1, **selection["pool"][index], probability=ranking["scores"][index],
                 text_mean=None if ranking["text_mean"] is None else ranking["text_mean"][index],
                 image_mean=None if ranking["image_mean"] is None else ranking["image_mean"][index])
            for rank, index in enumerate(order)]


def prefix_selection(source, ranking, count):
    """Internal encoder input, never passed to the old five-view file loader."""
    import adaptive_views as views
    import adaptive_view_shuffle as shuffle
    result = copy.deepcopy(source)
    result["selected"] = [dict(label=row["label"], aspect=row["aspect"]) for row in ranking[:count]]
    result["views_fingerprint"] = views.views_fingerprint(result["selected"], source["selection_modality"],
        pmi=views.is_pmi_selection(source), shuffle=source.get(shuffle.FIELD), pool_mode=views.pool_mode(source),
        example_configuration=source.get(views.examples.FIELD))
    result["fingerprint"] = views.digest({key: value for key, value in result.items() if key != "fingerprint"})
    return result


def prefix_vectors(encoder, records, logits, converter=None):
    if converter is None:
        from adaptive_view_encoding import sparse_vectors as converter
    if any(len(row) != max(COUNTS) for row in logits):
        raise ValueError("Expected exactly seven unmerged sparse-view logits per sample")
    return {count: converter(encoder, records, [row[:count] for row in logits]) for count in COUNTS}


def artifact_paths(root, job):
    from dense_representation import mode_path
    def mode(path):
        return Path(mode_path(path, job["USE_FREERET_DENSE"], job["USE_FREERET_PROMPT_CONSTRAINTS"], job["USE_FREERET_FULL_PROMPT"]))
    return mode(Path(root) / "dense"), {count: mode(Path(root) / f"top_{count}" / "sparse") for count in COUNTS}


def encode_artifacts(job, args, context, runtime, selection, ranking, root, configuration, api):
    """Stream one corpus pass into seven JSONL collections; keep logits batch-local."""
    import numpy as np
    import adaptive_views as views
    import adaptive_view_encoding as adaptive
    from dense_representation import write_manifest
    from qwen_vl_pipeline import dataset_records, corpus_kind
    target = Path(root) / "artifacts"
    ready = adaptive.root_action(context.rank, lambda: completed(target, configuration))
    if ready:
        return target
    staging = Path(adaptive.root_action(context.rank, lambda: tempfile.mkdtemp(prefix=".encoding-", dir=root)))
    try:
        model_args, data = api._retrieval_namespaces(args, eol_type="disassembleeol_separate", calculate_type="concat", sparse_manual=False)
        data.per_device_batch_size = job["PER_DEVICE_BATCH_SIZE"]
        encoder = rank_action(lambda: adaptive.AdaptiveEncoder(runtime.encoder, runtime.processor, model_args, data, job["task"],
                                           prefix_selection(selection, ranking, max(COUNTS))))
        dense_root, sparse_roots = artifact_paths(staging, job)
        def prepare():
            dense_root.mkdir(parents=True)
            (staging / "queries").mkdir()
            for count in COUNTS:
                (staging / "queries" / f"top_{count}").mkdir()
            for sparse in sparse_roots.values():
                (sparse / "collection").mkdir(parents=True)
        adaptive.root_action(context.rank, prepare)
        error = None
        try:
            dataset = api.build_dataset(args)
            corpus = dataset_records(dataset, job["task"], corpus_kind(job["task"], job["query_type"]),
                prompt_constraints_mode=bool(job["USE_FREERET_PROMPT_CONSTRAINTS"] or job["USE_FREERET_FULL_PROMPT"]))[context.rank::context.world_size]
            hidden_size = getattr(runtime.encoder.config, "text_config", runtime.encoder.config).hidden_size
            dense_parts, identifiers = [], []
            with ExitStack() as stack:
                streams = {count: stack.enter_context((sparse / "collection" / f"corpus_{context.rank}.jsonl").open("w", encoding="utf-8"))
                           for count, sparse in sparse_roots.items()}
                for start in api._progress(range(0, len(corpus), data.per_device_batch_size), desc="[topk] corpus encoding", unit="batch"):
                    batch = corpus[start:start + data.per_device_batch_size]
                    logits, dense = encoder.encode(batch)
                    vectors = prefix_vectors(encoder, batch, logits)
                    dense_parts.append(dense.float().cpu().numpy())
                    identifiers.extend(str(record["id"]) for record in batch)
                    for count, rows in vectors.items():
                        for record, vector in zip(batch, rows):
                            streams[count].write(json.dumps(dict(id=str(record["id"]), contents="", vector=vector), ensure_ascii=False) + "\n")
            dense = np.concatenate(dense_parts) if dense_parts else np.empty((0, hidden_size), dtype="float32")
            with (dense_root / f"corpus_{context.rank}.pkl").open("wb") as stream:
                pickle.dump((dense, identifiers), stream)
            records = api._qwen_query_records(args, api._query_records(args, dataset))[context.rank::context.world_size]
            ids, dense_parts = [], []
            # Store compact vectors, not seven sets of weight-expanded topic strings.
            # Only the current prefix is expanded when its retrieval actually starts.
            with ExitStack() as stack:
                streams = {count: stack.enter_context((staging / "queries" / f"top_{count}" / f"query_{context.rank}.jsonl").open("w", encoding="utf-8"))
                           for count in COUNTS}
                for start in api._progress(range(0, len(records), job["RETRIEVAL_BATCH_SIZE"]), desc="[topk] query encoding", unit="batch"):
                    batch = records[start:start + job["RETRIEVAL_BATCH_SIZE"]]
                    logits, dense = encoder.encode(batch)
                    vectors = prefix_vectors(encoder, batch, logits)
                    ids.extend(str(record["id"]) for record in batch)
                    dense_parts.append(dense.float().cpu().numpy())
                    for count, rows in vectors.items():
                        for record, vector in zip(batch, rows):
                            streams[count].write(json.dumps(dict(id=str(record["id"]), vector=vector), ensure_ascii=False) + "\n")
            dense = np.concatenate(dense_parts) if dense_parts else np.empty((0, hidden_size), dtype="float32")
            with (staging / "queries" / f"query_{context.rank}.pkl").open("wb") as stream:
                pickle.dump((ids, dense), stream)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        adaptive.worker_errors(error)
        def publish():
            for path in (dense_root, *sparse_roots.values()):
                write_manifest(path, job["USE_FREERET_DENSE"], job["USE_FREERET_PROMPT_CONSTRAINTS"], job["USE_FREERET_FULL_PROMPT"])
            files = [path for path in staging.rglob("*") if path.is_file()]
            stamp(staging, configuration, files)
            if target.exists():
                raise ValueError(f"Uncompleted artifacts already exist; choose another OUTPUT_ROOT: {target}")
            staging.rename(target)
        adaptive.root_action(context.rank, publish)
        return target
    finally:
        if context.is_main and staging.exists():
            if staging.resolve().parent != Path(root).resolve() or not staging.name.startswith(".encoding-"):
                raise ValueError(f"Refusing to clean outside the owned staging directory: {staging}")
            shutil.rmtree(staging)


def ensure_index(root, configuration, *, threads=16):
    from qwen_sparse_index import build_sparse_index
    root = Path(root)
    name = "topk_index_manifest.json"
    if completed(root, configuration, name):
        return
    index = root / "collection" / "index"
    if index.exists():
        raise ValueError(f"Index exists without a matching completion marker: {index}; choose another OUTPUT_ROOT")
    build_sparse_index(root, threads=threads)
    stamp(root, configuration, [item for item in index.rglob("*") if item.is_file() and item.name != "write.lock"], name)


def retrieve_prefix(args, encoded, sparse_root, dense_root, api):
    """Same three-stage primitives, with independently validated Top-K artifacts."""
    api._validate_encoded_queries(encoded)
    first, second = api.StageTimer(), api.StageTimer()
    stage1 = api.retrieve_sparse_topics_from_lucene(Path(sparse_root), encoded.sparse_topics, encoded.query_ids,
        depth=args.stage1_depth, threads=args.sparse_threads, progress_prefix="[stage1]", timing=first)
    candidates = {query: set(api._top_ids(scores, args.stage1_depth)) for query, scores in stage1.items()}
    if not stage1:
        return stage1, {}, {}, {}, first, second
    dense = api.retrieve_dense_from_corpus(encoded.dense_representations, encoded.query_ids, Path(dense_root), list(stage1),
        depth=args.stage2_depth, candidate_filter=candidates, normalize=not args.no_dense_normalize, timing=second)
    sparse = api.retrieve_sparse_topics_from_lucene(Path(sparse_root), encoded.sparse_topics, list(stage1),
        depth=max(args.stage2_depth, args.stage1_depth), threads=args.sparse_threads, progress_prefix="[stage2]", timing=second)
    hybrid = api.fuse_restricted_runs(stage1, dense, sparse, args.hybrid_alpha, args.stage1_depth, timing=second)
    return stage1, dense, sparse, hybrid, first, second


def summary_rows(job, ranking, count, report, result_dir):
    return [dict(model_family=job["model_family"], model=job["model_path"], dataset=job["dataset"],
                 split=job["DATASET_SPLIT"], task=job["task"], query_type=job["query_type"], view_count=count,
                 selected_labels=" ".join(row["label"] for row in ranking[:count]),
                 selected_aspects=json.dumps([row["aspect"] for row in ranking[:count]], ensure_ascii=False),
                 execution_mode="two_stage" if job["STOP_AFTER_STAGE2"] else "three_stage", hybrid_alpha=job["HYBRID_ALPHA"],
                 stage=row["stage"], method=row["variant"], category=row["category"], query_count=row["query_count"],
                 **{f"R@{key}": value for key, value in row["recall"].items()}, result_dir=str(result_dir))
            for row in report["results"]]


def publish_summary(job, rows, experiment):
    write_csv(Path(experiment) / "topk_summary.csv", rows)
    if job.get("batch_dir"):
        batch = Path(job["batch_dir"])
        atomic_json(batch / f"summary_{job['job_index']:05d}.json", rows)
        combined = []
        for path in sorted(batch.glob("summary_*.json")):
            combined.extend(json.loads(path.read_text(encoding="utf-8")))
        write_csv(Path(job["output_root"]) / "topk_summary.csv", combined)


def run_prefixes(job, args, context, selection, ranking, experiment, artifacts, encoding_config, api):
    import adaptive_view_encoding as adaptive
    import adaptive_views as views
    dense_root, sparse_roots = artifact_paths(artifacts, job)
    def read_queries():
        with (Path(artifacts) / "queries" / f"query_{context.rank}.pkl").open("rb") as stream:
            return pickle.load(stream)
    ids, dense = rank_action(read_queries)
    summary = []
    for count in COUNTS:
        configuration = dict(version=VERSION, encoding=digest(encoding_config), count=count,
            stop_after_stage2=job["STOP_AFTER_STAGE2"], stage1_depth=args.stage1_depth, stage2_depth=args.stage2_depth,
            alpha=args.hybrid_alpha, sparse_threads=args.sparse_threads,
            ranker_identity=None if job["STOP_AFTER_STAGE2"] else views.checkpoint_identity(args.ranker_model_name_or_path),
            ranker={} if job["STOP_AFTER_STAGE2"] else {key: value for key, value in vars(args).items()
                if key.startswith("ranker_") or key.startswith("rerank_") or key == "modify_type"})
        mode = "two_stage" if job["STOP_AFTER_STAGE2"] else "three_stage"
        result_dir = Path(experiment) / f"top_{count}" / f"stage1_{args.stage1_depth}_stage2_{args.stage2_depth}" / mode / digest(configuration)[:16]
        ready = adaptive.root_action(context.rank, lambda: completed(result_dir, configuration))
        if ready:
            if context.is_main:
                report = json.loads((result_dir / "recall_metrics.json").read_text(encoding="utf-8"))
                summary.extend(summary_rows(job, ranking, count, report, result_dir))
                publish_summary(job, summary, experiment)
            continue
        adaptive.root_action(context.rank, lambda: ensure_index(sparse_roots[count],
            dict(version=VERSION, encoding=digest(encoding_config), count=count), threads=job["SPARSE_THREADS"]))
        adaptive.root_action(context.rank, lambda: result_dir.mkdir(parents=True, exist_ok=True))
        def read_topics():
            topics = {}
            with (Path(artifacts) / "queries" / f"top_{count}" / f"query_{context.rank}.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["id"] in topics:
                        raise ValueError(f"Duplicate query ID: {row['id']}")
                    topics[row["id"]] = api._vector_to_topic(row["vector"])
            return topics
        encoded = api.EncodedQueries(ids, dense, rank_action(read_topics))
        error, local = None, None
        try:
            local = retrieve_prefix(args, encoded, sparse_roots[count], dense_root, api)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        adaptive.worker_errors(error)
        del encoded  # Release expanded topics before the next prefix or reranker.
        first, dense_run, sparse_run, hybrid_run = [api._gather_run(run, context) for run in local[:4]]
        timers = {"stage1": api._collect_stage_timing(local[4], context), "stage2": api._collect_stage_timing(local[5], context)}
        reporter = None
        def save_retrieval():
            nonlocal reporter
            if not first:
                raise ValueError("No query IDs available for Top-K evaluation")
            api._validate_query_coverage(first, hybrid_run, "Stage 2")
            reporter = api.RecallReporter(args, first, result_dir)
            for name, run in (("stage1", first), ("stage2_dense", dense_run), ("stage2_sparse", sparse_run), ("stage2_hybrid", hybrid_run)):
                api.write_trec_run(run, result_dir / f"{name}_run.trec", name)
                api.write_json_run(run, result_dir / f"{name}_run.json")
            reporter.record("stage1", {"sparse_concat": first})
            reporter.record("stage2", {"dense": dense_run, "sparse": sparse_run, "hybrid": hybrid_run})
            report = json.loads((result_dir / "recall_metrics.json").read_text(encoding="utf-8"))
            publish_summary(job, summary + summary_rows(job, ranking, count, report, result_dir), experiment)
        adaptive.root_action(context.rank, save_retrieval)
        if not job["STOP_AFTER_STAGE2"]:
            timer = api.StageTimer()
            error, local_ranker = None, None
            try:
                local_ranker = api.run_stage3(args, local[3], context, timing=timer)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            adaptive.worker_errors(error)
            ranked = api._gather_run(local_ranker, context)
            timers["stage3"] = api._collect_stage_timing(timer, context)
            def save_ranker():
                api._validate_query_coverage(first, ranked, "Stage 3")
                api.write_trec_run(ranked, result_dir / "stage3_ranker_run.trec", "stage3_ranker")
                api.write_json_run(ranked, result_dir / "stage3_ranker_run.json")
                reporter.record("stage3", {"ranker": ranked})
            adaptive.root_action(context.rank, save_ranker)
        def finish():
            atomic_json(result_dir / "pipeline_metadata.json", dict(configuration=configuration,
                selection_fingerprint=selection["fingerprint"], selected=ranking[:count],
                stage3_status="skipped" if job["STOP_AFTER_STAGE2"] else "completed", dense_shared=True,
                artifacts=str(artifacts), timing_scope="retrieval/reranking compute; excludes shared encoding, setup and I/O"))
            atomic_json(result_dir / "timing_metrics.json", timers)
            write_csv(result_dir / "timing_metrics.csv", [dict(stage=stage, seconds=value["compute_seconds"])
                for stage, value in timers.items()])
            stamp(result_dir, configuration, [file for file in result_dir.iterdir() if file.is_file() and file.name != "complete.json"])
            report = json.loads((result_dir / "recall_metrics.json").read_text(encoding="utf-8"))
            summary.extend(summary_rows(job, ranking, count, report, result_dir))
            publish_summary(job, summary, experiment)
            print(f"[topk] completed Top-{count}: {result_dir}", flush=True)
        adaptive.root_action(context.rank, finish)


def run_worker(job, local_rank=-1):
    import adaptive_view_encoding as adaptive
    import adaptive_views as views
    from transformers import AutoConfig
    api = adaptive.pipeline_module()
    args = pipeline_args(job, api)
    context = api._initialise_distributed(local_rank)
    runtime = None
    successful = False
    def get_runtime():
        nonlocal runtime
        if runtime is None:
            runtime = api._load_retrieval_runtime(args, context)
            adaptive.configure_processor(runtime.encoder.config, runtime.processor, args.retrieval_reps_loc)
        return runtime
    try:
        def load_identity():
            identity = adaptive.load_model_identity(job["model_path"], dtype=job["DTYPE"], reps_loc=job["REPS_LOC"],
                min_pixels=job["QWEN_MIN_PIXELS"], max_pixels=job["QWEN_MAX_PIXELS"], image_mode=job["LLAVA_IMAGE_MODE"])
            return identity, AutoConfig.from_pretrained(job["model_path"], local_files_only=True)
        identity, config = rank_action(load_identity)
        selection, source_path = adaptive.root_action(context.rank,
            lambda: resolve_selection(job, args, identity, config, get_runtime, api))
        if views.is_overlap_selection(selection):
            raise ValueError("Use a probability-only encode_adaptive_views selection, not an overlap-vote selection")
        ranking = ranked_views(selection)
        model_args, data = api._retrieval_namespaces(args, eol_type="disassembleeol_separate", calculate_type="concat", sparse_manual=False)
        from qwen_vl_pipeline import corpus_kind
        configuration = dict(version=VERSION, model=identity, task=job["task"], query_type=job["query_type"],
            dataset=job["dataset"], split=job["DATASET_SPLIT"], data_root=job["data_root"], source_fingerprint=selection["fingerprint"],
            counts=list(COUNTS), selected=ranking[:7], world_size=context.world_size,
            encoding=adaptive.encoding_configuration(model_args, data, job["task"], corpus_kind(job["task"], job["query_type"]),
                identity, prefix_selection(selection, ranking, 7)),
            prompt_sources={name: file_hash(PROJECT / "src" / name) for name in (
                "freeret_prompt_catalog.py", "freeret_prompt_constraints.py", "freeret_full_prompt.py")})
        experiment = Path(job["output_root"]) / job["model_family"] / Path(job["model_path"]).name / job["dataset"] / job["query_type"] / job["DATASET_SPLIT"] / digest(configuration)[:16]
        def prepare():
            experiment.mkdir(parents=True, exist_ok=True)
            atomic_json(experiment / "experiment.json", dict(configuration=configuration, fingerprint=digest(configuration), source_file=source_path))
            atomic_json(experiment / "view_selection.json", selection)
            atomic_json(experiment / "view_ranking.json", ranking)
            write_csv(experiment / "view_ranking.csv", ranking)
            for count in COUNTS:
                atomic_json(experiment / f"top_{count}" / "selected_views.json", dict(count=count,
                    source_fingerprint=selection["fingerprint"], selected=ranking[:count]))
            print(f"[topk] experiment: {experiment}\n[topk] source: {source_path}", flush=True)
            for row in ranking[:7]:
                print(f"[topk] rank={row['rank']} {row['label']}: {row['aspect']} probability={row['probability']:.8f}", flush=True)
        adaptive.root_action(context.rank, prepare)
        ready = adaptive.root_action(context.rank, lambda: completed(experiment / "artifacts", configuration))
        if not ready:
            error = None
            try:
                get_runtime()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            adaptive.worker_errors(error)
            def check_identity():
                actual = views.model_identity(job["model_path"], runtime.encoder.config, runtime.processor, dtype=job["DTYPE"])
                views.require_equal(actual, identity, "Top-K runtime model/processor")
            rank_action(check_identity)
            artifacts = encode_artifacts(job, args, context, runtime, selection, ranking, experiment, configuration, api)
        else:
            artifacts = experiment / "artifacts"
        api._release_retrieval_runtime(runtime)
        runtime = None
        run_prefixes(job, args, context, selection, ranking, experiment, artifacts, configuration, api)
        successful = True
    finally:
        api._release_retrieval_runtime(runtime)
        api._shutdown_distributed(context, synchronize=successful)


def run_batch(env, *, runner=subprocess.run):
    jobs = build_jobs(env)
    from adaptive_encode_batch import annotation_files
    runnable, missing = [], []
    for job in jobs:
        required = [Path(job["model_path"]) / "config.json",
                    *annotation_files(Path(job["data_root"]), job["dataset"], job["DATASET_SPLIT"])]
        if job["VIEW_SELECTION_FILE"]:
            required.append(Path(job["VIEW_SELECTION_FILE"]))
        else:
            required.extend(annotation_files(Path(job["data_root"]), job["dataset"], job["VIEW_SELECTION_SPLIT"] or job["DATASET_SPLIT"]))
        if not job["STOP_AFTER_STAGE2"]:
            required.append(Path(job["RANKER_MODEL_PATH"]) / "config.json")
        absent = [str(item) for item in dict.fromkeys(required) if not item.is_file()]
        (missing if absent else runnable).append((job, absent) if absent else job)
    for job, absent in missing:
        print(f"Skip missing {job['model_family']}/{job['dataset']}/{job['query_type']}: {absent}", flush=True)
    if not runnable:
        raise ValueError("No runnable Top-K jobs; check model config.json and dataset annotation paths")
    dry = boolean(env.get("DRY_RUN", "false"))
    batch = Path(runnable[0]["output_root"]) / ".batches" / uuid4().hex
    if not dry:
        atomic_json(batch / "batch_plan.json", runnable)
    for index, job in enumerate(runnable, 1):
        job = dict(job, batch_dir=str(batch), job_index=index)
        path = batch / f"job_{index:05d}.json"
        command = [env.get("DEEPSPEED_BIN") or "deepspeed", f"--num_gpus={job['GPUS']}", f"--num_nodes={job['NUM_NODES']}",
                   str(PROJECT / "src" / "adaptive_views_topk_experiment.py"), "--worker", str(path)]
        print(f"[topk] job {index}/{len(runnable)}: {job['model_family']}/{job['dataset']}/{job['query_type']}; "
              f"Top-1..7; {'two stages' if job['STOP_AFTER_STAGE2'] else 'three stages'}", flush=True)
        print(json.dumps(dict(command=command, configuration=job), ensure_ascii=False), flush=True)
        if dry:
            continue
        atomic_json(path, job)
        try:
            result = runner(command, cwd=job["project"], env=dict(os.environ, **env), check=False)
        except OSError as exc:
            raise ValueError(f"Cannot launch DeepSpeed: {exc}") from exc
        if result.returncode:
            print(f"[topk] failed job {index}; completed CSV results are preserved", file=sys.stderr)
            return result.returncode if result.returncode > 0 else 128 - result.returncode
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.worker:
            run_worker(json.loads(args.worker.read_text(encoding="utf-8")), args.local_rank)
            return 0
        return run_batch(dict(os.environ))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
