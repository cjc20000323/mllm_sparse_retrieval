"""Resolve one adaptive experiment by its encoding settings and run three-stage retrieval."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from types import SimpleNamespace

import adaptive_views as views
import adaptive_view_shuffle as option_shuffle
from adaptive_encode_batch import (CHECKPOINTS, GROUPS, PROJECT, annotation_files, boolean,
                                   configured_list, encoding_controls, resolved_path, split_name)
from qwen_vl_paths import resolve_sparse_index


FAMILIES = dict(qwen2="qwen2_vl", qwen25="qwen2_5_vl", qwen3="qwen3_vl",
                llava_llama="llava_llama", llava_mistral="llava_mistral")
RETRIEVAL_DEFAULTS = {
    "STAGE1_DEPTH": "1000", "STAGE2_DEPTH": "1000", "SPARSE_THREADS": "16",
    "RETRIEVAL_BATCH_SIZE": "1", "RERANK_NUM": "5", "RERANK_BATCH_SIZE": "1",
    "HYBRID_ALPHA": "0.5", "SELECT_HYBRID_ALPHA": "false", "OVERWRITE": "true",
}
PIPELINE_NAMES = (
    "VIEW_EXAMPLE_MODE", "VIEW_EXAMPLE_MAX_TOKENS",
    "VIEW_SELECTION_RUN_ID",
    "USE_HIERARCHICAL_VIEWS",
    "USE_FREERET_DENSE", "USE_FREERET_PROMPT_CONSTRAINTS", "USE_FREERET_FULL_PROMPT", "GPUS", "NUM_NODES", "DEEPSPEED_BIN", "DTYPE", "LLAVA_IMAGE_MODE", "RETRIEVAL_DTYPE", "RANKER_DTYPE",
    "RETRIEVAL_LLAVA_IMAGE_MODE", "RANKER_LLAVA_IMAGE_MODE", "REPS_LOC", "SPARSE_LENGTH", "TEXT_SPARSE_LENGTH",
    "IMAGE_SPARSE_LENGTH", "SPARSE_VALUE_TYPE", "SPARSE_LOWER_OR_UPPER", "SPARSE_VALUE_MEAN", "IS_FILTERED",
    "NUM_EXPENDED_TOKENS", "TBPR_TYPE", "QWEN_MIN_PIXELS", "QWEN_MAX_PIXELS",
    "RANKER_QWEN_MIN_PIXELS", "RANKER_QWEN_MAX_PIXELS", *RETRIEVAL_DEFAULTS,
)
VALIDATION_INPUTS = ("VAL_PASSAGE_REPS", "VAL_SPARSE_INDEX", "VAL_VIEW_SELECTION_FILE",
                     "VALIDATION_DATASET_NAME", "VALIDATION_SPLIT", "VAL_STAGE1_SPARSE_INDEX")


def selection_options(env, controls):
    from adaptive_view_overlap import OverlapViewArguments
    options = OverlapViewArguments(
        adaptive_views=True, view_selection_modality=controls.get("VIEW_SELECTION_MODALITY", "both"),
        view_example_mode=controls["VIEW_EXAMPLE_MODE"], view_example_max_tokens=int(controls["VIEW_EXAMPLE_MAX_TOKENS"]),
        view_selection_hierarchical=boolean(controls["USE_HIERARCHICAL_VIEWS"], "USE_HIERARCHICAL_VIEWS"),
        view_selection_pmi=boolean(env.get("USE_VIEW_PMI", "false"), "USE_VIEW_PMI"),
        view_overlap_selection=boolean(env.get("USE_VIEW_OVERLAP_SELECTION", "false"), "USE_VIEW_OVERLAP_SELECTION"),
        **{name.lower(): int(controls[name]) for name in (
            "VIEW_SELECTION_ROUNDS", "VIEW_TEXTS_PER_ROUND", "VIEW_IMAGES_PER_ROUND", "VIEW_SELECTION_SEED")})
    if controls["VIEW_SELECTION_RUN_ID"]:
        options.view_selection_shuffle = True
        options.view_selection_shuffle_run_id = option_shuffle.validate_run_id(controls["VIEW_SELECTION_RUN_ID"])
    for name, convert in (("VIEW_SELECTED_COUNT", int), ("VIEW_OVERLAP_THRESHOLD", float),
                          ("VIEW_OVERLAP_SAMPLES", int), ("VIEW_OVERLAP_SEED", int)):
        if env.get(name):
            try:
                setattr(options, name.lower(), convert(env[name]))
            except ValueError as exc:
                raise ValueError(f"Invalid {name}: {env[name]!r}") from exc
    views.examples.validate_options(options)
    if options.view_selection_rounds < 1:
        raise ValueError("VIEW_SELECTION_ROUNDS must be positive")
    if options.view_example_mode == "fixed":
        views.selection_sampling(options, options.view_selection_modality)
    return options


def pipeline_controls(env):
    # Retrieval computes its own selection paths; a leftover single-job path must
    # not alter sampling validation or determine which experiment is selected.
    controls = encoding_controls({k: v for k, v in env.items() if k != "VIEW_SELECTION_FILE"})
    controls["VIEW_SELECTION_RUN_ID"] = env.get("VIEW_SELECTION_RUN_ID", "")
    if controls["SHUFFLE_VIEW_OPTIONS"] == "true" and not controls["VIEW_SELECTION_RUN_ID"]:
        raise ValueError("Shuffled retrieval needs VIEW_SELECTION_RUN_ID from encoding; retrieval never generates a new permutation")
    controls.update({name: env.get(name) or default for name, default in RETRIEVAL_DEFAULTS.items()})
    for role in ("RETRIEVAL", "RANKER"):
        name = f"{role}_DTYPE"
        controls[name] = env.get(name) or controls["DTYPE"]
        if controls[name] not in {"float16", "bfloat16", "float32"}:
            raise ValueError(f"Invalid {name}: {controls[name]}")
        name = f"{role}_LLAVA_IMAGE_MODE"
        controls[name] = env.get(name) or controls["LLAVA_IMAGE_MODE"]
        if controls[name] not in {"anyres", "base"}:
            raise ValueError(f"Invalid {name}: {controls[name]}")
    for name in ("RANKER_QWEN_MIN_PIXELS", "RANKER_QWEN_MAX_PIXELS"):
        controls[name] = env.get(name) or controls[name.removeprefix("RANKER_")]
        if int(controls[name]) < 0:
            raise ValueError(f"{name} must be nonnegative")
    for name in ("STAGE1_DEPTH", "STAGE2_DEPTH", "SPARSE_THREADS", "RETRIEVAL_BATCH_SIZE", "RERANK_NUM", "RERANK_BATCH_SIZE"):
        if int(controls[name]) < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("SELECT_HYBRID_ALPHA", "OVERWRITE"):
        controls[name] = str(boolean(controls[name], name)).lower()
    alpha = float(controls["HYBRID_ALPHA"])
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("HYBRID_ALPHA must be finite and in [0,1]")
    controls["HYBRID_ALPHA"] = "0.5" if controls["SELECT_HYBRID_ALPHA"] == "true" else str(alpha)
    if any(env.get(name) for name in VALIDATION_INPUTS):
        raise ValueError("Custom validation inputs require the single-job launcher; batch alpha scanning uses the current dataset")
    for prefix in ("", "RANKER_"):
        low, high = (int(controls[prefix + "QWEN_" + bound + "_PIXELS"]) for bound in ("MIN", "MAX"))
        if low and high and low > high:
            raise ValueError(f"{prefix}QWEN_MIN_PIXELS must not exceed {prefix}QWEN_MAX_PIXELS")
    return controls


@dataclass(frozen=True)
class Settings:
    group: str
    families: tuple[str, ...]
    datasets: tuple[str, ...]
    queries: tuple[str, ...]
    split: str
    selection_split: str
    models: dict[str, Path]
    rankers: dict[str, Path]
    roots: dict[str, Path]
    output_dir: Path | None
    controls: dict[str, str]
    options: object
    dry_run: bool
    bash: str
    environment: dict[str, str]
    project: Path
    overlap_only: bool = False
    recover_missing_selection: bool = False
    selection_root: Path | None = None

    @classmethod
    def from_environment(cls, group, env=None, *, project=PROJECT, overlap_only=False, recover_missing_selection=False):
        env = dict(os.environ if env is None else env)
        project = Path(project).resolve()
        families = configured_list(env, "MODEL_FAMILY", "MODEL_FAMILIES", CHECKPOINTS, CHECKPOINTS)
        datasets = configured_list(env, "DATASET", "DATASETS", GROUPS[group]["datasets"], GROUPS[group]["datasets"])
        queries = configured_list(env, "IR_QUERY_TYPE", "IR_QUERY_TYPES", ("image", "text"), ("image", "text")) if group == "ir" else ()
        if len(families) > 1 and (env.get("MODEL_PATH") or env.get("RANKER_MODEL_PATH")):
            raise ValueError("Global MODEL_PATH/RANKER_MODEL_PATH require one family; use <NAME>_<FAMILY> overrides")
        total = len(families) * len(datasets) * (len(queries) or 1)
        if env.get("OUTPUT_DIR") and total > 1:
            raise ValueError("OUTPUT_DIR requires one retrieval job; use ADAPTIVE_OUTPUT_ROOT for batches")
        def path(name, default):
            return resolved_path(env.get(name, default), project, name)
        checkpoint_root = path("CHECKPOINT_ROOT", "/root/autodl-fs/checkpoints")
        models, rankers = {}, {}
        for family in families:
            models[family] = resolved_path(env.get("MODEL_PATH") or env.get(f"MODEL_PATH_{family.upper()}")
                                           or str(checkpoint_root / CHECKPOINTS[family]), project, "model path")
            rankers[family] = resolved_path(env.get("RANKER_MODEL_PATH") or env.get(f"RANKER_MODEL_PATH_{family.upper()}")
                                            or str(models[family]), project, "ranker model path")
        tags = [os.path.normcase(model.name) for model in models.values()]
        if not all(tags) or len(set(tags)) != len(tags):
            raise ValueError("Selected model paths must have distinct directory names")
        controls = pipeline_controls(env)
        if controls["RETRIEVAL_LLAVA_IMAGE_MODE"] == "base" and any(not f.startswith("llava_") for f in families):
            raise ValueError("Retrieval image mode base requires LLaVA-NeXT families only")
        options = selection_options(env, controls)
        if overlap_only and not options.view_overlap_selection:
            raise ValueError("Overlap retrieval launchers require USE_VIEW_OVERLAP_SELECTION=true; use the original launcher for ordinary/PMI retrieval")
        selection_root = path("VIEW_SELECTION_ROOT", "") if env.get("VIEW_SELECTION_ROOT") else None
        split = split_name(env.get("DATASET_SPLIT", "test"), "DATASET_SPLIT")
        return cls(group, families, datasets, queries, split,
                   split_name(env.get("VIEW_SELECTION_SPLIT") or split, "VIEW_SELECTION_SPLIT"), models, rankers,
                   dict(data=path("DATA_ROOT", "/root/autodl-tmp"), dense=path("DENSE_OUTPUT_ROOT", "./dense_output"),
                        sparse=path("SPARSE_OUTPUT_ROOT", "./sparse_output"),
                        results=path("ADAPTIVE_OUTPUT_ROOT", "./three_stage_adaptive_results")),
                   path("OUTPUT_DIR", "") if env.get("OUTPUT_DIR") else None, controls, options,
                   boolean(env.get("DRY_RUN", "false"), "DRY_RUN"), env.get("BASH_BIN") or "bash", env, project,
                   overlap_only, recover_missing_selection, selection_root)


def job_types(settings, dataset):
    if settings.group == "ir":
        return [("ir", query, "text" if query == "image" else "image") for query in settings.queries]
    return [("tbpr", "text", "image")] if settings.group == "tbpr" else [
        ("t2it", "text", "corpus") if dataset == "webqa" else ("it2t", "image", "corpus")]


def caption_modify_type(task, query):
    """Only candidate-text directions have the existing text-prior correction."""
    modes = {("ir", "image"): "modify_single", ("ir", "text"): "no",
             ("tbpr", "text"): "no", ("t2it", "text"): "no", ("it2t", "image"): "modify_single"}
    try:
        return modes[task, query]
    except KeyError as exc:
        raise ValueError(f"Unsupported adaptive caption reranking direction: {task}/{query}") from exc


def namespaces(settings, family, dataset, task, kind):
    c = settings.controls
    model = SimpleNamespace(model_name_or_path=str(settings.models[family]),
                            eol_type="disassembleeol_separate", calculate_type="concat")
    data = SimpleNamespace(dataset_name=dataset, dataset_split=settings.split, prompt_type="prompt_5", cir_type="type",
                           sparse_manual=False, dense_output_dir=str(settings.roots["dense"]),
                           sparse_output_dir=str(settings.roots["sparse"]), llava_image_mode=c["RETRIEVAL_LLAVA_IMAGE_MODE"],
                           **{name.lower(): c[name] for name in ("REPS_LOC", "SPARSE_VALUE_TYPE", "SPARSE_LOWER_OR_UPPER", "TBPR_TYPE")},
                           **{name.lower(): int(c[name]) for name in ("SPARSE_LENGTH", "TEXT_SPARSE_LENGTH", "IMAGE_SPARSE_LENGTH",
                                                                     "NUM_EXPENDED_TOKENS")},
                           **{name.lower(): boolean(c[name], name) for name in ("SPARSE_VALUE_MEAN", "IS_FILTERED")})
    if boolean(c["USE_FREERET_DENSE"], "USE_FREERET_DENSE"):
        model.use_freeret_dense = True
    if boolean(c["USE_FREERET_FULL_PROMPT"], "USE_FREERET_FULL_PROMPT"):
        model.use_freeret_full_prompt = True
    if boolean(c["USE_FREERET_PROMPT_CONSTRAINTS"], "USE_FREERET_PROMPT_CONSTRAINTS"):
        model.use_freeret_prompt_constraints = True
    return model, data, SimpleNamespace(task_type=task, encode_type=kind)


def selection_location(settings, model, data, training, identity):
    strategy = None
    if settings.options.view_overlap_selection:
        from adaptive_view_overlap import OverlapSelectionStrategy
        strategy = OverlapSelectionStrategy(settings.options, model, data, training.task_type)
    mode = settings.options.view_selection_modality
    candidate_mode = views.options_pool_mode(settings.options)
    example_policy = views.examples.configuration_for_path(settings.options, model.model_name_or_path)
    key = views.selection_cache_key(identity, data.dataset_name, settings.selection_split, mode,
                                    views.selection_sampling(settings.options, mode, example_configuration=example_policy), settings.roots["data"],
                                    pmi=settings.options.view_selection_pmi,
                                    strategy=strategy.configuration if strategy else None,
                                    shuffle=option_shuffle.for_options(settings.options,identity,data.dataset_name,views.view_pool(candidate_mode)),
                                    pool_mode=candidate_mode)
    return views.selection_cache_path(settings.roots["sparse"], model.model_name_or_path, data.dataset_name, key), key, strategy


class MissingInput(FileNotFoundError):
    """An absent experiment is skipped; a present but inconsistent one is an error."""


def recover_selection_cache(settings, model, data, training, identity, expected_path, backend, resolver):
    """Recover only an experiment explicitly linked to the expected automatic cache key."""
    pmi = settings.options.view_selection_pmi
    mode = settings.options.view_selection_modality
    overlap = settings.options.view_overlap_selection
    version = (views.PMI_OVERLAP_VERSION if pmi else views.OVERLAP_VERSION) if overlap else (
        views.PMI_VERSION if pmi else views.VERSION)
    candidate_mode = views.options_pool_mode(settings.options)
    probe = views.add_pool_mode(dict(version=version, selected=[], selection_modality=mode, views_fingerprint="0" * 64), candidate_mode)
    probe = views.examples.add_configuration(probe, views.examples.configuration_for_path(settings.options, model.model_name_or_path))
    shuffle = option_shuffle.for_options(settings.options,identity,data.dataset_name,views.view_pool(candidate_mode))
    probe = option_shuffle.add_metadata(probe,shuffle)
    if pmi:
        probe["calibration"] = dict(configuration=views.pmi_configuration(shuffle, pool_mode=candidate_mode))
    parent = backend.artifact_dirs(model, data, training, probe)[1].parent.resolve()
    expected_tail = tuple(os.path.normcase(part) for part in expected_path.as_posix().split("/")[-5:])
    matches = []
    for path in sorted(parent.glob(f"*/{views.MANIFEST}")):
        root = path.parent
        if path.resolve() != root.resolve() / views.MANIFEST or root.resolve() != root:
            raise ValueError(f"Refusing linked adaptive manifest/corpus during cache recovery: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        reference = manifest.get("selection_file")
        if not isinstance(reference, str) or not reference:
            raise ValueError(f"Missing selection_file during cache recovery: {path}")
        tail = tuple(os.path.normcase(part) for part in reference.replace("\\", "/").split("/")[-5:])
        if tail != expected_tail:
            continue
        configuration = manifest["configuration"]
        if manifest.get("fingerprint") != views.digest(configuration):
            raise ValueError(f"Corrupt adaptive completion fingerprint during cache recovery: {path}")
        candidate = dict(probe, selected=configuration.get("selected"), views_fingerprint=configuration.get("views_fingerprint"))
        expected = backend.metadata(model, data, training.task_type, training.encode_type, identity, candidate)
        if configuration != expected["configuration"]:
            continue
        if manifest.get("fingerprint") != expected["fingerprint"]:
            raise ValueError(f"Corrupt adaptive completion fingerprint during cache recovery: {path}")
        fingerprint = manifest.get("selection_fingerprint")
        if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise ValueError(f"Missing/invalid full selection_fingerprint during cache recovery: {path}")
        view_fingerprint = configuration.get("views_fingerprint")
        if not isinstance(view_fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", view_fingerprint):
            raise ValueError(f"Invalid views_fingerprint during cache recovery: {path}")
        if backend.artifact_dirs(model, data, training, candidate)[1].resolve() != root:
            raise ValueError(f"Directory/view fingerprint mismatch during cache recovery: {path}")
        matches.append((root, manifest))
    if not matches:
        raise MissingInput(f"Missing selection cache: {expected_path}; no completion manifest linked to the exact cache/configuration under {parent}. "
                           "Match encoding model/data paths, precision and selection settings, or restore the original cache.")
    fingerprints = {manifest["selection_fingerprint"] for _, manifest in matches}
    if len(fingerprints) != 1:
        raise ValueError(f"Ambiguous adaptive experiments for selection cache {expected_path}: "
                         f"different full selection fingerprints={sorted(fingerprints)}; manifests={[str(root / views.MANIFEST) for root, _ in matches]}. "
                         "Restore the intended original cache; no latest/available experiment is selected automatically.")
    root, manifest = matches[0]
    actual_path = Path(manifest["selection_file"]).expanduser()
    if not actual_path.is_absolute():
        actual_path = root / actual_path
    try:
        if not actual_path.exists():
            actual_path = resolver.recover(actual_path, root, manifest)
        checked = dict(manifest, selection_file=str(actual_path))
        selection = views.validate_index_selection(root, checked, resolver=resolver)
    except FileNotFoundError as exc:
        raise MissingInput(f"Cannot recover original selection for {expected_path}: {exc}") from exc
    print(f"[adaptive retrieval] recovered exact selection cache: {expected_path} -> {actual_path}; "
          f"manifest={root / views.MANIFEST}; selection_fingerprint={selection['fingerprint']}", flush=True)
    return actual_path, selection


def validate_corpus(root, expected, selection, *, dense):
    if not root.exists():
        raise MissingInput(f"Missing {'dense' if dense else 'sparse'} corpus: {root}")
    if not root.is_dir() or root.resolve() != root:
        raise ValueError(f"Expected an unlinked corpus directory: {root}")
    manifest = views.validate_artifact(root, expected)
    count = manifest.get("num_shards")
    if type(count) is not int or count < 1:
        raise ValueError(f"Invalid shard count at {root}")
    if manifest.get("selection_fingerprint") != selection["fingerprint"]:
        raise ValueError(f"Selection fingerprint differs from the encoded experiment at {root}")
    files = list(root.glob("corpus_*.pkl")) if dense else list((root / "collection").glob("corpus_*.jsonl"))
    if len(files) != count or any(not p.is_file() or p.resolve() != p for p in files):
        raise ValueError(f"Incomplete {'dense' if dense else 'sparse'} shards at {root}; expected {count}, found {len(files)}")
    return manifest


@dataclass(frozen=True)
class Job:
    key: str
    output: Path
    variables: dict[str, str]
    fingerprint: str
    configuration: dict


def prepare_jobs(settings, *, identity_loader=None, backend=None, auto_modify=False):
    if backend is None:
        import adaptive_view_encoding as backend
    identity_loader = backend.load_model_identity if identity_loader is None else identity_loader
    for root in settings.roots.values():
        if root.exists() and not root.is_dir():
            raise ValueError(f"Expected a directory: {root}")
    c = settings.controls
    resolver = views.SelectionFileResolver(settings.selection_root, sparse_root=settings.roots["sparse"]) if settings.recover_missing_selection else None
    jobs, missing, seen_outputs = [], [], set()
    for family in settings.families:
        model, ranker = settings.models[family], settings.rankers[family]
        model_missing = [str(p) for p in (model / "config.json", ranker / "config.json") if not p.is_file()]
        identity = None
        ranker_checked = False
        for dataset in settings.datasets:
            for task, query, kind in job_types(settings, dataset):
                job_key = f"{family}|{dataset}|{query}|{settings.split}"
                absent = model_missing + [str(p) for p in annotation_files(settings.roots["data"], dataset, settings.split) if not p.is_file()]
                if absent:
                    missing.append((job_key, "Missing inputs: " + ", ".join(dict.fromkeys(absent))))
                    continue
                if identity is None:
                    identity = identity_loader(str(model), dtype=c["RETRIEVAL_DTYPE"], reps_loc=c["REPS_LOC"],
                                               min_pixels=int(c["QWEN_MIN_PIXELS"]), max_pixels=int(c["QWEN_MAX_PIXELS"]),
                                               image_mode=c["RETRIEVAL_LLAVA_IMAGE_MODE"])
                    if identity["family"] != FAMILIES[family]:
                        raise ValueError(f"Model family mismatch: {family} points to {identity['family']} at {model}")
                if not ranker_checked:
                    if c["RANKER_LLAVA_IMAGE_MODE"] == "base":
                        from llava_resolution import validate_checkpoint
                        validate_checkpoint(str(ranker), "base")
                    ranker_checked = True
                model_args, data, training = namespaces(settings, family, dataset, task, kind)
                selection_path, cache_key, strategy = selection_location(settings, model_args, data, training, identity)
                try:
                    if not selection_path.exists():
                        if not settings.recover_missing_selection:
                            raise MissingInput(f"Missing selection cache: {selection_path}; match encoding model/data paths, precision and selection settings")
                        selection_path, selection = recover_selection_cache(settings, model_args, data, training, identity,
                                                                            selection_path, backend, resolver)
                    else:
                        selection = views.load_selection(selection_path, identity=identity, dataset=dataset,
                                                         selection_modality=settings.options.view_selection_modality,
                                                         view_selection_pmi=settings.options.view_selection_pmi,
                                                         view_selection_hierarchical=settings.options.view_selection_hierarchical,
                                                         view_example_mode=settings.options.view_example_mode,
                                                         view_example_max_tokens=settings.options.view_example_max_tokens)
                    if settings.overlap_only and not views.is_overlap_selection(selection):
                        raise ValueError(f"Expected an overlap selection (16/17): {selection_path}")
                    views.require_equal(dict(split=selection.get("selection_split")), dict(split=settings.selection_split),
                                        "Selection source split")
                    views.require_equal(selection.get("sampling"), cache_key["sampling"], "Selection sampling")
                    if strategy is not None:
                        strategy.validate(selection)
                    elif views.is_overlap_selection(selection):
                        raise ValueError(f"Expected ordinary/PMI selection, found overlap: {selection_path}")
                    dense, sparse = (p.resolve() for p in backend.artifact_dirs(model_args, data, training, selection))
                    expected = backend.metadata(model_args, data, task, kind, identity, selection)
                    dense_manifest = validate_corpus(dense, expected, selection, dense=True)
                    sparse_manifest = validate_corpus(sparse, expected, selection, dense=False)
                    if dense_manifest["num_shards"] != sparse_manifest["num_shards"]:
                        raise ValueError(f"Dense/sparse shard counts differ for {job_key}")
                    index = resolve_sparse_index(sparse, require=False)
                    if not index.exists():
                        index_launcher = (f"sparse_index_adaptive_views_overlap_{settings.group}.sh" if settings.overlap_only
                                          else f"sparse_index_adaptive_{settings.group}.sh")
                        raise MissingInput(f"Missing Lucene index: {index}; run {index_launcher} first")
                    if not index.is_dir() or not any(p.is_file() for p in index.glob("segments_*")):
                        raise ValueError(f"Incomplete Lucene index at {index}")
                except MissingInput as exc:
                    missing.append((job_key, str(exc)))
                    continue
                except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                    raise ValueError(f"{job_key}: invalid adaptive experiment at {selection_path}: {exc}") from exc
                variables = {name: c[name] for name in PIPELINE_NAMES}
                variables.update(MODEL_FAMILY=family, MODEL_PATH=str(model), RANKER_MODEL_PATH=str(ranker), DATASET=dataset,
                                 DATASET_SPLIT=settings.split, IR_QUERY_TYPE=query, DATA_ROOT=str(settings.roots["data"]),
                                 VIEW_SELECTION_FILE=str(selection_path), DENSE_CORPUS_DIR=str(dense),
                                 SPARSE_CORPUS_DIR=str(sparse), STAGE1_SPARSE_INDEX=str(index))
                if auto_modify:
                    # Always derive from this job; inherited MODIFY_TYPE/previous jobs cannot override it.
                    variables["JOB_MODIFY_TYPE"] = caption_modify_type(task, query)
                config = dict(version=1, task=task, query_type=query, encoding_fingerprint=expected["fingerprint"],
                              views_fingerprint=selection["views_fingerprint"], selection_fingerprint=selection["fingerprint"],
                              selection_cache_fingerprint=views.digest(cache_key), ranker=views.checkpoint_identity(ranker),
                              parameters={k: v for k, v in variables.items() if k not in {
                                  "VIEW_EXAMPLE_MODE", "VIEW_EXAMPLE_MAX_TOKENS",
                                  "GPUS", "NUM_NODES", "DEEPSPEED_BIN", "OVERWRITE", "JOB_MODIFY_TYPE", "USE_FREERET_DENSE", "USE_FREERET_PROMPT_CONSTRAINTS", "USE_FREERET_FULL_PROMPT", "SHUFFLE_VIEW_OPTIONS", "VIEW_SELECTION_RUN_ID", "USE_HIERARCHICAL_VIEWS"}})
                if auto_modify:
                    config["modify_type"] = variables["JOB_MODIFY_TYPE"]
                from dense_representation import configuration as dense_configuration, mode_path
                use_dense = boolean(c["USE_FREERET_DENSE"], "USE_FREERET_DENSE")
                config = dense_configuration(config, use_dense, boolean(c["USE_FREERET_PROMPT_CONSTRAINTS"], "USE_FREERET_PROMPT_CONSTRAINTS"), boolean(c["USE_FREERET_FULL_PROMPT"], "USE_FREERET_FULL_PROMPT"))
                config = views.add_pool_mode(option_shuffle.add_metadata(config,selection.get(option_shuffle.FIELD)), views.pool_mode(selection))
                config = views.examples.add_configuration(config, selection.get(views.examples.FIELD))
                fingerprint = views.digest(config)
                output = settings.output_dir or (settings.roots["results"] / model.name / dataset / query / settings.split /
                         "adaptive_views" / selection["views_fingerprint"][:16] / expected["fingerprint"][:16] /
                         f"stage1_{c['STAGE1_DEPTH']}_stage2_{c['STAGE2_DEPTH']}" / fingerprint[:16])
                output = option_shuffle.mark_path(output,selection.get(option_shuffle.FIELD))
                output = mode_path(output, use_dense, boolean(c["USE_FREERET_PROMPT_CONSTRAINTS"], "USE_FREERET_PROMPT_CONSTRAINTS"), boolean(c["USE_FREERET_FULL_PROMPT"], "USE_FREERET_FULL_PROMPT")).resolve()
                if output in seen_outputs:
                    raise ValueError(f"Duplicate retrieval output directory: {output}")
                seen_outputs.add(output)
                if output.exists() and (not output.is_dir() or c["OVERWRITE"] != "true"):
                    raise ValueError(f"Retrieval output exists: {output}; enable OVERWRITE or use another ADAPTIVE_OUTPUT_ROOT")
                variables["OUTPUT_DIR"] = str(output)
                jobs.append(Job(job_key, output, variables, fingerprint, config))
    return jobs, missing


def report_summary(jobs, missing, completed, planned, failed=0):
    print(f"\nSummary: runnable={len(jobs)} completed={completed} planned={planned} skipped_missing={len(missing)} "
          f"failed={failed} remaining={len(jobs) - completed - planned - failed}", flush=True)


def run(settings, *, runner=None, identity_loader=None, backend=None, launcher=None):
    if launcher is None:
        command = [settings.bash, "scripts/run_adaptive_views_three_stage.sh"]
    else:
        script = resolved_path(str(launcher), settings.project, "launcher")
        if not script.is_file():
            raise ValueError(f"Retrieval launcher does not exist: {script}")
        command = [settings.bash, script.as_posix(), "--run-one"]
    jobs, missing = prepare_jobs(settings, identity_loader=identity_loader, backend=backend,
                                auto_modify=launcher is not None)
    print(f"Adaptive retrieval batch: group={settings.group} split={settings.split} dry_run={settings.dry_run} "
          f"PMI={settings.options.view_selection_pmi} overlap={settings.options.view_overlap_selection}", flush=True)
    print(f"Preflight: runnable={len(jobs)} skipped_missing={len(missing)}; every runnable job runs all three stages.", flush=True)
    for key, reason in missing:
        print(f"Skip missing: {key}\n  {reason}", flush=True)
    if not jobs:
        report_summary(jobs, missing, 0, 0)
        raise ValueError("No runnable adaptive retrieval jobs; check the reported paths and encoding settings")
    completed = planned = 0
    runner = subprocess.run if runner is None else runner
    for i, job in enumerate(jobs, 1):
        print(f"\nJob [{i}/{len(jobs)}]: {job.key}\nSelection cache fingerprint: {job.configuration['selection_cache_fingerprint']}\n"
              f"Views fingerprint: {job.configuration['views_fingerprint']}\nEncoding fingerprint: {job.configuration['encoding_fingerprint']}\n"
              f"Retrieval fingerprint: {job.fingerprint}", flush=True)
        if "modify_type" in job.configuration:
            print(f"Rerank: what_caption_generation modify_type={job.configuration['modify_type']}", flush=True)
        print(shlex.join(["env", *(f"{k}={v}" for k, v in job.variables.items()), *command]), flush=True)
        if settings.dry_run:
            planned += 1
            continue
        try:
            result = runner(command, cwd=settings.project, env=dict(settings.environment, **job.variables), check=False)
            status = result.returncode if result.returncode >= 0 else 128 - result.returncode
            if status == 0:
                views.atomic_json(job.output / "batch_job.json", dict(fingerprint=job.fingerprint, configuration=job.configuration))
        except OSError as exc:
            print(f"Cannot launch {job.key}: {exc}", file=sys.stderr, flush=True)
            status = 2
        if status:
            print(f"Failed: {job.key} (exit {status}); stopping batch.", file=sys.stderr, flush=True)
            report_summary(jobs, missing, completed, planned, failed=1)
            return status
        completed += 1
        print(f"Completed [{i}/{len(jobs)}]: {job.key}", flush=True)
    report_summary(jobs, missing, completed, planned)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, epilog="""
Configure environment variables or edit the batch Bash launchers; only -h/--help CLI arguments are accepted.
Full-command launchers list every setting and execute their own DeepSpeed command through internal --run-one calls.
MODEL_FAMILIES='qwen2 qwen25 qwen3 llava_llama llava_mistral'; nonempty MODEL_FAMILY selects one.
DATASETS: ir='flickr coco'; tbpr='CUHK-PEDES ICFG-PEDES RSTPReid'; fused='webqa remuq llava'. DATASET selects one.
IR_QUERY_TYPES='image text'; IR_QUERY_TYPE selects one direction. TBPR is text->image; fused uses corpus artifacts.
USE_HIERARCHICAL_VIEWS=true locates encodings using the ten primary/secondary dimension candidates.
CHECKPOINT_ROOT=/root/autodl-fs/checkpoints; MODEL_PATH_<FAMILY> overrides each family (uppercase suffix).
RANKER_MODEL_PATH_<FAMILY> defaults to that family's model. Global MODEL_PATH/RANKER_MODEL_PATH need one family.
DATA_ROOT=/root/autodl-tmp; DATASET_SPLIT=test; VIEW_SELECTION_SPLIT defaults to the dataset split.
DENSE_OUTPUT_ROOT=./dense_output; SPARSE_OUTPUT_ROOT=./sparse_output. Match ALL settings to adaptive encoding.
VIEW_SELECTION_MODALITY=both VIEW_SELECTION_ROUNDS=20 VIEW_SELECTION_SEED=42.
VIEW_EXAMPLE_MODE=auto|fixed defaults auto; VIEW_EXAMPLE_MAX_TOKENS=0 uses the model context budget.
Match encoding mode/budget. Fixed mode uses VIEW_TEXTS_PER_ROUND=8 VIEW_IMAGES_PER_ROUND=2 and preserves old caches.
USE_VIEW_PMI=false USE_VIEW_OVERLAP_SELECTION=false; enable both to retrieve PMI-calibrated overlap artifacts.
Overlap: VIEW_SELECTED_COUNT=5 VIEW_OVERLAP_THRESHOLD=0.1 VIEW_OVERLAP_SAMPLES=20 VIEW_OVERLAP_SEED=42.
Overlap versions 16/17 use shared tokens / candidate unique tokens, accepting strictly below threshold.
Old overlap versions 13/15 are incompatible; reselect, re-encode and rebuild their indexes.
The exact encoding selection-cache hash is recomputed; its locked views determine the corpus fingerprint/path.
No latest/all-fingerprint experiment selection is performed. Single-job artifact path variables are recomputed.
Overlap multi launchers require USE_VIEW_OVERLAP_SELECTION=true and enable strict missing-cache recovery.
--recover-missing-selection uses only completion manifests linked to the exact expected cache/configuration.
Different full selection fingerprints are an ambiguity error. VIEW_SELECTION_ROOT optionally locates exact JSON backups.
Recovery is read-only; original launchers retain their missing-cache behavior unless explicitly opted in.
DTYPE=float16 REPS_LOC=after_pad SPARSE_LENGTH=40; all sparse, filtering and pixel settings must match encoding.
RETRIEVAL_DTYPE/RANKER_DTYPE override DTYPE. LLAVA_IMAGE_MODE=anyres; stage-specific image modes override it.
GPUS=1 NUM_NODES=1 STAGE1_DEPTH=1000 STAGE2_DEPTH=1000 SPARSE_THREADS=16 RETRIEVAL_BATCH_SIZE=1.
RERANK_NUM=5 RERANK_BATCH_SIZE=1; ranker uses what_caption_generation.
New full-command launchers always choose modify_type per job: IR image->text and ReMuQ/LLaVA use modify_single;
IR text->image, TBPR and WebQA use no. There is no manual switch; inherited MODIFY_TYPE is ignored.
Actual modify_type is printed and included in the result fingerprint/batch_job.json; no prior cache is enabled.
Legacy launchers retain their fixed modify_type=no behavior and existing result fingerprints.
HYBRID_ALPHA=0.5 SELECT_HYBRID_ALPHA=false; true scans the current dataset. Custom validation uses the single-job launcher.
ADAPTIVE_OUTPUT_ROOT=./three_stage_adaptive_results; per-experiment results include encoding/retrieval fingerprints.
OUTPUT_DIR is allowed for one job. Legacy three_stage_adaptive_<task>_multi.sh launchers default OVERWRITE=true.
New three_stage_retrieval_pipeline_adaptive_views_<task>.sh launchers default OVERWRITE=false; set true to rerun.
Missing inputs are skipped/summarized; zero runnable jobs is an error. Corrupt/inconsistent inputs fail preflight.
DRY_RUN=true loads only configs/processors, prints fingerprints/commands, and writes no output or model weights.
No corpus encoding, selection, overlap probing, PMI prior calculation or index building is performed.
PYTHON_BIN selects the Python environment; BASH_BIN=bash DEEPSPEED_BIN=deepspeed select the launch commands.
Relative paths are resolved from the project root. Actual job failure stops the batch and preserves the exit code.
""")
    parser.add_argument("--task-group", choices=GROUPS, required=True)
    parser.add_argument("--launcher", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--overlap-only", action="store_true", help="Require overlap encoding experiments")
    parser.add_argument("--recover-missing-selection", action="store_true", help="Recover missing exact selection caches from linked completion manifests")
    args = parser.parse_args(argv)
    try:
        return run(Settings.from_environment(args.task_group, overlap_only=args.overlap_only,
                                              recover_missing_selection=args.recover_missing_selection), launcher=args.launcher)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        print("Retrieval batch interrupted.", file=sys.stderr, flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
