"""CPU-only planning and preflight for the ordinary LLaVA task launchers."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from dense_representation import artifact_root, mode_path, validate_artifact
from stage2_retrieval_mode import output_path

PROJECT = Path(__file__).resolve().parents[1]
MODELS = {"llava_mistral": "llava-hf-llava-v1.6-mistral-7b-hf",
          "llava_llama": "llava-hf-llama3-llava-next-8b-hf"}
DATASETS = {"ir": ("flickr", "coco"), "tbpr": ("CUHK-PEDES", "ICFG-PEDES", "RSTPReid"),
            "fused": ("webqa", "remuq", "llava")}
ANNOTATIONS = {"flickr": ("flickr", ("flickr_{split}.csv",)), "coco": ("coco", ("coco_{split}.csv",)),
               "CUHK-PEDES": ("CUHK-PEDES", ("reid_raw.json",)),
               "ICFG-PEDES": ("ICFG-PEDES", ("ICFG-PEDES.json",)),
               "RSTPReid": ("RSTPReid", ("data_captions.json",)),
               "webqa": ("WebQA", ("query-00000-of-00001.parquet", "qrels-00000-of-00001.parquet",
                                     "corpus-00000-of-00002.parquet", "corpus-00001-of-00002.parquet")),
               "remuq": ("UMRB-ReMuQ", ("query.parquet", "qrels.parquet", "corpus.parquet")),
               "llava": ("UMRB-LLaVA", ("query.parquet", "qrels.parquet", "corpus.parquet"))}
DEFAULTS = dict(PROMPT_TYPE="prompt_shared_5", TBPR_TYPE="type", DATASET_SPLIT="test",
    CHECKPOINT_ROOT="/root/autodl-fs/checkpoints", DATA_ROOT="/root/autodl-tmp",
    DENSE_OUTPUT_ROOT="./dense_output", SPARSE_OUTPUT_ROOT="./sparse_output", RESULT_ROOT="./three_stage_results",
    GPUS="1", NUM_NODES="1", DTYPE="float16", PER_DEVICE_BATCH_SIZE="1", DEEPSPEED_BIN="deepspeed",
    PYTHON_BIN=sys.executable, EOL_TYPE="disassembleeol_separate", CALCULATE_TYPE="concat", SPARSE_MANUAL="false",
    REPS_LOC="after_pad", SPARSE_LENGTH="40", SPARSE_VALUE_TYPE="sum", SPARSE_LOWER_OR_UPPER="lower",
    SPARSE_VALUE_MEAN="false", IS_FILTERED="false", NUM_EXPENDED_TOKENS="0", RANDOM_VIEW_SEED="42",
    LLAVA_IMAGE_MODE="anyres", DRY_RUN="false", SPARSE_THREADS="16", USE_FREERET_DENSE="false",
    USE_FREERET_PROMPT_CONSTRAINTS="false", USE_FREERET_FULL_PROMPT="false", STAGE1_METHOD="sparse_concat",
    STAGE1_DEPTH="1000", STAGE2_DEPTH="1000", STAGE2_METHOD="hybrid", HYBRID_ALPHA="0.5",
    SELECT_HYBRID_ALPHA="false", RETRIEVAL_BATCH_SIZE="1", RETRIEVAL_DEVICE="auto", RETRIEVAL_DEVICE_MAP="auto",
    RANKER_DEVICE_MAP="auto", RERANK_NUM="5", RERANK_BATCH_SIZE="1", RERANK_TEMPLATE="caption_generation",
    RERANK_TYPE="pointwise", MODIFY_TYPE="auto", RERANK_REPS_LOC="after_pad", OVERWRITE="false",
    USE_RERANK_PRIOR_CACHE="auto")
BOOLS = ("DRY_RUN", "SPARSE_MANUAL", "SPARSE_VALUE_MEAN", "IS_FILTERED", "USE_FREERET_DENSE",
         "USE_FREERET_PROMPT_CONSTRAINTS", "USE_FREERET_FULL_PROMPT", "SELECT_HYBRID_ALPHA", "OVERWRITE")
CLI_ENV = dict(dataset_name="DATASET", dataset_split="DATASET_SPLIT", task_type="TASK_TYPE", query_type="QUERY_TYPE",
    prompt_type="PROMPT_TYPE", tbpr_type="TBPR_TYPE", data_root="DATA_ROOT", output_dir="OUTPUT_DIR",
    retrieval_model_name_or_path="RETRIEVAL_MODEL_PATH", ranker_model_name_or_path="RANKER_MODEL_PATH",
    stage1_sparse_index="STAGE1_SPARSE_INDEX", stage2_dense_index="STAGE2_DENSE_INDEX", stage2_sparse_index="STAGE2_SPARSE_INDEX",
    stage1_run="STAGE1_RUN", stage1_command="STAGE1_COMMAND", stage1_method="STAGE1_METHOD", stage2_run="STAGE2_RUN",
    stage1_depth="STAGE1_DEPTH", stage2_depth="STAGE2_DEPTH", stage2_method="STAGE2_METHOD", hybrid_alpha="HYBRID_ALPHA",
    select_hybrid_alpha="SELECT_HYBRID_ALPHA", retrieval_dtype="RETRIEVAL_DTYPE", ranker_dtype="RANKER_DTYPE",
    retrieval_reps_loc="RETRIEVAL_REPS_LOC", retrieval_llava_image_mode="RETRIEVAL_LLAVA_IMAGE_MODE",
    ranker_llava_image_mode="RANKER_LLAVA_IMAGE_MODE", sparse_length="SPARSE_LENGTH", text_sparse_length="TEXT_SPARSE_LENGTH",
    image_sparse_length="IMAGE_SPARSE_LENGTH", sparse_value_type="SPARSE_VALUE_TYPE", sparse_lower_or_upper="SPARSE_LOWER_OR_UPPER",
    sparse_value_mean="SPARSE_VALUE_MEAN", is_filtered="IS_FILTERED", num_expended_tokens="NUM_EXPENDED_TOKENS",
    random_view_seed="RANDOM_VIEW_SEED", use_freeret_dense="USE_FREERET_DENSE", use_freeret_full_prompt="USE_FREERET_FULL_PROMPT",
    use_freeret_prompt_constraints="USE_FREERET_PROMPT_CONSTRAINTS", use_rerank_prior_cache="USE_RERANK_PRIOR_CACHE", overwrite="OVERWRITE",
    modify_type="MODIFY_TYPE", rerank_template="RERANK_TEMPLATE", rerank_type="RERANK_TYPE")


def boolean(value, name):
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"true", "1", "yes"}:
        return True
    if str(value).lower() in {"false", "0", "no"}:
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}")


def configured(env, single, plural, default, allowed):
    values = (env.get(single) or env.get(plural) or default).split()
    if not values or (env.get(single) and len(values) != 1) or len(values) != len(set(values)) or any(value not in allowed for value in values):
        raise ValueError(f"Invalid {plural}: {values}; choose from {', '.join(allowed)}")
    return values


def resolve_modify_type(value, task, query, template):
    """Resolve auto per job; the worker accepts only no or modify_single."""
    if value not in {"auto", "no", "modify_single"}:
        raise ValueError("MODIFY_TYPE must be auto, no or modify_single")
    if value != "auto":
        return value
    supported = task == "it2t" or (task == "ir" and query == "image")
    return "modify_single" if supported and "caption_generation" in template else "no"


def pipeline_overrides(extra, env):
    """Use the worker's parser so CLI paths/settings precede automatic lookup."""
    if not extra:
        return
    from three_stage_retrieval_pipeline import _parser
    parser = _parser()
    for action in parser._actions:
        if action.dest != "help":
            action.default = argparse.SUPPRESS
            action.required = False
    for name, value in vars(parser.parse_args(extra)).items():
        if name in CLI_ENV:
            env[CLI_ENV[name]] = str(value).lower() if isinstance(value, bool) else str(value)


@dataclass
class Job:
    key: str
    action: str
    family: str
    dataset: str
    task: str
    query: str
    model: Path
    ranker: Path
    dense: Path
    sparse: Path
    stage1: Path
    stage2: Path
    output: Path
    command: list[str]


def build_jobs(action, group, environment=None, *, extra=(), project=PROJECT):
    env = dict(os.environ if environment is None else environment)
    if extra:
        if action != "retrieve":
            raise ValueError("Configure encoding/indexing in the script or environment; only --help and --dry-run are accepted")
        pipeline_overrides(extra, env)
    c = {**env, **{name: env.get(name) or value for name, value in DEFAULTS.items()}}
    for name in BOOLS:
        c[name] = str(boolean(c[name], name)).lower()
    cache_name = "USE_RERANK_PRIOR_CACHE"
    cache_mode = env.get(cache_name)
    if cache_mode is None or cache_mode == "":
        cache_mode = DEFAULTS[cache_name]
    if cache_mode == "auto":
        c[cache_name] = "auto"
    else:
        try:
            c[cache_name] = str(boolean(cache_mode, cache_name)).lower()
        except ValueError as exc:
            raise ValueError("USE_RERANK_PRIOR_CACHE must be auto, true or false") from exc
    for name in ("GPUS", "NUM_NODES", "PER_DEVICE_BATCH_SIZE", "SPARSE_LENGTH", "SPARSE_THREADS", "STAGE1_DEPTH", "STAGE2_DEPTH",
                 "RETRIEVAL_BATCH_SIZE", "RERANK_NUM", "RERANK_BATCH_SIZE"):
        if int(c[name]) < 1:
            raise ValueError(f"{name} must be positive")
        c[name] = str(int(c[name]))
    for name in ("TEXT_SPARSE_LENGTH", "IMAGE_SPARSE_LENGTH"):
        if c.get(name):
            if int(c[name]) < 1:
                raise ValueError(f"{name} must be positive")
            c[name] = str(int(c[name]))
    if int(c["NUM_EXPENDED_TOKENS"]) < 0:
        raise ValueError("NUM_EXPENDED_TOKENS must be nonnegative")
    c["NUM_EXPENDED_TOKENS"] = str(int(c["NUM_EXPENDED_TOKENS"]))
    c["RANDOM_VIEW_SEED"] = str(int(c["RANDOM_VIEW_SEED"]))
    for name in ("DTYPE", "RETRIEVAL_DTYPE", "RANKER_DTYPE"):
        if c.get(name, c["DTYPE"]) not in {"float16", "bfloat16", "float32"}:
            raise ValueError(f"Invalid {name}")
    if c["SPARSE_MANUAL"] != "false" or (c["EOL_TYPE"], c["CALCULATE_TYPE"]) != ("disassembleeol_separate", "concat"):
        raise ValueError("These paired launchers require disassembleeol_separate + concat and SPARSE_MANUAL=false")
    if not c["DATASET_SPLIT"] or any(char in c["DATASET_SPLIT"] for char in "/\\"):
        raise ValueError("DATASET_SPLIT must be a split name")
    if c["TBPR_TYPE"] not in {"origin_type", "type", "type_1", "type_2"}:
        raise ValueError("Invalid TBPR_TYPE")
    if c["PROMPT_TYPE"] not in {"prompt_shared_5", "prompt_random_5", *{f"prompt_{n}" for n in range(1, 8)}}:
        raise ValueError("Unsupported PROMPT_TYPE")
    if c["STAGE2_METHOD"] not in {"hybrid", "dense"} or not 0 <= float(c["HYBRID_ALPHA"]) <= 1:
        raise ValueError("Invalid STAGE2_METHOD or HYBRID_ALPHA")
    if c["SPARSE_VALUE_TYPE"] not in {"sum", "max", "replace"} or c["SPARSE_LOWER_OR_UPPER"] not in {"lower", "upper"}:
        raise ValueError("Invalid sparse value settings")
    reps = c.get("RETRIEVAL_REPS_LOC", c["REPS_LOC"]) if action == "retrieve" else c["REPS_LOC"]
    image_mode = c.get("RETRIEVAL_LLAVA_IMAGE_MODE", c["LLAVA_IMAGE_MODE"]) if action == "retrieve" else c["LLAVA_IMAGE_MODE"]
    if reps not in {"after_pad", "before_pad"} or image_mode not in {"anyres", "base"}:
        raise ValueError("Invalid REPS_LOC or LLAVA_IMAGE_MODE")
    if c.get("RANKER_LLAVA_IMAGE_MODE", c["LLAVA_IMAGE_MODE"]) not in {"anyres", "base"}:
        raise ValueError("Invalid RANKER_LLAVA_IMAGE_MODE")
    families = configured(c, "MODEL_FAMILY", "MODEL_FAMILIES", "llava_mistral", MODELS)
    if c.get("BASE_MODEL") and not c.get("MODEL_FAMILY") and families == ["llava_mistral"]:
        families = [next((name for name, checkpoint in MODELS.items() if checkpoint == Path(c["BASE_MODEL"]).name), "llava_mistral")]
    datasets = configured(c, "DATASET", "DATASETS", DATASETS[group][0], DATASETS[group])
    queries = configured(c, "QUERY_TYPE", "QUERY_TYPES", "image text", ("image", "text")) if group == "ir" else [None]
    total = len(families) * len(datasets) * len(queries)
    if len(families) > 1 and any(c.get(name) for name in ("MODEL_PATH", "RETRIEVAL_MODEL_PATH", "BASE_MODEL", "ARTIFACT_MODEL_NAME")):
        raise ValueError("Batch models need MODEL_PATH_LLAVA_MISTRAL / MODEL_PATH_LLAVA_LLAMA, not a shared MODEL_PATH")
    single_paths = ("DENSE_CORPUS_DIR", "SPARSE_CORPUS_DIR", "STAGE1_SPARSE_INDEX", "STAGE2_DENSE_INDEX", "STAGE2_SPARSE_INDEX",
                    "OUTPUT_DIR", "ENCODE_OUTPUT_DIR", "VAL_PASSAGE_REPS", "VAL_SPARSE_INDEX", "VAL_STAGE1_SPARSE_INDEX", "VAL_STAGE1_RUN", "STAGE1_RUN", "STAGE2_RUN")
    if total > 1 and any(c.get(name) for name in single_paths):
        raise ValueError("Explicit corpus/index/output/validation paths require one model/dataset/direction combination")
    if action == "encode" and any(c.get(name) for name in single_paths[:5]):
        raise ValueError("Encoding uses DENSE_OUTPUT_ROOT/SPARSE_OUTPUT_ROOT; explicit corpus/index paths are for indexing/retrieval")
    use = tuple(c[name] == "true" for name in ("USE_FREERET_DENSE", "USE_FREERET_PROMPT_CONSTRAINTS", "USE_FREERET_FULL_PROMPT"))
    if (use[1] or use[2]) and not use[0]:
        raise ValueError("FreeRet prompt modes require USE_FREERET_DENSE=true")
    project = Path(project).resolve()
    def path(value):
        result = Path(value).expanduser()
        return result if result.is_absolute() else project / result
    def mode(value):
        return Path(mode_path(path(value), *use))
    c["DATA_ROOT"] = str(path(c["DATA_ROOT"]))
    if c.get("STAGE1_RUN"):
        c["STAGE1_RUN"] = str(path(c["STAGE1_RUN"]))
    jobs, seen = [], set()
    for family in families:
        supplied = c.get("RETRIEVAL_MODEL_PATH") or c.get("MODEL_PATH") or c.get("MODEL_PATH_" + family.upper())
        if not supplied and c.get("BASE_MODEL"):
            supplied = c["BASE_MODEL"] if "/" in c["BASE_MODEL"] else str(Path(c["CHECKPOINT_ROOT"]) / c["BASE_MODEL"])
        model = path(supplied or str(Path(c["CHECKPOINT_ROOT"]) / MODELS[family]))
        ranker = path(c.get("RANKER_MODEL_PATH_" + family.upper()) or c.get("RANKER_MODEL_PATH") or str(model))
        tag = c.get("ARTIFACT_MODEL_NAME") or model.name
        if not tag or tag in {".", ".."} or any(char in tag for char in "/\\"):
            raise ValueError("ARTIFACT_MODEL_NAME must be a directory basename")
        for dataset, ir_query in product(datasets, queries):
            if group == "ir":
                task, query = "ir", ir_query
                kind, direction = ("text", "image_to_text") if query == "image" else ("image", "text_to_image")
            else:
                task, query, kind, direction = ("tbpr", "text", "image", "text_to_image") if group == "tbpr" else (
                    ("t2it", "text", "corpus", "text_to_imagetext") if dataset == "webqa" else ("it2t", "image", "corpus", "imagetext_to_text"))
            if c.get("TASK_TYPE") and c["TASK_TYPE"] != task:
                raise ValueError(f"{dataset} requires TASK_TYPE={task}")
            if task != "ir":
                query = c.get("QUERY_TYPE") or query
            if query not in ({"image", "text"} if task == "ir" else {"image", "composed"} if task == "it2t" else {"text"}):
                raise ValueError(f"Invalid QUERY_TYPE={query} for {dataset}")
            if c.get("ENCODE_TYPE") and c["ENCODE_TYPE"] != kind:
                raise ValueError(f"{dataset} requires ENCODE_TYPE={kind}")
            if action == "retrieve":
                modify_type = resolve_modify_type(c["MODIFY_TYPE"], task, query, c["RERANK_TEMPLATE"])
                use_prior_cache = modify_type == "modify_single" if c[cache_name] == "auto" else c[cache_name] == "true"
            if action == "retrieve" and use_prior_cache:
                if task != "it2t" and not (task == "ir" and query == "image"):
                    raise ValueError("Rerank prior cache supports only it2t and IR image-to-text; for IR select QUERY_TYPE=image")
                from rerank_prior_cache import validate_prior_cache_options
                validate_prior_cache_options(True, modify_type, c["RERANK_TEMPLATE"], task, query, c["RERANK_TYPE"])
            if action == "retrieve" and c["PROMPT_TYPE"] == "prompt_shared_5" and c["STAGE1_METHOD"] == "sparse_manual" and not (c.get("STAGE1_RUN") or c.get("STAGE1_COMMAND")):
                raise ValueError("prompt_shared_5 requires sparse_concat for online sparse Stage 1")
            relative = Path(tag) / dataset / kind / ("filter" if c["IS_FILTERED"] == "true" else "no_filter") / "concat" / c["PROMPT_TYPE"]
            if task == "tbpr" and c["PROMPT_TYPE"] == "prompt_random_5":
                relative /= c["TBPR_TYPE"]
            relative /= c["DATASET_SPLIT"]
            if task == "tbpr" and c["PROMPT_TYPE"] != "prompt_random_5":
                relative /= c["TBPR_TYPE"]
            leaf = f"{c['NUM_EXPENDED_TOKENS']}_no_manual_{c['SPARSE_LENGTH']}_{c['SPARSE_VALUE_TYPE']}_no_cluster_{reps}_disassembleeol_separate_{c['SPARSE_LOWER_OR_UPPER']}_{'mean' if c['SPARSE_VALUE_MEAN']=='true' else 'no_mean'}"
            if image_mode == "base" and (kind == "image" or task == "t2it"):
                leaf += "_llava_base"
            dense = mode(c.get("STAGE2_DENSE_INDEX") or c.get("DENSE_CORPUS_DIR") or str(Path(c["DENSE_OUTPUT_ROOT"]) / relative / leaf))
            sparse = mode(c.get("SPARSE_CORPUS_DIR") or c.get("STAGE2_SPARSE_INDEX") or str(Path(c["SPARSE_OUTPUT_ROOT"]) / relative / leaf))
            stage2 = mode(c.get("STAGE2_SPARSE_INDEX") or str(sparse))
            stage1 = mode(c.get("STAGE1_SPARSE_INDEX") or str(stage2))
            result = path(c["RESULT_ROOT"]) / tag / dataset / direction / c["DATASET_SPLIT"] / "sparse_concat_stage1"
            if task == "tbpr":
                result /= c["TBPR_TYPE"]
            result = output_path(result, c["STAGE2_METHOD"])
            if c["PROMPT_TYPE"] == "prompt_shared_5":
                result /= c["PROMPT_TYPE"]
            if c["SELECT_HYBRID_ALPHA"] == "true":
                result /= "alpha_select"
            ranker_mode = c.get("RANKER_LLAVA_IMAGE_MODE", c["LLAVA_IMAGE_MODE"])
            if image_mode == "base" or ranker_mode == "base":
                result = Path(str(result) + f"_retrieval_{image_mode}_ranker_{ranker_mode}")
            result /= f"stage1_{c['STAGE1_DEPTH']}_stage2_{c['STAGE2_DEPTH']}"
            result = path(c["OUTPUT_DIR"]) if c.get("OUTPUT_DIR") else Path(mode_path(result, *use))
            flags = [item for name in ("USE_FREERET_DENSE", "USE_FREERET_PROMPT_CONSTRAINTS", "USE_FREERET_FULL_PROMPT") for item in ("--" + name.lower(), c[name])]
            if action == "encode":
                command = [c["DEEPSPEED_BIN"], f"--num_gpus={c['GPUS']}", f"--num_nodes={c['NUM_NODES']}", "src/encode_standard_task.py", *flags,
                    "--model_name_or_path", str(model), "--dataset_name", dataset, "--dataset_split", c["DATASET_SPLIT"],
                    "--task_type", task, "--encode_type", kind,
                    "--dense_output_dir", str(path(c["DENSE_OUTPUT_ROOT"])), "--sparse_output_dir", str(path(c["SPARSE_OUTPUT_ROOT"])),
                    "--output_dir", str(path(c.get("ENCODE_OUTPUT_DIR") or c.get("OUTPUT_DIR") or str(Path("output/standard_tasks") / tag / dataset / c["DATASET_SPLIT"] / c["PROMPT_TYPE"] / kind))),
                    "--per_device_batch_size", c["PER_DEVICE_BATCH_SIZE"],
                    "--prompt_type", c["PROMPT_TYPE"], "--tbpr_type", c["TBPR_TYPE"], "--eol_type", c["EOL_TYPE"], "--calculate_type", c["CALCULATE_TYPE"],
                    "--sparse_manual", "false", "--reps_loc", reps, "--llava_image_mode", image_mode]
                for name in ("SPARSE_LENGTH", "TEXT_SPARSE_LENGTH", "IMAGE_SPARSE_LENGTH", "SPARSE_VALUE_TYPE", "SPARSE_LOWER_OR_UPPER",
                             "NUM_EXPENDED_TOKENS", "RANDOM_VIEW_SEED"):
                    command += ["--" + name.lower(), c.get(name) or c["SPARSE_LENGTH"]]
                for name in ("SPARSE_VALUE_MEAN", "IS_FILTERED"):
                    command += ["--" + name.lower(), c[name]]
                command += ["--fp16"] if c["DTYPE"] == "float16" else ["--bf16"] if c["DTYPE"] == "bfloat16" else []
            elif action == "index":
                command = [c["PYTHON_BIN"], "-B", "src/freeret_sparse_index.py", "build", "--family", family,
                           "--sparse-root", str(sparse), "--threads", c["SPARSE_THREADS"], *flags]
            else:
                command = [c["DEEPSPEED_BIN"], f"--num_gpus={c['GPUS']}", f"--num_nodes={c['NUM_NODES']}",
                           "src/three_stage_retrieval_pipeline.py", *flags]
                values = dict(dataset_name=dataset, dataset_split=c["DATASET_SPLIT"], task_type=task, query_type=query,
                    tbpr_type=c["TBPR_TYPE"], data_root=str(path(c["DATA_ROOT"])), prompt_type=c["PROMPT_TYPE"],
                    stage1_sparse_index=str(stage1), stage2_dense_index=str(dense), stage2_sparse_index=str(stage2),
                    retrieval_model_name_or_path=str(model), ranker_model_name_or_path=str(ranker), output_dir=str(result),
                    retrieval_dtype=c.get("RETRIEVAL_DTYPE", c["DTYPE"]), ranker_dtype=c.get("RANKER_DTYPE", c["DTYPE"]),
                    retrieval_reps_loc=reps, retrieval_llava_image_mode=image_mode, ranker_llava_image_mode=ranker_mode)
                for name in ("STAGE1_METHOD", "STAGE1_DEPTH", "STAGE2_DEPTH", "STAGE2_METHOD", "HYBRID_ALPHA", "SPARSE_THREADS", "SPARSE_LENGTH",
                    "SPARSE_VALUE_TYPE", "SPARSE_LOWER_OR_UPPER", "NUM_EXPENDED_TOKENS", "RANDOM_VIEW_SEED", "RETRIEVAL_BATCH_SIZE",
                    "RETRIEVAL_DEVICE", "RETRIEVAL_DEVICE_MAP", "RANKER_DEVICE_MAP", "RERANK_TYPE", "RERANK_NUM", "RERANK_BATCH_SIZE",
                    "RERANK_TEMPLATE", "RERANK_REPS_LOC"):
                    values[name.lower()] = c[name]
                values["modify_type"] = modify_type
                for name in ("TEXT_SPARSE_LENGTH", "IMAGE_SPARSE_LENGTH"):
                    values[name.lower()] = c.get(name) or c["SPARSE_LENGTH"]
                for name in ("VAL_PASSAGE_REPS", "VAL_SPARSE_INDEX", "VALIDATION_DATASET_NAME", "VALIDATION_SPLIT", "VAL_STAGE1_SPARSE_INDEX",
                    "VAL_STAGE1_RUN", "STAGE1_RUN", "STAGE1_COMMAND", "STAGE2_RUN", "RERANK_PRIOR_CACHE_DIR",
                    "RETRIEVAL_QWEN_MIN_PIXELS", "RETRIEVAL_QWEN_MAX_PIXELS", "RANKER_QWEN_MIN_PIXELS", "RANKER_QWEN_MAX_PIXELS"):
                    if c.get(name):
                        values[name.lower()] = c[name]
                for name, value in values.items():
                    command += [("--modify_type" if name == "modify_type" else "--" + name.replace("_", "-")), str(value)]
                for name in ("SELECT_HYBRID_ALPHA", "SPARSE_VALUE_MEAN", "IS_FILTERED", "OVERWRITE"):
                    if c[name] == "true":
                        command += ["--" + name.lower().replace("_", "-")]
                if use_prior_cache:
                    command += ["--use-rerank-prior-cache"]
                command += list(extra)
            identity = (str(sparse), str(result))
            if identity in seen:
                raise ValueError(f"Duplicate artifact/output paths: {sparse}")
            seen.add(identity)
            key = f"{family}|{dataset}|{c['DATASET_SPLIT']}" + (f"|{query}" if task == "ir" else "")
            jobs.append(Job(key, action, family, dataset, task, query, model, ranker,
                            dense, sparse, stage1, stage2, result, command))
    return jobs, c


def index_directory(path):
    path = Path(path)
    return path if path.name == "index" else artifact_root(path) / "index"


def require_index(path):
    target = index_directory(path)
    if not target.is_dir() or not any(item.is_file() for item in target.glob("segments_*")):
        raise ValueError(f"No Lucene segments_* at {target}; run the matching sparse_index script")


def preflight(jobs, c):
    errors = []
    for job in jobs:
        try:
            if not shutil.which(job.command[0]):
                raise ValueError(f"Executable not found: {job.command[0]}")
            if job.action != "index":
                for checkpoint in (job.model, job.ranker) if job.action == "retrieve" else (job.model,):
                    if not (checkpoint / "config.json").is_file():
                        raise ValueError(f"Checkpoint config.json missing: {checkpoint}")
                directory, filenames = ANNOTATIONS[job.dataset]
                root = Path(c["DATA_ROOT"]).expanduser()
                if not root.is_absolute():
                    root = PROJECT / root
                for name in filenames:
                    name = name.format(split=c["DATASET_SPLIT"])
                    if not (root / "data" / directory / name).is_file():
                        raise ValueError(f"Dataset annotation missing: {root / 'data' / directory / name}")
            if job.action == "index":
                from freeret_sparse_index import checked_root, sparse_layout
                target = index_directory(job.sparse)
                if target.is_symlink():
                    raise ValueError(f"Refusing a linked index: {target}")
                if target.exists():
                    checked_root(job.sparse, *(c[name] == "true" for name in ("USE_FREERET_DENSE", "USE_FREERET_PROMPT_CONSTRAINTS", "USE_FREERET_FULL_PROMPT")))
                    require_index(target)
                else:
                    sparse_layout(job.sparse, job.family, *(c[name] == "true" for name in ("USE_FREERET_DENSE", "USE_FREERET_PROMPT_CONSTRAINTS", "USE_FREERET_FULL_PROMPT")))
            elif job.action == "retrieve":
                if not (job.dense.is_file() and job.dense.name.startswith("corpus") and job.dense.suffix == ".pkl") and not list(job.dense.glob("corpus*.pkl")):
                    raise ValueError(f"No dense corpus*.pkl at {job.dense}; run the matching encode script")
                for path in (job.dense,) + ((job.stage2,) if c["STAGE2_METHOD"] == "hybrid" else ()):
                    validate_artifact(path, *(c[name] == "true" for name in ("USE_FREERET_DENSE", "USE_FREERET_PROMPT_CONSTRAINTS", "USE_FREERET_FULL_PROMPT")))
                if c.get("STAGE1_RUN"):
                    if not Path(c["STAGE1_RUN"]).is_file():
                        raise ValueError(f"Stage-1 run missing: {c['STAGE1_RUN']}")
                elif not c.get("STAGE1_COMMAND") and c["STAGE1_METHOD"] in {"sparse_concat", "sparse_manual"}:
                    validate_artifact(job.stage1, *(c[name] == "true" for name in ("USE_FREERET_DENSE", "USE_FREERET_PROMPT_CONSTRAINTS", "USE_FREERET_FULL_PROMPT")))
                    require_index(job.stage1)
                if c["STAGE2_METHOD"] == "hybrid":
                    require_index(job.stage2)
                if job.output.exists() and c["OVERWRITE"] != "true":
                    raise ValueError(f"Output already exists: {job.output}; use a new OUTPUT_DIR or OVERWRITE=true")
        except (OSError, ValueError) as exc:
            errors.append(f"{job.key}: {exc}")
    if errors:
        raise ValueError("Preflight failed before starting any job:\n" + "\n".join(errors))


def run(action, group, environment=None, *, extra=(), runner=subprocess.run, project=PROJECT):
    jobs, c = build_jobs(action, group, environment, extra=extra, project=project)
    dry = c["DRY_RUN"] == "true"
    if not dry:
        preflight(jobs, c)
    completed = skipped = 0
    for job in jobs:
        print(f"[{action}] {job.key}\n  model: {job.model}\n  dense: {job.dense}\n  sparse: {job.sparse}", flush=True)
        if action == "retrieve":
            print(f"  Stage 1: {job.stage1}\n  Stage 2: {c['STAGE2_METHOD']}\n  results: {job.output}", flush=True)
        print("  command: " + shlex.join(job.command), flush=True)
        if not dry and action == "index" and index_directory(job.sparse).exists():
            print("  skipped: existing valid index", flush=True)
            skipped += 1
            continue
        if not dry:
            result = runner(job.command, cwd=Path(project), check=False, env={**os.environ, "DATA_ROOT": c["DATA_ROOT"]})
            if result.returncode:
                print(f"Failed {job.key}: exit {result.returncode}; completed={completed}, skipped={skipped}", file=sys.stderr)
                return result.returncode if result.returncode > 0 else 128 - result.returncode
            completed += 1
    print(f"Summary: jobs={len(jobs)} completed={completed} skipped={skipped} planned={len(jobs) if dry else 0}", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog="Edit the .sh configuration or environment. Default: prompt_shared_5, llava_mistral, one dataset; TBPR_TYPE=type. IR defaults to Flickr and both directions (QUERY_TYPES='image text'); QUERY_TYPE=image|text selects one. MODIFY_TYPE=auto selects modify_single for caption-generation IR image-to-text / fused image-text-to-text, otherwise no. MODIFY_TYPE=no|modify_single or the native CLI overrides auto. USE_RERANK_PRIOR_CACHE=auto follows the resolved modify_type per job; true/false override it. Cache requires pointwise caption-generation reranking in a supported direction. MODEL_FAMILIES / DATASETS enable batching. DRY_RUN=true prints paths/commands without reading data or writing files.")
    parser.add_argument("action", choices=("encode", "index", "retrieve"))
    parser.add_argument("group", choices=DATASETS)
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args(argv)
    env = dict(os.environ)
    if args.dry_run:
        env["DRY_RUN"] = "true"
    try:
        return run(args.action, args.group, env, extra=extra)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
