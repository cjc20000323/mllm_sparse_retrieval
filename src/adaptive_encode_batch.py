"""Run adaptive candidate encoding sequentially across models, datasets and modalities."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    "qwen2": "Qwen-Qwen2-VL-7B-Instruct",
    "qwen25": "Qwen-Qwen2.5-VL-7B-Instruct",
    "qwen3": "Qwen-Qwen3-VL-8B-Instruct",
    "llava_llama": "llava-hf-llama3-llava-next-8b-hf",
    "llava_mistral": "llava-hf-llava-v1.6-mistral-7b-hf",
}
GROUPS = {
    "ir": dict(datasets=("flickr", "coco"), kinds=("image", "text"), allowed=("image", "text")),
    "tbpr": dict(datasets=("CUHK-PEDES", "ICFG-PEDES", "RSTPReid"), kinds=("image",),
                 allowed=("image", "text")),
    "fused": dict(datasets=("webqa", "remuq", "llava"), kinds=("corpus",), allowed=("corpus",)),
}
# These defaults match the single-job launcher. Only these public settings are printed;
# the rest of the parent environment (including credentials) is never logged.
DEFAULTS = {
    "GPUS": "1", "NUM_NODES": "1", "DTYPE": "float16", "LLAVA_IMAGE_MODE": "anyres",
    "PER_DEVICE_BATCH_SIZE": "1", "SPARSE_LENGTH": "40", "SPARSE_VALUE_TYPE": "sum",
    "SPARSE_LOWER_OR_UPPER": "lower", "IS_FILTERED": "false", "SPARSE_VALUE_MEAN": "false",
    "NUM_EXPENDED_TOKENS": "0", "TBPR_TYPE": "origin_type", "REPS_LOC": "after_pad",
    "QWEN_MIN_PIXELS": "0", "QWEN_MAX_PIXELS": "0", "VIEW_SELECTION_ROUNDS": "20",
    "VIEW_TEXTS_PER_ROUND": "8", "VIEW_IMAGES_PER_ROUND": "2", "VIEW_SELECTION_SEED": "42",
    "VIEW_EXAMPLE_MODE": "auto", "VIEW_EXAMPLE_MAX_TOKENS": "0",
    "DEEPSPEED_BIN": "deepspeed", "USE_FREERET_DENSE": "false", "USE_FREERET_PROMPT_CONSTRAINTS": "false", "USE_FREERET_FULL_PROMPT": "false",
    "SHUFFLE_VIEW_OPTIONS": "false",
    "USE_HIERARCHICAL_VIEWS": "false",
}


def boolean(value, name):
    if value.lower() in {"true", "1", "yes"}:
        return True
    if value.lower() in {"false", "0", "no"}:
        return False
    raise ValueError(f"{name} must be true/false or 1/0, got {value!r}")


def configured_list(env, singular, plural, defaults, allowed):
    values = (env.get(singular) or env.get(plural, " ".join(defaults))).split()
    if not values:
        raise ValueError(f"{plural} must contain at least one value")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate values in {plural}: {values}")
    if any(value not in allowed for value in values):
        raise ValueError(f"Unsupported {plural}: {values}; choose from {', '.join(allowed)}")
    return tuple(values)


def resolved_path(value, project, name):
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project / path).resolve()


def split_name(value, name):
    if not value.strip() or value in {".", ".."} or any(char in value for char in "/\\"):
        raise ValueError(f"{name} must be a nonempty split name, got {value!r}")
    return value


def encoding_controls(env):
    controls = {name: env.get(name) or default for name, default in DEFAULTS.items()}
    for name in ("TEXT_SPARSE_LENGTH", "IMAGE_SPARSE_LENGTH"):
        controls[name] = env.get(name) or controls["SPARSE_LENGTH"]
    for name in ("VIEW_SELECTION_MODALITY", "USE_VIEW_PMI"):
        if name in env and (env[name] or name == "USE_VIEW_PMI"):
            controls[name] = env[name]
    for name in ("IS_FILTERED", "SPARSE_VALUE_MEAN", "USE_VIEW_PMI", "USE_FREERET_DENSE", "USE_FREERET_PROMPT_CONSTRAINTS", "USE_FREERET_FULL_PROMPT", "SHUFFLE_VIEW_OPTIONS", "USE_HIERARCHICAL_VIEWS"):
        if name in controls:
            controls[name] = str(boolean(controls[name], name)).lower()
    if controls["USE_FREERET_PROMPT_CONSTRAINTS"] == "true" and controls["USE_FREERET_DENSE"] != "true":
        raise ValueError("USE_FREERET_PROMPT_CONSTRAINTS=true requires USE_FREERET_DENSE=true")
    if controls["USE_FREERET_FULL_PROMPT"] == "true" and controls["USE_FREERET_DENSE"] != "true":
        raise ValueError("USE_FREERET_FULL_PROMPT=true requires USE_FREERET_DENSE=true")
    choices = {
        "DTYPE": ("float16", "bfloat16", "float32"), "LLAVA_IMAGE_MODE": ("anyres", "base"),
        "REPS_LOC": ("before_pad", "after_pad"), "SPARSE_VALUE_TYPE": ("replace", "sum", "max"),
        "SPARSE_LOWER_OR_UPPER": ("lower", "upper"), "TBPR_TYPE": ("origin_type", "type"),
        "VIEW_SELECTION_MODALITY": ("text", "image", "both"),
        "VIEW_EXAMPLE_MODE": ("auto", "fixed"),
    }
    for name, allowed in choices.items():
        if name in controls and controls[name] not in allowed:
            raise ValueError(f"{name} must be one of {', '.join(allowed)}")
    positive = {"GPUS", "NUM_NODES", "PER_DEVICE_BATCH_SIZE", "SPARSE_LENGTH", "TEXT_SPARSE_LENGTH",
                "IMAGE_SPARSE_LENGTH", "VIEW_SELECTION_ROUNDS"}
    nonnegative = {"NUM_EXPENDED_TOKENS", "QWEN_MIN_PIXELS", "QWEN_MAX_PIXELS", "VIEW_EXAMPLE_MAX_TOKENS"}
    mode = controls.get("VIEW_SELECTION_MODALITY", "both")
    # A locked file can supply the mode; let the encoder validate its inherited counts.
    if controls["VIEW_EXAMPLE_MODE"] == "fixed" and (not env.get("VIEW_SELECTION_FILE") or "VIEW_SELECTION_MODALITY" in controls):
        if mode in {"text", "both"}:
            positive.add("VIEW_TEXTS_PER_ROUND")
        if mode in {"image", "both"}:
            positive.add("VIEW_IMAGES_PER_ROUND")
    integers = positive | nonnegative | {"VIEW_TEXTS_PER_ROUND", "VIEW_IMAGES_PER_ROUND", "VIEW_SELECTION_SEED"}
    for name in sorted(integers):
        try:
            value = int(controls[name])
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {controls[name]!r}") from exc
        if (name in positive and value < 1) or (name in nonnegative and value < 0):
            raise ValueError(f"{name} must be {'positive' if name in positive else 'nonnegative'}")
    return controls


def overlap_controls(env):
    """Validate only explicit overrides; omitted tuning values can inherit a locked file."""
    result = {"USE_VIEW_OVERLAP_SELECTION": str(boolean(
        env.get("USE_VIEW_OVERLAP_SELECTION", "false"), "USE_VIEW_OVERLAP_SELECTION")).lower()}
    for name in ("VIEW_SELECTED_COUNT", "VIEW_OVERLAP_SAMPLES", "VIEW_OVERLAP_SEED", "VIEW_OVERLAP_THRESHOLD"):
        if name not in env:
            continue
        try:
            value = float(env[name]) if name == "VIEW_OVERLAP_THRESHOLD" else int(env[name])
        except ValueError as exc:
            raise ValueError(f"Invalid {name}: {env[name]!r}") from exc
        size = 10 if boolean(env.get("USE_HIERARCHICAL_VIEWS", "false"), "USE_HIERARCHICAL_VIEWS") else 20
        if ((name == "VIEW_SELECTED_COUNT" and not 1 <= value <= size)
                or (name == "VIEW_OVERLAP_SAMPLES" and value < 1)
                or (name == "VIEW_OVERLAP_THRESHOLD" and (not math.isfinite(value) or not 0 <= value <= 1))):
            raise ValueError(f"Invalid {name}: {env[name]!r}")
        result[name] = str(value)
    return result


@dataclass(frozen=True)
class Settings:
    group: str
    families: tuple[str, ...]
    datasets: tuple[str, ...]
    kinds: tuple[str, ...]
    split: str
    selection_split: str
    models: dict[str, Path]
    data_root: Path
    dense_root: Path
    sparse_root: Path
    log_root: Path
    output_dir: Path | None
    selection_file: Path | None
    dry_run: bool
    bash: str
    controls: dict[str, str]
    environment: dict[str, str]
    project: Path
    encoder: str = "ordinary"
    shuffle_run_id: str | None = None

    @classmethod
    def from_environment(cls, group, env=None, *, project=PROJECT, encoder="ordinary"):
        if encoder not in {"ordinary", "overlap"}:
            raise ValueError(f"Invalid encoder: {encoder}")
        env = dict(os.environ if env is None else env)
        project = Path(project).resolve()
        defaults = GROUPS[group]
        families = configured_list(env, "MODEL_FAMILY", "MODEL_FAMILIES", CHECKPOINTS, CHECKPOINTS)
        datasets = configured_list(env, "DATASET", "DATASETS", defaults["datasets"], defaults["datasets"])
        kinds = configured_list(env, "ENCODE_TYPE", "ENCODE_TYPES", defaults["kinds"], defaults["allowed"])
        split = split_name(env.get("DATASET_SPLIT", "test"), "DATASET_SPLIT")
        selection_split = split_name(env.get("VIEW_SELECTION_SPLIT") or split, "VIEW_SELECTION_SPLIT")
        if env.get("MODEL_PATH") and len(families) != 1:
            raise ValueError("MODEL_PATH requires one model family; use MODEL_PATH_<FAMILY> for multiple families")
        if env.get("VIEW_SELECTION_FILE") and len(families) * len(datasets) != 1:
            raise ValueError("VIEW_SELECTION_FILE requires one model/dataset pair; omit it to use automatic caches")
        if env.get("OUTPUT_DIR") and len(families) * len(datasets) * len(kinds) != 1:
            raise ValueError("OUTPUT_DIR requires one encoding job; use ENCODE_LOG_ROOT for batches")
        def path(name, default):
            return resolved_path(env.get(name, default), project, name)

        checkpoint_root = path("CHECKPOINT_ROOT", "/root/autodl-fs/checkpoints")
        models = {
            family: resolved_path(env.get("MODEL_PATH") or env.get(f"MODEL_PATH_{family.upper()}")
                                  or str(checkpoint_root / CHECKPOINTS[family]), project, f"{family} model path")
            for family in families
        }
        tags = [os.path.normcase(model.name) for model in models.values()]
        if not all(tags) or len(tags) != len(set(tags)):
            raise ValueError("Selected model paths must have distinct directory names to avoid artifact collisions")
        controls = encoding_controls(env)
        if encoder == "overlap":
            controls.update(overlap_controls(env))
        if controls["LLAVA_IMAGE_MODE"] == "base" and any(not family.startswith("llava_") for family in families):
            raise ValueError("LLAVA_IMAGE_MODE=base requires LLaVA-NeXT families only")
        from adaptive_view_shuffle import new_run_id
        shuffle_run_id = new_run_id() if controls["SHUFFLE_VIEW_OPTIONS"] == "true" else None
        return cls(group, families, datasets, kinds, split, selection_split, models,
                   path("DATA_ROOT", "/root/autodl-tmp"), path("DENSE_OUTPUT_ROOT", "./dense_output"),
                   path("SPARSE_OUTPUT_ROOT", "./sparse_output"), path("ENCODE_LOG_ROOT", "./adaptive_encode_logs"),
                   path("OUTPUT_DIR", "") if env.get("OUTPUT_DIR") else None,
                   path("VIEW_SELECTION_FILE", "") if env.get("VIEW_SELECTION_FILE") else None,
                   boolean(env.get("DRY_RUN", "false"), "DRY_RUN"), env.get("BASH_BIN") or "bash",
                   controls, env, project, encoder, shuffle_run_id)


def annotation_files(data_root, dataset, split):
    """Match dataset.py plus the adaptive pipeline's _patch_dataset_paths mapping."""
    base = data_root / "data"
    if dataset in {"flickr", "coco"}:
        return (base / dataset / f"{dataset}_{split}.csv",)
    tbpr = {"CUHK-PEDES": "reid_raw.json", "ICFG-PEDES": "ICFG-PEDES.json", "RSTPReid": "data_captions.json"}
    if dataset in tbpr:
        return (base / dataset / tbpr[dataset],)
    if dataset == "webqa":
        return tuple(base / "WebQA" / name for name in (
            "query-00000-of-00001.parquet", "qrels-00000-of-00001.parquet",
            "corpus-00000-of-00002.parquet", "corpus-00001-of-00002.parquet"))
    directory = {"remuq": "UMRB-ReMuQ", "llava": "UMRB-LLaVA"}[dataset]
    return tuple(base / directory / name for name in ("query.parquet", "qrels.parquet", "corpus.parquet"))


@dataclass(frozen=True)
class Job:
    family: str
    dataset: str
    kind: str
    variables: dict[str, str]

    @property
    def key(self):
        return f"{self.family}|{self.dataset}|{self.kind}"


def prepare_jobs(settings):
    """Inspect the entire matrix before the first model process can start."""
    for root in (settings.dense_root, settings.sparse_root, settings.log_root):
        if root.exists() and not root.is_dir():
            raise ValueError(f"Expected an output root directory: {root}")
    if settings.selection_file and settings.selection_file.exists() and not settings.selection_file.is_file():
        raise ValueError(f"Expected a view selection file: {settings.selection_file}")
    if settings.shuffle_run_id and settings.selection_file and settings.selection_file.exists():
        raise ValueError("Shuffled encoding starts a new experiment; omit VIEW_SELECTION_FILE or choose a nonexistent file")
    jobs, skipped = [], []
    for family, dataset, kind in product(settings.families, settings.datasets, settings.kinds):
        model = settings.models[family]
        required = dict.fromkeys((model / "config.json", *annotation_files(settings.data_root, dataset, settings.split),
                                  *annotation_files(settings.data_root, dataset, settings.selection_split)))
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            skipped.append((f"{family}|{dataset}|{kind}", missing))
            continue
        output = settings.output_dir or (settings.log_root / family / model.name / dataset / settings.split / kind)
        from adaptive_view_shuffle import mark_path
        if settings.shuffle_run_id:
            output = mark_path(output,{"run_id":settings.shuffle_run_id})
        from dense_representation import mode_path
        output = mode_path(output, boolean(settings.controls["USE_FREERET_DENSE"], "USE_FREERET_DENSE"),
                           boolean(settings.controls["USE_FREERET_PROMPT_CONSTRAINTS"], "USE_FREERET_PROMPT_CONSTRAINTS"),
                           boolean(settings.controls["USE_FREERET_FULL_PROMPT"], "USE_FREERET_FULL_PROMPT"))
        if output.exists() and not output.is_dir():
            raise ValueError(f"Expected a job output directory: {output}")
        variables = dict(settings.controls, MODEL_FAMILY=family, MODEL_PATH=str(model), DATASET=dataset,
                         ENCODE_TYPE=kind, IR_QUERY_TYPE="text" if kind == "image" else "image",
                         DATASET_SPLIT=settings.split, VIEW_SELECTION_SPLIT=settings.selection_split,
                         DATA_ROOT=str(settings.data_root), DENSE_OUTPUT_ROOT=str(settings.dense_root),
                         SPARSE_OUTPUT_ROOT=str(settings.sparse_root), OUTPUT_DIR=str(output))
        if settings.selection_file:
            variables["VIEW_SELECTION_FILE"] = str(settings.selection_file)
        if settings.shuffle_run_id:
            variables["_ADAPTIVE_SHUFFLE_RUN_ID"] = settings.shuffle_run_id
        jobs.append(Job(family, dataset, kind, variables))
    return jobs, skipped


def report_summary(jobs, skipped, completed, planned, failed=0):
    print(f"\nSummary: runnable={len(jobs)} completed={completed} planned={planned} "
          f"skipped_missing={len(skipped)} failed={failed} remaining={len(jobs) - completed - planned - failed}", flush=True)


def run(settings, *, runner=None, launcher=None):
    if launcher is None:
        name = "encode_adaptive_views_overlap.sh" if settings.encoder == "overlap" else "encode_adaptive_views.sh"
        command = [settings.bash, "scripts/" + name]
    else:
        script = resolved_path(str(launcher), settings.project, "launcher")
        if not script.is_file():
            raise ValueError(f"Encoding launcher does not exist: {script}")
        command = [settings.bash, script.as_posix(), "--run-one"]
    jobs, skipped = prepare_jobs(settings)
    if settings.shuffle_run_id:
        print(f"Option shuffle run: {settings.shuffle_run_id}; one mapping per model/dataset",flush=True)
    print(f"Adaptive encode batch: group={settings.group} families={' '.join(settings.families)} "
          f"datasets={' '.join(settings.datasets)} kinds={' '.join(settings.kinds)} "
          f"split={settings.split} dry_run={settings.dry_run} encoder={settings.encoder}", flush=True)
    if settings.shuffle_run_id:
        print("Each model/dataset starts a fresh shuffled experiment; sibling modality jobs share only this run's cache.",flush=True)
    else:
        print("Every runnable job re-encodes its corpus; existing view-selection caches are reused.\n"
              "The encoder replaces old shards and backs up existing indexes for the same artifact.", flush=True)
    print(f"Preflight: runnable={len(jobs)} skipped_missing={len(skipped)}", flush=True)
    for key, missing in skipped:
        print(f"Skip missing: {key}\n  " + "\n  ".join(missing), flush=True)
    if not jobs:
        report_summary(jobs, skipped, 0, 0)
        raise ValueError("No runnable encoding jobs; check model config.json files and dataset annotations")
    completed = planned = 0
    runner = subprocess.run if runner is None else runner
    for index, job in enumerate(jobs, 1):
        print(f"\nJob [{index}/{len(jobs)}]: {job.key}", flush=True)
        print(shlex.join(["env", *(f"{name}={value}" for name, value in job.variables.items()), *command]), flush=True)
        if settings.dry_run:
            planned += 1
            continue
        try:
            result = runner(command, cwd=settings.project, env=dict(settings.environment, **job.variables), check=False)
            status = result.returncode if result.returncode >= 0 else 128 - result.returncode
        except OSError as exc:
            print(f"Cannot launch {job.key}: {exc}", file=sys.stderr, flush=True)
            status = 2
        if status:
            print(f"Failed: {job.key} (exit {status}); stopping batch.", file=sys.stderr, flush=True)
            report_summary(jobs, skipped, completed, planned, failed=1)
            return status
        completed += 1
        print(f"Completed [{index}/{len(jobs)}]: {job.key}", flush=True)
    report_summary(jobs, skipped, completed, planned)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="""
Configure the Bash launchers' configuration blocks or environment variables (no extra encode CLI arguments).
Each Bash launcher lists every setting and executes its own full DeepSpeed command for each planned job.
The shared scheduler performs preflight/iteration; its internal --run-one calls do not recurse into the batch.
MODEL_FAMILIES='qwen2 qwen25 qwen3 llava_llama llava_mistral'.
DATASETS: ir='flickr coco'; tbpr='CUHK-PEDES ICFG-PEDES RSTPReid'; fused='webqa remuq llava'.
ENCODE_TYPES: ir='image text'; tbpr='image' (text also allowed); fused='corpus'.
Nonempty MODEL_FAMILY, DATASET and ENCODE_TYPE override the corresponding lists. Lists are whitespace-separated.
ENCODE_TYPES describes candidate modalities, not query modalities; IR_QUERY_TYPE is derived for each job.
CHECKPOINT_ROOT=/root/autodl-fs/checkpoints uses the existing single-job checkpoint directory names.
MODEL_PATH_QWEN2, MODEL_PATH_QWEN25, MODEL_PATH_QWEN3, MODEL_PATH_LLAVA_LLAMA, MODEL_PATH_LLAVA_MISTRAL
override individual families. Global MODEL_PATH requires one selected family.
DATA_ROOT=/root/autodl-tmp; DATASET_SPLIT=test; VIEW_SELECTION_SPLIT defaults to DATASET_SPLIT.
DENSE_OUTPUT_ROOT=./dense_output; SPARSE_OUTPUT_ROOT=./sparse_output.
ENCODE_LOG_ROOT=./adaptive_encode_logs creates per-family/checkpoint/dataset/split/modality job output paths.
OUTPUT_DIR requires one job. VIEW_SELECTION_FILE requires one model/dataset pair (both IR modalities allowed).
VIEW_EXAMPLE_MODE=auto|fixed defaults auto. VIEW_EXAMPLE_MAX_TOKENS=0 uses the model context limit.
Positive budgets may lower that limit; one answer token is reserved. Auto mode ignores fixed per-round counts.
Fixed mode preserves legacy cache/artifact fingerprints; set it to reuse existing fixed-count selections.
Overlap launchers use encode_adaptive_views_overlap.py and default USE_VIEW_OVERLAP_SELECTION=true.
USE_VIEW_PMI=true can be combined with overlap; corrected probabilities determine the probe order.
USE_HIERARCHICAL_VIEWS=true replaces A-T with the ten primary/secondary dimensions A-J (default false).
It can be combined with PMI, overlap and SHUFFLE_VIEW_OPTIONS; overlap count must then be 1-10.
VIEW_SELECTED_COUNT=5 VIEW_OVERLAP_THRESHOLD=0.1 VIEW_OVERLAP_SAMPLES=20 VIEW_OVERLAP_SEED=42.
Overlap is shared tokens / candidate unique tokens; accept strictly below threshold. Raise it to relax selection.
New overlap/PMI+overlap versions are 16/17; old versions 13/15 cannot be reused.
Omitted PMI/modality/overlap options inherit a named selection file. Probes use top-30 sparse terms.
GPUS=1 NUM_NODES=1 DTYPE=float16 LLAVA_IMAGE_MODE=anyres PER_DEVICE_BATCH_SIZE=1 SPARSE_LENGTH=40.
All encoding environment settings documented by scripts/encode_adaptive_views.sh are forwarded unchanged.
DRY_RUN=true|false (default false) prints runnable jobs/commands and missing inputs without writing output.
Missing model config.json/annotation files are skipped and summarized; zero runnable jobs is an error.
Each runnable job ALWAYS re-encodes, replacing old shards and backing up old indexes through the encoder.
View selection still uses its existing caches. Jobs run sequentially and stop at the first actual failure.
Only annotation/config existence is checked, not every image or weight file. No indexing/retrieval is started.
PYTHON_BIN selects the launcher's interpreter; BASH_BIN=bash and DEEPSPEED_BIN=deepspeed select executables.
Relative paths are resolved from the project root, even when launched from another directory.
""")
    parser.add_argument("--task-group", choices=GROUPS, required=True, help="Fixed by each Bash launcher")
    parser.add_argument("--launcher", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--encoder", choices=("ordinary", "overlap"), default="ordinary", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        return run(Settings.from_environment(args.task_group, encoder=args.encoder), launcher=args.launcher)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        print("Encoding batch interrupted.", file=sys.stderr, flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
