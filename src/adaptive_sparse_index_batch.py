"""Discover completed adaptive corpora and build separate impact indexes on CPU."""
from __future__ import annotations

import freeret_full_prompt as full_prompt

import argparse
from dataclasses import dataclass
from itertools import product
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from adaptive_views import MANIFEST, digest, validate_index_selection, is_overlap_selection, IncompatibleViewSelectionError, SelectionFileResolver
from qwen_sparse_index import build_sparse_index
from qwen_vl_paths import resolve_sparse_index


FAMILIES = {
    "qwen2": "qwen2_vl", "qwen25": "qwen2_5_vl", "qwen3": "qwen3_vl",
    "llava_llama": "llava_llama", "llava_mistral": "llava_mistral",
}
GROUPS = {
    "ir": dict(datasets={"flickr": "ir", "coco": "ir"}, kinds=("image", "text"), allowed=("image", "text")),
    "tbpr": dict(datasets={"CUHK-PEDES": "tbpr", "ICFG-PEDES": "tbpr", "RSTPReid": "tbpr"},
                 kinds=("image",), allowed=("image", "text")),
    "fused": dict(datasets={"webqa": "t2it", "remuq": "it2t", "llava": "it2t"},
                  kinds=("corpus",), allowed=("corpus",)),
}
PROJECT = Path(__file__).resolve().parents[1]


def boolean(value):
    if value.lower() in {"true", "1", "yes"}:
        return True
    if value.lower() in {"false", "0", "no"}:
        return False
    raise ValueError(f"DRY_RUN must be true/false or 1/0, got {value!r}")


def configured_list(env, singular, plural, defaults, allowed):
    values = (env.get(singular) or env.get(plural, " ".join(defaults))).split()
    if not values:
        raise ValueError(f"{plural} must contain at least one value")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate values in {plural}: {values}")
    unsupported = [value for value in values if value not in allowed]
    if unsupported:
        raise ValueError(f"Unsupported {plural}: {unsupported}; choose from {', '.join(allowed)}")
    return tuple(values)


@dataclass(frozen=True)
class Settings:
    group: str
    families: tuple[str, ...]
    datasets: tuple[str, ...]
    kinds: tuple[str, ...]
    split: str
    sparse_root: Path
    threads: int
    pmi_mode: str
    dry_run: bool
    overlap_only: bool = False
    selection_root: Path | None = None
    use_freeret_dense: bool = False
    use_freeret_prompt_constraints: bool = False
    exact_encoding: bool = False
    environment: dict[str, str] | None = None
    use_freeret_full_prompt: bool = False

    @classmethod
    def from_environment(cls, group, env=None, *, overlap_only=False, exact_encoding=False):
        env = dict(os.environ if env is None else env)
        defaults = GROUPS[group]
        families = configured_list(env, "MODEL_FAMILY", "MODEL_FAMILIES", FAMILIES, FAMILIES)
        datasets = configured_list(env, "DATASET", "DATASETS", defaults["datasets"], defaults["datasets"])
        kinds = configured_list(env, "ENCODE_TYPE", "ENCODE_TYPES", defaults["kinds"], defaults["allowed"])
        split = env.get("DATASET_SPLIT", "test")
        if not split.strip():
            raise ValueError("DATASET_SPLIT must not be empty")
        threads = int(env.get("THREADS", env.get("SPARSE_THREADS", "16")))
        if threads < 1:
            raise ValueError("THREADS must be a positive integer")
        pmi_mode = env.get("PMI_MODE", "all").lower()
        if pmi_mode not in {"all", "on", "off"}:
            raise ValueError("PMI_MODE must be all, on or off")
        supplied_root = env.get("SPARSE_OUTPUT_ROOT", "sparse_output")
        if not supplied_root.strip():
            raise ValueError("SPARSE_OUTPUT_ROOT must not be empty")
        root = Path(supplied_root).expanduser()
        root = root if root.is_absolute() else PROJECT / root
        selection_root = env.get("VIEW_SELECTION_ROOT")
        if selection_root:
            selection_root = Path(selection_root).expanduser()
            selection_root = selection_root if selection_root.is_absolute() else PROJECT / selection_root
            selection_root = selection_root.resolve()
        if boolean(env.get("USE_FREERET_PROMPT_CONSTRAINTS", "false")) and not boolean(env.get("USE_FREERET_DENSE", "false")):
            raise ValueError("USE_FREERET_PROMPT_CONSTRAINTS=true requires USE_FREERET_DENSE=true")
        full = boolean(env.get("USE_FREERET_FULL_PROMPT", "false"))
        if full and not boolean(env.get("USE_FREERET_DENSE", "false")):
            raise ValueError("USE_FREERET_FULL_PROMPT=true requires USE_FREERET_DENSE=true")
        return cls(group, families, datasets, kinds, split, root.resolve(), threads, pmi_mode,
                   boolean(env.get("DRY_RUN", "false")), overlap_only, selection_root or None,
                   boolean(env.get("USE_FREERET_DENSE", "false")), boolean(env.get("USE_FREERET_PROMPT_CONSTRAINTS", "false")) and not full,
                   exact_encoding, env if exact_encoding else None, full)


@dataclass(frozen=True)
class Job:
    root: Path
    family: str
    dataset: str
    kind: str
    model: str
    fingerprint: str
    pmi: bool
    existing_index: Path | None

    @property
    def key(self):
        return self.family, self.dataset, self.kind


def check_artifact(root, manifest, configuration, *, selection=None, resolver=None):
    """Validate all selected corpora before publishing the first index."""
    if manifest.get("fingerprint") != digest(configuration):
        raise ValueError(f"Corrupt adaptive configuration fingerprint: {root / MANIFEST}")
    fingerprint = configuration.get("views_fingerprint", "")
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError(f"Invalid views_fingerprint at {root}")
    expected_names = {fingerprint[:16], fingerprint[:16] + "_llava_base"}
    from adaptive_view_shuffle import mark_path, FIELD, validate
    from adaptive_views import view_pool, pool_mode
    shuffle = configuration.get(FIELD)
    validate(shuffle, view_pool(pool_mode(configuration)), identity=configuration.get("model"), dataset=configuration.get("dataset"))
    expected_names = {mark_path(Path(name),shuffle).name for name in expected_names}
    from dense_representation import SUFFIX, PROMPT_SUFFIX, FULL_PROMPT_SUFFIX, validate_artifact
    use_dense = bool(configuration.get("dense_representation"))
    if use_dense:
        suffix = FULL_PROMPT_SUFFIX if configuration.get("freeret_full_prompt") else PROMPT_SUFFIX if configuration.get("freeret_prompt_constraints") else ""
        expected_names = {name + SUFFIX + suffix for name in expected_names}
    validate_artifact(root, use_dense, bool(configuration.get("freeret_prompt_constraints")), bool(configuration.get("freeret_full_prompt")))
    if root.name not in expected_names:
        raise ValueError(f"Directory/view fingerprint mismatch at {root}; expected {fingerprint[:16]}")
    if selection is None:
        validate_index_selection(root, manifest, resolver=resolver)
    collection, target = root / "collection", root / "collection/index"
    if collection.resolve() != collection or target.resolve() != target:
        raise ValueError(f"Refusing linked adaptive collection/index at {root}")
    count = manifest.get("num_shards")
    if type(count) is not int or count < 1:
        raise ValueError(f"Invalid num_shards in {root / MANIFEST}")
    documents = sorted(collection.glob("corpus_*.jsonl"))
    if len(documents) != count:
        raise ValueError(f"Sparse shard count mismatch at {collection}: found {len(documents)}, expected {count}")
    for document in documents:
        if not document.is_file() or document.resolve() != document:
            raise ValueError(f"Expected an unlinked corpus file: {document}")
    index = resolve_sparse_index(root, require=False)
    if index.exists() or index.is_symlink():
        if index.resolve() != index or not index.is_dir():
            raise ValueError(f"Expected an unlinked index directory: {index}")
        if not any(path.is_file() for path in index.glob("segments_*")):
            raise ValueError(f"Incomplete existing index (no segments_*) at {index}; existing indexes are not overwritten")
        return index
    return None


def discover(settings, *, incompatible=None, non_overlap=None):
    if not settings.sparse_root.is_dir():
        raise ValueError(f"Sparse output root does not exist: {settings.sparse_root}")
    family_alias = {native: alias for alias, native in FAMILIES.items()}
    resolver = SelectionFileResolver(settings.selection_root, sparse_root=settings.sparse_root)
    jobs, seen = [], set()
    for path in sorted(settings.sparse_root.rglob(MANIFEST)):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            configuration = manifest["configuration"]
            # Validate the namespace before filtering, so a mislabeled experiment
            # cannot be silently interpreted as the other representation mode.
            from dense_representation import marked, prompts_marked, full_prompts_marked
            use_dense = bool(configuration.get("dense_representation"))
            if marked(path.parent) != use_dense:
                raise ValueError(f"Dense mode directory/configuration mismatch: {path}")
            use_prompts = bool(configuration.get("freeret_prompt_constraints"))
            use_full = bool(configuration.get("freeret_full_prompt"))
            if full_prompts_marked(path.parent) != use_full:
                raise ValueError(f"Complete FreeRet prompt directory/configuration mismatch: {path}")
            if use_full != settings.use_freeret_full_prompt:
                if path.parent.resolve() == settings.sparse_root.resolve():
                    raise ValueError(f"Complete FreeRet prompt mode mismatch at explicit target: {path}")
                continue
            if prompts_marked(path.parent) != use_prompts:
                raise ValueError(f"Prompt mode directory/configuration mismatch: {path}")
            if use_prompts != settings.use_freeret_prompt_constraints:
                continue
            if use_dense != settings.use_freeret_dense:
                continue
            model = configuration["model"]
            family = family_alias.get(model["family"])
            dataset, kind = configuration["dataset"], configuration["kind"]
            if (family not in settings.families or dataset not in settings.datasets
                    or kind not in settings.kinds or configuration.get("split") != settings.split):
                continue
            from dense_representation import LEGACY_PROMPT_SUFFIXES, SUFFIX
            if (use_prompts and configuration["freeret_prompt_constraints"].get("version") == 1
                    and path.parent.name.endswith(tuple(SUFFIX + item for item in LEGACY_PROMPT_SUFFIXES))):
                reason = ("FreeRet prompt version v1 lacks task_description; re-encode and index "
                          "with _freeret_constraints_v2")
                if path.parent.resolve() == settings.sparse_root.resolve():
                    raise ValueError(reason)
                if incompatible is not None:
                    incompatible.append((path, reason))
                continue
            if configuration.get("task") != GROUPS[settings.group]["datasets"][dataset]:
                raise ValueError(f"Dataset/task mismatch for {dataset} at {path}")
            calibration = configuration.get("selection_calibration", {})
            pmi = calibration.get("method") == "pmi"
            if settings.pmi_mode != "all" and pmi != (settings.pmi_mode == "on"):
                continue
            root = path.parent.resolve()
            if root in seen:
                continue
            if path.resolve() != root / MANIFEST:
                raise ValueError(f"Refusing a linked completion manifest: {path}")
            selection = None
            if settings.overlap_only:
                selection = validate_index_selection(root, manifest, resolver=resolver)
                if not is_overlap_selection(selection):
                    if non_overlap is not None:
                        non_overlap.append((path, selection["version"]))
                    continue
            existing = check_artifact(root, manifest, configuration, selection=selection, resolver=resolver)
            checkpoint = model.get("checkpoint", {}).get("path", str(root.parent.parent))
            jobs.append(Job(root, family, dataset, kind, checkpoint,
                            configuration["views_fingerprint"], pmi, existing))
            seen.add(root)
        except IncompatibleViewSelectionError as exc:
            if incompatible is not None:
                incompatible.append((path, str(exc)))
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            raise ValueError(f"Cannot use adaptive manifest {path}: {exc}") from exc
    family_order = {value: index for index, value in enumerate(settings.families)}
    dataset_order = {value: index for index, value in enumerate(settings.datasets)}
    kind_order = {value: index for index, value in enumerate(settings.kinds)}
    jobs.sort(key=lambda job: (family_order[job.family], dataset_order[job.dataset], kind_order[job.kind], str(job.root)))
    found = {job.key for job in jobs}
    missing = [key for key in product(settings.families, settings.datasets, settings.kinds) if key not in found]
    return jobs, missing


def encoding_context(settings):
    """Supply the shared path helpers with encoding settings, without retrieval preflight."""
    from types import SimpleNamespace
    from adaptive_encode_batch import CHECKPOINTS, encoding_controls, overlap_controls, resolved_path, split_name
    from adaptive_retrieval_batch import selection_options
    import adaptive_view_shuffle as shuffle

    env = settings.environment or {}
    controls = encoding_controls({k: v for k, v in env.items() if k != "VIEW_SELECTION_FILE"})
    controls.update(overlap_controls(env))
    controls["VIEW_SELECTION_RUN_ID"] = env.get("VIEW_SELECTION_RUN_ID", "")
    if controls["SHUFFLE_VIEW_OPTIONS"] == "true" and not controls["VIEW_SELECTION_RUN_ID"]:
        raise ValueError("Exact indexing needs the original VIEW_SELECTION_RUN_ID when SHUFFLE_VIEW_OPTIONS=true; "
                         "indexing never generates a new shuffle run ID")
    if controls["VIEW_SELECTION_RUN_ID"]:
        shuffle.validate_run_id(controls["VIEW_SELECTION_RUN_ID"])
    controls["RETRIEVAL_LLAVA_IMAGE_MODE"] = controls["LLAVA_IMAGE_MODE"]
    if controls["LLAVA_IMAGE_MODE"] == "base" and any(not f.startswith("llava_") for f in settings.families):
        raise ValueError("LLAVA_IMAGE_MODE=base requires LLaVA-NeXT families only; use anyres for Qwen")
    low, high = int(controls["QWEN_MIN_PIXELS"]), int(controls["QWEN_MAX_PIXELS"])
    if low and high and low > high:
        raise ValueError("QWEN_MIN_PIXELS must not exceed QWEN_MAX_PIXELS")
    if env.get("VIEW_SELECTION_FILE"):
        raise ValueError("Exact indexing resolves the automatic selection cache; for a custom VIEW_SELECTION_FILE, "
                         "set TARGET_SPARSE_DIR to its completed encoding directory in the Bash launcher")
    options = selection_options(env, controls)
    if settings.overlap_only and not options.view_overlap_selection:
        raise ValueError("--overlap-only with --exact-encoding requires USE_VIEW_OVERLAP_SELECTION=true")
    if settings.pmi_mode != "all" and options.view_selection_pmi != (settings.pmi_mode == "on"):
        raise ValueError("PMI_MODE conflicts with USE_VIEW_PMI; exact indexing uses the encoding USE_VIEW_PMI setting")
    def path(name, default):
        return resolved_path(env.get(name) or default, PROJECT, name)
    checkpoint_root = path("CHECKPOINT_ROOT", "/root/autodl-fs/checkpoints")
    if len(settings.families) > 1 and env.get("MODEL_PATH"):
        raise ValueError("Global MODEL_PATH requires one family; use MODEL_PATH_<FAMILY> for batches")
    models = {family: resolved_path(env.get("MODEL_PATH") or env.get(f"MODEL_PATH_{family.upper()}")
                                    or str(checkpoint_root / CHECKPOINTS[family]), PROJECT, "model path")
              for family in settings.families}
    tags = [os.path.normcase(model.name) for model in models.values()]
    if len(set(tags)) != len(tags):
        raise ValueError("Selected model paths must have distinct directory names")
    return SimpleNamespace(models=models, controls=controls, options=options, split=split_name(settings.split, "DATASET_SPLIT"),
                           selection_split=split_name(env.get("VIEW_SELECTION_SPLIT") or settings.split, "VIEW_SELECTION_SPLIT"),
                           roots=dict(data=path("DATA_ROOT", "/root/autodl-tmp"), sparse=settings.sparse_root,
                                      dense=path("DENSE_OUTPUT_ROOT", "./dense_output")))


def discover_exact(settings, *, identity_loader=None, backend=None):
    """Locate one locked experiment per combination; unrelated manifests are never scanned."""
    import adaptive_views as views
    import adaptive_view_shuffle as shuffle
    import adaptive_view_encoding as encoding
    from adaptive_retrieval_batch import namespaces, selection_location

    context = encoding_context(settings)
    c, options = context.controls, context.options
    backend = encoding if backend is None else backend
    identity_loader = encoding.load_model_identity if identity_loader is None else identity_loader
    resolver = SelectionFileResolver(settings.selection_root, sparse_root=settings.sparse_root)
    jobs = []
    for family in settings.families:
        identity = identity_loader(str(context.models[family]), dtype=c["DTYPE"], reps_loc=c["REPS_LOC"],
                                   min_pixels=int(c["QWEN_MIN_PIXELS"]), max_pixels=int(c["QWEN_MAX_PIXELS"]),
                                   image_mode=c["LLAVA_IMAGE_MODE"])
        if identity["family"] != FAMILIES[family]:
            raise ValueError(f"Model family mismatch: {family} points to {identity['family']} at {context.models[family]}")
        for dataset, kind in product(settings.datasets, settings.kinds):
            task = GROUPS[settings.group]["datasets"][dataset]
            model, data, training = namespaces(context, family, dataset, task, kind)
            selection_path, key, strategy = selection_location(context, model, data, training, identity)
            print(f"Exact selection cache: {selection_path}\nSelection cache fingerprint: {digest(key)}", flush=True)
            if not selection_path.is_file():
                details = dict(model=str(context.models[family]), data_root=str(context.roots["data"]),
                               dataset=dataset, split=context.selection_split, dtype=c["DTYPE"],
                               image_mode=c["LLAVA_IMAGE_MODE"], min_pixels=c["QWEN_MIN_PIXELS"], max_pixels=c["QWEN_MAX_PIXELS"],
                               sampling=key["sampling"], modality=options.view_selection_modality,
                               pmi=options.view_selection_pmi, hierarchical=options.view_selection_hierarchical,
                               overlap=options.view_overlap_selection, run_id=c["VIEW_SELECTION_RUN_ID"])
                details.update(example_mode=options.view_example_mode, example_max_tokens=options.view_example_max_tokens)
                if strategy is not None:
                    details["overlap_configuration"] = strategy.configuration
                raise ValueError(f"Missing exact selection cache: {selection_path}; encoding settings={json.dumps(details, ensure_ascii=False)}. "
                                 "Match the original encoding parameters and model/data paths, or set TARGET_SPARSE_DIR explicitly; "
                                 "no other experiment is selected automatically")
            try:
                selection = views.load_selection(selection_path, identity=identity, dataset=dataset,
                                                 selection_modality=options.view_selection_modality,
                                                 view_selection_pmi=options.view_selection_pmi,
                                                 view_selection_hierarchical=options.view_selection_hierarchical,
                                                 view_example_mode=options.view_example_mode,
                                                 view_example_max_tokens=options.view_example_max_tokens)
                views.require_equal(dict(split=selection.get("selection_split")), dict(split=context.selection_split), "Selection source split")
                views.require_equal(selection.get("sampling"), key["sampling"], "Selection sampling")
                views.require_equal(dict(shuffle=selection.get(shuffle.FIELD)), dict(shuffle=key.get(shuffle.FIELD)), "Selection option shuffle")
                if strategy is not None:
                    strategy.validate(selection)
                elif views.is_overlap_selection(selection):
                    raise ValueError("Expected ordinary/PMI selection, found overlap")
                root = backend.artifact_dirs(model, data, training, selection)[1].resolve()
                expected = backend.metadata(model, data, task, kind, identity, selection)
                if (root / MANIFEST).resolve() != root / MANIFEST:
                    raise ValueError(f"Refusing a linked completion manifest: {root / MANIFEST}")
                manifest = views.validate_artifact(root, expected)
                locked = validate_index_selection(root, manifest, resolver=resolver)
                views.require_equal(dict(fingerprint=locked["fingerprint"]), dict(fingerprint=selection["fingerprint"]),
                                    "Automatic cache/manifest selection fingerprint")
                existing = check_artifact(root, manifest, expected["configuration"], selection=locked)
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                raise ValueError(f"Cannot use exact adaptive experiment {family}|{dataset}|{kind}; selection={selection_path}: {exc}") from exc
            jobs.append(Job(root, family, dataset, kind, str(context.models[family]), selection["views_fingerprint"],
                            options.view_selection_pmi, existing))
    return jobs, []


def report_summary(settings, jobs, missing, built, skipped, planned, incompatible=(), non_overlap=()):
    print(f"\nSummary: matched={len(jobs)} built={built} skipped_existing={skipped} "
          f"planned={planned} missing_combinations={len(missing)} skipped_incompatible={len(incompatible)} "
          f"skipped_non_overlap={len(non_overlap)}", flush=True)
    for family, dataset, kind in missing:
        print(f"Missing: {family}|{dataset}|{kind}|{settings.split} (PMI_MODE={settings.pmi_mode})", flush=True)


def run(settings, *, builder=None, identity_loader=None, backend=None):
    incompatible, non_overlap = [], []
    if settings.exact_encoding:
        jobs, missing = discover_exact(settings, identity_loader=identity_loader, backend=backend)
    else:
        jobs, missing = discover(settings, incompatible=incompatible, non_overlap=non_overlap)
    print(f"Adaptive sparse batch: group={settings.group} families={' '.join(settings.families)} "
          f"datasets={' '.join(settings.datasets)} kinds={' '.join(settings.kinds)} split={settings.split} "
          f"PMI_MODE={settings.pmi_mode} threads={settings.threads} dry_run={settings.dry_run} "
          f"overlap_only={settings.overlap_only} exact_encoding={settings.exact_encoding}", flush=True)
    print(f"Sparse output root: {settings.sparse_root}", flush=True)
    for path, reason in incompatible:
        print(f"Skip incompatible: {path}\n  {reason}", flush=True)
    for path, version in non_overlap:
        print(f"Skip non-overlap: {path} (selection version {version})", flush=True)
    built = skipped = planned = 0
    if not jobs:
        report_summary(settings, jobs, missing, built, skipped, planned, incompatible, non_overlap)
        raise ValueError("No matching completed adaptive sparse artifacts; check the filters and SPARSE_OUTPUT_ROOT")
    builder = build_sparse_index if builder is None else builder
    for index, job in enumerate(jobs, 1):
        print(f"\nJob [{index}/{len(jobs)}]: {'|'.join(job.key)} split={settings.split} "
              f"PMI={'on' if job.pmi else 'off'}\nModel: {job.model}\nViews fingerprint: {job.fingerprint}\n"
              f"Input: {job.root / 'collection'}\nOutput: {job.existing_index or job.root / 'collection/index'}", flush=True)
        if job.existing_index is not None:
            skipped += 1
            print("Skip: an existing index has segments_*; it is not overwritten.", flush=True)
            continue
        if settings.dry_run:
            planned += 1
            print("Dry run: would build this fingerprint only.", flush=True)
            continue
        try:
            builder(job.root, threads=settings.threads)
        except (subprocess.CalledProcessError, OSError, ValueError) as exc:
            status = exc.returncode if isinstance(exc, subprocess.CalledProcessError) else 2
            print(f"Failed: {job.root} (exit {status}); stopping batch: {exc}", file=sys.stderr, flush=True)
            report_summary(settings, jobs, missing, built, skipped, planned, incompatible, non_overlap)
            return status
        built += 1
        print(f"Completed [{index}/{len(jobs)}]: {job.root}", flush=True)
    report_summary(settings, jobs, missing, built, skipped, planned, incompatible, non_overlap)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog="""
Configure the Bash launchers by editing their configuration block or environment variables.
MODEL_FAMILIES='qwen2 qwen25 qwen3 llava_llama llava_mistral'; nonempty MODEL_FAMILY selects one.
DATASETS: ir='flickr coco'; tbpr='CUHK-PEDES ICFG-PEDES RSTPReid'; fused='webqa remuq llava'.
Nonempty DATASET selects one. ENCODE_TYPES defaults: ir='image text', tbpr='image', fused='corpus';
nonempty ENCODE_TYPE selects one. TBPR may also select text; fused accepts corpus only.
Lists are whitespace-separated; empty, duplicate and unsupported entries fail.
DATASET_SPLIT=test; SPARSE_OUTPUT_ROOT=<project>/sparse_output; THREADS=16 (SPARSE_THREADS is an alias).
PMI_MODE=all|on|off (default all); on includes PMI and PMI+overlap;
off includes ordinary and overlap artifacts without PMI.
--overlap-only restricts indexing to versions 16/17, as used by sparse_index_adaptive_views_overlap_<task>.sh.
With --overlap-only: PMI_MODE=all selects 16/17; off selects 16; on selects 17.
Ordinary/PMI-only corpora are skipped and summarized as skipped_non_overlap. Original launchers keep all modes.
DRY_RUN=true|false (default false); PYTHON_BIN in the launcher selects the active environment.
All matching fingerprints, checkpoint names and encoding variants are processed separately.
Valid existing indexes are skipped. Missing combinations are summarized; no matches is an error.
The completion manifest must reference its locked selection file; fingerprints must agree.
Missing references are recovered read-only by the full selection_fingerprint; no short-hash or selected-view fallback.
VIEW_SELECTION_ROOT optionally supplies the search directory (relative to project root in batch mode).
Otherwise use SPARSE_OUTPUT_ROOT/view_selection, or the nearest view_selection directory among artifact ancestors.
Only missing references trigger recovery; existing corrupt/mismatched files remain errors. Each directory is scanned once per batch.
Incompatible selection versions/pools (including old overlap 13/15) are skipped and summarized as skipped_incompatible.
Diagnostics include stored/supported versions, candidate counts and differing labels; reselect, encode and rebuild.
Incomplete artifacts/indexes fail before any build. Jobs run sequentially and stop on failure.
--exact-encoding resolves only the automatic cache and corpus matching encoding parameters (no global scan).
It needs local checkpoint config/tokenizer/processor files and the encoding environment, but no weights,
original datasets, dense corpus or GPU. Match DATA_ROOT, model paths, DTYPE, selection/sparse/image settings,
USE_VIEW_PMI, USE_HIERARCHICAL_VIEWS and USE_VIEW_OVERLAP_SELECTION to encoding.
VIEW_EXAMPLE_MODE=auto|fixed defaults auto; match VIEW_EXAMPLE_MAX_TOKENS (default 0) to encoding.
Use fixed to locate legacy fixed-count caches. Manifest-scanning/manual modes read the stored example policy.
Shuffled encoding requires its original VIEW_SELECTION_RUN_ID; no new permutation is generated.
Missing exact caches fail with the expected path; no latest/other experiment fallback.
Without --exact-encoding no checkpoint files are needed. Actual builds require Pyserini and Java.
""", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-group", choices=GROUPS, required=True, help="Task group fixed by each Bash launcher")
    parser.add_argument("--overlap-only", action="store_true", help="Index only overlap and PMI+overlap selections")
    parser.add_argument("--exact-encoding", action="store_true", help="Compute the exact cache/corpus path from encoding parameters")
    args = parser.parse_args(argv)
    try:
        return run(Settings.from_environment(args.task_group, overlap_only=args.overlap_only, exact_encoding=args.exact_encoding))
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
