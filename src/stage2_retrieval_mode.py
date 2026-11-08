"""Stage-2 selection policy; query encodings and corpus fingerprints stay unchanged."""
from pathlib import Path
import os
import re


def method(args):
    value = getattr(args, "stage2_method", "hybrid")
    if value not in {"hybrid", "dense"}:
        raise ValueError("STAGE2_METHOD must be hybrid or dense")
    return value


def artifact_paths(args):
    paths = [(args.stage2_dense_index, True)]
    if method(args) == "hybrid":
        paths.append((args.stage2_sparse_index, False))
    return paths


def configure(args):
    """Drop inactive inputs before generic artifact/alpha validation touches them."""
    args.stage2_method = method(args)
    if args.stage2_method != "dense":
        return
    if int(os.environ.get("RANK", "0")) == 0:
        print("[stage2] dense within Stage-1 candidates; hybrid alpha, alpha scanning and validation inputs are ignored")
    args.hybrid_alpha = None
    args.select_hybrid_alpha = False
    args.stage2_sparse_index = None
    for name in ("val_passage_reps", "val_sparse_index", "val_stage1_run", "val_stage1_sparse_index",
                 "validation_dataset_name", "val_view_selection_file"):
        if hasattr(args, name):
            setattr(args, name, None)


def output_path(path, mode):
    """Mark generated dense results, preserving the existing FreeRet namespace suffix."""
    path = Path(path)
    if mode == "hybrid":
        return path
    if mode != "dense":
        raise ValueError("STAGE2_METHOD must be hybrid or dense")
    name, suffix = path.name, ""
    match = re.search(r"_dense_freeret_pre_mlp(?:_freeret_(?:constraints|full_prompt)_v\d+)?$", name)
    if match:
        suffix, name = match.group(), name[:match.start()]
    if not name.endswith("_stage2_dense"):
        name += "_stage2_dense"
    return path.with_name(name + suffix)


def clear_inactive_reports(path, args):
    if method(args) != "dense" or not getattr(args, "overwrite", False):
        return
    inputs = {Path(value).expanduser().resolve() for name in ("stage1_run", "stage2_run", "ranker_run")
              if (value := getattr(args, name, None))}
    for name in ("stage2_hybrid_run.trec", "stage2_hybrid_run.json", "stage2_sparse_run.trec",
                 "stage2_sparse_run.json", "stage2_alpha_sweep.csv", "alpha_selection.json"):
        report = Path(path) / name
        if report.resolve() not in inputs:
            report.unlink(missing_ok=True)


def adaptive_output_path(path, selection):
    """Make dense-run view markers idempotent, including shuffle + mass + FreeRet."""
    import adaptive_view_shuffle as shuffle
    import adaptive_view_probability_mass as mass
    path = Path(path)
    name, tail = path.name, ""
    suffix = re.search(r"_dense_freeret_pre_mlp(?:_freeret_(?:constraints|full_prompt)_v\d+)?$", name)
    if suffix:
        name, tail = name[:suffix.start()], suffix.group()
    expected = {}
    if selection.get(shuffle.FIELD):
        expected["shuffle"] = selection[shuffle.FIELD]["run_id"]
    if mass.is_selection(selection):
        expected["pm"] = selection["views_fingerprint"][:16]
    while True:
        marker = re.search(r"_(shuffle|pm)_([0-9a-f]{16})$", name)
        if not marker:
            break
        if expected.get(marker[1]) != marker[2]:
            raise ValueError(f"Output path belongs to another adaptive experiment: {path}")
        name = name[:marker.start()]
    for key, value in expected.items():
        name += f"_{key}_{value}"
    return path.with_name(name + tail)
