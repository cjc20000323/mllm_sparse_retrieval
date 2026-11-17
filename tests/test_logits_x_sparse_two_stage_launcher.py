"""Exercise the unified Bash launcher with recording children, without GPU models."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from test_adaptive_rgp import BASH, ROOT


ENTRY = "logits_x_sparse_two_stage_ir.sh"
CHILDREN = {
    "encode": "encode_adaptive_views_ir.sh",
    "index": "sparse_index_adaptive_ir.sh",
    "search": "two_stage_retrieval_pipeline_adaptive_views_ir.sh",
}
CLEARED = (
    "OUTPUT_DIR", "ENCODE_TYPE", "IR_QUERY_TYPE", "VIEW_SELECTION_FILE", "TARGET_SPARSE_DIR",
    "DENSE_CORPUS_DIR", "SPARSE_CORPUS_DIR", "STAGE1_SPARSE_INDEX", "_ADAPTIVE_SHUFFLE_RUN_ID",
    "VAL_PASSAGE_REPS", "VAL_SPARSE_INDEX", "VAL_VIEW_SELECTION_FILE", "VAL_STAGE1_SPARSE_INDEX",
    "VALIDATION_DATASET_NAME", "VALIDATION_SPLIT",
)
FIXED = dict(
    SINGLE_VIEW_REPEAT_SELECTED_COUNT="true", STAGE2_METHOD="hybrid", SELECT_HYBRID_ALPHA="false",
    USE_RERANK_PRIOR_CACHE="false", SPARSE_VALUE_MEAN="false", USE_VIEW_OVERLAP_SELECTION="false",
    USE_STC_SPARSE="false", USE_LENS_SPARSE="false", USE_FREERET_FULL_PROMPT="false",
    SHUFFLE_VIEW_OPTIONS="false", VIEW_SELECTION_RUN_ID="",
)
DEFAULTS = dict(
    MODEL_FAMILY="llava_mistral", DATASET="flickr", DATASET_SPLIT="test",
    ENCODE_TYPES="image text", IR_QUERY_TYPES="image text",
    CHECKPOINT_ROOT="/root/autodl-fs/checkpoints", MODEL_PATH="", DATA_ROOT="/root/autodl-tmp",
    DENSE_OUTPUT_ROOT="/root/autodl-fs/dense_output", SPARSE_OUTPUT_ROOT="/root/autodl-fs/sparse_output",
    VIEW_SELECTION_ROOT="/root/autodl-fs/sparse_output/view_selection",
    ENCODE_LOG_ROOT="/root/autodl-fs/adaptive_encode_logs",
    ADAPTIVE_OUTPUT_ROOT="/root/autodl-fs/two_stage_adaptive_results",
    GPUS="2", NUM_NODES="1", DTYPE="float16", RETRIEVAL_DTYPE="float16",
    PER_DEVICE_BATCH_SIZE="1", RETRIEVAL_BATCH_SIZE="4", SPARSE_THREADS="16", THREADS="16",
    DRY_RUN="false", OVERWRITE="true", DEEPSPEED_BIN="deepspeed", REPS_LOC="after_pad",
    SPARSE_LENGTH="30", TEXT_SPARSE_LENGTH="30", IMAGE_SPARSE_LENGTH="30",
    SPARSE_VALUE_TYPE="sum", SPARSE_LOWER_OR_UPPER="lower", IS_FILTERED="false", NUM_EXPENDED_TOKENS="0",
    TBPR_TYPE="type", LLAVA_IMAGE_MODE="base", RETRIEVAL_LLAVA_IMAGE_MODE="base",
    QWEN_MIN_PIXELS="0", QWEN_MAX_PIXELS="0", USE_FREERET_DENSE="false", USE_FREERET_PROMPT_CONSTRAINTS="false",
    VIEW_SELECTION_MODALITY="both", VIEW_SELECTION_SPLIT="test", VIEW_SELECTION_ROUNDS="20",
    VIEW_TEXTS_PER_ROUND="30", VIEW_IMAGES_PER_ROUND="10", VIEW_SELECTION_SEED="42",
    VIEW_EXAMPLE_MODE="auto", VIEW_EXAMPLE_MAX_TOKENS="0", USE_HIERARCHICAL_VIEWS="true",
    USE_VIEW_PROBABILITY_MASS="true", VIEW_PROBABILITY_MASS_THRESHOLD="0.9", USE_VIEW_PMI="false",
    USE_VIEW_TOKEN_VARIANTS="false", USE_VIEW_PREFIX_TOKENS="false",
    STAGE1_DEPTH="200", STAGE2_DEPTH="200", HYBRID_ALPHA="0.7",
)
CAPTURE_NAMES = (*DEFAULTS, *FIXED, *CLEARED, "PYTHON_BIN", "BASH_BIN",
                 "MODEL_PATH_LLAVA_LLAMA", "MODEL_PATH_LLAVA_MISTRAL")


@pytest.fixture
def launcher(tmp_path):
    if BASH is None:
        pytest.skip("Bash not installed")
    root = tmp_path / "project with spaces"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    entry = scripts / ENTRY
    shutil.copy2(ROOT / "scripts" / ENTRY, entry)
    capture = tmp_path / "recorded stages.jsonl"
    (root / "capture.py").write_text(
        "import json, os, sys\nfrom pathlib import Path\n"
        f"names = {CAPTURE_NAMES!r}\n"
        "stage = sys.argv[1]\n"
        "record = dict(stage=stage, cwd=os.getcwd(), args=sys.argv[2:], "
        "configuration={name: os.environ.get(name) for name in names})\n"
        "with Path(os.environ['CAPTURE']).open('a', encoding='utf-8') as stream:\n"
        "    stream.write(json.dumps(record) + '\\n')\n"
        "sys.exit(int(os.environ.get('FAIL_CODE', '23')) if os.environ.get('FAIL_STAGE') == stage else 0)\n",
        encoding="utf-8",
    )
    for stage, filename in CHILDREN.items():
        (scripts / filename).write_text(
            f'#!/usr/bin/env bash\nset -euo pipefail\n"${{PYTHON_BIN}}" -B capture.py {stage} "$@"\n',
            encoding="utf-8",
        )

    def run(*args, overrides=None):
        capture.unlink(missing_ok=True)
        environment = {name: value for name, value in os.environ.items()
                       if name not in {*CAPTURE_NAMES, "FAIL_STAGE", "FAIL_CODE"}}
        # Keep Linux default paths literal when Git Bash launches native Windows Python.
        environment.update(PYTHON_BIN=Path(sys.executable).as_posix(), BASH_BIN=Path(BASH).as_posix(),
                           CAPTURE=capture.as_posix(), MSYS2_ENV_CONV_EXCL=";".join(CAPTURE_NAMES))
        environment.update(overrides or {})
        result = subprocess.run([BASH, entry.as_posix(), *args], cwd=tmp_path, env=environment,
                                capture_output=True, text=True, encoding="utf-8", timeout=30)
        records = [json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines()] if capture.exists() else []
        return result, records

    return NS(root=root, entry=entry, run=run)


def test_default_runs_three_stages_with_identical_configuration(launcher):
    result, records = launcher.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert [record["stage"] for record in records] == list(CHILDREN)
    assert all(record["configuration"] == records[0]["configuration"] for record in records)
    for record in records:
        assert Path(record["cwd"]).resolve() == launcher.root.resolve()
        assert record["args"] == []
        assert {name: record["configuration"][name] for name in DEFAULTS} == DEFAULTS
        assert {name: record["configuration"][name] for name in FIXED} == FIXED
    assert "K = len(selected)" in result.stdout
    assert "Completed search" in result.stdout


@pytest.mark.parametrize("stage", list(CHILDREN))
def test_single_stage_dispatch(launcher, stage):
    result, records = launcher.run(stage)
    assert result.returncode == 0, result.stdout + result.stderr
    assert [record["stage"] for record in records] == [stage]


def test_overrides_and_space_containing_paths_reach_every_stage(launcher):
    roots = {name: str(launcher.root / name.lower() / "custom data") for name in (
        "CHECKPOINT_ROOT", "MODEL_PATH", "DATA_ROOT", "DENSE_OUTPUT_ROOT", "SPARSE_OUTPUT_ROOT",
        "ENCODE_LOG_ROOT", "ADAPTIVE_OUTPUT_ROOT")}
    overrides = dict(roots, MODEL_FAMILY="llava_llama", DATASET="coco", DATASET_SPLIT="val",
                     IR_QUERY_TYPES="text", DTYPE="bfloat16", RETRIEVAL_DTYPE="float32",
                     LLAVA_IMAGE_MODE="anyres", RETRIEVAL_LLAVA_IMAGE_MODE="base",
                     SPARSE_LENGTH="42", TEXT_SPARSE_LENGTH="37", IMAGE_SPARSE_LENGTH="39",
                     GPUS="1", PER_DEVICE_BATCH_SIZE="2", RETRIEVAL_BATCH_SIZE="3",
                     VIEW_EXAMPLE_MODE="fixed", VIEW_SELECTION_ROUNDS="3", VIEW_TEXTS_PER_ROUND="8",
                     VIEW_IMAGES_PER_ROUND="2", VIEW_SELECTION_SEED="7", VIEW_PROBABILITY_MASS_THRESHOLD="0.8",
                     STAGE1_DEPTH="100", STAGE2_DEPTH="50", HYBRID_ALPHA="0.6")
    result, records = launcher.run("all", overrides=overrides)
    assert result.returncode == 0, result.stdout + result.stderr
    expected = dict(overrides, RETRIEVAL_DTYPE="bfloat16", RETRIEVAL_LLAVA_IMAGE_MODE="anyres",
                    VIEW_SELECTION_SPLIT="val")
    for record in records:
        assert {name: record["configuration"][name] for name in expected} == expected
        assert record["configuration"]["MODEL_PATH_LLAVA_LLAMA"] == roots["CHECKPOINT_ROOT"] + "/llava-hf-llama3-llava-next-8b-hf"


def test_old_job_paths_and_incompatible_modes_do_not_leak(launcher):
    inherited = dict.fromkeys(CLEARED, "stale job with spaces")
    inherited.update({name: "true" for name in FIXED})
    inherited.update(SINGLE_VIEW_REPEAT_SELECTED_COUNT="false", STAGE2_METHOD="dense",
                     VIEW_SELECTION_RUN_ID="old-shuffled-selection")
    result, records = launcher.run("all", overrides=inherited)
    assert result.returncode == 0, result.stdout + result.stderr
    for record in records:
        assert all(record["configuration"][name] is None for name in CLEARED)
        assert {name: record["configuration"][name] for name in FIXED} == FIXED


def test_real_child_entries_forward_shared_settings_to_batch_schedulers(launcher):
    for filename in [*CHILDREN.values(), "stc_sparse_mode.sh", "lens_sparse_mode.sh",
                     "freeret_dense_mode.sh", "stage2_retrieval_mode.sh"]:
        shutil.copy2(ROOT / "scripts" / filename, launcher.root / "scripts" / filename)
    runner = launcher.root / "record python.sh"
    runner.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\n'
        'case "${2:-}" in\n'
        '  src/adaptive_encode_batch.py) stage=encode ;;\n'
        '  src/adaptive_sparse_index_batch.py) stage=index ;;\n'
        '  src/adaptive_retrieval_batch.py) stage=search ;;\n'
        '  *) echo "Unexpected program: $*" >&2; exit 2 ;;\n'
        'esac\n"${CAPTURE_PYTHON}" -B capture.py "${stage}" "$@"\n',
        encoding="utf-8",
    )
    runner.chmod(0o755)
    result, records = launcher.run("all", overrides=dict(
        PYTHON_BIN=runner.as_posix(), CAPTURE_PYTHON=Path(sys.executable).as_posix()))
    assert result.returncode == 0, result.stdout + result.stderr
    assert [record["stage"] for record in records] == list(CHILDREN)
    for record in records:
        assert {name: record["configuration"][name] for name in DEFAULTS} == DEFAULTS
        assert {name: record["configuration"][name] for name in FIXED} == FIXED
    assert "--exact-encoding" in records[1]["args"]
    assert "--stop-after-stage2" in records[2]["args"]


@pytest.mark.parametrize("stage", list(CHILDREN))
def test_child_failure_stops_later_stages_and_preserves_exit_code(launcher, stage):
    result, records = launcher.run("all", overrides=dict(FAIL_STAGE=stage, FAIL_CODE="23"))
    assert result.returncode == 23
    assert [record["stage"] for record in records] == list(CHILDREN)[:list(CHILDREN).index(stage)+1]
    assert f"Failed {stage} (exit 23)" in result.stderr
    assert f"Completed {stage}" not in result.stdout


@pytest.mark.parametrize("option", ["-h", "--help"])
def test_help_needs_no_programs_or_model_configuration(launcher, option):
    result, records = launcher.run(option, overrides=dict(BASH_BIN="missing bash", MODEL_FAMILY="unsupported"))
    assert result.returncode == 0
    assert not records
    assert "all|encode|index|search" in result.stdout
    assert "IR_QUERY_TYPES=text" in result.stdout


@pytest.mark.parametrize("args", [("unknown",), ("all", "search")])
def test_invalid_cli_does_not_start_children(launcher, args):
    result, records = launcher.run(*args)
    assert result.returncode == 2
    assert not records
    assert "Usage:" in result.stderr


@pytest.mark.parametrize("overrides,message", [
    (dict(MODEL_FAMILY="qwen25"), "MODEL_FAMILY"),
    (dict(DATASET="webqa"), "DATASET"),
    (dict(BASH_BIN="missing bash"), "Bash not found"),
])
def test_unsupported_configuration_fails_before_children(launcher, overrides, message):
    result, records = launcher.run(overrides=overrides)
    assert result.returncode == 2
    assert not records
    assert message in result.stderr


def test_bash_syntax(launcher):
    result = subprocess.run([BASH, "-n", launcher.entry.as_posix()], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
