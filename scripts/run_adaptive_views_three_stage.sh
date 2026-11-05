#!/usr/bin/env bash
USE_HIERARCHICAL_VIEWS="${USE_HIERARCHICAL_VIEWS-false}"  # true: ten primary/secondary dimensions (A-J); false: original A-T
case "${USE_HIERARCHICAL_VIEWS,,}" in
    true|1|yes) USE_HIERARCHICAL_VIEWS=true ;;
    false|0|no) USE_HIERARCHICAL_VIEWS=false ;;
    *) echo 'Invalid USE_HIERARCHICAL_VIEWS: use true or false' >&2; exit 2 ;;
esac
VIEW_SELECTION_RUN_ID="${VIEW_SELECTION_RUN_ID-}"  # Shuffle experiment identifier printed by encoding
USE_FREERET_DENSE="${USE_FREERET_DENSE-false}"  # true: FreeRet pre-MLP dense; prompts/sparse unchanged
USE_FREERET_FULL_PROMPT="${USE_FREERET_FULL_PROMPT-false}"  # Complete FreeRet dense/sparse prompts; requires FreeRet dense
USE_FREERET_PROMPT_CONSTRAINTS="${USE_FREERET_PROMPT_CONSTRAINTS-false}"  # Prepend FreeRet constraints to all representation views
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/freeret_dense_mode.sh" || exit $?
# Run retrieval against completed adaptive-view artifacts and existing Lucene indexes.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
    cat <<'USAGE'
Usage: [environment] bash scripts/run_adaptive_views_three_stage.sh [extra pipeline arguments]

Required outputs from encode_adaptive_views.sh or encode_adaptive_views_overlap.sh:
  VIEW_SELECTION_FILE  Path to view_selection.json.
  DENSE_CORPUS_DIR     Dense leaf containing corpus_<rank>.pkl and adaptive_views_manifest.json.
  SPARSE_CORPUS_DIR    Sparse leaf, collection, or index; the Lucene index must already exist.
Optional STAGE1_SPARSE_INDEX overrides the Stage-1 index (default: SPARSE_CORPUS_DIR).
CLI --view-selection-file, --stage1-sparse-index, --stage2-dense-index and
--stage2-sparse-index override the corresponding environment paths; last value wins.

MODEL_FAMILY=qwen2|qwen25|qwen3|llava_llama|llava_mistral (default: qwen25).
MODEL_PATH overrides the checkpoint. DATASET=flickr DATASET_SPLIT=test IR_QUERY_TYPE=image.
Match MODEL_PATH, DTYPE, SPARSE_LENGTH, filtering, pixel and image-mode settings to encode.
Defaults: GPUS=1 DTYPE=float16 SPARSE_LENGTH=40 STAGE1_DEPTH=1000 STAGE2_DEPTH=1000.
RERANK_NUM=5 RERANK_BATCH_SIZE=1 HYBRID_ALPHA=0.5 SELECT_HYBRID_ALPHA=false.
All existing adaptive launcher variables and extra pipeline arguments are supported.
PYTHON_BIN defaults to python in the active environment, used only for artifact checks.
OUTPUT_DIR/--output-dir and OVERWRITE/--overwrite retain the existing launcher behavior.

Stages: adaptive sparse recall -> dense+sparse hybrid ranking -> what_caption_generation reranking.
Reads locked ordinary/overlap views; queries are encoded online and reused in Stage 2.
No candidate encoding, view selection or index construction is performed.
Missing indexes cause an error with a manual build command before models are started.

Example (after encode and sparse indexing, using the actual printed artifact paths):
  VIEW_SELECTION_FILE=/path/to/view_selection.json \
  DENSE_CORPUS_DIR=/path/to/dense/adaptive_views/fingerprint \
  SPARSE_CORPUS_DIR=/path/to/sparse/adaptive_views/fingerprint \
  MODEL_FAMILY=qwen25 DATASET=flickr IR_QUERY_TYPE=image GPUS=1 \
  bash scripts/run_adaptive_views_three_stage.sh
USAGE
}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 2
}

# Resolve artifact overrides before checking them. Other options keep the
# existing launcher's parsing, precedence and result-directory behavior.
PIPELINE_ARGS=()
while (($#)); do
    case "${1%%=*}" in
        -h|--help) usage; exit 0 ;;
        --view-selection-file|--stage1-sparse-index|--stage2-dense-index|--stage2-sparse-index)
            CLI_OPTION="${1%%=*}"
            if [[ "$1" == *=* ]]; then
                CLI_VALUE="${1#*=}"
                shift
            else
                (($# >= 2)) && [[ "$2" != --* ]] || die "${CLI_OPTION} requires a value."
                CLI_VALUE="$2"
                shift 2
            fi
            [[ -n "${CLI_VALUE}" ]] || die "${CLI_OPTION} requires a non-empty value."
            case "${CLI_OPTION}" in
                --view-selection-file) VIEW_SELECTION_FILE="${CLI_VALUE}" ;;
                --stage1-sparse-index) STAGE1_SPARSE_INDEX="${CLI_VALUE}" ;;
                --stage2-dense-index) DENSE_CORPUS_DIR="${CLI_VALUE}" ;;
                --stage2-sparse-index) SPARSE_CORPUS_DIR="${CLI_VALUE}" ;;
            esac
            ;;
        --) PIPELINE_ARGS+=("$@"); break ;;
        *) PIPELINE_ARGS+=("$1"); shift ;;
    esac
done

[[ -n "${VIEW_SELECTION_FILE:-}" ]] || die "Set VIEW_SELECTION_FILE to the selection JSON printed by adaptive encode."
[[ -n "${DENSE_CORPUS_DIR:-}" ]] || die "Set DENSE_CORPUS_DIR to the adaptive dense output directory."
[[ -n "${SPARSE_CORPUS_DIR:-}" ]] || die "Set SPARSE_CORPUS_DIR to the indexed adaptive sparse output directory."
export VIEW_SELECTION_FILE DENSE_CORPUS_DIR SPARSE_CORPUS_DIR
export STAGE1_SPARSE_INDEX="${STAGE1_SPARSE_INDEX:-${SPARSE_CORPUS_DIR}}"
dense_mode_paths
PYTHON_BIN="${PYTHON_BIN:-python}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Python not found: ${PYTHON_BIN}; activate the retrieval environment or set PYTHON_BIN."

"${PYTHON_BIN}" -B - "${VIEW_SELECTION_FILE}" "${DENSE_CORPUS_DIR}" "${SPARSE_CORPUS_DIR}" "${STAGE1_SPARSE_INDEX}" <<'PY'
import json
from pathlib import Path
import shlex
import sys

sys.path.insert(0, str(Path.cwd() / "src"))
from adaptive_views import MANIFEST, load_selection
from qwen_vl_paths import artifact_root, resolve_sparse_index


def fail(message):
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(2)


def completion(root, label):
    path = root / MANIFEST
    if not path.is_file():
        fail(f"Missing {label} completion manifest: {path}; finish adaptive corpus encoding first.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or type(value.get("num_shards")) is not int or value["num_shards"] < 1:
        fail(f"Invalid {label} completion manifest: {path}")
    return value


try:
    selection_path, dense, sparse, first_sparse = (Path(value).expanduser() for value in sys.argv[1:])
    if not selection_path.is_file():
        fail(f"View selection file does not exist: {selection_path}")
    selection = load_selection(selection_path)
    if not dense.is_dir():
        fail(f"Dense corpus directory does not exist: {dense}")
    dense_manifest = completion(dense, "dense")
    shards = [path for path in dense.glob("corpus_*.pkl") if path.is_file()]
    if len(shards) != dense_manifest["num_shards"]:
        fail(f"Dense shard count mismatch at {dense}: found {len(shards)}, expected {dense_manifest['num_shards']}.")
    print(f"Adaptive selection: {selection_path}; modality={selection['selection_modality']} selected_count={len(selection['selected'])}", flush=True)
    print(f"Adaptive dense: {dense}; shards={len(shards)}", flush=True)
    for label, path in (("Stage-1 sparse", first_sparse), ("Stage-2 sparse", sparse)):
        root = artifact_root(path)
        completion(root, label)
        index = resolve_sparse_index(path, require=False)
        if not any(item.is_file() for item in index.glob("segments_*")):
            command = shlex.join([sys.executable, "-B", "src/adaptive_views.py", "index", "--sparse-root", str(root)])
            fail(f"Missing {label} Lucene index (no segments_*) at {index}.\nBuild it separately:\n  {command}")
        print(f"{label}: {index}", flush=True)
except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
    fail(exc)
PY

exec env "USE_HIERARCHICAL_VIEWS=${USE_HIERARCHICAL_VIEWS}" "VIEW_SELECTION_RUN_ID=${VIEW_SELECTION_RUN_ID}" "USE_FREERET_DENSE=${USE_FREERET_DENSE}" "USE_FREERET_PROMPT_CONSTRAINTS=${USE_FREERET_PROMPT_CONSTRAINTS}" "USE_FREERET_FULL_PROMPT=${USE_FREERET_FULL_PROMPT}" bash "${REPO_ROOT}/scripts/three_stage_retrieval_pipeline_adaptive_views.sh" "${PIPELINE_ARGS[@]}"
