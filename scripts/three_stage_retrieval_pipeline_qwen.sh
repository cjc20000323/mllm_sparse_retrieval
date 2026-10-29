#!/usr/bin/env bash
# Run one Qwen checkpoint over multiple datasets, sequentially.
# Only this new launcher is configured; the existing pipeline is reused unchanged.
#
# Examples (run from any directory):
#   bash scripts/three_stage_retrieval_pipeline_qwen.sh
#   QWEN_FAMILY=qwen3 DATASETS="webqa remuq" bash scripts/three_stage_retrieval_pipeline_qwen.sh
#   QWEN_FAMILY=qwen2 IR_QUERY_TYPE=text bash scripts/three_stage_retrieval_pipeline_qwen.sh
#   QWEN_MODEL_PATH="/root/checkpoints/my Qwen" GPUS=2 bash scripts/three_stage_retrieval_pipeline_qwen.sh
#
# Build matching disassembleeol_separate+concat corpus indexes first. By default,
# the existing launcher resolves dense_output/ and sparse_output/ per dataset.
# Custom per-dataset indexes can be entered in the two tables below. Stage 1
# and Stage 2 share the selected sparse_concat index. Pixel settings must match
# corpus encoding. Models reload per dataset; the first failing job stops the batch.
# Other existing launcher settings can be exported, e.g. RERANK_TEMPLATE,
# RERANK_NUM, PROMPT_TYPE, SPARSE_LENGTH, SPARSE_VALUE_TYPE and TBPR_TYPE.

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: bash scripts/three_stage_retrieval_pipeline_qwen.sh
Configure through the block below or environment variables:
  QWEN_FAMILY       qwen2 | qwen25 | qwen3 (default: qwen25)
  DATASETS          whitespace-separated names (default: "flickr coco")
  QWEN_MODEL_PATH   overrides the family checkpoint under QWEN_CHECKPOINT_ROOT
  QWEN_OUTPUT_ROOT  independent model/dataset/direction/split output directories
  IR_QUERY_TYPE    image | text, applies only to COCO/Flickr (default: image)
  GPUS             processes per dataset (default: 1)
  SELECT_HYBRID_ALPHA  true scans 0.1..0.9; false keeps HYBRID_ALPHA (default: false)
  VAL_PASSAGE_REPS / VAL_SPARSE_INDEX  optional paired validation indexes
  VALIDATION_DATASET_NAME  required for shared validation paths across multiple jobs
  VALIDATION_SPLIT  default: val; per-dataset validation tables are configured below
Supported datasets: coco flickr CUHK-PEDES ICFG-PEDES RSTPReid webqa remuq llava.
Use the per-dataset index tables in this new script for custom corpus paths.
Other CLI arguments are rejected rather than forwarded to every dataset.
USAGE
}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 2
}

if (($#)); then
    if (($# == 1)) && [[ "$1" == --help || "$1" == -h ]]; then
        usage
        exit 0
    fi
    die "Use the configuration block or environment variables; arbitrary CLI arguments are not forwarded. See --help."
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LAUNCHER="${SCRIPT_DIR}/three_stage_retrieval_pipeline.sh"
cd "${REPO_ROOT}"

# ---------------- Configuration: edit here or set environment variables. ----------------
QWEN_FAMILY="${QWEN_FAMILY:-qwen25}"
QWEN_CHECKPOINT_ROOT="${QWEN_CHECKPOINT_ROOT:-/root/autodl-fs/checkpoints}"
DATASETS="${DATASETS-flickr coco}"
QWEN_OUTPUT_ROOT="${QWEN_OUTPUT_ROOT:-${REPO_ROOT}/three_stage_qwen_results}"
IR_QUERY_TYPE="${IR_QUERY_TYPE:-image}"
export GPUS="${GPUS:-1}"
export DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
export DATASET_SPLIT="${DATASET_SPLIT:-test}"
export RETRIEVAL_DTYPE="${RETRIEVAL_DTYPE:-float16}"
export RANKER_DTYPE="${RANKER_DTYPE:-float16}"
export RETRIEVAL_QWEN_MIN_PIXELS="${RETRIEVAL_QWEN_MIN_PIXELS:-0}"
export RETRIEVAL_QWEN_MAX_PIXELS="${RETRIEVAL_QWEN_MAX_PIXELS:-0}"
export RANKER_QWEN_MIN_PIXELS="${RANKER_QWEN_MIN_PIXELS:-0}"
export RANKER_QWEN_MAX_PIXELS="${RANKER_QWEN_MAX_PIXELS:-0}"
export SELECT_HYBRID_ALPHA="${SELECT_HYBRID_ALPHA:-false}"
export VALIDATION_SPLIT="${VALIDATION_SPLIT:-val}"
case "${SELECT_HYBRID_ALPHA,,}" in
    true|1|yes) SELECT_HYBRID_ALPHA=true ;;
    false|0|no) SELECT_HYBRID_ALPHA=false ;;
    *) die "SELECT_HYBRID_ALPHA must be true/false or 1/0." ;;
esac

# Leave empty to use the existing launcher's model/dataset-specific paths.
# Example entries: [flickr]="/path with spaces/flickr/dense" [coco]="/path/coco/dense"
declare -A DENSE_INDEXES=()
# Example entries: [flickr]="/path/flickr/sparse" [coco]="/path/coco/sparse"
declare -A SPARSE_INDEXES=()
# Optional validation indexes, keyed by TARGET dataset. For example:
# VAL_DENSE_INDEXES=([coco]="/path/flickr/val/dense")
# VAL_SPARSE_INDEXES=([coco]="/path/flickr/val/sparse")
# VALIDATION_DATASETS=([coco]="flickr")
# No validation indexes means select on that job's target split.
declare -A VAL_DENSE_INDEXES=()
declare -A VAL_SPARSE_INDEXES=()
declare -A VALIDATION_DATASETS=()
declare -A VAL_STAGE1_INDEXES=()
declare -A VAL_STAGE1_RUNS=()
# ----------------------------------------------------------------------------------------

case "${QWEN_FAMILY}" in
    qwen2) CHECKPOINT_NAME=Qwen2-VL-7B-Instruct ;;
    qwen25) CHECKPOINT_NAME=Qwen2.5-VL-7B-Instruct ;;
    qwen3) CHECKPOINT_NAME=Qwen3-VL-8B-Instruct ;;
    *) die "QWEN_FAMILY must be qwen2, qwen25 or qwen3, got: ${QWEN_FAMILY}" ;;
esac
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${QWEN_CHECKPOINT_ROOT}/${CHECKPOINT_NAME}}"
while [[ "${QWEN_MODEL_PATH}" == */ && "${QWEN_MODEL_PATH}" != / ]]; do
    QWEN_MODEL_PATH="${QWEN_MODEL_PATH%/}"
done
ARTIFACT_TAG="${QWEN_MODEL_PATH##*/}"
[[ -n "${ARTIFACT_TAG}" && "${ARTIFACT_TAG}" != . && "${ARTIFACT_TAG}" != .. ]] || die "Invalid QWEN_MODEL_PATH."
[[ "${IR_QUERY_TYPE}" == image || "${IR_QUERY_TYPE}" == text ]] || die "IR_QUERY_TYPE must be image or text."
[[ "${GPUS}" =~ ^[1-9][0-9]*$ ]] || die "GPUS must be a positive integer."
[[ -f "${LAUNCHER}" ]] || die "Existing launcher not found: ${LAUNCHER}"

read -r -a DATASET_LIST <<< "${DATASETS//$'\n'/ }"
((${#DATASET_LIST[@]})) || die "DATASETS must contain at least one dataset."
declare -A SEEN=()
dataset_task() {
    case "$1" in
        flickr|coco) printf ir ;;
        CUHK-PEDES|ICFG-PEDES|RSTPReid) printf tbpr ;;
        webqa) printf t2it ;;
        remuq|llava) printf it2t ;;
        *) die "Unsupported validation dataset: $1" ;;
    esac
}
validation_job_env() {
    local dataset="$1" dense sparse name first_index first_run
    QWEN_VALIDATION_ENV=()
    name="${VALIDATION_DATASETS[${dataset}]-${VALIDATION_DATASET_NAME:-}}"
    first_index="${VAL_STAGE1_INDEXES[${dataset}]-${VAL_STAGE1_SPARSE_INDEX:-}}"
    first_run="${VAL_STAGE1_RUNS[${dataset}]-${VAL_STAGE1_RUN:-}}"
    if [[ -n "${VAL_DENSE_INDEXES[${dataset}]-}" || -n "${VAL_SPARSE_INDEXES[${dataset}]-}" ]]; then
        dense="${VAL_DENSE_INDEXES[${dataset}]-}"
        sparse="${VAL_SPARSE_INDEXES[${dataset}]-}"
    else
        dense="${VAL_PASSAGE_REPS:-}"
        sparse="${VAL_SPARSE_INDEX:-}"
        if [[ -n "${dense}${sparse}" && ${#DATASET_LIST[@]} -gt 1 && -z "${name}" ]]; then
            die "Shared validation paths require VALIDATION_DATASET_NAME for multiple jobs; otherwise use per-dataset validation tables."
        fi
    fi
    if [[ -n "${dense}${sparse}${name}${first_index}${first_run}" ]]; then
        [[ "${SELECT_HYBRID_ALPHA}" == true ]] || die "Validation inputs require SELECT_HYBRID_ALPHA=true."
        [[ -n "${dense}" && -n "${sparse}" ]] || die "Validation dense/sparse paths must both be provided for ${dataset}."
        name="${name:-${dataset}}"
        [[ "$(dataset_task "${name}")" == "$(dataset_task "${dataset}")" ]] || die "Validation dataset ${name} has a different task from ${dataset}."
        QWEN_VALIDATION_ENV+=("VAL_PASSAGE_REPS=${dense}" "VAL_SPARSE_INDEX=${sparse}" "VALIDATION_DATASET_NAME=${name}")
        [[ -z "${first_index}" ]] || QWEN_VALIDATION_ENV+=("VAL_STAGE1_SPARSE_INDEX=${first_index}")
        [[ -z "${first_run}" ]] || QWEN_VALIDATION_ENV+=("VAL_STAGE1_RUN=${first_run}")
    fi
    return 0
}
for dataset in "${DATASET_LIST[@]}"; do
    case "${dataset}" in
        coco|flickr|CUHK-PEDES|ICFG-PEDES|RSTPReid|webqa|remuq|llava) ;;
        *) die "Unsupported dataset: ${dataset}" ;;
    esac
    [[ -z "${SEEN[${dataset}]+present}" ]] || die "Duplicate dataset: ${dataset}"
    SEEN["${dataset}"]=1
    validation_job_env "${dataset}"
done

printf 'Qwen batch: family=%s model=%s GPUs=%s datasets=%s\n' "${QWEN_FAMILY}" "${QWEN_MODEL_PATH}" "${GPUS}" "${DATASET_LIST[*]}"
for dataset in "${DATASET_LIST[@]}"; do
    case "${dataset}" in
        coco|flickr)
            QUERY="${IR_QUERY_TYPE}"
            if [[ "${QUERY}" == image ]]; then DIRECTION=image_to_text; else DIRECTION=text_to_image; fi
            ;;
        CUHK-PEDES|ICFG-PEDES|RSTPReid) QUERY=text; DIRECTION=text_to_image ;;
        webqa) QUERY=text; DIRECTION=text_to_imagetext ;;
        remuq|llava) QUERY=image; DIRECTION=imagetext_to_text ;;
    esac
    JOB_OUTPUT="${QWEN_OUTPUT_ROOT}/${ARTIFACT_TAG}/${dataset}/${DIRECTION}/${DATASET_SPLIT}"
    if [[ "${DIRECTION}" == text_to_image && "${dataset}" != coco && "${dataset}" != flickr ]]; then
        JOB_OUTPUT+="/${TBPR_TYPE:-origin_type}"
    fi
    if [[ "${SELECT_HYBRID_ALPHA}" == true ]]; then JOB_OUTPUT+="/alpha_select"; fi
    validation_job_env "${dataset}"
    JOB_ENV=(
        "DATASET=${dataset}" "QUERY_TYPE=${QUERY}"
        "BASE_MODEL=${ARTIFACT_TAG}" "ARTIFACT_MODEL_NAME=${ARTIFACT_TAG}"
        "RETRIEVAL_MODEL_PATH=${QWEN_MODEL_PATH}" "RANKER_MODEL_PATH=${QWEN_MODEL_PATH}"
        "OUTPUT_DIR=${JOB_OUTPUT}"
        "${QWEN_VALIDATION_ENV[@]}"
    )
    if [[ -n "${DENSE_INDEXES[${dataset}]-}" ]]; then
        JOB_ENV+=("STAGE2_DENSE_INDEX=${DENSE_INDEXES[${dataset}]}")
    fi
    if [[ -n "${SPARSE_INDEXES[${dataset}]-}" ]]; then
        JOB_ENV+=("STAGE1_SPARSE_INDEX=${SPARSE_INDEXES[${dataset}]}" "STAGE2_SPARSE_INDEX=${SPARSE_INDEXES[${dataset}]}")
    fi
    printf '\n========== %s: %s -> %s ==========\n' "${dataset}" "${QWEN_FAMILY}" "${JOB_OUTPUT}"
    # Clear inherited job-specific values. No unchecked "$@" can override this job.
    if env -u TASK_TYPE -u QUERY_TYPE -u STAGE1_SPARSE_INDEX -u STAGE2_DENSE_INDEX \
        -u STAGE2_SPARSE_INDEX -u OUTPUT_DIR -u BASE_MODEL -u ARTIFACT_MODEL_NAME \
        -u RETRIEVAL_MODEL_PATH -u RANKER_MODEL_PATH \
        -u VAL_PASSAGE_REPS -u VAL_SPARSE_INDEX -u VALIDATION_DATASET_NAME \
        -u VAL_STAGE1_SPARSE_INDEX -u VAL_STAGE1_RUN \
        "${JOB_ENV[@]}" bash "${LAUNCHER}"; then
        printf 'Completed: %s\n' "${dataset}"
    else
        status=$?
        printf 'Failed: %s (exit %s); stopping Qwen batch.\n' "${dataset}" "${status}" >&2
        exit "${status}"
    fi
done
printf '\nCompleted all %s dataset jobs.\n' "${#DATASET_LIST[@]}"
