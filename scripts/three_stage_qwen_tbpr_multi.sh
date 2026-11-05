#!/usr/bin/env bash
USE_FREERET_DENSE="${USE_FREERET_DENSE-false}"  # true: FreeRet pre-MLP dense; prompts/sparse unchanged
USE_FREERET_FULL_PROMPT="${USE_FREERET_FULL_PROMPT-false}"  # Complete FreeRet dense/sparse prompts; requires FreeRet dense
USE_FREERET_PROMPT_CONSTRAINTS="${USE_FREERET_PROMPT_CONSTRAINTS-false}"  # Prepend FreeRet constraints to all representation views
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/freeret_dense_mode.sh" || exit $?
# Standalone multi-model, multi-dataset Qwen TBPR: sparse -> hybrid -> rerank.
# Prepare each model/dataset with encode_qwen_tbpr.sh and sparse_index_qwen.sh.
# Examples (run from any directory):
#   bash scripts/three_stage_qwen_tbpr_multi.sh
#   QWEN_FAMILIES="qwen25 qwen3" DATASETS="CUHK-PEDES ICFG-PEDES" bash scripts/three_stage_qwen_tbpr_multi.sh
#   QWEN_FAMILY=qwen3 DATASET=RSTPReid SELECT_HYBRID_ALPHA=true bash scripts/three_stage_qwen_tbpr_multi.sh
set -euo pipefail

die() { printf 'Error: %s\n' "$*" >&2; exit 2; }
usage() {
    cat <<'USAGE'
Usage: bash scripts/three_stage_qwen_tbpr_multi.sh [-h|--help]
Default result paths end in stage1_<STAGE1_DEPTH>_stage2_<STAGE2_DEPTH>.
Explicit OUTPUT_DIR stays unchanged.
Default: all 3 Qwen families x all 3 TBPR datasets = 9 sequential jobs.
QWEN_FAMILIES="qwen2 qwen25 qwen3"; DATASETS="CUHK-PEDES ICFG-PEDES RSTPReid".
QWEN_FAMILY selects one family instead of QWEN_FAMILIES when nonempty.
DATASET selects one dataset instead of DATASETS when nonempty.
Order: each family in list order, then every dataset in list order.
TASK_TYPE=tbpr, QUERY_TYPE=text, candidate artifacts=image.
TBPR_TYPE=origin_type|type (default: origin_type) must match encoding.
Prior correction/cache is unsupported; MODIFY_TYPE=no and USE_RERANK_PRIOR_CACHE=false.

Edit the configuration block or set environment variables. This standalone
script directly launches src/three_stage_retrieval_pipeline.py.
Model paths default to QWEN_CHECKPOINT_ROOT/<family-checkpoint>:
  qwen2: Qwen2-VL-7B-Instruct; qwen25: Qwen2.5-VL-7B-Instruct;
  qwen3: Qwen3-VL-8B-Instruct. QWEN_CHECKPOINT_ROOT defaults to /root/autodl-fs/checkpoints.
MODEL_PATHS/RANKER_MODEL_PATHS in the configuration block override each family.
Each ranker defaults to the current retrieval model, independently for every family.
QWEN_MODEL_PATH/RANKER_MODEL_PATH override the tables only with ONE selected family.
Family and dataset lists are whitespace-separated; duplicates/empty lists are rejected.
Defaults: test, 1 GPU, float16, prompt_5, disassembleeol_separate+concat,
SPARSE_LENGTH=40, Stage-1/2 depths=1000, what_caption_generation top 5.
Prepare matching encode_qwen_tbpr.sh artifacts and sparse_index_qwen.sh
indexes first. Keep prompt, sparse, seed and retrieval pixel settings identical.
DENSE_OUTPUT_ROOT/SPARSE_OUTPUT_ROOT default to project dense_output/sparse_output.
STAGE2_DENSE_INDEX/STAGE2_SPARSE_INDEX/STAGE1_SPARSE_INDEX override a single job;
use configuration tables keyed by "family|dataset" for batch overrides.
Sparse inputs accept an encoding leaf, collection/, index/ or collection/index/.
collection/index is preferred; legacy index is used only if the nested path is absent.
SELECT_HYBRID_ALPHA=false uses HYBRID_ALPHA=0.5 and ignores validation inputs.
SELECT_HYBRID_ALPHA=true scans 0.1..0.9 using paired VAL_PASSAGE_REPS/VAL_SPARSE_INDEX;
omit both to select on the target split. VALIDATION_SPLIT defaults to val.
VAL_PASSAGE_REPS/VAL_SPARSE_INDEX/VALIDATION_DATASET_NAME are single-job overrides;
use paired VAL_DENSE_INDEXES/VAL_SPARSE_INDEXES and optional VALIDATION_DATASETS
keyed by "family|dataset" in batches. Fixed alpha ignores all validation inputs.
The Python pipeline writes each alpha and the best result to CSV.
QWEN_OUTPUT_ROOT defaults to project three_stage_qwen_results; derived output
paths distinguish model, dataset, encoding settings, direction and alpha mode.
OUTPUT_DIR overrides a single job only. OVERWRITE defaults to false.
All combinations are checked before launching any GPU process. Existing output
directories require OVERWRITE=true. Duplicate model directory names/output paths
are rejected. Any failing job stops the entire batch with its original exit code.
GPUS/NUM_NODES/DEEPSPEED_BIN default to 1/1/deepspeed.
RETRIEVAL_DTYPE/RANKER_DTYPE accept float16|bfloat16|float32; DTYPE supplies both defaults.
RETRIEVAL_QWEN_MIN_PIXELS/MAX_PIXELS and RANKER_QWEN_MIN_PIXELS/MAX_PIXELS
inherit QWEN_MIN_PIXELS/MAX_PIXELS, or 0 for checkpoint defaults.
Only -h/--help is accepted; arbitrary CLI arguments are not forwarded.

Examples:
  bash scripts/three_stage_qwen_tbpr_multi.sh
  QWEN_FAMILIES="qwen25 qwen3" DATASETS="CUHK-PEDES ICFG-PEDES" bash scripts/three_stage_qwen_tbpr_multi.sh
  QWEN_FAMILY=qwen3 DATASET=RSTPReid SELECT_HYBRID_ALPHA=true bash scripts/three_stage_qwen_tbpr_multi.sh
USAGE
}
if (($#)); then
    if (($# == 1)) && [[ "$1" == -h || "$1" == --help ]]; then usage; exit 0; fi
    die 'Configure this script or environment variables. See --help.'
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# ---------------- Configuration: edit here or use environment variables. ----------------
QWEN_FAMILY="${QWEN_FAMILY:-}"
QWEN_FAMILIES="${QWEN_FAMILIES-qwen2 qwen25 qwen3}"
QWEN_CHECKPOINT_ROOT="${QWEN_CHECKPOINT_ROOT:-/root/autodl-fs/checkpoints}"
REQUESTED_MODEL_PATH="${QWEN_MODEL_PATH:-}"
REQUESTED_RANKER_PATH="${RANKER_MODEL_PATH:-}"
DATASET="${DATASET:-}"
DATASETS="${DATASETS-CUHK-PEDES ICFG-PEDES RSTPReid}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
DENSE_OUTPUT_ROOT="${DENSE_OUTPUT_ROOT:-${REPO_ROOT}/dense_output}"
SPARSE_OUTPUT_ROOT="${SPARSE_OUTPUT_ROOT:-${REPO_ROOT}/sparse_output}"
QWEN_OUTPUT_ROOT="${QWEN_OUTPUT_ROOT:-${REPO_ROOT}/three_stage_qwen_results}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
GPUS="${GPUS:-1}"
NUM_NODES="${NUM_NODES:-1}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"
STAGE1_DEPTH="${STAGE1_DEPTH:-1000}"
STAGE2_DEPTH="${STAGE2_DEPTH:-1000}"
SPARSE_THREADS="${SPARSE_THREADS:-${THREADS:-16}}"
RETRIEVAL_BATCH_SIZE="${RETRIEVAL_BATCH_SIZE:-1}"
RETRIEVAL_DEVICE="${RETRIEVAL_DEVICE:-auto}"
RETRIEVAL_DEVICE_MAP="${RETRIEVAL_DEVICE_MAP:-auto}"
RETRIEVAL_DTYPE="${RETRIEVAL_DTYPE:-${DTYPE:-float16}}"
RETRIEVAL_REPS_LOC="${RETRIEVAL_REPS_LOC:-${REPS_LOC:-after_pad}}"
RETRIEVAL_QWEN_MIN_PIXELS="${RETRIEVAL_QWEN_MIN_PIXELS:-${QWEN_MIN_PIXELS:-0}}"
RETRIEVAL_QWEN_MAX_PIXELS="${RETRIEVAL_QWEN_MAX_PIXELS:-${QWEN_MAX_PIXELS:-0}}"
PROMPT_TYPE="${PROMPT_TYPE:-prompt_5}"
RANDOM_VIEW_SEED="${RANDOM_VIEW_SEED:-42}"
SPARSE_LENGTH="${SPARSE_LENGTH:-40}"
TEXT_SPARSE_LENGTH="${TEXT_SPARSE_LENGTH:-${SPARSE_LENGTH}}"
IMAGE_SPARSE_LENGTH="${IMAGE_SPARSE_LENGTH:-${SPARSE_LENGTH}}"
NUM_EXPENDED_TOKENS="${NUM_EXPENDED_TOKENS:-0}"
SPARSE_VALUE_TYPE="${SPARSE_VALUE_TYPE:-sum}"
SPARSE_LOWER_OR_UPPER="${SPARSE_LOWER_OR_UPPER:-lower}"
SPARSE_VALUE_MEAN="${SPARSE_VALUE_MEAN:-false}"
IS_FILTERED="${IS_FILTERED:-false}"
TBPR_TYPE="${TBPR_TYPE:-origin_type}"
CIR_TYPE="${CIR_TYPE:-type}"
SELECT_HYBRID_ALPHA="${SELECT_HYBRID_ALPHA:-false}"
HYBRID_ALPHA="${HYBRID_ALPHA:-0.5}"
VALIDATION_SPLIT="${VALIDATION_SPLIT:-val}"
RANKER_DEVICE_MAP="${RANKER_DEVICE_MAP:-auto}"
RANKER_DTYPE="${RANKER_DTYPE:-${DTYPE:-float16}}"
RANKER_QWEN_MIN_PIXELS="${RANKER_QWEN_MIN_PIXELS:-${QWEN_MIN_PIXELS:-0}}"
RANKER_QWEN_MAX_PIXELS="${RANKER_QWEN_MAX_PIXELS:-${QWEN_MAX_PIXELS:-0}}"
RERANK_NUM="${RERANK_NUM:-5}"
RERANK_BATCH_SIZE="${RERANK_BATCH_SIZE:-1}"
RERANK_TEMPLATE="${RERANK_TEMPLATE:-what_caption_generation}"
RERANK_REPS_LOC="${RERANK_REPS_LOC:-after_pad}"
MODIFY_TYPE="${MODIFY_TYPE:-no}"
USE_RERANK_PRIOR_CACHE="${USE_RERANK_PRIOR_CACHE:-false}"
OVERWRITE="${OVERWRITE:-false}"

# Optional per-family model overrides. Edit the array declarations below.
# Example: declare -A MODEL_PATHS=([qwen25]="/my/Qwen2.5-VL-7B-Instruct")
# Example: declare -A RANKER_MODEL_PATHS=([qwen3]="/my/Qwen3-VL-8B-Instruct")
declare -A MODEL_PATHS=()
declare -A RANKER_MODEL_PATHS=()
# Optional per-job artifact overrides. Keys are "family|dataset".
# Example: declare -A DENSE_INDEXES=(["qwen25|CUHK-PEDES"]="/my/dense/leaf")
# Example: declare -A SPARSE_INDEXES=(["qwen25|CUHK-PEDES"]="/my/sparse/leaf/collection")
declare -A DENSE_INDEXES=()
declare -A SPARSE_INDEXES=()
declare -A STAGE1_INDEXES=()
# Paired validation inputs; omit both to select on the target split.
# Example: declare -A VAL_DENSE_INDEXES=(["qwen25|CUHK-PEDES"]="/my/val/dense")
# Example: declare -A VAL_SPARSE_INDEXES=(["qwen25|CUHK-PEDES"]="/my/val/sparse/collection")
# Example: declare -A VALIDATION_DATASETS=(["qwen25|CUHK-PEDES"]="CUHK-PEDES")
declare -A VAL_DENSE_INDEXES=()
declare -A VAL_SPARSE_INDEXES=()
declare -A VALIDATION_DATASETS=()
# ---------------------------------------------------------------------------------------

bool() {
    case "${!1,,}" in
        true|1|yes) printf -v "$1" '%s' true ;;
        false|0|no) printf -v "$1" '%s' false ;;
        *) die "$1 must be true/false or 1/0." ;;
    esac
}
for name in SELECT_HYBRID_ALPHA SPARSE_VALUE_MEAN IS_FILTERED USE_RERANK_PRIOR_CACHE OVERWRITE; do bool "${name}"; done
[[ "${MODIFY_TYPE}" == no && "${USE_RERANK_PRIOR_CACHE}" == false ]] || die "TBPR does not support prior correction/cache; set MODIFY_TYPE=no and USE_RERANK_PRIOR_CACHE=false."
[[ "${EOL_TYPE:-disassembleeol_separate}" == disassembleeol_separate && "${CALCULATE_TYPE:-concat}" == concat ]] || die 'Stages 1/2 require disassembleeol_separate+concat.'
SPARSE_MANUAL="${SPARSE_MANUAL:-false}"
bool SPARSE_MANUAL
[[ "${SPARSE_MANUAL}" == false ]] || die 'Stages 1/2 require SPARSE_MANUAL=false.'
for name in GPUS NUM_NODES STAGE1_DEPTH STAGE2_DEPTH SPARSE_THREADS RETRIEVAL_BATCH_SIZE RERANK_NUM RERANK_BATCH_SIZE SPARSE_LENGTH TEXT_SPARSE_LENGTH IMAGE_SPARSE_LENGTH; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer."
done
for name in NUM_EXPENDED_TOKENS RETRIEVAL_QWEN_MIN_PIXELS RETRIEVAL_QWEN_MAX_PIXELS RANKER_QWEN_MIN_PIXELS RANKER_QWEN_MAX_PIXELS; do
    [[ "${!name}" =~ ^(0|[1-9][0-9]*)$ ]] || die "${name} must be a nonnegative integer."
done
for prefix in RETRIEVAL RANKER; do
    minimum="${prefix}_QWEN_MIN_PIXELS"; maximum="${prefix}_QWEN_MAX_PIXELS"
    if ((${!minimum} > 0 && ${!maximum} > 0 && ${!minimum} > ${!maximum})); then die "${minimum} must not exceed ${maximum}."; fi
    name="${prefix}_DTYPE"
    case "${!name}" in float16|bfloat16|float32) ;; *) die "Invalid ${name}." ;; esac
done
for name in RETRIEVAL_REPS_LOC RERANK_REPS_LOC; do
    [[ "${!name}" == after_pad || "${!name}" == before_pad ]] || die "Invalid ${name}."
done
[[ "${RANDOM_VIEW_SEED}" =~ ^-?[0-9]+$ ]] || die 'RANDOM_VIEW_SEED must be an integer.'
[[ "${HYBRID_ALPHA}" =~ ^(0|0?\.[0-9]+|1(\.0+)?)$ ]] || die 'HYBRID_ALPHA must be in [0,1].'
[[ "${TBPR_TYPE}" == origin_type || "${TBPR_TYPE}" == type ]] || die 'TBPR_TYPE must be origin_type or type.'
[[ "${SPARSE_VALUE_TYPE}" == sum || "${SPARSE_VALUE_TYPE}" == max || "${SPARSE_VALUE_TYPE}" == replace ]] || die 'Invalid SPARSE_VALUE_TYPE.'
[[ "${SPARSE_LOWER_OR_UPPER}" == lower || "${SPARSE_LOWER_OR_UPPER}" == upper ]] || die 'Invalid SPARSE_LOWER_OR_UPPER.'
[[ "${MODIFY_TYPE}" == no || "${MODIFY_TYPE}" == modify_single ]] || die 'Invalid MODIFY_TYPE.'
for name in DATASET_SPLIT PROMPT_TYPE RERANK_TEMPLATE CIR_TYPE; do
    [[ "${!name}" =~ ^[a-zA-Z0-9_-]+$ ]] || die "Invalid ${name}."
done
if [[ "${SELECT_HYBRID_ALPHA}" == true ]]; then
    [[ "${VALIDATION_SPLIT}" =~ ^[a-zA-Z0-9_-]+$ ]] || die 'Invalid VALIDATION_SPLIT.'
fi
if [[ -n "${QWEN_FAMILY}" ]]; then FAMILY_LIST=("${QWEN_FAMILY}"); else read -r -a FAMILY_LIST <<< "${QWEN_FAMILIES//$'\n'/ }"; fi
((${#FAMILY_LIST[@]})) || die 'QWEN_FAMILIES must contain at least one family.'
if [[ -n "${DATASET}" ]]; then DATASET_LIST=("${DATASET}"); else read -r -a DATASET_LIST <<< "${DATASETS//$'\n'/ }"; fi
((${#DATASET_LIST[@]})) || die 'DATASETS must contain at least one dataset.'
TOTAL_JOBS=$((${#FAMILY_LIST[@]} * ${#DATASET_LIST[@]}))
if ((${#FAMILY_LIST[@]} > 1)); then
    [[ -z "${REQUESTED_MODEL_PATH}${REQUESTED_RANKER_PATH}" ]] || die 'QWEN_MODEL_PATH/RANKER_MODEL_PATH require one family; use MODEL_PATHS/RANKER_MODEL_PATHS for multiple families.'
fi
if ((TOTAL_JOBS > 1)); then
    [[ -z "${OUTPUT_DIR}" ]] || die 'OUTPUT_DIR is only for a single job; use QWEN_OUTPUT_ROOT for batches.'
    [[ -z "${STAGE2_DENSE_INDEX:-}${STAGE2_SPARSE_INDEX:-}${STAGE1_SPARSE_INDEX:-}" ]] || die 'For batches, use input tables keyed by family|dataset instead of shared STAGE*_INDEX overrides.'
    if [[ "${SELECT_HYBRID_ALPHA}" == true ]]; then
        [[ -z "${VAL_PASSAGE_REPS:-}${VAL_SPARSE_INDEX:-}${VALIDATION_DATASET_NAME:-}" ]] || die 'For batches, use validation tables keyed by family|dataset instead of shared validation overrides.'
    fi
fi

checkpoint_for() {
    case "$1" in
        qwen2) printf Qwen2-VL-7B-Instruct ;;
        qwen25) printf Qwen2.5-VL-7B-Instruct ;;
        qwen3) printf Qwen3-VL-8B-Instruct ;;
        *) die "Unsupported Qwen family: $1; choose qwen2, qwen25 or qwen3." ;;
    esac
}
resolve_model() {
    JOB_FAMILY="$1"
    local checkpoint
    checkpoint="$(checkpoint_for "${JOB_FAMILY}")"
    # Resolve from the original overrides, never a previous job's paths.
    QWEN_MODEL_PATH="${REQUESTED_MODEL_PATH:-${MODEL_PATHS[${JOB_FAMILY}]:-${QWEN_CHECKPOINT_ROOT}/${checkpoint}}}"
    while [[ "${QWEN_MODEL_PATH}" == */ && "${QWEN_MODEL_PATH}" != / ]]; do QWEN_MODEL_PATH="${QWEN_MODEL_PATH%/}"; done
    MODEL_TAG="${QWEN_MODEL_PATH##*/}"
    [[ -n "${MODEL_TAG}" && "${MODEL_TAG}" != . && "${MODEL_TAG}" != .. ]] || die "Invalid model path for ${JOB_FAMILY}: ${QWEN_MODEL_PATH}"
    RANKER_MODEL_PATH="${REQUESTED_RANKER_PATH:-${RANKER_MODEL_PATHS[${JOB_FAMILY}]:-${QWEN_MODEL_PATH}}}"
    while [[ "${RANKER_MODEL_PATH}" == */ && "${RANKER_MODEL_PATH}" != / ]]; do RANKER_MODEL_PATH="${RANKER_MODEL_PATH%/}"; done
}
task_for() {
    case "$1" in
        CUHK-PEDES|ICFG-PEDES|RSTPReid) printf tbpr ;;
        *) die "Unsupported dataset: $1; choose CUHK-PEDES, ICFG-PEDES or RSTPReid." ;;
    esac
}
relative_path() {
    local split="$1" filter=no_filter mean=no_mean
    [[ "${IS_FILTERED}" == false ]] || filter=filter
    [[ "${SPARSE_VALUE_MEAN}" == false ]] || mean=mean
    local path="${MODEL_TAG}/${JOB_DATASET}/${TARGET_TYPE}/${filter}/concat/${PROMPT_TYPE}/${split}"
    [[ "${TASK_TYPE}" != tbpr ]] || path+="/${TBPR_TYPE}"
    printf '%s/%s_no_manual_%s_%s_no_cluster_%s_disassembleeol_separate_%s_%s' "${path}" "${NUM_EXPENDED_TOKENS}" "${SPARSE_LENGTH}" "${SPARSE_VALUE_TYPE}" "${RETRIEVAL_REPS_LOC}" "${SPARSE_LOWER_OR_UPPER}" "${mean}"
}
resolve_index() {
    local path="$1"
    while [[ "${path}" == */ && "${path}" != / ]]; do path="${path%/}"; done
    if [[ "${path##*/}" == index ]]; then printf '%s' "${path}"; return; fi
    [[ "${path##*/}" != collection ]] || path="$(dirname -- "${path}")"
    if [[ -e "${path}/collection/index" || -L "${path}/collection/index" ]]; then
        printf '%s/collection/index' "${path}"
    elif [[ -e "${path}/index" || -L "${path}/index" ]]; then
        printf '%s/index' "${path}"
    else
        printf '%s/collection/index' "${path}"
    fi
}
require_file() { [[ -f "$1" ]] || die "$2 file not found: $1"; }
require_dataset() {
    local dataset="$1" split="$2" directory file
    local -a files
    case "${dataset}" in
        CUHK-PEDES) directory="${dataset}"; files=(reid_raw.json) ;;
        ICFG-PEDES) directory="${dataset}"; files=(ICFG-PEDES.json) ;;
        RSTPReid) directory="${dataset}"; files=(data_captions.json) ;;
        *) die "Unsupported dataset: ${dataset}" ;;
    esac
    for file in "${files[@]}"; do require_file "${DATA_ROOT}/data/${directory}/${file}" "${dataset}/${split} annotation"; done
}
shopt -s nullglob
require_corpus() {
    local label="$1" dense="$2" sparse="$3" index
    local -a shards=("${dense}"/corpus*.pkl)
    if [[ -f "${dense}" ]]; then
        [[ "${dense##*/}" == corpus*.pkl ]] || die "${label}: expected a corpus*.pkl file: ${dense}"
    else
        ((${#shards[@]})) && [[ -f "${shards[0]}" ]] || die "${label}: no corpus*.pkl in ${dense}; run encode_qwen_tbpr.sh with matching settings."
    fi
    index="$(resolve_index "${sparse}")"
    local -a segments=("${index}"/segments_*)
    ((${#segments[@]})) && [[ -f "${segments[0]}" ]] || die "${label}: no Lucene segments_* in ${index}; run sparse_index_qwen.sh."
}
resolve_job() {
    JOB_DATASET="$1"
    JOB_KEY="${JOB_FAMILY}|${JOB_DATASET}"
    TASK_TYPE="$(task_for "${JOB_DATASET}")"
    QUERY_TYPE=text; TARGET_TYPE=image; DIRECTION=text_to_image
    RELATIVE="$(relative_path "${DATASET_SPLIT}")"
    JOB_DENSE="${DENSE_INDEXES[${JOB_KEY}]:-${STAGE2_DENSE_INDEX:-${DENSE_OUTPUT_ROOT}/${RELATIVE}}}"
    JOB_SPARSE="${SPARSE_INDEXES[${JOB_KEY}]:-${STAGE2_SPARSE_INDEX:-${SPARSE_OUTPUT_ROOT}/${RELATIVE}/collection}}"
    JOB_FIRST="${STAGE1_INDEXES[${JOB_KEY}]:-${STAGE1_SPARSE_INDEX:-${JOB_SPARSE}}}"
    ALPHA_ARGS=()
    ALPHA_LABEL="alpha_${HYBRID_ALPHA}"
    JOB_VAL_DENSE=''; JOB_VAL_SPARSE=''; JOB_VAL_DATASET=''
    if [[ "${SELECT_HYBRID_ALPHA}" == true ]]; then
        ALPHA_ARGS+=(--select-hybrid-alpha)
        ALPHA_LABEL="alpha_${DATASET_SPLIT}"
        if [[ -n "${VAL_DENSE_INDEXES[${JOB_KEY}]:-}${VAL_SPARSE_INDEXES[${JOB_KEY}]:-}" ]]; then
            JOB_VAL_DENSE="${VAL_DENSE_INDEXES[${JOB_KEY}]:-}"
            JOB_VAL_SPARSE="${VAL_SPARSE_INDEXES[${JOB_KEY}]:-}"
        else
            JOB_VAL_DENSE="${VAL_PASSAGE_REPS:-}"
            JOB_VAL_SPARSE="${VAL_SPARSE_INDEX:-}"
        fi
        JOB_VAL_DATASET="${VALIDATION_DATASETS[${JOB_KEY}]:-${VALIDATION_DATASET_NAME:-}}"
        if [[ -n "${JOB_VAL_DENSE}${JOB_VAL_SPARSE}${JOB_VAL_DATASET}" ]]; then
            [[ -n "${JOB_VAL_DENSE}" && -n "${JOB_VAL_SPARSE}" ]] || die "Validation dense/sparse paths must both be provided for ${JOB_KEY}."
            JOB_VAL_DATASET="${JOB_VAL_DATASET:-${JOB_DATASET}}"
            [[ "$(task_for "${JOB_VAL_DATASET}")" == "${TASK_TYPE}" ]] || die "Validation task mismatch for ${JOB_DATASET}: ${JOB_VAL_DATASET}."
            ALPHA_ARGS+=(--val-passage-reps "${JOB_VAL_DENSE}" --val-sparse-index "${JOB_VAL_SPARSE}" --validation-dataset-name "${JOB_VAL_DATASET}" --validation-split "${VALIDATION_SPLIT}")
            ALPHA_LABEL=alpha_val
        fi
    fi
    JOB_OUTPUT="${OUTPUT_DIR:-${QWEN_OUTPUT_ROOT}/${RELATIVE}/${DIRECTION}/three_stage/${ALPHA_LABEL}/stage1_${STAGE1_DEPTH}_stage2_${STAGE2_DEPTH}}"
}
preflight_job() {
    dense_mode_paths
    require_dataset "${JOB_DATASET}" "${DATASET_SPLIT}"
    require_corpus "${JOB_KEY} Stage 2" "${JOB_DENSE}" "${JOB_SPARSE}"
    local first="$(resolve_index "${JOB_FIRST}")"
    local -a segments=("${first}"/segments_*)
    ((${#segments[@]})) && [[ -f "${segments[0]}" ]] || die "${JOB_KEY} Stage 1: no Lucene segments_* in ${first}; run sparse_index_qwen.sh."
    if [[ -n "${JOB_VAL_DENSE}" ]]; then
        require_dataset "${JOB_VAL_DATASET}" "${VALIDATION_SPLIT}"
        require_corpus "${JOB_KEY} Validation" "${JOB_VAL_DENSE}" "${JOB_VAL_SPARSE}"
    fi
    [[ ! -e "${JOB_OUTPUT}" && ! -L "${JOB_OUTPUT}" || "${OVERWRITE}" == true ]] || die "${JOB_KEY}: output already exists: ${JOB_OUTPUT}; set OVERWRITE=true or choose a new QWEN_OUTPUT_ROOT."
}
build_command() {
    dense_mode_paths
    COMMAND=("${DEEPSPEED_BIN}" --num_gpus "${GPUS}" --num_nodes "${NUM_NODES}" src/three_stage_retrieval_pipeline.py --use_freeret_dense "${USE_FREERET_DENSE}" --use_freeret_prompt_constraints "${USE_FREERET_PROMPT_CONSTRAINTS}" --use_freeret_full_prompt "${USE_FREERET_FULL_PROMPT}"
        --dataset-name "${JOB_DATASET}" --dataset-split "${DATASET_SPLIT}" --task-type "${TASK_TYPE}" --query-type "${QUERY_TYPE}"
        --tbpr-type "${TBPR_TYPE}" --cir-type "${CIR_TYPE}" --data-root "${DATA_ROOT}" --stage1-method sparse_concat
        --stage1-sparse-index "${JOB_FIRST}" --stage1-depth "${STAGE1_DEPTH}"
        --stage2-dense-index "${JOB_DENSE}" --stage2-sparse-index "${JOB_SPARSE}" --stage2-depth "${STAGE2_DEPTH}"
        --hybrid-alpha "${HYBRID_ALPHA}" "${ALPHA_ARGS[@]}" --sparse-threads "${SPARSE_THREADS}"
        --retrieval-model-name-or-path "${QWEN_MODEL_PATH}" --retrieval-device "${RETRIEVAL_DEVICE}" --retrieval-device-map "${RETRIEVAL_DEVICE_MAP}"
        --retrieval-dtype "${RETRIEVAL_DTYPE}" --retrieval-batch-size "${RETRIEVAL_BATCH_SIZE}" --retrieval-reps-loc "${RETRIEVAL_REPS_LOC}"
        --retrieval-qwen-min-pixels "${RETRIEVAL_QWEN_MIN_PIXELS}" --retrieval-qwen-max-pixels "${RETRIEVAL_QWEN_MAX_PIXELS}"
        --prompt-type "${PROMPT_TYPE}" --random-view-seed "${RANDOM_VIEW_SEED}" --sparse-length "${SPARSE_LENGTH}"
        --text-sparse-length "${TEXT_SPARSE_LENGTH}" --image-sparse-length "${IMAGE_SPARSE_LENGTH}" --num-expended-tokens "${NUM_EXPENDED_TOKENS}"
        --sparse-value-type "${SPARSE_VALUE_TYPE}" --sparse-lower-or-upper "${SPARSE_LOWER_OR_UPPER}"
        --ranker-model-name-or-path "${RANKER_MODEL_PATH}" --ranker-device-map "${RANKER_DEVICE_MAP}" --ranker-dtype "${RANKER_DTYPE}"
        --ranker-qwen-min-pixels "${RANKER_QWEN_MIN_PIXELS}" --ranker-qwen-max-pixels "${RANKER_QWEN_MAX_PIXELS}"
        --rerank-type pointwise --rerank-template "${RERANK_TEMPLATE}" --rerank-num "${RERANK_NUM}" --rerank-batch-size "${RERANK_BATCH_SIZE}"
        --rerank-reps-loc "${RERANK_REPS_LOC}" --modify-type "${MODIFY_TYPE}" --output-dir "${JOB_OUTPUT}")
    [[ "${IS_FILTERED}" == false ]] || COMMAND+=(--is-filtered)
    [[ "${SPARSE_VALUE_MEAN}" == false ]] || COMMAND+=(--sparse-value-mean)
    [[ "${OVERWRITE}" == false ]] || COMMAND+=(--overwrite)
}

command -v "${DEEPSPEED_BIN}" >/dev/null 2>&1 || die "DeepSpeed executable not found: ${DEEPSPEED_BIN}"
require_file src/three_stage_retrieval_pipeline.py 'Python entrypoint'
declare -A SEEN_FAMILIES=() SEEN_DATASETS=() SEEN_MODEL_TAGS=() SEEN_OUTPUTS=()
for family in "${FAMILY_LIST[@]}"; do
    checkpoint_for "${family}" >/dev/null
    [[ -z "${SEEN_FAMILIES[${family}]+present}" ]] || die "Duplicate Qwen family: ${family}"
    SEEN_FAMILIES["${family}"]=1
done
for dataset in "${DATASET_LIST[@]}"; do
    task_for "${dataset}" >/dev/null
    [[ -z "${SEEN_DATASETS[${dataset}]+present}" ]] || die "Duplicate dataset: ${dataset}"
    SEEN_DATASETS["${dataset}"]=1
done
# Preflight the entire model x dataset matrix before launching any GPU process.
for family in "${FAMILY_LIST[@]}"; do
    resolve_model "${family}"
    [[ -z "${SEEN_MODEL_TAGS[${MODEL_TAG}]+present}" ]] || die "Duplicate model directory name: ${MODEL_TAG}; each family needs a distinct name for artifacts/results."
    SEEN_MODEL_TAGS["${MODEL_TAG}"]="${family}"
    require_file "${QWEN_MODEL_PATH}/config.json" "${family} retrieval checkpoint config"
    require_file "${RANKER_MODEL_PATH}/config.json" "${family} ranker checkpoint config"
    for dataset in "${DATASET_LIST[@]}"; do
        resolve_job "${dataset}"
        [[ -z "${SEEN_OUTPUTS[${JOB_OUTPUT}]+present}" ]] || die "Output collision for ${JOB_KEY} and ${SEEN_OUTPUTS[${JOB_OUTPUT}]}: ${JOB_OUTPUT}"
        SEEN_OUTPUTS["${JOB_OUTPUT}"]="${JOB_KEY}"
        preflight_job
    done
done
printf 'Qwen TBPR batch: families=%s datasets=%s jobs=%s GPUs=%s\n' "${FAMILY_LIST[*]}" "${DATASET_LIST[*]}" "${TOTAL_JOBS}" "${GPUS}"
completed=0
for family in "${FAMILY_LIST[@]}"; do
    resolve_model "${family}"
    for dataset in "${DATASET_LIST[@]}"; do
        resolve_job "${dataset}"
        build_command
        printf '\nJob [%s/%s]: %s (%s, %s)\nRetrieval model: %s\nRanker model: %s\nDense corpus: %s\n' "$((completed + 1))" "${TOTAL_JOBS}" "${JOB_KEY}" "${DATASET_SPLIT}" "${DIRECTION}" "${QWEN_MODEL_PATH}" "${RANKER_MODEL_PATH}" "${JOB_DENSE}"
        printf 'Stage-1 Lucene index: %s\nStage-2 Lucene index: %s\n' "$(resolve_index "${JOB_FIRST}")" "$(resolve_index "${JOB_SPARSE}")"
        if [[ "${SELECT_HYBRID_ALPHA}" == false ]]; then
            printf 'Alpha: fixed %s\n' "${HYBRID_ALPHA}"
        elif [[ -n "${JOB_VAL_DENSE}" ]]; then
            printf 'Alpha: select 0.1..0.9 on %s/%s\nValidation dense: %s\nValidation Lucene index: %s\n' "${JOB_VAL_DATASET}" "${VALIDATION_SPLIT}" "${JOB_VAL_DENSE}" "$(resolve_index "${JOB_VAL_SPARSE}")"
        else
            printf 'Skipping validation: no val paths specified.\nAlpha: select 0.1..0.9 on target %s/%s\n' "${dataset}" "${DATASET_SPLIT}"
        fi
        printf 'Results: %s\nCommand:' "${JOB_OUTPUT}"; printf ' %q' "${COMMAND[@]}"; printf '\n'
        if "${COMMAND[@]}"; then
            completed=$((completed + 1))
            printf 'Completed [%s/%s]: %s\n' "${completed}" "${TOTAL_JOBS}" "${JOB_KEY}"
        else
            status=$?
            printf 'Failed [%s/%s]: %s (exit %s); completed %s jobs, stopping Qwen batch.\n' "$((completed + 1))" "${TOTAL_JOBS}" "${JOB_KEY}" "${status}" "${completed}" >&2
            exit "${status}"
        fi
    done
done
printf '\nCompleted all %s model/dataset jobs.\n' "${TOTAL_JOBS}"
