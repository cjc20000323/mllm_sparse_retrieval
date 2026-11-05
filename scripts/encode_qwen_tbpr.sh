#!/usr/bin/env bash
USE_FREERET_DENSE="${USE_FREERET_DENSE-false}"  # true: FreeRet pre-MLP dense; prompts/sparse unchanged
USE_FREERET_FULL_PROMPT="${USE_FREERET_FULL_PROMPT-false}"  # Complete FreeRet dense/sparse prompts; requires FreeRet dense
USE_FREERET_PROMPT_CONSTRAINTS="${USE_FREERET_PROMPT_CONSTRAINTS-false}"  # Prepend FreeRet constraints to all representation views
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/freeret_dense_mode.sh" || exit $?
# Standalone Qwen TBPR candidate encoding.
# Directly runs src/encode.py; queries are encoded online during retrieval.
# Examples (one model per invocation, datasets run sequentially):
#   bash scripts/encode_qwen_tbpr.sh
#   QWEN_FAMILY=qwen3 DATASETS="CUHK-PEDES ICFG-PEDES RSTPReid" bash scripts/encode_qwen_tbpr.sh
#   DATASET=CUHK-PEDES DATASET_SPLIT=val bash scripts/encode_qwen_tbpr.sh
#   EOL_TYPE=prompteol bash scripts/encode_qwen_tbpr.sh
#   DENSE_OUTPUT_ROOT=/my/dense SPARSE_OUTPUT_ROOT=/my/sparse bash scripts/encode_qwen_tbpr.sh
set -euo pipefail

QWEN_FAMILY="${QWEN_FAMILY:-qwen25}"
QWEN_CHECKPOINT_ROOT="${QWEN_CHECKPOINT_ROOT:-/root/autodl-fs/checkpoints}"
GPUS="${GPUS:-1}"
NUM_NODES="${NUM_NODES:-1}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"
DATASET="${DATASET:-}"
DATASETS="${DATASETS-CUHK-PEDES}"
REQUESTED_TASK_TYPE="${TASK_TYPE:-}"
ENCODE_TYPE="${ENCODE_TYPE:-image}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
PROMPT_TYPE="${PROMPT_TYPE:-prompt_5}"
RANDOM_VIEW_SEED="${RANDOM_VIEW_SEED:-42}"
EOL_TYPE="${EOL_TYPE:-disassembleeol_separate}"
REPS_LOC="${REPS_LOC:-after_pad}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
DTYPE="${DTYPE:-float16}"
QWEN_MIN_PIXELS="${QWEN_MIN_PIXELS:-0}"
QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS:-0}"
SPARSE_LENGTH="${SPARSE_LENGTH:-40}"
TEXT_SPARSE_LENGTH="${TEXT_SPARSE_LENGTH:-${SPARSE_LENGTH}}"
IMAGE_SPARSE_LENGTH="${IMAGE_SPARSE_LENGTH:-${SPARSE_LENGTH}}"
SPARSE_VALUE_TYPE="${SPARSE_VALUE_TYPE:-sum}"
SPARSE_LOWER_OR_UPPER="${SPARSE_LOWER_OR_UPPER:-lower}"
NUM_EXPENDED_TOKENS="${NUM_EXPENDED_TOKENS:-0}"
SPARSE_MANUAL="${SPARSE_MANUAL:-false}"
SPARSE_VALUE_MEAN="${SPARSE_VALUE_MEAN:-false}"
IS_FILTERED="${IS_FILTERED:-false}"
TBPR_TYPE="${TBPR_TYPE:-origin_type}"
CIR_TYPE="${CIR_TYPE:-type}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
USE_GENERATED_VALIDATION_DATA="${USE_GENERATED_VALIDATION_DATA:-false}"
GENERATED_VALIDATION_DATA_ROOT="${GENERATED_VALIDATION_DATA_ROOT:-./generated_llava70b_data}"

die() { printf 'Error: %s\n' "$*" >&2; exit 2; }
if (($#)); then
    if (($# == 1)) && [[ "$1" == -h || "$1" == --help ]]; then
        cat <<'USAGE'
Usage: bash scripts/encode_qwen_tbpr.sh [-h|--help]
Edit the configuration block or set environment variables.
Supported datasets: CUHK-PEDES ICFG-PEDES RSTPReid. Default dataset: CUHK-PEDES.
TBPR candidates: person images; TASK_TYPE=tbpr and ENCODE_TYPE=image.
DATASET selects one job; otherwise DATASETS is a whitespace-separated list.
Datasets run sequentially; all names/configuration are checked before launching,
and the first failing encoding job stops the batch. No query files are generated.
QWEN_FAMILY=qwen2|qwen25|qwen3 (default: qwen25); QWEN_MODEL_PATH overrides
the local checkpoint under QWEN_CHECKPOINT_ROOT (default: /root/autodl-fs/checkpoints).
Defaults: test, 1 GPU, float16, prompt_5, sparse length 40, non-manual sparse,
disassembleeol_separate+concat. EOL_TYPE=prompteol selects separate.
TBPR_TYPE=origin_type|type (default: origin_type) and CIR_TYPE=type are forwarded
unchanged for encoding fingerprints. TBPR_TYPE must match the retrieval script.
DTYPE=float16|bfloat16|float32. GPUS, NUM_NODES, DEEPSPEED_BIN,
PER_DEVICE_BATCH_SIZE and all existing prompt/sparse/pixel settings are configurable.
QWEN_MIN_PIXELS=0 and QWEN_MAX_PIXELS=0 use the checkpoint's image defaults.
DENSE_OUTPUT_ROOT/SPARSE_OUTPUT_ROOT default to project dense_output/sparse_output.
The native encoding directory and sparse collection/ path are printed per dataset.
OUTPUT_DIR defaults to ./output (runtime output, not corpus representations).
DATASET_SPLIT selects the split. USE_GENERATED_VALIDATION_DATA=true forces val
and forwards GENERATED_VALIDATION_DATA_ROOT (default: ./generated_llava70b_data).
Only -h/--help is accepted; other CLI arguments are rejected, not forwarded.
Build Lucene separately with sparse_index_qwen.sh after encoding.
Three-stage retrieval uses disassembleeol_separate+concat and SPARSE_MANUAL=false;
keep prompt, sparse, seed, TBPR and retrieval pixel settings identical across steps.

Examples:
  bash scripts/encode_qwen_tbpr.sh
  QWEN_FAMILY=qwen3 DATASETS="CUHK-PEDES ICFG-PEDES RSTPReid" bash scripts/encode_qwen_tbpr.sh
  DATASET=CUHK-PEDES DATASET_SPLIT=val bash scripts/encode_qwen_tbpr.sh
  EOL_TYPE=prompteol bash scripts/encode_qwen_tbpr.sh
  DENSE_OUTPUT_ROOT=/my/dense SPARSE_OUTPUT_ROOT=/my/sparse bash scripts/encode_qwen_tbpr.sh
USAGE
        exit 0
    fi
    die "Configure variables in this script or the environment. See --help."
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
DENSE_OUTPUT_ROOT="${DENSE_OUTPUT_ROOT:-${REPO_ROOT}/dense_output}"
SPARSE_OUTPUT_ROOT="${SPARSE_OUTPUT_ROOT:-${REPO_ROOT}/sparse_output}"

task_for() {
    case "$1" in
        CUHK-PEDES|ICFG-PEDES|RSTPReid) printf tbpr ;;
        *) die "Unsupported dataset: $1; choose CUHK-PEDES ICFG-PEDES RSTPReid." ;;
    esac
}
if [[ -n "${DATASET}" ]]; then
    DATASET_LIST=("${DATASET}")
else
    read -r -a DATASET_LIST <<< "${DATASETS//$'\n'/ }"
fi
((${#DATASET_LIST[@]})) || die 'DATASETS must contain at least one dataset.'
[[ "${ENCODE_TYPE}" == image ]] || die 'This launcher encodes candidates; ENCODE_TYPE must be image.'
declare -A SEEN=()
for dataset in "${DATASET_LIST[@]}"; do
    expected_task="$(task_for "${dataset}")"
    [[ -z "${REQUESTED_TASK_TYPE}" || "${REQUESTED_TASK_TYPE}" == "${expected_task}" ]] || die "${dataset} requires TASK_TYPE=${expected_task}."
    [[ -z "${SEEN[${dataset}]+present}" ]] || die "Duplicate dataset: ${dataset}"
    SEEN["${dataset}"]=1
done

case "${QWEN_FAMILY}" in
    qwen2) CHECKPOINT=Qwen2-VL-7B-Instruct ;;
    qwen25) CHECKPOINT=Qwen2.5-VL-7B-Instruct ;;
    qwen3) CHECKPOINT=Qwen3-VL-8B-Instruct ;;
    *) die "QWEN_FAMILY must be qwen2, qwen25 or qwen3." ;;
esac
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${QWEN_CHECKPOINT_ROOT}/${CHECKPOINT}}"
while [[ "${QWEN_MODEL_PATH}" == */ && "${QWEN_MODEL_PATH}" != / ]]; do
    QWEN_MODEL_PATH="${QWEN_MODEL_PATH%/}"
done
MODEL_TAG="${QWEN_MODEL_PATH##*/}"
[[ -n "${MODEL_TAG}" && "${MODEL_TAG}" != . && "${MODEL_TAG}" != .. ]] || die "Invalid QWEN_MODEL_PATH."
case "${EOL_TYPE}" in
    prompteol) EXPECTED_CALCULATE_TYPE=separate ;;
    disassembleeol_separate) EXPECTED_CALCULATE_TYPE=concat ;;
    *) die "EOL_TYPE must be prompteol or disassembleeol_separate." ;;
esac
CALCULATE_TYPE="${CALCULATE_TYPE:-${EXPECTED_CALCULATE_TYPE}}"
[[ "${CALCULATE_TYPE}" == "${EXPECTED_CALCULATE_TYPE}" ]] || die "${EOL_TYPE} requires CALCULATE_TYPE=${EXPECTED_CALCULATE_TYPE}."

PRECISION_ARGS=()
case "${DTYPE}" in
    float16) PRECISION_ARGS+=(--fp16) ;;
    bfloat16) PRECISION_ARGS+=(--bf16) ;;
    float32) ;;
    *) die "DTYPE must be float16, bfloat16 or float32." ;;
esac
SPARSE_ARGS=()
for name in SPARSE_MANUAL SPARSE_VALUE_MEAN IS_FILTERED; do
    case "${!name,,}" in
        true|1|yes) printf -v "${name}" '%s' true; SPARSE_ARGS+=("--${name,,}") ;;
        false|0|no) printf -v "${name}" '%s' false ;;
        *) die "${name} must be true/false or 1/0." ;;
    esac
done
GENERATED_VALIDATION_ARGS=()
case "${USE_GENERATED_VALIDATION_DATA,,}" in
    true|1|yes)
        DATASET_SPLIT=val
        GENERATED_VALIDATION_ARGS+=(--use_generated_validation_data
            --generated_validation_data_root "${GENERATED_VALIDATION_DATA_ROOT}")
        ;;
    false|0|no) ;;
    *) die "USE_GENERATED_VALIDATION_DATA must be true/false or 1/0." ;;
esac
for name in GPUS NUM_NODES PER_DEVICE_BATCH_SIZE SPARSE_LENGTH TEXT_SPARSE_LENGTH IMAGE_SPARSE_LENGTH; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer."
done
for name in QWEN_MIN_PIXELS QWEN_MAX_PIXELS NUM_EXPENDED_TOKENS; do
    [[ "${!name}" =~ ^(0|[1-9][0-9]*)$ ]] || die "${name} must be nonnegative."
done
if ((QWEN_MIN_PIXELS > 0 && QWEN_MAX_PIXELS > 0 && QWEN_MIN_PIXELS > QWEN_MAX_PIXELS)); then
    die "QWEN_MIN_PIXELS must not exceed QWEN_MAX_PIXELS."
fi
[[ "${DATASET_SPLIT}" =~ ^[a-zA-Z0-9_-]+$ ]] || die 'Invalid DATASET_SPLIT.'
[[ "${PROMPT_TYPE}" =~ ^[a-zA-Z0-9_]+$ ]] || die 'Invalid PROMPT_TYPE.'
[[ "${RANDOM_VIEW_SEED}" =~ ^-?[0-9]+$ ]] || die 'RANDOM_VIEW_SEED must be an integer.'
[[ "${REPS_LOC}" == after_pad || "${REPS_LOC}" == before_pad ]] || die 'Invalid REPS_LOC.'
[[ "${TBPR_TYPE}" == origin_type || "${TBPR_TYPE}" == type ]] || die 'TBPR_TYPE must be origin_type or type.'
[[ "${CIR_TYPE}" =~ ^[a-zA-Z0-9_]+$ ]] || die 'Invalid CIR_TYPE.'
case "${SPARSE_VALUE_TYPE}" in sum|max|replace) ;; *) die 'Invalid SPARSE_VALUE_TYPE.' ;; esac
case "${SPARSE_LOWER_OR_UPPER}" in lower|upper) ;; *) die 'Invalid SPARSE_LOWER_OR_UPPER.' ;; esac
[[ -f src/encode.py ]] || die 'Python entrypoint not found: src/encode.py'
command -v "${DEEPSPEED_BIN}" >/dev/null 2>&1 || die "DeepSpeed executable not found: ${DEEPSPEED_BIN}"
[[ -f "${QWEN_MODEL_PATH}/config.json" ]] || die "Checkpoint config.json not found: ${QWEN_MODEL_PATH}"

# Same directory contract as qwen_vl_pipeline.artifact_dirs().
relative_path() {
    local filter=no_filter manual=no_manual lengths="${SPARSE_LENGTH}" mean=no_mean relative
    [[ "${IS_FILTERED}" == false ]] || filter=filter
    [[ "${SPARSE_VALUE_MEAN}" == false ]] || mean=mean
    if [[ "${SPARSE_MANUAL}" == true ]]; then manual=manual; lengths="${TEXT_SPARSE_LENGTH}_${IMAGE_SPARSE_LENGTH}"; fi
    relative="${MODEL_TAG}/${DATASET}/${ENCODE_TYPE}/${filter}/${CALCULATE_TYPE}/${PROMPT_TYPE}/${DATASET_SPLIT}"
    relative+="/${TBPR_TYPE}"
    printf '%s/%s_%s_%s_%s_no_cluster_%s_%s_%s_%s' "${relative}" "${NUM_EXPENDED_TOKENS}" "${manual}" "${lengths}" "${SPARSE_VALUE_TYPE}" "${REPS_LOC}" "${EOL_TYPE}" "${SPARSE_LOWER_OR_UPPER}" "${mean}"
}
printf 'Qwen candidate encoding: model=%s datasets=%s method=%s+%s\n' "${QWEN_MODEL_PATH}" "${DATASET_LIST[*]}" "${EOL_TYPE}" "${CALCULATE_TYPE}"
for DATASET in "${DATASET_LIST[@]}"; do
    TASK_TYPE="$(task_for "${DATASET}")"
    RELATIVE="$(relative_path)"
    printf '\nDataset: %s (%s, %s, candidates=%s)\n' "${DATASET}" "${DATASET_SPLIT}" "${TASK_TYPE}" "${ENCODE_TYPE}"
    printf 'Dense corpus: %s/%s\nSparse collection: %s/%s/collection\n' "${DENSE_OUTPUT_ROOT%/}" "${RELATIVE}" "${SPARSE_OUTPUT_ROOT%/}" "${RELATIVE}"
    printf 'Runtime output: %s\n' "${OUTPUT_DIR}"
    if "${DEEPSPEED_BIN}" --num_gpus="${GPUS}" --num_nodes="${NUM_NODES}" src/encode.py --use_freeret_dense "${USE_FREERET_DENSE}" --use_freeret_prompt_constraints "${USE_FREERET_PROMPT_CONSTRAINTS}" --use_freeret_full_prompt "${USE_FREERET_FULL_PROMPT}" \
        --model_name_or_path "${QWEN_MODEL_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --dense_output_dir "${DENSE_OUTPUT_ROOT}" \
        --sparse_output_dir "${SPARSE_OUTPUT_ROOT}" \
        "${PRECISION_ARGS[@]}" \
        --dataset_name "${DATASET}" \
        --task_type "${TASK_TYPE}" \
        --encode_type "${ENCODE_TYPE}" \
        --per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
        --dataset_split "${DATASET_SPLIT}" \
        --reps_loc "${REPS_LOC}" \
        --num_expended_tokens "${NUM_EXPENDED_TOKENS}" \
        --eol_type "${EOL_TYPE}" \
        --sparse_value_type "${SPARSE_VALUE_TYPE}" \
        --sparse_lower_or_upper "${SPARSE_LOWER_OR_UPPER}" \
        --calculate_type "${CALCULATE_TYPE}" \
        --prompt_type "${PROMPT_TYPE}" \
        --random_view_seed "${RANDOM_VIEW_SEED}" \
        --sparse_length "${SPARSE_LENGTH}" \
        --text_sparse_length "${TEXT_SPARSE_LENGTH}" \
        --image_sparse_length "${IMAGE_SPARSE_LENGTH}" \
        --qwen_min_pixels "${QWEN_MIN_PIXELS}" \
        --qwen_max_pixels "${QWEN_MAX_PIXELS}" \
        --tbpr_type "${TBPR_TYPE}" \
        --cir_type "${CIR_TYPE}" \
        "${SPARSE_ARGS[@]}" \
        "${GENERATED_VALIDATION_ARGS[@]}"; then
        printf 'Completed encoding: %s\n' "${DATASET}"
    else
        status=$?
        printf 'Failed encoding: %s (exit %s); stopping batch.\n' "${DATASET}" "${status}" >&2
        exit "${status}"
    fi
done
printf '\nCompleted all %s dataset encoding jobs. Build Lucene indexes separately.\n' "${#DATASET_LIST[@]}"
