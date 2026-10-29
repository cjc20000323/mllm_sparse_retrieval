#!/bin/bash
# Standalone Qwen encoding, configured like encode.sh. One dataset per run.
# Examples:
#   bash scripts/encode_qwen.sh
#   QWEN_FAMILY=qwen3 DATASET=coco bash scripts/encode_qwen.sh
#   DATASET=flickr ENCODE_TYPE=image bash scripts/encode_qwen.sh
#   DATASET=CUHK-PEDES TASK_TYPE=tbpr ENCODE_TYPE=image bash scripts/encode_qwen.sh
# Afterwards, use DATASETS=flickr bash scripts/sparse_index_qwen.sh for this corpus.
set -euo pipefail

QWEN_FAMILY="${QWEN_FAMILY:-qwen25}"
QWEN_CHECKPOINT_ROOT="${QWEN_CHECKPOINT_ROOT:-/root/autodl-fs/checkpoints}"
GPUS="${GPUS:-1}"
NUM_NODES="${NUM_NODES:-1}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"
# DATASET defaults to flickr after the single-value DATASETS compatibility check.
DATASET="${DATASET:-}"
TASK_TYPE="${TASK_TYPE:-ir}"
ENCODE_TYPE="${ENCODE_TYPE:-text}"
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
Usage: bash scripts/encode_qwen.sh [-h|--help]
Edit the configuration block above or set environment variables.
Defaults: Qwen2.5-VL-7B-Instruct, DATASET=flickr, TASK_TYPE=ir, ENCODE_TYPE=text,
DATASET_SPLIT=test, GPUS=1, disassembleeol_separate+concat, float16.
QWEN_FAMILY=qwen2|qwen25|qwen3; QWEN_MODEL_PATH overrides the local checkpoint.
EOL_TYPE=prompteol selects separate. DTYPE=float16|bfloat16|float32.
DENSE_OUTPUT_ROOT/SPARSE_OUTPUT_ROOT default to project dense_output/sparse_output.
OUTPUT_DIR defaults to ./output (runtime output, not corpus representations).
USE_GENERATED_VALIDATION_DATA=true selects val and GENERATED_VALIDATION_DATA_ROOT.
Single-value DATASETS is accepted when DATASET is unset; multiple values are rejected.
ENCODE_TYPE explicitly selects the corpus modality; IR_QUERY_TYPE is not read.
No other CLI arguments are forwarded. Build Lucene separately after encoding.
USAGE
        exit 0
    fi
    die "Configure variables in this script or the environment. See --help."
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
DENSE_OUTPUT_ROOT="${DENSE_OUTPUT_ROOT:-${REPO_ROOT}/dense_output}"
SPARSE_OUTPUT_ROOT="${SPARSE_OUTPUT_ROOT:-${REPO_ROOT}/sparse_output}"

if [[ -z "${DATASET}" ]]; then
    if [[ -v DATASETS ]]; then
        read -r -a DATASET_LIST <<< "${DATASETS//$'\n'/ }"
        ((${#DATASET_LIST[@]} == 1)) || die "Set one DATASET; run encode_qwen.sh separately for each dataset."
        DATASET="${DATASET_LIST[0]}"
    else
        DATASET=flickr
    fi
fi
[[ "${DATASET}" != *[[:space:]]* ]] || die "DATASET must name one dataset."

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
        true|1|yes) SPARSE_ARGS+=("--${name,,}") ;;
        false|0|no) ;;
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
command -v "${DEEPSPEED_BIN}" >/dev/null 2>&1 || die "DeepSpeed executable not found: ${DEEPSPEED_BIN}"
[[ -f "${QWEN_MODEL_PATH}/config.json" ]] || die "Checkpoint config.json not found: ${QWEN_MODEL_PATH}"

"${DEEPSPEED_BIN}" --num_gpus="${GPUS}" --num_nodes="${NUM_NODES}" src/encode.py \
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
    "${GENERATED_VALIDATION_ARGS[@]}"
