#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Run the independent search_similarity.py entry point.
#
# Usage:
#   bash scripts/search_similarity.sh [both|no_manual|manual]
#
# Legacy prompteol/separate defaults to both settings (paired report).
# disassembleeol_separate/concat defaults to no_manual (single report):
#   EOL_TYPE=disassembleeol_separate CALCULATE_TYPE=concat \
#     bash scripts/search_similarity.sh
# Original-text sparse tokens use the same single-report default:
#   EOL_TYPE=disassembleeol_separate_origin_text CALCULATE_TYPE=concat \
#     SPARSE_TYPE=single EXPENDED_TOKENS=0 bash scripts/search_similarity.sh
# Encode targets and build the sparse index with the same settings first;
# this launcher does not create indices or reuse the old text sparse index.

# The Python entry point reads this environment variable.  Set it to false
# explicitly when only the original search behavior is wanted.
export ENABLE_RETRIEVAL_SIMILARITY_STATS="${ENABLE_RETRIEVAL_SIMILARITY_STATS:-true}"

GPUS="${GPUS:-2}"
NUM_NODES="${NUM_NODES:-1}"
MASTER_PORT="${MASTER_PORT:-29501}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"

BASE_MODEL="${BASE_MODEL:-llava-hf-llava-v1.6-mistral-7b-hf}"
MODEL_PATH="${MODEL_PATH:-./checkpoints/${BASE_MODEL}}"
DATASET="${DATASET:-flickr}"
TARGET_TYPE="${TARGET_TYPE:-text}"
QUERY_TYPE="${QUERY_TYPE:-image}"
FILTER="${FILTER:-no_filter}"
PROMPT_TYPE="${PROMPT_TYPE:-prompt_5}"
VAL_SPLIT="${VAL_SPLIT:-val}"
TEST_SPLIT="${TEST_SPLIT:-test}"

# These must match the already encoded target representations and index.
CALCULATE_TYPE="${CALCULATE_TYPE:-separate}"
EOL_TYPE="${EOL_TYPE:-prompteol}"
case "${EOL_TYPE}/${CALCULATE_TYPE}" in
  prompteol/separate) DEFAULT_RUN_MODE="both" ;;
  disassembleeol_separate/concat) DEFAULT_RUN_MODE="no_manual" ;;
  disassembleeol_separate_origin_text/concat) DEFAULT_RUN_MODE="no_manual" ;;
  *)
    echo "Unsupported EOL_TYPE/CALCULATE_TYPE=${EOL_TYPE}/${CALCULATE_TYPE}; use prompteol/separate, disassembleeol_separate/concat, or disassembleeol_separate_origin_text/concat." >&2
    exit 1
    ;;
esac
RUN_MODE="${1:-${DEFAULT_RUN_MODE}}"
case "${RUN_MODE}" in
  both|no_manual|manual) ;;
  *)
    echo "Usage: bash scripts/search_similarity.sh [both|no_manual|manual]" >&2
    exit 1
    ;;
esac
EXPENDED_TOKENS="${EXPENDED_TOKENS:-0}"
MANUAL_LENGTH="${MANUAL_LENGTH:-128}"
SPARSE_VALUE_TYPE="${SPARSE_VALUE_TYPE:-sum}"
SPARSE_TYPE="${SPARSE_TYPE:-single}"
SPARSE_LOWER_OR_UPPER="${SPARSE_LOWER_OR_UPPER:-lower}"
SPARSE_LENGTH="${SPARSE_LENGTH:-${MANUAL_LENGTH}}"
TEXT_SPARSE_LENGTH="${TEXT_SPARSE_LENGTH:-${SPARSE_LENGTH}}"
IMAGE_SPARSE_LENGTH="${IMAGE_SPARSE_LENGTH:-${SPARSE_LENGTH}}"
CLUSTER="${CLUSTER:-no_cluster}"
REPS_LOC="${REPS_LOC:-after_pad}"
USE_MEAN="${USE_MEAN:-no_mean}"

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
RETRIEVAL_BATCH_SIZE="${RETRIEVAL_BATCH_SIZE:-1}"
THREADS="${THREADS:-16}"
DEPTH="${DEPTH:-1000}"
ALPHA="${ALPHA:-0.9}"
BETA="${BETA:-0.1}"
PRECISION="${PRECISION:-fp16}"
USE_FAISS_GPU="${USE_FAISS_GPU:-1}"
REMOVE_QUERY="${REMOVE_QUERY:-1}"

case "${PRECISION}" in
  bf16) PRECISION_ARGS=(--bf16) ;;
  fp16) PRECISION_ARGS=(--fp16) ;;
  fp32) PRECISION_ARGS=() ;;
  *)
    echo "Unsupported PRECISION=${PRECISION}; use bf16, fp16, or fp32." >&2
    exit 1
    ;;
esac

FAISS_ARGS=()
if [[ "${USE_FAISS_GPU}" == "1" ]]; then
  FAISS_ARGS+=(--use_gpu)
fi

REMOVE_QUERY_ARGS=()
if [[ "${REMOVE_QUERY}" == "1" ]]; then
  REMOVE_QUERY_ARGS+=(--remove_query)
fi

COMMON_PATH="${BASE_MODEL}/${DATASET}/${TARGET_TYPE}/${FILTER}/${CALCULATE_TYPE}/${PROMPT_TYPE}"

run_search() {
  local manual_label="$1"
  local manual_args=()

  if [[ "${manual_label}" == "manual" ]]; then
    manual_args+=(--sparse_manual)
  fi

  local path_suffix
  if [[ "${manual_label}" == "manual" ]]; then
    path_suffix="${EXPENDED_TOKENS}_${manual_label}_${TEXT_SPARSE_LENGTH}_${IMAGE_SPARSE_LENGTH}_${SPARSE_VALUE_TYPE}_${CLUSTER}_${REPS_LOC}_${EOL_TYPE}_${SPARSE_LOWER_OR_UPPER}_${USE_MEAN}"
  else
    path_suffix="${EXPENDED_TOKENS}_${manual_label}_${SPARSE_LENGTH}_${SPARSE_VALUE_TYPE}_${CLUSTER}_${REPS_LOC}_${EOL_TYPE}_${SPARSE_LOWER_OR_UPPER}_${USE_MEAN}"
  fi
  local val_dense_dir="dense_output/${COMMON_PATH}/${VAL_SPLIT}/${path_suffix}"
  local val_sparse_dir="sparse_output/${COMMON_PATH}/${VAL_SPLIT}/${path_suffix}"
  local test_dense_dir="dense_output/${COMMON_PATH}/${TEST_SPLIT}/${path_suffix}"
  local test_sparse_dir="sparse_output/${COMMON_PATH}/${TEST_SPLIT}/${path_suffix}"
  local output_dir="output/${COMMON_PATH}/${TEST_SPLIT}/${path_suffix}/result"

  echo "Running ${EOL_TYPE}/${CALCULATE_TYPE} search with sparse_manual=${manual_label}"
  echo "  dense reps:  ${test_dense_dir}"
  echo "  sparse index: ${test_sparse_dir}"
  echo "  output:      ${output_dir}"

  "${DEEPSPEED_BIN}" \
    --num_gpus="${GPUS}" \
    --num_nodes="${NUM_NODES}" \
    --master_port="${MASTER_PORT}" \
    src/search_similarity.py \
    --model_name_or_path "${MODEL_PATH}" \
    --output_dir "${output_dir}" \
    "${PRECISION_ARGS[@]}" \
    --dataset_name "${DATASET}" \
    --dataset_split "${TEST_SPLIT}" \
    --per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --threads "${THREADS}" \
    --passage_reps "${test_dense_dir}" \
    --sparse_index "${test_sparse_dir}" \
    --depth "${DEPTH}" \
    --query_type "${QUERY_TYPE}" \
    --retrieval_batch_size "${RETRIEVAL_BATCH_SIZE}" \
    --alpha "${ALPHA}" \
    --beta "${BETA}" \
    --num_expended_tokens "${EXPENDED_TOKENS}" \
    --reps_loc "${REPS_LOC}" \
    --eol_type "${EOL_TYPE}" \
    --sparse_value_type "${SPARSE_VALUE_TYPE}" \
    --sparse_lower_or_upper "${SPARSE_LOWER_OR_UPPER}" \
    --sparse_length "${SPARSE_LENGTH}" \
    --text_sparse_length "${TEXT_SPARSE_LENGTH}" \
    --image_sparse_length "${IMAGE_SPARSE_LENGTH}" \
    --sparse_type "${SPARSE_TYPE}" \
    --calculate_type "${CALCULATE_TYPE}" \
    --prompt_type "${PROMPT_TYPE}" \
    "${FAISS_ARGS[@]}" \
    "${REMOVE_QUERY_ARGS[@]}" \
    "${manual_args[@]}"
}

case "${RUN_MODE}" in
  both)
    run_search no_manual
    run_search manual
    ;;
  no_manual)
    run_search no_manual
    ;;
  manual)
    run_search manual
    ;;
esac

echo "Search run(s) completed."
if [[ "${CALCULATE_TYPE}" == "concat" ]]; then
  echo "Similarity statistics (when enabled) include each available setting; a second setting adds the paired comparison."
elif [[ "${RUN_MODE}" == "both" ]]; then
  echo "The Excel report is generated after both test runs are available."
fi
