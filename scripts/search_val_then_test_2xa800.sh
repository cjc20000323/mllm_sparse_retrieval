#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

GPUS="${GPUS:-2}"
NUM_NODES="${NUM_NODES:-1}"
MASTER_PORT="${MASTER_PORT:-29501}"

BASE_MODEL="${BASE_MODEL:-llava-hf-llava-v1.6-mistral-7b-hf}"
DATASET="${DATASET:-flickr}"
TASK_TYPE="${TASK_TYPE:-ir}"
TARGET_TYPE="${TARGET_TYPE:-text}"
QUERY_TYPE="${QUERY_TYPE:-image}"
TEST_SPLIT="${TEST_SPLIT:-test}"
VAL_SPLIT="${VAL_SPLIT:-val}"

FILTER="${FILTER:-no_filter}"
CALCULATE_TYPE="${CALCULATE_TYPE:-concat}"
PROMPT_TYPE="${PROMPT_TYPE:-prompt_5}"
EXPENDED_TOKENS="${EXPENDED_TOKENS:-0}"
MANUAL="${MANUAL:-no_manual}"
SPARSE_LENGTH="${SPARSE_LENGTH:-40}"
SPARSE_VALUE_TYPE="${SPARSE_VALUE_TYPE:-sum}"
CLUSTER="${CLUSTER:-no_cluster}"
REPS_LOC="${REPS_LOC:-after_pad}"
EOL_TYPE="${EOL_TYPE:-disassembleeol_separate}"
SPARSE_LOWER_OR_UPPER="${SPARSE_LOWER_OR_UPPER:-lower}"
USE_MEAN="${USE_MEAN:-no_mean}"
SPARSE_TYPE="${SPARSE_TYPE:-single}"

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
RETRIEVAL_BATCH_SIZE="${RETRIEVAL_BATCH_SIZE:-1}"
DEPTH="${DEPTH:-1000}"
THREADS="${THREADS:-16}"
PRECISION="${PRECISION:-bf16}"
USE_FAISS_GPU="${USE_FAISS_GPU:-0}"

PATH_SUFFIX="${PATH_SUFFIX:-${EXPENDED_TOKENS}_${MANUAL}_${SPARSE_LENGTH}_${SPARSE_VALUE_TYPE}_${CLUSTER}_${REPS_LOC}_${EOL_TYPE}_${SPARSE_LOWER_OR_UPPER}_${USE_MEAN}_${SPARSE_TYPE}}"
LEGACY_PATH_SUFFIX="${EXPENDED_TOKENS}_${MANUAL}_${SPARSE_LENGTH}_${SPARSE_VALUE_TYPE}_${CLUSTER}_${REPS_LOC}_${EOL_TYPE}_${SPARSE_LOWER_OR_UPPER}_${USE_MEAN}"

COMMON_PATH="${BASE_MODEL}/${DATASET}/${TARGET_TYPE}/${FILTER}/${CALCULATE_TYPE}/${PROMPT_TYPE}"

PASSAGE_REPS="${PASSAGE_REPS:-dense_output/${COMMON_PATH}/${TEST_SPLIT}/${PATH_SUFFIX}}"
SPARSE_INDEX="${SPARSE_INDEX:-sparse_output/${COMMON_PATH}/${TEST_SPLIT}/${PATH_SUFFIX}}"
VAL_PASSAGE_REPS="${VAL_PASSAGE_REPS:-dense_output/${COMMON_PATH}/${VAL_SPLIT}/${PATH_SUFFIX}}"
VAL_SPARSE_INDEX="${VAL_SPARSE_INDEX:-sparse_output/${COMMON_PATH}/${VAL_SPLIT}/${PATH_SUFFIX}}"

if [[ ! -d "${PASSAGE_REPS}" && -d "dense_output/${COMMON_PATH}/${TEST_SPLIT}/${LEGACY_PATH_SUFFIX}" ]]; then
  PASSAGE_REPS="dense_output/${COMMON_PATH}/${TEST_SPLIT}/${LEGACY_PATH_SUFFIX}"
fi

if [[ ! -d "${SPARSE_INDEX}" && -d "sparse_output/${COMMON_PATH}/${TEST_SPLIT}/${LEGACY_PATH_SUFFIX}" ]]; then
  SPARSE_INDEX="sparse_output/${COMMON_PATH}/${TEST_SPLIT}/${LEGACY_PATH_SUFFIX}"
fi

if [[ ! -d "${VAL_PASSAGE_REPS}" && -d "dense_output/${COMMON_PATH}/${VAL_SPLIT}/${LEGACY_PATH_SUFFIX}" ]]; then
  VAL_PASSAGE_REPS="dense_output/${COMMON_PATH}/${VAL_SPLIT}/${LEGACY_PATH_SUFFIX}"
fi

if [[ ! -d "${VAL_SPARSE_INDEX}" && -d "sparse_output/${COMMON_PATH}/${VAL_SPLIT}/${LEGACY_PATH_SUFFIX}" ]]; then
  VAL_SPARSE_INDEX="sparse_output/${COMMON_PATH}/${VAL_SPLIT}/${LEGACY_PATH_SUFFIX}"
fi

PRECISION_ARGS=()
case "${PRECISION}" in
  bf16) PRECISION_ARGS+=(--bf16) ;;
  fp16) PRECISION_ARGS+=(--fp16) ;;
  fp32) ;;
  *) echo "Unsupported PRECISION=${PRECISION}; use bf16, fp16, or fp32" >&2; exit 1 ;;
esac

FAISS_ARGS=()
if [[ "${USE_FAISS_GPU}" == "1" ]]; then
  FAISS_ARGS+=(--use_gpu)
fi

echo "Running search_val_then_test.py on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "PASSAGE_REPS=${PASSAGE_REPS}"
echo "SPARSE_INDEX=${SPARSE_INDEX}"
echo "VAL_PASSAGE_REPS=${VAL_PASSAGE_REPS}"
echo "VAL_SPARSE_INDEX=${VAL_SPARSE_INDEX}"

deepspeed \
  --num_gpus="${GPUS}" \
  --num_nodes="${NUM_NODES}" \
  --master_port="${MASTER_PORT}" \
  src/search_val_then_test.py \
  --model_name_or_path "./checkpoints/${BASE_MODEL}" \
  --output_dir "./output/${COMMON_PATH}/${TEST_SPLIT}/${PATH_SUFFIX}/result" \
  --dataset_name "${DATASET}" \
  --dataset_split "${TEST_SPLIT}" \
  --task_type "${TASK_TYPE}" \
  --per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --threads "${THREADS}" \
  "${PRECISION_ARGS[@]}" \
  --passage_reps "${PASSAGE_REPS}" \
  --sparse_index "${SPARSE_INDEX}" \
  --val_passage_reps "${VAL_PASSAGE_REPS}" \
  --val_sparse_index "${VAL_SPARSE_INDEX}" \
  --depth "${DEPTH}" \
  --remove_query \
  --query_type "${QUERY_TYPE}" \
  --retrieval_batch_size "${RETRIEVAL_BATCH_SIZE}" \
  "${FAISS_ARGS[@]}" \
  --num_expended_tokens "${EXPENDED_TOKENS}" \
  --reps_loc "${REPS_LOC}" \
  --eol_type "${EOL_TYPE}" \
  --sparse_value_type "${SPARSE_VALUE_TYPE}" \
  --sparse_lower_or_upper "${SPARSE_LOWER_OR_UPPER}" \
  --sparse_length "${SPARSE_LENGTH}" \
  --calculate_type "${CALCULATE_TYPE}" \
  --prompt_type "${PROMPT_TYPE}" \
  --sparse_type "${SPARSE_TYPE}"
