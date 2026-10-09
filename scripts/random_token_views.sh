#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Usage:
#   bash scripts/random_token_views.sh [all|encode|index|search]
#
# Examples:
#   RANDOM_VIEW_SEED=42 bash scripts/random_token_views.sh all
#   BASE_MODEL=llava-hf-llava-v1.6-mistral-7b-hf bash scripts/random_token_views.sh encode

STAGE="${1:-all}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

GPUS="${GPUS:-2}"
NUM_NODES="${NUM_NODES:-1}"
MASTER_PORT="${MASTER_PORT:-29501}"

BASE_MODEL="${BASE_MODEL:-llava-hf-llava-v1.6-mistral-7b-hf}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-fs/checkpoints/${BASE_MODEL}}"
DATASET="${DATASET:-flickr}"
TASK_TYPE="ir"
TARGET_TYPE="${TARGET_TYPE:-image}"
QUERY_TYPE="${QUERY_TYPE:-text}"
VAL_SPLIT="${VAL_SPLIT:-val}"
TEST_SPLIT="${TEST_SPLIT:-test}"

PROMPT_TYPE="${PROMPT_TYPE:-prompt_random_5}"
RANDOM_VIEW_SEED="${RANDOM_VIEW_SEED:-42}"
CALCULATE_TYPE="${CALCULATE_TYPE:-concat}"
EOL_TYPE="${EOL_TYPE:-disassembleeol_separate}"
REPS_LOC="${REPS_LOC:-after_pad}"
SPARSE_TYPE="${SPARSE_TYPE:-single}"
SPARSE_LENGTH="${SPARSE_LENGTH:-30}"
SPARSE_VALUE_TYPE="${SPARSE_VALUE_TYPE:-sum}"
SPARSE_LOWER_OR_UPPER="${SPARSE_LOWER_OR_UPPER:-lower}"
SPARSE_VALUE_MEAN="${SPARSE_VALUE_MEAN:-0}"
EXPENDED_TOKENS="${EXPENDED_TOKENS:-0}"

FILTER="no_filter"
MANUAL="no_manual"
CLUSTER="no_cluster"

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
RETRIEVAL_BATCH_SIZE="${RETRIEVAL_BATCH_SIZE:-4}"
THREADS="${THREADS:-16}"
DEPTH="${DEPTH:-1000}"
ALPHA="${ALPHA:-0.9}"
BETA="${BETA:-0.1}"
PRECISION="${PRECISION:-fp16}"
USE_FAISS_GPU="${USE_FAISS_GPU:-1}"
REMOVE_QUERY="${REMOVE_QUERY:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "${MODEL_PATH}" in
  *llava-hf-llama3-llava-next-8b-hf*|*llava-hf-llava-v1.6-mistral-7b-hf*) ;;
  *)
    echo "prompt_random_5 only supports Llama3-8B and LLaVA-Mistral-7B." >&2
    exit 1
    ;;
esac

case "${STAGE}" in
  all|encode|index|search) ;;
  *)
    echo "Usage: bash scripts/random_token_views.sh [all|encode|index|search]" >&2
    exit 1
    ;;
esac

PRECISION_ARGS=()
case "${PRECISION}" in
  bf16) PRECISION_ARGS+=(--bf16) ;;
  fp16) PRECISION_ARGS+=(--fp16) ;;
  fp32) ;;
  *)
    echo "Unsupported PRECISION=${PRECISION}; use bf16, fp16, or fp32." >&2
    exit 1
    ;;
esac

MEAN_ARGS=()
if [[ "${SPARSE_VALUE_MEAN}" == "1" ]]; then
  MEAN_ARGS+=(--sparse_value_mean)
  USE_MEAN="mean"
else
  USE_MEAN="no_mean"
fi

FAISS_ARGS=()
if [[ "${USE_FAISS_GPU}" == "1" ]]; then
  FAISS_ARGS+=(--use_gpu)
fi

REMOVE_QUERY_ARGS=()
if [[ "${REMOVE_QUERY}" == "1" ]]; then
  REMOVE_QUERY_ARGS+=(--remove_query)
fi

# encode.py/search.py use model_name_or_path[28:] as the model output directory name.
MODEL_OUTPUT_NAME="${MODEL_PATH:28}"
if [[ -z "${MODEL_OUTPUT_NAME}" ]]; then
  echo "MODEL_PATH must contain at least 29 characters to match the current output-path logic." >&2
  exit 1
fi

PATH_SUFFIX="${EXPENDED_TOKENS}_${MANUAL}_${SPARSE_LENGTH}_${SPARSE_VALUE_TYPE}_${CLUSTER}_${REPS_LOC}_${EOL_TYPE}_${SPARSE_LOWER_OR_UPPER}_${USE_MEAN}_random_view_seed_${RANDOM_VIEW_SEED}"
# Match encode.py: only fusion representation directories include the sparse type.
if [[ "${SPARSE_TYPE}" == "fusion" ]]; then
  PATH_SUFFIX="${PATH_SUFFIX}_${SPARSE_TYPE}"
fi
COMMON_PATH="${MODEL_OUTPUT_NAME}/${DATASET}/${TARGET_TYPE}/${FILTER}/${CALCULATE_TYPE}/${PROMPT_TYPE}"

VAL_DENSE_DIR="dense_output/${COMMON_PATH}/${VAL_SPLIT}/${PATH_SUFFIX}"
VAL_SPARSE_DIR="sparse_output/${COMMON_PATH}/${VAL_SPLIT}/${PATH_SUFFIX}"
TEST_DENSE_DIR="dense_output/${COMMON_PATH}/${TEST_SPLIT}/${PATH_SUFFIX}"
TEST_SPARSE_DIR="sparse_output/${COMMON_PATH}/${TEST_SPLIT}/${PATH_SUFFIX}"
SEARCH_OUTPUT_DIR="output/${COMMON_PATH}/${TEST_SPLIT}/${PATH_SUFFIX}/result"

run_encode() {
  local split="$1"

  echo "Encoding ${DATASET}/${split} with ${MODEL_OUTPUT_NAME}, seed=${RANDOM_VIEW_SEED}"
  deepspeed \
    --num_gpus="${GPUS}" \
    --num_nodes="${NUM_NODES}" \
    --master_port="${MASTER_PORT}" \
    src/encode.py \
    --model_name_or_path "${MODEL_PATH}" \
    --output_dir ./output \
    "${PRECISION_ARGS[@]}" \
    --dataset_name "${DATASET}" \
    --dataset_split "${split}" \
    --task_type "${TASK_TYPE}" \
    --encode_type "${TARGET_TYPE}" \
    --per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --reps_loc "${REPS_LOC}" \
    --num_expended_tokens "${EXPENDED_TOKENS}" \
    --eol_type "${EOL_TYPE}" \
    --sparse_value_type "${SPARSE_VALUE_TYPE}" \
    --sparse_lower_or_upper "${SPARSE_LOWER_OR_UPPER}" \
    --sparse_length "${SPARSE_LENGTH}" \
    --sparse_type "${SPARSE_TYPE}" \
    --calculate_type "${CALCULATE_TYPE}" \
    --prompt_type "${PROMPT_TYPE}" \
    --random_view_seed "${RANDOM_VIEW_SEED}" \
    "${MEAN_ARGS[@]}"
}

run_index() {
  local sparse_dir="$1"

  echo "Building sparse index: ${sparse_dir}/index"
  "${PYTHON_BIN}" -m pyserini.index.lucene \
    --collection JsonVectorCollection \
    --input "${sparse_dir}" \
    --index "${sparse_dir}/index" \
    --generator DefaultLuceneDocumentGenerator \
    --threads "${THREADS}" \
    --impact \
    --pretokenized
}

run_search() {
  echo "Searching ${DATASET} with random token views, seed=${RANDOM_VIEW_SEED}"
  deepspeed \
    --num_gpus="${GPUS}" \
    --num_nodes="${NUM_NODES}" \
    --master_port="${MASTER_PORT}" \
    src/search.py \
    --model_name_or_path "${MODEL_PATH}" \
    --output_dir "${SEARCH_OUTPUT_DIR}" \
    "${PRECISION_ARGS[@]}" \
    --dataset_name "${DATASET}" \
    --dataset_split "${TEST_SPLIT}" \
    --task_type "${TASK_TYPE}" \
    --per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --threads "${THREADS}" \
    --passage_reps "${TEST_DENSE_DIR}" \
    --sparse_index "${TEST_SPARSE_DIR}" \
    --val_passage_reps "${VAL_DENSE_DIR}" \
    --val_sparse_index "${VAL_SPARSE_DIR}" \
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
    --sparse_type "${SPARSE_TYPE}" \
    --calculate_type "${CALCULATE_TYPE}" \
    --prompt_type "${PROMPT_TYPE}" \
    --random_view_seed "${RANDOM_VIEW_SEED}" \
    "${MEAN_ARGS[@]}" \
    "${FAISS_ARGS[@]}" \
    "${REMOVE_QUERY_ARGS[@]}"
}

if [[ "${STAGE}" == "all" || "${STAGE}" == "encode" ]]; then
  run_encode "${VAL_SPLIT}"
  run_encode "${TEST_SPLIT}"
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "index" ]]; then
  run_index "${VAL_SPARSE_DIR}"
  run_index "${TEST_SPARSE_DIR}"
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "search" ]]; then
  run_search
fi
