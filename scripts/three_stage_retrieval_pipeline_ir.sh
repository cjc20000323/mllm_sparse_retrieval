#!/usr/bin/env bash
# Ordinary LLaVA IR retrieve; edit this configuration or override it in the environment.
set -euo pipefail
# Single values override the lists; empty single values enable the configured matrix.
MODEL_FAMILY="${MODEL_FAMILY-}"
MODEL_FAMILIES="${MODEL_FAMILIES-llava_mistral}" # llava_mistral llava_llama
MODEL_PATH="${MODEL_PATH-}" # single model only; per-family paths work with a batch
MODEL_PATH_LLAVA_MISTRAL="${MODEL_PATH_LLAVA_MISTRAL-}"
MODEL_PATH_LLAVA_LLAMA="${MODEL_PATH_LLAVA_LLAMA-}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT-/root/autodl-fs/checkpoints}"
DATASET="${DATASET-}"
DATASETS="${DATASETS-flickr}"
DATASET_SPLIT="${DATASET_SPLIT-test}"
QUERY_TYPE="${QUERY_TYPE-}" # image or text overrides QUERY_TYPES
QUERY_TYPES="${QUERY_TYPES-image text}" # image-to-text, then text-to-image
PROMPT_TYPE="${PROMPT_TYPE-prompt_shared_5}" # set prompt_5 explicitly for the old prompt group
DATA_ROOT="${DATA_ROOT-/root/autodl-tmp}"
DENSE_OUTPUT_ROOT="${DENSE_OUTPUT_ROOT-./dense_output}"
SPARSE_OUTPUT_ROOT="${SPARSE_OUTPUT_ROOT-./sparse_output}"
GPUS="${GPUS-1}"
NUM_NODES="${NUM_NODES-1}"
DTYPE="${DTYPE-float16}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE-1}"
REPS_LOC="${REPS_LOC-after_pad}"
SPARSE_LENGTH="${SPARSE_LENGTH-40}"
TEXT_SPARSE_LENGTH="${TEXT_SPARSE_LENGTH-${SPARSE_LENGTH}}"
IMAGE_SPARSE_LENGTH="${IMAGE_SPARSE_LENGTH-${SPARSE_LENGTH}}"
SPARSE_VALUE_TYPE="${SPARSE_VALUE_TYPE-sum}"
SPARSE_LOWER_OR_UPPER="${SPARSE_LOWER_OR_UPPER-lower}"
SPARSE_VALUE_MEAN="${SPARSE_VALUE_MEAN-false}"
IS_FILTERED="${IS_FILTERED-false}"
NUM_EXPENDED_TOKENS="${NUM_EXPENDED_TOKENS-0}"
RANDOM_VIEW_SEED="${RANDOM_VIEW_SEED-42}"
LLAVA_IMAGE_MODE="${LLAVA_IMAGE_MODE-anyres}"
EOL_TYPE="${EOL_TYPE-disassembleeol_separate}"
CALCULATE_TYPE="${CALCULATE_TYPE-concat}"
SPARSE_MANUAL="${SPARSE_MANUAL-false}"
USE_FREERET_DENSE="${USE_FREERET_DENSE-false}"
USE_FREERET_PROMPT_CONSTRAINTS="${USE_FREERET_PROMPT_CONSTRAINTS-false}"
USE_FREERET_FULL_PROMPT="${USE_FREERET_FULL_PROMPT-false}"
PYTHON_BIN="${PYTHON_BIN-python}"
DEEPSPEED_BIN="${DEEPSPEED_BIN-deepspeed}"
DRY_RUN="${DRY_RUN-false}"
SPARSE_THREADS="${SPARSE_THREADS-16}"
SPARSE_CORPUS_DIR="${SPARSE_CORPUS_DIR-}" # optional single-job corpus/index override
# sparse_concat -> hybrid (or dense) -> the existing pointwise reranker.
RESULT_ROOT="${RESULT_ROOT-./three_stage_results}"
STAGE1_METHOD="${STAGE1_METHOD-sparse_concat}"
STAGE1_DEPTH="${STAGE1_DEPTH-1000}"
STAGE2_DEPTH="${STAGE2_DEPTH-1000}"
STAGE2_METHOD="${STAGE2_METHOD-hybrid}"
HYBRID_ALPHA="${HYBRID_ALPHA-0.5}"
SELECT_HYBRID_ALPHA="${SELECT_HYBRID_ALPHA-false}"
RETRIEVAL_BATCH_SIZE="${RETRIEVAL_BATCH_SIZE-1}"
RETRIEVAL_DTYPE="${RETRIEVAL_DTYPE-${DTYPE}}"
RANKER_DTYPE="${RANKER_DTYPE-${DTYPE}}"
RERANK_TEMPLATE="${RERANK_TEMPLATE-caption_generation}"
RERANK_TYPE="${RERANK_TYPE-pointwise}"
RERANK_NUM="${RERANK_NUM-5}"
RERANK_BATCH_SIZE="${RERANK_BATCH_SIZE-1}"
MODIFY_TYPE="${MODIFY_TYPE-auto}" # auto: caption image-to-text uses modify_single; text-to-image uses no
USE_RERANK_PRIOR_CACHE="${USE_RERANK_PRIOR_CACHE-auto}" # auto follows resolved modify_single per direction; true/false override
OVERWRITE="${OVERWRITE-false}"
OUTPUT_DIR="${OUTPUT_DIR-}" # explicit paths are single-job overrides
STAGE1_SPARSE_INDEX="${STAGE1_SPARSE_INDEX-}"
STAGE2_DENSE_INDEX="${STAGE2_DENSE_INDEX-}"
STAGE2_SPARSE_INDEX="${STAGE2_SPARSE_INDEX-}"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/shared_views_task_common.sh"
shared_views_task_main retrieve ir "$@"
