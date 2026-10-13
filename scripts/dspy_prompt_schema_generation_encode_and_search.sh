#!/usr/bin/env bash

set -euo pipefail

# Run this script from any directory.  The Python program still writes its
# search results under the path_prefix configured in the Python source.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Hardware and launcher settings.
GPUS="${GPUS:-4}"
NUM_NODES="${NUM_NODES:-1}"

# Model/data settings.  The default model path matches the path layout used
# by src/dspy_prompt_schema_generation_encode_and_search.py (model path
# components after /root/autodl-fs/checkpoints/ are used in output paths).
BASE_MODEL="${BASE_MODEL:-llava-hf-llava-v1.6-mistral-7b-hf}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/root/autodl-fs/checkpoints/${BASE_MODEL}}"
DSPY_MODEL_PATH="${DSPY_MODEL_PATH:-/root/autodl-fs/checkpoints/Meta-Llama-3.1-8B-Instruct}"
DATASET="${DATASET:-flickr}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
TASK_TYPE="${TASK_TYPE:-ir}"
ENCODE_TYPE="${ENCODE_TYPE:-text}"

# Representation/output settings.  Override these variables when the
# representations need to be stored somewhere other than the defaults.
DENSE_OUTPUT_DIR="${DENSE_OUTPUT_DIR:-./dense_output}"
SPARSE_OUTPUT_DIR="${SPARSE_OUTPUT_DIR:-./sparse_output}"
REPS_LOC="${REPS_LOC:-after_pad}"
EOL_TYPE="${EOL_TYPE:-disassembleeol_separate}"
CALCULATE_TYPE="${CALCULATE_TYPE:-concat}"
PROMPT_TYPE="${PROMPT_TYPE:-prompt_5}"
SPARSE_VALUE_TYPE="${SPARSE_VALUE_TYPE:-sum}"
SPARSE_LOWER_OR_UPPER="${SPARSE_LOWER_OR_UPPER:-lower}"
SPARSE_LENGTH="${SPARSE_LENGTH:-40}"
NUM_EXPENDED_TOKENS="${NUM_EXPENDED_TOKENS:-0}"
DATASET_SUFFIX="${DATASET_SUFFIX:-no}"

# DSPy settings.
DEMONSTRATION_NUM="${DEMONSTRATION_NUM:-1}"
DSPY_STRENGTH="${DSPY_STRENGTH:-light}"
FORCE_REGENERATE_DSPY_ASPECTS="${FORCE_REGENERATE_DSPY_ASPECTS:-false}"

# Validation-dataset arguments are effective when
# USE_VALIDATION_DATASET_RESOLVER=True is set in the Python script.  With that
# switch left False, the Python script keeps its original validation behavior.
USE_FIXED_VALIDATION_DATASET="${USE_FIXED_VALIDATION_DATASET:-false}"
VALIDATION_DATASET_NAME="${VALIDATION_DATASET_NAME:-}"

VALIDATION_ARGS=()
if [[ "${USE_FIXED_VALIDATION_DATASET}" == "true" ]]; then
    VALIDATION_ARGS+=(--use_fixed_validation_dataset)
fi
if [[ -n "${VALIDATION_DATASET_NAME}" ]]; then
    VALIDATION_ARGS+=(--validation_dataset_name "${VALIDATION_DATASET_NAME}")
fi

REGENERATE_ARGS=()
if [[ "${FORCE_REGENERATE_DSPY_ASPECTS}" == "true" ]]; then
    REGENERATE_ARGS+=(--force_regenerate_dspy_aspects)
fi

# Warm-start behavior is controlled by USE_INITIAL_DSPY_ASPECTS and
# INITIAL_DSPY_ASPECTS in src/dspy_prompt_schema_generation_encode_and_search.py.
# Set USE_INITIAL_DSPY_ASPECTS=True there to evaluate the supplied initial
# aspects before accepting a newly generated DSPy candidate.
deepspeed \
    --num_gpus="${GPUS}" \
    --num_nodes="${NUM_NODES}" \
    src/dspy_prompt_schema_generation_encode_and_search.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --dspy_model_path "${DSPY_MODEL_PATH}" \
    --dataset_name "${DATASET}" \
    --dataset_split "${DATASET_SPLIT}" \
    --task_type "${TASK_TYPE}" \
    --encode_type "${ENCODE_TYPE}" \
    --dense_output_dir "${DENSE_OUTPUT_DIR}" \
    --sparse_output_dir "${SPARSE_OUTPUT_DIR}" \
    --per_device_batch_size 1 \
    --threads 16 \
    --depth 1000 \
    --retrieval_batch_size 1 \
    --remove_query \
    --fp16 \
    --reps_loc "${REPS_LOC}" \
    --eol_type "${EOL_TYPE}" \
    --calculate_type "${CALCULATE_TYPE}" \
    --prompt_type "${PROMPT_TYPE}" \
    --sparse_value_type "${SPARSE_VALUE_TYPE}" \
    --sparse_lower_or_upper "${SPARSE_LOWER_OR_UPPER}" \
    --sparse_length "${SPARSE_LENGTH}" \
    --text_sparse_length "${SPARSE_LENGTH}" \
    --image_sparse_length "${SPARSE_LENGTH}" \
    --num_expended_tokens "${NUM_EXPENDED_TOKENS}" \
    --dataset_suffix "${DATASET_SUFFIX}" \
    --demonstration_num "${DEMONSTRATION_NUM}" \
    --dspy_strength "${DSPY_STRENGTH}" \
    "${VALIDATION_ARGS[@]}" \
    "${REGENERATE_ARGS[@]}"
