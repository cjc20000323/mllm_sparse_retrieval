#!/usr/bin/env bash

set -euo pipefail

# Dedicated entry point for optimizing DSPy schema aspects against validation
# data produced by generate_llava70b_dataset_text.py.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Hardware and launcher settings.
GPUS="${GPUS:-4}"
NUM_NODES="${NUM_NODES:-1}"

# Model/data settings.
BASE_MODEL="${BASE_MODEL:-llava-hf-llava-v1.6-mistral-7b-hf}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/root/autodl-fs/checkpoints/${BASE_MODEL}}"
# This is the API's served model ID, not necessarily its checkpoint directory.
# Deploy the HF-format NeXT-72B checkpoint separately, for example:
# CUDA_VISIBLE_DEVICES=<server GPUs> vllm serve /path/to/llava-next-72b-hf \
#   --served-model-name llava-hf/llava-next-72b-hf --tensor-parallel-size <N> \
#   --max-num-seqs 1 --limit-mm-per-prompt '{"image":1}'
# Keep the service's GPU allocation separate from this DeepSpeed job.
DSPY_MODEL_PATH="${DSPY_MODEL_PATH:-llava-hf/llava-next-72b-hf}"
DSPY_API_BASE="${DSPY_API_BASE:-http://127.0.0.1:8000/v1}"
# Optional authentication is read directly from the DSPY_API_KEY environment.
DSPY_REFLECTION_MODE="${DSPY_REFLECTION_MODE:-multimodal}"
DSPY_FEEDBACK_FAILURES_PER_DIRECTION="${DSPY_FEEDBACK_FAILURES_PER_DIRECTION:-2}"
DSPY_REFLECTION_IMAGE_MAX_EDGE="${DSPY_REFLECTION_IMAGE_MAX_EDGE:-672}"
DSPY_REFLECTION_CASE_MAX_TOKENS="${DSPY_REFLECTION_CASE_MAX_TOKENS:-512}"
DSPY_REFLECTION_INSTRUCTION_MAX_TOKENS="${DSPY_REFLECTION_INSTRUCTION_MAX_TOKENS:-2048}"
DSPY_API_TIMEOUT="${DSPY_API_TIMEOUT:-300}"
DATASET="${DATASET:-flickr}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
TASK_TYPE="${TASK_TYPE:-ir}"
ENCODE_TYPE="${ENCODE_TYPE:-text}"
GENERATED_VALIDATION_DATA_ROOT="${GENERATED_VALIDATION_DATA_ROOT:-./generated_llava70b_data}"

# The current CIR entry point does not forward generated-validation settings to
# ComposedTextImageRetrievalDataset, so accepting it here would silently use the
# original Fashion-IQ validation data.
if [[ "${TASK_TYPE,,}" == "cir" ]]; then
    echo "Generated validation data is not currently supported for TASK_TYPE=cir." >&2
    exit 2
fi

if [[ ! -d "${GENERATED_VALIDATION_DATA_ROOT}/data" ]]; then
    echo "Generated validation data directory was not found: ${GENERATED_VALIDATION_DATA_ROOT}/data" >&2
    echo "Generate it first with generate_llava70b_dataset_text.py and --split val." >&2
    exit 2
fi

# Representation/output settings.
DENSE_OUTPUT_DIR="${DENSE_OUTPUT_DIR:-./dense_output_dspy}"
SPARSE_OUTPUT_DIR="${SPARSE_OUTPUT_DIR:-./sparse_output_dspy}"
REPS_LOC="${REPS_LOC:-after_pad}"
EOL_TYPE="${EOL_TYPE:-disassembleeol_separate}"
CALCULATE_TYPE="${CALCULATE_TYPE:-concat}"
PROMPT_TYPE="${PROMPT_TYPE:-prompt_5}"
SPARSE_VALUE_TYPE="${SPARSE_VALUE_TYPE:-sum}"
SPARSE_LOWER_OR_UPPER="${SPARSE_LOWER_OR_UPPER:-lower}"
SPARSE_LENGTH="${SPARSE_LENGTH:-40}"
NUM_EXPENDED_TOKENS="${NUM_EXPENDED_TOKENS:-0}"
DATASET_SUFFIX="${DATASET_SUFFIX:-no}"

# DSPy GEPA settings.  The hand-written views are GEPA round 1; subsequent
# rounds use failure-grounded feedback from the generated validation set.
# Regeneration is enabled by default because the aspects lock key does not
# include the generated-data root or its contents.
DEMONSTRATION_NUM="${DEMONSTRATION_NUM:-1}"
DSPY_STRENGTH="${DSPY_STRENGTH:-light}"
FORCE_REGENERATE_DSPY_ASPECTS="${FORCE_REGENERATE_DSPY_ASPECTS:-true}"
USE_INITIAL_DSPY_ASPECTS="${USE_INITIAL_DSPY_ASPECTS:-true}"
# Restrict optimization to a JSON aspect pool when DSPY_ASPECT_MODE=pool.
DSPY_ASPECT_MODE="${DSPY_ASPECT_MODE:-free}"
DSPY_ASPECT_POOL_FILE="${DSPY_ASPECT_POOL_FILE:-}"
ASPECT_SELECTION_ARGS=(--dspy_aspect_mode "${DSPY_ASPECT_MODE}")
if [[ -n "${DSPY_ASPECT_POOL_FILE}" ]]; then
    ASPECT_SELECTION_ARGS+=(--dspy_aspect_pool_file "${DSPY_ASPECT_POOL_FILE}")
fi
case "${DSPY_ASPECT_MODE}" in
    free) ;;
    pool) [[ -f "${DSPY_ASPECT_POOL_FILE}" ]] || { echo "pool mode requires an existing DSPY_ASPECT_POOL_FILE." >&2; exit 2; } ;;
    *) echo "DSPY_ASPECT_MODE must be free or pool." >&2; exit 2 ;;
esac

# These arguments become effective when USE_VALIDATION_DATASET_RESOLVER=True is
# enabled in the Python program.
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

INITIAL_ASPECT_ARGS=()
if [[ "${USE_INITIAL_DSPY_ASPECTS,,}" == "true" ]]; then
    INITIAL_ASPECT_ARGS+=(--use_initial_dspy_aspects)
fi

deepspeed \
    --num_gpus="${GPUS}" \
    --num_nodes="${NUM_NODES}" \
    src/dspy_prompt_schema_generation_encode_and_search.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --dspy_model_path "${DSPY_MODEL_PATH}" \
    --dspy_api_base "${DSPY_API_BASE}" \
    --dspy_reflection_mode "${DSPY_REFLECTION_MODE}" \
    --dspy_feedback_failures_per_direction "${DSPY_FEEDBACK_FAILURES_PER_DIRECTION}" \
    --dspy_reflection_image_max_edge "${DSPY_REFLECTION_IMAGE_MAX_EDGE}" \
    --dspy_reflection_case_max_tokens "${DSPY_REFLECTION_CASE_MAX_TOKENS}" \
    --dspy_reflection_instruction_max_tokens "${DSPY_REFLECTION_INSTRUCTION_MAX_TOKENS}" \
    --dspy_api_timeout "${DSPY_API_TIMEOUT}" \
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
    --use_generated_validation_data \
    --generated_validation_data_root "${GENERATED_VALIDATION_DATA_ROOT}" \
    "${VALIDATION_ARGS[@]}" \
    "${REGENERATE_ARGS[@]}" \
    "${INITIAL_ASPECT_ARGS[@]}" \
    "${ASPECT_SELECTION_ARGS[@]}"
