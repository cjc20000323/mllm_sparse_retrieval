#!/usr/bin/env bash

# Encode validation data generated with the LLaVA-NeXT-style training prompts.
# This dedicated entry point always enables generated validation data and keeps
# its representations separate from both original and normally generated data.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

BASE_MODEL="${BASE_MODEL:-llava-hf-llava-v1.6-mistral-7b-hf}"
MODEL_PATH="${MODEL_PATH:-./checkpoints/${BASE_MODEL}}"
DATASET="${DATASET:-flickr}"
TASK_TYPE="${TASK_TYPE:-ir}"
ENCODE_TYPE="${ENCODE_TYPE:-image}"
PROMPT_TYPE="${PROMPT_TYPE:-prompt_5}"

GPUS="${GPUS:-4}"
NUM_NODES="${NUM_NODES:-1}"
USE_FEW_SHOT_EXAMPLES="${USE_FEW_SHOT_EXAMPLES:-false}"
FEW_SHOT_NUM="${FEW_SHOT_NUM:-2}"
FEW_SHOT_SPLIT="${FEW_SHOT_SPLIT:-test}"

FEW_SHOT_ARGS=()
case "${USE_FEW_SHOT_EXAMPLES,,}" in
  true|1|yes)
    if ! [[ "${FEW_SHOT_NUM}" =~ ^[1-9][0-9]*$ ]]; then
      echo "FEW_SHOT_NUM must be a positive integer." >&2
      exit 2
    fi
    if [[ "${FEW_SHOT_SPLIT}" != "val" && "${FEW_SHOT_SPLIT}" != "test" ]]; then
      echo "FEW_SHOT_SPLIT must be val or test." >&2
      exit 2
    fi
    FEW_SHOT_MODE="enabled"
    DEFAULT_GENERATED_VALIDATION_DATA_ROOT="./generated_llava70b_data_llava_next_training_few_shot_${FEW_SHOT_SPLIT}_${FEW_SHOT_NUM}"
    FEW_SHOT_ARGS+=(
      --use_few_shot_examples
      --few_shot_num "${FEW_SHOT_NUM}"
      --few_shot_split "${FEW_SHOT_SPLIT}"
    )
    ;;
  false|0|no|"")
    FEW_SHOT_MODE="disabled"
    DEFAULT_GENERATED_VALIDATION_DATA_ROOT="./generated_llava70b_data_llava_next_training"
    ;;
  *)
    echo "USE_FEW_SHOT_EXAMPLES must be true or false." >&2
    exit 2
    ;;
esac

# An explicitly supplied source root always takes priority over the
# mode-specific defaults above.
GENERATED_VALIDATION_DATA_ROOT="${GENERATED_VALIDATION_DATA_ROOT:-${DEFAULT_GENERATED_VALIDATION_DATA_ROOT}}"

echo "dataset=${DATASET}"
echo "task_type=${TASK_TYPE}"
echo "encode_type=${ENCODE_TYPE}"
echo "model_path=${MODEL_PATH}"
echo "generated_validation_data_root=${GENERATED_VALIDATION_DATA_ROOT}"
echo "generated_prompt_mode=llava_next_training"
echo "few_shot_mode=${FEW_SHOT_MODE}"
if [[ "${FEW_SHOT_MODE}" == "enabled" ]]; then
    echo "few_shot_num=${FEW_SHOT_NUM}"
    echo "few_shot_split=${FEW_SHOT_SPLIT}"
fi

deepspeed --num_gpus="${GPUS}" --num_nodes="${NUM_NODES}" src/encode.py \
    --model_name_or_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR:-./output}" \
    --fp16 \
    --dataset_name "${DATASET}" \
    --task_type "${TASK_TYPE}" \
    --encode_type "${ENCODE_TYPE}" \
    --per_device_batch_size "${PER_DEVICE_BATCH_SIZE:-1}" \
    --dataset_split val \
    --reps_loc "${REPS_LOC:-after_pad}" \
    --num_expended_tokens "${EXPENDED_TOKENS:-0}" \
    --eol_type "${EOL_TYPE:-disassembleeol_separate}" \
    --sparse_value_type "${SPARSE_VALUE_TYPE:-sum}" \
    --sparse_lower_or_upper "${SPARSE_LOWER_OR_UPPER:-lower}" \
    --calculate_type "${CALCULATE_TYPE:-concat}" \
    --prompt_type "${PROMPT_TYPE}" \
    --sparse_length "${SPARSE_LENGTH:-40}" \
    --use_generated_validation_data \
    --generated_validation_data_root "${GENERATED_VALIDATION_DATA_ROOT}" \
    --use_llava_next_training_prompts \
    "${FEW_SHOT_ARGS[@]}"
