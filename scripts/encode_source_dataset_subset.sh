#!/usr/bin/env bash
set -euo pipefail

# Encode an original COCO/Flickr validation set or the generated 200-image
# source-dataset subset.  The subset switch controls both data loading and the
# _source_subset output-path suffix in encode.py.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

BASE_MODEL="${BASE_MODEL:-llava-hf-llava-v1.6-mistral-7b-hf}"
MODEL_PATH="${MODEL_PATH:-./checkpoints/${BASE_MODEL}}"
DATASET="${DATASET:-flickr}"
TASK_TYPE="${TASK_TYPE:-ir}"
ENCODE_TYPE="${ENCODE_TYPE:-text}"
PROMPT_TYPE="${PROMPT_TYPE:-prompt_5}"

GPUS="${GPUS:-4}"
NUM_NODES="${NUM_NODES:-1}"
USE_SOURCE_DATASET_SUBSET="${USE_SOURCE_DATASET_SUBSET:-false}"
GENERATED_VALIDATION_DATA_ROOT="${GENERATED_VALIDATION_DATA_ROOT:-./generated_val_subsets/coco_flickr_val_200_seed_2026}"

if [[ "${DATASET}" != "coco" && "${DATASET}" != "flickr" ]]; then
    echo "DATASET must be coco or flickr for source-dataset subset encoding." >&2
    exit 2
fi

VALIDATION_ARGS=()
SUBSET_MODE="disabled"
case "${USE_SOURCE_DATASET_SUBSET,,}" in
    true|1|yes)
        SUBSET_MODE="enabled"
        VALIDATION_ARGS+=(
            --use_generated_validation_data
            --generated_validation_data_root "${GENERATED_VALIDATION_DATA_ROOT}"
            --use_source_dataset_subset
        )
        ;;
    false|0|no|"")
        ;;
    *)
        echo "USE_SOURCE_DATASET_SUBSET must be true or false." >&2
        exit 2
        ;;
esac

echo "dataset=${DATASET}"
echo "task_type=${TASK_TYPE}"
echo "encode_type=${ENCODE_TYPE}"
echo "model_path=${MODEL_PATH}"
echo "source_dataset_subset_mode=${SUBSET_MODE}"
if [[ "${SUBSET_MODE}" == "enabled" ]]; then
    echo "generated_validation_data_root=${GENERATED_VALIDATION_DATA_ROOT}"
    echo "output_suffix=_generated_val_source_subset"
else
    echo "output_suffix=(unchanged)"
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
    "${VALIDATION_ARGS[@]}"
