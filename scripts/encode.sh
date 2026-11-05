#!/bin/bash
USE_FREERET_DENSE="${USE_FREERET_DENSE-false}"  # true: FreeRet pre-MLP dense; prompts/sparse unchanged
USE_FREERET_FULL_PROMPT="${USE_FREERET_FULL_PROMPT-false}"  # Complete FreeRet dense/sparse prompts; requires FreeRet dense
USE_FREERET_PROMPT_CONSTRAINTS="${USE_FREERET_PROMPT_CONSTRAINTS-false}"  # Prepend FreeRet constraints to all representation views
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/freeret_dense_mode.sh" || exit $?

LLAVA_IMAGE_MODE="${LLAVA_IMAGE_MODE:-anyres}"
export LLAVA_IMAGE_MODE
case "${LLAVA_IMAGE_MODE}" in anyres|base) ;; *) echo "LLAVA_IMAGE_MODE must be anyres or base" >&2; exit 2 ;; esac

PROMPT_TYPE="${PROMPT_TYPE:-prompt_5}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
USE_GENERATED_VALIDATION_DATA="${USE_GENERATED_VALIDATION_DATA:-false}"
GENERATED_VALIDATION_DATA_ROOT="${GENERATED_VALIDATION_DATA_ROOT:-./generated_llava70b_data}"

GENERATED_VALIDATION_ARGS=()
if [[ "${USE_GENERATED_VALIDATION_DATA}" == "true" ]]; then
    DATASET_SPLIT=val
    GENERATED_VALIDATION_ARGS+=(--use_generated_validation_data)
    GENERATED_VALIDATION_ARGS+=(--generated_validation_data_root "${GENERATED_VALIDATION_DATA_ROOT}")
fi

if [[ "${PROMPT_TYPE}" == "prompt_random_5" ]]; then
    # Reuse the random-view settings and encode both validation and test splits.
    export PROMPT_TYPE USE_GENERATED_VALIDATION_DATA GENERATED_VALIDATION_DATA_ROOT
    env "USE_FREERET_DENSE=${USE_FREERET_DENSE}" "USE_FREERET_PROMPT_CONSTRAINTS=${USE_FREERET_PROMPT_CONSTRAINTS}" "USE_FREERET_FULL_PROMPT=${USE_FREERET_FULL_PROMPT}" bash "$(dirname "$0")/random_token_views.sh" encode
else
    GPUS=4
    NUM_NODES=1

    deepspeed --num_gpus=$GPUS --num_nodes=$NUM_NODES src/encode.py --use_freeret_dense "${USE_FREERET_DENSE}" --use_freeret_prompt_constraints "${USE_FREERET_PROMPT_CONSTRAINTS}" --use_freeret_full_prompt "${USE_FREERET_FULL_PROMPT}" \
        --llava_image_mode "${LLAVA_IMAGE_MODE}" \
        --model_name_or_path ./checkpoints/llava-hf-llava-v1.6-mistral-7b-hf \
        --output_dir ./output \
        --fp16 \
        --dataset_name flickr \
        --encode_type image \
        --per_device_batch_size 1 \
        --dataset_split "${DATASET_SPLIT}" \
        --reps_loc 'after_pad' \
        --num_expended_tokens 0 \
        --eol_type disassembleeol_separate \
        --sparse_value_type sum \
        --sparse_lower_or_upper lower \
        --calculate_type concat \
        --prompt_type "${PROMPT_TYPE}" \
        --sparse_length 40 \
        "${GENERATED_VALIDATION_ARGS[@]}"
fi
