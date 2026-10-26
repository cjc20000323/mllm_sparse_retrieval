#!/bin/bash

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
    bash "$(dirname "$0")/random_token_views.sh" encode
else
    GPUS=4
    NUM_NODES=1

    deepspeed --num_gpus=$GPUS --num_nodes=$NUM_NODES src/encode.py \
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
