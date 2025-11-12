#!/bin/bash

GPUS=4
NUM_NODES=1

deepspeed --num_gpus=$GPUS --num_nodes=$NUM_NODES src/encode.py \
	--model_name_or_path ./checkpoints/llava-hf-llava-v1.6-mistral-7b-hf \
	--output_dir ./output \
	--fp16 \
    --dataset_name flickr \
    --encode_type image \
    --per_device_batch_size 1 \
    --dataset_split test \
    --reps_loc 'after_pad' \
    --num_expended_tokens 0 \
    --eol_type disassembleeol_separate \
    --sparse_value_type sum \
    --sparse_lower_or_upper lower \
    --calculate_type concat \
    --prompt_type prompt_5 \
    --sparse_length 40 \
