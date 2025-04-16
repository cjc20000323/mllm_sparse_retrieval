#!/bin/bash

GPUS=4
NUM_NODES=1

python src/encode_intern.py \
	--model_name_or_path ./checkpoints/OpenGVLab-InternVL2_5-8B \
	--output_dir ./output \
	--bf16 \
    --dataset_name flickr \
    --encode_type text \
    --per_device_batch_size 4 \
    --dataset_split test \
    --reps_loc 'before_pad' \
    --num_expended_tokens 0 \
