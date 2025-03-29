#!/bin/bash

GPUS=4
NUM_NODES=1

deepspeed --num_gpus=$GPUS --num_nodes=$NUM_NODES src/encode.py \
	--model_name_or_path ./checkpoints/llava-hf-llama3-llava-next-8b-hf \
	--output_dir ./output \
	--fp16 \
    --dataset_name flickr \
    --encode_type image \
    --per_device_batch_size 4 \
    --dataset_split test \
    --is_filtered \


