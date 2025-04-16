#!/bin/bash

GPUS=4
NUM_NODES=1

deepspeed --num_gpus=$GPUS --num_nodes=$NUM_NODES src/encode_run.py \
	--model_name_or_path ./checkpoints/llava-hf-llava-1.5-7b-hf \
	--output_dir ./output \
	--fp16 \
