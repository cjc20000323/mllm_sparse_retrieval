#!/bin/bash

GPUS=4
NUM_NODES=1

deepspeed --num_gpus=$GPUS --num_nodes=$NUM_NODES src/train.py \
	--model_name_or_path ./checkpoints/llava-hf-llama3-llava-next-8b-hf \
	--output_dir ./output/llava-hf-llama3-llava-next-8b-hf \
	--fp16 \
    --dataset_name coco \
    --encode_type text \
    --per_device_batch_size 4 \
    --dataset_split test \
    --reps_loc 'before_pad' \
    --num_expended_tokens 0 \
    --load_kbit 4 \
    --lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_bias none \
    --use_few_shot \
    --few_shot_sum 200 \
    --learning_rate 5e-5 \
    --train_mode dense_emb \
    --num_train_epochs 5 \
    --deepspeed ./ds_configs/zero1.json \
    --gather_save_gradient \
    --tau 0.05 \
    
