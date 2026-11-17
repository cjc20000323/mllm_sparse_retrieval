#!/bin/bash

# 从项目根目录运行；三个脚本的这两项配置保持一致。
ENCODE_TYPE=text       # text：图检文；image：文检图
SPARSE_MANUAL=false    # false：原文取词（最多128）；true：全词表预测Top-30

GPUS=2
NUM_NODES=1

deepspeed --num_gpus=$GPUS --num_nodes=$NUM_NODES src/encode.py \
    --model_name_or_path /root/gpufree-share/checkpoints/llava-hf-llava-v1.6-mistral-7b-hf \
    --output_dir /root/gpufree-share/output \
    --dense_output_dir /root/gpufree-share/dense_output \
    --sparse_output_dir /root/gpufree-share/sparse_output \
    --llava_image_mode base \
    --use_freeret_dense false \
    --use_freeret_full_prompt false \
    --use_freeret_prompt_constraints false \
    --fp16 \
    --dataset_name flickr \
    --dataset_split test \
    --task_type ir \
    --encode_type "$ENCODE_TYPE" \
    --per_device_batch_size 1 \
    --reps_loc after_pad \
    --num_expended_tokens 0 \
    --eol_type prompteol \
    --calculate_type separate \
    --prompt_type prompt_5 \
    --sparse_manual "$SPARSE_MANUAL" \
    --sparse_length 30 \
    --text_sparse_length 30 \
    --image_sparse_length 30 \
    --sparse_value_type sum \
    --sparse_lower_or_upper lower
