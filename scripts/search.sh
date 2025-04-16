#!/bin/bash

BASE_MODEL=llava-hf-llama3-llava-next-8b-hf
GPUS=4
NUM_NODES=1
TARGET_TYPE=text
DATASET=coco
FILTER=no_filter
EXPENDED_TOKENS=0
MANUAL=manual
MANUAL_LENGTH=128


deepspeed --num_gpus=$GPUS --num_nodes=$NUM_NODES src/search.py \
    --model_name_or_path ./checkpoints/${BASE_MODEL} \
    --lora \
    --lora_model_path ./output/${BASE_MODEL} \
    --per_device_batch_size 2 \
    --threads 16 \
    --dataset_name coco \
    --dataset_split test \
    --fp16 \
    --passage_reps dense_output/${BASE_MODEL}/${DATASET}/${TARGET_TYPE}/${FILTER}/${EXPENDED_TOKENS}_${MANUAL}_${MANUAL_LENGTH}_lora \
    --sparse_index sparse_output/${BASE_MODEL}/${DATASET}/${TARGET_TYPE}/${FILTER}/${EXPENDED_TOKENS}_${MANUAL}_${MANUAL_LENGTH}_lora \
    --depth 1000 \
    --save_dir output/${BASE_MODEL}/${DATASET}/${QUERY_TYPE}/${FILTER}/${EXPENDED_TOKENS}_${MANUAL}_${MANUAL_LENGTH}_lora/results/ \
    --use_gpu \
    --remove_query \
    --query_type image \
    --batch_size 2 \
    --num_expended_tokens 0 \
    --alpha 0.5 \
    --reps_loc 'before_pad' \
    --sparse_manual \
    --sparse_length 128 \
