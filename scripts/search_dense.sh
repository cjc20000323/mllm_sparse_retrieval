#!/bin/bash

BASE_MODEL=llava-hf-llama3-llava-next-8b-hf
GPUS=4
NUM_NODES=1
TARGET_TYPE=text
DATASET=coco
FILTER=no_filter
EXPENDED_TOKENS=0


deepspeed --num_gpus=$GPUS --num_nodes=$NUM_NODES src/search.py \
    --model_name_or_path ./checkpoints/${BASE_MODEL} \
    --per_device_batch_size 1 \
    --threads 16 \
    --dataset_name coco \
    --dataset_split test \
    --fp16 \
    --passage_reps dense_output/${BASE_MODEL}/${DATASET}/${TARGET_TYPE}/${FILTER}/${EXPENDED_TOKENS} \
    --depth 10 \
    --save_dir output/${BASE_MODEL}/${DATASET}/${QUERY_TYPE}/${FILTER}/${EXPENDED_TOKENS}/results/ \
    --remove_query \
    --query_type image \
    --batch_size 1 \
    --num_expended_tokens 0 \
    # --use_gpu \
