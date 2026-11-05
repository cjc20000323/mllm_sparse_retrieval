#!/bin/bash
USE_FREERET_DENSE="${USE_FREERET_DENSE-false}"  # true: FreeRet pre-MLP dense; prompts/sparse unchanged
USE_FREERET_FULL_PROMPT="${USE_FREERET_FULL_PROMPT-false}"  # Complete FreeRet dense/sparse prompts; requires FreeRet dense
USE_FREERET_PROMPT_CONSTRAINTS="${USE_FREERET_PROMPT_CONSTRAINTS-false}"  # Prepend FreeRet constraints to all representation views
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/freeret_dense_mode.sh" || exit $?

BASE_MODEL=llava-hf-llama3-llava-next-8b-hf
GPUS=4
NUM_NODES=1
TARGET_TYPE=text
DATASET=coco
FILTER=no_filter
EXPENDED_TOKENS=0
MANUAL=no_manual
MANUAL_LENGTH=128


deepspeed --num_gpus=$GPUS --num_nodes=$NUM_NODES src/search.py --use_freeret_dense "${USE_FREERET_DENSE}" --use_freeret_prompt_constraints "${USE_FREERET_PROMPT_CONSTRAINTS}" --use_freeret_full_prompt "${USE_FREERET_FULL_PROMPT}" \
    --model_name_or_path ./checkpoints/${BASE_MODEL} \
    --per_device_batch_size 4 \
    --threads 16 \
    --dataset_name coco \
    --dataset_split test \
    --fp16 \
    --sparse_index sparse_output/${BASE_MODEL}/${DATASET}/${TARGET_TYPE}/${FILTER}/${EXPENDED_TOKENS}_${MANUAL}_${MANUAL_LENGTH} \
    --depth 10 \
    --save_dir output/${BASE_MODEL}/${DATASET}/${QUERY_TYPE}/${FILTER}/${EXPENDED_TOKENS}_${MANUAL}_${MANUAL_LENGTH}/results/ \
    --use_gpu \
    --remove_query \
    --query_type image \
    --batch_size 4 \
    --num_expended_tokens 0 \
