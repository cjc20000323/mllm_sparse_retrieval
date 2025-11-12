#!/bin/bash

BASE_MODEL=llava-hf-llava-v1.6-mistral-7b-hf
LORA_MODEL=llava-hf-llama3-llava-next-8b-hf_3epoch-lr4e-4-perbatch-8-32-flickr_lora_64_16_0_05_no_cutoff_and_pad_to_after_pad
GPUS=4
NUM_NODES=1
TARGET_TYPE=text
DATASET=flickr
FILTER=no_filter
EXPENDED_TOKENS=0
MANUAL=no_manual
MANUAL_LENGTH=20
CLUSTER=no_cluster
REPS_LOC=after_pad
EOL_TYPE=disassembleeol_separate
SPARSE_VALUE_TYPE=sum
SPARSE_LOWER_OR_UPPER=lower
CALCULATE_TYPE=concat
PROMPT_TYPE=prompt_5
USE_MEAN=no_mean
DATASET_SPLIT=test


deepspeed --num_gpus=$GPUS --num_nodes=$NUM_NODES src/rerank.py \
    --model_name_or_path ./checkpoints/${BASE_MODEL} \
    --per_device_batch_size 1 \
    --threads 16 \
    --dataset_name flickr \
    --dataset_split test \
    --fp16 \
    --passage_reps dense_output/${BASE_MODEL}/${DATASET}/${TARGET_TYPE}/${FILTER}/${CALCULATE_TYPE}/${PROMPT_TYPE}/test/${EXPENDED_TOKENS}_${MANUAL}_${MANUAL_LENGTH}_${SPARSE_VALUE_TYPE}_${CLUSTER}_${REPS_LOC}_${EOL_TYPE}_${SPARSE_LOWER_OR_UPPER}_${USE_MEAN} \
    --sparse_index sparse_output/${BASE_MODEL}/${DATASET}/${TARGET_TYPE}/${FILTER}/${CALCULATE_TYPE}/${PROMPT_TYPE}/test/${EXPENDED_TOKENS}_${MANUAL}_${MANUAL_LENGTH}_${SPARSE_VALUE_TYPE}_${CLUSTER}_${REPS_LOC}_${EOL_TYPE}_${SPARSE_LOWER_OR_UPPER}_${USE_MEAN} \
    --output_dir ./output/${BASE_MODEL}/${DATASET}/${TARGET_TYPE}/${FILTER}/${CALCULATE_TYPE}/${PROMPT_TYPE}/test/${EXPENDED_TOKENS}_${MANUAL}_${MANUAL_LENGTH}_${SPARSE_VALUE_TYPE}_${CLUSTER}_${REPS_LOC}_${EOL_TYPE}_${SPARSE_LOWER_OR_UPPER}_${USE_MEAN}/result \
    --depth 1000 \
    --remove_query \
    --query_type image \
    --use_gpu \
    --retrieval_batch_size 1 \
    --num_expended_tokens 0 \
    --alpha 0.9 \
    --beta 0.1 \
    --reps_loc 'after_pad' \
    --eol_type disassembleeol_separate \
    --sparse_value_type sum \
    --sparse_lower_or_upper lower \
    --sparse_length 20 \
    --calculate_type concat \
    --prompt_type prompt_5 \
    --rerank_type pointwise \
    --rerank_num 5 \
    --rerank_template caption_generation
