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
# Set USE_FIXED_ALPHA=true to use ALPHA directly; false keeps existing selection logic.
ALPHA="${ALPHA:-0.9}"
BETA="${BETA:-0.1}"
USE_FIXED_ALPHA="${USE_FIXED_ALPHA:-false}"
USE_GENERATED_VALIDATION_DATA="${USE_GENERATED_VALIDATION_DATA:-false}"
GENERATED_VALIDATION_DATA_ROOT="${GENERATED_VALIDATION_DATA_ROOT:-./generated_llava70b_data}"
# Complete retrieval, then stop reranking once five >=5s rates after 20
# warmup queries have CV <5%. Save seconds under --output_dir/time_estimate.json.
# Example: ESTIMATE_TIME_ONLY=true bash scripts/rerank.sh
ESTIMATE_TIME_ONLY="${ESTIMATE_TIME_ONLY:-false}"
# Candidates per model forward, per GPU process; not the query retrieval batch.
RERANK_BATCH_SIZE="${RERANK_BATCH_SIZE:-1}"
# Preserve existing templates and scores; the cache only replaces repeated
# text-only prior forwards. Requires MODIFY_TYPE=modify_single.
MODIFY_TYPE="${MODIFY_TYPE:-no}"
RERANK_TEMPLATE="${RERANK_TEMPLATE:-caption_generation}"
USE_RERANK_PRIOR_CACHE="${USE_RERANK_PRIOR_CACHE:-false}"
RERANK_PRIOR_CACHE_DIR="${RERANK_PRIOR_CACHE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/cache/rerank_prior}"
PRIOR_CACHE_ARGS=()
case "${USE_RERANK_PRIOR_CACHE,,}" in
    true|1) PRIOR_CACHE_ARGS+=(--use_rerank_prior_cache --rerank_prior_cache_dir "${RERANK_PRIOR_CACHE_DIR}") ;;
    false|0) ;;
    *) echo "USE_RERANK_PRIOR_CACHE must be true/false or 1/0." >&2; exit 2 ;;
esac
TIME_ESTIMATE_ARGS=()
case "${ESTIMATE_TIME_ONLY,,}" in
    true|1) TIME_ESTIMATE_ARGS+=(--estimate_time_only) ;;
    false|0) ;;
    *) echo "ESTIMATE_TIME_ONLY must be true/false or 1/0." >&2; exit 2 ;;
esac

FUSION_ALPHA_ARGS=()
if [[ "${USE_FIXED_ALPHA}" == "true" ]]; then
    FUSION_ALPHA_ARGS+=(--use_fixed_alpha)
fi

GENERATED_VALIDATION_ARGS=()
if [[ "${USE_GENERATED_VALIDATION_DATA}" == "true" ]]; then
    GENERATED_VALIDATION_ARGS+=(--use_generated_validation_data)
    GENERATED_VALIDATION_ARGS+=(--generated_validation_data_root "${GENERATED_VALIDATION_DATA_ROOT}")
fi


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
    --alpha "${ALPHA}" \
    --beta "${BETA}" \
    "${FUSION_ALPHA_ARGS[@]}" \
    --reps_loc 'after_pad' \
    --eol_type disassembleeol_separate \
    --sparse_value_type sum \
    --sparse_lower_or_upper lower \
    --sparse_length 20 \
    --calculate_type concat \
    --prompt_type prompt_5 \
    --rerank_type pointwise \
    --rerank_num 5 \
    --rerank_batch_size "${RERANK_BATCH_SIZE}" \
    --rerank_template "${RERANK_TEMPLATE}" \
    --modify_type "${MODIFY_TYPE}" \
    "${PRIOR_CACHE_ARGS[@]}" \
    "${TIME_ESTIMATE_ARGS[@]}" \
    "${GENERATED_VALIDATION_ARGS[@]}"
