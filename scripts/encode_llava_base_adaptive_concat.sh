#!/bin/bash
set -e

# 从项目根目录运行；三个脚本的 ENCODE_TYPE 保持一致。
ENCODE_TYPE=text       # text：图检文；image：文检图
# 文本端和图像端均从全词表提取每个视角的Top-30。
USE_VIEW_PMI=true
VIEW_PMI_PRIOR_PROMPT_MODE=instruction_only
# true：按累计概率选择；false：固定Top-5。至少保留一个视角。
USE_VIEW_PROBABILITY_MASS="${USE_VIEW_PROBABILITY_MASS:-true}"
VIEW_PROBABILITY_MASS_THRESHOLD="${VIEW_PROBABILITY_MASS_THRESHOLD:-0.9}"
case "${USE_VIEW_PROBABILITY_MASS,,}" in
    true|1|yes) USE_VIEW_PROBABILITY_MASS=true ;;
    false|0|no) USE_VIEW_PROBABILITY_MASS=false ;;
    *) echo 'USE_VIEW_PROBABILITY_MASS must be true or false.' >&2; exit 1 ;;
esac

# 模式和阈值使用不同的选择文件；也可指定已有文件作为概率来源。
DEFAULT_VIEW_SELECTION_FILE=/root/gpufree-share/output/adaptive_concat/flickr_view_selection.json
if [[ "$USE_VIEW_PROBABILITY_MASS" == true ]]; then
    DEFAULT_VIEW_SELECTION_FILE="/root/gpufree-share/output/adaptive_concat/flickr_view_selection_pm_${VIEW_PROBABILITY_MASS_THRESHOLD}.json"
fi
VIEW_SELECTION_FILE="${VIEW_SELECTION_FILE:-$DEFAULT_VIEW_SELECTION_FILE}"
ARTIFACTS_FILE="/root/gpufree-share/output/adaptive_concat/flickr_${ENCODE_TYPE}.json"

GPUS=2
NUM_NODES=1

deepspeed --num_gpus=$GPUS --num_nodes=$NUM_NODES src/encode_adaptive_views.py \
    --model_name_or_path /root/gpufree-share/checkpoints/llava-hf-llava-v1.6-mistral-7b-hf \
    --output_dir /root/gpufree-share/output \
    --dense_output_dir /root/gpufree-share/dense_output \
    --sparse_output_dir /root/gpufree-share/sparse_output \
    --adaptive_views \
    --adaptive_data_root /root/gpufree-share \
    --adaptive_artifacts_file "$ARTIFACTS_FILE" \
    --view_selection_file "$VIEW_SELECTION_FILE" \
    --view_selection_modality both \
    --view_selection_split test \
    --view_selection_rounds 20 \
    --view_selection_seed 42 \
    --view_example_mode auto \
    --view_selection_pmi "$USE_VIEW_PMI" \
    --view_pmi_prior_prompt_mode "$VIEW_PMI_PRIOR_PROMPT_MODE" \
    --view_selection_hierarchical false \
    --view_probability_mass "$USE_VIEW_PROBABILITY_MASS" \
    --view_probability_mass_threshold "$VIEW_PROBABILITY_MASS_THRESHOLD" \
    --view_selection_token_variants false \
    --view_selection_prefix_tokens false \
    --view_selection_shuffle false \
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
    --eol_type disassembleeol_separate \
    --calculate_type concat \
    --prompt_type prompt_5 \
    --sparse_manual false \
    --sparse_type single \
    --sparse_length 30 \
    --text_sparse_length 30 \
    --image_sparse_length 30 \
    --sparse_value_type sum \
    --sparse_lower_or_upper lower
