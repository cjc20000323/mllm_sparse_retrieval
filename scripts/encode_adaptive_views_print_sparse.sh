#!/usr/bin/env bash
set -e

# 从项目根目录运行。text 编码文本候选（图检文）；image 编码图像候选（文检图）。
MODEL_PATH="${MODEL_PATH:-/root/autodl-fs/checkpoints/llava-hf-llama3-llava-next-8b-hf}"
DATASET="${DATASET:-flickr}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
TASK_TYPE="${TASK_TYPE:-ir}"
ENCODE_TYPE="${ENCODE_TYPE:-text}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-fs/output/adaptive_print_sparse}"
DENSE_OUTPUT_ROOT="${DENSE_OUTPUT_ROOT:-/root/autodl-fs/dense_output}"
SPARSE_OUTPUT_ROOT="${SPARSE_OUTPUT_ROOT:-/root/autodl-fs/sparse_output}"

# true：所有 GPU 打印各自样本的已选视角；只影响日志，实际编码仍保留每个视角 Top-30。
PRINT_SPARSE="${PRINT_SPARSE:-true}"
PRINT_SPARSE_TOP_K="${PRINT_SPARSE_TOP_K:-10}"
PRINT_SPARSE_FORMAT="${PRINT_SPARSE_FORMAT:-text}" # text：易读文本；json：原有结构化输出。
PRINT_SPARSE_TIMING="${PRINT_SPARSE_TIMING:-end}" # end：成功结束后集中显示；immediate：命中时立即显示。
# 仍全量编码；Flickr 目标句 sentid=89110 对应 imgid=17822、411175971.jpg。
DEFAULT_PRINT_SPARSE_SAMPLE_ID=89110
if [[ "$DATASET" == flickr && "$ENCODE_TYPE" == image ]]; then
    DEFAULT_PRINT_SPARSE_SAMPLE_ID=17822
fi
# 显式设置可覆盖默认 ID；留空则显示所有样本。
PRINT_SPARSE_SAMPLE_ID="${PRINT_SPARSE_SAMPLE_ID-${DEFAULT_PRINT_SPARSE_SAMPLE_ID}}"
# 全部样本（包括目标文本或图像）的逐视角 token 按 GPU 分文件保存。
PRINT_SPARSE_LOG_DIR="${PRINT_SPARSE_LOG_DIR:-${OUTPUT_DIR}/sparse_token_logs}"
case "${PRINT_SPARSE,,}" in
    true|1|yes) PRINT_SPARSE=true ;;
    false|0|no) PRINT_SPARSE=false ;;
    *) echo 'PRINT_SPARSE must be true or false.' >&2; exit 2 ;;
esac
case "$PRINT_SPARSE_FORMAT" in
    text|json) ;;
    *) echo 'PRINT_SPARSE_FORMAT must be text or json.' >&2; exit 2 ;;
esac
case "$PRINT_SPARSE_TIMING" in
    immediate|end) ;;
    *) echo 'PRINT_SPARSE_TIMING must be immediate or end.' >&2; exit 2 ;;
esac
if [[ "$PRINT_SPARSE" == true && ! "$PRINT_SPARSE_TOP_K" =~ ^0*[1-9][0-9]*$ ]]; then
    echo 'PRINT_SPARSE_TOP_K must be a positive integer.' >&2
    exit 2
fi

# 默认使用累计概率选择 A-J 视角；选择配置必须与已有编码实验一致。
USE_VIEW_PROBABILITY_MASS="${USE_VIEW_PROBABILITY_MASS:-true}"
VIEW_PROBABILITY_MASS_THRESHOLD="${VIEW_PROBABILITY_MASS_THRESHOLD:-0.9}"
USE_VIEW_PMI="${USE_VIEW_PMI:-false}"
VIEW_PMI_PRIOR_PROMPT_MODE="${VIEW_PMI_PRIOR_PROMPT_MODE:-false}"
VIEW_SELECTION_FILE="${VIEW_SELECTION_FILE:-}"  # 留空使用自动缓存，也可指定已有选择文件。
GPUS="${GPUS:-2}"
NUM_NODES="${NUM_NODES:-1}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"

"$DEEPSPEED_BIN" --num_gpus="$GPUS" --num_nodes="$NUM_NODES" src/encode_adaptive_views.py \
    --model_name_or_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --dense_output_dir "$DENSE_OUTPUT_ROOT" \
    --sparse_output_dir "$SPARSE_OUTPUT_ROOT" \
    --adaptive_views \
    --adaptive_data_root "$DATA_ROOT" \
    --view_selection_file "$VIEW_SELECTION_FILE" \
    --view_selection_modality both \
    --view_selection_split "$DATASET_SPLIT" \
    --view_selection_rounds 20 \
    --view_selection_seed 42 \
    --view_example_mode auto \
    --view_example_max_tokens 0 \
    --view_selection_hierarchical true \
    --view_selection_pmi "$USE_VIEW_PMI" \
    --view_pmi_prior_prompt_mode "$VIEW_PMI_PRIOR_PROMPT_MODE" \
    --view_probability_mass "$USE_VIEW_PROBABILITY_MASS" \
    --view_probability_mass_threshold "$VIEW_PROBABILITY_MASS_THRESHOLD" \
    --view_selection_token_variants false \
    --view_selection_prefix_tokens false \
    --view_selection_shuffle false \
    --use_stc_sparse false \
    --use_lens_sparse false \
    --use_freeret_dense false \
    --use_freeret_full_prompt false \
    --use_freeret_prompt_constraints false \
    --llava_image_mode base \
    --fp16 \
    --dataset_name "$DATASET" \
    --dataset_split "$DATASET_SPLIT" \
    --task_type "$TASK_TYPE" \
    --encode_type "$ENCODE_TYPE" \
    --per_device_batch_size 1 \
    --reps_loc after_pad \
    --random_view_seed 42 \
    --num_expended_tokens 0 \
    --eol_type disassembleeol_separate \
    --calculate_type concat \
    --prompt_type prompt_5 \
    --tbpr_type type \
    --cir_type type \
    --sparse_manual false \
    --sparse_type single \
    --sparse_length 30 \
    --text_sparse_length 30 \
    --image_sparse_length 30 \
    --sparse_value_type sum \
    --sparse_lower_or_upper lower \
    --print_sparse "$PRINT_SPARSE" \
    --print_sparse_top_k "$PRINT_SPARSE_TOP_K" \
    --print_sparse_format "$PRINT_SPARSE_FORMAT" \
    --print_sparse_timing "$PRINT_SPARSE_TIMING" \
    --print_sparse_sample_id "$PRINT_SPARSE_SAMPLE_ID" \
    --print_sparse_log_dir "$PRINT_SPARSE_LOG_DIR"
