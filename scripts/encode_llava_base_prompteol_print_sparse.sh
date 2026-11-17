#!/usr/bin/env bash
set -e

# 从项目根目录运行；PromptEOL 只有整条输入的一组输出，不使用自适应视角。
MODEL_PATH="${MODEL_PATH:-/root/autodl-fs/checkpoints/llava-hf-llava-v1.6-mistral-7b-hf}"
DATASET="${DATASET:-flickr}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
ENCODE_TYPE="${ENCODE_TYPE:-text}" # text：图搜文；image：文搜图。
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-fs/output/llava_base_prompteol_print_sparse}" # 根目录；自动追加 manual 模式和模态。
DENSE_OUTPUT_ROOT="${DENSE_OUTPUT_ROOT:-/root/autodl-fs/dense_output}"
SPARSE_OUTPUT_ROOT="${SPARSE_OUTPUT_ROOT:-/root/autodl-fs/sparse_output}"

# false：文本诊断从实际保留的原文 token 中取 Top-K；true 和图像编码显示全词表 Top-K。
SPARSE_MANUAL="${SPARSE_MANUAL:-false}"
SPARSE_LENGTH="${SPARSE_LENGTH:-30}"
TEXT_SPARSE_LENGTH="${TEXT_SPARSE_LENGTH:-$SPARSE_LENGTH}"
IMAGE_SPARSE_LENGTH="${IMAGE_SPARSE_LENGTH:-$SPARSE_LENGTH}"
PRINT_SPARSE="${PRINT_SPARSE:-true}"
PRINT_SPARSE_TOP_K="${PRINT_SPARSE_TOP_K:-10}"
PRINT_SPARSE_FORMAT="${PRINT_SPARSE_FORMAT:-text}"
PRINT_SPARSE_TIMING="${PRINT_SPARSE_TIMING:-end}" # end：成功结束后集中显示；immediate：即时显示。
# Flickr 目标句 sentid=89110 对应 imgid=17822、411175971.jpg；留空打印全部样本。
DEFAULT_PRINT_SPARSE_SAMPLE_ID=89110
if [[ "$DATASET" == flickr && "$ENCODE_TYPE" == image ]]; then
    DEFAULT_PRINT_SPARSE_SAMPLE_ID=17822
fi
PRINT_SPARSE_SAMPLE_ID="${PRINT_SPARSE_SAMPLE_ID-${DEFAULT_PRINT_SPARSE_SAMPLE_ID}}"
for name in PRINT_SPARSE SPARSE_MANUAL; do
    case "${!name,,}" in
        true|1|yes) printf -v "$name" '%s' true ;;
        false|0|no) printf -v "$name" '%s' false ;;
        *) printf '%s must be true or false.\n' "$name" >&2; exit 2 ;;
    esac
done
case "$ENCODE_TYPE" in text|image) ;; *) echo 'ENCODE_TYPE must be text or image.' >&2; exit 2 ;; esac
case "$PRINT_SPARSE_FORMAT" in text|json) ;; *) echo 'PRINT_SPARSE_FORMAT must be text or json.' >&2; exit 2 ;; esac
case "$PRINT_SPARSE_TIMING" in immediate|end) ;; *) echo 'PRINT_SPARSE_TIMING must be immediate or end.' >&2; exit 2 ;; esac
if [[ "$PRINT_SPARSE" == true && ! "$PRINT_SPARSE_TOP_K" =~ ^0*[1-9][0-9]*$ ]]; then
    echo 'PRINT_SPARSE_TOP_K must be a positive integer.' >&2
    exit 2
fi

# 四种组合分开保存；自定义 PRINT_SPARSE_LOG_DIR 也作为根目录追加同样的子目录。
RESULT_SUBDIR="sparse_manual_${SPARSE_MANUAL}/${ENCODE_TYPE}"
OUTPUT_DIR="${OUTPUT_DIR%/}/${RESULT_SUBDIR}"
if [[ -n "${PRINT_SPARSE_LOG_DIR:-}" ]]; then
    PRINT_SPARSE_LOG_DIR="${PRINT_SPARSE_LOG_DIR%/}/${RESULT_SUBDIR}"
else
    PRINT_SPARSE_LOG_DIR="${OUTPUT_DIR}/sparse_token_logs"
fi
GPUS="${GPUS:-2}"
NUM_NODES="${NUM_NODES:-1}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"

printf 'Output directory: %s\nSparse token logs: %s\n' "$OUTPUT_DIR" "$PRINT_SPARSE_LOG_DIR"
"$DEEPSPEED_BIN" --num_gpus="$GPUS" --num_nodes="$NUM_NODES" src/encode.py \
    --model_name_or_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --dense_output_dir "$DENSE_OUTPUT_ROOT" \
    --sparse_output_dir "$SPARSE_OUTPUT_ROOT" \
    --llava_image_mode base \
    --use_freeret_dense false \
    --use_freeret_full_prompt false \
    --use_freeret_prompt_constraints false \
    --fp16 \
    --dataset_name "$DATASET" \
    --dataset_split "$DATASET_SPLIT" \
    --task_type ir \
    --encode_type "$ENCODE_TYPE" \
    --per_device_batch_size 1 \
    --reps_loc after_pad \
    --num_expended_tokens 0 \
    --eol_type prompteol \
    --calculate_type separate \
    --prompt_type prompt_5 \
    --sparse_manual "$SPARSE_MANUAL" \
    --sparse_length "$SPARSE_LENGTH" \
    --text_sparse_length "$TEXT_SPARSE_LENGTH" \
    --image_sparse_length "$IMAGE_SPARSE_LENGTH" \
    --sparse_value_type sum \
    --sparse_lower_or_upper lower \
    --print_sparse_diagnostics true \
    --diagnostic_data_root "$DATA_ROOT" \
    --print_sparse "$PRINT_SPARSE" \
    --print_sparse_top_k "$PRINT_SPARSE_TOP_K" \
    --print_sparse_format "$PRINT_SPARSE_FORMAT" \
    --print_sparse_timing "$PRINT_SPARSE_TIMING" \
    --print_sparse_sample_id "$PRINT_SPARSE_SAMPLE_ID" \
    --print_sparse_log_dir "$PRINT_SPARSE_LOG_DIR"
