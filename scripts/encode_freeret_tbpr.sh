#!/usr/bin/env bash
# 候选编码：使用普通 encode.py 固定视角；两个 FreeRet 开关默认关闭。
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
USE_FREERET_DENSE="${USE_FREERET_DENSE-false}"
USE_FREERET_FULL_PROMPT="${USE_FREERET_FULL_PROMPT-false}"  # Complete FreeRet dense/sparse prompts; requires FreeRet dense
USE_FREERET_PROMPT_CONSTRAINTS="${USE_FREERET_PROMPT_CONSTRAINTS-false}"
source "${SCRIPT_DIR}/freeret_task_common.sh"
if (($#)); then
    if (($# == 1)) && [[ "$1" == -h || "$1" == --help ]]; then freeret_help; exit 0; fi
    freeret_die 'Configure variables in this file or the environment; see --help.'
fi
TASK_GROUP=tbpr
# ===== 单模型、单数据集；在本文件或环境变量中修改配置 =====
MODEL_FAMILY="${MODEL_FAMILY-llava_mistral}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT-/root/autodl-fs/checkpoints}"
MODEL_PATH="${MODEL_PATH-}"
DATASET="${DATASET-CUHK-PEDES}"
DATASET_SPLIT="${DATASET_SPLIT-test}"
QUERY_TYPE="${QUERY_TYPE-}"             # IR=image/text；TBPR/WebQA=text；ReMuQ/LLaVA=image/composed
ENCODE_TYPE="${ENCODE_TYPE-}"           # 留空时按方向推导候选模态；显式值必须匹配
DENSE_OUTPUT_ROOT="${DENSE_OUTPUT_ROOT-./dense_output}"
SPARSE_OUTPUT_ROOT="${SPARSE_OUTPUT_ROOT-./sparse_output}"
GPUS="${GPUS-1}"
NUM_NODES="${NUM_NODES-1}"
DTYPE="${DTYPE-float16}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE-1}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"
DRY_RUN="${DRY_RUN-false}"               # true 只打印路径和命令
PROMPT_TYPE="${PROMPT_TYPE-prompt_5}"
RANDOM_VIEW_SEED="${RANDOM_VIEW_SEED-42}"
EOL_TYPE="${EOL_TYPE-disassembleeol_separate}"
CALCULATE_TYPE="${CALCULATE_TYPE-concat}"
SPARSE_MANUAL="${SPARSE_MANUAL-false}"
REPS_LOC="${REPS_LOC-after_pad}"
SPARSE_LENGTH="${SPARSE_LENGTH-40}"
TEXT_SPARSE_LENGTH="${TEXT_SPARSE_LENGTH-${SPARSE_LENGTH}}"
IMAGE_SPARSE_LENGTH="${IMAGE_SPARSE_LENGTH-${SPARSE_LENGTH}}"
SPARSE_VALUE_TYPE="${SPARSE_VALUE_TYPE-sum}"
SPARSE_LOWER_OR_UPPER="${SPARSE_LOWER_OR_UPPER-lower}"
SPARSE_VALUE_MEAN="${SPARSE_VALUE_MEAN-false}"
IS_FILTERED="${IS_FILTERED-false}"
NUM_EXPENDED_TOKENS="${NUM_EXPENDED_TOKENS-0}"
TBPR_TYPE="${TBPR_TYPE-origin_type}"
CIR_TYPE="${CIR_TYPE-type}"
LLAVA_IMAGE_MODE="${LLAVA_IMAGE_MODE-anyres}"
QWEN_MIN_PIXELS="${QWEN_MIN_PIXELS-0}"
QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS-0}"

# ===== 编码输出及实际启动命令 =====
OUTPUT_DIR="${OUTPUT_DIR-}"
[[ -z "${DENSE_CORPUS_DIR:-}" && -z "${SPARSE_CORPUS_DIR:-}" ]] || freeret_die 'Encode sets output roots; explicit corpus directories are for index/retrieval.'
freeret_configure
OUTPUT_DIR="$(dense_mode_path "${OUTPUT_DIR:-./output/freeret/${MODEL_TAG}/${DATASET}/${DATASET_SPLIT}/${ENCODE_TYPE}}")"
EXTRA=()
[[ "${SPARSE_VALUE_MEAN}" == false ]] || EXTRA+=(--sparse_value_mean)
[[ "${IS_FILTERED}" == false ]] || EXTRA+=(--is_filtered)
COMMAND=(
    "${DEEPSPEED_BIN}" --num_gpus="${GPUS}" --num_nodes="${NUM_NODES}" src/encode.py
    "${FREERET_ARGS[@]}"
    --model_name_or_path "${MODEL_PATH}" --dataset_name "${DATASET}"
    --task_type "${TASK_TYPE}" --encode_type "${ENCODE_TYPE}" --dataset_split "${DATASET_SPLIT}"
    --output_dir "${OUTPUT_DIR}" --dense_output_dir "${DENSE_OUTPUT_ROOT}" --sparse_output_dir "${SPARSE_OUTPUT_ROOT}"
    --per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" --reps_loc "${REPS_LOC}"
    --prompt_type "${PROMPT_TYPE}" --random_view_seed "${RANDOM_VIEW_SEED}"
    --eol_type "${EOL_TYPE}" --calculate_type "${CALCULATE_TYPE}" --sparse_manual false --sparse_type single
    --sparse_length "${SPARSE_LENGTH}" --text_sparse_length "${TEXT_SPARSE_LENGTH}" --image_sparse_length "${IMAGE_SPARSE_LENGTH}"
    --sparse_value_type "${SPARSE_VALUE_TYPE}" --sparse_lower_or_upper "${SPARSE_LOWER_OR_UPPER}"
    --num_expended_tokens "${NUM_EXPENDED_TOKENS}" --tbpr_type "${TBPR_TYPE}" --cir_type "${CIR_TYPE}"
    --llava_image_mode "${LLAVA_IMAGE_MODE}" --qwen_min_pixels "${QWEN_MIN_PIXELS}" --qwen_max_pixels "${QWEN_MAX_PIXELS}"
    "${PRECISION_ARGS[@]}" "${EXTRA[@]}"
)
freeret_report
printf 'Runtime output: %s\n' "${OUTPUT_DIR}"
[[ "${DRY_RUN}" != true ]] || exit 0
freeret_require_command "${DEEPSPEED_BIN}"
freeret_require_checkpoint "${MODEL_PATH}"
exec "${COMMAND[@]}"
