#!/usr/bin/env bash
# 稀疏索引：匹配普通 encode.py 产物；两个 FreeRet 开关默认关闭。
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

# ===== 索引输入：可覆盖为实验目录、collection 或 index =====
PYTHON_BIN="${PYTHON_BIN:-python}"
SPARSE_THREADS="${SPARSE_THREADS-16}"
SPARSE_CORPUS_DIR="${SPARSE_CORPUS_DIR-}"
freeret_configure
[[ "${SPARSE_THREADS}" =~ ^[1-9][0-9]*$ ]] || freeret_die 'SPARSE_THREADS must be positive.'
COMMAND=(
    "${PYTHON_BIN}" -B src/freeret_sparse_index.py build
    --family "${MODEL_FAMILY}" --sparse-root "${SPARSE_CORPUS_DIR}" --threads "${SPARSE_THREADS}"
    "${FREERET_ARGS[@]}"
)
freeret_report
[[ "${DRY_RUN}" != true ]] || exit 0
freeret_require_command "${PYTHON_BIN}"
exec "${COMMAND[@]}"
