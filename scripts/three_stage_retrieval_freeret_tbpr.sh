#!/usr/bin/env bash
# 三阶段检索：使用普通固定视角产物；两个 FreeRet 开关默认关闭。
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

# ===== 三阶段资源、路径、融合与重排 =====
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT-/root/autodl-tmp}"   # 与 src/arguments.py 中编码使用的数据位置一致
RANKER_MODEL_PATH="${RANKER_MODEL_PATH-}"
DENSE_CORPUS_DIR="${DENSE_CORPUS_DIR-}"
SPARSE_CORPUS_DIR="${SPARSE_CORPUS_DIR-}"
STAGE1_SPARSE_INDEX="${STAGE1_SPARSE_INDEX-}"
STAGE2_DENSE_INDEX="${STAGE2_DENSE_INDEX-}"
STAGE2_SPARSE_INDEX="${STAGE2_SPARSE_INDEX-}"
OUTPUT_DIR="${OUTPUT_DIR-}"
STAGE1_DEPTH="${STAGE1_DEPTH-1000}"
STAGE2_DEPTH="${STAGE2_DEPTH-1000}"
SPARSE_THREADS="${SPARSE_THREADS-16}"
HYBRID_ALPHA="${HYBRID_ALPHA-0.5}"
SELECT_HYBRID_ALPHA="${SELECT_HYBRID_ALPHA-false}"
VAL_PASSAGE_REPS="${VAL_PASSAGE_REPS-}"
VAL_SPARSE_INDEX="${VAL_SPARSE_INDEX-}"
VAL_STAGE1_SPARSE_INDEX="${VAL_STAGE1_SPARSE_INDEX-}"
VALIDATION_DATASET_NAME="${VALIDATION_DATASET_NAME-}"
VALIDATION_SPLIT="${VALIDATION_SPLIT-val}"
RETRIEVAL_DEVICE="${RETRIEVAL_DEVICE-auto}"
RETRIEVAL_DEVICE_MAP="${RETRIEVAL_DEVICE_MAP-auto}"
RETRIEVAL_DTYPE="${RETRIEVAL_DTYPE-${DTYPE}}"
RETRIEVAL_BATCH_SIZE="${RETRIEVAL_BATCH_SIZE-${PER_DEVICE_BATCH_SIZE}}"
RETRIEVAL_LLAVA_IMAGE_MODE="${RETRIEVAL_LLAVA_IMAGE_MODE-${LLAVA_IMAGE_MODE}}"
RANKER_DTYPE="${RANKER_DTYPE-${DTYPE}}"
RANKER_DEVICE_MAP="${RANKER_DEVICE_MAP-auto}"
RANKER_LLAVA_IMAGE_MODE="${RANKER_LLAVA_IMAGE_MODE-${LLAVA_IMAGE_MODE}}"
RANKER_QWEN_MIN_PIXELS="${RANKER_QWEN_MIN_PIXELS-${QWEN_MIN_PIXELS}}"
RANKER_QWEN_MAX_PIXELS="${RANKER_QWEN_MAX_PIXELS-${QWEN_MAX_PIXELS}}"
RERANK_TEMPLATE="${RERANK_TEMPLATE-caption_generation}"
MODIFY_TYPE="${MODIFY_TYPE-no}"
RERANK_NUM="${RERANK_NUM-5}"
RERANK_BATCH_SIZE="${RERANK_BATCH_SIZE-1}"
RERANK_REPS_LOC="${RERANK_REPS_LOC-after_pad}"
USE_RERANK_PRIOR_CACHE="${USE_RERANK_PRIOR_CACHE-false}"
RERANK_PRIOR_CACHE_DIR="${RERANK_PRIOR_CACHE_DIR-./cache/rerank_prior}"
OVERWRITE="${OVERWRITE-false}"
freeret_configure
RANKER_MODEL_PATH="${RANKER_MODEL_PATH:-${MODEL_PATH}}"
[[ "${RETRIEVAL_LLAVA_IMAGE_MODE}" == "${LLAVA_IMAGE_MODE}" ]] || freeret_die 'Retrieval image mode must match the encoded corpus setting LLAVA_IMAGE_MODE.'
for name in RETRIEVAL_DTYPE RANKER_DTYPE; do
    case "${!name}" in float16|bfloat16|float32) ;; *) freeret_die "Invalid ${name}." ;; esac
done
case "${RANKER_LLAVA_IMAGE_MODE}" in anyres|base) ;; *) freeret_die 'Invalid RANKER_LLAVA_IMAGE_MODE.' ;; esac
[[ "${RANKER_LLAVA_IMAGE_MODE}" != base || "${MODEL_FAMILY}" == llava_* ]] || freeret_die 'Ranker base mode requires LLaVA-NeXT.'
case "${MODIFY_TYPE}" in no|modify_single) ;; *) freeret_die 'Invalid MODIFY_TYPE.' ;; esac
[[ "${HYBRID_ALPHA}" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]] || freeret_die 'HYBRID_ALPHA must be in [0,1].'
for name in STAGE1_DEPTH STAGE2_DEPTH SPARSE_THREADS RETRIEVAL_BATCH_SIZE RERANK_NUM RERANK_BATCH_SIZE; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || freeret_die "${name} must be positive."
done
for name in SELECT_HYBRID_ALPHA USE_RERANK_PRIOR_CACHE OVERWRITE; do freeret_bool "${name}"; done
STAGE1_SPARSE_INDEX="$(dense_mode_path "${STAGE1_SPARSE_INDEX:-${DEFAULT_SPARSE_INDEX}}")"
STAGE2_DENSE_INDEX="$(dense_mode_path "${STAGE2_DENSE_INDEX:-${DENSE_CORPUS_DIR}}")"
STAGE2_SPARSE_INDEX="$(dense_mode_path "${STAGE2_SPARSE_INDEX:-${DEFAULT_SPARSE_INDEX}}")"
OUTPUT_VARIANT=sparse_concat_stage1
[[ "${TASK_TYPE}" != tbpr ]] || OUTPUT_VARIANT+="/${TBPR_TYPE}"
[[ "${SELECT_HYBRID_ALPHA}" != true ]] || OUTPUT_VARIANT+=/alpha_select
if [[ "${RETRIEVAL_LLAVA_IMAGE_MODE}" == base || "${RANKER_LLAVA_IMAGE_MODE}" == base ]]; then
    OUTPUT_VARIANT+="_retrieval_${RETRIEVAL_LLAVA_IMAGE_MODE}_ranker_${RANKER_LLAVA_IMAGE_MODE}"
fi
OUTPUT_DIR="$(dense_mode_path "${OUTPUT_DIR:-./three_stage_results/${MODEL_TAG}/${DATASET}/${DIRECTION_LABEL}/${DATASET_SPLIT}/${OUTPUT_VARIANT}/stage1_${STAGE1_DEPTH}_stage2_${STAGE2_DEPTH}}")"
EXTRA=()
CHECK_ARGS=(--dense-root "${STAGE2_DENSE_INDEX}" --sparse-root "${STAGE1_SPARSE_INDEX}" --sparse-root "${STAGE2_SPARSE_INDEX}")
[[ "${SPARSE_VALUE_MEAN}" == false ]] || EXTRA+=(--sparse-value-mean)
[[ "${IS_FILTERED}" == false ]] || EXTRA+=(--is-filtered)
[[ "${OVERWRITE}" == false ]] || EXTRA+=(--overwrite)
if [[ "${SELECT_HYBRID_ALPHA}" == true ]]; then
    EXTRA+=(--select-hybrid-alpha --validation-split "${VALIDATION_SPLIT}")
    [[ -z "${VALIDATION_DATASET_NAME}" ]] || EXTRA+=(--validation-dataset-name "${VALIDATION_DATASET_NAME}")
else
    [[ -z "${VAL_PASSAGE_REPS}${VAL_SPARSE_INDEX}${VAL_STAGE1_SPARSE_INDEX}${VALIDATION_DATASET_NAME}" ]] || freeret_die 'Validation inputs require SELECT_HYBRID_ALPHA=true.'
fi
[[ -n "${VAL_PASSAGE_REPS}" && -n "${VAL_SPARSE_INDEX}" || -z "${VAL_PASSAGE_REPS}${VAL_SPARSE_INDEX}" ]] || freeret_die 'Provide both VAL_PASSAGE_REPS and VAL_SPARSE_INDEX.'
if [[ -n "${VAL_PASSAGE_REPS}" ]]; then
    VAL_PASSAGE_REPS="$(dense_mode_path "${VAL_PASSAGE_REPS}")"
    VAL_SPARSE_INDEX="$(dense_mode_path "${VAL_SPARSE_INDEX}")"
    EXTRA+=(--val-passage-reps "${VAL_PASSAGE_REPS}" --val-sparse-index "${VAL_SPARSE_INDEX}")
    CHECK_ARGS+=(--dense-root "${VAL_PASSAGE_REPS}" --sparse-root "${VAL_SPARSE_INDEX}")
fi
if [[ -n "${VAL_STAGE1_SPARSE_INDEX}" ]]; then
    [[ -n "${VAL_PASSAGE_REPS}" ]] || freeret_die 'VAL_STAGE1_SPARSE_INDEX requires validation Stage-2 inputs.'
    VAL_STAGE1_SPARSE_INDEX="$(dense_mode_path "${VAL_STAGE1_SPARSE_INDEX}")"
    EXTRA+=(--val-stage1-sparse-index "${VAL_STAGE1_SPARSE_INDEX}")
    CHECK_ARGS+=(--sparse-root "${VAL_STAGE1_SPARSE_INDEX}")
fi
[[ "${USE_RERANK_PRIOR_CACHE}" == false ]] || EXTRA+=(--use-rerank-prior-cache --rerank-prior-cache-dir "${RERANK_PRIOR_CACHE_DIR}")
COMMAND=(
    "${DEEPSPEED_BIN}" --num_gpus="${GPUS}" --num_nodes="${NUM_NODES}" src/three_stage_retrieval_pipeline.py
    "${FREERET_ARGS[@]}"
    --dataset-name "${DATASET}" --dataset-split "${DATASET_SPLIT}" --task-type "${TASK_TYPE}" --query-type "${QUERY_TYPE}" --data-root "${DATA_ROOT}"
    --stage1-method sparse_concat --stage1-sparse-index "${STAGE1_SPARSE_INDEX}" --stage1-depth "${STAGE1_DEPTH}"
    --stage2-dense-index "${STAGE2_DENSE_INDEX}" --stage2-sparse-index "${STAGE2_SPARSE_INDEX}" --stage2-depth "${STAGE2_DEPTH}"
    --hybrid-alpha "${HYBRID_ALPHA}" --sparse-threads "${SPARSE_THREADS}"
    --retrieval-model-name-or-path "${MODEL_PATH}" --retrieval-device "${RETRIEVAL_DEVICE}" --retrieval-device-map "${RETRIEVAL_DEVICE_MAP}"
    --retrieval-dtype "${RETRIEVAL_DTYPE}" --retrieval-batch-size "${RETRIEVAL_BATCH_SIZE}" --retrieval-reps-loc "${REPS_LOC}"
    --retrieval-llava-image-mode "${RETRIEVAL_LLAVA_IMAGE_MODE}" --retrieval-qwen-min-pixels "${QWEN_MIN_PIXELS}" --retrieval-qwen-max-pixels "${QWEN_MAX_PIXELS}"
    --prompt-type "${PROMPT_TYPE}" --random-view-seed "${RANDOM_VIEW_SEED}"
    --sparse-length "${SPARSE_LENGTH}" --text-sparse-length "${TEXT_SPARSE_LENGTH}" --image-sparse-length "${IMAGE_SPARSE_LENGTH}"
    --sparse-value-type "${SPARSE_VALUE_TYPE}" --sparse-lower-or-upper "${SPARSE_LOWER_OR_UPPER}" --num-expended-tokens "${NUM_EXPENDED_TOKENS}"
    --tbpr-type "${TBPR_TYPE}" --cir-type "${CIR_TYPE}"
    --ranker-model-name-or-path "${RANKER_MODEL_PATH}" --ranker-dtype "${RANKER_DTYPE}" --ranker-device-map "${RANKER_DEVICE_MAP}"
    --ranker-llava-image-mode "${RANKER_LLAVA_IMAGE_MODE}" --ranker-qwen-min-pixels "${RANKER_QWEN_MIN_PIXELS}" --ranker-qwen-max-pixels "${RANKER_QWEN_MAX_PIXELS}"
    --rerank-type pointwise --rerank-template "${RERANK_TEMPLATE}" --modify-type "${MODIFY_TYPE}"
    --rerank-num "${RERANK_NUM}" --rerank-batch-size "${RERANK_BATCH_SIZE}" --rerank-reps-loc "${RERANK_REPS_LOC}"
    --output-dir "${OUTPUT_DIR}" "${EXTRA[@]}"
)
freeret_report
printf 'Stage 1 index: %s\nStage 2 dense: %s\nStage 2 sparse: %s\nResults: %s\n' "${STAGE1_SPARSE_INDEX}" "${STAGE2_DENSE_INDEX}" "${STAGE2_SPARSE_INDEX}" "${OUTPUT_DIR}"
[[ "${DRY_RUN}" != true ]] || exit 0
freeret_require_command "${PYTHON_BIN}"
freeret_require_command "${DEEPSPEED_BIN}"
freeret_require_checkpoint "${MODEL_PATH}"
freeret_require_checkpoint "${RANKER_MODEL_PATH}"
freeret_require_data
"${PYTHON_BIN}" -B src/freeret_sparse_index.py check --family "${MODEL_FAMILY}" "${FREERET_ARGS[@]}" "${CHECK_ARGS[@]}"
exec "${COMMAND[@]}"
