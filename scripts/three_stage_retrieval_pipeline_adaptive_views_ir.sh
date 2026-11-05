#!/usr/bin/env bash
# auto 按实际上下文长度填充示例；fixed 使用 VIEW_TEXTS_PER_ROUND / VIEW_IMAGES_PER_ROUND。
VIEW_EXAMPLE_MODE="${VIEW_EXAMPLE_MODE-auto}"
VIEW_EXAMPLE_MAX_TOKENS="${VIEW_EXAMPLE_MAX_TOKENS-0}" # 0 使用模型上限；可下调，总预算包含一个回答 token
USE_HIERARCHICAL_VIEWS="${USE_HIERARCHICAL_VIEWS-false}"  # true: ten primary/secondary dimensions (A-J); false: original A-T
case "${USE_HIERARCHICAL_VIEWS,,}" in
    true|1|yes) USE_HIERARCHICAL_VIEWS=true ;;
    false|0|no) USE_HIERARCHICAL_VIEWS=false ;;
    *) echo 'Invalid USE_HIERARCHICAL_VIEWS: use true or false' >&2; exit 2 ;;
esac
VIEW_SELECTION_RUN_ID="${VIEW_SELECTION_RUN_ID-}"  # Shuffle experiment identifier printed by encoding
USE_FREERET_DENSE="${USE_FREERET_DENSE-false}"  # true: FreeRet pre-MLP dense; prompts/sparse unchanged
USE_FREERET_FULL_PROMPT="${USE_FREERET_FULL_PROMPT-false}"  # Complete FreeRet dense/sparse prompts; requires FreeRet dense
USE_FREERET_PROMPT_CONSTRAINTS="${USE_FREERET_PROMPT_CONSTRAINTS-false}"  # Prepend FreeRet constraints to all representation views
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/freeret_dense_mode.sh" || exit $?
# Adaptive 三阶段批量检索：全部配置及实际 DeepSpeed 命令均在本文件中展开。
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="${REPO_ROOT}/scripts/${BASH_SOURCE[0]##*/}"
cd "${REPO_ROOT}"
RUN_ONE=false
if (($# == 1)) && [[ "$1" == --run-one ]]; then
    RUN_ONE=true
elif (($#)) && ! { (($# == 1)) && [[ "$1" == -h || "$1" == --help ]]; }; then
    echo 'Configure environment variables or the configuration block; see --help.' >&2
    exit 2
fi

# ===== 模型、数据集与查询方向 =====
# 非空单值覆盖优先；默认留空，以保留多模型、多数据集范围。
MODEL_FAMILIES="${MODEL_FAMILIES-qwen2 qwen25 qwen3 llava_llama llava_mistral}"
MODEL_FAMILY="${MODEL_FAMILY-}"
DATASETS="${DATASETS-flickr coco}"
DATASET="${DATASET-}"
IR_QUERY_TYPES="${IR_QUERY_TYPES-image text}"  # image=图搜文；text=文搜图
IR_QUERY_TYPE="${IR_QUERY_TYPE-}"
DATASET_SPLIT="${DATASET_SPLIT-test}"

# ===== 检索模型：五种基座的检查点目录逐项列出 =====
CHECKPOINT_ROOT="${CHECKPOINT_ROOT-/root/autodl-fs/checkpoints}"
MODEL_PATH_QWEN2="${MODEL_PATH_QWEN2-${CHECKPOINT_ROOT}/Qwen-Qwen2-VL-7B-Instruct}"
MODEL_PATH_QWEN25="${MODEL_PATH_QWEN25-${CHECKPOINT_ROOT}/Qwen-Qwen2.5-VL-7B-Instruct}"
MODEL_PATH_QWEN3="${MODEL_PATH_QWEN3-${CHECKPOINT_ROOT}/Qwen-Qwen3-VL-8B-Instruct}"
MODEL_PATH_LLAVA_LLAMA="${MODEL_PATH_LLAVA_LLAMA-${CHECKPOINT_ROOT}/llava-hf-llama3-llava-next-8b-hf}"
MODEL_PATH_LLAVA_MISTRAL="${MODEL_PATH_LLAVA_MISTRAL-${CHECKPOINT_ROOT}/llava-hf-llava-v1.6-mistral-7b-hf}"
MODEL_PATH="${MODEL_PATH-}"  # 非空仅允许一个模型族

# ===== 重排模型：留空时跟随当前任务的检索模型 =====
RANKER_MODEL_PATH_QWEN2="${RANKER_MODEL_PATH_QWEN2-}"
RANKER_MODEL_PATH_QWEN25="${RANKER_MODEL_PATH_QWEN25-}"
RANKER_MODEL_PATH_QWEN3="${RANKER_MODEL_PATH_QWEN3-}"
RANKER_MODEL_PATH_LLAVA_LLAMA="${RANKER_MODEL_PATH_LLAVA_LLAMA-}"
RANKER_MODEL_PATH_LLAVA_MISTRAL="${RANKER_MODEL_PATH_LLAVA_MISTRAL-}"
RANKER_MODEL_PATH="${RANKER_MODEL_PATH-}"  # 非空仅允许一个模型族

# ===== 数据、编码产物和检索结果目录 =====
DATA_ROOT="${DATA_ROOT-/root/autodl-tmp}"
DENSE_OUTPUT_ROOT="${DENSE_OUTPUT_ROOT-./dense_output}"
SPARSE_OUTPUT_ROOT="${SPARSE_OUTPUT_ROOT-./sparse_output}"
ADAPTIVE_OUTPUT_ROOT="${ADAPTIVE_OUTPUT_ROOT-./three_stage_adaptive_results}"
OUTPUT_DIR="${OUTPUT_DIR-}"  # 非空仅允许一个检索任务
# 调度器按下方配置精确定位每个任务的 VIEW_SELECTION_FILE、DENSE_CORPUS_DIR、
# SPARSE_CORPUS_DIR、STAGE1_SPARSE_INDEX；这些路径无需手动逐一填写。

# ===== 运行资源与程序 =====
GPUS="${GPUS-1}"
NUM_NODES="${NUM_NODES-1}"
DTYPE="${DTYPE-float16}"  # float16 / bfloat16 / float32
RETRIEVAL_DTYPE="${RETRIEVAL_DTYPE-${DTYPE}}"  # 必须与编码精度一致
RANKER_DTYPE="${RANKER_DTYPE-${DTYPE}}"
DRY_RUN="${DRY_RUN-false}"  # true 仅预检并打印任务，不加载模型权重或写结果
OVERWRITE="${OVERWRITE-false}"  # true 允许重新运行并覆写已有结果
PYTHON_BIN="${PYTHON_BIN:-python}"
BASH_BIN="${BASH_BIN:-bash}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"

# ===== 稀疏配置：必须与 adaptive 编码一致 =====
REPS_LOC="${REPS_LOC-after_pad}"  # before_pad / after_pad
SPARSE_LENGTH="${SPARSE_LENGTH-40}"
TEXT_SPARSE_LENGTH="${TEXT_SPARSE_LENGTH-${SPARSE_LENGTH}}"
IMAGE_SPARSE_LENGTH="${IMAGE_SPARSE_LENGTH-${SPARSE_LENGTH}}"
SPARSE_VALUE_TYPE="${SPARSE_VALUE_TYPE-sum}"  # replace / sum / max
SPARSE_LOWER_OR_UPPER="${SPARSE_LOWER_OR_UPPER-lower}"  # lower / upper
SPARSE_VALUE_MEAN="${SPARSE_VALUE_MEAN-false}"
IS_FILTERED="${IS_FILTERED-false}"
NUM_EXPENDED_TOKENS="${NUM_EXPENDED_TOKENS-0}"
TBPR_TYPE="${TBPR_TYPE-origin_type}"  # origin_type / type

# ===== 图像处理 =====
LLAVA_IMAGE_MODE="${LLAVA_IMAGE_MODE-anyres}"  # anyres / base；base 仅支持 LLaVA-NeXT
RETRIEVAL_LLAVA_IMAGE_MODE="${RETRIEVAL_LLAVA_IMAGE_MODE-${LLAVA_IMAGE_MODE}}"
RANKER_LLAVA_IMAGE_MODE="${RANKER_LLAVA_IMAGE_MODE-${LLAVA_IMAGE_MODE}}"
QWEN_MIN_PIXELS="${QWEN_MIN_PIXELS-0}"
QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS-0}"
RANKER_QWEN_MIN_PIXELS="${RANKER_QWEN_MIN_PIXELS-${QWEN_MIN_PIXELS}}"
RANKER_QWEN_MAX_PIXELS="${RANKER_QWEN_MAX_PIXELS-${QWEN_MAX_PIXELS}}"

# ===== 视角选择配置：只用于定位已有编码实验，不重新选择视角 =====
# 精确匹配要求此处与编码时一致；PMI 与 overlap 可同时开启。
USE_VIEW_PMI="${USE_VIEW_PMI-false}"
USE_VIEW_OVERLAP_SELECTION="${USE_VIEW_OVERLAP_SELECTION-false}"
VIEW_SELECTION_MODALITY="${VIEW_SELECTION_MODALITY-both}"  # text / image / both
VIEW_SELECTION_SPLIT="${VIEW_SELECTION_SPLIT-}"  # 空时跟随 DATASET_SPLIT
VIEW_SELECTION_ROUNDS="${VIEW_SELECTION_ROUNDS-20}"
VIEW_TEXTS_PER_ROUND="${VIEW_TEXTS_PER_ROUND-8}"
VIEW_IMAGES_PER_ROUND="${VIEW_IMAGES_PER_ROUND-2}"
VIEW_SELECTION_SEED="${VIEW_SELECTION_SEED-42}"
VIEW_SELECTED_COUNT="${VIEW_SELECTED_COUNT-5}"
VIEW_OVERLAP_THRESHOLD="${VIEW_OVERLAP_THRESHOLD-0.1}"
VIEW_OVERLAP_SAMPLES="${VIEW_OVERLAP_SAMPLES-20}"
VIEW_OVERLAP_SEED="${VIEW_OVERLAP_SEED-42}"

# ===== 三阶段检索与重排 =====
STAGE1_DEPTH="${STAGE1_DEPTH-1000}"
STAGE2_DEPTH="${STAGE2_DEPTH-1000}"
SPARSE_THREADS="${SPARSE_THREADS-16}"
RETRIEVAL_BATCH_SIZE="${RETRIEVAL_BATCH_SIZE-1}"
HYBRID_ALPHA="${HYBRID_ALPHA-0.5}"
SELECT_HYBRID_ALPHA="${SELECT_HYBRID_ALPHA-false}"  # true 在当前数据集扫描 0.1～0.9
RERANK_NUM="${RERANK_NUM-5}"
RERANK_BATCH_SIZE="${RERANK_BATCH_SIZE-1}"
# what_caption_generation 的 modify_type 始终按当前任务方向自动选择：
# IR 图搜文、ReMuQ/LLaVA 图文搜文本使用 modify_single，其余方向使用 no。
# 由调度器逐任务传入 JOB_MODIFY_TYPE；外部 MODIFY_TYPE 不覆盖此规则，也不启用先验缓存。
# 自定义独立验证集继续使用原单任务脚本。

# ===== 内部单任务分支：执行与原单任务脚本一致的完整三阶段命令 =====
if [[ "${RUN_ONE}" == true ]]; then
    shift
    : "${MODEL_FAMILY:?Internal job requires MODEL_FAMILY}"
    : "${MODEL_PATH:?Internal job requires MODEL_PATH}"
    : "${DATASET:?Internal job requires DATASET}"
    : "${OUTPUT_DIR:?Internal job requires OUTPUT_DIR}"
    : "${VIEW_SELECTION_FILE:?Internal job requires VIEW_SELECTION_FILE}"
    : "${DENSE_CORPUS_DIR:?Internal job requires DENSE_CORPUS_DIR}"
    : "${SPARSE_CORPUS_DIR:?Internal job requires SPARSE_CORPUS_DIR}"
    : "${JOB_MODIFY_TYPE:?Internal job requires JOB_MODIFY_TYPE from the batch scheduler}"
    case "${JOB_MODIFY_TYPE}" in
        no|modify_single) ;;
        *) echo 'Invalid internal JOB_MODIFY_TYPE' >&2; exit 2 ;;
    esac
    case "${DRY_RUN,,}" in
        true|1|yes) echo 'Use the normal batch entry with DRY_RUN=true; --run-one does not retrieve in dry-run mode.' >&2; exit 2 ;;
        false|0|no) ;;
        *) echo 'Invalid DRY_RUN: use true or false' >&2; exit 2 ;;
    esac
    PIPELINE_ARGS=()
    case "${MODEL_FAMILY:-qwen25}" in
        qwen2) CHECKPOINT=Qwen-Qwen2-VL-7B-Instruct ;;
        qwen25) CHECKPOINT=Qwen-Qwen2.5-VL-7B-Instruct ;;
        qwen3) CHECKPOINT=Qwen-Qwen3-VL-8B-Instruct ;;
        llava_llama) CHECKPOINT=llava-hf-llama3-llava-next-8b-hf ;;
        llava_mistral) CHECKPOINT=llava-hf-llava-v1.6-mistral-7b-hf ;;
        *) echo 'Invalid MODEL_FAMILY' >&2; exit 2 ;;
    esac
    MODEL_PATH="${MODEL_PATH:-${CHECKPOINT_ROOT:-/root/autodl-fs/checkpoints}/${CHECKPOINT}}"
    MODEL_PATH="${MODEL_PATH%/}"
    RETRIEVAL_LLAVA_IMAGE_MODE="${RETRIEVAL_LLAVA_IMAGE_MODE:-${LLAVA_IMAGE_MODE:-anyres}}"
    RANKER_LLAVA_IMAGE_MODE="${RANKER_LLAVA_IMAGE_MODE:-${LLAVA_IMAGE_MODE:-anyres}}"
    LLAVA_RESULT_SUFFIX=""
    for LLAVA_MODE in "${RETRIEVAL_LLAVA_IMAGE_MODE}" "${RANKER_LLAVA_IMAGE_MODE}"; do
        case "${LLAVA_MODE}" in anyres|base) ;; *) echo "LLaVA image mode must be anyres or base" >&2; exit 2 ;; esac
    done
    if [[ "${RETRIEVAL_LLAVA_IMAGE_MODE}" == base || "${RANKER_LLAVA_IMAGE_MODE}" == base ]]; then
        LLAVA_RESULT_SUFFIX="_retrieval_${RETRIEVAL_LLAVA_IMAGE_MODE}_ranker_${RANKER_LLAVA_IMAGE_MODE}"
    fi
    DATASET="${DATASET:-flickr}"
    case "${DATASET}" in
        flickr|coco) TASK=ir; QUERY="${IR_QUERY_TYPE:-image}" ;;
        CUHK-PEDES|ICFG-PEDES|RSTPReid) TASK=tbpr; QUERY=text ;;
        webqa) TASK=t2it; QUERY=text ;;
        remuq|llava) TASK=it2t; QUERY=image ;;
        *) echo 'Unsupported DATASET' >&2; exit 2 ;;
    esac
    EXTRA=()
    case "${SELECT_HYBRID_ALPHA:-false}" in
        true|1|yes)
            EXTRA+=(--select-hybrid-alpha)
            [[ -z "${VAL_PASSAGE_REPS:-}" ]] || EXTRA+=(--val-passage-reps "${VAL_PASSAGE_REPS}")
            [[ -z "${VAL_SPARSE_INDEX:-}" ]] || EXTRA+=(--val-sparse-index "${VAL_SPARSE_INDEX}")
            [[ -z "${VALIDATION_DATASET_NAME:-}" ]] || EXTRA+=(--validation-dataset-name "${VALIDATION_DATASET_NAME}")
            [[ -z "${VALIDATION_SPLIT:-}" ]] || EXTRA+=(--validation-split "${VALIDATION_SPLIT}")
            [[ -z "${VAL_VIEW_SELECTION_FILE:-}" ]] || EXTRA+=(--val-view-selection-file "${VAL_VIEW_SELECTION_FILE}") ;;
        false|0|no) ;; *) echo 'Invalid SELECT_HYBRID_ALPHA' >&2; exit 2 ;;
    esac
    for pair in 'IS_FILTERED:is-filtered' 'SPARSE_VALUE_MEAN:sparse-value-mean' 'OVERWRITE:overwrite'; do
        name="${pair%%:*}"; option="${pair#*:}"
        case "${!name:-false}" in true|1|yes) EXTRA+=("--${option}") ;; false|0|no) ;; *) printf 'Invalid %s\n' "${name}" >&2; exit 2 ;; esac
    done
    OUTPUT_DIR="${OUTPUT_DIR:-./three_stage_adaptive_results/${MODEL_PATH##*/}/${DATASET}/${QUERY}/${DATASET_SPLIT:-test}${LLAVA_RESULT_SUFFIX}/stage1_${STAGE1_DEPTH}_stage2_${STAGE2_DEPTH}}"
    printf 'Results: %s\n' "${OUTPUT_DIR}"
    exec "${DEEPSPEED_BIN:-deepspeed}" --num_gpus="${GPUS:-1}" --num_nodes="${NUM_NODES:-1}" src/three_stage_retrieval_pipeline_adaptive_views.py --view-selection-run-id "${VIEW_SELECTION_RUN_ID}" --use_freeret_dense "${USE_FREERET_DENSE}" --use_freeret_prompt_constraints "${USE_FREERET_PROMPT_CONSTRAINTS}" --use_freeret_full_prompt "${USE_FREERET_FULL_PROMPT}" \
        --retrieval-llava-image-mode "${RETRIEVAL_LLAVA_IMAGE_MODE}" --ranker-llava-image-mode "${RANKER_LLAVA_IMAGE_MODE}" \
        --adaptive-views --view-selection-file "${VIEW_SELECTION_FILE}" --dataset-name "${DATASET}" \
        --dataset-split "${DATASET_SPLIT:-test}" --task-type "${TASK}" --query-type "${QUERY}" \
        --data-root "${DATA_ROOT:-/root/autodl-tmp}" --tbpr-type "${TBPR_TYPE:-origin_type}" \
        --stage1-method sparse_concat --stage1-sparse-index "${STAGE1_SPARSE_INDEX:-${SPARSE_CORPUS_DIR}}" --stage1-depth "${STAGE1_DEPTH}" \
        --stage2-dense-index "${DENSE_CORPUS_DIR}" --stage2-sparse-index "${SPARSE_CORPUS_DIR}" --stage2-depth "${STAGE2_DEPTH}" \
        --hybrid-alpha "${HYBRID_ALPHA:-0.5}" --sparse-threads "${SPARSE_THREADS:-16}" \
        --retrieval-model-name-or-path "${MODEL_PATH}" --retrieval-dtype "${RETRIEVAL_DTYPE:-${DTYPE:-float16}}" \
        --retrieval-reps-loc "${REPS_LOC:-after_pad}" --retrieval-batch-size "${RETRIEVAL_BATCH_SIZE:-1}" \
        --retrieval-qwen-min-pixels "${QWEN_MIN_PIXELS:-0}" --retrieval-qwen-max-pixels "${QWEN_MAX_PIXELS:-0}" \
        --sparse-length "${SPARSE_LENGTH:-40}" --text-sparse-length "${TEXT_SPARSE_LENGTH:-${SPARSE_LENGTH:-40}}" \
        --image-sparse-length "${IMAGE_SPARSE_LENGTH:-${SPARSE_LENGTH:-40}}" --num-expended-tokens "${NUM_EXPENDED_TOKENS:-0}" \
        --sparse-value-type "${SPARSE_VALUE_TYPE:-sum}" --sparse-lower-or-upper "${SPARSE_LOWER_OR_UPPER:-lower}" \
        --ranker-model-name-or-path "${RANKER_MODEL_PATH:-${MODEL_PATH}}" --ranker-dtype "${RANKER_DTYPE:-${DTYPE:-float16}}" \
        --ranker-qwen-min-pixels "${RANKER_QWEN_MIN_PIXELS:-${QWEN_MIN_PIXELS:-0}}" --ranker-qwen-max-pixels "${RANKER_QWEN_MAX_PIXELS:-${QWEN_MAX_PIXELS:-0}}" \
        --rerank-type pointwise --rerank-template what_caption_generation --modify-type "${JOB_MODIFY_TYPE}" \
        --rerank-num "${RERANK_NUM:-5}" --rerank-batch-size "${RERANK_BATCH_SIZE:-1}" \
        --output-dir "${OUTPUT_DIR}" \
        "${EXTRA[@]}" "${PIPELINE_ARGS[@]}"
fi

# ===== 批量调度：普通变量通过一次 env 集中传给 Python =====
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "Python not found: ${PYTHON_BIN}; activate the retrieval environment or set PYTHON_BIN." >&2; exit 2; }
BATCH_ENV=("USE_HIERARCHICAL_VIEWS=${USE_HIERARCHICAL_VIEWS}" "VIEW_SELECTION_RUN_ID=${VIEW_SELECTION_RUN_ID}" "USE_FREERET_DENSE=${USE_FREERET_DENSE}" "USE_FREERET_PROMPT_CONSTRAINTS=${USE_FREERET_PROMPT_CONSTRAINTS}" "USE_FREERET_FULL_PROMPT=${USE_FREERET_FULL_PROMPT}")
for name in VIEW_EXAMPLE_MODE VIEW_EXAMPLE_MAX_TOKENS \
    MODEL_FAMILIES MODEL_FAMILY DATASETS DATASET IR_QUERY_TYPES IR_QUERY_TYPE DATASET_SPLIT \
    CHECKPOINT_ROOT MODEL_PATH_QWEN2 MODEL_PATH_QWEN25 MODEL_PATH_QWEN3 MODEL_PATH_LLAVA_LLAMA MODEL_PATH_LLAVA_MISTRAL MODEL_PATH \
    RANKER_MODEL_PATH_QWEN2 RANKER_MODEL_PATH_QWEN25 RANKER_MODEL_PATH_QWEN3 RANKER_MODEL_PATH_LLAVA_LLAMA RANKER_MODEL_PATH_LLAVA_MISTRAL RANKER_MODEL_PATH DATA_ROOT \
    DENSE_OUTPUT_ROOT SPARSE_OUTPUT_ROOT ADAPTIVE_OUTPUT_ROOT OUTPUT_DIR GPUS NUM_NODES DTYPE \
    RETRIEVAL_DTYPE RANKER_DTYPE DRY_RUN OVERWRITE BASH_BIN DEEPSPEED_BIN REPS_LOC \
    SPARSE_LENGTH TEXT_SPARSE_LENGTH IMAGE_SPARSE_LENGTH SPARSE_VALUE_TYPE SPARSE_LOWER_OR_UPPER SPARSE_VALUE_MEAN IS_FILTERED \
    NUM_EXPENDED_TOKENS TBPR_TYPE LLAVA_IMAGE_MODE RETRIEVAL_LLAVA_IMAGE_MODE RANKER_LLAVA_IMAGE_MODE QWEN_MIN_PIXELS QWEN_MAX_PIXELS \
    RANKER_QWEN_MIN_PIXELS RANKER_QWEN_MAX_PIXELS USE_VIEW_PMI USE_VIEW_OVERLAP_SELECTION VIEW_SELECTION_MODALITY VIEW_SELECTION_SPLIT VIEW_SELECTION_ROUNDS \
    VIEW_TEXTS_PER_ROUND VIEW_IMAGES_PER_ROUND VIEW_SELECTION_SEED VIEW_SELECTED_COUNT VIEW_OVERLAP_THRESHOLD VIEW_OVERLAP_SAMPLES VIEW_OVERLAP_SEED \
    STAGE1_DEPTH STAGE2_DEPTH SPARSE_THREADS RETRIEVAL_BATCH_SIZE HYBRID_ALPHA SELECT_HYBRID_ALPHA RERANK_NUM \
    RERANK_BATCH_SIZE; do
    BATCH_ENV+=("${name}=${!name}")
done
exec env "VIEW_SELECTION_RUN_ID=${VIEW_SELECTION_RUN_ID}" "${BATCH_ENV[@]}" "${PYTHON_BIN}" -B src/adaptive_retrieval_batch.py --task-group ir --launcher "${SCRIPT_PATH}" "$@"
