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
SHUFFLE_VIEW_OPTIONS="${SHUFFLE_VIEW_OPTIONS-false}"  # One recorded permutation per model/dataset in this run
case "${SHUFFLE_VIEW_OPTIONS,,}" in
    true|1|yes) SHUFFLE_VIEW_OPTIONS=true ;;
    false|0|no) SHUFFLE_VIEW_OPTIONS=false ;;
    *) echo 'Invalid SHUFFLE_VIEW_OPTIONS: use true or false' >&2; exit 2 ;;
esac
ADAPTIVE_SHUFFLE_ARGS=(--view_selection_shuffle "${SHUFFLE_VIEW_OPTIONS}")
if [[ "${SHUFFLE_VIEW_OPTIONS}" == true && -n "${_ADAPTIVE_SHUFFLE_RUN_ID:-}" ]]; then
    ADAPTIVE_SHUFFLE_ARGS+=(--view_selection_shuffle_run_id "${_ADAPTIVE_SHUFFLE_RUN_ID}")
fi
USE_FREERET_DENSE="${USE_FREERET_DENSE-false}"  # true: FreeRet pre-MLP dense; prompts/sparse unchanged
USE_FREERET_FULL_PROMPT="${USE_FREERET_FULL_PROMPT-false}"  # Complete FreeRet dense/sparse prompts; requires FreeRet dense
USE_FREERET_PROMPT_CONSTRAINTS="${USE_FREERET_PROMPT_CONSTRAINTS-false}"  # Prepend FreeRet constraints to all representation views
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/freeret_dense_mode.sh" || exit $?
# Adaptive overlap / PMI 批量编码：全部配置及实际启动参数均在本文件中展开。
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

# ===== 模型、数据集与候选编码模态 =====
# 修改默认列表可批量运行；非空单值覆盖优先，默认留空以保留批量范围。
MODEL_FAMILIES="${MODEL_FAMILIES-qwen2 qwen25 qwen3 llava_llama llava_mistral}"
MODEL_FAMILY="${MODEL_FAMILY-}"
DATASETS="${DATASETS-webqa remuq llava}"
DATASET="${DATASET-}"
ENCODE_TYPES="${ENCODE_TYPES-corpus}"
ENCODE_TYPE="${ENCODE_TYPE-}"
DATASET_SPLIT="${DATASET_SPLIT-test}"
# IR_QUERY_TYPE 由每个任务的 ENCODE_TYPE 推导：image 候选对应 text 查询，反之亦然。
# TBPR 使用 task_type=tbpr；WebQA=t2it；ReMuQ/LLaVA=it2t。任务类型由数据集推导。

# ===== 检查点：五种模型的默认目录逐项列出 =====
CHECKPOINT_ROOT="${CHECKPOINT_ROOT-/root/autodl-fs/checkpoints}"
MODEL_PATH_QWEN2="${MODEL_PATH_QWEN2-${CHECKPOINT_ROOT}/Qwen-Qwen2-VL-7B-Instruct}"
MODEL_PATH_QWEN25="${MODEL_PATH_QWEN25-${CHECKPOINT_ROOT}/Qwen-Qwen2.5-VL-7B-Instruct}"
MODEL_PATH_QWEN3="${MODEL_PATH_QWEN3-${CHECKPOINT_ROOT}/Qwen-Qwen3-VL-8B-Instruct}"
MODEL_PATH_LLAVA_LLAMA="${MODEL_PATH_LLAVA_LLAMA-${CHECKPOINT_ROOT}/llava-hf-llama3-llava-next-8b-hf}"
MODEL_PATH_LLAVA_MISTRAL="${MODEL_PATH_LLAVA_MISTRAL-${CHECKPOINT_ROOT}/llava-hf-llava-v1.6-mistral-7b-hf}"
# 全局 MODEL_PATH 仅允许一个模型族；多模型使用上面的逐模型配置。
MODEL_PATH="${MODEL_PATH-}"

# ===== 数据及输出目录 =====
DATA_ROOT="${DATA_ROOT-/root/autodl-tmp}"
DENSE_OUTPUT_ROOT="${DENSE_OUTPUT_ROOT-./dense_output}"
SPARSE_OUTPUT_ROOT="${SPARSE_OUTPUT_ROOT-./sparse_output}"
ENCODE_LOG_ROOT="${ENCODE_LOG_ROOT-./adaptive_encode_logs}"
# OUTPUT_DIR 非空仅允许单个编码任务；批量日志按模型/数据集/split/模态自动分目录。
OUTPUT_DIR="${OUTPUT_DIR-}"

# ===== 运行资源与程序 =====
GPUS="${GPUS-1}"
NUM_NODES="${NUM_NODES-1}"
DTYPE="${DTYPE-float16}"                 # float16 / bfloat16 / float32
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE-1}"
DRY_RUN="${DRY_RUN-false}"               # true 仅预检并打印任务，不加载模型或写编码文件
PYTHON_BIN="${PYTHON_BIN:-python}"
BASH_BIN="${BASH_BIN:-bash}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"

# ===== 稀疏编码 =====
REPS_LOC="${REPS_LOC-after_pad}"         # before_pad / after_pad
SPARSE_LENGTH="${SPARSE_LENGTH-40}"
TEXT_SPARSE_LENGTH="${TEXT_SPARSE_LENGTH-${SPARSE_LENGTH}}"
IMAGE_SPARSE_LENGTH="${IMAGE_SPARSE_LENGTH-${SPARSE_LENGTH}}"
SPARSE_VALUE_TYPE="${SPARSE_VALUE_TYPE-sum}"              # replace / sum / max
SPARSE_LOWER_OR_UPPER="${SPARSE_LOWER_OR_UPPER-lower}"     # lower / upper
SPARSE_VALUE_MEAN="${SPARSE_VALUE_MEAN-false}"
IS_FILTERED="${IS_FILTERED-false}"
NUM_EXPENDED_TOKENS="${NUM_EXPENDED_TOKENS-0}"
TBPR_TYPE="${TBPR_TYPE-origin_type}"     # origin_type / type

# ===== 图像处理 =====
LLAVA_IMAGE_MODE="${LLAVA_IMAGE_MODE-anyres}"              # anyres / base；base 仅支持 LLaVA-NeXT
QWEN_MIN_PIXELS="${QWEN_MIN_PIXELS-0}"    # 0 沿用 processor 默认值
QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS-0}"

# ===== 数据集级视角选择及 PMI =====
# 一个模型/数据集组合可指定锁定文件；多个组合留空，分别使用自动缓存。
VIEW_SELECTION_FILE="${VIEW_SELECTION_FILE-}"
VIEW_SELECTION_SPLIT="${VIEW_SELECTION_SPLIT-}"           # 空时跟随 DATASET_SPLIT
VIEW_SELECTION_ROUNDS="${VIEW_SELECTION_ROUNDS-20}"
VIEW_TEXTS_PER_ROUND="${VIEW_TEXTS_PER_ROUND-8}"
VIEW_IMAGES_PER_ROUND="${VIEW_IMAGES_PER_ROUND-2}"
VIEW_SELECTION_SEED="${VIEW_SELECTION_SEED-42}"
# 下列两项故意不赋默认值：未设置时，新选择使用 false/both，复用文件继承其记录。
# 如需显式设置，取消相应行的注释；外部环境变量同样有效。
# USE_VIEW_PMI=true                       # true / false
# VIEW_SELECTION_MODALITY=both            # text / image / both

# ===== overlap 筛选：可与 PMI 同时启用 =====
USE_VIEW_OVERLAP_SELECTION="${USE_VIEW_OVERLAP_SELECTION-true}"
# 不指定选择文件时使用以下默认值；显式复用文件时，未设置的值继承文件。
if [[ -z "${VIEW_SELECTION_FILE}" ]]; then
    VIEW_SELECTED_COUNT="${VIEW_SELECTED_COUNT-5}"
    VIEW_OVERLAP_THRESHOLD="${VIEW_OVERLAP_THRESHOLD-0.1}"  # 重合数/当前候选有效token数，严格小于阈值才接受
    VIEW_OVERLAP_SAMPLES="${VIEW_OVERLAP_SAMPLES-20}"      # 每个启用模态独立采样
    VIEW_OVERLAP_SEED="${VIEW_OVERLAP_SEED-42}"
fi
# 若需覆盖已指定选择文件的设置，可在此使用普通变量赋值；冲突会报错。
# VIEW_SELECTED_COUNT=5
# VIEW_OVERLAP_THRESHOLD=0.1
# VIEW_OVERLAP_SAMPLES=20
# VIEW_OVERLAP_SEED=42
# 探测固定 top-30；最终语料编码长度仍由 SPARSE_LENGTH 等控制。

# ===== 内部单任务分支：调度器传入已确定的模型、数据集、模态和输出目录 =====
# 以下完整命令与 encode_adaptive_views.sh 对齐；固定参数直接写在命令中。
# eol_type=disassembleeol_separate，calculate_type=concat，sparse_manual=false；
# prompt_type=prompt_5，random_view_seed=42（不是 VIEW_SELECTION_SEED），cir_type=type（兼容字段）。
if [[ "${RUN_ONE}" == true ]]; then
    shift
    : "${MODEL_FAMILY:?Internal job requires MODEL_FAMILY}"
    : "${MODEL_PATH:?Internal job requires MODEL_PATH}"
    : "${DATASET:?Internal job requires DATASET}"
    : "${ENCODE_TYPE:?Internal job requires ENCODE_TYPE}"
    : "${OUTPUT_DIR:?Internal job requires OUTPUT_DIR}"
    case "${DRY_RUN,,}" in
        true|1|yes) echo 'Use the normal batch entry with DRY_RUN=true; --run-one does not encode in dry-run mode.' >&2; exit 2 ;;
        false|0|no) ;;
        *) echo 'Invalid DRY_RUN: use true or false' >&2; exit 2 ;;
    esac
    MODEL_FAMILY="${MODEL_FAMILY:-qwen25}"
    case "${MODEL_FAMILY}" in
        qwen2) CHECKPOINT=Qwen-Qwen2-VL-7B-Instruct ;;
        qwen25) CHECKPOINT=Qwen-Qwen2.5-VL-7B-Instruct ;;
        qwen3) CHECKPOINT=Qwen-Qwen3-VL-8B-Instruct ;;
        llava_llama) CHECKPOINT=llava-hf-llama3-llava-next-8b-hf ;;
        llava_mistral) CHECKPOINT=llava-hf-llava-v1.6-mistral-7b-hf ;;
        *) printf 'Invalid MODEL_FAMILY: %s\n' "${MODEL_FAMILY}" >&2; exit 2 ;;
    esac
    MODEL_PATH="${MODEL_PATH:-${CHECKPOINT_ROOT:-/root/autodl-fs/checkpoints}/${CHECKPOINT}}"
    DATASET="${DATASET:-flickr}"
    IR_QUERY_TYPE="${IR_QUERY_TYPE:-image}"
    case "${DATASET}" in
        flickr|coco)
            TASK=ir
            case "${IR_QUERY_TYPE}" in image) KIND=text ;; text) KIND=image ;; *) echo 'Invalid IR_QUERY_TYPE' >&2; exit 2 ;; esac ;;
        CUHK-PEDES|ICFG-PEDES|RSTPReid) TASK=tbpr; KIND=image ;;
        webqa) TASK=t2it; KIND=corpus ;;
        remuq|llava) TASK=it2t; KIND=corpus ;;
        *) printf 'Unsupported DATASET: %s\n' "${DATASET}" >&2; exit 2 ;;
    esac
    PRECISION=()
    case "${DTYPE:-float16}" in float16) PRECISION+=(--fp16) ;; bfloat16) PRECISION+=(--bf16) ;; float32) ;; *) echo 'Invalid DTYPE' >&2; exit 2 ;; esac
    EXTRA=()
    if [[ -v USE_VIEW_PMI ]]; then
        case "${USE_VIEW_PMI,,}" in
            true|1|yes) EXTRA+=(--view_selection_pmi true) ;;
            false|0|no) EXTRA+=(--view_selection_pmi false) ;;
            *) echo 'Invalid USE_VIEW_PMI: use true or false' >&2; exit 2 ;;
        esac
    fi
    case "${USE_VIEW_OVERLAP_SELECTION,,}" in
        true|1|yes) EXTRA+=(--view_overlap_selection) ;;
        false|0|no) ;;
        *) echo 'Invalid USE_VIEW_OVERLAP_SELECTION: use true or false' >&2; exit 2 ;;
    esac
    for pair in 'VIEW_SELECTED_COUNT:view_selected_count' 'VIEW_OVERLAP_THRESHOLD:view_overlap_threshold' \
                'VIEW_OVERLAP_SAMPLES:view_overlap_samples' 'VIEW_OVERLAP_SEED:view_overlap_seed'; do
        name="${pair%%:*}"; option="${pair#*:}"
        [[ ! -v "${name}" ]] || EXTRA+=("--${option}" "${!name}")
    done
    if [[ -n "${VIEW_SELECTION_MODALITY:-}" ]]; then
        case "${VIEW_SELECTION_MODALITY}" in text|image|both) ;; *) echo 'Invalid VIEW_SELECTION_MODALITY: use text, image or both' >&2; exit 2 ;; esac
        EXTRA+=(--view_selection_modality "${VIEW_SELECTION_MODALITY}")
    fi
    [[ -z "${VIEW_SELECTION_FILE:-}" ]] || EXTRA+=(--view_selection_file "${VIEW_SELECTION_FILE}")
    [[ -z "${VIEW_SELECTION_SPLIT:-}" ]] || EXTRA+=(--view_selection_split "${VIEW_SELECTION_SPLIT}")
    for pair in 'IS_FILTERED:is_filtered' 'SPARSE_VALUE_MEAN:sparse_value_mean'; do
        name="${pair%%:*}"; option="${pair#*:}"
        case "${!name:-false}" in true|1|yes) EXTRA+=("--${option}") ;; false|0|no) ;; *) printf 'Invalid %s\n' "${name}" >&2; exit 2 ;; esac
    done
    exec "${DEEPSPEED_BIN:-deepspeed}" --num_gpus="${GPUS:-1}" --num_nodes="${NUM_NODES:-1}" src/encode_adaptive_views_overlap.py --use_freeret_dense "${USE_FREERET_DENSE}" --use_freeret_prompt_constraints "${USE_FREERET_PROMPT_CONSTRAINTS}" --use_freeret_full_prompt "${USE_FREERET_FULL_PROMPT}" --view_selection_hierarchical "${USE_HIERARCHICAL_VIEWS}" "${ADAPTIVE_SHUFFLE_ARGS[@]}" \
        --llava_image_mode "${LLAVA_IMAGE_MODE:-anyres}" \
        --adaptive_views --model_name_or_path "${MODEL_PATH}" --dataset_name "${DATASET}" \
        --dataset_split "${DATASET_SPLIT:-test}" --task_type "${TASK}" --encode_type "${ENCODE_TYPE:-${KIND}}" \
        --adaptive_data_root "${DATA_ROOT:-/root/autodl-tmp}" --output_dir "${OUTPUT_DIR:-./adaptive_encode_logs}" \
        --dense_output_dir "${DENSE_OUTPUT_ROOT:-./dense_output}" --sparse_output_dir "${SPARSE_OUTPUT_ROOT:-./sparse_output}" \
        --eol_type disassembleeol_separate --calculate_type concat --sparse_manual false \
        --prompt_type prompt_5 --random_view_seed 42 --reps_loc "${REPS_LOC:-after_pad}" \
        --per_device_batch_size "${PER_DEVICE_BATCH_SIZE:-1}" --sparse_length "${SPARSE_LENGTH:-40}" \
        --text_sparse_length "${TEXT_SPARSE_LENGTH:-${SPARSE_LENGTH:-40}}" --image_sparse_length "${IMAGE_SPARSE_LENGTH:-${SPARSE_LENGTH:-40}}" \
        --sparse_value_type "${SPARSE_VALUE_TYPE:-sum}" --sparse_lower_or_upper "${SPARSE_LOWER_OR_UPPER:-lower}" \
        --num_expended_tokens "${NUM_EXPENDED_TOKENS:-0}" --tbpr_type "${TBPR_TYPE:-origin_type}" --cir_type type \
        --qwen_min_pixels "${QWEN_MIN_PIXELS:-0}" --qwen_max_pixels "${QWEN_MAX_PIXELS:-0}" \
        --view_example_mode "${VIEW_EXAMPLE_MODE}" --view_example_max_tokens "${VIEW_EXAMPLE_MAX_TOKENS}" \
        --view_selection_rounds "${VIEW_SELECTION_ROUNDS:-20}" --view_texts_per_round "${VIEW_TEXTS_PER_ROUND:-8}" \
        --view_images_per_round "${VIEW_IMAGES_PER_ROUND:-2}" --view_selection_seed "${VIEW_SELECTION_SEED:-42}" \
        "${PRECISION[@]}" "${EXTRA[@]}" "$@"
fi

# ===== 批量调度：每个任务回到本文件的 --run-one 分支，不再委托单任务脚本 =====
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "Python not found: ${PYTHON_BIN}; activate the encoding environment or set PYTHON_BIN." >&2; exit 2; }
# 配置使用普通变量赋值；只在进入 Python 调度器时集中传入子进程环境。
BATCH_ENV=("USE_HIERARCHICAL_VIEWS=${USE_HIERARCHICAL_VIEWS}" "SHUFFLE_VIEW_OPTIONS=${SHUFFLE_VIEW_OPTIONS}" "USE_VIEW_OVERLAP_SELECTION=${USE_VIEW_OVERLAP_SELECTION}" "USE_FREERET_DENSE=${USE_FREERET_DENSE}" "USE_FREERET_PROMPT_CONSTRAINTS=${USE_FREERET_PROMPT_CONSTRAINTS}" "USE_FREERET_FULL_PROMPT=${USE_FREERET_FULL_PROMPT}")
for name in VIEW_EXAMPLE_MODE VIEW_EXAMPLE_MAX_TOKENS MODEL_FAMILIES MODEL_FAMILY DATASETS DATASET ENCODE_TYPES ENCODE_TYPE DATASET_SPLIT \
    CHECKPOINT_ROOT MODEL_PATH_QWEN2 MODEL_PATH_QWEN25 MODEL_PATH_QWEN3 MODEL_PATH_LLAVA_LLAMA MODEL_PATH_LLAVA_MISTRAL MODEL_PATH \
    DATA_ROOT DENSE_OUTPUT_ROOT SPARSE_OUTPUT_ROOT ENCODE_LOG_ROOT OUTPUT_DIR \
    GPUS NUM_NODES DTYPE PER_DEVICE_BATCH_SIZE DRY_RUN BASH_BIN DEEPSPEED_BIN \
    REPS_LOC SPARSE_LENGTH TEXT_SPARSE_LENGTH IMAGE_SPARSE_LENGTH SPARSE_VALUE_TYPE SPARSE_LOWER_OR_UPPER \
    SPARSE_VALUE_MEAN IS_FILTERED NUM_EXPENDED_TOKENS TBPR_TYPE LLAVA_IMAGE_MODE QWEN_MIN_PIXELS QWEN_MAX_PIXELS \
    VIEW_SELECTION_FILE VIEW_SELECTION_SPLIT VIEW_SELECTION_ROUNDS VIEW_TEXTS_PER_ROUND VIEW_IMAGES_PER_ROUND VIEW_SELECTION_SEED; do
    BATCH_ENV+=("${name}=${!name}")
done
# 未设置的可选项保持未设置，以便复用选择文件时继承其 PMI/模态记录。
for name in USE_VIEW_PMI VIEW_SELECTION_MODALITY VIEW_SELECTED_COUNT VIEW_OVERLAP_THRESHOLD VIEW_OVERLAP_SAMPLES VIEW_OVERLAP_SEED; do
    if [[ -v "${name}" ]]; then BATCH_ENV+=("${name}=${!name}"); fi
done
exec env "${BATCH_ENV[@]}" "${PYTHON_BIN}" -B src/adaptive_encode_batch.py --task-group fused --encoder overlap --launcher "${SCRIPT_PATH}" "$@"
