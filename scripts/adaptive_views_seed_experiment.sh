#!/usr/bin/env bash
# 仅运行示例采样和视角概率分配；不同 seed 共用模型、数据池及选项映射。
set -euo pipefail
# Opt in to summing all vocabulary variants of each view option; match this across encoding/index/search.
export USE_VIEW_TOKEN_VARIANTS="${USE_VIEW_TOKEN_VARIANTS-false}"
# Optional PMI prior prompt: false keeps the candidate list; instruction_only removes it.
# VIEW_PMI_PRIOR_PROMPT_MODE=false
if [[ -v VIEW_PMI_PRIOR_PROMPT_MODE ]]; then
    case "${VIEW_PMI_PRIOR_PROMPT_MODE,,}" in
        false|instruction_only) export VIEW_PMI_PRIOR_PROMPT_MODE="${VIEW_PMI_PRIOR_PROMPT_MODE,,}" ;;
        *) echo 'Invalid VIEW_PMI_PRIOR_PROMPT_MODE: use false or instruction_only' >&2; exit 2 ;;
    esac
fi
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# ===== 每次一个模型和数据集，按 seed 列表依次计算 =====
MODEL_FAMILY="${MODEL_FAMILY-qwen25}" # qwen2 qwen25 qwen3 llava_llama llava_mistral
CHECKPOINT_ROOT="${CHECKPOINT_ROOT-/root/gpufree-share/checkpoints}"
MODEL_PATH="${MODEL_PATH-}"          # 留空：按模型族在 CHECKPOINT_ROOT 中定位
LORA_MODEL_PATH="${LORA_MODEL_PATH-}"
DATASET="${DATASET-flickr}"          # flickr coco CUHK-PEDES ICFG-PEDES RSTPReid webqa remuq llava
DATASET_SPLIT="${DATASET_SPLIT-test}"
VIEW_SELECTION_SPLIT="${VIEW_SELECTION_SPLIT-}" # 空时跟随 DATASET_SPLIT
DATA_ROOT="${DATA_ROOT-/root/gpufree-share}"
OUTPUT_ROOT="${OUTPUT_ROOT-${REPO_ROOT}/adaptive_view_seed_results}"
VIEW_SELECTION_SEEDS="${VIEW_SELECTION_SEEDS-0 1 2 3 42}" # 首个 seed 是对比基准；重复值去重
INFERENCE_SEED="${INFERENCE_SEED-42}" # 固定模型推理随机状态，仅改变示例采样 seed

# ===== 模型运行及图像处理：单进程，无需 DeepSpeed =====
DEVICE="${DEVICE-auto}"             # auto / cpu / cuda / cuda:0；可用 CUDA_VISIBLE_DEVICES 选择显卡
DTYPE="${DTYPE-float16}"
REPS_LOC="${REPS_LOC-after_pad}"
LLAVA_IMAGE_MODE="${LLAVA_IMAGE_MODE-anyres}" # Qwen 必须 anyres；LLaVA-NeXT 可用 base
QWEN_MIN_PIXELS="${QWEN_MIN_PIXELS-0}"         # 0 保留 processor 默认值
QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS-0}"

# ===== 每个 seed 默认20轮，固定每轮8条文本、2张图像 =====
VIEW_SELECTION_ROUNDS="${VIEW_SELECTION_ROUNDS-20}"
VIEW_SELECTION_MODALITY="${VIEW_SELECTION_MODALITY-both}" # text / image / both；两种模态独立评分
VIEW_EXAMPLE_MODE="${VIEW_EXAMPLE_MODE-fixed}"            # fixed / auto
VIEW_TEXTS_PER_ROUND="${VIEW_TEXTS_PER_ROUND-8}"           # 仅 fixed 使用；不足时使用池中全部样本
VIEW_IMAGES_PER_ROUND="${VIEW_IMAGES_PER_ROUND-2}"
VIEW_EXAMPLE_MAX_TOKENS="${VIEW_EXAMPLE_MAX_TOKENS-0}"     # 仅 auto 使用；0 为模型上下文上限
USE_VIEW_PMI="${USE_VIEW_PMI-false}"
USE_HIERARCHICAL_VIEWS="${USE_HIERARCHICAL_VIEWS-false}"   # true: A-J层级视角；false: 原A-T视角
SHUFFLE_VIEW_OPTIONS="${SHUFFLE_VIEW_OPTIONS-false}"
VIEW_SELECTION_RUN_ID="${VIEW_SELECTION_RUN_ID-}" # 打乱时留空会生成一次映射，全部 seed 共用

# ===== 预检与程序 =====
DRY_RUN="${DRY_RUN-false}" # true: 检查配置/本地输入路径并打印，不加载模型或写文件
PYTHON_BIN="${PYTHON_BIN:-python}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "Python not found: ${PYTHON_BIN}" >&2; exit 2; }
read -r -a SEEDS <<< "${VIEW_SELECTION_SEEDS}"
if ((${#SEEDS[@]} == 0)); then echo 'VIEW_SELECTION_SEEDS must not be empty' >&2; exit 2; fi
EXTRA=()
[[ -z "${MODEL_PATH}" ]] || EXTRA+=(--model-path "${MODEL_PATH}")
[[ -z "${LORA_MODEL_PATH}" ]] || EXTRA+=(--lora-model-path "${LORA_MODEL_PATH}")
if [[ -v VIEW_PMI_PRIOR_PROMPT_MODE ]]; then
    EXTRA+=(--view-pmi-prior-prompt-mode "${VIEW_PMI_PRIOR_PROMPT_MODE}")
fi
exec "${PYTHON_BIN}" -B src/adaptive_views_seed_experiment.py \
    --model-family "${MODEL_FAMILY}" --checkpoint-root "${CHECKPOINT_ROOT}" \
    --dataset "${DATASET}" --dataset-split "${DATASET_SPLIT}" --view-selection-split "${VIEW_SELECTION_SPLIT}" \
    --data-root "${DATA_ROOT}" --output-root "${OUTPUT_ROOT}" --seeds "${SEEDS[@]}" \
    --inference-seed "${INFERENCE_SEED}" --device "${DEVICE}" --dtype "${DTYPE}" --reps-loc "${REPS_LOC}" \
    --llava-image-mode "${LLAVA_IMAGE_MODE}" --qwen-min-pixels "${QWEN_MIN_PIXELS}" --qwen-max-pixels "${QWEN_MAX_PIXELS}" \
    --view-selection-rounds "${VIEW_SELECTION_ROUNDS}" --view-selection-modality "${VIEW_SELECTION_MODALITY}" \
    --view-example-mode "${VIEW_EXAMPLE_MODE}" --view-example-max-tokens "${VIEW_EXAMPLE_MAX_TOKENS}" \
    --view-texts-per-round "${VIEW_TEXTS_PER_ROUND}" --view-images-per-round "${VIEW_IMAGES_PER_ROUND}" \
    --use-view-token-variants "${USE_VIEW_TOKEN_VARIANTS}" --use-view-pmi "${USE_VIEW_PMI}" --use-hierarchical-views "${USE_HIERARCHICAL_VIEWS}" \
    --shuffle-view-options "${SHUFFLE_VIEW_OPTIONS}" --view-selection-run-id "${VIEW_SELECTION_RUN_ID}" \
    --dry-run "${DRY_RUN}" "${EXTRA[@]}" "$@"
