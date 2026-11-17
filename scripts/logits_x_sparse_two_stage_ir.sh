#!/usr/bin/env bash
# 单视角 sparse 权重乘以实际选中视角数量 K；编码、索引、两阶段检索统一入口。
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/logits_x_sparse_two_stage_ir.sh [all|encode|index|search]

  all      Encode image/text corpora, build indexes, then run sparse -> hybrid (default).
  encode   Encode with K = len(selected) from the shared view-selection record.
  index    Index the matching encoded corpora.
  search   Run two-stage retrieval using existing matching corpora and indexes.
  -h, --help  Show this help without loading models or starting a stage.

Examples (run in the GPU environment with Transformers, FAISS and Pyserini):
  bash scripts/logits_x_sparse_two_stage_ir.sh
  GPUS=1 bash scripts/logits_x_sparse_two_stage_ir.sh all
  MODEL_FAMILY=llava_llama DATASET=coco bash scripts/logits_x_sparse_two_stage_ir.sh all
  IR_QUERY_TYPES=text bash scripts/logits_x_sparse_two_stage_ir.sh search

Edit the configuration block below, or override its defaults with environment variables.
  MODEL_FAMILY: llava_mistral (default) / llava_llama
  DATASET: flickr (default) / coco; DATASET_SPLIT: test (default)
  ENCODE_TYPES: "image text"; IR_QUERY_TYPES: "image text"
    image queries retrieve text; text queries retrieve images.
  CHECKPOINT_ROOT, MODEL_PATH, DATA_ROOT: model and dataset locations
  DENSE_OUTPUT_ROOT, SPARSE_OUTPUT_ROOT: encoded corpus locations
  ENCODE_LOG_ROOT, ADAPTIVE_OUTPUT_ROOT: encoding logs and retrieval results
  GPUS, NUM_NODES, DTYPE, PER_DEVICE_BATCH_SIZE, RETRIEVAL_BATCH_SIZE: resources
  SPARSE_LENGTH, LLAVA_IMAGE_MODE, VIEW_* and USE_VIEW_*: shared encoding/selection settings
  STAGE1_DEPTH, STAGE2_DEPTH, HYBRID_ALPHA: retrieval settings (defaults 200 / 200 / 0.7)
  PYTHON_BIN, BASH_BIN, DEEPSPEED_BIN: program locations

This dedicated entry always enables selected-count scaling and hybrid Stage 2.
Mean aggregation, overlap, STC, LENS, full FreeRet prompts, option shuffling,
alpha scanning and prior-cache preparation are disabled. Stage 3 is skipped.
The dense representation is not scaled; K is read automatically, never set to a fixed 5.

Selection caches and artifact paths are located by the existing batch schedulers.
Inherited single-job OUTPUT_DIR, ENCODE_TYPE, IR_QUERY_TYPE, VIEW_SELECTION_FILE,
TARGET_SPARSE_DIR and corpus/index paths are cleared. Use the list/root settings above.
Keep the same settings for separate encode/index/search invocations.
The all/encode stages re-encode corpora; search alone requires the new mode's artifacts.
DRY_RUN=true delegates each stage's existing preflight; index/search still need encoded artifacts.
EOF
}

if (($# > 1)); then
    usage >&2
    exit 2
fi
RUN_STAGE="${1:-all}"
case "${RUN_STAGE}" in
    -h|--help) usage; exit 0 ;;
    all|encode|index|search) ;;
    *) printf 'Unknown stage: %s\n' "${RUN_STAGE}" >&2; usage >&2; exit 2 ;;
esac

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# ===== 模型、数据集与方向：单模型、单数据集，默认双向检索 =====
export MODEL_FAMILY="${MODEL_FAMILY:-llava_mistral}"
export DATASET="${DATASET:-flickr}"
export DATASET_SPLIT="${DATASET_SPLIT:-test}"
export ENCODE_TYPES="${ENCODE_TYPES:-image text}"
export IR_QUERY_TYPES="${IR_QUERY_TYPES:-image text}"
case "${MODEL_FAMILY}" in
    llava_mistral|llava_llama) ;;
    *) printf 'MODEL_FAMILY must be llava_mistral or llava_llama: %s\n' "${MODEL_FAMILY}" >&2; exit 2 ;;
esac
case "${DATASET}" in
    flickr|coco) ;;
    *) printf 'DATASET must be flickr or coco: %s\n' "${DATASET}" >&2; exit 2 ;;
esac

# ===== 路径：三个阶段共享同一组根目录，具体实验路径由调度器计算 =====
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/root/autodl-fs/checkpoints}"
export MODEL_PATH_LLAVA_LLAMA="${MODEL_PATH_LLAVA_LLAMA:-${CHECKPOINT_ROOT}/llava-hf-llama3-llava-next-8b-hf}"
export MODEL_PATH_LLAVA_MISTRAL="${MODEL_PATH_LLAVA_MISTRAL:-${CHECKPOINT_ROOT}/llava-hf-llava-v1.6-mistral-7b-hf}"
export MODEL_PATH="${MODEL_PATH:-}"
export DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
export DENSE_OUTPUT_ROOT="${DENSE_OUTPUT_ROOT:-/root/autodl-fs/dense_output}"
export SPARSE_OUTPUT_ROOT="${SPARSE_OUTPUT_ROOT:-/root/autodl-fs/sparse_output}"
export VIEW_SELECTION_ROOT="${VIEW_SELECTION_ROOT:-${SPARSE_OUTPUT_ROOT}/view_selection}"
export ENCODE_LOG_ROOT="${ENCODE_LOG_ROOT:-/root/autodl-fs/adaptive_encode_logs}"
export ADAPTIVE_OUTPUT_ROOT="${ADAPTIVE_OUTPUT_ROOT:-/root/autodl-fs/two_stage_adaptive_results}"

# ===== 运行资源：检索精度与图像模式始终跟随编码配置 =====
export GPUS="${GPUS:-2}"
export NUM_NODES="${NUM_NODES:-1}"
export DTYPE="${DTYPE:-float16}"
export RETRIEVAL_DTYPE="${DTYPE}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
export RETRIEVAL_BATCH_SIZE="${RETRIEVAL_BATCH_SIZE:-4}"
export SPARSE_THREADS="${SPARSE_THREADS:-16}"
export THREADS="${THREADS:-${SPARSE_THREADS}}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export BASH_BIN="${BASH_BIN:-bash}"
export DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"
export DRY_RUN="${DRY_RUN:-false}"
export OVERWRITE="${OVERWRITE:-true}"

# ===== 共享编码参数：含 IR 中不参与打分、但参与产物指纹的兼容字段 =====
export REPS_LOC="${REPS_LOC:-after_pad}"
export SPARSE_LENGTH="${SPARSE_LENGTH:-30}"
export TEXT_SPARSE_LENGTH="${TEXT_SPARSE_LENGTH:-${SPARSE_LENGTH}}"
export IMAGE_SPARSE_LENGTH="${IMAGE_SPARSE_LENGTH:-${SPARSE_LENGTH}}"
export SPARSE_VALUE_TYPE="${SPARSE_VALUE_TYPE:-sum}"
export SPARSE_LOWER_OR_UPPER="${SPARSE_LOWER_OR_UPPER:-lower}"
export IS_FILTERED="${IS_FILTERED:-false}"
export NUM_EXPENDED_TOKENS="${NUM_EXPENDED_TOKENS:-0}"
export TBPR_TYPE="${TBPR_TYPE:-type}"
export LLAVA_IMAGE_MODE="${LLAVA_IMAGE_MODE:-base}"
export RETRIEVAL_LLAVA_IMAGE_MODE="${LLAVA_IMAGE_MODE}"
export QWEN_MIN_PIXELS="${QWEN_MIN_PIXELS:-0}"
export QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS:-0}"
export USE_FREERET_DENSE="${USE_FREERET_DENSE:-false}"
export USE_FREERET_PROMPT_CONSTRAINTS="${USE_FREERET_PROMPT_CONSTRAINTS:-false}"

# ===== 共享视角选择配置：复用匹配的记录，K 取记录中的 len(selected) =====
export VIEW_SELECTION_MODALITY="${VIEW_SELECTION_MODALITY:-both}"
export VIEW_SELECTION_SPLIT="${VIEW_SELECTION_SPLIT:-${DATASET_SPLIT}}"
export VIEW_SELECTION_ROUNDS="${VIEW_SELECTION_ROUNDS:-20}"
export VIEW_TEXTS_PER_ROUND="${VIEW_TEXTS_PER_ROUND:-30}"
export VIEW_IMAGES_PER_ROUND="${VIEW_IMAGES_PER_ROUND:-10}"
export VIEW_SELECTION_SEED="${VIEW_SELECTION_SEED:-42}"
export VIEW_EXAMPLE_MODE="${VIEW_EXAMPLE_MODE:-auto}"
export VIEW_EXAMPLE_MAX_TOKENS="${VIEW_EXAMPLE_MAX_TOKENS:-0}"
export USE_HIERARCHICAL_VIEWS="${USE_HIERARCHICAL_VIEWS:-true}"
export USE_VIEW_PROBABILITY_MASS="${USE_VIEW_PROBABILITY_MASS:-true}"
export VIEW_PROBABILITY_MASS_THRESHOLD="${VIEW_PROBABILITY_MASS_THRESHOLD:-0.9}"
export USE_VIEW_PMI="${USE_VIEW_PMI:-false}"
export USE_VIEW_TOKEN_VARIANTS="${USE_VIEW_TOKEN_VARIANTS:-false}"
export USE_VIEW_PREFIX_TOKENS="${USE_VIEW_PREFIX_TOKENS:-false}"

# ===== 两阶段打分参数 =====
export STAGE1_DEPTH="${STAGE1_DEPTH:-200}"
export STAGE2_DEPTH="${STAGE2_DEPTH:-200}"
export HYBRID_ALPHA="${HYBRID_ALPHA:-0.7}"

# ===== 专用实验的固定开关：覆盖其他实验遗留的开关 =====
export SINGLE_VIEW_REPEAT_SELECTED_COUNT=true
export STAGE2_METHOD=hybrid
export SELECT_HYBRID_ALPHA=false
export USE_RERANK_PRIOR_CACHE=false
export SPARSE_VALUE_MEAN=false
export USE_VIEW_OVERLAP_SELECTION=false
export USE_STC_SPARSE=false
export USE_LENS_SPARSE=false
export USE_FREERET_FULL_PROMPT=false
export SHUFFLE_VIEW_OPTIONS=false
export VIEW_SELECTION_RUN_ID=""

# 单任务路径不跨阶段传递；方向通过列表参数配置，输出通过各自的根目录管理。
unset OUTPUT_DIR ENCODE_TYPE IR_QUERY_TYPE VIEW_SELECTION_FILE TARGET_SPARSE_DIR \
    DENSE_CORPUS_DIR SPARSE_CORPUS_DIR STAGE1_SPARSE_INDEX _ADAPTIVE_SHUFFLE_RUN_ID \
    VAL_PASSAGE_REPS VAL_SPARSE_INDEX VAL_VIEW_SELECTION_FILE VAL_STAGE1_SPARSE_INDEX \
    VALIDATION_DATASET_NAME VALIDATION_SPLIT

command -v "${BASH_BIN}" >/dev/null 2>&1 || { printf 'Bash not found: %s\n' "${BASH_BIN}" >&2; exit 2; }
printf 'Selected-count sparse -> hybrid: model=%s dataset=%s/%s stage=%s\n' \
    "${MODEL_FAMILY}" "${DATASET}" "${DATASET_SPLIT}" "${RUN_STAGE}"
printf 'Sparse multiplier: K = len(selected) from the shared selection record; dense is unscaled.\n'
printf 'Dense root: %s\nSparse root: %s\nResults root: %s\n' \
    "${DENSE_OUTPUT_ROOT}" "${SPARSE_OUTPUT_ROOT}" "${ADAPTIVE_OUTPUT_ROOT}"

run_stage() {
    local stage_name="$1" entry="$2" status
    printf '\n[logits_x_sparse] Starting %s\n' "${stage_name}"
    if "${BASH_BIN}" "${REPO_ROOT}/scripts/${entry}"; then
        printf '[logits_x_sparse] Completed %s\n' "${stage_name}"
    else
        status=$?
        printf '[logits_x_sparse] Failed %s (exit %s); stopping.\n' "${stage_name}" "${status}" >&2
        exit "${status}"
    fi
}

if [[ "${RUN_STAGE}" == all || "${RUN_STAGE}" == encode ]]; then
    run_stage encode encode_adaptive_views_ir.sh
fi
if [[ "${RUN_STAGE}" == all || "${RUN_STAGE}" == index ]]; then
    run_stage index sparse_index_adaptive_ir.sh
fi
if [[ "${RUN_STAGE}" == all || "${RUN_STAGE}" == search ]]; then
    run_stage search two_stage_retrieval_pipeline_adaptive_views_ir.sh
fi
