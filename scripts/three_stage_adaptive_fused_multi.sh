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
USE_FREERET_DENSE="${USE_FREERET_DENSE-false}"
USE_FREERET_FULL_PROMPT="${USE_FREERET_FULL_PROMPT-false}"  # Complete FreeRet dense/sparse prompts; requires FreeRet dense
USE_FREERET_PROMPT_CONSTRAINTS="${USE_FREERET_PROMPT_CONSTRAINTS-false}"
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/freeret_dense_mode.sh" || exit $?
# Run one adaptive encoding configuration across models/datasets through all three retrieval stages.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
if (($#)) && ! { (($# == 1)) && [[ "$1" == -h || "$1" == --help ]]; }; then
    echo 'Configure environment variables or the configuration block; see --help.' >&2
    exit 2
fi

# Configuration: edit defaults here, or override with environment variables.
# Nonempty MODEL_FAMILY/DATASET/IR_QUERY_TYPE take priority over their lists.
export MODEL_FAMILIES="${MODEL_FAMILIES-qwen2 qwen25 qwen3 llava_llama llava_mistral}"
export DATASETS="${DATASETS-webqa remuq llava}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT-/root/autodl-fs/checkpoints}"
export DATA_ROOT="${DATA_ROOT-/root/autodl-tmp}"
export DATASET_SPLIT="${DATASET_SPLIT-test}"
export DENSE_OUTPUT_ROOT="${DENSE_OUTPUT_ROOT-./dense_output}"
export SPARSE_OUTPUT_ROOT="${SPARSE_OUTPUT_ROOT-./sparse_output}"
export ADAPTIVE_OUTPUT_ROOT="${ADAPTIVE_OUTPUT_ROOT-./three_stage_adaptive_results}"

# These settings MUST match the corresponding adaptive encoding run.
export DTYPE="${DTYPE-float16}"
export REPS_LOC="${REPS_LOC-after_pad}"
export SPARSE_LENGTH="${SPARSE_LENGTH-40}"
export USE_VIEW_PMI="${USE_VIEW_PMI-false}"
export USE_VIEW_OVERLAP_SELECTION="${USE_VIEW_OVERLAP_SELECTION-false}"
export VIEW_SELECTION_MODALITY="${VIEW_SELECTION_MODALITY-both}"
export VIEW_SELECTION_ROUNDS="${VIEW_SELECTION_ROUNDS-20}"
export VIEW_TEXTS_PER_ROUND="${VIEW_TEXTS_PER_ROUND-8}"
export VIEW_IMAGES_PER_ROUND="${VIEW_IMAGES_PER_ROUND-2}"
export VIEW_SELECTION_SEED="${VIEW_SELECTION_SEED-42}"
# export VIEW_SELECTION_SPLIT=train
# export TEXT_SPARSE_LENGTH=40 IMAGE_SPARSE_LENGTH=40 TBPR_TYPE=origin_type
# export LLAVA_IMAGE_MODE=anyres QWEN_MIN_PIXELS=0 QWEN_MAX_PIXELS=0
# export VIEW_SELECTED_COUNT=5 VIEW_OVERLAP_THRESHOLD=0.1 VIEW_OVERLAP_SAMPLES=20 VIEW_OVERLAP_SEED=42

export GPUS="${GPUS-1}"
export NUM_NODES="${NUM_NODES-1}"
export STAGE1_DEPTH="${STAGE1_DEPTH-1000}"
export STAGE2_DEPTH="${STAGE2_DEPTH-1000}"
export HYBRID_ALPHA="${HYBRID_ALPHA-0.5}"
export SELECT_HYBRID_ALPHA="${SELECT_HYBRID_ALPHA-false}"
export RERANK_NUM="${RERANK_NUM-5}"
export OVERWRITE="${OVERWRITE-true}"
export DRY_RUN="${DRY_RUN-false}"

# Optional per-family paths; ranker defaults to the same family's retrieval model.
# export MODEL_PATH_QWEN2="/path/to/Qwen-Qwen2-VL-7B-Instruct"
# export MODEL_PATH_QWEN25="/path/to/Qwen-Qwen2.5-VL-7B-Instruct"
# export MODEL_PATH_QWEN3="/path/to/Qwen-Qwen3-VL-8B-Instruct"
# export MODEL_PATH_LLAVA_LLAMA="/path/to/llava-hf-llama3-llava-next-8b-hf"
# export MODEL_PATH_LLAVA_MISTRAL="/path/to/llava-hf-llava-v1.6-mistral-7b-hf"
# export RANKER_MODEL_PATH_QWEN25="/path/to/custom-ranker"
# OUTPUT_DIR requires one job; otherwise use ADAPTIVE_OUTPUT_ROOT.
PYTHON_BIN="${PYTHON_BIN:-python}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "Python not found: ${PYTHON_BIN}; activate the retrieval environment or set PYTHON_BIN." >&2; exit 2; }
exec env "VIEW_EXAMPLE_MODE=${VIEW_EXAMPLE_MODE}" "VIEW_EXAMPLE_MAX_TOKENS=${VIEW_EXAMPLE_MAX_TOKENS}" "USE_HIERARCHICAL_VIEWS=${USE_HIERARCHICAL_VIEWS}" "VIEW_SELECTION_RUN_ID=${VIEW_SELECTION_RUN_ID}" "USE_FREERET_DENSE=${USE_FREERET_DENSE}" "USE_FREERET_PROMPT_CONSTRAINTS=${USE_FREERET_PROMPT_CONSTRAINTS}" "USE_FREERET_FULL_PROMPT=${USE_FREERET_FULL_PROMPT}" "${PYTHON_BIN}" -B src/adaptive_retrieval_batch.py --task-group fused "$@"
