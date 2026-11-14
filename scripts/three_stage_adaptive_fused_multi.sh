#!/usr/bin/env bash
# STC sparse vocabulary compression; false preserves the original sparse representation.
USE_STC_SPARSE="${USE_STC_SPARSE-false}"
STC_ALGORITHM="${STC_ALGORITHM-agglomerative}" # agglomerative / minibatch_kmeans
STC_N_CLUSTERS="${STC_N_CLUSTERS-16000}" # semantic clusters; numeric singleton dimensions are additional
STC_SEED="${STC_SEED-42}"
STC_BATCH_SIZE="${STC_BATCH_SIZE-4096}"
STC_MAX_ITER="${STC_MAX_ITER-100}"
STC_WEIGHT_SCALE="${STC_WEIGHT_SCALE-100}"
STC_CACHE_ROOT="${STC_CACHE_ROOT-}" # empty: SPARSE_OUTPUT_ROOT/stc_vocab
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/stc_sparse_mode.sh" || exit $?
# LENS centroid LM head; false preserves the original/STC representation.
USE_LENS_SPARSE="${USE_LENS_SPARSE-false}"
LENS_ALGORITHM="${LENS_ALGORITHM-kmeans}" # kmeans / minibatch_kmeans
LENS_N_CLUSTERS="${LENS_N_CLUSTERS-8000}"
LENS_SEED="${LENS_SEED-42}"
LENS_N_INIT="${LENS_N_INIT-1}"
LENS_MAX_ITER="${LENS_MAX_ITER-300}"
LENS_BATCH_SIZE="${LENS_BATCH_SIZE-4096}" # MiniBatch only
LENS_CACHE_ROOT="${LENS_CACHE_ROOT-}" # empty: SPARSE_OUTPUT_ROOT/lens_heads
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lens_sparse_mode.sh" || exit $?
STAGE2_METHOD="${STAGE2_METHOD-hybrid}" # hybrid / dense; dense sorts only Stage-1 candidates
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/stage2_retrieval_mode.sh" || exit $?
stage2_mode_configure || exit $?
# true: stop before the next probability-ranked view would exceed the threshold; keep at least the first.
USE_VIEW_PROBABILITY_MASS="${USE_VIEW_PROBABILITY_MASS-false}"
VIEW_PROBABILITY_MASS_THRESHOLD="${VIEW_PROBABILITY_MASS_THRESHOLD-0.9}" # [0,1]; 1 selects all; incompatible with overlap
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
# Opt in to summing all vocabulary variants of each view option; match this across encoding/index/search.
export USE_VIEW_TOKEN_VARIANTS="${USE_VIEW_TOKEN_VARIANTS-false}"
# Exact uppercase prefix token for LLaVA only; incompatible with vocabulary variant sums.
export USE_VIEW_PREFIX_TOKENS="${USE_VIEW_PREFIX_TOKENS-false}"
case "${USE_VIEW_PREFIX_TOKENS,,}" in
    true|1|yes) USE_VIEW_PREFIX_TOKENS=true ;;
    false|0|no) USE_VIEW_PREFIX_TOKENS=false ;;
    *) echo 'Invalid USE_VIEW_PREFIX_TOKENS: use true or false' >&2; exit 2 ;;
esac
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
stage2_mode_configure || exit $?
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
exec env "${STC_ENV[@]}" "${LENS_ENV[@]}" "USE_VIEW_PROBABILITY_MASS=${USE_VIEW_PROBABILITY_MASS}" "VIEW_PROBABILITY_MASS_THRESHOLD=${VIEW_PROBABILITY_MASS_THRESHOLD}" "VIEW_EXAMPLE_MODE=${VIEW_EXAMPLE_MODE}" "VIEW_EXAMPLE_MAX_TOKENS=${VIEW_EXAMPLE_MAX_TOKENS}" "USE_HIERARCHICAL_VIEWS=${USE_HIERARCHICAL_VIEWS}" "VIEW_SELECTION_RUN_ID=${VIEW_SELECTION_RUN_ID}" "USE_FREERET_DENSE=${USE_FREERET_DENSE}" "USE_FREERET_PROMPT_CONSTRAINTS=${USE_FREERET_PROMPT_CONSTRAINTS}" "USE_FREERET_FULL_PROMPT=${USE_FREERET_FULL_PROMPT}" "${PYTHON_BIN}" -B src/adaptive_retrieval_batch.py --task-group fused "$@"
