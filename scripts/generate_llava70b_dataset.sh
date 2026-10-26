#!/usr/bin/env bash
set -euo pipefail

# This script is intentionally independent from the existing search/encode
# scripts.  Change these variables instead of editing the Python program.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/root/autodl-fs/checkpoints/llava-70b}"
SOURCE_ROOT="${SOURCE_ROOT:-/root/autodl-tmp}"

# The requested n is controlled here.  It is applied independently to every
# dataset selected below.
NUM_IMAGES="${NUM_IMAGES:-10}"
SPLIT="${SPLIT:-test}"
SEED="${SEED:-2026}"

# Use "all" for every supported dataset.  LLaVA is handled only as an
# image+text-to-text (it2t) dataset.
DATASETS="${DATASETS:-all}"

# A 70B checkpoint normally needs multiple visible GPUs.  Set
# CUDA_VISIBLE_DEVICES outside this script if a particular GPU list is needed.
DTYPE="${DTYPE:-float16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.9}"
USE_LLAVA_NEXT_TRAINING_PROMPTS="${USE_LLAVA_NEXT_TRAINING_PROMPTS:-false}"
USE_FEW_SHOT_EXAMPLES="${USE_FEW_SHOT_EXAMPLES:-false}"
FEW_SHOT_NUM="${FEW_SHOT_NUM:-2}"
FEW_SHOT_SPLIT="${FEW_SHOT_SPLIT:-${SPLIT}}"

PROMPT_ARGS=()
case "${USE_LLAVA_NEXT_TRAINING_PROMPTS,,}" in
  true|1|yes)
    PROMPT_ARGS+=(--use_llava_next_training_prompts)
    PROMPT_MODE="llava_next_training"
    BASE_OUTPUT_ROOT="${PROJECT_ROOT}/generated_llava70b_data_llava_next_training"
    ;;
  false|0|no|"")
    PROMPT_MODE="existing"
    BASE_OUTPUT_ROOT="${PROJECT_ROOT}/generated_llava70b_data"
    ;;
  *)
    echo "USE_LLAVA_NEXT_TRAINING_PROMPTS must be true or false." >&2
    exit 2
    ;;
esac

FEW_SHOT_ARGS=()
case "${USE_FEW_SHOT_EXAMPLES,,}" in
  true|1|yes)
    if ! [[ "${FEW_SHOT_NUM}" =~ ^[1-9][0-9]*$ ]]; then
      echo "FEW_SHOT_NUM must be a positive integer." >&2
      exit 2
    fi
    if [[ "${FEW_SHOT_SPLIT}" != "val" && "${FEW_SHOT_SPLIT}" != "test" ]]; then
      echo "FEW_SHOT_SPLIT must be val or test." >&2
      exit 2
    fi
    FEW_SHOT_ARGS+=(
      --use_few_shot_examples
      --few_shot_num "${FEW_SHOT_NUM}"
      --few_shot_split "${FEW_SHOT_SPLIT}"
    )
    FEW_SHOT_MODE="enabled"
    DEFAULT_OUTPUT_ROOT="${BASE_OUTPUT_ROOT}_few_shot_${FEW_SHOT_SPLIT}_${FEW_SHOT_NUM}"
    ;;
  false|0|no|"")
    FEW_SHOT_MODE="disabled"
    DEFAULT_OUTPUT_ROOT="${BASE_OUTPUT_ROOT}"
    ;;
  *)
    echo "USE_FEW_SHOT_EXAMPLES must be true or false." >&2
    exit 2
    ;;
esac

# An explicitly supplied OUTPUT_ROOT takes priority over the mode-specific
# defaults above.
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}"

echo "prompt_mode=${PROMPT_MODE}"
echo "few_shot_mode=${FEW_SHOT_MODE}"
if [[ "${FEW_SHOT_MODE}" == "enabled" ]]; then
  echo "few_shot_num=${FEW_SHOT_NUM}"
  echo "few_shot_split=${FEW_SHOT_SPLIT}"
fi
echo "output_root=${OUTPUT_ROOT}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/src/generate_llava70b_dataset_text.py" \
  --datasets "${DATASETS}" \
  --split "${SPLIT}" \
  --num_images "${NUM_IMAGES}" \
  --seed "${SEED}" \
  --source_root "${SOURCE_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --device_map "${DEVICE_MAP}" \
  --dtype "${DTYPE}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  "${PROMPT_ARGS[@]}" \
  "${FEW_SHOT_ARGS[@]}" \
  --copy_images \
  --overwrite
