#!/usr/bin/env bash
set -euo pipefail

# Generate dataset.py-compatible validation text with fixed few-shot
# demonstrations. This entrypoint is independent from the existing zero-shot
# generation script and always enables --use_few_shot_examples.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/root/autodl-fs/checkpoints/llava-70b}"
SOURCE_ROOT="${SOURCE_ROOT:-/root/autodl-tmp}"

# NUM_IMAGES is applied independently to every selected dataset.
NUM_IMAGES="${NUM_IMAGES:-10}"
SPLIT="${SPLIT:-test}"
SEED="${SEED:-2026}"
DATASETS="${DATASETS:-all}"

DTYPE="${DTYPE:-float16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.9}"

# Few-shot generation is mandatory in this script. The demonstrations may
# come from either the validation or test split.
FEW_SHOT_NUM="${FEW_SHOT_NUM:-2}"
FEW_SHOT_SPLIT="${FEW_SHOT_SPLIT:-test}"

# Use LLaVA-NeXT training-style prompts by default so that this script's output
# matches encode_generated_validation_llava_next_training.sh.
USE_LLAVA_NEXT_TRAINING_PROMPTS="${USE_LLAVA_NEXT_TRAINING_PROMPTS:-true}"

if ! [[ "${NUM_IMAGES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_IMAGES must be a positive integer." >&2
  exit 2
fi

if ! [[ "${FEW_SHOT_NUM}" =~ ^[1-9][0-9]*$ ]]; then
  echo "FEW_SHOT_NUM must be a positive integer." >&2
  exit 2
fi

if [[ "${FEW_SHOT_SPLIT}" != "val" && "${FEW_SHOT_SPLIT}" != "test" ]]; then
  echo "FEW_SHOT_SPLIT must be val or test." >&2
  exit 2
fi

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

DEFAULT_OUTPUT_ROOT="${BASE_OUTPUT_ROOT}_few_shot_${FEW_SHOT_SPLIT}_${FEW_SHOT_NUM}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}"

echo "datasets=${DATASETS}"
echo "split=${SPLIT}"
echo "num_images=${NUM_IMAGES}"
echo "prompt_mode=${PROMPT_MODE}"
echo "few_shot_mode=enabled"
echo "few_shot_example_mode=text_only"
echo "few_shot_num=${FEW_SHOT_NUM}"
echo "few_shot_split=${FEW_SHOT_SPLIT}"
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
  --use_few_shot_examples \
  --few_shot_text_only \
  --few_shot_num "${FEW_SHOT_NUM}" \
  --few_shot_split "${FEW_SHOT_SPLIT}" \
  --copy_images \
  --overwrite
