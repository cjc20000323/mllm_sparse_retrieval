#!/usr/bin/env bash
set -euo pipefail

# This script is intentionally independent from the existing search/encode
# scripts.  Change these variables instead of editing the Python program.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/root/autodl-fs/checkpoints/llava-70b}"
SOURCE_ROOT="${SOURCE_ROOT:-/root/autodl-tmp}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/generated_llava70b_data}"

# The requested n is controlled here.  It is applied independently to every
# dataset selected below.
NUM_IMAGES="${NUM_IMAGES:-10}"
SPLIT="${SPLIT:-test}"
SEED="${SEED:-2026}"

# Use "all" for every dataset currently supported by dataset.py.  The alias
# "llava" expands to both llava_t2it and llava_it2t layouts.
DATASETS="${DATASETS:-all}"

# A 70B checkpoint normally needs multiple visible GPUs.  Set
# CUDA_VISIBLE_DEVICES outside this script if a particular GPU list is needed.
DTYPE="${DTYPE:-float16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.9}"

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
  --copy_images \
  --overwrite
