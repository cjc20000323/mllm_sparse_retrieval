#!/usr/bin/env bash
# Standalone FreeRet estimate-only launcher, not the three-stage pipeline.
# Default: all eight datasets; COCO/Flickr run both directions.
#   bash scripts/estimate_freeret_retrieval_time.sh
#   CUDA_VISIBLE_DEVICES=2,3 NUM_PROCESSES=2 DATASETS=webqa \
#     bash scripts/estimate_freeret_retrieval_time.sh
# Configure this launcher through environment variables, as with freeret_retrieval.sh.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

die() {
  echo "[freeret-time][error] $*" >&2
  exit 2
}

(($# == 0)) || die "Use environment variables (for example DATASETS=webqa); positional/CLI arguments are not supported."

PYTHON_BIN="${PYTHON_BIN:-python}"
# Same model/default retrieval settings as scripts/freeret_retrieval.sh.
# Other model families supported by the FreeRet entrypoint can be selected
# using MODEL_PATH and their matching PYTHON_BIN environment.
MODEL_PATH="${MODEL_PATH:-/root/autodl-fs/checkpoints/Qwen2.5-VL-7B-Instruct}"
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/freeret_time_estimates}"
# coco, flickr, CUHK-PEDES, ICFG-PEDES, RSTPReid, webqa, remuq, llava,
# a comma-separated subset, or all. FreeRet processes these tasks sequentially
# inside one distributed launch, with each task's work sharded across workers.
DATASETS="${DATASETS:-all}"
SPLIT="${SPLIT:-test}"
# auto = COCO/Flickr t2i+i2t; TBPR t2i; WebQA t2it; ReMuQ/LLaVA it2t.
DIRECTION="${DIRECTION:-auto}"
DTYPE="${DTYPE:-bfloat16}"
NUM_PROCESSES="${NUM_PROCESSES:-auto}"
NUM_NODES="${NUM_NODES:-1}"
# Compatibility argument only: workers bind a complete model to their local
# GPU and ignore DEVICE_MAP, including NUM_PROCESSES=1.
DEVICE_MAP="${DEVICE_MAP:-auto}"
MASTER_PORT="${MASTER_PORT:-29500}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
# 0 uses the dataset-specific FreeRet image setting.
MAX_PIXELS="${MAX_PIXELS:-0}"
TARGET_BATCH_SIZE="${TARGET_BATCH_SIZE:-8}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-4}"
RERANK_BATCH_SIZE="${RERANK_BATCH_SIZE:-2}"
RERANK_NUM="${RERANK_NUM:-50}"
STAGE1_TOPK="${STAGE1_TOPK:-50}"
SIMILARITY_CHUNK_SIZE="${SIMILARITY_CHUNK_SIZE:-8192}"
MAX_TARGET_GPU_GB="${MAX_TARGET_GPU_GB:-4}"
METRICS_KS="${METRICS_KS:-1 5 10}"
MAX_QUERIES="${MAX_QUERIES:--1}"
SEED="${SEED:-2026}"

# Always estimate-only, regardless of an inherited ESTIMATE_TIME_ONLY value.
# Targets remain complete (or cached). Query sampling skips 20 warmup query
# batches, samples every >=5 seconds, and stops after 5 rates with CV <5%.
# Short/unstable shards finish and report actual time instead. The existing
# Python entrypoint owns this policy; this script does not reimplement it.
EXTRA_ARGS=(--estimate_time_only)
if [[ "${CACHE_TARGETS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--cache_targets)
fi

[[ "${NUM_NODES}" == "1" ]] || die "NUM_NODES must be 1; only single-node launches are supported."
if [[ "${NUM_PROCESSES}" != "auto" && ! "${NUM_PROCESSES}" =~ ^[1-9][0-9]*$ ]]; then
  die "NUM_PROCESSES must be auto or a positive integer."
fi
if [[ ! "${MASTER_PORT}" =~ ^[1-9][0-9]{0,4}$ ]] || (( MASTER_PORT > 65535 )); then
  die "MASTER_PORT must be an integer between 1 and 65535."
fi
VISIBLE_GPU_COUNT="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')" || \
  die "Cannot query GPUs using PYTHON_BIN=${PYTHON_BIN}."
[[ "${VISIBLE_GPU_COUNT}" =~ ^(0|[1-9][0-9]*)$ ]] || die "Invalid visible GPU count: ${VISIBLE_GPU_COUNT}."
(( VISIBLE_GPU_COUNT > 0 )) || die "DeepSpeed requires at least one visible CUDA GPU; no Python/CPU fallback is enabled."

# Preserve physical GPU selection. --num_gpus can discard CUDA_VISIBLE_DEVICES;
# using logical IDs with --include could select different physical devices.
VISIBLE_GPU_IDS=()
if [[ "${CUDA_VISIBLE_DEVICES+x}" == "x" ]]; then
  [[ "${CUDA_VISIBLE_DEVICES}" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]] || \
    die "CUDA_VISIBLE_DEVICES must be a comma-separated list of numeric GPU IDs (for example 2,3); UUID/MIG selectors are not supported by this launcher."
  IFS=',' read -r -a VISIBLE_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
  SEEN_GPU_IDS=","
  for GPU_ID in "${VISIBLE_GPU_IDS[@]}"; do
    [[ "${SEEN_GPU_IDS}" != *",${GPU_ID},"* ]] || die "CUDA_VISIBLE_DEVICES contains duplicate GPU ID ${GPU_ID}."
    SEEN_GPU_IDS+="${GPU_ID},"
  done
  (( ${#VISIBLE_GPU_IDS[@]} == VISIBLE_GPU_COUNT )) || \
    die "CUDA_VISIBLE_DEVICES lists ${#VISIBLE_GPU_IDS[@]} GPUs but PyTorch sees ${VISIBLE_GPU_COUNT}; check the selected GPU IDs."
else
  for ((GPU_ID=0; GPU_ID<VISIBLE_GPU_COUNT; GPU_ID++)); do
    VISIBLE_GPU_IDS+=("${GPU_ID}")
  done
fi
if [[ "${NUM_PROCESSES}" == "auto" ]]; then
  NUM_PROCESSES="${VISIBLE_GPU_COUNT}"
fi
if (( ${#NUM_PROCESSES} > ${#VISIBLE_GPU_COUNT} )) || (( NUM_PROCESSES > VISIBLE_GPU_COUNT )); then
  die "NUM_PROCESSES=${NUM_PROCESSES} exceeds visible GPU count ${VISIBLE_GPU_COUNT}."
fi
SELECTED_GPU_IDS=("${VISIBLE_GPU_IDS[@]:0:NUM_PROCESSES}")
SELECTED_GPU_LIST="$(IFS=','; echo "${SELECTED_GPU_IDS[*]}")"
if ! "${PYTHON_BIN}" -c 'import deepspeed.launcher.runner'; then
  die "DeepSpeed could not be imported using PYTHON_BIN=${PYTHON_BIN}; use an environment with a working DeepSpeed installation."
fi

read -r -a METRICS_K_VALUES <<< "${METRICS_KS}"
COMMAND=(
  env -u CUDA_VISIBLE_DEVICES "${PYTHON_BIN}" -m deepspeed.launcher.runner
  # Empty hostfile disables automatic /job/hostfile discovery.
  --hostfile "" --include "localhost:${SELECTED_GPU_LIST}"
  --master_addr 127.0.0.1 --master_port "${MASTER_PORT}"
  "${PROJECT_ROOT}/src/freeret_retrieval.py"
  --datasets "${DATASETS}"
  --split "${SPLIT}"
  --direction "${DIRECTION}"
  --data_root "${DATASET_ROOT}"
  --output_dir "${OUTPUT_DIR}"
  --model_path "${MODEL_PATH}"
  --dtype "${DTYPE}"
  --device_map "${DEVICE_MAP}"
  --attn_implementation "${ATTN_IMPLEMENTATION}"
  --max_pixels "${MAX_PIXELS}"
  --target_batch_size "${TARGET_BATCH_SIZE}"
  --query_batch_size "${QUERY_BATCH_SIZE}"
  --rerank_batch_size "${RERANK_BATCH_SIZE}"
  --rerank_num "${RERANK_NUM}"
  --stage1_topk "${STAGE1_TOPK}"
  --similarity_chunk_size "${SIMILARITY_CHUNK_SIZE}"
  --max_target_gpu_gb "${MAX_TARGET_GPU_GB}"
  --metrics_ks "${METRICS_K_VALUES[@]}"
  --max_queries "${MAX_QUERIES}"
  --seed "${SEED}"
  "${EXTRA_ARGS[@]}"
)
echo "[freeret-time] Estimate-only; DATASETS=${DATASETS}; DIRECTION=${DIRECTION}; OUTPUT_DIR=${OUTPUT_DIR}."
echo "[freeret-time] Saves per-task time_estimate.json and model-level time_estimate_summary.json; no retrieval results or Recall."
echo "[deepspeed] single node; ${NUM_PROCESSES} process(es); physical GPU IDs=${SELECTED_GPU_LIST}; MASTER_PORT=${MASTER_PORT}."
echo "[deepspeed] One complete model per GPU; DEVICE_MAP is ignored, including NUM_PROCESSES=1."
printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
exec "${COMMAND[@]}"
