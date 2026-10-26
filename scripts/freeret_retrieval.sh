#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-fs/checkpoints/Qwen2.5-VL-7B-Instruct}"
# Supported model families include Qwen2.5-VL, LLaVA-NeXT Mistral-7B
# (llava-hf/llava-v1.6-mistral-7b-hf), LLaVA-NeXT Llama-3-8B
# (llava-hf/llama3-llava-next-8b-hf), LLaVA-OV
# (llava-hf/llava-onevision-qwen2-7b-ov-hf), and LLaVA-OV-1.5
# (lmms-lab/LLaVA-OneVision-1.5-8B-Instruct). Set MODEL_PATH to the local
# checkpoint. OV-1.5 uses official remote code and therefore requires a
# dedicated environment with transformers==4.53.0 and qwen-vl-utils, e.g.:
#   PYTHON_BIN=/opt/conda/envs/freeret_ov15/bin/python \
#   MODEL_PATH=/root/gpufree-share/checkpoints/lmms-lab-LLaVA-OneVision-1.5-8B-Instruct \
#   bash scripts/freeret_retrieval.sh
# The Python entrypoint checks this version before allocating model weights.
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/freeret}"

# Supported values: coco, flickr, CUHK-PEDES, ICFG-PEDES, RSTPReid,
# webqa, remuq, llava, or all.
DATASETS="${DATASETS:-coco}"
SPLIT="${SPLIT:-test}"
# auto = coco/flickr both directions; other datasets use their dataset.py
# direction: TBPR=t2i, WebQA=t2it, ReMuQ/LLaVA=it2t.
DIRECTION="${DIRECTION:-auto}"

DTYPE="${DTYPE:-bfloat16}"
# Always launch through DeepSpeed, including NUM_PROCESSES=1.
# auto = one process per visible GPU; each GPU must fit a complete model.
# Example: CUDA_VISIBLE_DEVICES=2,3 NUM_PROCESSES=2 MASTER_PORT=29501 \
#   bash scripts/freeret_retrieval.sh
NUM_PROCESSES="${NUM_PROCESSES:-auto}"
# Retained for CLI compatibility; DeepSpeed workers ignore DEVICE_MAP and
# bind the entire model to their local GPU, even with one worker.
DEVICE_MAP="${DEVICE_MAP:-auto}"
MASTER_PORT="${MASTER_PORT:-29500}"
# OV-1.5 is forced to SDPA by the Python entrypoint. SDPA is also the safe
# default for all other supported model families when flash-attn is absent.
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
# 0 follows the per-dataset values registered in freeret_retrieval.py
# (official MSCOCO/WebQA values are 256); set a positive value to override.
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
# Estimate each configured dataset/direction, then exit without retrieval
# results or recall metrics. Targets must still be fully encoded (or cached).
# Skip 20 batches per stage; sample every >=5s; require 5 rates with CV <5%.
# Example: ESTIMATE_TIME_ONLY=true bash scripts/freeret_retrieval.sh
# false (default) completes all queries and saves actual retrieval/rerank seconds
# in each task's timing_metrics.json and the model directory's timing_summary.json.
ESTIMATE_TIME_ONLY="${ESTIMATE_TIME_ONLY:-false}"

EXTRA_ARGS=()
case "${ESTIMATE_TIME_ONLY,,}" in
  true|1) EXTRA_ARGS+=(--estimate_time_only) ;;
  false|0) ;;
  *) echo "ESTIMATE_TIME_ONLY must be true/false or 1/0." >&2; exit 2 ;;
esac
if [[ "${CACHE_TARGETS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--cache_targets)
fi

die() {
  echo "[freeret][error] $*" >&2
  exit 2
}

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

# Resolve physical IDs before DeepSpeed runs. Passing --num_gpus would discard
# CUDA_VISIBLE_DEVICES, while passing logical IDs to --include could select
# different physical GPUs. The runner receives only our explicit physical list.
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
  # An empty path disables /job/hostfile discovery and guarantees local launch.
  # /dev/null is NOT suitable: DeepSpeed rejects an existing empty hostfile.
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
echo "[deepspeed] single node; ${NUM_PROCESSES} process(es); physical GPU IDs=${SELECTED_GPU_LIST}; MASTER_PORT=${MASTER_PORT}."
echo "[deepspeed] One complete model per GPU; DEVICE_MAP is ignored, including NUM_PROCESSES=1."
printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
exec "${COMMAND[@]}"
