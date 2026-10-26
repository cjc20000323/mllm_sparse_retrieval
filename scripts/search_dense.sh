#!/usr/bin/env bash
# Online dense queries against a matching encode_dense.py corpus.
#   PASSAGE_REPS=/path/to/corpus/shards bash scripts/search_dense.sh
#   MODEL_NAME_OR_PATH=./checkpoints/DeepGlint-AI-UniME-LLaVA-1.6-7B \
#     DATASET=remuq PASSAGE_REPS=/path/to/remuq/index bash scripts/search_dense.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
die() { echo "[dense-search][error] $*" >&2; exit 2; }

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${PROJECT_ROOT}/checkpoints/BAAI-BGE-VL-MLLM-S1}"
DENSE_MODEL_TYPE="${DENSE_MODEL_TYPE:-auto}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
DATASET="${DATASET:-flickr}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
PASSAGE_REPS="${PASSAGE_REPS:-}"
SAVE_DIR="${SAVE_DIR:-}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DEPTH="${DEPTH:-1000}"
DTYPE="${DTYPE:-float16}"
NUM_PROCESSES="${NUM_PROCESSES:-${GPUS:-1}}"
MASTER_PORT="${MASTER_PORT:-29500}"

EXTRA_ARGS=()
while (($#)); do
    case "${1%%=*}" in
        --dataset_name|--dataset_split|--task_type|--query_type|--data_root|--model_name_or_path|--dense_model_type|--passage_reps|--save_dir|--per_device_batch_size|--depth)
            OPTION="${1%%=*}"
            if [[ "$1" == *=* ]]; then
                VALUE="${1#*=}"; shift
            else
                (($# >= 2)) || die "${OPTION} requires a value."
                [[ "$2" != --* ]] || die "${OPTION} requires a value."
                VALUE="$2"; shift 2
            fi
            [[ -n "${VALUE}" ]] || die "${OPTION} requires a nonempty value."
            case "${OPTION}" in
                --dataset_name) DATASET="${VALUE}" ;;
                --dataset_split) DATASET_SPLIT="${VALUE}" ;;
                --task_type) TASK_TYPE="${VALUE}" ;;
                --query_type) QUERY_TYPE="${VALUE}" ;;
                --data_root) DATA_ROOT="${VALUE}" ;;
                --model_name_or_path) MODEL_NAME_OR_PATH="${VALUE}" ;;
                --dense_model_type) DENSE_MODEL_TYPE="${VALUE}" ;;
                --passage_reps) PASSAGE_REPS="${VALUE}" ;;
                --save_dir) SAVE_DIR="${VALUE}" ;;
                --per_device_batch_size) BATCH_SIZE="${VALUE}" ;;
                --depth) DEPTH="${VALUE}" ;;
            esac
            ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

case "${DATASET}" in
    coco|flickr) EXPECTED_TASK=ir; QUERY_TYPE="${QUERY_TYPE:-text}" ;;
    CUHK-PEDES|ICFG-PEDES|RSTPReid) EXPECTED_TASK=tbpr; QUERY_TYPE="${QUERY_TYPE:-text}" ;;
    webqa) EXPECTED_TASK=t2it; QUERY_TYPE="${QUERY_TYPE:-text}" ;;
    remuq|llava) EXPECTED_TASK=it2t; QUERY_TYPE="${QUERY_TYPE:-image}" ;;
    *) die "Unsupported dataset: ${DATASET}" ;;
esac
TASK_TYPE="${TASK_TYPE:-${EXPECTED_TASK}}"
[[ "${TASK_TYPE}" == "${EXPECTED_TASK}" ]] || die "${DATASET} requires TASK_TYPE=${EXPECTED_TASK}."
case "${TASK_TYPE}:${QUERY_TYPE}" in
    ir:text|ir:image|tbpr:text|t2it:text|it2t:image|it2t:composed) ;;
    *) die "Invalid QUERY_TYPE=${QUERY_TYPE} for ${DATASET}/${TASK_TYPE}." ;;
esac
case "${DENSE_MODEL_TYPE}" in auto|mmret|unime) ;; *) die "Invalid DENSE_MODEL_TYPE." ;; esac
[[ -n "${PASSAGE_REPS}" ]] || die "Set PASSAGE_REPS or --passage_reps to the matching corpus index."
[[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || die "BATCH_SIZE must be a positive integer."
[[ "${DEPTH}" =~ ^[1-9][0-9]*$ ]] || die "DEPTH must be a positive integer."
[[ "${NUM_PROCESSES}" =~ ^[1-9][0-9]*$ ]] || die "NUM_PROCESSES must be a positive integer."
[[ "${MASTER_PORT}" =~ ^[1-9][0-9]{0,4}$ ]] && ((MASTER_PORT <= 65535)) || die "Invalid MASTER_PORT."

PRECISION_ARGS=()
case "${DTYPE}" in
    float16) PRECISION_ARGS+=(--fp16) ;;
    bfloat16) PRECISION_ARGS+=(--bf16) ;;
    float32) ;;
    *) die "DTYPE must be float16, bfloat16 or float32." ;;
esac
OUTPUT_ARGS=()
if [[ -n "${SAVE_DIR}" ]]; then OUTPUT_ARGS+=(--save_dir "${SAVE_DIR}"); fi

COMMAND=("${PYTHON_BIN}")
if ((NUM_PROCESSES > 1)); then
    LAUNCH_ARGS=(--hostfile "" --master_addr 127.0.0.1 --master_port "${MASTER_PORT}")
    if [[ "${CUDA_VISIBLE_DEVICES+x}" == x ]]; then
        [[ "${CUDA_VISIBLE_DEVICES}" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]] || \
            die "CUDA_VISIBLE_DEVICES must contain numeric GPU IDs, e.g. 2,3."
        IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
        ((NUM_PROCESSES <= ${#GPU_IDS[@]})) || die "NUM_PROCESSES exceeds the selected GPU count."
        SEEN=","
        for GPU_ID in "${GPU_IDS[@]}"; do
            [[ "${SEEN}" != *",${GPU_ID},"* ]] || die "Duplicate GPU ID: ${GPU_ID}"
            SEEN+="${GPU_ID},"
        done
        SELECTED=("${GPU_IDS[@]:0:NUM_PROCESSES}")
        SELECTED_LIST="$(IFS=','; echo "${SELECTED[*]}")"
        LAUNCH_ARGS+=(--include "localhost:${SELECTED_LIST}")
        COMMAND=(env -u CUDA_VISIBLE_DEVICES "${PYTHON_BIN}")
    else
        LAUNCH_ARGS+=(--num_gpus "${NUM_PROCESSES}")
    fi
    COMMAND+=(-m deepspeed.launcher.runner "${LAUNCH_ARGS[@]}")
fi
COMMAND+=("${PROJECT_ROOT}/src/search_dense.py"
    --model_name_or_path "${MODEL_NAME_OR_PATH}" --dense_model_type "${DENSE_MODEL_TYPE}"
    --dataset_name "${DATASET}" --dataset_split "${DATASET_SPLIT}"
    --task_type "${TASK_TYPE}" --query_type "${QUERY_TYPE}"
    --data_root "${DATA_ROOT}" --passage_reps "${PASSAGE_REPS}"
    --per_device_batch_size "${BATCH_SIZE}" --depth "${DEPTH}"
    --output_dir "${PROJECT_ROOT}/output/.dense-search-trainer" --report_to none
    "${OUTPUT_ARGS[@]}" "${PRECISION_ARGS[@]}" "${EXTRA_ARGS[@]}")
echo "[dense-search] ${DATASET}/${DATASET_SPLIT}, ${TASK_TYPE}/${QUERY_TYPE}; processes=${NUM_PROCESSES}; ${DTYPE}"
printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
exec "${COMMAND[@]}"
