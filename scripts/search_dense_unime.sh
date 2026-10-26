#!/usr/bin/env bash
# Standalone unime search launcher; does not call another shell script.
# Download the complete checkpoint:
#   hf download DeepGlint-AI/UniME-LLaVA-1.6-7B --local-dir ./checkpoints/DeepGlint-AI-UniME-LLaVA-1.6-7B
# Run one dataset per invocation, encoding before searching:
#   DATASET=webqa bash scripts/encode_dense_unime.sh
#   DATASET=webqa bash scripts/search_dense_unime.sh
# Eight datasets: coco, flickr, CUHK-PEDES, ICFG-PEDES, RSTPReid, webqa, remuq, llava.
# COCO/Flickr image queries: use QUERY_TYPE=image on BOTH commands (text candidates).
# ReMuQ/LLaVA image/composed queries always include the image AND question text.
# Defaults: Flickr/test, one process, batch 1, FP16. CPU: DTYPE=float32.
# Matching path labels must be supplied to both commands when overriding defaults.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
die() { echo "[unime-search][error] $*" >&2; exit 2; }
normalize_bool() {
    case "${1,,}" in
        true|1|yes|on|y|t) printf true ;;
        false|0|no|off|n|f) printf false ;;
        *) die "Expected a boolean value, got: $1" ;;
    esac
}
normalize_integer() {
    local value="$1"
    [[ "${value}" =~ ^[0-9]+$ ]] || die "Expected a nonnegative integer, got: ${value}"
    while [[ "${value}" == 0* && "${#value}" -gt 1 ]]; do value="${value#0}"; done
    printf '%s' "${value}"
}

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${PROJECT_ROOT}/checkpoints/DeepGlint-AI-UniME-LLaVA-1.6-7B}"
DENSE_MODEL_TYPE="${DENSE_MODEL_TYPE:-unime}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
DATASET="${DATASET:-flickr}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
TASK_TYPE="${TASK_TYPE:-}"
QUERY_TYPE="${QUERY_TYPE:-}"
ENCODE_TYPE="${ENCODE_TYPE:-}"
DENSE_OUTPUT_DIR="${DENSE_OUTPUT_DIR:-${PROJECT_ROOT}/dense_output}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DTYPE="${DTYPE:-float16}"
NUM_PROCESSES="${NUM_PROCESSES:-${GPUS:-1}}"
MASTER_PORT="${MASTER_PORT:-29500}"

# These compatibility labels mirror dense_encoding.output_directory().
PROMPT_TYPE="${PROMPT_TYPE:-prompt_5}"
CALCULATE_TYPE="${CALCULATE_TYPE:-separate}"
EOL_TYPE="${EOL_TYPE:-prompteol}"
TBPR_TYPE="${TBPR_TYPE:-origin_type}"
NUM_EXPENDED_TOKENS="${NUM_EXPENDED_TOKENS:-0}"
SPARSE_LENGTH="${SPARSE_LENGTH:-128}"
SPARSE_VALUE_TYPE="${SPARSE_VALUE_TYPE:-replace}"
REPS_LOC="${REPS_LOC:-before_pad}"
SPARSE_LOWER_OR_UPPER="${SPARSE_LOWER_OR_UPPER:-lower}"
IS_FILTERED="${IS_FILTERED:-false}"
SPARSE_MANUAL="${SPARSE_MANUAL:-false}"
USE_OUTPUT_EMBEDDING_CLUSTER="${USE_OUTPUT_EMBEDDING_CLUSTER:-false}"
CLUSTER_SUM="${CLUSTER_SUM:-8000}"
SPARSE_VALUE_MEAN="${SPARSE_VALUE_MEAN:-false}"
PASSAGE_REPS="${PASSAGE_REPS:-}"
SAVE_DIR="${SAVE_DIR:-}"
DEPTH="${DEPTH:-1000}"
USE_GPU="${USE_GPU:-false}"

# Resolve all path/direction overrides before deriving the index directory.
EXTRA_ARGS=()
while (($#)); do
    OPTION="${1%%=*}"
    case "${OPTION}" in
        --dataset_name|--dataset_split|--task_type|--query_type|--encode_type|--data_root|--model_name_or_path|--dense_model_type|--dense_output_dir|--per_device_batch_size|--prompt_type|--calculate_type|--eol_type|--tbpr_type|--num_expended_tokens|--sparse_length|--sparse_value_type|--reps_loc|--sparse_lower_or_upper|--cluster_sum|--passage_reps|--save_dir|--depth)
            if [[ "$1" == *=* ]]; then
                VALUE="${1#*=}"; shift
            else
                (($# >= 2)) || die "${OPTION} requires a value."
                [[ "$2" != --* ]] || die "${OPTION} requires a value."
                VALUE="$2"; shift 2
            fi
            [[ -n "${VALUE}" ]] || die "${OPTION} requires a nonempty value."
            case "${OPTION}" in
                --dataset_name) VARIABLE=DATASET ;;
                --per_device_batch_size) VARIABLE=BATCH_SIZE ;;
                *) VARIABLE="${OPTION#--}"; VARIABLE="${VARIABLE^^}" ;;
            esac
            printf -v "${VARIABLE}" '%s' "${VALUE}"
            ;;
        --is_filtered|--sparse_manual|--use_output_embedding_cluster|--sparse_value_mean|--fp16|--bf16|--use_gpu)
            if [[ "$1" == *=* ]]; then
                VALUE="${1#*=}"; shift
            elif (($# >= 2)) && [[ "$2" != --* ]]; then
                VALUE="$2"; shift 2
            else
                VALUE=true; shift
            fi
            VALUE="$(normalize_bool "${VALUE}")"
            case "${OPTION}:${VALUE}" in
                --fp16:true) DTYPE=float16 ;;
                --bf16:true) DTYPE=bfloat16 ;;
                --fp16:false) if [[ "${DTYPE}" == float16 ]]; then DTYPE=float32; fi ;;
                --bf16:false) if [[ "${DTYPE}" == bfloat16 ]]; then DTYPE=float32; fi ;;
                *) VARIABLE="${OPTION#--}"; VARIABLE="${VARIABLE^^}"; printf -v "${VARIABLE}" '%s' "${VALUE}" ;;
            esac
            ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

[[ "${DENSE_MODEL_TYPE}" == unime || "${DENSE_MODEL_TYPE}" == auto ]] || \
    die "This launcher requires DENSE_MODEL_TYPE=unime (or auto)."
case "${DATASET}" in
    coco|flickr) EXPECTED_TASK=ir ;;
    CUHK-PEDES|ICFG-PEDES|RSTPReid) EXPECTED_TASK=tbpr ;;
    webqa) EXPECTED_TASK=t2it ;;
    remuq|llava) EXPECTED_TASK=it2t ;;
    *) die "Unsupported dataset: ${DATASET}" ;;
esac
TASK_TYPE="${TASK_TYPE:-${EXPECTED_TASK}}"
[[ "${TASK_TYPE}" == "${EXPECTED_TASK}" ]] || die "${DATASET} requires TASK_TYPE=${EXPECTED_TASK}."
case "${TASK_TYPE}" in
    ir)
        if [[ -z "${QUERY_TYPE}" ]]; then
            if [[ "${ENCODE_TYPE}" == text ]]; then QUERY_TYPE=image; else QUERY_TYPE=text; fi
        fi
        case "${QUERY_TYPE}" in
            text) ENCODE_TYPE="${ENCODE_TYPE:-image}" ;;
            image) ENCODE_TYPE="${ENCODE_TYPE:-text}" ;;
            *) die "COCO/Flickr require QUERY_TYPE=text or image." ;;
        esac
        ;;
    tbpr) QUERY_TYPE="${QUERY_TYPE:-text}"; ENCODE_TYPE="${ENCODE_TYPE:-image}" ;;
    t2it) QUERY_TYPE="${QUERY_TYPE:-text}"; ENCODE_TYPE="${ENCODE_TYPE:-corpus}" ;;
    it2t) QUERY_TYPE="${QUERY_TYPE:-image}"; ENCODE_TYPE="${ENCODE_TYPE:-corpus}" ;;
esac
case "${TASK_TYPE}:${QUERY_TYPE}:${ENCODE_TYPE}" in
    ir:text:image|ir:image:text|tbpr:text:image|t2it:text:corpus|it2t:image:corpus|it2t:composed:corpus) ;;
    *) die "Conflicting query/candidate direction: ${DATASET}/${TASK_TYPE}/${QUERY_TYPE}/${ENCODE_TYPE}." ;;
esac
[[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || die "BATCH_SIZE must be a positive integer."
[[ "${NUM_PROCESSES}" =~ ^[1-9][0-9]*$ ]] || die "NUM_PROCESSES must be a positive integer."
[[ "${MASTER_PORT}" =~ ^[1-9][0-9]{0,4}$ ]] && ((MASTER_PORT <= 65535)) || die "Invalid MASTER_PORT."
NUM_EXPENDED_TOKENS="$(normalize_integer "${NUM_EXPENDED_TOKENS}")"
SPARSE_LENGTH="$(normalize_integer "${SPARSE_LENGTH}")"
CLUSTER_SUM="$(normalize_integer "${CLUSTER_SUM}")"
IS_FILTERED="$(normalize_bool "${IS_FILTERED}")"
SPARSE_MANUAL="$(normalize_bool "${SPARSE_MANUAL}")"
USE_OUTPUT_EMBEDDING_CLUSTER="$(normalize_bool "${USE_OUTPUT_EMBEDDING_CLUSTER}")"
SPARSE_VALUE_MEAN="$(normalize_bool "${SPARSE_VALUE_MEAN}")"

MODEL_BASENAME="${MODEL_NAME_OR_PATH//\\//}"
while [[ "${MODEL_BASENAME}" == */ ]]; do MODEL_BASENAME="${MODEL_BASENAME%/}"; done
MODEL_BASENAME="${MODEL_BASENAME##*/}"
[[ -n "${MODEL_BASENAME}" && "${MODEL_BASENAME}" != . && "${MODEL_BASENAME}" != .. ]] || \
    die "MODEL_NAME_OR_PATH must identify a checkpoint directory or HF repository."
FILTER_LABEL=no_filter; MANUAL_LABEL=no_manual; CLUSTER_LABEL=no_cluster; MEAN_LABEL=no_mean
if [[ "${IS_FILTERED}" == true ]]; then FILTER_LABEL=filter; fi
if [[ "${SPARSE_MANUAL}" == true ]]; then MANUAL_LABEL=manual; fi
if [[ "${USE_OUTPUT_EMBEDDING_CLUSTER}" == true ]]; then CLUSTER_LABEL="cluster_${CLUSTER_SUM}"; fi
if [[ "${SPARSE_VALUE_MEAN}" == true ]]; then MEAN_LABEL=mean; fi
INDEX_DIR="${DENSE_OUTPUT_DIR%/}/${MODEL_BASENAME}/${DATASET}/${ENCODE_TYPE}/${FILTER_LABEL}/${CALCULATE_TYPE}/${PROMPT_TYPE}/${DATASET_SPLIT}"
if [[ "${TASK_TYPE}" == tbpr ]]; then INDEX_DIR+="/${TBPR_TYPE}"; fi
INDEX_DIR+="/${NUM_EXPENDED_TOKENS}_${MANUAL_LABEL}_${SPARSE_LENGTH}_${SPARSE_VALUE_TYPE}_${CLUSTER_LABEL}_${REPS_LOC}_${EOL_TYPE}_${SPARSE_LOWER_OR_UPPER}_${MEAN_LABEL}"

[[ "${DEPTH}" =~ ^[1-9][0-9]*$ ]] || die "DEPTH must be a positive integer."
USE_GPU="$(normalize_bool "${USE_GPU}")"
PASSAGE_REPS="${PASSAGE_REPS:-${INDEX_DIR}}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/search_results/${MODEL_BASENAME}/${DATASET}/${QUERY_TYPE}/${DATASET_SPLIT}}"
if [[ -f "${PASSAGE_REPS}" ]]; then
    [[ "${PASSAGE_REPS##*/}" == corpus*.pkl ]] || die "PASSAGE_REPS must be a corpus*.pkl file or index directory."
elif [[ -d "${PASSAGE_REPS}" ]]; then
    shopt -s nullglob
    CORPUS_FILES=("${PASSAGE_REPS}"/corpus*.pkl)
    shopt -u nullglob
    FOUND_CORPUS=false
    for CORPUS_FILE in "${CORPUS_FILES[@]}"; do
        if [[ -f "${CORPUS_FILE}" ]]; then FOUND_CORPUS=true; break; fi
    done
    [[ "${FOUND_CORPUS}" == true ]] || die "No corpus*.pkl files in ${PASSAGE_REPS}; run the matching encoder first."
else
    die "Dense corpus index does not exist: ${PASSAGE_REPS}; run scripts/encode_dense_unime.sh with matching settings."
fi
SEARCH_ARGS=(--query_type "${QUERY_TYPE}" --passage_reps "${PASSAGE_REPS}" --save_dir "${SAVE_DIR}"
             --depth "${DEPTH}" --use_gpu "${USE_GPU}")


PRECISION_ARGS=()
case "${DTYPE}" in
    float16) PRECISION_ARGS+=(--fp16) ;;
    bfloat16) PRECISION_ARGS+=(--bf16) ;;
    float32) ;;
    *) die "DTYPE must be float16, bfloat16 or float32." ;;
esac

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
    --model_name_or_path "${MODEL_NAME_OR_PATH}" --dense_model_type unime
    --dataset_name "${DATASET}" --dataset_split "${DATASET_SPLIT}"
    --task_type "${TASK_TYPE}" --encode_type "${ENCODE_TYPE}"
    --data_root "${DATA_ROOT}" --dense_output_dir "${DENSE_OUTPUT_DIR}"
    --per_device_batch_size "${BATCH_SIZE}"
    --prompt_type "${PROMPT_TYPE}" --calculate_type "${CALCULATE_TYPE}" --eol_type "${EOL_TYPE}"
    --tbpr_type "${TBPR_TYPE}" --num_expended_tokens "${NUM_EXPENDED_TOKENS}"
    --sparse_length "${SPARSE_LENGTH}" --sparse_value_type "${SPARSE_VALUE_TYPE}"
    --reps_loc "${REPS_LOC}" --sparse_lower_or_upper "${SPARSE_LOWER_OR_UPPER}"
    --is_filtered "${IS_FILTERED}" --sparse_manual "${SPARSE_MANUAL}"
    --use_output_embedding_cluster "${USE_OUTPUT_EMBEDDING_CLUSTER}" --cluster_sum "${CLUSTER_SUM}"
    --sparse_value_mean "${SPARSE_VALUE_MEAN}"
    --output_dir "${SAVE_DIR%/}/.trainer" "${SEARCH_ARGS[@]}"
    --report_to none "${PRECISION_ARGS[@]}" "${EXTRA_ARGS[@]}")

echo "[unime-search] ${DATASET}/${DATASET_SPLIT}, ${TASK_TYPE}; query=${QUERY_TYPE}; candidates=${ENCODE_TYPE}"
echo "[unime-search] model=${MODEL_NAME_OR_PATH}; processes=${NUM_PROCESSES}; batch=${BATCH_SIZE}; ${DTYPE}"
echo "[unime-search] corpus index: ${PASSAGE_REPS}"
echo "[unime-search] results: ${SAVE_DIR}/dense.xlsx"
printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
exec "${COMMAND[@]}"
