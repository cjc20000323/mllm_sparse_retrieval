#!/usr/bin/env bash
# Estimate-only, eight datasets, one GPU/process; does not call another shell script.
# Stage 1/2 run completely and reuse query encodings; Stage 3 uses FreeRet ETA and stops early.
# Default: WebQA/test, Mistral-7B, FP16, rerank 5, batch 1. No Recall/partial runs.
#   DATASET=webqa bash scripts/estimate_three_stage_retrieval_time.sh
#   CUDA_VISIBLE_DEVICES=2,3 NUM_PROCESSES=2 DATASET=remuq bash scripts/estimate_three_stage_retrieval_time.sh
#   DATASET=coco QUERY_TYPE=text bash scripts/estimate_three_stage_retrieval_time.sh
# Datasets: coco, flickr, CUHK-PEDES, ICFG-PEDES, RSTPReid, webqa, remuq, llava.
# COCO/Flickr default image -> text. ReMuQ/LLaVA image/composed retain question text.
# Match existing disassembleeol_separate+concat corpus indexes; NUM_PROCESSES=auto uses all visible GPUs.
# GPUS is a process-count alias. Single node, always DeepSpeed (including one worker).
# Sampling: 20 complete queries warm up, >=5s intervals, 5 rates with CV strictly <5%.
# Small shards finish fully. CLI overrides environment; last repeated value wins.
# See docs/three_stage_time_estimation.md for timing boundaries and outputs.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

die() {
    echo "Error: $*" >&2
    exit 2
}

require_directory() {
    local path="$1"
    local description="$2"
    [[ -d "${path}" ]] || die "${description} directory does not exist: ${path}"
}

require_file() {
    local path="$1"
    local description="$2"
    [[ -f "${path}" ]] || die "${description} file does not exist: ${path}"
}

require_lucene_index() {
    local path="$1"
    local description="$2"
    local index_dir="${path}"
    if [[ -d "${path}/index" ]]; then
        index_dir="${path}/index"
    fi
    require_directory "${index_dir}" "${description}"
    local -a segments=("${index_dir}"/segments_*)
    [[ -f "${segments[0]:-}" ]] || die \
        "no Lucene segments_* file found in ${index_dir}; build it with pyserini.index.lucene"
}

is_true() {
    case "${1,,}" in
        true|1|yes) return 0 ;;
        false|0|no|"") return 1 ;;
        *) die "expected a boolean value (true/false), got: $1" ;;
    esac
}

# Resolve path-related options before inferring task/query/corpus and direction.
# Leaving them in the trailing arguments would override Python's query settings
# without updating the shell-generated output/index paths or preflight checks.
EXTRA_ARGS=()
while (($# > 0)); do
    case "${1%%=*}" in
        --dataset-name|--dataset-split|--task-type|--query-type|--tbpr-type|--data-root|--output-dir|--stage1-sparse-index|--stage2-dense-index|--stage2-sparse-index|--modify_type|--modify-type|--rerank-template|--rerank-type|--rerank-prior-cache-dir|--rerank_prior_cache_dir|--retrieval-model-name-or-path|--ranker-model-name-or-path|--retrieval-lora-model-path|--retrieval-reps-loc|--retrieval-dtype|--ranker-dtype|--retrieval-batch-size|--rerank-batch-size|--rerank-num|--stage1-depth|--stage2-depth|--hybrid-alpha|--sparse-threads|--prompt-type|--sparse-length|--text-sparse-length|--image-sparse-length|--num-expended-tokens|--sparse-value-type|--sparse-lower-or-upper|--base-model|--artifact-model-name)
            CLI_OPTION="${1%%=*}"
            if [[ "$1" == *=* ]]; then
                CLI_VALUE="${1#*=}"
                shift
            else
                (($# >= 2)) || die "${CLI_OPTION} requires a value."
                [[ "$2" != --* ]] || die "${CLI_OPTION} requires a value."
                CLI_VALUE="$2"
                shift 2
            fi
            [[ -n "${CLI_VALUE}" ]] || die "${CLI_OPTION} requires a non-empty value."
            case "${CLI_OPTION}" in
                --dataset-name) VARIABLE=DATASET ;;
                --retrieval-model-name-or-path) VARIABLE=RETRIEVAL_MODEL_PATH ;;
                --ranker-model-name-or-path) VARIABLE=RANKER_MODEL_PATH ;;
                --modify_type|--modify-type) VARIABLE=MODIFY_TYPE ;;
                --rerank_prior_cache_dir|--rerank-prior-cache-dir) VARIABLE=RERANK_PRIOR_CACHE_DIR ;;
                *) VARIABLE="${CLI_OPTION#--}"; VARIABLE="${VARIABLE//-/_}"; VARIABLE="${VARIABLE^^}" ;;
            esac
            printf -v "${VARIABLE}" '%s' "${CLI_VALUE}"
            ;;
        --use-rerank-prior-cache|--use_rerank_prior_cache|--overwrite|--is-filtered|--sparse-value-mean)
            [[ "$1" != *=* ]] || die "${1%%=*} is a flag and does not take a value."
            case "$1" in
                --use-rerank-prior-cache|--use_rerank_prior_cache) USE_RERANK_PRIOR_CACHE=true ;;
                --overwrite) OVERWRITE=true ;;
                --is-filtered) IS_FILTERED=true ;;
                --sparse-value-mean) SPARSE_VALUE_MEAN=true ;;
            esac
            shift
            ;;
        --)
            # There are no positional arguments; continue normalizing forwarded options.
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_PROCESSES="${NUM_PROCESSES:-${GPUS:-auto}}"
MASTER_PORT="${MASTER_PORT:-29500}"
NUM_NODES="${NUM_NODES:-1}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
DATASET="${DATASET:-webqa}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"

[[ "${NUM_NODES}" == 1 ]] || die "Time estimation supports a single local node (NUM_NODES=1)."

TBPR_TYPE="${TBPR_TYPE:-origin_type}"
case "${DATASET}" in
    flickr|coco)
        EXPECTED_TASK_TYPE="ir"
        QUERY_TYPE="${QUERY_TYPE:-image}"
        DATASET_DIRECTORY="${DATASET}"
        DATASET_FILES=("${DATASET}_${DATASET_SPLIT}.csv")
        case "${QUERY_TYPE}" in
            image) TARGET_TYPE="text"; DIRECTION_LABEL="image_to_text" ;;
            text) TARGET_TYPE="image"; DIRECTION_LABEL="text_to_image" ;;
            *) die "${DATASET} requires QUERY_TYPE=image or text." ;;
        esac
        ;;
    CUHK-PEDES|ICFG-PEDES|RSTPReid)
        EXPECTED_TASK_TYPE="tbpr"
        QUERY_TYPE="${QUERY_TYPE:-text}"
        [[ "${QUERY_TYPE}" == "text" ]] || die "${DATASET} requires QUERY_TYPE=text."
        [[ "${TBPR_TYPE}" == "origin_type" || "${TBPR_TYPE}" == "type" ]] || \
            die "TBPR_TYPE must be origin_type or type."
        TARGET_TYPE="image"
        DIRECTION_LABEL="text_to_image"
        DATASET_DIRECTORY="${DATASET}"
        case "${DATASET}" in
            CUHK-PEDES) DATASET_FILES=("reid_raw.json") ;;
            ICFG-PEDES) DATASET_FILES=("ICFG-PEDES.json") ;;
            RSTPReid) DATASET_FILES=("data_captions.json") ;;
        esac
        ;;
    webqa)
        EXPECTED_TASK_TYPE="t2it"
        QUERY_TYPE="${QUERY_TYPE:-text}"
        [[ "${QUERY_TYPE}" == "text" ]] || die "webqa requires QUERY_TYPE=text."
        TARGET_TYPE="corpus"
        DIRECTION_LABEL="text_to_imagetext"
        DATASET_DIRECTORY="WebQA"
        DATASET_FILES=("query-00000-of-00001.parquet" "qrels-00000-of-00001.parquet"
                       "corpus-00000-of-00002.parquet" "corpus-00001-of-00002.parquet")
        ;;
    remuq|llava)
        EXPECTED_TASK_TYPE="it2t"
        QUERY_TYPE="${QUERY_TYPE:-image}"
        [[ "${QUERY_TYPE}" == "image" || "${QUERY_TYPE}" == "composed" ]] || \
            die "${DATASET} requires QUERY_TYPE=image or composed (image+text query)."
        TARGET_TYPE="corpus"
        DIRECTION_LABEL="imagetext_to_text"
        if [[ "${DATASET}" == "remuq" ]]; then
            DATASET_DIRECTORY="UMRB-ReMuQ"
        else
            DATASET_DIRECTORY="UMRB-LLaVA"
        fi
        DATASET_FILES=("query.parquet" "qrels.parquet" "corpus.parquet")
        ;;
    *) die "DATASET must be coco, flickr, CUHK-PEDES, ICFG-PEDES, RSTPReid, webqa, remuq or llava, got: ${DATASET}" ;;
esac
TASK_TYPE="${TASK_TYPE:-${EXPECTED_TASK_TYPE}}"
[[ "${TASK_TYPE}" == "${EXPECTED_TASK_TYPE}" ]] || \
    die "${DATASET} requires TASK_TYPE=${EXPECTED_TASK_TYPE}, got: ${TASK_TYPE}."

BASE_MODEL="${BASE_MODEL:-llava-hf-llava-v1.6-mistral-7b-hf}"
ARTIFACT_MODEL_NAME="${ARTIFACT_MODEL_NAME:-${BASE_MODEL##*/}}"
RETRIEVAL_MODEL_PATH="${RETRIEVAL_MODEL_PATH:-./checkpoints/${BASE_MODEL}}"
RANKER_MODEL_PATH="${RANKER_MODEL_PATH:-${RETRIEVAL_MODEL_PATH}}"

STAGE1_DEPTH="${STAGE1_DEPTH:-1000}"

STAGE2_DEPTH="${STAGE2_DEPTH:-1000}"
HYBRID_ALPHA="${HYBRID_ALPHA:-0.5}"
# Lucene threads per process; total CPU concurrency scales with GPU workers.
SPARSE_THREADS="${SPARSE_THREADS:-16}"
RETRIEVAL_DEVICE="${RETRIEVAL_DEVICE:-auto}"
RETRIEVAL_DEVICE_MAP="${RETRIEVAL_DEVICE_MAP:-auto}"
RETRIEVAL_DTYPE="${RETRIEVAL_DTYPE:-float16}"
RETRIEVAL_BATCH_SIZE="${RETRIEVAL_BATCH_SIZE:-1}"
RETRIEVAL_REPS_LOC="${RETRIEVAL_REPS_LOC:-after_pad}"

PROMPT_TYPE="${PROMPT_TYPE:-prompt_5}"
SPARSE_LENGTH="${SPARSE_LENGTH:-40}"
TEXT_SPARSE_LENGTH="${TEXT_SPARSE_LENGTH:-${SPARSE_LENGTH}}"
IMAGE_SPARSE_LENGTH="${IMAGE_SPARSE_LENGTH:-${SPARSE_LENGTH}}"
NUM_EXPENDED_TOKENS="${NUM_EXPENDED_TOKENS:-0}"
SPARSE_VALUE_TYPE="${SPARSE_VALUE_TYPE:-sum}"
SPARSE_LOWER_OR_UPPER="${SPARSE_LOWER_OR_UPPER:-lower}"

RANKER_DEVICE_MAP="${RANKER_DEVICE_MAP:-auto}"
RANKER_DTYPE="${RANKER_DTYPE:-float16}"
RETRIEVAL_QWEN_MIN_PIXELS="${RETRIEVAL_QWEN_MIN_PIXELS:-0}"
RETRIEVAL_QWEN_MAX_PIXELS="${RETRIEVAL_QWEN_MAX_PIXELS:-0}"
RANKER_QWEN_MIN_PIXELS="${RANKER_QWEN_MIN_PIXELS:-0}"
RANKER_QWEN_MAX_PIXELS="${RANKER_QWEN_MAX_PIXELS:-0}"
RERANK_TYPE="${RERANK_TYPE:-pointwise}"
[[ "${RERANK_TYPE}" == pointwise ]] || die "Time estimation requires RERANK_TYPE=pointwise."
RERANK_NUM="${RERANK_NUM:-5}"
RERANK_BATCH_SIZE="${RERANK_BATCH_SIZE:-1}"
RERANK_TEMPLATE="${RERANK_TEMPLATE:-caption_generation}"
MODIFY_TYPE="${MODIFY_TYPE:-no}"
# Existing prior correction/cache supports IR image -> text and it2t only.
# Enabling the cache requires MODIFY_TYPE=modify_single; TBPR/WebQA reject it.
USE_RERANK_PRIOR_CACHE="${USE_RERANK_PRIOR_CACHE:-false}"
RERANK_PRIOR_CACHE_DIR="${RERANK_PRIOR_CACHE_DIR:-${REPO_ROOT}/cache/rerank_prior}"
PRIOR_CACHE_ARGS=()
case "${USE_RERANK_PRIOR_CACHE,,}" in
    true|1) PRIOR_CACHE_ARGS+=(--use-rerank-prior-cache --rerank-prior-cache-dir "${RERANK_PRIOR_CACHE_DIR}") ;;
    false|0) ;;
    *) die "USE_RERANK_PRIOR_CACHE must be true/false or 1/0." ;;
esac
RERANK_REPS_LOC="${RERANK_REPS_LOC:-after_pad}"

# These labels must match the corpus-side encode.py configuration. Stage 2 in
# three_stage_retrieval_pipeline.py is fixed to disassembleeol_separate+concat.
FILTER_LABEL="no_filter"
if is_true "${IS_FILTERED:-false}"; then FILTER_LABEL=filter; fi
CALCULATE_TYPE="concat"
EOL_TYPE="disassembleeol_separate"
MANUAL_LABEL="no_manual"
CLUSTER_LABEL="no_cluster"
USE_MEAN_LABEL="no_mean"
if is_true "${SPARSE_VALUE_MEAN:-false}"; then USE_MEAN_LABEL=mean; fi
LABEL_FLAGS=()
if [[ "${FILTER_LABEL}" == filter ]]; then LABEL_FLAGS+=(--is-filtered); fi
if [[ "${USE_MEAN_LABEL}" == mean ]]; then LABEL_FLAGS+=(--sparse-value-mean); fi
# Keep numeric index labels canonical, matching Python.
for VARIABLE in SPARSE_LENGTH TEXT_SPARSE_LENGTH IMAGE_SPARSE_LENGTH NUM_EXPENDED_TOKENS; do
    VALUE="${!VARIABLE}"
    [[ "${VALUE}" =~ ^[0-9]+$ ]] || die "${VARIABLE} must be a nonnegative integer."
    while [[ "${VALUE}" == 0* && "${#VALUE}" -gt 1 ]]; do VALUE="${VALUE#0}"; done
    printf -v "${VARIABLE}" '%s' "${VALUE}"
done

EXPERIMENT_LEAF="${NUM_EXPENDED_TOKENS}_${MANUAL_LABEL}_${SPARSE_LENGTH}_${SPARSE_VALUE_TYPE}_${CLUSTER_LABEL}_${RETRIEVAL_REPS_LOC}_${EOL_TYPE}_${SPARSE_LOWER_OR_UPPER}_${USE_MEAN_LABEL}"
INDEX_PREFIX="${ARTIFACT_MODEL_NAME}/${DATASET}/${TARGET_TYPE}/${FILTER_LABEL}/${CALCULATE_TYPE}/${PROMPT_TYPE}/${DATASET_SPLIT}"
# encode.py places TBPR_TYPE between the split and experiment leaf.
if [[ "${TASK_TYPE}" == "tbpr" ]]; then
    INDEX_PREFIX+="/${TBPR_TYPE}"
fi
INDEX_RELATIVE_PATH="${INDEX_PREFIX}/${EXPERIMENT_LEAF}"
STAGE2_DENSE_INDEX="${STAGE2_DENSE_INDEX:-dense_output/${INDEX_RELATIVE_PATH}}"
STAGE2_SPARSE_INDEX="${STAGE2_SPARSE_INDEX:-sparse_output/${INDEX_RELATIVE_PATH}}"
# Both stages use the same query encoder settings. A different Stage-1 index
# must still have been built with those sparse_concat settings.
STAGE1_SPARSE_INDEX="${STAGE1_SPARSE_INDEX:-${STAGE2_SPARSE_INDEX}}"
OUTPUT_VARIANT="sparse_concat_stage1"
if [[ "${TASK_TYPE}" == "tbpr" ]]; then
    OUTPUT_VARIANT+="/${TBPR_TYPE}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-./three_stage_time_estimates/${ARTIFACT_MODEL_NAME}/${DATASET}/${DIRECTION_LABEL}/${DATASET_SPLIT}/${OUTPUT_VARIANT}}"
OVERWRITE="${OVERWRITE:-false}"

require_file "src/estimate_three_stage_retrieval_time.py" "estimation entrypoint"
require_directory "${DATA_ROOT}/data/${DATASET_DIRECTORY}" "dataset"
for DATASET_FILE in "${DATASET_FILES[@]}"; do
    require_file "${DATA_ROOT}/data/${DATASET_DIRECTORY}/${DATASET_FILE}" "dataset annotation"
done
require_directory "${RETRIEVAL_MODEL_PATH}" "retrieval checkpoint"
require_directory "${RANKER_MODEL_PATH}" "ranker checkpoint"

if [[ -f "${STAGE2_DENSE_INDEX}" ]]; then
    [[ "$(basename -- "${STAGE2_DENSE_INDEX}")" == corpus*.pkl ]] || \
        die "STAGE2_DENSE_INDEX is a file but is not a corpus*.pkl file: ${STAGE2_DENSE_INDEX}"
else
    require_directory "${STAGE2_DENSE_INDEX}" "Stage-2 dense index"
    shopt -s nullglob
    dense_corpus_files=("${STAGE2_DENSE_INDEX}"/corpus*.pkl)
    shopt -u nullglob
    ((${#dense_corpus_files[@]} > 0)) || die \
        "no corpus*.pkl found in ${STAGE2_DENSE_INDEX}; generate the candidate index with encode.py --encode_type ${TARGET_TYPE} --task_type ${TASK_TYPE}"
fi

require_lucene_index "${STAGE1_SPARSE_INDEX}" "Stage-1 Lucene index"
require_lucene_index "${STAGE2_SPARSE_INDEX}" "Stage-2 Lucene index"

OVERWRITE_ARGS=()
if is_true "${OVERWRITE}"; then
    OVERWRITE_ARGS+=(--overwrite)
fi

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

LORA_ARGS=()
if [[ -n "${RETRIEVAL_LORA_MODEL_PATH:-}" ]]; then
    LORA_ARGS+=(--retrieval-lora-model-path "${RETRIEVAL_LORA_MODEL_PATH}")
fi

COMMAND=(
    env -u CUDA_VISIBLE_DEVICES "${PYTHON_BIN}" -m deepspeed.launcher.runner
    --hostfile "" --include "localhost:${SELECTED_GPU_LIST}"
    --master_addr 127.0.0.1 --master_port "${MASTER_PORT}"
    "${REPO_ROOT}/src/estimate_three_stage_retrieval_time.py"
    --dataset-name "${DATASET}"
    --dataset-split "${DATASET_SPLIT}"
    --task-type "${TASK_TYPE}"
    --query-type "${QUERY_TYPE}"
    --tbpr-type "${TBPR_TYPE}"
    --data-root "${DATA_ROOT}"
    --stage1-method sparse_concat
    --stage1-sparse-index "${STAGE1_SPARSE_INDEX}"
    --stage1-depth "${STAGE1_DEPTH}"
    --stage2-dense-index "${STAGE2_DENSE_INDEX}"
    --stage2-sparse-index "${STAGE2_SPARSE_INDEX}"
    --stage2-depth "${STAGE2_DEPTH}"
    --hybrid-alpha "${HYBRID_ALPHA}"
    --sparse-threads "${SPARSE_THREADS}"
    --retrieval-model-name-or-path "${RETRIEVAL_MODEL_PATH}"
    --retrieval-qwen-min-pixels "${RETRIEVAL_QWEN_MIN_PIXELS}"
    --retrieval-qwen-max-pixels "${RETRIEVAL_QWEN_MAX_PIXELS}"
    --retrieval-device "${RETRIEVAL_DEVICE}"
    --retrieval-device-map "${RETRIEVAL_DEVICE_MAP}"
    --retrieval-dtype "${RETRIEVAL_DTYPE}"
    --retrieval-batch-size "${RETRIEVAL_BATCH_SIZE}"
    --retrieval-reps-loc "${RETRIEVAL_REPS_LOC}"
    --prompt-type "${PROMPT_TYPE}"
    --sparse-length "${SPARSE_LENGTH}"
    --text-sparse-length "${TEXT_SPARSE_LENGTH}"
    --image-sparse-length "${IMAGE_SPARSE_LENGTH}"
    --num-expended-tokens "${NUM_EXPENDED_TOKENS}"
    --sparse-value-type "${SPARSE_VALUE_TYPE}"
    --sparse-lower-or-upper "${SPARSE_LOWER_OR_UPPER}"
    --ranker-model-name-or-path "${RANKER_MODEL_PATH}"
    --ranker-qwen-min-pixels "${RANKER_QWEN_MIN_PIXELS}"
    --ranker-qwen-max-pixels "${RANKER_QWEN_MAX_PIXELS}"
    --ranker-device-map "${RANKER_DEVICE_MAP}"
    --ranker-dtype "${RANKER_DTYPE}"
    --rerank-type "${RERANK_TYPE}"
    --rerank-num "${RERANK_NUM}"
    --rerank-batch-size "${RERANK_BATCH_SIZE}"
    --rerank-template "${RERANK_TEMPLATE}"
    --modify_type "${MODIFY_TYPE}"
    "${PRIOR_CACHE_ARGS[@]}"
    --rerank-reps-loc "${RERANK_REPS_LOC}"
    --output-dir "${OUTPUT_DIR}"
    "${OVERWRITE_ARGS[@]}" "${LABEL_FLAGS[@]}" "${LORA_ARGS[@]}"
    "${EXTRA_ARGS[@]}"
)

echo "Dataset:            ${DATASET} (${DATASET_SPLIT}, ${DIRECTION_LABEL})"
echo "Task/query/corpus:  ${TASK_TYPE} / ${QUERY_TYPE} / ${TARGET_TYPE}"
if [[ "${TASK_TYPE}" == "tbpr" ]]; then
    echo "TBPR view type:     ${TBPR_TYPE}"
fi
echo "Distributed workers: ${NUM_PROCESSES}; physical GPU IDs=${SELECTED_GPU_LIST}; single node; MASTER_PORT=${MASTER_PORT}"
echo "Estimate mode: Stage 1/2 complete; Stage 3 stops after stable complete-query throughput."
echo "Compute total excludes setup, evaluation, output and estimation-control collectives."
echo "Stage 1:            sparse_concat (query-sharded across workers, local encodings reused by Stage 2)"
echo "Stage-1 sparse index:${STAGE1_SPARSE_INDEX}"
echo "Retrieval model:    ${RETRIEVAL_MODEL_PATH}"
echo "Dense corpus index: ${STAGE2_DENSE_INDEX}"
echo "Sparse corpus index:${STAGE2_SPARSE_INDEX}"
echo "Rerank modify type: ${MODIFY_TYPE}"
echo "Time report:        ${OUTPUT_DIR}/time_estimate.json"
printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

exec "${COMMAND[@]}"
