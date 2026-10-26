#!/usr/bin/env bash

# Standalone WebQA/ReMuQ/LLaVA launcher; starts DeepSpeed directly, not another
# shell script. Run one dataset per invocation (default: WebQA):
#   bash scripts/three_stage_retrieval_pipeline_multimodal.sh
#   DATASET=remuq bash scripts/three_stage_retrieval_pipeline_multimodal.sh
#   DATASET=llava GPUS=2 RERANK_NUM=20 bash scripts/three_stage_retrieval_pipeline_multimodal.sh
#   bash scripts/three_stage_retrieval_pipeline_multimodal.sh --dataset-name remuq --query-type composed
# WebQA: text -> image+text; ReMuQ/LLaVA: image+text -> text. For the latter,
# QUERY_TYPE=image and composed both include the query text, not just its image.
# Precompute corpus indexes with the matching task_type and --encode_type corpus
# using disassembleeol_separate+concat. Explicit output/index paths are preserved.
# Full path-related CLI options override environment settings; last value wins.
# Prior caching is supported only for ReMuQ/LLaVA with MODIFY_TYPE=modify_single
# and a caption_generation template. It is disabled by default.

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

# Normalize path and cache-related options before selecting task defaults,
# building paths or validating cache compatibility. Forward other Python args.
EXTRA_ARGS=()
while (($# > 0)); do
    case "${1%%=*}" in
        --dataset-name|--dataset-split|--task-type|--query-type|--data-root|--output-dir|--stage1-sparse-index|--stage2-dense-index|--stage2-sparse-index|--modify_type|--modify-type|--rerank-template|--rerank-type|--rerank-prior-cache-dir|--rerank_prior_cache_dir)
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
                --dataset-name) DATASET="${CLI_VALUE}" ;;
                --dataset-split) DATASET_SPLIT="${CLI_VALUE}" ;;
                --task-type) TASK_TYPE="${CLI_VALUE}" ;;
                --query-type) QUERY_TYPE="${CLI_VALUE}" ;;
                --data-root) DATA_ROOT="${CLI_VALUE}" ;;
                --output-dir) OUTPUT_DIR="${CLI_VALUE}" ;;
                --stage1-sparse-index) STAGE1_SPARSE_INDEX="${CLI_VALUE}" ;;
                --stage2-dense-index) STAGE2_DENSE_INDEX="${CLI_VALUE}" ;;
                --stage2-sparse-index) STAGE2_SPARSE_INDEX="${CLI_VALUE}" ;;
                --modify_type|--modify-type) MODIFY_TYPE="${CLI_VALUE}" ;;
                --rerank-template) RERANK_TEMPLATE="${CLI_VALUE}" ;;
                --rerank-type) RERANK_TYPE="${CLI_VALUE}" ;;
                --rerank-prior-cache-dir|--rerank_prior_cache_dir) RERANK_PRIOR_CACHE_DIR="${CLI_VALUE}" ;;
            esac
            ;;
        --use-rerank-prior-cache|--use_rerank_prior_cache)
            [[ "$1" != *=* ]] || die "${1%%=*} is a flag and does not take a value."
            USE_RERANK_PRIOR_CACHE="true"
            shift
            ;;
        --)
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

GPUS="${GPUS:-4}"
NUM_NODES="${NUM_NODES:-1}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
DATASET="${DATASET:-webqa}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"

[[ "${GPUS}" =~ ^[1-9][0-9]*$ ]] || die "GPUS must be a positive integer, got: ${GPUS}"
[[ "${NUM_NODES}" =~ ^[1-9][0-9]*$ ]] || die "NUM_NODES must be a positive integer, got: ${NUM_NODES}"

TARGET_TYPE="corpus"
case "${DATASET}" in
    webqa)
        EXPECTED_TASK_TYPE="t2it"
        QUERY_TYPE="${QUERY_TYPE:-text}"
        [[ "${QUERY_TYPE}" == "text" ]] || die "webqa requires QUERY_TYPE=text."
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
        DIRECTION_LABEL="imagetext_to_text"
        if [[ "${DATASET}" == "remuq" ]]; then
            DATASET_DIRECTORY="UMRB-ReMuQ"
        else
            DATASET_DIRECTORY="UMRB-LLaVA"
        fi
        DATASET_FILES=("query.parquet" "qrels.parquet" "corpus.parquet")
        ;;
    *) die "Multimodal DATASET must be webqa, remuq or llava, got: ${DATASET}." ;;
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
RERANK_TYPE="${RERANK_TYPE:-pointwise}"
RERANK_NUM="${RERANK_NUM:-5}"
RERANK_BATCH_SIZE="${RERANK_BATCH_SIZE:-1}"
RERANK_TEMPLATE="${RERANK_TEMPLATE:-caption_generation}"
MODIFY_TYPE="${MODIFY_TYPE:-no}"
RERANK_REPS_LOC="${RERANK_REPS_LOC:-after_pad}"
[[ "${RERANK_TYPE}" == "pointwise" ]] || die "This pipeline requires RERANK_TYPE=pointwise."

USE_RERANK_PRIOR_CACHE="${USE_RERANK_PRIOR_CACHE:-false}"
RERANK_PRIOR_CACHE_DIR="${RERANK_PRIOR_CACHE_DIR:-${REPO_ROOT}/cache/rerank_prior}"
PRIOR_CACHE_ARGS=()
if is_true "${USE_RERANK_PRIOR_CACHE}"; then
    [[ "${TASK_TYPE}" == "it2t" ]] || die "webqa does not support the offline rerank prior cache."
    [[ "${MODIFY_TYPE}" == "modify_single" ]] || die "Prior caching requires MODIFY_TYPE=modify_single."
    [[ "${RERANK_TEMPLATE}" == *caption_generation* ]] || \
        die "Prior caching requires a caption_generation rerank template."
    PRIOR_CACHE_ARGS+=(--use-rerank-prior-cache)
fi

# Corpus-side layout matches encode.py; these tasks have no TBPR_TYPE layer.
FILTER_LABEL="no_filter"
CALCULATE_TYPE="concat"
EOL_TYPE="disassembleeol_separate"
MANUAL_LABEL="no_manual"
CLUSTER_LABEL="no_cluster"
USE_MEAN_LABEL="no_mean"
EXPERIMENT_LEAF="${NUM_EXPENDED_TOKENS}_${MANUAL_LABEL}_${SPARSE_LENGTH}_${SPARSE_VALUE_TYPE}_${CLUSTER_LABEL}_${RETRIEVAL_REPS_LOC}_${EOL_TYPE}_${SPARSE_LOWER_OR_UPPER}_${USE_MEAN_LABEL}"
INDEX_RELATIVE_PATH="${ARTIFACT_MODEL_NAME}/${DATASET}/${TARGET_TYPE}/${FILTER_LABEL}/${CALCULATE_TYPE}/${PROMPT_TYPE}/${DATASET_SPLIT}/${EXPERIMENT_LEAF}"
STAGE2_DENSE_INDEX="${STAGE2_DENSE_INDEX:-dense_output/${INDEX_RELATIVE_PATH}}"
STAGE2_SPARSE_INDEX="${STAGE2_SPARSE_INDEX:-sparse_output/${INDEX_RELATIVE_PATH}}"
# Each worker reuses its Stage-1 sparse_concat query encoding in Stage 2.
STAGE1_SPARSE_INDEX="${STAGE1_SPARSE_INDEX:-${STAGE2_SPARSE_INDEX}}"
OUTPUT_DIR="${OUTPUT_DIR:-./three_stage_results/${ARTIFACT_MODEL_NAME}/${DATASET}/${DIRECTION_LABEL}/${DATASET_SPLIT}/sparse_concat_stage1}"
OVERWRITE="${OVERWRITE:-false}"

command -v "${DEEPSPEED_BIN}" >/dev/null 2>&1 || die "DeepSpeed executable was not found: ${DEEPSPEED_BIN}"
require_file "src/three_stage_retrieval_pipeline.py" "pipeline entrypoint"
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
        "no corpus*.pkl found in ${STAGE2_DENSE_INDEX}; generate the candidate index with encode.py --encode_type corpus --task_type ${TASK_TYPE}"
fi
require_lucene_index "${STAGE1_SPARSE_INDEX}" "Stage-1 Lucene index"
require_lucene_index "${STAGE2_SPARSE_INDEX}" "Stage-2 Lucene index"

OVERWRITE_ARGS=()
if is_true "${OVERWRITE}"; then
    OVERWRITE_ARGS+=(--overwrite)
fi

COMMAND=(
    "${DEEPSPEED_BIN}"
    --num_gpus "${GPUS}"
    --num_nodes "${NUM_NODES}"
    src/three_stage_retrieval_pipeline.py
    --dataset-name "${DATASET}"
    --dataset-split "${DATASET_SPLIT}"
    --task-type "${TASK_TYPE}"
    --query-type "${QUERY_TYPE}"
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
    --ranker-device-map "${RANKER_DEVICE_MAP}"
    --ranker-dtype "${RANKER_DTYPE}"
    --rerank-type "${RERANK_TYPE}"
    --rerank-num "${RERANK_NUM}"
    --rerank-batch-size "${RERANK_BATCH_SIZE}"
    --rerank-template "${RERANK_TEMPLATE}"
    --modify_type "${MODIFY_TYPE}"
    "${PRIOR_CACHE_ARGS[@]}"
    --rerank-prior-cache-dir "${RERANK_PRIOR_CACHE_DIR}"
    --rerank-reps-loc "${RERANK_REPS_LOC}"
    --output-dir "${OUTPUT_DIR}"
    "${OVERWRITE_ARGS[@]}"
    "${EXTRA_ARGS[@]}"
)

echo "Dataset:            ${DATASET} (${DATASET_SPLIT}, ${DIRECTION_LABEL})"
echo "Task/query/corpus:  ${TASK_TYPE} / ${QUERY_TYPE} / ${TARGET_TYPE}"
echo "Distributed workers: ${GPUS} GPU process(es) across ${NUM_NODES} node(s)"
echo "Stage 1:            sparse_concat (query-sharded across workers, local encodings reused by Stage 2)"
echo "Stage-1 sparse index:${STAGE1_SPARSE_INDEX}"
echo "Retrieval model:    ${RETRIEVAL_MODEL_PATH}"
echo "Dense corpus index: ${STAGE2_DENSE_INDEX}"
echo "Sparse corpus index:${STAGE2_SPARSE_INDEX}"
echo "Rerank modify type: ${MODIFY_TYPE}"
echo "Prior cache:        ${USE_RERANK_PRIOR_CACHE} (${RERANK_PRIOR_CACHE_DIR})"
echo "Output directory:   ${OUTPUT_DIR}"
printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

exec "${COMMAND[@]}"
