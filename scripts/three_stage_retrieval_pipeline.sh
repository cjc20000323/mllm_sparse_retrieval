#!/usr/bin/env bash

# Run the three-stage retrieval pipeline on the original dataset:
#   1. LLaVA sparse_concat coarse retrieval (query-sharded, one encoding per query)
#   2. LLaVA dense+sparse hybrid retrieval (reuse each rank's local encodings)
#   3. LLaVA pointwise reranking
# Select one dataset per run, for example:
#   DATASET=CUHK-PEDES TBPR_TYPE=origin_type bash scripts/three_stage_retrieval_pipeline.sh
#   DATASET=webqa bash scripts/three_stage_retrieval_pipeline.sh
#   DATASET=remuq bash scripts/three_stage_retrieval_pipeline.sh
# COCO/Flickr default to image -> text; QUERY_TYPE=text selects text -> image.
# The equivalent CLI form is: bash scripts/three_stage_retrieval_pipeline.sh --query-type text
# Path-related CLI options override environment variables before paths/checks
# are resolved; repeated options use the last value. Explicit paths are kept.
# WebQA retrieves image+text corpus items; ReMuQ/LLaVA use image+text queries.

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
        --dataset-name|--dataset-split|--task-type|--query-type|--tbpr-type|--data-root|--output-dir|--stage1-sparse-index|--stage2-dense-index|--stage2-sparse-index)
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
                --tbpr-type) TBPR_TYPE="${CLI_VALUE}" ;;
                --data-root) DATA_ROOT="${CLI_VALUE}" ;;
                --output-dir) OUTPUT_DIR="${CLI_VALUE}" ;;
                --stage1-sparse-index) STAGE1_SPARSE_INDEX="${CLI_VALUE}" ;;
                --stage2-dense-index) STAGE2_DENSE_INDEX="${CLI_VALUE}" ;;
                --stage2-sparse-index) STAGE2_SPARSE_INDEX="${CLI_VALUE}" ;;
            esac
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
DATASET="${DATASET:-flickr}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"

[[ "${GPUS}" =~ ^[1-9][0-9]*$ ]] || die "GPUS must be a positive integer, got: ${GPUS}"
[[ "${NUM_NODES}" =~ ^[1-9][0-9]*$ ]] || die "NUM_NODES must be a positive integer, got: ${NUM_NODES}"

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
CALCULATE_TYPE="concat"
EOL_TYPE="disassembleeol_separate"
MANUAL_LABEL="no_manual"
CLUSTER_LABEL="no_cluster"
USE_MEAN_LABEL="no_mean"

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
OUTPUT_DIR="${OUTPUT_DIR:-./three_stage_results/${ARTIFACT_MODEL_NAME}/${DATASET}/${DIRECTION_LABEL}/${DATASET_SPLIT}/${OUTPUT_VARIANT}}"
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
        "no corpus*.pkl found in ${STAGE2_DENSE_INDEX}; generate the candidate index with encode.py --encode_type ${TARGET_TYPE} --task_type ${TASK_TYPE}"
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
    --rerank-type pointwise
    --rerank-num "${RERANK_NUM}"
    --rerank-batch-size "${RERANK_BATCH_SIZE}"
    --rerank-template "${RERANK_TEMPLATE}"
    --modify_type "${MODIFY_TYPE}"
    "${PRIOR_CACHE_ARGS[@]}"
    --rerank-reps-loc "${RERANK_REPS_LOC}"
    --output-dir "${OUTPUT_DIR}"
    "${OVERWRITE_ARGS[@]}"
    "${EXTRA_ARGS[@]}"
)

echo "Dataset:            ${DATASET} (${DATASET_SPLIT}, ${DIRECTION_LABEL})"
echo "Task/query/corpus:  ${TASK_TYPE} / ${QUERY_TYPE} / ${TARGET_TYPE}"
if [[ "${TASK_TYPE}" == "tbpr" ]]; then
    echo "TBPR view type:     ${TBPR_TYPE}"
fi
echo "Distributed workers: ${GPUS} GPU process(es) across ${NUM_NODES} node(s)"
echo "Stage 1:            sparse_concat (query-sharded across workers, local encodings reused by Stage 2)"
echo "Stage-1 sparse index:${STAGE1_SPARSE_INDEX}"
echo "Retrieval model:    ${RETRIEVAL_MODEL_PATH}"
echo "Dense corpus index: ${STAGE2_DENSE_INDEX}"
echo "Sparse corpus index:${STAGE2_SPARSE_INDEX}"
echo "Rerank modify type: ${MODIFY_TYPE}"
echo "Output directory:   ${OUTPUT_DIR}"
printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

exec "${COMMAND[@]}"
