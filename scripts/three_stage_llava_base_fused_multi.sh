#!/usr/bin/env bash
USE_FREERET_DENSE="${USE_FREERET_DENSE-false}"  # true: FreeRet pre-MLP dense; prompts/sparse unchanged
USE_FREERET_FULL_PROMPT="${USE_FREERET_FULL_PROMPT-false}"  # Complete FreeRet dense/sparse prompts; requires FreeRet dense
USE_FREERET_PROMPT_CONSTRAINTS="${USE_FREERET_PROMPT_CONSTRAINTS-false}"  # Prepend FreeRet constraints to all representation views
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/freeret_dense_mode.sh" || exit $?
# Standalone LLaVA-NeXT base fused-modal: sparse -> hybrid -> rerank.
# Multiple models x multiple datasets, run sequentially; originals are unchanged.
# Examples:
#   bash scripts/three_stage_llava_base_fused_multi.sh
#   DATASETS="webqa remuq llava" bash scripts/three_stage_llava_base_fused_multi.sh
#   BASE_MODEL=llava-hf-llama3-llava-next-8b-hf bash scripts/three_stage_llava_base_fused_multi.sh
#   SELECT_HYBRID_ALPHA=true STAGE1_DEPTH=200 STAGE2_DEPTH=200 bash scripts/three_stage_llava_base_fused_multi.sh
set -euo pipefail

die() { printf 'Error: %s\n' "$*" >&2; exit 2; }
usage() {
    cat <<'USAGE'
Usage: bash scripts/three_stage_llava_base_fused_multi.sh [-h|--help]
Standalone launcher for src/three_stage_retrieval_pipeline.py; no other shell is called.
Supported/default datasets: webqa remuq llava.
WebQA: text->image+text (t2it). ReMuQ/LLaVA: image+text->text (it2t);
QUERY_TYPE=image retains both query modalities. Auto mode uses no/cache=false
for WebQA and modify_single for ReMuQ/LLaVA. Manual mode rejects enabled prior
correction/cache in any batch containing WebQA.
Default: two models x 3 datasets = 6 sequential jobs.
BASE_MODELS="llava-hf-llava-v1.6-mistral-7b-hf llava-hf-llama3-llava-next-8b-hf"
DATASETS="webqa remuq llava". BASE_MODEL/DATASET select one entry when nonempty.
Order: each model in list order, then every dataset in list order.
Lists are whitespace-separated; duplicate and empty lists are rejected.
Defaults: test, 4 GPUs, float16, prompt_5, sparse length 40, Stage-1/2 depth 1000,
disassembleeol_separate+concat (non-manual sparse), caption_generation top 5.
CHECKPOINT_ROOT defaults to /root/autodl-fs/checkpoints. Each model resolves to
CHECKPOINT_ROOT/<model name>. Edit MODEL_PATHS/RANKER_MODEL_PATHS/ARTIFACT_MODEL_NAMES
for per-model overrides. Every ranker defaults to the current retrieval model.
MODEL_PATH/RETRIEVAL_MODEL_PATH/RANKER_MODEL_PATH/ARTIFACT_MODEL_NAME are single-model
overrides only. Artifact directory names default to retrieval checkpoint basenames.
LLAVA_IMAGE_MODE=base|anyres defaults to base. RETRIEVAL_LLAVA_IMAGE_MODE and
RANKER_LLAVA_IMAGE_MODE independently override it. Base requires LLaVA-NeXT.
DENSE_OUTPUT_ROOT/SPARSE_OUTPUT_ROOT default to project dense_output/sparse_output.
Only image/image+text candidate leaves append _llava_base, according to retrieval
mode. Pure-text candidate paths are reused. Prompt, seed and sparse settings must
match encode.py. Build sparse indexes separately before running this launcher.
STAGE2_DENSE_INDEX/STAGE2_SPARSE_INDEX/STAGE1_SPARSE_INDEX override single-job inputs.
For batches, edit DENSE_INDEXES/SPARSE_INDEXES/STAGE1_INDEXES, keyed by "model|dataset".
Sparse aliases: leaf, collection/, index/, collection/index/; collection/index wins
over legacy index when both exist. Explicit paths are used without adding suffixes.
SELECT_HYBRID_ALPHA=false uses HYBRID_ALPHA=0.5 and ignores all validation inputs.
SELECT_HYBRID_ALPHA=true selects 0.1..0.9 on paired VAL_PASSAGE_REPS/VAL_SPARSE_INDEX,
or on the target split if neither is given. Use VAL_DENSE_INDEXES/VAL_SPARSE_INDEXES
and optional VALIDATION_DATASETS keyed by "model|dataset" for batch overrides.
VAL_PASSAGE_REPS/VAL_SPARSE_INDEX/VALIDATION_DATASET_NAME are single-job overrides.
The Python pipeline writes all alpha results and the best result to CSV.
LLAVA_OUTPUT_ROOT defaults to project three_stage_llava_base_results. Default paths
include model/dataset/direction/split, alpha, retrieval/ranker modes and end with
<modify_type>_prior_<true|false>/stage1_<STAGE1_DEPTH>_stage2_<STAGE2_DEPTH>.
OUTPUT_DIR is an exact single-job override and is not changed.
TBPR_TYPE=origin_type|type and CIR_TYPE=type are forwarded unchanged.
Prior correction/cache is supported only for IR image->text and ReMuQ/LLaVA.
It requires a caption_generation template; cache also requires MODIFY_TYPE=modify_single.
AUTO_MODIFY_BY_QUERY_TYPE=true (default) chooses per job: text -> no,
image/composed -> modify_single. Global MODIFY_TYPE is ignored in auto mode.
Auto no jobs disable prior cache; modify_single jobs use USE_RERANK_PRIOR_CACHE.
AUTO_MODIFY_BY_QUERY_TYPE=false restores global MODIFY_TYPE (default: no).
USE_RERANK_PRIOR_CACHE=false and OVERWRITE=false by default.
GPUS/NUM_NODES/DEEPSPEED_BIN, batch sizes, RETRIEVAL_DTYPE/RANKER_DTYPE (or DTYPE),
RETRIEVAL_REPS_LOC (or REPS_LOC), prompts and sparse settings are configurable.
All jobs are checked before the first GPU process. The first failure stops the batch.
Only -h/--help is accepted; arbitrary CLI arguments are rejected, not forwarded.
USAGE
}
if (($#)); then
    if (($# == 1)) && [[ "$1" == -h || "$1" == --help ]]; then usage; exit 0; fi
    die 'Configure environment variables or the configuration block; see --help.'
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# ---------------- Configuration: edit here or set environment variables. ----------------
BASE_MODEL="${BASE_MODEL:-}"
BASE_MODELS="${BASE_MODELS-llava-hf-llava-v1.6-mistral-7b-hf llava-hf-llama3-llava-next-8b-hf}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/root/autodl-fs/checkpoints}"
REQUESTED_MODEL_PATH="${RETRIEVAL_MODEL_PATH:-${MODEL_PATH:-}}"
REQUESTED_RANKER_PATH="${RANKER_MODEL_PATH:-}"
REQUESTED_ARTIFACT_NAME="${ARTIFACT_MODEL_NAME:-}"
LLAVA_IMAGE_MODE="${LLAVA_IMAGE_MODE:-base}"
RETRIEVAL_LLAVA_IMAGE_MODE="${RETRIEVAL_LLAVA_IMAGE_MODE:-${LLAVA_IMAGE_MODE}}"
RANKER_LLAVA_IMAGE_MODE="${RANKER_LLAVA_IMAGE_MODE:-${LLAVA_IMAGE_MODE}}"
DATASET="${DATASET:-}"
DATASETS="${DATASETS-webqa remuq llava}"
REQUESTED_QUERY_TYPE="${QUERY_TYPE:-}"
REQUESTED_TASK_TYPE="${TASK_TYPE:-}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
DENSE_OUTPUT_ROOT="${DENSE_OUTPUT_ROOT:-${REPO_ROOT}/dense_output}"
SPARSE_OUTPUT_ROOT="${SPARSE_OUTPUT_ROOT:-${REPO_ROOT}/sparse_output}"
LLAVA_OUTPUT_ROOT="${LLAVA_OUTPUT_ROOT:-${REPO_ROOT}/three_stage_llava_base_results}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
GPUS="${GPUS:-4}"
NUM_NODES="${NUM_NODES:-1}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"
STAGE1_DEPTH="${STAGE1_DEPTH:-1000}"
STAGE2_DEPTH="${STAGE2_DEPTH:-1000}"
SPARSE_THREADS="${SPARSE_THREADS:-${THREADS:-16}}"
RETRIEVAL_DEVICE="${RETRIEVAL_DEVICE:-auto}"
RETRIEVAL_DEVICE_MAP="${RETRIEVAL_DEVICE_MAP:-auto}"
RETRIEVAL_DTYPE="${RETRIEVAL_DTYPE:-${DTYPE:-float16}}"
RETRIEVAL_BATCH_SIZE="${RETRIEVAL_BATCH_SIZE:-1}"
RETRIEVAL_REPS_LOC="${RETRIEVAL_REPS_LOC:-${REPS_LOC:-after_pad}}"
PROMPT_TYPE="${PROMPT_TYPE:-prompt_5}"
RANDOM_VIEW_SEED="${RANDOM_VIEW_SEED:-42}"
SPARSE_LENGTH="${SPARSE_LENGTH:-40}"
TEXT_SPARSE_LENGTH="${TEXT_SPARSE_LENGTH:-${SPARSE_LENGTH}}"
IMAGE_SPARSE_LENGTH="${IMAGE_SPARSE_LENGTH:-${SPARSE_LENGTH}}"
NUM_EXPENDED_TOKENS="${NUM_EXPENDED_TOKENS:-0}"
SPARSE_VALUE_TYPE="${SPARSE_VALUE_TYPE:-sum}"
SPARSE_LOWER_OR_UPPER="${SPARSE_LOWER_OR_UPPER:-lower}"
SPARSE_VALUE_MEAN="${SPARSE_VALUE_MEAN:-false}"
IS_FILTERED="${IS_FILTERED:-false}"
TBPR_TYPE="${TBPR_TYPE:-origin_type}"
CIR_TYPE="${CIR_TYPE:-type}"
SELECT_HYBRID_ALPHA="${SELECT_HYBRID_ALPHA:-false}"
HYBRID_ALPHA="${HYBRID_ALPHA:-0.5}"
VALIDATION_SPLIT="${VALIDATION_SPLIT:-val}"
RANKER_DEVICE_MAP="${RANKER_DEVICE_MAP:-auto}"
RANKER_DTYPE="${RANKER_DTYPE:-${DTYPE:-float16}}"
RERANK_NUM="${RERANK_NUM:-5}"
RERANK_BATCH_SIZE="${RERANK_BATCH_SIZE:-1}"
RERANK_TEMPLATE="${RERANK_TEMPLATE:-caption_generation}"
RERANK_REPS_LOC="${RERANK_REPS_LOC:-after_pad}"
AUTO_MODIFY_BY_QUERY_TYPE="${AUTO_MODIFY_BY_QUERY_TYPE:-true}"
MODIFY_TYPE="${MODIFY_TYPE:-no}"
USE_RERANK_PRIOR_CACHE="${USE_RERANK_PRIOR_CACHE:-false}"
RERANK_PRIOR_CACHE_DIR="${RERANK_PRIOR_CACHE_DIR:-${REPO_ROOT}/cache/rerank_prior}"
OVERWRITE="${OVERWRITE:-false}"

# Optional model overrides, keyed by names from BASE_MODELS (edit declarations).
# Example: declare -A MODEL_PATHS=([my_llava]="/models/checkpoint with spaces")
# Example: declare -A RANKER_MODEL_PATHS=([my_llava]="/models/ranker")
# Example: declare -A ARTIFACT_MODEL_NAMES=([my_llava]="my_encoded_model")
declare -A MODEL_PATHS=()
declare -A RANKER_MODEL_PATHS=()
declare -A ARTIFACT_MODEL_NAMES=()
# Corpus and paired validation overrides are keyed by "model|dataset".
# Example: declare -A DENSE_INDEXES=(["my_llava|webqa"]="/data/dense/leaf")
# Example: declare -A SPARSE_INDEXES=(["my_llava|webqa"]="/data/sparse/leaf/collection")
# Use VALIDATION_DATASETS to override the validation dataset for that same job.
declare -A DENSE_INDEXES=()
declare -A SPARSE_INDEXES=()
declare -A STAGE1_INDEXES=()
declare -A VAL_DENSE_INDEXES=()
declare -A VAL_SPARSE_INDEXES=()
declare -A VALIDATION_DATASETS=()
# ---------------------------------------------------------------------------------------

bool() {
    case "${!1,,}" in
        true|1|yes) printf -v "$1" '%s' true ;;
        false|0|no) printf -v "$1" '%s' false ;;
        *) die "$1 must be true/false or 1/0." ;;
    esac
}
for name in SELECT_HYBRID_ALPHA SPARSE_VALUE_MEAN IS_FILTERED AUTO_MODIFY_BY_QUERY_TYPE USE_RERANK_PRIOR_CACHE OVERWRITE; do bool "${name}"; done
[[ "${EOL_TYPE:-disassembleeol_separate}" == disassembleeol_separate && "${CALCULATE_TYPE:-concat}" == concat ]] || die 'Stages 1/2 require disassembleeol_separate+concat.'
SPARSE_MANUAL="${SPARSE_MANUAL:-false}"
bool SPARSE_MANUAL
[[ "${SPARSE_MANUAL}" == false && "${SPARSE_TYPE:-single}" == single ]] || die 'Stages 1/2 require SPARSE_MANUAL=false and SPARSE_TYPE=single.'
for mode in "${LLAVA_IMAGE_MODE}" "${RETRIEVAL_LLAVA_IMAGE_MODE}" "${RANKER_LLAVA_IMAGE_MODE}"; do
    case "${mode}" in base|anyres) ;; *) die 'LLaVA image modes must be base or anyres.' ;; esac
done
for name in GPUS NUM_NODES STAGE1_DEPTH STAGE2_DEPTH SPARSE_THREADS RETRIEVAL_BATCH_SIZE RERANK_NUM RERANK_BATCH_SIZE SPARSE_LENGTH TEXT_SPARSE_LENGTH IMAGE_SPARSE_LENGTH; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer."
done
[[ "${NUM_EXPENDED_TOKENS}" =~ ^(0|[1-9][0-9]*)$ ]] || die 'NUM_EXPENDED_TOKENS must be nonnegative.'
[[ "${RANDOM_VIEW_SEED}" =~ ^-?[0-9]+$ ]] || die 'RANDOM_VIEW_SEED must be an integer.'
[[ "${HYBRID_ALPHA}" =~ ^(0|0?\.[0-9]+|1(\.0+)?)$ ]] || die 'HYBRID_ALPHA must be in [0,1].'
for name in RETRIEVAL_DTYPE RANKER_DTYPE; do
    case "${!name}" in float16|bfloat16|float32) ;; *) die "Invalid ${name}." ;; esac
done
for name in RETRIEVAL_REPS_LOC RERANK_REPS_LOC; do
    [[ "${!name}" == after_pad || "${!name}" == before_pad ]] || die "Invalid ${name}."
done
for name in DATASET_SPLIT PROMPT_TYPE RERANK_TEMPLATE CIR_TYPE; do
    [[ "${!name}" =~ ^[a-zA-Z0-9_-]+$ ]] || die "Invalid ${name}."
done
[[ "${TBPR_TYPE}" == origin_type || "${TBPR_TYPE}" == type ]] || die 'Invalid TBPR_TYPE.'
[[ "${SPARSE_VALUE_TYPE}" == sum || "${SPARSE_VALUE_TYPE}" == max || "${SPARSE_VALUE_TYPE}" == replace ]] || die 'Invalid SPARSE_VALUE_TYPE.'
[[ "${SPARSE_LOWER_OR_UPPER}" == lower || "${SPARSE_LOWER_OR_UPPER}" == upper ]] || die 'Invalid SPARSE_LOWER_OR_UPPER.'
if [[ "${AUTO_MODIFY_BY_QUERY_TYPE}" == false ]]; then
    [[ "${MODIFY_TYPE}" == no || "${MODIFY_TYPE}" == modify_single ]] || die 'Invalid MODIFY_TYPE.'
fi
if [[ "${SELECT_HYBRID_ALPHA}" == true ]]; then
    [[ "${VALIDATION_SPLIT}" =~ ^[a-zA-Z0-9_-]+$ ]] || die 'Invalid VALIDATION_SPLIT.'
fi
if [[ -n "${BASE_MODEL}" ]]; then MODEL_LIST=("${BASE_MODEL}"); else read -r -a MODEL_LIST <<< "${BASE_MODELS//$'\n'/ }"; fi
((${#MODEL_LIST[@]})) || die 'BASE_MODELS must contain at least one model.'
if [[ -n "${DATASET}" ]]; then DATASET_LIST=("${DATASET}"); else read -r -a DATASET_LIST <<< "${DATASETS//$'\n'/ }"; fi
((${#DATASET_LIST[@]})) || die 'DATASETS must contain at least one dataset.'
TOTAL_JOBS=$((${#MODEL_LIST[@]} * ${#DATASET_LIST[@]}))
if ((${#MODEL_LIST[@]} > 1)); then
    [[ -z "${REQUESTED_MODEL_PATH}${REQUESTED_RANKER_PATH}${REQUESTED_ARTIFACT_NAME}" ]] || die 'Single-model path/name overrides require one model; use MODEL_PATHS/RANKER_MODEL_PATHS/ARTIFACT_MODEL_NAMES for batches.'
fi
if ((TOTAL_JOBS > 1)); then
    [[ -z "${OUTPUT_DIR}" ]] || die 'OUTPUT_DIR is only for a single job; use LLAVA_OUTPUT_ROOT for batches.'
    [[ -z "${STAGE2_DENSE_INDEX:-}${STAGE2_SPARSE_INDEX:-}${STAGE1_SPARSE_INDEX:-}" ]] || die 'For batches, use index tables keyed by model|dataset instead of shared STAGE*_INDEX overrides.'
    if [[ "${SELECT_HYBRID_ALPHA}" == true ]]; then
        [[ -z "${VAL_PASSAGE_REPS:-}${VAL_SPARSE_INDEX:-}${VALIDATION_DATASET_NAME:-}" ]] || die 'For batches, use validation tables keyed by model|dataset instead of shared validation overrides.'
    fi
fi

resolve_model() {
    JOB_MODEL="$1"
    RETRIEVAL_MODEL_PATH="${REQUESTED_MODEL_PATH:-${MODEL_PATHS[${JOB_MODEL}]:-${CHECKPOINT_ROOT}/${JOB_MODEL}}}"
    while [[ "${RETRIEVAL_MODEL_PATH}" == */ && "${RETRIEVAL_MODEL_PATH}" != / ]]; do RETRIEVAL_MODEL_PATH="${RETRIEVAL_MODEL_PATH%/}"; done
    RANKER_MODEL_PATH="${REQUESTED_RANKER_PATH:-${RANKER_MODEL_PATHS[${JOB_MODEL}]:-${RETRIEVAL_MODEL_PATH}}}"
    while [[ "${RANKER_MODEL_PATH}" == */ && "${RANKER_MODEL_PATH}" != / ]]; do RANKER_MODEL_PATH="${RANKER_MODEL_PATH%/}"; done
    ARTIFACT_MODEL_NAME="${REQUESTED_ARTIFACT_NAME:-${ARTIFACT_MODEL_NAMES[${JOB_MODEL}]:-${RETRIEVAL_MODEL_PATH##*/}}}"
    [[ -n "${ARTIFACT_MODEL_NAME}" && "${ARTIFACT_MODEL_NAME}" != . && "${ARTIFACT_MODEL_NAME}" != .. && "${ARTIFACT_MODEL_NAME}" != */* ]] || die 'ARTIFACT_MODEL_NAME must name one model directory.'
}

task_for() {
    case "$1" in
        webqa) printf t2it ;;
        remuq|llava) printf it2t ;;
        *) die "Unsupported fused dataset: $1; choose webqa, remuq or llava." ;;
    esac
}
relative_path() {
    local split="$1" filter=no_filter mean=no_mean suffix='' path
    [[ "${IS_FILTERED}" == false ]] || filter=filter
    [[ "${SPARSE_VALUE_MEAN}" == false ]] || mean=mean
    [[ "${PROMPT_TYPE}" != prompt_random_5 ]] || mean+="_random_view_seed_${RANDOM_VIEW_SEED}"
    path="${ARTIFACT_MODEL_NAME}/${JOB_DATASET}/${TARGET_TYPE}/${filter}/concat/${PROMPT_TYPE}"
    if [[ "${TASK_TYPE}" == tbpr && "${PROMPT_TYPE}" == prompt_random_5 ]]; then path+="/${TBPR_TYPE}"; fi
    path+="/${split}"
    if [[ "${TASK_TYPE}" == tbpr && "${PROMPT_TYPE}" != prompt_random_5 ]]; then path+="/${TBPR_TYPE}"; fi
    if [[ "${RETRIEVAL_LLAVA_IMAGE_MODE}" == base && "${HAS_IMAGES}" == true ]]; then suffix=_llava_base; fi
    printf '%s/%s_no_manual_%s_%s_no_cluster_%s_disassembleeol_separate_%s_%s%s' "${path}" "${NUM_EXPENDED_TOKENS}" "${SPARSE_LENGTH}" "${SPARSE_VALUE_TYPE}" "${RETRIEVAL_REPS_LOC}" "${SPARSE_LOWER_OR_UPPER}" "${mean}" "${suffix}"
}
resolve_index() {
    local path="$1"
    while [[ "${path}" == */ && "${path}" != / ]]; do path="${path%/}"; done
    local -a direct=("${path}"/segments_*)
    if [[ "${path##*/}" == index ]] || ((${#direct[@]})); then printf '%s' "${path}"; return; fi
    [[ "${path##*/}" != collection ]] || path="$(dirname -- "${path}")"
    if [[ -e "${path}/collection/index" || -L "${path}/collection/index" ]]; then printf '%s/collection/index' "${path}"
    elif [[ -e "${path}/index" || -L "${path}/index" ]]; then printf '%s/index' "${path}"
    else printf '%s/collection/index' "${path}"
    fi
}
require_file() { [[ -f "$1" ]] || die "$2 file not found: $1"; }
require_dataset() {
    local dataset="$1" split="$2" directory file
    local -a files
    case "${dataset}" in
        flickr|coco) directory="${dataset}"; files=("${dataset}_${split}.csv") ;;
        CUHK-PEDES) directory="${dataset}"; files=(reid_raw.json) ;;
        ICFG-PEDES) directory="${dataset}"; files=(ICFG-PEDES.json) ;;
        RSTPReid) directory="${dataset}"; files=(data_captions.json) ;;
        webqa) directory=WebQA; files=(query-00000-of-00001.parquet qrels-00000-of-00001.parquet corpus-00000-of-00002.parquet corpus-00001-of-00002.parquet) ;;
        remuq|llava) directory=UMRB-ReMuQ; [[ "${dataset}" != llava ]] || directory=UMRB-LLaVA; files=(query.parquet qrels.parquet corpus.parquet) ;;
        *) die "Unsupported dataset: ${dataset}" ;;
    esac
    for file in "${files[@]}"; do require_file "${DATA_ROOT}/data/${directory}/${file}" "${dataset}/${split} annotation"; done
}
shopt -s nullglob
require_corpus() {
    local label="$1" dense="$2" index="$3"
    local -a shards=("${dense}"/corpus*.pkl) segments=("${index}"/segments_*)
    if [[ -f "${dense}" ]]; then
        [[ "${dense##*/}" == corpus*.pkl ]] || die "${label}: expected corpus*.pkl: ${dense}"
    else
        ((${#shards[@]})) && [[ -f "${shards[0]}" ]] || die "${label}: no corpus*.pkl in ${dense}; run encode.py with matching settings."
    fi
    ((${#segments[@]})) && [[ -f "${segments[0]}" ]] || die "${label}: no Lucene segments_* in ${index}; build the sparse index first."
}
resolve_modify_type() {
    # Always reset from the original global settings, never from the last job.
    JOB_MODIFY_TYPE="${MODIFY_TYPE}"
    JOB_USE_RERANK_PRIOR_CACHE="${USE_RERANK_PRIOR_CACHE}"
    if [[ "${AUTO_MODIFY_BY_QUERY_TYPE}" == true ]]; then
        case "${QUERY_TYPE}" in
            text) JOB_MODIFY_TYPE=no; JOB_USE_RERANK_PRIOR_CACHE=false ;;
            image|composed) JOB_MODIFY_TYPE=modify_single ;;
            *) die "Cannot select modify_type for QUERY_TYPE=${QUERY_TYPE}." ;;
        esac
    fi
}
resolve_job() {
    JOB_DATASET="$1"
    JOB_KEY="${JOB_MODEL}|${JOB_DATASET}"
    TASK_TYPE="$(task_for "${JOB_DATASET}")"
    [[ -z "${REQUESTED_TASK_TYPE}" || "${REQUESTED_TASK_TYPE}" == "${TASK_TYPE}" ]] || die "${JOB_DATASET} requires TASK_TYPE=${TASK_TYPE}."
    HAS_IMAGES=true
    case "${TASK_TYPE}" in
        ir)
            QUERY_TYPE="${REQUESTED_QUERY_TYPE:-image}"
            case "${QUERY_TYPE}" in
                image) TARGET_TYPE=text; DIRECTION=image_to_text; HAS_IMAGES=false ;;
                text) TARGET_TYPE=image; DIRECTION=text_to_image ;;
                *) die 'IR QUERY_TYPE must be image or text.' ;;
            esac ;;
        tbpr)
            QUERY_TYPE="${REQUESTED_QUERY_TYPE:-text}"
            [[ "${QUERY_TYPE}" == text ]] || die 'TBPR requires QUERY_TYPE=text.'
            TARGET_TYPE=image; DIRECTION=text_to_image ;;
        t2it)
            QUERY_TYPE="${REQUESTED_QUERY_TYPE:-text}"
            [[ "${QUERY_TYPE}" == text ]] || die 'WebQA requires QUERY_TYPE=text.'
            TARGET_TYPE=corpus; DIRECTION=text_to_imagetext ;;
        it2t)
            QUERY_TYPE="${REQUESTED_QUERY_TYPE:-image}"
            [[ "${QUERY_TYPE}" == image || "${QUERY_TYPE}" == composed ]] || die 'ReMuQ/LLaVA require QUERY_TYPE=image or composed.'
            TARGET_TYPE=corpus; DIRECTION=imagetext_to_text; HAS_IMAGES=false ;;
    esac
    resolve_modify_type
    RELATIVE="$(relative_path "${DATASET_SPLIT}")"
    JOB_DENSE="${DENSE_INDEXES[${JOB_KEY}]:-${STAGE2_DENSE_INDEX:-${DENSE_OUTPUT_ROOT}/${RELATIVE}}}"
    JOB_SPARSE="$(resolve_index "${SPARSE_INDEXES[${JOB_KEY}]:-${STAGE2_SPARSE_INDEX:-${SPARSE_OUTPUT_ROOT}/${RELATIVE}/collection}}")"
    JOB_FIRST="$(resolve_index "${STAGE1_INDEXES[${JOB_KEY}]:-${STAGE1_SPARSE_INDEX:-${JOB_SPARSE}}}")"
    ALPHA_ARGS=(); ALPHA_LABEL="alpha_${HYBRID_ALPHA}"
    JOB_VAL_DENSE=''; JOB_VAL_SPARSE=''; JOB_VAL_DATASET=''
    if [[ "${SELECT_HYBRID_ALPHA}" == true ]]; then
        ALPHA_ARGS+=(--select-hybrid-alpha); ALPHA_LABEL="alpha_${DATASET_SPLIT}"
        if [[ -n "${VAL_DENSE_INDEXES[${JOB_KEY}]:-}${VAL_SPARSE_INDEXES[${JOB_KEY}]:-}" ]]; then
            JOB_VAL_DENSE="${VAL_DENSE_INDEXES[${JOB_KEY}]:-}"; JOB_VAL_SPARSE="${VAL_SPARSE_INDEXES[${JOB_KEY}]:-}"
        else
            JOB_VAL_DENSE="${VAL_PASSAGE_REPS:-}"; JOB_VAL_SPARSE="${VAL_SPARSE_INDEX:-}"
        fi
        JOB_VAL_DATASET="${VALIDATION_DATASETS[${JOB_KEY}]:-${VALIDATION_DATASET_NAME:-}}"
        if [[ -n "${JOB_VAL_DENSE}${JOB_VAL_SPARSE}${JOB_VAL_DATASET}" ]]; then
            [[ -n "${JOB_VAL_DENSE}" && -n "${JOB_VAL_SPARSE}" ]] || die "Validation dense/sparse paths must both be provided for ${JOB_DATASET}."
            JOB_VAL_DATASET="${JOB_VAL_DATASET:-${JOB_DATASET}}"
            [[ "$(task_for "${JOB_VAL_DATASET}")" == "${TASK_TYPE}" ]] || die "Validation task mismatch for ${JOB_DATASET}: ${JOB_VAL_DATASET}."
            JOB_VAL_SPARSE="$(resolve_index "${JOB_VAL_SPARSE}")"
            ALPHA_ARGS+=(--val-passage-reps "${JOB_VAL_DENSE}" --val-sparse-index "${JOB_VAL_SPARSE}" --validation-dataset-name "${JOB_VAL_DATASET}" --validation-split "${VALIDATION_SPLIT}")
            ALPHA_LABEL=alpha_val
        fi
    fi
    local result="${LLAVA_OUTPUT_ROOT}/${ARTIFACT_MODEL_NAME}/${JOB_DATASET}/${DIRECTION}/${DATASET_SPLIT}/sparse_concat_stage1"
    [[ "${TASK_TYPE}" != tbpr ]] || result+="/${TBPR_TYPE}"
    JOB_OUTPUT="${OUTPUT_DIR:-${result}/${ALPHA_LABEL}_retrieval_${RETRIEVAL_LLAVA_IMAGE_MODE}_ranker_${RANKER_LLAVA_IMAGE_MODE}/${JOB_MODIFY_TYPE}_prior_${JOB_USE_RERANK_PRIOR_CACHE}/stage1_${STAGE1_DEPTH}_stage2_${STAGE2_DEPTH}}"
}
preflight_job() {
    dense_mode_paths
    require_dataset "${JOB_DATASET}" "${DATASET_SPLIT}"
    require_corpus "${JOB_DATASET} Stage 2" "${JOB_DENSE}" "${JOB_SPARSE}"
    local -a segments=("${JOB_FIRST}"/segments_*)
    ((${#segments[@]})) && [[ -f "${segments[0]}" ]] || die "${JOB_DATASET} Stage 1: no Lucene segments_* in ${JOB_FIRST}"
    if [[ -n "${JOB_VAL_DENSE}" ]]; then
        require_dataset "${JOB_VAL_DATASET}" "${VALIDATION_SPLIT}"
        require_corpus "${JOB_DATASET} Validation" "${JOB_VAL_DENSE}" "${JOB_VAL_SPARSE}"
    fi
    if [[ "${JOB_MODIFY_TYPE}" == modify_single || "${JOB_USE_RERANK_PRIOR_CACHE}" == true ]]; then
        [[ "${RERANK_TEMPLATE}" == *caption_generation* ]] || die 'Prior correction requires a caption_generation template.'
        [[ "${TASK_TYPE}" == it2t || ( "${TASK_TYPE}" == ir && "${QUERY_TYPE}" == image ) ]] || die "Prior correction is unsupported for ${JOB_DATASET}/${QUERY_TYPE}."
        [[ "${JOB_USE_RERANK_PRIOR_CACHE}" == false || "${JOB_MODIFY_TYPE}" == modify_single ]] || die 'Prior cache requires MODIFY_TYPE=modify_single.'
    fi
    [[ ! -e "${JOB_OUTPUT}" && ! -L "${JOB_OUTPUT}" || "${OVERWRITE}" == true ]] || die "Output already exists: ${JOB_OUTPUT}; set OVERWRITE=true or choose a new LLAVA_OUTPUT_ROOT."
}
build_command() {
    dense_mode_paths
    COMMAND=("${DEEPSPEED_BIN}" --num_gpus "${GPUS}" --num_nodes "${NUM_NODES}" src/three_stage_retrieval_pipeline.py --use_freeret_dense "${USE_FREERET_DENSE}" --use_freeret_prompt_constraints "${USE_FREERET_PROMPT_CONSTRAINTS}" --use_freeret_full_prompt "${USE_FREERET_FULL_PROMPT}"
        --dataset-name "${JOB_DATASET}" --dataset-split "${DATASET_SPLIT}" --task-type "${TASK_TYPE}" --query-type "${QUERY_TYPE}"
        --tbpr-type "${TBPR_TYPE}" --cir-type "${CIR_TYPE}" --data-root "${DATA_ROOT}" --stage1-method sparse_concat
        --stage1-sparse-index "${JOB_FIRST}" --stage1-depth "${STAGE1_DEPTH}"
        --stage2-dense-index "${JOB_DENSE}" --stage2-sparse-index "${JOB_SPARSE}" --stage2-depth "${STAGE2_DEPTH}"
        --hybrid-alpha "${HYBRID_ALPHA}" "${ALPHA_ARGS[@]}" --sparse-threads "${SPARSE_THREADS}"
        --retrieval-model-name-or-path "${RETRIEVAL_MODEL_PATH}" --retrieval-device "${RETRIEVAL_DEVICE}" --retrieval-device-map "${RETRIEVAL_DEVICE_MAP}"
        --retrieval-dtype "${RETRIEVAL_DTYPE}" --retrieval-batch-size "${RETRIEVAL_BATCH_SIZE}" --retrieval-reps-loc "${RETRIEVAL_REPS_LOC}"
        --retrieval-llava-image-mode "${RETRIEVAL_LLAVA_IMAGE_MODE}"
        --prompt-type "${PROMPT_TYPE}" --random-view-seed "${RANDOM_VIEW_SEED}" --sparse-length "${SPARSE_LENGTH}"
        --text-sparse-length "${TEXT_SPARSE_LENGTH}" --image-sparse-length "${IMAGE_SPARSE_LENGTH}" --num-expended-tokens "${NUM_EXPENDED_TOKENS}"
        --sparse-value-type "${SPARSE_VALUE_TYPE}" --sparse-lower-or-upper "${SPARSE_LOWER_OR_UPPER}"
        --ranker-model-name-or-path "${RANKER_MODEL_PATH}" --ranker-device-map "${RANKER_DEVICE_MAP}" --ranker-dtype "${RANKER_DTYPE}"
        --ranker-llava-image-mode "${RANKER_LLAVA_IMAGE_MODE}"
        --rerank-type pointwise --rerank-template "${RERANK_TEMPLATE}" --rerank-num "${RERANK_NUM}" --rerank-batch-size "${RERANK_BATCH_SIZE}"
        --rerank-reps-loc "${RERANK_REPS_LOC}" --modify-type "${JOB_MODIFY_TYPE}" --output-dir "${JOB_OUTPUT}")
    [[ "${IS_FILTERED}" == false ]] || COMMAND+=(--is-filtered)
    [[ "${SPARSE_VALUE_MEAN}" == false ]] || COMMAND+=(--sparse-value-mean)
    [[ "${OVERWRITE}" == false ]] || COMMAND+=(--overwrite)
    [[ "${JOB_USE_RERANK_PRIOR_CACHE}" == false ]] || COMMAND+=(--use-rerank-prior-cache --rerank-prior-cache-dir "${RERANK_PRIOR_CACHE_DIR}")
}

command -v "${DEEPSPEED_BIN}" >/dev/null 2>&1 || die "DeepSpeed executable not found: ${DEEPSPEED_BIN}"
require_file src/three_stage_retrieval_pipeline.py 'Python entrypoint'
declare -A SEEN_MODELS=() SEEN_DATASETS=() SEEN_ARTIFACTS=() SEEN_OUTPUTS=()
for model in "${MODEL_LIST[@]}"; do
    [[ "${model}" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || die "Invalid model name: ${model}; use MODEL_PATHS for custom checkpoint paths."
    [[ -z "${SEEN_MODELS[${model}]+present}" ]] || die "Duplicate model: ${model}"
    SEEN_MODELS["${model}"]=1
done
for dataset in "${DATASET_LIST[@]}"; do
    task_for "${dataset}" >/dev/null
    [[ -z "${SEEN_DATASETS[${dataset}]+present}" ]] || die "Duplicate dataset: ${dataset}"
    SEEN_DATASETS["${dataset}"]=1
done
# Preflight the entire model x dataset matrix before the first GPU process.
for model in "${MODEL_LIST[@]}"; do
    resolve_model "${model}"
    [[ -z "${SEEN_ARTIFACTS[${ARTIFACT_MODEL_NAME}]+present}" ]] || die "Duplicate artifact model directory: ${ARTIFACT_MODEL_NAME}; set distinct ARTIFACT_MODEL_NAMES."
    SEEN_ARTIFACTS["${ARTIFACT_MODEL_NAME}"]="${model}"
    require_file "${RETRIEVAL_MODEL_PATH}/config.json" "${model} retrieval checkpoint config"
    require_file "${RANKER_MODEL_PATH}/config.json" "${model} ranker checkpoint config"
    for dataset in "${DATASET_LIST[@]}"; do
        resolve_job "${dataset}"
        [[ -z "${SEEN_OUTPUTS[${JOB_OUTPUT}]+present}" ]] || die "Output collision for ${JOB_KEY}: ${JOB_OUTPUT}"
        SEEN_OUTPUTS["${JOB_OUTPUT}"]="${JOB_KEY}"
        preflight_job
    done
done
printf 'LLaVA fused-modal batch: models=%s datasets=%s jobs=%s GPUs=%s\n' "${MODEL_LIST[*]}" "${DATASET_LIST[*]}" "${TOTAL_JOBS}" "${GPUS}"
completed=0
for model in "${MODEL_LIST[@]}"; do
    resolve_model "${model}"
    for dataset in "${DATASET_LIST[@]}"; do
        resolve_job "${dataset}"
        build_command
        printf '\nJob [%s/%s]: %s (%s, %s)\nRetrieval model: %s (%s)\nRanker model: %s (%s)\nDense: %s\nStage-1 sparse: %s\nStage-2 sparse: %s\n' "$((completed + 1))" "${TOTAL_JOBS}" "${JOB_KEY}" "${DATASET_SPLIT}" "${DIRECTION}" "${RETRIEVAL_MODEL_PATH}" "${RETRIEVAL_LLAVA_IMAGE_MODE}" "${RANKER_MODEL_PATH}" "${RANKER_LLAVA_IMAGE_MODE}" "${JOB_DENSE}" "${JOB_FIRST}" "${JOB_SPARSE}"
        printf 'Query type: %s; auto modify: %s; modify_type: %s; prior cache: %s\n' "${QUERY_TYPE}" "${AUTO_MODIFY_BY_QUERY_TYPE}" "${JOB_MODIFY_TYPE}" "${JOB_USE_RERANK_PRIOR_CACHE}"
        if [[ "${SELECT_HYBRID_ALPHA}" == false ]]; then printf 'Alpha: fixed %s\n' "${HYBRID_ALPHA}"
        elif [[ -n "${JOB_VAL_DENSE}" ]]; then printf 'Alpha: select 0.1..0.9 on %s/%s\nValidation dense: %s\nValidation sparse: %s\n' "${JOB_VAL_DATASET}" "${VALIDATION_SPLIT}" "${JOB_VAL_DENSE}" "${JOB_VAL_SPARSE}"
        else printf 'Skipping validation: no val paths specified.\nAlpha: select 0.1..0.9 on target %s/%s\n' "${dataset}" "${DATASET_SPLIT}"
        fi
        printf 'Results: %s\nCommand:' "${JOB_OUTPUT}"; printf ' %q' "${COMMAND[@]}"; printf '\n'
        if "${COMMAND[@]}"; then
            completed=$((completed + 1))
            printf 'Completed [%s/%s]: %s\n' "${completed}" "${TOTAL_JOBS}" "${JOB_KEY}"
        else
            status=$?
            printf 'Failed [%s/%s]: %s (exit %s); completed %s jobs, stopping batch.\n' "$((completed + 1))" "${TOTAL_JOBS}" "${JOB_KEY}" "${status}" "${completed}" >&2
            exit "${status}"
        fi
    done
done
printf '\nCompleted all %s model/dataset jobs.\n' "${completed}"
