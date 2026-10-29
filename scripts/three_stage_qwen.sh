#!/usr/bin/env bash
# Standalone Qwen launcher: sparse_concat -> hybrid retrieval -> pointwise rerank.
# Prepare matching encode_qwen.sh artifacts and sparse_index_qwen.sh indexes first.
# Examples (run from any directory):
#   bash scripts/three_stage_qwen.sh
#   DATASETS="flickr coco" QWEN_FAMILY=qwen3 bash scripts/three_stage_qwen.sh
#   DATASET=flickr IR_QUERY_TYPE=text bash scripts/three_stage_qwen.sh
#   SELECT_HYBRID_ALPHA=true bash scripts/three_stage_qwen.sh
set -euo pipefail

die() { printf 'Error: %s\n' "$*" >&2; exit 2; }
usage() {
    cat <<'USAGE'
Usage: bash scripts/three_stage_qwen.sh [-h|--help]
Edit this script's configuration block or set environment variables.
This standalone script directly launches src/three_stage_retrieval_pipeline.py.
Defaults: Qwen2.5-VL-7B-Instruct, Flickr image->text, test, 1 GPU, float16,
sparse_concat in Stages 1/2, fixed HYBRID_ALPHA=0.5, what_caption_generation top 5.
  QWEN_FAMILY=qwen2|qwen25|qwen3; QWEN_MODEL_PATH overrides the checkpoint.
  RANKER_MODEL_PATH defaults to QWEN_MODEL_PATH.
  DATASET selects one job; otherwise DATASETS is a whitespace-separated list (default: flickr).
  IR_QUERY_TYPE=image|text controls COCO/Flickr direction.
  DATA_ROOT defaults to /root/autodl-tmp (annotations are under DATA_ROOT/data).
  DENSE_OUTPUT_ROOT/SPARSE_OUTPUT_ROOT default to project dense_output/sparse_output.
  STAGE2_DENSE_INDEX/STAGE2_SPARSE_INDEX/STAGE1_SPARSE_INDEX override single-job inputs.
  For batches, use the per-dataset input tables in the configuration block.
  Sparse aliases: encoding leaf, collection/, index/, collection/index/.
  Prefer collection/index; use legacy index only when the nested path is absent.
  SELECT_HYBRID_ALPHA=false uses HYBRID_ALPHA (default: 0.5), ignoring all val inputs.
  SELECT_HYBRID_ALPHA=true scans 0.1..0.9 on explicit paired val inputs, otherwise target.
  VAL_PASSAGE_REPS/VAL_SPARSE_INDEX must be paired. VALIDATION_SPLIT defaults to val.
  Shared val paths in a batch require VALIDATION_DATASET_NAME and matching task types.
  QWEN_OUTPUT_ROOT defaults to project three_stage_qwen_results.
  OUTPUT_DIR overrides the exact destination for a single job only; OVERWRITE defaults to false.
  GPUS/NUM_NODES/DEEPSPEED_BIN default to 1/1/deepspeed.
  RETRIEVAL_DTYPE/RANKER_DTYPE accept float16|bfloat16|float32 (DTYPE supplies both defaults).
  RETRIEVAL_QWEN_MIN_PIXELS/MAX_PIXELS and RANKER_QWEN_MIN_PIXELS/MAX_PIXELS default
  to QWEN_MIN_PIXELS/MAX_PIXELS, or 0 for checkpoint defaults.
  Keep prompt, random seed, pixel and sparse settings identical to encoding.
Supported datasets: flickr coco CUHK-PEDES ICFG-PEDES RSTPReid webqa remuq llava.
Stages 1/2 require disassembleeol_separate+concat with non-manual sparse vectors.
Only -h/--help is accepted; arbitrary CLI arguments are not forwarded.
USAGE
}
if (($#)); then
    if (($# == 1)) && [[ "$1" == -h || "$1" == --help ]]; then usage; exit 0; fi
    die 'Configure this script or environment variables. See --help.'
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# ---------------- Configuration: edit here or use environment variables. ----------------
QWEN_FAMILY="${QWEN_FAMILY:-qwen25}"
QWEN_CHECKPOINT_ROOT="${QWEN_CHECKPOINT_ROOT:-/root/autodl-fs/checkpoints}"
DATASET="${DATASET:-}"
DATASETS="${DATASETS-flickr}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
IR_QUERY_TYPE="${IR_QUERY_TYPE:-image}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
DENSE_OUTPUT_ROOT="${DENSE_OUTPUT_ROOT:-${REPO_ROOT}/dense_output}"
SPARSE_OUTPUT_ROOT="${SPARSE_OUTPUT_ROOT:-${REPO_ROOT}/sparse_output}"
QWEN_OUTPUT_ROOT="${QWEN_OUTPUT_ROOT:-${REPO_ROOT}/three_stage_qwen_results}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
GPUS="${GPUS:-1}"
NUM_NODES="${NUM_NODES:-1}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"
STAGE1_DEPTH="${STAGE1_DEPTH:-1000}"
STAGE2_DEPTH="${STAGE2_DEPTH:-1000}"
SPARSE_THREADS="${SPARSE_THREADS:-${THREADS:-16}}"
RETRIEVAL_BATCH_SIZE="${RETRIEVAL_BATCH_SIZE:-1}"
RETRIEVAL_DEVICE="${RETRIEVAL_DEVICE:-auto}"
RETRIEVAL_DEVICE_MAP="${RETRIEVAL_DEVICE_MAP:-auto}"
RETRIEVAL_DTYPE="${RETRIEVAL_DTYPE:-${DTYPE:-float16}}"
RETRIEVAL_REPS_LOC="${RETRIEVAL_REPS_LOC:-${REPS_LOC:-after_pad}}"
RETRIEVAL_QWEN_MIN_PIXELS="${RETRIEVAL_QWEN_MIN_PIXELS:-${QWEN_MIN_PIXELS:-0}}"
RETRIEVAL_QWEN_MAX_PIXELS="${RETRIEVAL_QWEN_MAX_PIXELS:-${QWEN_MAX_PIXELS:-0}}"
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
SELECT_HYBRID_ALPHA="${SELECT_HYBRID_ALPHA:-false}"
HYBRID_ALPHA="${HYBRID_ALPHA:-0.5}"
VALIDATION_SPLIT="${VALIDATION_SPLIT:-val}"
RANKER_DEVICE_MAP="${RANKER_DEVICE_MAP:-auto}"
RANKER_DTYPE="${RANKER_DTYPE:-${DTYPE:-float16}}"
RANKER_QWEN_MIN_PIXELS="${RANKER_QWEN_MIN_PIXELS:-${QWEN_MIN_PIXELS:-0}}"
RANKER_QWEN_MAX_PIXELS="${RANKER_QWEN_MAX_PIXELS:-${QWEN_MAX_PIXELS:-0}}"
RERANK_NUM="${RERANK_NUM:-5}"
RERANK_BATCH_SIZE="${RERANK_BATCH_SIZE:-1}"
RERANK_TEMPLATE="${RERANK_TEMPLATE:-what_caption_generation}"
RERANK_REPS_LOC="${RERANK_REPS_LOC:-after_pad}"
MODIFY_TYPE="${MODIFY_TYPE:-no}"
USE_RERANK_PRIOR_CACHE="${USE_RERANK_PRIOR_CACHE:-false}"
RERANK_PRIOR_CACHE_DIR="${RERANK_PRIOR_CACHE_DIR:-${REPO_ROOT}/cache/rerank_prior}"
OVERWRITE="${OVERWRITE:-false}"

# Optional per-TARGET-dataset overrides. Empty entries use derived paths.
# DENSE_INDEXES=([flickr]="/my/dense/leaf" [coco]="/my/coco/dense/leaf")
# SPARSE_INDEXES=([flickr]="/my/sparse/leaf/collection")
declare -A DENSE_INDEXES=()
declare -A SPARSE_INDEXES=()
declare -A STAGE1_INDEXES=()
# Paired validation inputs; omit both to select on the target split.
# VAL_DENSE_INDEXES=([coco]="/my/flickr/val/dense")
# VAL_SPARSE_INDEXES=([coco]="/my/flickr/val/sparse/collection")
# VALIDATION_DATASETS=([coco]="flickr")
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
for name in SELECT_HYBRID_ALPHA SPARSE_VALUE_MEAN IS_FILTERED USE_RERANK_PRIOR_CACHE OVERWRITE; do bool "${name}"; done
[[ "${EOL_TYPE:-disassembleeol_separate}" == disassembleeol_separate && "${CALCULATE_TYPE:-concat}" == concat ]] || die 'Stages 1/2 require disassembleeol_separate+concat.'
SPARSE_MANUAL="${SPARSE_MANUAL:-false}"
bool SPARSE_MANUAL
[[ "${SPARSE_MANUAL}" == false ]] || die 'Stages 1/2 require SPARSE_MANUAL=false.'
for name in GPUS NUM_NODES STAGE1_DEPTH STAGE2_DEPTH SPARSE_THREADS RETRIEVAL_BATCH_SIZE RERANK_NUM RERANK_BATCH_SIZE SPARSE_LENGTH TEXT_SPARSE_LENGTH IMAGE_SPARSE_LENGTH; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer."
done
for name in NUM_EXPENDED_TOKENS RETRIEVAL_QWEN_MIN_PIXELS RETRIEVAL_QWEN_MAX_PIXELS RANKER_QWEN_MIN_PIXELS RANKER_QWEN_MAX_PIXELS; do
    [[ "${!name}" =~ ^(0|[1-9][0-9]*)$ ]] || die "${name} must be a nonnegative integer."
done
for prefix in RETRIEVAL RANKER; do
    minimum="${prefix}_QWEN_MIN_PIXELS"; maximum="${prefix}_QWEN_MAX_PIXELS"
    if ((${!minimum} > 0 && ${!maximum} > 0 && ${!minimum} > ${!maximum})); then die "${minimum} must not exceed ${maximum}."; fi
    name="${prefix}_DTYPE"
    case "${!name}" in float16|bfloat16|float32) ;; *) die "Invalid ${name}." ;; esac
done
for name in RETRIEVAL_REPS_LOC RERANK_REPS_LOC; do
    [[ "${!name}" == after_pad || "${!name}" == before_pad ]] || die "Invalid ${name}."
done
[[ "${RANDOM_VIEW_SEED}" =~ ^-?[0-9]+$ ]] || die 'RANDOM_VIEW_SEED must be an integer.'
[[ "${HYBRID_ALPHA}" =~ ^(0|0?\.[0-9]+|1(\.0+)?)$ ]] || die 'HYBRID_ALPHA must be in [0,1].'
[[ "${IR_QUERY_TYPE}" == image || "${IR_QUERY_TYPE}" == text ]] || die 'IR_QUERY_TYPE must be image or text.'
[[ "${TBPR_TYPE}" == origin_type || "${TBPR_TYPE}" == type ]] || die 'TBPR_TYPE must be origin_type or type.'
[[ "${SPARSE_VALUE_TYPE}" == sum || "${SPARSE_VALUE_TYPE}" == max || "${SPARSE_VALUE_TYPE}" == replace ]] || die 'Invalid SPARSE_VALUE_TYPE.'
[[ "${SPARSE_LOWER_OR_UPPER}" == lower || "${SPARSE_LOWER_OR_UPPER}" == upper ]] || die 'Invalid SPARSE_LOWER_OR_UPPER.'
[[ "${MODIFY_TYPE}" == no || "${MODIFY_TYPE}" == modify_single ]] || die 'Invalid MODIFY_TYPE.'
for name in DATASET_SPLIT PROMPT_TYPE RERANK_TEMPLATE; do
    [[ "${!name}" =~ ^[a-zA-Z0-9_-]+$ ]] || die "Invalid ${name}."
done
if [[ "${SELECT_HYBRID_ALPHA}" == true ]]; then
    [[ "${VALIDATION_SPLIT}" =~ ^[a-zA-Z0-9_-]+$ ]] || die 'Invalid VALIDATION_SPLIT.'
fi
case "${QWEN_FAMILY}" in
    qwen2) CHECKPOINT=Qwen2-VL-7B-Instruct ;;
    qwen25) CHECKPOINT=Qwen2.5-VL-7B-Instruct ;;
    qwen3) CHECKPOINT=Qwen3-VL-8B-Instruct ;;
    *) die 'QWEN_FAMILY must be qwen2, qwen25 or qwen3.' ;;
esac
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${QWEN_CHECKPOINT_ROOT}/${CHECKPOINT}}"
while [[ "${QWEN_MODEL_PATH}" == */ && "${QWEN_MODEL_PATH}" != / ]]; do QWEN_MODEL_PATH="${QWEN_MODEL_PATH%/}"; done
MODEL_TAG="${QWEN_MODEL_PATH##*/}"
[[ -n "${MODEL_TAG}" && "${MODEL_TAG}" != . && "${MODEL_TAG}" != .. ]] || die 'Invalid QWEN_MODEL_PATH.'
RANKER_MODEL_PATH="${RANKER_MODEL_PATH:-${QWEN_MODEL_PATH}}"
if [[ -n "${DATASET}" ]]; then DATASET_LIST=("${DATASET}"); else read -r -a DATASET_LIST <<< "${DATASETS//$'\n'/ }"; fi
((${#DATASET_LIST[@]})) || die 'DATASETS must contain at least one dataset.'
if ((${#DATASET_LIST[@]} > 1)); then
    [[ -z "${OUTPUT_DIR}" ]] || die 'OUTPUT_DIR is only for a single job; use QWEN_OUTPUT_ROOT for batches.'
    [[ -z "${STAGE2_DENSE_INDEX:-}${STAGE2_SPARSE_INDEX:-}${STAGE1_SPARSE_INDEX:-}" ]] || die 'For batches, use per-dataset input tables instead of shared STAGE*_INDEX overrides.'
fi

task_for() {
    case "$1" in
        flickr|coco) printf ir ;;
        CUHK-PEDES|ICFG-PEDES|RSTPReid) printf tbpr ;;
        webqa) printf t2it ;;
        remuq|llava) printf it2t ;;
        *) die "Unsupported dataset: $1" ;;
    esac
}
relative_path() {
    local split="$1" filter=no_filter mean=no_mean
    [[ "${IS_FILTERED}" == false ]] || filter=filter
    [[ "${SPARSE_VALUE_MEAN}" == false ]] || mean=mean
    local path="${MODEL_TAG}/${JOB_DATASET}/${TARGET_TYPE}/${filter}/concat/${PROMPT_TYPE}/${split}"
    [[ "${TASK_TYPE}" != tbpr ]] || path+="/${TBPR_TYPE}"
    printf '%s/%s_no_manual_%s_%s_no_cluster_%s_disassembleeol_separate_%s_%s' "${path}" "${NUM_EXPENDED_TOKENS}" "${SPARSE_LENGTH}" "${SPARSE_VALUE_TYPE}" "${RETRIEVAL_REPS_LOC}" "${SPARSE_LOWER_OR_UPPER}" "${mean}"
}
resolve_index() {
    local path="$1"
    while [[ "${path}" == */ && "${path}" != / ]]; do path="${path%/}"; done
    if [[ "${path##*/}" == index ]]; then printf '%s' "${path}"; return; fi
    [[ "${path##*/}" != collection ]] || path="$(dirname -- "${path}")"
    if [[ -e "${path}/collection/index" || -L "${path}/collection/index" ]]; then
        printf '%s/collection/index' "${path}"
    elif [[ -e "${path}/index" || -L "${path}/index" ]]; then
        printf '%s/index' "${path}"
    else
        printf '%s/collection/index' "${path}"
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
    local label="$1" dense="$2" sparse="$3" index
    local -a shards=("${dense}"/corpus*.pkl)
    if [[ -f "${dense}" ]]; then
        [[ "${dense##*/}" == corpus*.pkl ]] || die "${label}: expected a corpus*.pkl file: ${dense}"
    else
        ((${#shards[@]})) && [[ -f "${shards[0]}" ]] || die "${label}: no corpus*.pkl in ${dense}; run encode_qwen.sh with matching settings."
    fi
    index="$(resolve_index "${sparse}")"
    local -a segments=("${index}"/segments_*)
    ((${#segments[@]})) && [[ -f "${segments[0]}" ]] || die "${label}: no Lucene segments_* in ${index}; run sparse_index_qwen.sh."
}
resolve_job() {
    JOB_DATASET="$1"
    TASK_TYPE="$(task_for "${JOB_DATASET}")"
    case "${TASK_TYPE}" in
        ir)
            QUERY_TYPE="${IR_QUERY_TYPE}"
            if [[ "${QUERY_TYPE}" == image ]]; then TARGET_TYPE=text; DIRECTION=image_to_text; else TARGET_TYPE=image; DIRECTION=text_to_image; fi ;;
        tbpr) QUERY_TYPE=text; TARGET_TYPE=image; DIRECTION=text_to_image ;;
        t2it) QUERY_TYPE=text; TARGET_TYPE=corpus; DIRECTION=text_to_imagetext ;;
        it2t) QUERY_TYPE=image; TARGET_TYPE=corpus; DIRECTION=imagetext_to_text ;;
    esac
    RELATIVE="$(relative_path "${DATASET_SPLIT}")"
    JOB_DENSE="${DENSE_INDEXES[${JOB_DATASET}]:-${STAGE2_DENSE_INDEX:-${DENSE_OUTPUT_ROOT}/${RELATIVE}}}"
    JOB_SPARSE="${SPARSE_INDEXES[${JOB_DATASET}]:-${STAGE2_SPARSE_INDEX:-${SPARSE_OUTPUT_ROOT}/${RELATIVE}/collection}}"
    JOB_FIRST="${STAGE1_INDEXES[${JOB_DATASET}]:-${STAGE1_SPARSE_INDEX:-${JOB_SPARSE}}}"
    ALPHA_ARGS=()
    ALPHA_LABEL="alpha_${HYBRID_ALPHA}"
    JOB_VAL_DENSE=''; JOB_VAL_SPARSE=''; JOB_VAL_DATASET=''
    if [[ "${SELECT_HYBRID_ALPHA}" == true ]]; then
        ALPHA_ARGS+=(--select-hybrid-alpha)
        ALPHA_LABEL="alpha_${DATASET_SPLIT}"
        if [[ -n "${VAL_DENSE_INDEXES[${JOB_DATASET}]:-}${VAL_SPARSE_INDEXES[${JOB_DATASET}]:-}" ]]; then
            JOB_VAL_DENSE="${VAL_DENSE_INDEXES[${JOB_DATASET}]:-}"
            JOB_VAL_SPARSE="${VAL_SPARSE_INDEXES[${JOB_DATASET}]:-}"
        else
            JOB_VAL_DENSE="${VAL_PASSAGE_REPS:-}"
            JOB_VAL_SPARSE="${VAL_SPARSE_INDEX:-}"
            if [[ -n "${JOB_VAL_DENSE}${JOB_VAL_SPARSE}" && ${#DATASET_LIST[@]} -gt 1 ]]; then
                [[ -n "${VALIDATION_DATASET_NAME:-}" ]] || die 'Shared val paths in a batch require VALIDATION_DATASET_NAME.'
            fi
        fi
        JOB_VAL_DATASET="${VALIDATION_DATASETS[${JOB_DATASET}]:-${VALIDATION_DATASET_NAME:-}}"
        if [[ -n "${JOB_VAL_DENSE}${JOB_VAL_SPARSE}${JOB_VAL_DATASET}" ]]; then
            [[ -n "${JOB_VAL_DENSE}" && -n "${JOB_VAL_SPARSE}" ]] || die "Validation dense/sparse paths must both be provided for ${JOB_DATASET}."
            JOB_VAL_DATASET="${JOB_VAL_DATASET:-${JOB_DATASET}}"
            [[ "$(task_for "${JOB_VAL_DATASET}")" == "${TASK_TYPE}" ]] || die "Validation task mismatch for ${JOB_DATASET}: ${JOB_VAL_DATASET}."
            ALPHA_ARGS+=(--val-passage-reps "${JOB_VAL_DENSE}" --val-sparse-index "${JOB_VAL_SPARSE}" --validation-dataset-name "${JOB_VAL_DATASET}" --validation-split "${VALIDATION_SPLIT}")
            ALPHA_LABEL=alpha_val
        fi
    fi
    JOB_OUTPUT="${OUTPUT_DIR:-${QWEN_OUTPUT_ROOT}/${RELATIVE}/${DIRECTION}/three_stage/${ALPHA_LABEL}}"
}
preflight_job() {
    require_dataset "${JOB_DATASET}" "${DATASET_SPLIT}"
    require_corpus "${JOB_DATASET} Stage 2" "${JOB_DENSE}" "${JOB_SPARSE}"
    local first="$(resolve_index "${JOB_FIRST}")"
    local -a segments=("${first}"/segments_*)
    ((${#segments[@]})) && [[ -f "${segments[0]}" ]] || die "${JOB_DATASET} Stage 1: no Lucene segments_* in ${first}; run sparse_index_qwen.sh."
    if [[ -n "${JOB_VAL_DENSE}" ]]; then
        require_dataset "${JOB_VAL_DATASET}" "${VALIDATION_SPLIT}"
        require_corpus "${JOB_DATASET} Validation" "${JOB_VAL_DENSE}" "${JOB_VAL_SPARSE}"
    fi
    if [[ "${MODIFY_TYPE}" == modify_single || "${USE_RERANK_PRIOR_CACHE}" == true ]]; then
        [[ "${RERANK_TEMPLATE}" == *caption_generation* ]] || die 'Prior correction requires a caption_generation template.'
        [[ "${TASK_TYPE}" == it2t || ( "${TASK_TYPE}" == ir && "${QUERY_TYPE}" == image ) ]] || die "Prior correction is unsupported for ${JOB_DATASET}/${QUERY_TYPE}."
        [[ "${USE_RERANK_PRIOR_CACHE}" == false || "${MODIFY_TYPE}" == modify_single ]] || die 'Prior cache requires MODIFY_TYPE=modify_single.'
    fi
}
build_command() {
    COMMAND=("${DEEPSPEED_BIN}" --num_gpus "${GPUS}" --num_nodes "${NUM_NODES}" src/three_stage_retrieval_pipeline.py
        --dataset-name "${JOB_DATASET}" --dataset-split "${DATASET_SPLIT}" --task-type "${TASK_TYPE}" --query-type "${QUERY_TYPE}"
        --tbpr-type "${TBPR_TYPE}" --data-root "${DATA_ROOT}" --stage1-method sparse_concat
        --stage1-sparse-index "${JOB_FIRST}" --stage1-depth "${STAGE1_DEPTH}"
        --stage2-dense-index "${JOB_DENSE}" --stage2-sparse-index "${JOB_SPARSE}" --stage2-depth "${STAGE2_DEPTH}"
        --hybrid-alpha "${HYBRID_ALPHA}" "${ALPHA_ARGS[@]}" --sparse-threads "${SPARSE_THREADS}"
        --retrieval-model-name-or-path "${QWEN_MODEL_PATH}" --retrieval-device "${RETRIEVAL_DEVICE}" --retrieval-device-map "${RETRIEVAL_DEVICE_MAP}"
        --retrieval-dtype "${RETRIEVAL_DTYPE}" --retrieval-batch-size "${RETRIEVAL_BATCH_SIZE}" --retrieval-reps-loc "${RETRIEVAL_REPS_LOC}"
        --retrieval-qwen-min-pixels "${RETRIEVAL_QWEN_MIN_PIXELS}" --retrieval-qwen-max-pixels "${RETRIEVAL_QWEN_MAX_PIXELS}"
        --prompt-type "${PROMPT_TYPE}" --random-view-seed "${RANDOM_VIEW_SEED}" --sparse-length "${SPARSE_LENGTH}"
        --text-sparse-length "${TEXT_SPARSE_LENGTH}" --image-sparse-length "${IMAGE_SPARSE_LENGTH}" --num-expended-tokens "${NUM_EXPENDED_TOKENS}"
        --sparse-value-type "${SPARSE_VALUE_TYPE}" --sparse-lower-or-upper "${SPARSE_LOWER_OR_UPPER}"
        --ranker-model-name-or-path "${RANKER_MODEL_PATH}" --ranker-device-map "${RANKER_DEVICE_MAP}" --ranker-dtype "${RANKER_DTYPE}"
        --ranker-qwen-min-pixels "${RANKER_QWEN_MIN_PIXELS}" --ranker-qwen-max-pixels "${RANKER_QWEN_MAX_PIXELS}"
        --rerank-type pointwise --rerank-template "${RERANK_TEMPLATE}" --rerank-num "${RERANK_NUM}" --rerank-batch-size "${RERANK_BATCH_SIZE}"
        --rerank-reps-loc "${RERANK_REPS_LOC}" --modify-type "${MODIFY_TYPE}" --output-dir "${JOB_OUTPUT}")
    [[ "${IS_FILTERED}" == false ]] || COMMAND+=(--is-filtered)
    [[ "${SPARSE_VALUE_MEAN}" == false ]] || COMMAND+=(--sparse-value-mean)
    [[ "${OVERWRITE}" == false ]] || COMMAND+=(--overwrite)
    [[ "${USE_RERANK_PRIOR_CACHE}" == false ]] || COMMAND+=(--use-rerank-prior-cache --rerank-prior-cache-dir "${RERANK_PRIOR_CACHE_DIR}")
}

command -v "${DEEPSPEED_BIN}" >/dev/null 2>&1 || die "DeepSpeed executable not found: ${DEEPSPEED_BIN}"
require_file src/three_stage_retrieval_pipeline.py 'Python entrypoint'
require_file "${QWEN_MODEL_PATH}/config.json" 'Retrieval checkpoint config'
require_file "${RANKER_MODEL_PATH}/config.json" 'Ranker checkpoint config'
declare -A SEEN=()
# Validate every job before launching any GPU process.
for dataset in "${DATASET_LIST[@]}"; do
    task_for "${dataset}" >/dev/null
    [[ -z "${SEEN[${dataset}]+present}" ]] || die "Duplicate dataset: ${dataset}"
    SEEN["${dataset}"]=1
    resolve_job "${dataset}"
    preflight_job
done
printf 'Qwen three-stage: family=%s model=%s datasets=%s GPUs=%s\n' "${QWEN_FAMILY}" "${QWEN_MODEL_PATH}" "${DATASET_LIST[*]}" "${GPUS}"
for dataset in "${DATASET_LIST[@]}"; do
    resolve_job "${dataset}"
    build_command
    printf '\nDataset: %s (%s, %s)\nDense corpus: %s\n' "${dataset}" "${DATASET_SPLIT}" "${DIRECTION}" "${JOB_DENSE}"
    printf 'Stage-1 Lucene index: %s\nStage-2 Lucene index: %s\n' "$(resolve_index "${JOB_FIRST}")" "$(resolve_index "${JOB_SPARSE}")"
    if [[ "${SELECT_HYBRID_ALPHA}" == false ]]; then
        printf 'Alpha: fixed %s\n' "${HYBRID_ALPHA}"
    elif [[ -n "${JOB_VAL_DENSE}" ]]; then
        printf 'Alpha: select 0.1..0.9 on %s/%s\nValidation dense: %s\nValidation Lucene index: %s\n' "${JOB_VAL_DATASET}" "${VALIDATION_SPLIT}" "${JOB_VAL_DENSE}" "$(resolve_index "${JOB_VAL_SPARSE}")"
    else
        printf 'Skipping validation: no val paths specified.\nAlpha: select 0.1..0.9 on target %s/%s\n' "${dataset}" "${DATASET_SPLIT}"
    fi
    printf 'Results: %s\nCommand:' "${JOB_OUTPUT}"; printf ' %q' "${COMMAND[@]}"; printf '\n'
    if "${COMMAND[@]}"; then
        printf 'Completed: %s\n' "${dataset}"
    else
        status=$?
        printf 'Failed: %s (exit %s); stopping Qwen batch.\n' "${dataset}" "${status}" >&2
        exit "${status}"
    fi
done
printf '\nCompleted all %s dataset jobs.\n' "${#DATASET_LIST[@]}"
