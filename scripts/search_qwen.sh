#!/bin/bash
# Standalone Qwen retrieval, configured like search.sh. One dataset per run.
# Examples:
#   bash scripts/search_qwen.sh
#   QWEN_FAMILY=qwen3 DATASET=coco bash scripts/search_qwen.sh
#   USE_FIXED_ALPHA=true ALPHA=0.5 bash scripts/search_qwen.sh
# Prepare matching test representations/indexes. Val is used only when a val
# path is explicitly configured; otherwise select on the target split directly.
set -euo pipefail

QWEN_FAMILY="${QWEN_FAMILY:-qwen25}"
QWEN_CHECKPOINT_ROOT="${QWEN_CHECKPOINT_ROOT:-/root/autodl-fs/checkpoints}"
GPUS="${GPUS:-1}"
NUM_NODES="${NUM_NODES:-1}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"
# DATASET defaults to flickr after the single-value DATASETS compatibility check.
DATASET="${DATASET:-}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
IR_QUERY_TYPE="${IR_QUERY_TYPE:-image}"
EOL_TYPE="${EOL_TYPE:-disassembleeol_separate}"
PROMPT_TYPE="${PROMPT_TYPE:-prompt_5}"
RANDOM_VIEW_SEED="${RANDOM_VIEW_SEED:-42}"
REPS_LOC="${REPS_LOC:-after_pad}"
DTYPE="${DTYPE:-float16}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
RETRIEVAL_BATCH_SIZE="${RETRIEVAL_BATCH_SIZE:-1}"
DEPTH="${DEPTH:-1000}"
THREADS="${THREADS:-16}"
USE_GPU="${USE_GPU:-false}"
QWEN_MIN_PIXELS="${QWEN_MIN_PIXELS:-0}"
QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS:-0}"
SPARSE_LENGTH="${SPARSE_LENGTH:-40}"
TEXT_SPARSE_LENGTH="${TEXT_SPARSE_LENGTH:-${SPARSE_LENGTH}}"
IMAGE_SPARSE_LENGTH="${IMAGE_SPARSE_LENGTH:-${SPARSE_LENGTH}}"
SPARSE_VALUE_TYPE="${SPARSE_VALUE_TYPE:-sum}"
SPARSE_LOWER_OR_UPPER="${SPARSE_LOWER_OR_UPPER:-lower}"
NUM_EXPENDED_TOKENS="${NUM_EXPENDED_TOKENS:-0}"
SPARSE_MANUAL="${SPARSE_MANUAL:-false}"
SPARSE_VALUE_MEAN="${SPARSE_VALUE_MEAN:-false}"
IS_FILTERED="${IS_FILTERED:-false}"
TBPR_TYPE="${TBPR_TYPE:-origin_type}"
CIR_TYPE="${CIR_TYPE:-type}"
# Select alpha on explicitly configured val, otherwise on the target.
# Native Qwen fusion uses dense=alpha and sparse=1-alpha.
USE_FIXED_ALPHA="${USE_FIXED_ALPHA:-false}"
ALPHA="${ALPHA:-0.9}"
USE_GENERATED_VALIDATION_DATA="${USE_GENERATED_VALIDATION_DATA:-false}"
GENERATED_VALIDATION_DATA_ROOT="${GENERATED_VALIDATION_DATA_ROOT:-./generated_llava70b_data}"
# Empty paths are resolved from the corpus configuration below. Sparse paths are
# normalized to collection/; the searcher reads the sibling Lucene index/.
PASSAGE_REPS="${PASSAGE_REPS:-}"
SPARSE_INDEX="${SPARSE_INDEX:-}"
VAL_PASSAGE_REPS="${VAL_PASSAGE_REPS:-}"
VAL_SPARSE_INDEX="${VAL_SPARSE_INDEX:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

die() { printf 'Error: %s\n' "$*" >&2; exit 2; }
if (($#)); then
    if (($# == 1)) && [[ "$1" == -h || "$1" == --help ]]; then
        cat <<'USAGE'
Usage: bash scripts/search_qwen.sh [-h|--help]
Edit the configuration block above or set environment variables.
Defaults: Qwen2.5-VL-7B-Instruct, DATASET=flickr, IR_QUERY_TYPE=image, test,
GPUS=1, disassembleeol_separate+concat, float16, automatic alpha selection.
QWEN_FAMILY=qwen2|qwen25|qwen3; QWEN_MODEL_PATH overrides the local checkpoint.
EOL_TYPE=prompteol selects separate. DTYPE=float16|bfloat16|float32.
USE_FIXED_ALPHA=false selects alpha from 0.1..0.9 once, on rank 0.
Without explicit val paths it uses the target split (default: test), even if default
val directories exist. Explicit val paths must resolve to complete matching indexes.
USE_FIXED_ALPHA=true uses ALPHA directly (default: 0.9); no val indexes needed.
Native Qwen fusion weights are alpha for dense and 1-alpha for sparse.
DENSE_OUTPUT_ROOT/SPARSE_OUTPUT_ROOT default to project dense_output/sparse_output.
PASSAGE_REPS/SPARSE_INDEX and VAL_PASSAGE_REPS/VAL_SPARSE_INDEX override derived paths.
Sparse inputs (corpus root, index/ or collection/) are normalized to collection/.
The native searcher reads the parent manifest and sibling index/, not raw JSONL.
QWEN_OUTPUT_ROOT defaults to project qwen_retrieval_results; OUTPUT_DIR overrides
the derived per-model/dataset/direction/search/alpha_val, alpha_test (target split),
or alpha_VALUE (fixed) directory.
USE_GENERATED_VALIDATION_DATA and GENERATED_VALIDATION_DATA_ROOT affect val only.
Results: alpha_sweep.csv (nine weights), best.csv, recall_metrics.json and TREC runs.
Fixed alpha produces best.csv without a sweep. Recall values are percentages (0..100).
Single-value DATASETS is accepted when DATASET is unset; multiple values are rejected.
Supported datasets: flickr coco CUHK-PEDES ICFG-PEDES RSTPReid webqa remuq llava.
Keep prompt, sparse, random seed and pixel settings identical to corpus encoding.
Only -h/--help is accepted; arbitrary CLI arguments are not forwarded.
USAGE
        exit 0
    fi
    die "Configure variables in this script or the environment. See --help."
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
DENSE_OUTPUT_ROOT="${DENSE_OUTPUT_ROOT:-${REPO_ROOT}/dense_output}"
SPARSE_OUTPUT_ROOT="${SPARSE_OUTPUT_ROOT:-${REPO_ROOT}/sparse_output}"
QWEN_OUTPUT_ROOT="${QWEN_OUTPUT_ROOT:-${REPO_ROOT}/qwen_retrieval_results}"

if [[ -z "${DATASET}" ]]; then
    if [[ -v DATASETS ]]; then
        read -r -a DATASET_LIST <<< "${DATASETS//$'\n'/ }"
        ((${#DATASET_LIST[@]} == 1)) || die "Set one DATASET; run search_qwen.sh separately for each dataset."
        DATASET="${DATASET_LIST[0]}"
    else
        DATASET=flickr
    fi
fi
case "${QWEN_FAMILY}" in
    qwen2) CHECKPOINT=Qwen2-VL-7B-Instruct ;;
    qwen25) CHECKPOINT=Qwen2.5-VL-7B-Instruct ;;
    qwen3) CHECKPOINT=Qwen3-VL-8B-Instruct ;;
    *) die "QWEN_FAMILY must be qwen2, qwen25 or qwen3." ;;
esac
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${QWEN_CHECKPOINT_ROOT}/${CHECKPOINT}}"
while [[ "${QWEN_MODEL_PATH}" == */ && "${QWEN_MODEL_PATH}" != / ]]; do
    QWEN_MODEL_PATH="${QWEN_MODEL_PATH%/}"
done
MODEL_TAG="${QWEN_MODEL_PATH##*/}"
[[ -n "${MODEL_TAG}" && "${MODEL_TAG}" != . && "${MODEL_TAG}" != .. ]] || die "Invalid QWEN_MODEL_PATH."
case "${EOL_TYPE}" in
    prompteol) EXPECTED_CALCULATE_TYPE=separate ;;
    disassembleeol_separate) EXPECTED_CALCULATE_TYPE=concat ;;
    *) die "EOL_TYPE must be prompteol or disassembleeol_separate." ;;
esac
CALCULATE_TYPE="${CALCULATE_TYPE:-${EXPECTED_CALCULATE_TYPE}}"
[[ "${CALCULATE_TYPE}" == "${EXPECTED_CALCULATE_TYPE}" ]] || die "${EOL_TYPE} requires CALCULATE_TYPE=${EXPECTED_CALCULATE_TYPE}."
[[ "${IR_QUERY_TYPE}" == image || "${IR_QUERY_TYPE}" == text ]] || die "IR_QUERY_TYPE must be image or text."
case "${DATASET}" in
    flickr|coco)
        TASK_TYPE=ir; QUERY_TYPE="${IR_QUERY_TYPE}"
        if [[ "${QUERY_TYPE}" == image ]]; then
            TARGET_TYPE=text; DIRECTION=image_to_text
        else
            TARGET_TYPE=image; DIRECTION=text_to_image
        fi
        ;;
    CUHK-PEDES|ICFG-PEDES|RSTPReid)
        TASK_TYPE=tbpr; QUERY_TYPE=text; TARGET_TYPE=image; DIRECTION=text_to_image ;;
    webqa) TASK_TYPE=t2it; QUERY_TYPE=text; TARGET_TYPE=corpus; DIRECTION=text_to_imagetext ;;
    remuq|llava) TASK_TYPE=it2t; QUERY_TYPE=image; TARGET_TYPE=corpus; DIRECTION=imagetext_to_text ;;
    *) die "Unsupported dataset: ${DATASET}; set one DATASET." ;;
esac

for name in SPARSE_MANUAL SPARSE_VALUE_MEAN IS_FILTERED USE_GPU USE_FIXED_ALPHA USE_GENERATED_VALIDATION_DATA; do
    case "${!name,,}" in
        true|1|yes) printf -v "${name}" '%s' true ;;
        false|0|no) printf -v "${name}" '%s' false ;;
        *) die "${name} must be true/false or 1/0." ;;
    esac
done
[[ "${DATASET_SPLIT}" =~ ^[a-zA-Z0-9_-]+$ ]] || die "Invalid DATASET_SPLIT."
[[ "${PROMPT_TYPE}" =~ ^[a-zA-Z0-9_]+$ ]] || die "Invalid PROMPT_TYPE."
[[ "${REPS_LOC}" == after_pad || "${REPS_LOC}" == before_pad ]] || die "REPS_LOC must be after_pad or before_pad."
[[ "${TBPR_TYPE}" == origin_type || "${TBPR_TYPE}" == type ]] || die "TBPR_TYPE must be origin_type or type."
[[ "${SPARSE_VALUE_TYPE}" == sum || "${SPARSE_VALUE_TYPE}" == max || "${SPARSE_VALUE_TYPE}" == replace ]] || die "Invalid SPARSE_VALUE_TYPE."
[[ "${SPARSE_LOWER_OR_UPPER}" == lower || "${SPARSE_LOWER_OR_UPPER}" == upper ]] || die "Invalid SPARSE_LOWER_OR_UPPER."
[[ "${ALPHA}" =~ ^(0|0?\.[0-9]+|1(\.0+)?)$ ]] || die "ALPHA must be in [0,1]."
[[ "${RANDOM_VIEW_SEED}" =~ ^-?[0-9]+$ ]] || die "RANDOM_VIEW_SEED must be an integer."
for name in GPUS NUM_NODES PER_DEVICE_BATCH_SIZE RETRIEVAL_BATCH_SIZE SPARSE_LENGTH TEXT_SPARSE_LENGTH IMAGE_SPARSE_LENGTH DEPTH THREADS; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer."
done
for name in NUM_EXPENDED_TOKENS QWEN_MIN_PIXELS QWEN_MAX_PIXELS; do
    [[ "${!name}" =~ ^(0|[1-9][0-9]*)$ ]] || die "${name} must be a nonnegative integer."
done
if ((QWEN_MIN_PIXELS > 0 && QWEN_MAX_PIXELS > 0 && QWEN_MIN_PIXELS > QWEN_MAX_PIXELS)); then
    die "QWEN_MIN_PIXELS must not exceed QWEN_MAX_PIXELS."
fi
PRECISION_ARGS=()
case "${DTYPE}" in
    float16) PRECISION_ARGS+=(--fp16) ;;
    bfloat16) PRECISION_ARGS+=(--bf16) ;;
    float32) ;;
    *) die "DTYPE must be float16, bfloat16 or float32." ;;
esac

# Same directory contract as qwen_vl_pipeline.artifact_dirs(). An explicit val
# override may use the default val path for its unspecified counterpart.
relative_path() {
    local split="$1" filter=no_filter manual=no_manual lengths="${SPARSE_LENGTH}" mean=no_mean
    [[ "${IS_FILTERED}" == false ]] || filter=filter
    [[ "${SPARSE_VALUE_MEAN}" == false ]] || mean=mean
    if [[ "${SPARSE_MANUAL}" == true ]]; then
        manual=manual; lengths="${TEXT_SPARSE_LENGTH}_${IMAGE_SPARSE_LENGTH}"
    fi
    local relative="${MODEL_TAG}/${DATASET}/${TARGET_TYPE}/${filter}/${CALCULATE_TYPE}/${PROMPT_TYPE}/${split}"
    [[ "${TASK_TYPE}" != tbpr ]] || relative+="/${TBPR_TYPE}"
    printf '%s/%s_%s_%s_%s_no_cluster_%s_%s_%s_%s' "${relative}" "${NUM_EXPENDED_TOKENS}" "${manual}" "${lengths}" "${SPARSE_VALUE_TYPE}" "${REPS_LOC}" "${EOL_TYPE}" "${SPARSE_LOWER_OR_UPPER}" "${mean}"
}
sparse_collection_path() {
    local path="$1"
    while [[ "${path}" == */ && "${path}" != / ]]; do path="${path%/}"; done
    case "${path##*/}" in
        collection) printf '%s' "${path}" ;;
        index)
            path="$(dirname -- "${path}")"
            if [[ "${path##*/}" == collection ]]; then
                printf '%s' "${path}"
            else
                printf '%s/collection' "${path}"
            fi
            ;;
        *) printf '%s/collection' "${path%/}" ;;
    esac
}
sparse_index_path() {
    local path="$1"
    if [[ "${path##*/}" == index ]]; then
        printf '%s' "${path}"
        return
    fi
    [[ "${path##*/}" == collection ]] || path="${path%/}/collection"
    # An incomplete nested index must fail validation instead of silently
    # selecting an older sibling index.
    local legacy="$(dirname -- "${path}")/index"
    if [[ -e "${path}/index" || -L "${path}/index" ]]; then
        printf '%s/index' "${path}"
    elif [[ -e "${legacy}" || -L "${legacy}" ]]; then
        printf '%s' "${legacy}"
    else
        printf '%s/index' "${path}"
    fi
}
shopt -s nullglob
corpus_problem() {
    local dense="$1" sparse="$2" index
    local -a shards=("${dense}"/corpus*.pkl)
    if [[ ! -f "${dense}" ]] && { ((${#shards[@]} == 0)) || [[ ! -f "${shards[0]}" ]]; }; then
        printf 'no dense corpus*.pkl found in %s; run encode_qwen.sh with matching settings.' "${dense}"
        return 1
    fi
    index="$(sparse_index_path "${sparse}")"
    local -a segments=("${index}"/segments_*)
    if ((${#segments[@]} == 0)) || [[ ! -f "${segments[0]}" ]]; then
        printf 'no Lucene segments_* found in %s; run sparse_index_qwen.sh.' "${index}"
        return 1
    fi
}
require_corpus() {
    local label="$1" problem
    if problem="$(corpus_problem "$2" "$3")"; then
        return 0
    else
        die "${label}: ${problem}"
    fi
}
RELATIVE="$(relative_path "${DATASET_SPLIT}")"
PASSAGE_REPS="${PASSAGE_REPS:-${DENSE_OUTPUT_ROOT}/${RELATIVE}}"
SPARSE_INDEX="$(sparse_collection_path "${SPARSE_INDEX:-${SPARSE_OUTPUT_ROOT}/${RELATIVE}/collection}")"
command -v "${DEEPSPEED_BIN}" >/dev/null 2>&1 || die "DeepSpeed executable not found: ${DEEPSPEED_BIN}"
[[ -f "${QWEN_MODEL_PATH}/config.json" ]] || die "Checkpoint config.json not found: ${QWEN_MODEL_PATH}"
[[ -f src/search.py ]] || die "Python entrypoint not found: src/search.py"
require_corpus Test "${PASSAGE_REPS}" "${SPARSE_INDEX}"
ALPHA_LABEL="alpha_${ALPHA}"
ALPHA_SELECTION_SOURCE=fixed
FUSION_ALPHA_ARGS=()
if [[ "${USE_FIXED_ALPHA}" == true ]]; then
    FUSION_ALPHA_ARGS+=(--use_fixed_alpha)
else
    VAL_PATHS_EXPLICIT=false
    [[ -z "${VAL_PASSAGE_REPS}" && -z "${VAL_SPARSE_INDEX}" ]] || VAL_PATHS_EXPLICIT=true
    if [[ "${VAL_PATHS_EXPLICIT}" == true ]]; then
        VAL_RELATIVE="$(relative_path val)"
        VAL_PASSAGE_REPS="${VAL_PASSAGE_REPS:-${DENSE_OUTPUT_ROOT}/${VAL_RELATIVE}}"
        VAL_SPARSE_INDEX="$(sparse_collection_path "${VAL_SPARSE_INDEX:-${SPARSE_OUTPUT_ROOT}/${VAL_RELATIVE}/collection}")"
        require_corpus Validation "${VAL_PASSAGE_REPS}" "${VAL_SPARSE_INDEX}"
        ALPHA_SELECTION_SOURCE=validation
        FUSION_ALPHA_ARGS+=(--val_passage_reps "${VAL_PASSAGE_REPS}" --val_sparse_index "${VAL_SPARSE_INDEX}")
        ALPHA_LABEL=alpha_val
        if [[ "${USE_GENERATED_VALIDATION_DATA}" == true ]]; then
            FUSION_ALPHA_ARGS+=(--use_generated_validation_data --generated_validation_data_root "${GENERATED_VALIDATION_DATA_ROOT}")
        fi
    else
        ALPHA_SELECTION_SOURCE=target
        ALPHA_LABEL="alpha_${DATASET_SPLIT}"
        printf 'Skipping validation: no val paths specified.\nAlpha selection will use %s/%s.\n' "${DATASET}" "${DATASET_SPLIT}"
    fi
fi
OUTPUT_DIR="${OUTPUT_DIR:-${QWEN_OUTPUT_ROOT}/${RELATIVE}/${DIRECTION}/search/${ALPHA_LABEL}}"
SPARSE_ARGS=()
for name in SPARSE_MANUAL SPARSE_VALUE_MEAN IS_FILTERED; do
    [[ "${!name}" == false ]] || SPARSE_ARGS+=("--${name,,}")
done
GPU_ARGS=()
[[ "${USE_GPU}" == false ]] || GPU_ARGS+=(--use_gpu)

printf 'Qwen search: model=%s dataset=%s direction=%s method=%s+%s\n' "${QWEN_MODEL_PATH}" "${DATASET}" "${DIRECTION}" "${EOL_TYPE}" "${CALCULATE_TYPE}"
printf 'Test dense: %s\nTest sparse: %s\n' "${PASSAGE_REPS}" "${SPARSE_INDEX}"
printf 'Test Lucene index: %s\n' "$(sparse_index_path "${SPARSE_INDEX}")"
if [[ "${ALPHA_SELECTION_SOURCE}" == validation ]]; then
    printf 'Validation dense: %s\nValidation sparse: %s\nAlpha: select 0.1..0.9 on val\n' "${VAL_PASSAGE_REPS}" "${VAL_SPARSE_INDEX}"
    printf 'Validation Lucene index: %s\n' "$(sparse_index_path "${VAL_SPARSE_INDEX}")"
elif [[ "${ALPHA_SELECTION_SOURCE}" == target ]]; then
    printf 'Alpha: select 0.1..0.9 on target %s/%s\n' "${DATASET}" "${DATASET_SPLIT}"
else
    printf 'Alpha: fixed %s\n' "${ALPHA}"
fi
printf 'Results: %s\n' "${OUTPUT_DIR}"

"${DEEPSPEED_BIN}" --num_gpus="${GPUS}" --num_nodes="${NUM_NODES}" src/search.py \
    --model_name_or_path "${QWEN_MODEL_PATH}" \
    --dataset_name "${DATASET}" \
    --dataset_split "${DATASET_SPLIT}" \
    --task_type "${TASK_TYPE}" \
    --encode_type "${TARGET_TYPE}" \
    --query_type "${QUERY_TYPE}" \
    --passage_reps "${PASSAGE_REPS}" \
    --sparse_index "${SPARSE_INDEX}" \
    --output_dir "${OUTPUT_DIR}" \
    --per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --retrieval_batch_size "${RETRIEVAL_BATCH_SIZE}" \
    --depth "${DEPTH}" \
    --threads "${THREADS}" \
    --alpha "${ALPHA}" \
    --reps_loc "${REPS_LOC}" \
    --eol_type "${EOL_TYPE}" \
    --calculate_type "${CALCULATE_TYPE}" \
    --prompt_type "${PROMPT_TYPE}" \
    --random_view_seed "${RANDOM_VIEW_SEED}" \
    --sparse_length "${SPARSE_LENGTH}" \
    --text_sparse_length "${TEXT_SPARSE_LENGTH}" \
    --image_sparse_length "${IMAGE_SPARSE_LENGTH}" \
    --sparse_value_type "${SPARSE_VALUE_TYPE}" \
    --sparse_lower_or_upper "${SPARSE_LOWER_OR_UPPER}" \
    --num_expended_tokens "${NUM_EXPENDED_TOKENS}" \
    --qwen_min_pixels "${QWEN_MIN_PIXELS}" \
    --qwen_max_pixels "${QWEN_MAX_PIXELS}" \
    --tbpr_type "${TBPR_TYPE}" \
    --cir_type "${CIR_TYPE}" \
    "${PRECISION_ARGS[@]}" \
    "${SPARSE_ARGS[@]}" \
    "${GPU_ARGS[@]}" \
    "${FUSION_ALPHA_ARGS[@]}"
