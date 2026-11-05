#!/usr/bin/env bash
# Standalone LLaVA-NeXT retrieval and reranking. Prepare matching encode.py outputs and indexes first.
# Examples:
#   bash scripts/rerank_llava_base.sh
#   QUERY_TYPE=text LLAVA_IMAGE_MODE=base bash scripts/rerank_llava_base.sh
#   BASE_MODEL=llava-hf-llama3-llava-next-8b-hf bash scripts/rerank_llava_base.sh
#   LLAVA_IMAGE_MODE=anyres USE_FIXED_ALPHA=true ALPHA=0.5 bash scripts/rerank_llava_base.sh
#   DATASET=CUHK-PEDES TBPR_TYPE=type bash scripts/rerank_llava_base.sh
#   PASSAGE_REPS=/my/dense/leaf SPARSE_INDEX=/my/sparse/leaf/index bash scripts/rerank_llava_base.sh
set -euo pipefail
die() { printf 'Error: %s\n' "$*" >&2; exit 2; }
if (($#)); then
    if (($# == 1)) && [[ "$1" == -h || "$1" == --help ]]; then
        cat <<'USAGE'
Usage: bash scripts/rerank_llava_base.sh [-h|--help]
Independent launcher for src/rerank.py; existing shell launchers are unchanged.
Configure variables below or through the environment. Default: base LLaVA-NeXT
Mistral-7B, Flickr image->text, test, 4 GPUs, float16, prompt_5,
disassembleeol_separate+concat, sparse length 40, alpha selection on validation.
LLAVA_IMAGE_MODE=base|anyres (default: base). BASE_MODEL or MODEL_PATH selects
the checkpoint; CHECKPOINT_ROOT defaults to /root/autodl-fs/checkpoints.
QUERY_TYPE controls IR direction. Dataset selects task automatically; TASK_TYPE
may be specified explicitly. TBPR_TYPE=origin_type|type. One dataset per run.
EOL_TYPE=prompteol uses separate; disassembleeol_separate uses concat.
PASSAGE_REPS/SPARSE_INDEX and VAL_PASSAGE_REPS/VAL_SPARSE_INDEX override derived
paths. Index aliases: corpus root, collection/, index/, collection/index/.
Image candidate leaves append _llava_base; text candidate paths are reused.
DENSE_OUTPUT_ROOT/SPARSE_OUTPUT_ROOT default to project dense_output/sparse_output.
ARTIFACT_MODEL_NAME overrides the model directory used in derived artifact paths.
USE_FIXED_ALPHA=true uses ALPHA (default: 0.9), ignoring validation paths.
USE_GENERATED_VALIDATION_DATA=true uses generated val; its path marker is
_generated_val before _llava_base. VALIDATION_SPLIT overrides the inferred index
directory split; Python retains its existing validation-dataset selection.
GPUS, NUM_NODES, DEEPSPEED_BIN, DTYPE, DEPTH and batch sizes are configurable.
RERANK_NUM=5, RERANK_BATCH_SIZE=1, RERANK_TYPE=pointwise,
RERANK_TEMPLATE=caption_generation, MODIFY_TYPE=no. USE_RERANK_PRIOR_CACHE=true
requires MODIFY_TYPE=modify_single and a supported caption text-candidate task.
ESTIMATE_TIME_ONLY=true writes output_dir/time_estimate.json.
OUTPUT_DIR overrides the auxiliary output directory. Recall results are written
by Python under its existing search_results tree, with _llava_base on experiment
leaves in base mode. Both destinations are printed before launch.
Only -h/--help is accepted; arbitrary CLI options are not forwarded.
USAGE
        exit 0
    fi
    die 'Use environment variables or the configuration block; see --help.'
fi
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

LLAVA_IMAGE_MODE="${LLAVA_IMAGE_MODE:-base}"
BASE_MODEL="${BASE_MODEL:-llava-hf-llava-v1.6-mistral-7b-hf}"
MODEL_PATH="${MODEL_PATH:-${CHECKPOINT_ROOT:-/root/autodl-fs/checkpoints}/${BASE_MODEL}}"
MODEL_PATH="${MODEL_PATH%/}"
ARTIFACT_MODEL_NAME="${ARTIFACT_MODEL_NAME:-${MODEL_PATH##*/}}"
GPUS="${GPUS:-4}"
NUM_NODES="${NUM_NODES:-1}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"
DTYPE="${DTYPE:-float16}"
DATASET="${DATASET:-flickr}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
PROMPT_TYPE="${PROMPT_TYPE:-prompt_5}"
RANDOM_VIEW_SEED="${RANDOM_VIEW_SEED:-42}"
REPS_LOC="${REPS_LOC:-after_pad}"
EOL_TYPE="${EOL_TYPE:-disassembleeol_separate}"
SPARSE_LENGTH="${SPARSE_LENGTH:-${MANUAL_LENGTH:-40}}"
TEXT_SPARSE_LENGTH="${TEXT_SPARSE_LENGTH:-${SPARSE_LENGTH}}"
IMAGE_SPARSE_LENGTH="${IMAGE_SPARSE_LENGTH:-${SPARSE_LENGTH}}"
NUM_EXPENDED_TOKENS="${NUM_EXPENDED_TOKENS:-${EXPENDED_TOKENS:-0}}"
SPARSE_VALUE_TYPE="${SPARSE_VALUE_TYPE:-sum}"
SPARSE_LOWER_OR_UPPER="${SPARSE_LOWER_OR_UPPER:-lower}"
SPARSE_TYPE="${SPARSE_TYPE:-single}"
SPARSE_MANUAL="${SPARSE_MANUAL:-false}"
SPARSE_VALUE_MEAN="${SPARSE_VALUE_MEAN:-false}"
IS_FILTERED="${IS_FILTERED:-false}"
TBPR_TYPE="${TBPR_TYPE:-origin_type}"
CIR_TYPE="${CIR_TYPE:-type}"
DEPTH="${DEPTH:-1000}"
THREADS="${THREADS:-16}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
RETRIEVAL_BATCH_SIZE="${RETRIEVAL_BATCH_SIZE:-1}"
USE_GPU="${USE_GPU:-true}"
REMOVE_QUERY="${REMOVE_QUERY:-true}"
ALPHA="${ALPHA:-0.9}"
BETA="${BETA:-0.1}"
USE_FIXED_ALPHA="${USE_FIXED_ALPHA:-false}"
USE_GENERATED_VALIDATION_DATA="${USE_GENERATED_VALIDATION_DATA:-false}"
GENERATED_VALIDATION_DATA_ROOT="${GENERATED_VALIDATION_DATA_ROOT:-${REPO_ROOT}/generated_llava70b_data}"
DENSE_OUTPUT_ROOT="${DENSE_OUTPUT_ROOT:-${REPO_ROOT}/dense_output}"
SPARSE_OUTPUT_ROOT="${SPARSE_OUTPUT_ROOT:-${REPO_ROOT}/sparse_output}"
RERANK_NUM="${RERANK_NUM:-5}"
RERANK_BATCH_SIZE="${RERANK_BATCH_SIZE:-1}"
RERANK_TYPE="${RERANK_TYPE:-pointwise}"
RERANK_TEMPLATE="${RERANK_TEMPLATE:-caption_generation}"
MODIFY_TYPE="${MODIFY_TYPE:-no}"
USE_RERANK_PRIOR_CACHE="${USE_RERANK_PRIOR_CACHE:-false}"
RERANK_PRIOR_CACHE_DIR="${RERANK_PRIOR_CACHE_DIR:-${REPO_ROOT}/cache/rerank_prior}"
ESTIMATE_TIME_ONLY="${ESTIMATE_TIME_ONLY:-false}"

bool() {
    case "${!1,,}" in
        true|1|yes) printf -v "$1" '%s' true ;;
        false|0|no) printf -v "$1" '%s' false ;;
        *) die "$1 must be true/false or 1/0." ;;
    esac
}
for name in SPARSE_MANUAL SPARSE_VALUE_MEAN IS_FILTERED USE_GPU REMOVE_QUERY USE_FIXED_ALPHA USE_GENERATED_VALIDATION_DATA; do bool "${name}"; done
for name in USE_RERANK_PRIOR_CACHE ESTIMATE_TIME_ONLY; do bool "${name}"; done
for name in RERANK_NUM RERANK_BATCH_SIZE; do [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || die "Invalid ${name}."; done
case "${RERANK_TYPE}" in pointwise|listwise) ;; *) die 'Invalid RERANK_TYPE.' ;; esac
case "${MODIFY_TYPE}" in no|modify_single) ;; *) die 'Invalid MODIFY_TYPE.' ;; esac
case "${LLAVA_IMAGE_MODE}" in anyres|base) ;; *) die 'LLAVA_IMAGE_MODE must be anyres or base.' ;; esac
case "${EOL_TYPE}" in
    prompteol) EXPECTED_CALCULATE=separate ;;
    disassembleeol_separate) EXPECTED_CALCULATE=concat ;;
    *) die 'EOL_TYPE must be prompteol or disassembleeol_separate.' ;;
esac
CALCULATE_TYPE="${CALCULATE_TYPE:-${EXPECTED_CALCULATE}}"
[[ "${CALCULATE_TYPE}" == "${EXPECTED_CALCULATE}" ]] || die "${EOL_TYPE} requires CALCULATE_TYPE=${EXPECTED_CALCULATE}."
case "${DATASET}" in
    flickr|coco) EXPECTED_TASK=ir ;;
    CUHK-PEDES|ICFG-PEDES|RSTPReid) EXPECTED_TASK=tbpr ;;
    webqa|edis) EXPECTED_TASK=t2it ;;
    remuq|llava) EXPECTED_TASK=it2t ;;
    fashion-iq) EXPECTED_TASK=cir ;;
    *) die "Unsupported DATASET: ${DATASET}" ;;
esac
TASK_TYPE="${TASK_TYPE:-${EXPECTED_TASK}}"
[[ "${TASK_TYPE}" == "${EXPECTED_TASK}" ]] || die "${DATASET} requires TASK_TYPE=${EXPECTED_TASK}."
HAS_IMAGES=true
case "${TASK_TYPE}" in
    ir)
        QUERY_TYPE="${QUERY_TYPE:-image}"
        case "${QUERY_TYPE}" in image) EXPECTED_TARGET=text; HAS_IMAGES=false ;; text) EXPECTED_TARGET=image ;; *) die 'IR QUERY_TYPE must be image or text.' ;; esac ;;
    tbpr|t2it)
        QUERY_TYPE="${QUERY_TYPE:-text}"
        [[ "${QUERY_TYPE}" == text ]] || die "${TASK_TYPE} requires QUERY_TYPE=text."
        EXPECTED_TARGET=image
        [[ "${TASK_TYPE}" != t2it ]] || EXPECTED_TARGET=corpus ;;
    it2t)
        QUERY_TYPE="${QUERY_TYPE:-image}"
        [[ "${QUERY_TYPE}" == image || "${QUERY_TYPE}" == composed ]] || die 'it2t QUERY_TYPE must be image or composed.'
        EXPECTED_TARGET=corpus; HAS_IMAGES=false ;;
    cir) QUERY_TYPE="${QUERY_TYPE:-composed}"; [[ "${QUERY_TYPE}" == composed ]] || die 'CIR requires QUERY_TYPE=composed.'; EXPECTED_TARGET=image ;;
esac
TARGET_TYPE="${TARGET_TYPE:-${EXPECTED_TARGET}}"
[[ "${TARGET_TYPE}" == "${EXPECTED_TARGET}" ]] || die "${TASK_TYPE}/${QUERY_TYPE} requires TARGET_TYPE=${EXPECTED_TARGET}."
if [[ "${MODIFY_TYPE}" == modify_single || "${USE_RERANK_PRIOR_CACHE}" == true ]]; then
    [[ "${RERANK_TEMPLATE}" == *caption_generation* ]] || die 'Prior correction requires a caption_generation template.'
    [[ "${TASK_TYPE}" == it2t || ( "${TASK_TYPE}" == ir && "${QUERY_TYPE}" == image ) ]] || die 'Prior correction requires a text-candidate caption task.'
fi
if [[ "${USE_RERANK_PRIOR_CACHE}" == true ]]; then
    [[ "${MODIFY_TYPE}" == modify_single && "${RERANK_TYPE}" == pointwise ]] || die 'Prior cache requires modify_single and pointwise reranking.'
fi
for name in GPUS NUM_NODES DEPTH THREADS PER_DEVICE_BATCH_SIZE RETRIEVAL_BATCH_SIZE SPARSE_LENGTH TEXT_SPARSE_LENGTH IMAGE_SPARSE_LENGTH; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer."
done
[[ "${NUM_EXPENDED_TOKENS}" =~ ^[0-9]+$ ]] || die 'NUM_EXPENDED_TOKENS must be nonnegative.'
[[ "${RANDOM_VIEW_SEED}" =~ ^-?[0-9]+$ ]] || die 'RANDOM_VIEW_SEED must be an integer.'
for name in ALPHA BETA; do [[ "${!name}" =~ ^(0|0?\.[0-9]+|1(\.0+)?)$ ]] || die "${name} must be in [0,1]."; done
[[ "${REPS_LOC}" == after_pad || "${REPS_LOC}" == before_pad ]] || die 'Invalid REPS_LOC.'
[[ "${TBPR_TYPE}" == origin_type || "${TBPR_TYPE}" == type ]] || die 'Invalid TBPR_TYPE.'
case "${SPARSE_VALUE_TYPE}" in sum|max|replace) ;; *) die 'Invalid SPARSE_VALUE_TYPE.' ;; esac
case "${SPARSE_LOWER_OR_UPPER}" in lower|upper) ;; *) die 'Invalid SPARSE_LOWER_OR_UPPER.' ;; esac
for name in DATASET_SPLIT PROMPT_TYPE CIR_TYPE SPARSE_TYPE; do [[ "${!name}" =~ ^[a-zA-Z0-9_-]+$ ]] || die "Invalid ${name}."; done
PRECISION_ARGS=()
case "${DTYPE}" in float16) PRECISION_ARGS+=(--fp16) ;; bfloat16) PRECISION_ARGS+=(--bf16) ;; float32) ;; *) die 'Invalid DTYPE.' ;; esac

FILTER=no_filter; MANUAL=no_manual; MEAN=no_mean; LENGTHS="${SPARSE_LENGTH}"
[[ "${IS_FILTERED}" == false ]] || FILTER=filter
[[ "${SPARSE_VALUE_MEAN}" == false ]] || MEAN=mean
if [[ "${SPARSE_MANUAL}" == true ]]; then MANUAL=manual; LENGTHS="${TEXT_SPARSE_LENGTH}_${IMAGE_SPARSE_LENGTH}"; fi
if [[ "${PROMPT_TYPE}" == prompt_random_5 ]]; then MEAN+="_random_view_seed_${RANDOM_VIEW_SEED}"; fi
CORPUS_MODE_SUFFIX=''; RESULT_MODE_SUFFIX=''
if [[ "${LLAVA_IMAGE_MODE}" == base ]]; then
    RESULT_MODE_SUFFIX=_llava_base
    [[ "${HAS_IMAGES}" == false ]] || CORPUS_MODE_SUFFIX=_llava_base
fi
relative_path() {
    local split="$1" generated="${2:-}" path leaf
    path="${ARTIFACT_MODEL_NAME}/${DATASET}/${TARGET_TYPE}/${FILTER}/${CALCULATE_TYPE}/${PROMPT_TYPE}"
    if [[ "${PROMPT_TYPE}" == prompt_random_5 && "${TASK_TYPE}" == tbpr ]]; then path+="/${TBPR_TYPE}"; fi
    path+="/${split}"
    if [[ "${SPARSE_MANUAL}" == false && "${SPARSE_TYPE}" != fusion && "${PROMPT_TYPE}" != prompt_random_5 ]]; then
        [[ "${TASK_TYPE}" != tbpr ]] || path+="/${TBPR_TYPE}"
        [[ "${TASK_TYPE}" != cir ]] || path+="/${CIR_TYPE}"
    fi
    leaf="${NUM_EXPENDED_TOKENS}_${MANUAL}_${LENGTHS}_${SPARSE_VALUE_TYPE}_no_cluster_${REPS_LOC}_${EOL_TYPE}_${SPARSE_LOWER_OR_UPPER}_${MEAN}"
    [[ "${SPARSE_TYPE}" == fusion ]] || leaf+="${generated}"
    # Only fusion artifacts include sparse_type, matching native encode.py.
    [[ "${SPARSE_TYPE}" != fusion ]] || leaf+="_${SPARSE_TYPE}${generated}"
    printf '%s/%s%s' "${path}" "${leaf}" "${CORPUS_MODE_SUFFIX}"
}
RELATIVE="$(relative_path "${DATASET_SPLIT}")"
PASSAGE_REPS="${PASSAGE_REPS:-${DENSE_OUTPUT_ROOT}/${RELATIVE}}"
SPARSE_INDEX="${SPARSE_INDEX:-${SPARSE_OUTPUT_ROOT}/${RELATIVE}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/${ARTIFACT_MODEL_NAME}/${DATASET}/${QUERY_TYPE}/${EOL_TYPE}_${PROMPT_TYPE}${RESULT_MODE_SUFFIX}/result}"
FUSION_ARGS=()
if [[ "${USE_FIXED_ALPHA}" == true ]]; then
    FUSION_ARGS+=(--use_fixed_alpha)
else
    INFERRED_VAL_SPLIT=val
    if [[ "${TASK_TYPE}" == tbpr && "${DATASET}" != RSTPReid ]]; then INFERRED_VAL_SPLIT=train; fi
    if [[ "${TASK_TYPE}" == t2it || "${TASK_TYPE}" == it2t ]]; then INFERRED_VAL_SPLIT="${DATASET_SPLIT}"; fi
    VALIDATION_PATH_SUFFIX=''
    if [[ "${USE_GENERATED_VALIDATION_DATA}" == true ]]; then INFERRED_VAL_SPLIT=val; VALIDATION_PATH_SUFFIX=_generated_val; fi
    VALIDATION_SPLIT="${VALIDATION_SPLIT:-${INFERRED_VAL_SPLIT}}"
    [[ "${VALIDATION_SPLIT}" =~ ^[a-zA-Z0-9_-]+$ ]] || die 'Invalid VALIDATION_SPLIT.'
    VAL_RELATIVE="$(relative_path "${VALIDATION_SPLIT}" "${VALIDATION_PATH_SUFFIX}")"
    VAL_PASSAGE_REPS="${VAL_PASSAGE_REPS:-${DENSE_OUTPUT_ROOT}/${VAL_RELATIVE}}"
    VAL_SPARSE_INDEX="${VAL_SPARSE_INDEX:-${SPARSE_OUTPUT_ROOT}/${VAL_RELATIVE}}"
    FUSION_ARGS+=(--val_passage_reps "${VAL_PASSAGE_REPS}" --val_sparse_index "${VAL_SPARSE_INDEX}")
fi
OPTIONAL_ARGS=()
for name in SPARSE_MANUAL SPARSE_VALUE_MEAN IS_FILTERED USE_GPU REMOVE_QUERY; do
    [[ "${!name}" == false ]] || OPTIONAL_ARGS+=("--${name,,}")
done
if [[ "${USE_GENERATED_VALIDATION_DATA}" == true ]]; then
    OPTIONAL_ARGS+=(--use_generated_validation_data --generated_validation_data_root "${GENERATED_VALIDATION_DATA_ROOT}")
fi
[[ "${ESTIMATE_TIME_ONLY}" == false ]] || OPTIONAL_ARGS+=(--estimate_time_only)
[[ "${USE_RERANK_PRIOR_CACHE}" == false ]] || OPTIONAL_ARGS+=(--use_rerank_prior_cache --rerank_prior_cache_dir "${RERANK_PRIOR_CACHE_DIR}")
resolve_index() {
    local root="${1%/}"
    if [[ "${root##*/}" == index ]]; then printf '%s' "${root}"
    elif [[ -d "${root}/index" ]]; then printf '%s/index' "${root}"
    elif [[ -d "${root}/collection/index" ]]; then printf '%s/collection/index' "${root}"
    else printf '%s' "${root}"
    fi
}
shopt -s nullglob
require_corpus() {
    local label="$1" dense="$2" index
    local -a files=("${dense}"/corpus*.pkl)
    ((${#files[@]})) || die "${label}: no corpus*.pkl in ${dense}; encode with matching settings."
    index="$(resolve_index "$3")"
    local -a segments=("${index}"/segments_*)
    ((${#segments[@]})) || die "${label}: no Lucene segments_* in ${index}; build the sparse index first."
}
command -v "${DEEPSPEED_BIN}" >/dev/null 2>&1 || die "DeepSpeed not found: ${DEEPSPEED_BIN}"
[[ -f "${MODEL_PATH}/config.json" ]] || die "Model config.json not found: ${MODEL_PATH}"
require_corpus Test "${PASSAGE_REPS}" "${SPARSE_INDEX}"
[[ "${USE_FIXED_ALPHA}" == true ]] || require_corpus Validation "${VAL_PASSAGE_REPS}" "${VAL_SPARSE_INDEX}"
printf 'LLaVA rerank: mode=%s model=%s task=%s dataset=%s query=%s\n' "${LLAVA_IMAGE_MODE}" "${MODEL_PATH}" "${TASK_TYPE}" "${DATASET}" "${QUERY_TYPE}"
printf 'Dense: %s\nSparse: %s\nAuxiliary output: %s\n' "${PASSAGE_REPS}" "${SPARSE_INDEX}" "${OUTPUT_DIR}"
RECALL_OUTPUT="/root/autodl-fs/search_results/${MODEL_PATH:28}/${DATASET}/${QUERY_TYPE}/${FILTER}/${CALCULATE_TYPE}/${PROMPT_TYPE}"
[[ "${TASK_TYPE}" != tbpr ]] || RECALL_OUTPUT+="/${TBPR_TYPE}"
[[ "${TASK_TYPE}" != cir ]] || RECALL_OUTPUT+="/${CIR_TYPE}"
RECALL_OUTPUT+="/${NUM_EXPENDED_TOKENS}_${MANUAL}_${SPARSE_LENGTH}_${SPARSE_VALUE_TYPE}_no_cluster_${REPS_LOC}_${EOL_TYPE}_${SPARSE_LOWER_OR_UPPER}"
# The standalone Python result paths retain their existing model-path slice
# and sparse-length labels, independently of the corpus directory contract.
RESULT_MEAN=no_mean
[[ "${SPARSE_VALUE_MEAN}" == false ]] || RESULT_MEAN=mean
RECALL_OUTPUT+="_${RESULT_MEAN}_${SPARSE_TYPE}"
RECALL_OUTPUT+="${RESULT_MODE_SUFFIX}"
printf 'Recall output: %s\n' "${RECALL_OUTPUT}"
# Rerank output is a sibling experiment directory; mode precedes fixed-alpha subdirs.
RERANK_OUTPUT="${RECALL_OUTPUT%${RESULT_MODE_SUFFIX}}_rerank_${RERANK_TYPE}_${RERANK_NUM}_${RERANK_TEMPLATE}${RESULT_MODE_SUFFIX}"
[[ "${USE_FIXED_ALPHA}" == false ]] || RERANK_OUTPUT+="/use_fixed_alpha"
printf 'Rerank output: %s\n' "${RERANK_OUTPUT}"
if [[ "${USE_FIXED_ALPHA}" == false ]]; then printf 'Validation dense: %s\nValidation sparse: %s\n' "${VAL_PASSAGE_REPS}" "${VAL_SPARSE_INDEX}"; fi

COMMAND=("${DEEPSPEED_BIN}" --num_gpus "${GPUS}" --num_nodes "${NUM_NODES}" src/rerank.py
    --llava_image_mode "${LLAVA_IMAGE_MODE}" --model_name_or_path "${MODEL_PATH}"
    --task_type "${TASK_TYPE}" --dataset_name "${DATASET}" --dataset_split "${DATASET_SPLIT}"
    --encode_type "${TARGET_TYPE}" --query_type "${QUERY_TYPE}"
    --passage_reps "${PASSAGE_REPS}" --sparse_index "${SPARSE_INDEX}" --output_dir "${OUTPUT_DIR}"
    --per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" --threads "${THREADS}" --depth "${DEPTH}"
    --retrieval_batch_size "${RETRIEVAL_BATCH_SIZE}" --alpha "${ALPHA}" --beta "${BETA}"
    --reps_loc "${REPS_LOC}" --eol_type "${EOL_TYPE}" --calculate_type "${CALCULATE_TYPE}" --prompt_type "${PROMPT_TYPE}"
    --random_view_seed "${RANDOM_VIEW_SEED}" --sparse_length "${SPARSE_LENGTH}"
    --text_sparse_length "${TEXT_SPARSE_LENGTH}" --image_sparse_length "${IMAGE_SPARSE_LENGTH}"
    --num_expended_tokens "${NUM_EXPENDED_TOKENS}" --sparse_value_type "${SPARSE_VALUE_TYPE}"
    --sparse_lower_or_upper "${SPARSE_LOWER_OR_UPPER}" --sparse_type "${SPARSE_TYPE}"
    --tbpr_type "${TBPR_TYPE}" --cir_type "${CIR_TYPE}"
    --rerank_num "${RERANK_NUM}" --rerank_batch_size "${RERANK_BATCH_SIZE}"
    --rerank_type "${RERANK_TYPE}" --rerank_template "${RERANK_TEMPLATE}" --modify_type "${MODIFY_TYPE}"
    "${PRECISION_ARGS[@]}" "${OPTIONAL_ARGS[@]}" "${FUSION_ARGS[@]}")
"${COMMAND[@]}"
