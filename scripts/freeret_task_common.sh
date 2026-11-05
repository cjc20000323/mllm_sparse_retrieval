#!/usr/bin/env bash
# Shared validation/path rules only; each task launcher contains its own command.
source "${SCRIPT_DIR}/freeret_dense_mode.sh" || exit $?

freeret_die() { printf 'Error: %s\n' "$*" >&2; exit 2; }

freeret_bool() {
    local name="$1"
    case "${!name,,}" in
        true|1|yes) printf -v "${name}" '%s' true ;;
        false|0|no) printf -v "${name}" '%s' false ;;
        *) freeret_die "${name} must be true or false." ;;
    esac
}

freeret_help() {
    cat <<USAGE
Usage: [environment] bash scripts/${BASH_SOURCE[1]##*/} [-h|--help]
One model and one dataset per run. Edit the full configuration block or set environment variables.
MODEL_FAMILY=llava_mistral|llava_llama|qwen2|qwen25|qwen3; MODEL_PATH overrides its checkpoint.
IR: DATASET=flickr|coco, QUERY_TYPE=image|text; candidates are the opposite modality.
TBPR: DATASET=CUHK-PEDES|ICFG-PEDES|RSTPReid, QUERY_TYPE=text, image candidates.
fused: DATASET=webqa|remuq|llava; WebQA text->image+text; ReMuQ/LLaVA image+text->text.
USE_FREERET_DENSE=false and USE_FREERET_PROMPT_CONSTRAINTS=false by default.
USE_FREERET_FULL_PROMPT=false by default; true requires FreeRet dense and replaces Summary with complete FreeRet prompts.
Constraints require FreeRet dense. Use the same switches and encoding settings in all three steps.
DRY_RUN=true prints paths and the command without requiring artifacts or loading models.
Defaults: prompt_5, disassembleeol_separate+concat, sparse_length=40, non-manual,
after_pad, float16, batch size 1, one GPU. TBPR_TYPE=origin_type|type.
DENSE_OUTPUT_ROOT/SPARSE_OUTPUT_ROOT configure encoded artifacts; OUTPUT_DIR is runtime/results output.
Index/retrieval accept explicit DENSE_CORPUS_DIR/SPARSE_CORPUS_DIR. Existing indexes are not overwritten.
Index paths: LLaVA <experiment>/index; Qwen <experiment>/collection/index.
Encoding reads the dataset paths configured in src/arguments.py (default /root/autodl-tmp/data).
Retrieval DATA_ROOT must point to that same dataset tree. These launchers use original data and fixed views.
Only -h/--help is accepted; configure options using variables, not trailing Python arguments.
USAGE
}

freeret_configure() {
    local checkpoint candidate filter=no_filter mean=no_mean leaf relative name sparse_leaf
    for name in MODEL_FAMILIES DATASETS ENCODE_TYPES; do
        [[ -z "${!name:-}" ]] || freeret_die "${name} is a batch setting; these launchers run a single task."
    done
    for name in DRY_RUN SPARSE_VALUE_MEAN IS_FILTERED; do freeret_bool "${name}"; done
    case "${MODEL_FAMILY}" in
        llava_mistral) checkpoint=llava-hf-llava-v1.6-mistral-7b-hf ;;
        llava_llama) checkpoint=llava-hf-llama3-llava-next-8b-hf ;;
        qwen2) checkpoint=Qwen-Qwen2-VL-7B-Instruct ;;
        qwen25) checkpoint=Qwen-Qwen2.5-VL-7B-Instruct ;;
        qwen3) checkpoint=Qwen-Qwen3-VL-8B-Instruct ;;
        *) freeret_die 'Unsupported MODEL_FAMILY.' ;;
    esac
    MODEL_PATH="${MODEL_PATH:-${CHECKPOINT_ROOT%/}/${checkpoint}}"
    while [[ "${MODEL_PATH}" == */ && "${MODEL_PATH}" != / ]]; do MODEL_PATH="${MODEL_PATH%/}"; done
    MODEL_TAG="${MODEL_PATH##*/}"
    [[ -n "${MODEL_TAG}" && "${MODEL_TAG}" != . && "${MODEL_TAG}" != .. ]] || freeret_die 'Invalid MODEL_PATH.'
    case "${TASK_GROUP}:${DATASET}" in
        ir:flickr|ir:coco)
            TASK_TYPE=ir; QUERY_TYPE="${QUERY_TYPE:-image}"
            case "${QUERY_TYPE}" in
                image) candidate=text; DIRECTION_LABEL=image_to_text ;;
                text) candidate=image; DIRECTION_LABEL=text_to_image ;;
                *) freeret_die 'IR requires QUERY_TYPE=image or text.' ;;
            esac ;;
        tbpr:CUHK-PEDES|tbpr:ICFG-PEDES|tbpr:RSTPReid)
            TASK_TYPE=tbpr; QUERY_TYPE="${QUERY_TYPE:-text}"; candidate=image; DIRECTION_LABEL=text_to_image
            [[ "${QUERY_TYPE}" == text ]] || freeret_die 'TBPR requires QUERY_TYPE=text.' ;;
        fused:webqa)
            TASK_TYPE=t2it; QUERY_TYPE="${QUERY_TYPE:-text}"; candidate=corpus; DIRECTION_LABEL=text_to_imagetext
            [[ "${QUERY_TYPE}" == text ]] || freeret_die 'WebQA requires QUERY_TYPE=text.' ;;
        fused:remuq|fused:llava)
            TASK_TYPE=it2t; QUERY_TYPE="${QUERY_TYPE:-image}"; candidate=corpus; DIRECTION_LABEL=imagetext_to_text
            [[ "${QUERY_TYPE}" == image || "${QUERY_TYPE}" == composed ]] || freeret_die 'ReMuQ/LLaVA requires QUERY_TYPE=image or composed (both include text).' ;;
        *) freeret_die "Dataset ${DATASET} does not belong to task group ${TASK_GROUP}." ;;
    esac
    [[ -z "${ENCODE_TYPE}" || "${ENCODE_TYPE}" == "${candidate}" ]] || freeret_die "This direction requires ENCODE_TYPE=${candidate}."
    ENCODE_TYPE="${candidate}"
    [[ "${TBPR_TYPE}" == origin_type || "${TBPR_TYPE}" == type ]] || freeret_die 'Invalid TBPR_TYPE.'
    [[ "${PROMPT_TYPE}" == prompt_5 ]] || freeret_die 'These paired launchers use PROMPT_TYPE=prompt_5.'
    [[ "${EOL_TYPE}" == disassembleeol_separate && "${CALCULATE_TYPE}" == concat && "${SPARSE_MANUAL,,}" == false ]] ||
        freeret_die 'Three-stage pairing requires disassembleeol_separate+concat and SPARSE_MANUAL=false.'
    [[ "${DATASET_SPLIT}" =~ ^[a-zA-Z0-9_-]+$ ]] || freeret_die 'Invalid DATASET_SPLIT.'
    [[ "${REPS_LOC}" == after_pad || "${REPS_LOC}" == before_pad ]] || freeret_die 'Invalid REPS_LOC.'
    case "${SPARSE_VALUE_TYPE}" in sum|max|replace) ;; *) freeret_die 'Invalid SPARSE_VALUE_TYPE.' ;; esac
    case "${SPARSE_LOWER_OR_UPPER}" in lower|upper) ;; *) freeret_die 'Invalid SPARSE_LOWER_OR_UPPER.' ;; esac
    case "${LLAVA_IMAGE_MODE}" in anyres|base) ;; *) freeret_die 'Invalid LLAVA_IMAGE_MODE.' ;; esac
    [[ "${LLAVA_IMAGE_MODE}" != base || "${MODEL_FAMILY}" == llava_* ]] || freeret_die 'LLAVA_IMAGE_MODE=base requires LLaVA-NeXT.'
    case "${DTYPE}" in float16|bfloat16|float32) ;; *) freeret_die 'Invalid DTYPE.' ;; esac
    for name in GPUS NUM_NODES PER_DEVICE_BATCH_SIZE SPARSE_LENGTH TEXT_SPARSE_LENGTH IMAGE_SPARSE_LENGTH; do
        [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || freeret_die "${name} must be a positive integer."
    done
    for name in QWEN_MIN_PIXELS QWEN_MAX_PIXELS NUM_EXPENDED_TOKENS; do
        [[ "${!name}" =~ ^(0|[1-9][0-9]*)$ ]] || freeret_die "${name} must be nonnegative."
    done
    [[ "${RANDOM_VIEW_SEED}" =~ ^-?[0-9]+$ ]] || freeret_die 'RANDOM_VIEW_SEED must be an integer.'
    if ((QWEN_MIN_PIXELS > 0 && QWEN_MAX_PIXELS > 0 && QWEN_MIN_PIXELS > QWEN_MAX_PIXELS)); then
        freeret_die 'QWEN_MIN_PIXELS exceeds QWEN_MAX_PIXELS.'
    fi
    [[ "${IS_FILTERED}" == false ]] || filter=filter
    [[ "${SPARSE_VALUE_MEAN}" == false ]] || mean=mean
    relative="${MODEL_TAG}/${DATASET}/${ENCODE_TYPE}/${filter}/${CALCULATE_TYPE}/${PROMPT_TYPE}/${DATASET_SPLIT}"
    [[ "${TASK_TYPE}" != tbpr ]] || relative+="/${TBPR_TYPE}"
    leaf="${NUM_EXPENDED_TOKENS}_no_manual_${SPARSE_LENGTH}_${SPARSE_VALUE_TYPE}_no_cluster_${REPS_LOC}_${EOL_TYPE}_${SPARSE_LOWER_OR_UPPER}_${mean}"
    if [[ "${LLAVA_IMAGE_MODE}" == base && ( "${ENCODE_TYPE}" == image || "${TASK_TYPE}" == t2it ) ]]; then leaf+="_llava_base"; fi
    DENSE_CORPUS_DIR="$(dense_mode_path "${DENSE_CORPUS_DIR:-${DENSE_OUTPUT_ROOT%/}/${relative}/${leaf}}")"
    SPARSE_CORPUS_DIR="$(dense_mode_path "${SPARSE_CORPUS_DIR:-${SPARSE_OUTPUT_ROOT%/}/${relative}/${leaf}}")"
    # Explicit sparse aliases are accepted by the Python index/check helper.
    sparse_leaf="${SPARSE_CORPUS_DIR%/}"
    while [[ "${sparse_leaf##*/}" == collection || "${sparse_leaf##*/}" == index ]]; do sparse_leaf="$(dirname -- "${sparse_leaf}")"; done
    SPARSE_COLLECTION_DIR="${sparse_leaf}"
    [[ "${MODEL_FAMILY}" != qwen* ]] || SPARSE_COLLECTION_DIR+="/collection"
    DEFAULT_SPARSE_INDEX="${SPARSE_COLLECTION_DIR%/}/index"
    [[ "${SPARSE_CORPUS_DIR##*/}" != index ]] || DEFAULT_SPARSE_INDEX="${SPARSE_CORPUS_DIR}"
    FREERET_ARGS=(--use_freeret_dense "${USE_FREERET_DENSE}" --use_freeret_prompt_constraints "${USE_FREERET_PROMPT_CONSTRAINTS}" --use_freeret_full_prompt "${USE_FREERET_FULL_PROMPT}")
    PRECISION_ARGS=()
    case "${DTYPE}" in float16) PRECISION_ARGS+=(--fp16) ;; bfloat16) PRECISION_ARGS+=(--bf16) ;; esac
}

freeret_report() {
    printf 'Task: %s / %s / %s (%s, %s)\n' "${TASK_GROUP}" "${DATASET}" "${DIRECTION_LABEL}" "${DATASET_SPLIT}" "${MODEL_FAMILY}"
    printf 'Model: %s\nFreeRet dense=%s constraints=%s\n' "${MODEL_PATH}" "${USE_FREERET_DENSE}" "${USE_FREERET_PROMPT_CONSTRAINTS}"
    if [[ "${USE_FREERET_FULL_PROMPT}" == true ]]; then
        printf 'FreeRet prompt mode: complete (replaces Summary and bypasses the old constraints flag)\n'
    fi
    printf 'Dense corpus: %s\nSparse corpus: %s\nLucene index: %s\n' "${DENSE_CORPUS_DIR}" "${SPARSE_CORPUS_DIR}" "${DEFAULT_SPARSE_INDEX}"
    printf 'Command:'; printf ' %q' "${COMMAND[@]}"; printf '\n'
}

freeret_require_checkpoint() {
    [[ -f "$1/config.json" ]] || freeret_die "Checkpoint config.json not found: $1"
}

freeret_require_command() { command -v "$1" >/dev/null 2>&1 || freeret_die "Executable not found: $1"; }

freeret_require_data() {
    local directory file
    local -a files
    case "${DATASET}" in
        flickr|coco) directory="${DATASET}"; files=("${DATASET}_${DATASET_SPLIT}.csv") ;;
        CUHK-PEDES) directory=CUHK-PEDES; files=(reid_raw.json) ;;
        ICFG-PEDES) directory=ICFG-PEDES; files=(ICFG-PEDES.json) ;;
        RSTPReid) directory=RSTPReid; files=(data_captions.json) ;;
        webqa) directory=WebQA; files=(query-00000-of-00001.parquet qrels-00000-of-00001.parquet corpus-00000-of-00002.parquet corpus-00001-of-00002.parquet) ;;
        remuq) directory=UMRB-ReMuQ; files=(query.parquet qrels.parquet corpus.parquet) ;;
        llava) directory=UMRB-LLaVA; files=(query.parquet qrels.parquet corpus.parquet) ;;
    esac
    for file in "${files[@]}"; do
        [[ -f "${DATA_ROOT%/}/data/${directory}/${file}" ]] || freeret_die "Dataset annotation missing: ${DATA_ROOT%/}/data/${directory}/${file}"
    done
}
