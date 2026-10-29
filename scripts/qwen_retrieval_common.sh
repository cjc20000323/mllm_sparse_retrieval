#!/usr/bin/env bash
# Shared configuration for sparse_index_qwen.sh and rerank_qwen.sh.
# encode_qwen.sh and search_qwen.sh are standalone; keep their settings aligned.
# Existing Python entrypoints own prompts and model behavior.
#
# Export the same settings before running each stage, for example:
#   export QWEN_FAMILY=qwen25 DATASET=flickr DATASETS=flickr IR_QUERY_TYPE=image
#   export TASK_TYPE=ir ENCODE_TYPE=text
#   export EOL_TYPE=disassembleeol_separate
#   bash scripts/encode_qwen.sh
#   bash scripts/sparse_index_qwen.sh
#   USE_FIXED_ALPHA=true ALPHA=0.5 bash scripts/search_qwen.sh
#   bash scripts/rerank_qwen.sh
# Set EOL_TYPE=prompteol to select prompteol+separate instead. Changing the
# method/prompt/sparse/pixel settings requires matching newly encoded indexes.
# Dataset locations follow src/arguments.py; these entrypoints have no DATA_ROOT
# option. Rerank performs retrieval again, using the same corpus as search.

qwen_die() {
    printf 'Error: %s\n' "$*" >&2
    exit 2
}

qwen_usage() {
    printf 'Usage: bash scripts/%s_qwen.sh [-h|--help]\n' "$1"
    cat <<'USAGE'
Run the encode -> sparse_index -> search -> rerank scripts separately.
Configure environment variables or this file's configuration block for indexing/rerank.
The standalone encode_qwen.sh and search_qwen.sh use one explicit DATASET each.
search_qwen.sh selects on target unless val paths are explicit; this file defaults to fixed alpha.
Keep all configurations consistent; set DATASETS=flickr here for a Flickr-only corpus.
  QWEN_FAMILY          qwen2 | qwen25 | qwen3 (default: qwen25)
  QWEN_CHECKPOINT_ROOT checkpoint parent (default: /root/autodl-fs/checkpoints)
  QWEN_MODEL_PATH      custom local checkpoint; overrides the family default
  DATASETS             whitespace-separated names (default: "flickr coco")
  DATASET_SPLIT        default: test; use val to prepare validation indexes
  IR_QUERY_TYPE        image | text for COCO/Flickr (default: image)
  EOL_TYPE             disassembleeol_separate | prompteol (default: first)
  CALCULATE_TYPE       inferred as concat | separate; must match EOL_TYPE
  DENSE_OUTPUT_ROOT / SPARSE_OUTPUT_ROOT   default: dense_output / sparse_output
  QWEN_OUTPUT_ROOT     result/log root (default: qwen_retrieval_results)
  GPUS / NUM_NODES     default: 1 / 1; DEEPSPEED_BIN defaults to deepspeed
  PYTHON_BIN           interpreter for sparse_index (default: python)
  DTYPE                float16 | bfloat16 | float32 (default: float16)
  PER_DEVICE_BATCH_SIZE / RETRIEVAL_BATCH_SIZE   default: 1 / 1
  QWEN_MIN_PIXELS / QWEN_MAX_PIXELS   default: 0 / 0 (checkpoint defaults)
  PROMPT_TYPE / REPS_LOC   default: prompt_5 / after_pad
  SPARSE_LENGTH / SPARSE_VALUE_TYPE / SPARSE_LOWER_OR_UPPER   default: 40 / sum / lower
  TEXT_SPARSE_LENGTH / IMAGE_SPARSE_LENGTH   default: SPARSE_LENGTH
  SPARSE_MANUAL / SPARSE_VALUE_MEAN / IS_FILTERED   default: false
  NUM_EXPENDED_TOKENS   default: 0; TBPR_TYPE defaults to origin_type
  DEPTH / THREADS / USE_GPU   default: 1000 / 16 / false (USE_GPU controls FAISS)
  USE_FIXED_ALPHA / ALPHA   default: true / 0.5; false requires same-dataset val indexes
  RERANK_TEMPLATE / RERANK_NUM / RERANK_BATCH_SIZE   default: what_caption_generation / 20 / 1
  MODIFY_TYPE / USE_RERANK_PRIOR_CACHE   default: no / false
  RERANK_PRIOR_CACHE_DIR   default: cache/rerank_prior
Supported datasets: coco flickr CUHK-PEDES ICFG-PEDES RSTPReid webqa remuq llava.
Model defaults: Qwen2-VL-7B-Instruct, Qwen2.5-VL-7B-Instruct, Qwen3-VL-8B-Instruct.
Dataset paths follow src/arguments.py (default: /root/autodl-tmp/data).
Encoding does not build Lucene; sparse_index stages only corpus_*.jsonl and writes collection/index/.
Encoding replaces matching corpus artifacts and backs up old/nested indexes outside collection.
This index launcher does not overwrite an existing index. The first failing job stops the batch.
Only -h/--help is accepted; arbitrary CLI arguments are not forwarded.
USAGE
}

qwen_bool() {
    case "${!1,,}" in
        true|1|yes) printf -v "$1" '%s' true ;;
        false|0|no) printf -v "$1" '%s' false ;;
        *) qwen_die "$1 must be true/false or 1/0." ;;
    esac
}

qwen_configure() {
    QWEN_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
    cd "${QWEN_REPO_ROOT}"

    # ---------------- Configuration: edit here or export environment variables. ----------------
    QWEN_FAMILY="${QWEN_FAMILY:-qwen25}"
    QWEN_CHECKPOINT_ROOT="${QWEN_CHECKPOINT_ROOT:-/root/autodl-fs/checkpoints}"
    DATASETS="${DATASETS-flickr coco}"
    DATASET_SPLIT="${DATASET_SPLIT:-test}"
    IR_QUERY_TYPE="${IR_QUERY_TYPE:-image}"
    EOL_TYPE="${EOL_TYPE:-disassembleeol_separate}"
    DENSE_OUTPUT_ROOT="${DENSE_OUTPUT_ROOT:-${QWEN_REPO_ROOT}/dense_output}"
    SPARSE_OUTPUT_ROOT="${SPARSE_OUTPUT_ROOT:-${QWEN_REPO_ROOT}/sparse_output}"
    QWEN_OUTPUT_ROOT="${QWEN_OUTPUT_ROOT:-${QWEN_REPO_ROOT}/qwen_retrieval_results}"
    GPUS="${GPUS:-1}"
    NUM_NODES="${NUM_NODES:-1}"
    DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"
    PYTHON_BIN="${PYTHON_BIN:-python}"
    DTYPE="${DTYPE:-float16}"
    PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
    RETRIEVAL_BATCH_SIZE="${RETRIEVAL_BATCH_SIZE:-1}"
    QWEN_MIN_PIXELS="${QWEN_MIN_PIXELS:-0}"
    QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS:-0}"
    PROMPT_TYPE="${PROMPT_TYPE:-prompt_5}"
    REPS_LOC="${REPS_LOC:-after_pad}"
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
    DEPTH="${DEPTH:-1000}"
    THREADS="${THREADS:-16}"
    USE_GPU="${USE_GPU:-false}"
    USE_FIXED_ALPHA="${USE_FIXED_ALPHA:-true}"
    ALPHA="${ALPHA:-0.5}"
    RERANK_TEMPLATE="${RERANK_TEMPLATE:-what_caption_generation}"
    RERANK_NUM="${RERANK_NUM:-20}"
    RERANK_BATCH_SIZE="${RERANK_BATCH_SIZE:-1}"
    MODIFY_TYPE="${MODIFY_TYPE:-no}"
    USE_RERANK_PRIOR_CACHE="${USE_RERANK_PRIOR_CACHE:-false}"
    RERANK_PRIOR_CACHE_DIR="${RERANK_PRIOR_CACHE_DIR:-${QWEN_REPO_ROOT}/cache/rerank_prior}"
    # ------------------------------------------------------------------------------------------

    local checkpoint expected_calculate name
    case "${QWEN_FAMILY}" in
        qwen2) checkpoint=Qwen2-VL-7B-Instruct ;;
        qwen25) checkpoint=Qwen2.5-VL-7B-Instruct ;;
        qwen3) checkpoint=Qwen3-VL-8B-Instruct ;;
        *) qwen_die "QWEN_FAMILY must be qwen2, qwen25 or qwen3." ;;
    esac
    QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${QWEN_CHECKPOINT_ROOT}/${checkpoint}}"
    while [[ "${QWEN_MODEL_PATH}" == */ && "${QWEN_MODEL_PATH}" != / ]]; do
        QWEN_MODEL_PATH="${QWEN_MODEL_PATH%/}"
    done
    QWEN_MODEL_TAG="${QWEN_MODEL_PATH##*/}"
    [[ -n "${QWEN_MODEL_TAG}" && "${QWEN_MODEL_TAG}" != . && "${QWEN_MODEL_TAG}" != .. ]] || qwen_die "Invalid QWEN_MODEL_PATH."
    case "${EOL_TYPE}" in
        prompteol) expected_calculate=separate ;;
        disassembleeol_separate) expected_calculate=concat ;;
        *) qwen_die "EOL_TYPE must be prompteol or disassembleeol_separate." ;;
    esac
    CALCULATE_TYPE="${CALCULATE_TYPE:-${expected_calculate}}"
    [[ "${CALCULATE_TYPE}" == "${expected_calculate}" ]] || qwen_die "${EOL_TYPE} requires CALCULATE_TYPE=${expected_calculate}."
    [[ "${IR_QUERY_TYPE}" == image || "${IR_QUERY_TYPE}" == text ]] || qwen_die "IR_QUERY_TYPE must be image or text."
    [[ "${DATASET_SPLIT}" =~ ^[a-zA-Z0-9_-]+$ ]] || qwen_die "Invalid DATASET_SPLIT."
    [[ "${PROMPT_TYPE}" =~ ^[a-zA-Z0-9_]+$ ]] || qwen_die "Invalid PROMPT_TYPE."
    [[ "${REPS_LOC}" == after_pad || "${REPS_LOC}" == before_pad ]] || qwen_die "REPS_LOC must be after_pad or before_pad."
    [[ "${TBPR_TYPE}" == origin_type || "${TBPR_TYPE}" == type ]] || qwen_die "TBPR_TYPE must be origin_type or type."
    [[ "${SPARSE_VALUE_TYPE}" == sum || "${SPARSE_VALUE_TYPE}" == max || "${SPARSE_VALUE_TYPE}" == replace ]] || qwen_die "Invalid SPARSE_VALUE_TYPE."
    [[ "${SPARSE_LOWER_OR_UPPER}" == lower || "${SPARSE_LOWER_OR_UPPER}" == upper ]] || qwen_die "Invalid SPARSE_LOWER_OR_UPPER."
    [[ "${ALPHA}" =~ ^(0|0?\.[0-9]+|1(\.0+)?)$ ]] || qwen_die "ALPHA must be in [0,1]."
    [[ "${RERANK_TEMPLATE}" =~ ^[a-zA-Z0-9_]+$ ]] || qwen_die "Invalid RERANK_TEMPLATE."
    [[ "${MODIFY_TYPE}" == no || "${MODIFY_TYPE}" == modify_single ]] || qwen_die "MODIFY_TYPE must be no or modify_single."
    for name in GPUS NUM_NODES PER_DEVICE_BATCH_SIZE RETRIEVAL_BATCH_SIZE SPARSE_LENGTH TEXT_SPARSE_LENGTH IMAGE_SPARSE_LENGTH DEPTH THREADS RERANK_NUM RERANK_BATCH_SIZE; do
        [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || qwen_die "${name} must be a positive integer."
    done
    for name in NUM_EXPENDED_TOKENS QWEN_MIN_PIXELS QWEN_MAX_PIXELS; do
        [[ "${!name}" =~ ^(0|[1-9][0-9]*)$ ]] || qwen_die "${name} must be a nonnegative integer."
    done
    if ((QWEN_MIN_PIXELS > 0 && QWEN_MAX_PIXELS > 0 && QWEN_MIN_PIXELS > QWEN_MAX_PIXELS)); then
        qwen_die "QWEN_MIN_PIXELS must not exceed QWEN_MAX_PIXELS."
    fi
    for name in SPARSE_MANUAL SPARSE_VALUE_MEAN IS_FILTERED USE_GPU USE_FIXED_ALPHA USE_RERANK_PRIOR_CACHE; do
        qwen_bool "${name}"
    done
    QWEN_PRECISION_ARGS=()
    case "${DTYPE}" in
        float16) QWEN_PRECISION_ARGS+=(--fp16) ;;
        bfloat16) QWEN_PRECISION_ARGS+=(--bf16) ;;
        float32) ;;
        *) qwen_die "DTYPE must be float16, bfloat16 or float32." ;;
    esac

    read -r -a QWEN_DATASET_LIST <<< "${DATASETS//$'\n'/ }"
    ((${#QWEN_DATASET_LIST[@]})) || qwen_die "DATASETS must contain at least one dataset."
    local -A seen=()
    for name in "${QWEN_DATASET_LIST[@]}"; do
        case "${name}" in
            coco|flickr|CUHK-PEDES|ICFG-PEDES|RSTPReid|webqa|remuq|llava) ;;
            *) qwen_die "Unsupported dataset: ${name}" ;;
        esac
        [[ -z "${seen[${name}]+present}" ]] || qwen_die "Duplicate dataset: ${name}"
        seen["${name}"]=1
    done
    shopt -s nullglob
}

qwen_relative_path() {
    local split="$1" filter=no_filter manual=no_manual lengths="${SPARSE_LENGTH}" mean=no_mean
    [[ "${IS_FILTERED}" == false ]] || filter=filter
    [[ "${SPARSE_VALUE_MEAN}" == false ]] || mean=mean
    if [[ "${SPARSE_MANUAL}" == true ]]; then
        manual=manual
        lengths="${TEXT_SPARSE_LENGTH}_${IMAGE_SPARSE_LENGTH}"
    fi
    local relative="${QWEN_MODEL_TAG}/${QWEN_DATASET}/${QWEN_CORPUS}/${filter}/${CALCULATE_TYPE}/${PROMPT_TYPE}/${split}"
    [[ "${QWEN_TASK}" != tbpr ]] || relative+="/${TBPR_TYPE}"
    printf '%s/%s_%s_%s_%s_no_cluster_%s_%s_%s_%s' "${relative}" "${NUM_EXPENDED_TOKENS}" "${manual}" "${lengths}" "${SPARSE_VALUE_TYPE}" "${REPS_LOC}" "${EOL_TYPE}" "${SPARSE_LOWER_OR_UPPER}" "${mean}"
}

qwen_job_paths() {
    QWEN_DATASET="$1"
    case "${QWEN_DATASET}" in
        flickr|coco)
            QWEN_TASK=ir; QWEN_QUERY="${IR_QUERY_TYPE}"
            if [[ "${QWEN_QUERY}" == image ]]; then
                QWEN_CORPUS=text; QWEN_DIRECTION=image_to_text
            else
                QWEN_CORPUS=image; QWEN_DIRECTION=text_to_image
            fi
            ;;
        CUHK-PEDES|ICFG-PEDES|RSTPReid)
            QWEN_TASK=tbpr; QWEN_QUERY=text; QWEN_CORPUS=image; QWEN_DIRECTION=text_to_image ;;
        webqa) QWEN_TASK=t2it; QWEN_QUERY=text; QWEN_CORPUS=corpus; QWEN_DIRECTION=text_to_imagetext ;;
        remuq|llava) QWEN_TASK=it2t; QWEN_QUERY=image; QWEN_CORPUS=corpus; QWEN_DIRECTION=imagetext_to_text ;;
    esac
    local relative
    relative="$(qwen_relative_path "${DATASET_SPLIT}")"
    QWEN_DENSE_PATH="${DENSE_OUTPUT_ROOT}/${relative}"
    QWEN_SPARSE_PATH="${SPARSE_OUTPUT_ROOT}/${relative}"
    QWEN_RESULT_BASE="${QWEN_OUTPUT_ROOT}/${relative}/${QWEN_DIRECTION}"
    relative="$(qwen_relative_path val)"
    QWEN_VAL_DENSE_PATH="${DENSE_OUTPUT_ROOT}/${relative}"
    QWEN_VAL_SPARSE_PATH="${SPARSE_OUTPUT_ROOT}/${relative}"
}

qwen_sparse_index_path() {
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

qwen_require_corpus() {
    local dense="$1" sparse="$2" index
    index="$(qwen_sparse_index_path "${sparse}")"
    local -a shards=("${dense}"/corpus*.pkl) segments=("${index}"/segments_*)
    ((${#shards[@]})) && [[ -f "${shards[0]}" ]] || qwen_die "No dense corpus*.pkl found in ${dense}; run encode_qwen.sh with matching settings."
    ((${#segments[@]})) && [[ -f "${segments[0]}" ]] || qwen_die "No Lucene segments_* found in ${index}; run sparse_index_qwen.sh."
}

qwen_preflight_job() {
    local stage="$1"
    case "${stage}" in
        sparse_index)
            local -a documents=("${QWEN_SPARSE_PATH}"/collection/corpus*.jsonl)
            ((${#documents[@]})) && [[ -f "${documents[0]}" ]] || qwen_die "No corpus*.jsonl found in ${QWEN_SPARSE_PATH}/collection; run encode_qwen.sh."
            [[ ! -e "${QWEN_SPARSE_PATH}/collection/index" && ! -L "${QWEN_SPARSE_PATH}/collection/index" ]] || qwen_die "Index output already exists: ${QWEN_SPARSE_PATH}/collection/index; existing indexes are not overwritten."
            ;;
        search|rerank)
            qwen_require_corpus "${QWEN_DENSE_PATH}" "${QWEN_SPARSE_PATH}"
            if [[ "${USE_FIXED_ALPHA}" == false ]]; then
                qwen_require_corpus "${QWEN_VAL_DENSE_PATH}" "${QWEN_VAL_SPARSE_PATH}"
            fi
            ;;
    esac
}

qwen_command() {
    local stage="$1" alpha_label="alpha_${ALPHA}" name
    if [[ "${stage}" == sparse_index ]]; then
        QWEN_COMMAND=("${PYTHON_BIN}" -B src/qwen_sparse_index.py
            --sparse-root "${QWEN_SPARSE_PATH}" --threads "${THREADS}")
        return
    fi
    QWEN_COMMAND=("${DEEPSPEED_BIN}" --num_gpus "${GPUS}" --num_nodes "${NUM_NODES}" "src/${stage}.py"
        --model_name_or_path "${QWEN_MODEL_PATH}" --dataset_name "${QWEN_DATASET}"
        --dataset_split "${DATASET_SPLIT}" --task_type "${QWEN_TASK}" --encode_type "${QWEN_CORPUS}"
        --eol_type "${EOL_TYPE}" --calculate_type "${CALCULATE_TYPE}" --prompt_type "${PROMPT_TYPE}"
        --reps_loc "${REPS_LOC}" --tbpr_type "${TBPR_TYPE}" --per_device_batch_size "${PER_DEVICE_BATCH_SIZE}"
        --sparse_length "${SPARSE_LENGTH}" --text_sparse_length "${TEXT_SPARSE_LENGTH}"
        --image_sparse_length "${IMAGE_SPARSE_LENGTH}" --sparse_value_type "${SPARSE_VALUE_TYPE}"
        --sparse_lower_or_upper "${SPARSE_LOWER_OR_UPPER}" --num_expended_tokens "${NUM_EXPENDED_TOKENS}"
        --qwen_min_pixels "${QWEN_MIN_PIXELS}" --qwen_max_pixels "${QWEN_MAX_PIXELS}"
        "${QWEN_PRECISION_ARGS[@]}")
    for name in sparse_manual sparse_value_mean is_filtered; do
        local setting="${name^^}"
        [[ "${!setting}" == false ]] || QWEN_COMMAND+=("--${name}")
    done
    if [[ "${stage}" == encode ]]; then
        QWEN_COMMAND+=(--dense_output_dir "${DENSE_OUTPUT_ROOT}" --sparse_output_dir "${SPARSE_OUTPUT_ROOT}"
            --output_dir "${QWEN_RESULT_BASE}/encode")
        return
    fi
    QWEN_COMMAND+=(--query_type "${QWEN_QUERY}" --passage_reps "${QWEN_DENSE_PATH}"
        --sparse_index "${QWEN_SPARSE_PATH}" --depth "${DEPTH}" --threads "${THREADS}"
        --retrieval_batch_size "${RETRIEVAL_BATCH_SIZE}" --alpha "${ALPHA}")
    [[ "${USE_GPU}" == false ]] || QWEN_COMMAND+=(--use_gpu)
    if [[ "${USE_FIXED_ALPHA}" == true ]]; then
        QWEN_COMMAND+=(--use_fixed_alpha)
    else
        alpha_label=alpha_val
        QWEN_COMMAND+=(--val_passage_reps "${QWEN_VAL_DENSE_PATH}" --val_sparse_index "${QWEN_VAL_SPARSE_PATH}")
    fi
    if [[ "${stage}" == rerank ]]; then
        QWEN_COMMAND+=(--rerank_type pointwise --rerank_template "${RERANK_TEMPLATE}" --rerank_num "${RERANK_NUM}"
            --rerank_batch_size "${RERANK_BATCH_SIZE}" --modify_type "${MODIFY_TYPE}"
            --output_dir "${QWEN_RESULT_BASE}/rerank/${RERANK_TEMPLATE}/top_${RERANK_NUM}/${MODIFY_TYPE}/${alpha_label}")
        if [[ "${USE_RERANK_PRIOR_CACHE}" == true ]]; then
            QWEN_COMMAND+=(--use_rerank_prior_cache --rerank_prior_cache_dir "${RERANK_PRIOR_CACHE_DIR}")
        fi
    else
        QWEN_COMMAND+=(--output_dir "${QWEN_RESULT_BASE}/search/${alpha_label}")
    fi
}

qwen_main() {
    local stage="$1"
    shift
    if (($#)); then
        if (($# == 1)) && [[ "$1" == -h || "$1" == --help ]]; then
            qwen_usage "${stage}"
            return
        fi
        qwen_die "Configure environment variables or qwen_retrieval_common.sh; arbitrary CLI arguments are not forwarded. See --help."
    fi
    qwen_configure
    if [[ "${stage}" == sparse_index ]]; then
        command -v "${PYTHON_BIN}" >/dev/null 2>&1 || qwen_die "Python executable not found: ${PYTHON_BIN}"
        [[ -f src/qwen_sparse_index.py ]] || qwen_die "Python entrypoint does not exist: src/qwen_sparse_index.py"
    else
        command -v "${DEEPSPEED_BIN}" >/dev/null 2>&1 || qwen_die "DeepSpeed executable not found: ${DEEPSPEED_BIN}"
        [[ -d "${QWEN_MODEL_PATH}" ]] || qwen_die "Checkpoint directory does not exist: ${QWEN_MODEL_PATH}"
        [[ -f "${QWEN_MODEL_PATH}/config.json" ]] || qwen_die "Checkpoint config.json does not exist: ${QWEN_MODEL_PATH}"
        [[ -f "src/${stage}.py" ]] || qwen_die "Python entrypoint does not exist: src/${stage}.py"
    fi
    local dataset status
    # Validate every job before launching any of them.
    for dataset in "${QWEN_DATASET_LIST[@]}"; do
        qwen_job_paths "${dataset}"
        qwen_preflight_job "${stage}"
    done
    printf 'Qwen %s: family=%s model=%s method=%s+%s datasets=%s\n' "${stage}" "${QWEN_FAMILY}" "${QWEN_MODEL_PATH}" "${EOL_TYPE}" "${CALCULATE_TYPE}" "${QWEN_DATASET_LIST[*]}"
    for dataset in "${QWEN_DATASET_LIST[@]}"; do
        qwen_job_paths "${dataset}"
        qwen_command "${stage}"
        printf '\n========== %s: %s (%s) ==========\n' "${stage}" "${dataset}" "${QWEN_DIRECTION}"
        printf 'Command:'; printf ' %q' "${QWEN_COMMAND[@]}"; printf '\n'
        if "${QWEN_COMMAND[@]}"; then
            printf 'Completed: %s %s\n' "${stage}" "${dataset}"
        else
            status=$?
            printf 'Failed: %s %s (exit %s); stopping Qwen batch.\n' "${stage}" "${dataset}" "${status}" >&2
            return "${status}"
        fi
    done
    printf '\nCompleted all %s %s jobs.\n' "${#QWEN_DATASET_LIST[@]}" "${stage}"
}
