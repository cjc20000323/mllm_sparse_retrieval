#!/usr/bin/env bash
# Shared dense/prompt namespaces. Sparse settings and fusion weights are unchanged.
USE_FREERET_DENSE="${USE_FREERET_DENSE-false}"
case "${USE_FREERET_DENSE,,}" in
    true|TRUE|1|yes) USE_FREERET_DENSE=true ;;
    false|FALSE|0|no) USE_FREERET_DENSE=false ;;
    *) printf 'Error: USE_FREERET_DENSE must be true or false.\n' >&2; return 2 ;;
esac
USE_FREERET_FULL_PROMPT="${USE_FREERET_FULL_PROMPT-false}"  # Complete FreeRet dense/sparse prompts; requires FreeRet dense
case "${USE_FREERET_FULL_PROMPT,,}" in
    true|1|yes) USE_FREERET_FULL_PROMPT=true ;;
    false|0|no) USE_FREERET_FULL_PROMPT=false ;;
    *) printf 'Error: USE_FREERET_FULL_PROMPT must be true or false.\n' >&2; return 2 ;;
esac
if [[ "${USE_FREERET_FULL_PROMPT}" == true && "${USE_FREERET_DENSE}" != true ]]; then
    printf 'Error: USE_FREERET_FULL_PROMPT=true requires USE_FREERET_DENSE=true.\n' >&2
    return 2
fi
USE_FREERET_PROMPT_CONSTRAINTS="${USE_FREERET_PROMPT_CONSTRAINTS-false}"
case "${USE_FREERET_PROMPT_CONSTRAINTS,,}" in
    true|1|yes) USE_FREERET_PROMPT_CONSTRAINTS=true ;;
    false|0|no) USE_FREERET_PROMPT_CONSTRAINTS=false ;;
    *) printf 'Error: USE_FREERET_PROMPT_CONSTRAINTS must be true or false.\n' >&2; return 2 ;;
esac
if [[ "${USE_FREERET_PROMPT_CONSTRAINTS}" == true && "${USE_FREERET_DENSE}" != true ]]; then
    printf 'Error: USE_FREERET_PROMPT_CONSTRAINTS=true requires USE_FREERET_DENSE=true.\n' >&2
    return 2
fi

dense_mode_path() {
    local value="$1" tail=""
    if [[ "${USE_FREERET_DENSE}" != true || -z "${value}" ]]; then
        printf '%s' "${value}"
        return
    fi
    while [[ "${value}" == */ && "${value}" != / ]]; do value="${value%/}"; done
    case "${value##*/}" in
        *.pkl|*.pickle) tail="/${value##*/}"; value="$(dirname -- "${value}")" ;;
    esac
    while [[ "${value##*/}" == collection || "${value##*/}" == index ]]; do
        tail="/${value##*/}${tail}"
        value="$(dirname -- "${value}")"
    done
    # Preserve explicitly supplied old paths so Python reports the version mismatch.
    if [[ "${value}" != *_dense_freeret_pre_mlp_freeret_constraints_v1 && "${value}" != *_dense_freeret_pre_mlp_freeret_constraints_v2 && "${value}" != *_dense_freeret_pre_mlp_freeret_full_prompt_v1 ]]; then
        [[ "${value}" == *_dense_freeret_pre_mlp ]] || value+="_dense_freeret_pre_mlp"
        if [[ "${USE_FREERET_FULL_PROMPT}" == true ]]; then value+="_freeret_full_prompt_v1"
        elif [[ "${USE_FREERET_PROMPT_CONSTRAINTS}" == true ]]; then value+="_freeret_constraints_v2"; fi
    fi
    printf '%s%s' "${value}" "${tail}"
}

dense_mode_paths() {
    [[ "${USE_FREERET_DENSE}" == true ]] || return 0
    local name
    for name in PASSAGE_REPS SPARSE_INDEX VAL_PASSAGE_REPS VAL_SPARSE_INDEX \
        DENSE_CORPUS_DIR SPARSE_CORPUS_DIR STAGE1_SPARSE_INDEX STAGE2_DENSE_INDEX STAGE2_SPARSE_INDEX \
        VAL_STAGE1_SPARSE_INDEX OUTPUT_DIR JOB_DENSE JOB_SPARSE JOB_FIRST JOB_VAL_DENSE JOB_VAL_SPARSE JOB_OUTPUT \
        QWEN_DENSE_PATH QWEN_SPARSE_PATH QWEN_VAL_DENSE_PATH QWEN_VAL_SPARSE_PATH; do
        if [[ -v "${name}" && -n "${!name}" ]]; then
            printf -v "${name}" '%s' "$(dense_mode_path "${!name}")"
        fi
    done
}
