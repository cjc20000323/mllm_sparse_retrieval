#!/usr/bin/env bash
# Shared selection policy for three-stage launchers; hybrid remains the default.
STAGE2_METHOD="${STAGE2_METHOD-hybrid}" # hybrid / dense (dense reranks Stage-1 candidates)
case "${STAGE2_METHOD}" in hybrid|dense) ;; *) echo 'Error: STAGE2_METHOD must be hybrid or dense.' >&2; return 2 ;; esac
# Preserve the selected mode through wrapper scripts and their child launchers.
export STAGE2_METHOD

stage2_mode_configure() {
    case "${STAGE2_METHOD}" in
        hybrid) ;;
        dense)
            SELECT_HYBRID_ALPHA=false
            HYBRID_ALPHA=0.5 # unused; normalize before legacy launch-time validation
            local inactive
            for inactive in VAL_PASSAGE_REPS VAL_SPARSE_INDEX VAL_STAGE1_RUN VAL_STAGE1_SPARSE_INDEX VAL_VIEW_SELECTION_FILE VALIDATION_DATASET_NAME; do
                printf -v "${inactive}" '%s' ''
            done
            if [[ -z "${STAGE2_MODE_REPORTED:-}" ]]; then
                printf '[stage2] dense within Stage-1 candidates; hybrid weights and alpha scanning are ignored.\n'
                STAGE2_MODE_REPORTED=1
            fi
            ;;
        *) echo 'Error: STAGE2_METHOD must be hybrid or dense.' >&2; return 2 ;;
    esac
}

stage2_mode_path() {
    local value="$1" name suffix=""
    if [[ "${STAGE2_METHOD}" == hybrid ]]; then printf '%s' "${value}"; return; fi
    value="${value%/}"
    name="${value##*/}"
    if [[ "${name}" =~ _dense_freeret_pre_mlp(_freeret_(constraints|full_prompt)_v[0-9]+)?$ ]]; then
        suffix="${BASH_REMATCH[0]}"
        value="${value%"${suffix}"}"
    fi
    [[ "${value}" == *_stage2_dense ]] || value+="_stage2_dense"
    printf '%s%s' "${value}" "${suffix}"
}
