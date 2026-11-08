#!/usr/bin/env bash
# Launchers declare the public settings; this helper only forwards/validates them.
stc_sparse_configure() {
    STC_ARGS=()
    case "${USE_STC_SPARSE,,}" in
        true|1|yes) USE_STC_SPARSE=true ;;
        false|0|no) USE_STC_SPARSE=false ;;
        *) echo 'Error: USE_STC_SPARSE must be true or false.' >&2; return 2 ;;
    esac
    if [[ "${USE_STC_SPARSE}" == true ]]; then
        case "${STC_ALGORITHM}" in agglomerative|minibatch_kmeans) ;; *) echo 'Error: invalid STC_ALGORITHM.' >&2; return 2 ;; esac
        STC_ARGS=(--use_stc_sparse true --stc_algorithm "${STC_ALGORITHM}" --stc_n_clusters "${STC_N_CLUSTERS}"
                  --stc_seed "${STC_SEED}" --stc_batch_size "${STC_BATCH_SIZE}" --stc_max_iter "${STC_MAX_ITER}"
                  --stc_weight_scale "${STC_WEIGHT_SCALE}" --stc_cache_root "${STC_CACHE_ROOT}")
    fi
    STC_ENV=()
    local stc_name
    for stc_name in USE_STC_SPARSE STC_ALGORITHM STC_N_CLUSTERS STC_SEED STC_BATCH_SIZE STC_MAX_ITER STC_WEIGHT_SCALE STC_CACHE_ROOT; do
        STC_ENV+=("${stc_name}=${!stc_name}")
    done
}

stc_sparse_prepare() {
    [[ "${USE_STC_SPARSE}" == true ]] || return 0
    STC_CACHE_ROOT="${STC_CACHE_ROOT:-${SPARSE_OUTPUT_ROOT:-./sparse_output}/stc_vocab}"
    stc_sparse_configure || return $?
    # No CUDA/NCCL processes exist yet. Repeated model/dataset jobs hit this cache.
    "${PYTHON_BIN:-python}" -B src/stc_sparse.py build --model-name-or-path "${MODEL_PATH}" \
        --dtype "${DTYPE:-float16}" "${STC_ARGS[@]}"
}

stc_sparse_configure || return $?
