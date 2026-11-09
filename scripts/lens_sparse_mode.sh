#!/usr/bin/env bash
# Launchers declare defaults; this helper forwards controls and builds the centroid head before NCCL.
lens_sparse_configure() {
    LENS_ARGS=()
    case "${USE_LENS_SPARSE,,}" in
        true|1|yes) USE_LENS_SPARSE=true ;;
        false|0|no) USE_LENS_SPARSE=false ;;
        *) echo 'Error: USE_LENS_SPARSE must be true or false.' >&2; return 2 ;;
    esac
    if [[ "${USE_LENS_SPARSE}" == true ]]; then
        case "${USE_STC_SPARSE:-false}" in true|1|yes) echo 'Error: USE_LENS_SPARSE and USE_STC_SPARSE are mutually exclusive.' >&2; return 2 ;; esac
        case "${LENS_ALGORITHM}" in kmeans|minibatch_kmeans) ;; *) echo 'Error: invalid LENS_ALGORITHM.' >&2; return 2 ;; esac
        LENS_ARGS=(--use_lens_sparse true --lens_algorithm "${LENS_ALGORITHM}" --lens_n_clusters "${LENS_N_CLUSTERS}"
                   --lens_seed "${LENS_SEED}" --lens_n_init "${LENS_N_INIT}" --lens_max_iter "${LENS_MAX_ITER}"
                   --lens_batch_size "${LENS_BATCH_SIZE}" --lens_cache_root "${LENS_CACHE_ROOT}")
    fi
    LENS_ENV=()
    local lens_name
    for lens_name in USE_LENS_SPARSE LENS_ALGORITHM LENS_N_CLUSTERS LENS_SEED LENS_N_INIT LENS_MAX_ITER LENS_BATCH_SIZE LENS_CACHE_ROOT; do
        LENS_ENV+=("${lens_name}=${!lens_name}")
    done
}

lens_sparse_prepare() {
    [[ "${USE_LENS_SPARSE}" == true ]] || return 0
    LENS_CACHE_ROOT="${LENS_CACHE_ROOT:-${SPARSE_OUTPUT_ROOT:-./sparse_output}/lens_heads}"
    lens_sparse_configure || return $?
    "${PYTHON_BIN:-python}" -B src/lens_sparse.py build --model-name-or-path "${MODEL_PATH}" \
        --dtype "${DTYPE:-float16}" --is-filtered "${IS_FILTERED:-false}" \
        --sparse-lower-or-upper "${SPARSE_LOWER_OR_UPPER:-lower}" "${LENS_ARGS[@]}"
}

lens_sparse_configure || return $?
