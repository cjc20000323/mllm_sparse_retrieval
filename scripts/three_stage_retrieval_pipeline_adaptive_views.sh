#!/usr/bin/env bash
USE_HIERARCHICAL_VIEWS="${USE_HIERARCHICAL_VIEWS-false}"  # true: ten primary/secondary dimensions (A-J); false: original A-T
case "${USE_HIERARCHICAL_VIEWS,,}" in
    true|1|yes) USE_HIERARCHICAL_VIEWS=true ;;
    false|0|no) USE_HIERARCHICAL_VIEWS=false ;;
    *) echo 'Invalid USE_HIERARCHICAL_VIEWS: use true or false' >&2; exit 2 ;;
esac
VIEW_SELECTION_RUN_ID="${VIEW_SELECTION_RUN_ID-}"  # Shuffle experiment identifier printed by encoding
USE_FREERET_DENSE="${USE_FREERET_DENSE-false}"  # true: FreeRet pre-MLP dense; prompts/sparse unchanged
USE_FREERET_FULL_PROMPT="${USE_FREERET_FULL_PROMPT-false}"  # Complete FreeRet dense/sparse prompts; requires FreeRet dense
USE_FREERET_PROMPT_CONSTRAINTS="${USE_FREERET_PROMPT_CONSTRAINTS-false}"  # Prepend FreeRet constraints to all representation views
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/freeret_dense_mode.sh" || exit $?
# Step 3: read the locked views and the encoded/indexed artifacts printed in Step 1.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
    cat <<'USAGE'
Usage: [environment] bash scripts/three_stage_retrieval_pipeline_adaptive_views.sh [extra pipeline arguments]
Required: VIEW_SELECTION_FILE, DENSE_CORPUS_DIR, SPARSE_CORPUS_DIR (from adaptive encode).
Build collection/index before this step. SPARSE_CORPUS_DIR accepts a leaf, collection or index.
LLAVA_IMAGE_MODE=anyres|base; RETRIEVAL_LLAVA_IMAGE_MODE/RANKER_LLAVA_IMAGE_MODE override each stage.
MODEL_FAMILY=qwen2|qwen25|qwen3|llava_llama|llava_mistral, or set MODEL_PATH explicitly.
DATASET=flickr DATASET_SPLIT=test IR_QUERY_TYPE=image GPUS=1 DTYPE=float16.
STAGE1_DEPTH=1000 STAGE2_DEPTH=1000 RERANK_NUM=5 RERANK_BATCH_SIZE=1.
HYBRID_ALPHA=0.5 SELECT_HYBRID_ALPHA=false; optional paired VAL_PASSAGE_REPS/VAL_SPARSE_INDEX.
VAL_VIEW_SELECTION_FILE is required for cross-dataset adaptive validation.
SPARSE_LENGTH=40 and all sparse/pixel settings must match adaptive encode.
RANKER_MODEL_PATH defaults to MODEL_PATH. Reranking uses what_caption_generation, no prior correction.
OUTPUT_DIR defaults to ./three_stage_adaptive_results/<model>/<dataset>/<direction>/<split>/stage1_<depth>_stage2_<depth>.
Explicit OUTPUT_DIR/--output-dir stays unchanged. Trailing --stage1-depth and
--stage2-depth override environment defaults before the result path is derived.
OVERWRITE=false. Add --overwrite or set OVERWRITE=true explicitly to reuse a result directory.
USAGE
    exit 0
fi
: "${VIEW_SELECTION_FILE:?Set VIEW_SELECTION_FILE to the selected JSON from adaptive encode}"
: "${DENSE_CORPUS_DIR:?Set DENSE_CORPUS_DIR to the printed adaptive dense directory}"
: "${SPARSE_CORPUS_DIR:?Set SPARSE_CORPUS_DIR to the printed adaptive sparse directory}"
STAGE1_DEPTH="${STAGE1_DEPTH:-1000}"
STAGE2_DEPTH="${STAGE2_DEPTH:-1000}"
# Resolve depth/output overrides before deriving paths; forward other arguments.
PIPELINE_ARGS=()
while (($#)); do
    case "${1%%=*}" in
        --stage1-depth|--stage2-depth|--output-dir)
            CLI_OPTION="${1%%=*}"
            if [[ "$1" == *=* ]]; then
                CLI_VALUE="${1#*=}"
                shift
            else
                if (($# < 2)) || [[ "$2" == --* ]]; then
                    printf 'Error: %s requires a value.\n' "${CLI_OPTION}" >&2
                    exit 2
                fi
                CLI_VALUE="$2"
                shift 2
            fi
            [[ -n "${CLI_VALUE}" ]] || { printf 'Error: %s requires a non-empty value.\n' "${CLI_OPTION}" >&2; exit 2; }
            case "${CLI_OPTION}" in
                --stage1-depth) STAGE1_DEPTH="${CLI_VALUE}" ;;
                --stage2-depth) STAGE2_DEPTH="${CLI_VALUE}" ;;
                --output-dir) OUTPUT_DIR="${CLI_VALUE}" ;;
            esac
            ;;
        --) PIPELINE_ARGS+=("$@"); break ;;
        *) PIPELINE_ARGS+=("$1"); shift ;;
    esac
done
case "${MODEL_FAMILY:-qwen25}" in
    qwen2) CHECKPOINT=Qwen-Qwen2-VL-7B-Instruct ;;
    qwen25) CHECKPOINT=Qwen-Qwen2.5-VL-7B-Instruct ;;
    qwen3) CHECKPOINT=Qwen-Qwen3-VL-8B-Instruct ;;
    llava_llama) CHECKPOINT=llava-hf-llama3-llava-next-8b-hf ;;
    llava_mistral) CHECKPOINT=llava-hf-llava-v1.6-mistral-7b-hf ;;
    *) echo 'Invalid MODEL_FAMILY' >&2; exit 2 ;;
esac
MODEL_PATH="${MODEL_PATH:-${CHECKPOINT_ROOT:-/root/autodl-fs/checkpoints}/${CHECKPOINT}}"
MODEL_PATH="${MODEL_PATH%/}"
RETRIEVAL_LLAVA_IMAGE_MODE="${RETRIEVAL_LLAVA_IMAGE_MODE:-${LLAVA_IMAGE_MODE:-anyres}}"
RANKER_LLAVA_IMAGE_MODE="${RANKER_LLAVA_IMAGE_MODE:-${LLAVA_IMAGE_MODE:-anyres}}"
LLAVA_RESULT_SUFFIX=""
for LLAVA_MODE in "${RETRIEVAL_LLAVA_IMAGE_MODE}" "${RANKER_LLAVA_IMAGE_MODE}"; do
    case "${LLAVA_MODE}" in anyres|base) ;; *) echo "LLaVA image mode must be anyres or base" >&2; exit 2 ;; esac
done
if [[ "${RETRIEVAL_LLAVA_IMAGE_MODE}" == base || "${RANKER_LLAVA_IMAGE_MODE}" == base ]]; then
    LLAVA_RESULT_SUFFIX="_retrieval_${RETRIEVAL_LLAVA_IMAGE_MODE}_ranker_${RANKER_LLAVA_IMAGE_MODE}"
fi
DATASET="${DATASET:-flickr}"
case "${DATASET}" in
    flickr|coco) TASK=ir; QUERY="${IR_QUERY_TYPE:-image}" ;;
    CUHK-PEDES|ICFG-PEDES|RSTPReid) TASK=tbpr; QUERY=text ;;
    webqa) TASK=t2it; QUERY=text ;;
    remuq|llava) TASK=it2t; QUERY=image ;;
    *) echo 'Unsupported DATASET' >&2; exit 2 ;;
esac
EXTRA=()
case "${SELECT_HYBRID_ALPHA:-false}" in
    true|1|yes)
        EXTRA+=(--select-hybrid-alpha)
        [[ -z "${VAL_PASSAGE_REPS:-}" ]] || EXTRA+=(--val-passage-reps "${VAL_PASSAGE_REPS}")
        [[ -z "${VAL_SPARSE_INDEX:-}" ]] || EXTRA+=(--val-sparse-index "${VAL_SPARSE_INDEX}")
        [[ -z "${VALIDATION_DATASET_NAME:-}" ]] || EXTRA+=(--validation-dataset-name "${VALIDATION_DATASET_NAME}")
        [[ -z "${VALIDATION_SPLIT:-}" ]] || EXTRA+=(--validation-split "${VALIDATION_SPLIT}")
        [[ -z "${VAL_VIEW_SELECTION_FILE:-}" ]] || EXTRA+=(--val-view-selection-file "${VAL_VIEW_SELECTION_FILE}") ;;
    false|0|no) ;; *) echo 'Invalid SELECT_HYBRID_ALPHA' >&2; exit 2 ;;
esac
for pair in 'IS_FILTERED:is-filtered' 'SPARSE_VALUE_MEAN:sparse-value-mean' 'OVERWRITE:overwrite'; do
    name="${pair%%:*}"; option="${pair#*:}"
    case "${!name:-false}" in true|1|yes) EXTRA+=("--${option}") ;; false|0|no) ;; *) printf 'Invalid %s\n' "${name}" >&2; exit 2 ;; esac
done
OUTPUT_DIR="${OUTPUT_DIR:-./three_stage_adaptive_results/${MODEL_PATH##*/}/${DATASET}/${QUERY}/${DATASET_SPLIT:-test}${LLAVA_RESULT_SUFFIX}/stage1_${STAGE1_DEPTH}_stage2_${STAGE2_DEPTH}}"
printf 'Results: %s\n' "${OUTPUT_DIR}"
"${DEEPSPEED_BIN:-deepspeed}" --num_gpus="${GPUS:-1}" --num_nodes="${NUM_NODES:-1}" src/three_stage_retrieval_pipeline_adaptive_views.py --view-selection-run-id "${VIEW_SELECTION_RUN_ID}" --use_freeret_dense "${USE_FREERET_DENSE}" --use_freeret_prompt_constraints "${USE_FREERET_PROMPT_CONSTRAINTS}" --use_freeret_full_prompt "${USE_FREERET_FULL_PROMPT}" \
    --retrieval-llava-image-mode "${RETRIEVAL_LLAVA_IMAGE_MODE}" --ranker-llava-image-mode "${RANKER_LLAVA_IMAGE_MODE}" \
    --adaptive-views --view-selection-file "${VIEW_SELECTION_FILE}" --dataset-name "${DATASET}" \
    --dataset-split "${DATASET_SPLIT:-test}" --task-type "${TASK}" --query-type "${QUERY}" \
    --data-root "${DATA_ROOT:-/root/autodl-tmp}" --tbpr-type "${TBPR_TYPE:-origin_type}" \
    --stage1-method sparse_concat --stage1-sparse-index "${STAGE1_SPARSE_INDEX:-${SPARSE_CORPUS_DIR}}" --stage1-depth "${STAGE1_DEPTH}" \
    --stage2-dense-index "${DENSE_CORPUS_DIR}" --stage2-sparse-index "${SPARSE_CORPUS_DIR}" --stage2-depth "${STAGE2_DEPTH}" \
    --hybrid-alpha "${HYBRID_ALPHA:-0.5}" --sparse-threads "${SPARSE_THREADS:-16}" \
    --retrieval-model-name-or-path "${MODEL_PATH}" --retrieval-dtype "${RETRIEVAL_DTYPE:-${DTYPE:-float16}}" \
    --retrieval-reps-loc "${REPS_LOC:-after_pad}" --retrieval-batch-size "${RETRIEVAL_BATCH_SIZE:-1}" \
    --retrieval-qwen-min-pixels "${QWEN_MIN_PIXELS:-0}" --retrieval-qwen-max-pixels "${QWEN_MAX_PIXELS:-0}" \
    --sparse-length "${SPARSE_LENGTH:-40}" --text-sparse-length "${TEXT_SPARSE_LENGTH:-${SPARSE_LENGTH:-40}}" \
    --image-sparse-length "${IMAGE_SPARSE_LENGTH:-${SPARSE_LENGTH:-40}}" --num-expended-tokens "${NUM_EXPENDED_TOKENS:-0}" \
    --sparse-value-type "${SPARSE_VALUE_TYPE:-sum}" --sparse-lower-or-upper "${SPARSE_LOWER_OR_UPPER:-lower}" \
    --ranker-model-name-or-path "${RANKER_MODEL_PATH:-${MODEL_PATH}}" --ranker-dtype "${RANKER_DTYPE:-${DTYPE:-float16}}" \
    --ranker-qwen-min-pixels "${RANKER_QWEN_MIN_PIXELS:-${QWEN_MIN_PIXELS:-0}}" --ranker-qwen-max-pixels "${RANKER_QWEN_MAX_PIXELS:-${QWEN_MAX_PIXELS:-0}}" \
    --rerank-type pointwise --rerank-template what_caption_generation --modify-type no \
    --rerank-num "${RERANK_NUM:-5}" --rerank-batch-size "${RERANK_BATCH_SIZE:-1}" \
    --output-dir "${OUTPUT_DIR}" \
    "${EXTRA[@]}" "${PIPELINE_ARGS[@]}"
