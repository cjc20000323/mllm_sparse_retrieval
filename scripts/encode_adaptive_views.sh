#!/usr/bin/env bash
# auto 按实际上下文长度填充示例；fixed 使用 VIEW_TEXTS_PER_ROUND / VIEW_IMAGES_PER_ROUND。
VIEW_EXAMPLE_MODE="${VIEW_EXAMPLE_MODE-auto}"
VIEW_EXAMPLE_MAX_TOKENS="${VIEW_EXAMPLE_MAX_TOKENS-0}" # 0 使用模型上限；可下调，总预算包含一个回答 token
USE_HIERARCHICAL_VIEWS="${USE_HIERARCHICAL_VIEWS-false}"  # true: ten primary/secondary dimensions (A-J); false: original A-T
case "${USE_HIERARCHICAL_VIEWS,,}" in
    true|1|yes) USE_HIERARCHICAL_VIEWS=true ;;
    false|0|no) USE_HIERARCHICAL_VIEWS=false ;;
    *) echo 'Invalid USE_HIERARCHICAL_VIEWS: use true or false' >&2; exit 2 ;;
esac
SHUFFLE_VIEW_OPTIONS="${SHUFFLE_VIEW_OPTIONS-false}"  # One recorded permutation per model/dataset in this run
case "${SHUFFLE_VIEW_OPTIONS,,}" in
    true|1|yes) SHUFFLE_VIEW_OPTIONS=true ;;
    false|0|no) SHUFFLE_VIEW_OPTIONS=false ;;
    *) echo 'Invalid SHUFFLE_VIEW_OPTIONS: use true or false' >&2; exit 2 ;;
esac
ADAPTIVE_SHUFFLE_ARGS=(--view_selection_shuffle "${SHUFFLE_VIEW_OPTIONS}")
if [[ "${SHUFFLE_VIEW_OPTIONS}" == true && -n "${_ADAPTIVE_SHUFFLE_RUN_ID:-}" ]]; then
    ADAPTIVE_SHUFFLE_ARGS+=(--view_selection_shuffle_run_id "${_ADAPTIVE_SHUFFLE_RUN_ID}")
fi
USE_FREERET_DENSE="${USE_FREERET_DENSE-false}"  # true: FreeRet pre-MLP dense; prompts/sparse unchanged
USE_FREERET_FULL_PROMPT="${USE_FREERET_FULL_PROMPT-false}"  # Complete FreeRet dense/sparse prompts; requires FreeRet dense
USE_FREERET_PROMPT_CONSTRAINTS="${USE_FREERET_PROMPT_CONSTRAINTS-false}"  # Prepend FreeRet constraints to all representation views
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/freeret_dense_mode.sh" || exit $?
# Step 1: select perspectives and encode one candidate modality. Existing files are unchanged.
# Example: MODEL_FAMILY=llava_llama DATASET=coco IR_QUERY_TYPE=text bash scripts/encode_adaptive_views.sh
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
    cat <<'USAGE'
Usage: [environment] bash scripts/encode_adaptive_views.sh [extra encode arguments]
LLAVA_IMAGE_MODE=anyres|base (default: anyres; base requires LLaVA-NeXT).
MODEL_FAMILY=qwen2|qwen25|qwen3|llava_llama|llava_mistral (default: qwen25).
MODEL_PATH overrides the checkpoint under CHECKPOINT_ROOT=/root/autodl-fs/checkpoints.
DATASET=flickr, DATASET_SPLIT=test, IR_QUERY_TYPE=image|text (default: image).
ENCODE_TYPE optionally overrides the inferred candidate modality.
GPUS=1 NUM_NODES=1 DTYPE=float16 DATA_ROOT=/root/autodl-tmp.
DENSE_OUTPUT_ROOT=./dense_output SPARSE_OUTPUT_ROOT=./sparse_output.
SPARSE_LENGTH=40; set TEXT_SPARSE_LENGTH/IMAGE_SPARSE_LENGTH if encoding used different values.
VIEW_SELECTION_FILE: reuse a locked file; otherwise generate and print a new path.
VIEW_SELECTION_MODALITY=text|image|both. New selections default to both.
USE_VIEW_PMI=true|false: remove the no-example prior bias using per-round softmax PMI correction.
New selections default false; an omitted switch inherits an explicitly reused file's setting.
Explicit PMI/file conflicts fail. PMI selection caches and encoded artifacts are isolated.
When VIEW_SELECTION_MODALITY is omitted for an existing file, inherit its mode; explicit conflicts fail.
both scores all text rounds before all image rounds and averages the modalities equally.
VIEW_SELECTION_ROUNDS=20 VIEW_SELECTION_SEED=42.
VIEW_EXAMPLE_MODE=auto|fixed defaults auto; VIEW_EXAMPLE_MAX_TOKENS=0 uses the model context limit.
Positive budgets may lower that limit; one answer token is reserved, and inputs are never truncated.
Fixed mode uses VIEW_TEXTS_PER_ROUND=8 VIEW_IMAGES_PER_ROUND=2 and preserves legacy cache fingerprints.
VIEW_SELECTION_SPLIT defaults to DATASET_SPLIT. IS_FILTERED/SPARSE_VALUE_MEAN default false.
Rank 0 prints each round's sample IDs, top choice, all candidate probabilities, and aggregate rankings live.
Reused selections print the stored ranking and selected views without replaying rounds.
This step does not build Lucene or start retrieval. Read the printed next-step command.
USAGE
    exit 0
fi
MODEL_FAMILY="${MODEL_FAMILY:-qwen25}"
case "${MODEL_FAMILY}" in
    qwen2) CHECKPOINT=Qwen-Qwen2-VL-7B-Instruct ;;
    qwen25) CHECKPOINT=Qwen-Qwen2.5-VL-7B-Instruct ;;
    qwen3) CHECKPOINT=Qwen-Qwen3-VL-8B-Instruct ;;
    llava_llama) CHECKPOINT=llava-hf-llama3-llava-next-8b-hf ;;
    llava_mistral) CHECKPOINT=llava-hf-llava-v1.6-mistral-7b-hf ;;
    *) printf 'Invalid MODEL_FAMILY: %s\n' "${MODEL_FAMILY}" >&2; exit 2 ;;
esac
MODEL_PATH="${MODEL_PATH:-${CHECKPOINT_ROOT:-/root/autodl-fs/checkpoints}/${CHECKPOINT}}"
DATASET="${DATASET:-flickr}"
IR_QUERY_TYPE="${IR_QUERY_TYPE:-image}"
case "${DATASET}" in
    flickr|coco)
        TASK=ir
        case "${IR_QUERY_TYPE}" in image) KIND=text ;; text) KIND=image ;; *) echo 'Invalid IR_QUERY_TYPE' >&2; exit 2 ;; esac ;;
    CUHK-PEDES|ICFG-PEDES|RSTPReid) TASK=tbpr; KIND=image ;;
    webqa) TASK=t2it; KIND=corpus ;;
    remuq|llava) TASK=it2t; KIND=corpus ;;
    *) printf 'Unsupported DATASET: %s\n' "${DATASET}" >&2; exit 2 ;;
esac
PRECISION=()
case "${DTYPE:-float16}" in float16) PRECISION+=(--fp16) ;; bfloat16) PRECISION+=(--bf16) ;; float32) ;; *) echo 'Invalid DTYPE' >&2; exit 2 ;; esac
EXTRA=()
if [[ -v USE_VIEW_PMI ]]; then
    case "${USE_VIEW_PMI,,}" in
        true|1|yes) EXTRA+=(--view_selection_pmi true) ;;
        false|0|no) EXTRA+=(--view_selection_pmi false) ;;
        *) echo 'Invalid USE_VIEW_PMI: use true or false' >&2; exit 2 ;;
    esac
fi
if [[ -n "${VIEW_SELECTION_MODALITY:-}" ]]; then
    case "${VIEW_SELECTION_MODALITY}" in text|image|both) ;; *) echo 'Invalid VIEW_SELECTION_MODALITY: use text, image or both' >&2; exit 2 ;; esac
    EXTRA+=(--view_selection_modality "${VIEW_SELECTION_MODALITY}")
fi
[[ -z "${VIEW_SELECTION_FILE:-}" ]] || EXTRA+=(--view_selection_file "${VIEW_SELECTION_FILE}")
[[ -z "${VIEW_SELECTION_SPLIT:-}" ]] || EXTRA+=(--view_selection_split "${VIEW_SELECTION_SPLIT}")
for pair in 'IS_FILTERED:is_filtered' 'SPARSE_VALUE_MEAN:sparse_value_mean'; do
    name="${pair%%:*}"; option="${pair#*:}"
    case "${!name:-false}" in true|1|yes) EXTRA+=("--${option}") ;; false|0|no) ;; *) printf 'Invalid %s\n' "${name}" >&2; exit 2 ;; esac
done
"${DEEPSPEED_BIN:-deepspeed}" --num_gpus="${GPUS:-1}" --num_nodes="${NUM_NODES:-1}" src/encode_adaptive_views.py --use_freeret_dense "${USE_FREERET_DENSE}" --use_freeret_prompt_constraints "${USE_FREERET_PROMPT_CONSTRAINTS}" --use_freeret_full_prompt "${USE_FREERET_FULL_PROMPT}" --view_selection_hierarchical "${USE_HIERARCHICAL_VIEWS}" "${ADAPTIVE_SHUFFLE_ARGS[@]}" \
    --llava_image_mode "${LLAVA_IMAGE_MODE:-anyres}" \
    --adaptive_views --model_name_or_path "${MODEL_PATH}" --dataset_name "${DATASET}" \
    --dataset_split "${DATASET_SPLIT:-test}" --task_type "${TASK}" --encode_type "${ENCODE_TYPE:-${KIND}}" \
    --adaptive_data_root "${DATA_ROOT:-/root/autodl-tmp}" --output_dir "${OUTPUT_DIR:-./adaptive_encode_logs}" \
    --dense_output_dir "${DENSE_OUTPUT_ROOT:-./dense_output}" --sparse_output_dir "${SPARSE_OUTPUT_ROOT:-./sparse_output}" \
    --eol_type disassembleeol_separate --calculate_type concat --sparse_manual false \
    --prompt_type prompt_5 --random_view_seed 42 --reps_loc "${REPS_LOC:-after_pad}" \
    --per_device_batch_size "${PER_DEVICE_BATCH_SIZE:-1}" --sparse_length "${SPARSE_LENGTH:-40}" \
    --text_sparse_length "${TEXT_SPARSE_LENGTH:-${SPARSE_LENGTH:-40}}" --image_sparse_length "${IMAGE_SPARSE_LENGTH:-${SPARSE_LENGTH:-40}}" \
    --sparse_value_type "${SPARSE_VALUE_TYPE:-sum}" --sparse_lower_or_upper "${SPARSE_LOWER_OR_UPPER:-lower}" \
    --num_expended_tokens "${NUM_EXPENDED_TOKENS:-0}" --tbpr_type "${TBPR_TYPE:-origin_type}" --cir_type type \
    --qwen_min_pixels "${QWEN_MIN_PIXELS:-0}" --qwen_max_pixels "${QWEN_MAX_PIXELS:-0}" \
    --view_example_mode "${VIEW_EXAMPLE_MODE}" --view_example_max_tokens "${VIEW_EXAMPLE_MAX_TOKENS}" \
    --view_selection_rounds "${VIEW_SELECTION_ROUNDS:-20}" --view_texts_per_round "${VIEW_TEXTS_PER_ROUND:-8}" \
    --view_images_per_round "${VIEW_IMAGES_PER_ROUND:-2}" --view_selection_seed "${VIEW_SELECTION_SEED:-42}" \
    "${PRECISION[@]}" "${EXTRA[@]}" "$@"
