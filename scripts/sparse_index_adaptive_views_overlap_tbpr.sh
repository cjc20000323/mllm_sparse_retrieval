#!/usr/bin/env bash
USE_FREERET_DENSE="${USE_FREERET_DENSE-false}"  # true: FreeRet pre-MLP dense; prompts/sparse unchanged
USE_FREERET_FULL_PROMPT="${USE_FREERET_FULL_PROMPT-false}"  # Complete FreeRet dense/sparse prompts; requires FreeRet dense
USE_FREERET_PROMPT_CONSTRAINTS="${USE_FREERET_PROMPT_CONSTRAINTS-false}"  # Prepend FreeRet constraints to all representation views
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/freeret_dense_mode.sh" || exit $?
# 仅为 overlap / PMI+overlap 结果批量建索引；每个指纹独立建库，无需 GPU。
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
if (($#)) && ! { (($# == 1)) && [[ "$1" == -h || "$1" == --help ]]; }; then
    echo 'Configure environment variables or the configuration block; see --help.' >&2
    exit 2
fi

# ===== 模型族、数据集与候选模态 =====
# 非空单值覆盖对应列表；支持批量处理同一模型族下的多个检查点及所有匹配指纹。
MODEL_FAMILIES="${MODEL_FAMILIES-qwen2 qwen25 qwen3 llava_llama llava_mistral}"
MODEL_FAMILY="${MODEL_FAMILY-}"
DATASETS="${DATASETS-CUHK-PEDES ICFG-PEDES RSTPReid}"
DATASET="${DATASET-}"
ENCODE_TYPES="${ENCODE_TYPES-image}"
ENCODE_TYPE="${ENCODE_TYPE-}"
DATASET_SPLIT="${DATASET_SPLIT-test}"
# TBPR 默认索引图像候选，也支持 ENCODE_TYPE=text；自动处理已有 origin_type/type 变体。

# ===== 输入目录、线程与模式 =====
# 与 encode_adaptive_views_overlap 的 SPARSE_OUTPUT_ROOT 保持一致。
SPARSE_OUTPUT_ROOT="${SPARSE_OUTPUT_ROOT-${REPO_ROOT}/sparse_output}"
VIEW_SELECTION_ROOT="${VIEW_SELECTION_ROOT-}"  # 可选：原JSON引用失效时，按完整指纹查找副本；空时自动定位view_selection目录
THREADS="${THREADS-${SPARSE_THREADS-16}}"  # THREADS 优先，兼容 SPARSE_THREADS
PMI_MODE="${PMI_MODE-all}"               # all=16/17；off=16；on=17
DRY_RUN="${DRY_RUN-false}"               # true 仅预检、打印，不写索引或目录
PYTHON_BIN="${PYTHON_BIN:-python}"

# 从清单和选择文件识别模式，无需填写视角指纹或重新设置 overlap 参数。
# 普通/仅 PMI 结果跳过；版本或候选池不兼容（含旧13/15）汇总为 skipped_incompatible。
# 每个目标的 collection/corpus_*.jsonl -> collection/index；已有有效索引跳过。
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "Python not found: ${PYTHON_BIN}; activate the indexing environment or set PYTHON_BIN." >&2; exit 2; }
BATCH_ENV=("USE_FREERET_DENSE=${USE_FREERET_DENSE}" "USE_FREERET_PROMPT_CONSTRAINTS=${USE_FREERET_PROMPT_CONSTRAINTS}" "USE_FREERET_FULL_PROMPT=${USE_FREERET_FULL_PROMPT}")
for name in MODEL_FAMILIES MODEL_FAMILY DATASETS DATASET ENCODE_TYPES ENCODE_TYPE \
    DATASET_SPLIT SPARSE_OUTPUT_ROOT VIEW_SELECTION_ROOT THREADS PMI_MODE DRY_RUN; do
    BATCH_ENV+=("${name}=${!name}")
done
exec env "${BATCH_ENV[@]}" "${PYTHON_BIN}" -B src/adaptive_sparse_index_batch.py --task-group tbpr --overlap-only "$@"
