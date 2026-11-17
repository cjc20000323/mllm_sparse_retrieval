#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

# 从项目根目录运行；三个脚本的这两项配置保持一致。
ENCODE_TYPE=text       # text：图检文；image：文检图
SPARSE_MANUAL=false    # PromptEOL：false为原文取词，true为全词表Top-30

EOL_TYPE=prompteol
CALCULATE_TYPE=separate
# 另支持 disassembleeol_separate / concat 和 disassembleeol_separate_origin_text / concat。
# 这里只读取普通编码目录；adaptive 编码需使用对应的检索脚本。
export ENABLE_RETRIEVAL_SIMILARITY_STATS=true  # false：关闭统计

# 本地 Lucene 检索用占位值通过部分 Pyserini 版本的导入检查。
export OPENAI_API_KEY="${OPENAI_API_KEY:-local-placeholder}"

if [[ "$#" -ne 0 ]]; then
    echo '不再使用 both/no_manual/manual 参数，请修改脚本顶部的 SPARSE_MANUAL。' >&2
    exit 2
fi

GPUS=2
NUM_NODES=1

# 查询方向与候选库模态相反；只有图像候选库带base后缀。
QUERY_TYPE=image
IMAGE_SUFFIX=""
if [[ "$ENCODE_TYPE" == image ]]; then
    QUERY_TYPE=text
    IMAGE_SUFFIX=_llava_base
fi

SPARSE_MODE=no_manual_30
if [[ "$SPARSE_MANUAL" == true ]]; then
    SPARSE_MODE=manual_30_30
fi

CORPUS_DIR="llava-hf-llava-v1.6-mistral-7b-hf/flickr/$ENCODE_TYPE/no_filter/$CALCULATE_TYPE/prompt_5/test"
CORPUS_DIR="$CORPUS_DIR/0_${SPARSE_MODE}_sum_no_cluster_after_pad_${EOL_TYPE}_lower_no_mean${IMAGE_SUFFIX}"

deepspeed --num_gpus=$GPUS --num_nodes=$NUM_NODES src/search_similarity.py \
    --model_name_or_path /root/gpufree-share/checkpoints/llava-hf-llava-v1.6-mistral-7b-hf \
    --output_dir /root/gpufree-share/output \
    --llava_image_mode base \
    --use_freeret_dense false \
    --use_freeret_full_prompt false \
    --use_freeret_prompt_constraints false \
    --fp16 \
    --dataset_name flickr \
    --dataset_split test \
    --task_type ir \
    --encode_type "$ENCODE_TYPE" \
    --query_type "$QUERY_TYPE" \
    --passage_reps "/root/gpufree-share/dense_output/$CORPUS_DIR" \
    --sparse_index "/root/gpufree-share/sparse_output/$CORPUS_DIR/index" \
    --per_device_batch_size 1 \
    --retrieval_batch_size 1 \
    --threads 16 \
    --depth 1000 \
    --use_gpu \
    --alpha 0.7 --beta 0.3 --use_fixed_alpha \
    --reps_loc after_pad \
    --num_expended_tokens 0 \
    --eol_type "$EOL_TYPE" \
    --calculate_type "$CALCULATE_TYPE" \
    --prompt_type prompt_5 \
    --sparse_manual "$SPARSE_MANUAL" \
    --sparse_length 30 \
    --text_sparse_length 30 \
    --image_sparse_length 30 \
    --sparse_value_type sum \
    --sparse_lower_or_upper lower
