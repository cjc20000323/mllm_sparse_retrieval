#!/bin/bash
set -e

# 从项目根目录运行；三个脚本的 ENCODE_TYPE 保持一致。
ENCODE_TYPE=text       # text：图检文；image：文检图
ARTIFACTS_FILE="/root/gpufree-share/output/adaptive_concat/flickr_${ENCODE_TYPE}.json"

# 本地检索仅用占位值通过部分 Pyserini 版本的导入检查。
export OPENAI_API_KEY="${OPENAI_API_KEY:-local-placeholder}"

GPUS=2
NUM_NODES=1

QUERY_TYPE=image
if [[ "$ENCODE_TYPE" == image ]]; then
    QUERY_TYPE=text
fi

DENSE_DIR=$(python -B src/adaptive_artifacts.py "$ARTIFACTS_FILE" dense_root)
SPARSE_DIR=$(python -B src/adaptive_artifacts.py "$ARTIFACTS_FILE" sparse_root)
VIEW_SELECTION_FILE=$(python -B src/adaptive_artifacts.py "$ARTIFACTS_FILE" selection_file)

# 完整候选库检索；稀疏模式、视角和累计概率阈值均从编码记录继承。
deepspeed --num_gpus=$GPUS --num_nodes=$NUM_NODES src/search_adaptive_views.py \
    --model_name_or_path /root/gpufree-share/checkpoints/llava-hf-llava-v1.6-mistral-7b-hf \
    --passage_reps "$DENSE_DIR" \
    --sparse_index "$SPARSE_DIR" \
    --view_selection_file "$VIEW_SELECTION_FILE" \
    --adaptive_data_root /root/gpufree-share \
    --search_output_root /root/gpufree-share/adaptive_search_results \
    --dataset_name flickr \
    --dataset_split test \
    --task_type ir \
    --query_type "$QUERY_TYPE" \
    --llava_image_mode base \
    --dtype float16 \
    --view_selection_token_variants false \
    --view_selection_prefix_tokens false \
    --depth 1000 \
    --alpha 0.7 \
    --use_fixed_alpha true \
    --retrieval_batch_size 1 \
    --threads 16 \
    --use_gpu true
