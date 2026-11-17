#!/bin/bash

# 从项目根目录运行；三个脚本的这两项配置保持一致。
ENCODE_TYPE=text       # text：图检文；image：文检图
SPARSE_MANUAL=false    # false：原文取词；true：全词表预测Top-30

# 以下只用于匹配编码结果的目录，不用修改。
SPARSE_MODE=no_manual_30
if [[ "$SPARSE_MANUAL" == true ]]; then
    SPARSE_MODE=manual_30_30
fi

IMAGE_SUFFIX=""
if [[ "$ENCODE_TYPE" == image ]]; then
    IMAGE_SUFFIX=_llava_base
fi

SPARSE_DIR="/root/gpufree-share/sparse_output/llava-hf-llava-v1.6-mistral-7b-hf/flickr/$ENCODE_TYPE/no_filter/separate/prompt_5/test"
SPARSE_DIR="$SPARSE_DIR/0_${SPARSE_MODE}_sum_no_cluster_after_pad_prompteol_lower_no_mean${IMAGE_SUFFIX}"

# 复用现有建索引工具，FreeRet功能默认关闭；已有索引时无需重复运行。
python -B src/freeret_sparse_index.py build \
    --family llava_mistral \
    --sparse-root "$SPARSE_DIR" \
    --threads 16
