#!/bin/bash
set -e

# 从项目根目录运行；三个脚本的 ENCODE_TYPE 保持一致。
ENCODE_TYPE=text       # text：图检文；image：文检图

ARTIFACTS_FILE="/root/gpufree-share/output/adaptive_origin_text/flickr_${ENCODE_TYPE}.json"

# 读取编码成功后记录的真实目录，不手工拼接视角指纹。
SPARSE_DIR=$(python -B src/adaptive_artifacts.py "$ARTIFACTS_FILE" sparse_root)

# 验证 adaptive 编码记录，建立 collection/index。
python -B src/adaptive_views.py index \
    --sparse-root "$SPARSE_DIR" \
    --threads 16
