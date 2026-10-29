#!/usr/bin/env bash
# Index Qwen collection/ JSONL files with Pyserini; keep prompt manifests outside input.
# Uses the same configuration as encode_qwen.sh. Set PYTHON_BIN for your environment.
# Existing indexes are not overwritten. This stage does not load model weights.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/qwen_retrieval_common.sh"
qwen_main sparse_index "$@"
