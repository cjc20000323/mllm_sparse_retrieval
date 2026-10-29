#!/usr/bin/env bash
# Retrieve candidates then rerank using the existing native Qwen pointwise path.
# Default: what_caption_generation, top 20, rerank batch size 1, modify_type=no.
# Same corpus/settings as search_qwen.sh; does not read search's saved run files.
# Template support and prior-cache restrictions are checked by the Python adapter.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/qwen_retrieval_common.sh"
qwen_main rerank "$@"
