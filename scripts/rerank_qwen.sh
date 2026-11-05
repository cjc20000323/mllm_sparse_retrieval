#!/usr/bin/env bash
USE_FREERET_DENSE="${USE_FREERET_DENSE-false}"  # true: FreeRet pre-MLP dense; prompts/sparse unchanged
USE_FREERET_FULL_PROMPT="${USE_FREERET_FULL_PROMPT-false}"  # Complete FreeRet dense/sparse prompts; requires FreeRet dense
USE_FREERET_PROMPT_CONSTRAINTS="${USE_FREERET_PROMPT_CONSTRAINTS-false}"  # Prepend FreeRet constraints to all representation views
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/freeret_dense_mode.sh" || exit $?
# Retrieve candidates then rerank using the existing native Qwen pointwise path.
# Default: what_caption_generation, top 20, rerank batch size 1, modify_type=no.
# Same corpus/settings as search_qwen.sh; does not read search's saved run files.
# Template support and prior-cache restrictions are checked by the Python adapter.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/qwen_retrieval_common.sh"
qwen_main rerank "$@"
