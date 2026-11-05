#!/bin/bash
USE_FREERET_DENSE="${USE_FREERET_DENSE-false}"  # true: FreeRet pre-MLP dense; prompts/sparse unchanged
USE_FREERET_FULL_PROMPT="${USE_FREERET_FULL_PROMPT-false}"  # Complete FreeRet dense/sparse prompts; requires FreeRet dense
USE_FREERET_PROMPT_CONSTRAINTS="${USE_FREERET_PROMPT_CONSTRAINTS-false}"  # Prepend FreeRet constraints to all representation views
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/freeret_dense_mode.sh" || exit $?

OUTPUT_DIR=sparse_output
MODEL=llava-hf-llava-v1.6-mistral-7b-hf
DATASET=flickr
MODAL=text
FILTER=no_filter
EXPENDED_TOKENS=0
MANUAL=no_manual
MANUAL_LENGTH=40
CLUSTER=no_cluster
REPS_LOC=after_pad
EOL_TYPE=disassembleeol_separate
SPARSE_VALUE_TYPE=sum
SPARSE_LOWER_OR_UPPER=lower
CALCULATE_TYPE=concat
PROMPT_TYPE=prompt_5
USE_MEAN=no_mean
DATASET_SPLIT=test

DENSE_MODE_SUFFIX=""
[[ "${USE_FREERET_DENSE}" == false ]] || DENSE_MODE_SUFFIX="_dense_freeret_pre_mlp"
if [[ "${USE_FREERET_FULL_PROMPT}" == true ]]; then DENSE_MODE_SUFFIX+="_freeret_full_prompt_v1"
elif [[ "${USE_FREERET_PROMPT_CONSTRAINTS}" == true ]]; then DENSE_MODE_SUFFIX+="_freeret_constraints_v2"; fi
python -m pyserini.index.lucene \
  --collection JsonVectorCollection \
  --input ${OUTPUT_DIR}/${MODEL}/${DATASET}/${MODAL}/${FILTER}/${CALCULATE_TYPE}/${PROMPT_TYPE}/${DATASET_SPLIT}/${EXPENDED_TOKENS}_${MANUAL}_${MANUAL_LENGTH}_${SPARSE_VALUE_TYPE}_${CLUSTER}_${REPS_LOC}_${EOL_TYPE}_${SPARSE_LOWER_OR_UPPER}_${USE_MEAN}${DENSE_MODE_SUFFIX} \
  --index ${OUTPUT_DIR}/${MODEL}/${DATASET}/${MODAL}/${FILTER}/${CALCULATE_TYPE}/${PROMPT_TYPE}/${DATASET_SPLIT}/${EXPENDED_TOKENS}_${MANUAL}_${MANUAL_LENGTH}_${SPARSE_VALUE_TYPE}_${CLUSTER}_${REPS_LOC}_${EOL_TYPE}_${SPARSE_LOWER_OR_UPPER}_${USE_MEAN}${DENSE_MODE_SUFFIX}/index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 16 \
  --impact --pretokenized
