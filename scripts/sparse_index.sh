#!/bin/bash

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

python -m pyserini.index.lucene \
  --collection JsonVectorCollection \
  --input ${OUTPUT_DIR}/${MODEL}/${DATASET}/${MODAL}/${FILTER}/${CALCULATE_TYPE}/${PROMPT_TYPE}/${DATASET_SPLIT}/${EXPENDED_TOKENS}_${MANUAL}_${MANUAL_LENGTH}_${SPARSE_VALUE_TYPE}_${CLUSTER}_${REPS_LOC}_${EOL_TYPE}_${SPARSE_LOWER_OR_UPPER}_${USE_MEAN} \
  --index ${OUTPUT_DIR}/${MODEL}/${DATASET}/${MODAL}/${FILTER}/${CALCULATE_TYPE}/${PROMPT_TYPE}/${DATASET_SPLIT}/${EXPENDED_TOKENS}_${MANUAL}_${MANUAL_LENGTH}_${SPARSE_VALUE_TYPE}_${CLUSTER}_${REPS_LOC}_${EOL_TYPE}_${SPARSE_LOWER_OR_UPPER}_${USE_MEAN}/index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 16 \
  --impact --pretokenized
