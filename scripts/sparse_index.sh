#!/bin/bash

OUTPUT_DIR=sparse_output
MODEL=llava-hf-llama3-llava-next-8b-hf
DATASET=coco
MODAL=image
FILTER=no_filter
EXPENDED_TOKENS=0
MANUAL=manual
MANUAL_LENGTH=128

python -m pyserini.index.lucene \
  --collection JsonVectorCollection \
  --input ${OUTPUT_DIR}/${MODEL}/${DATASET}/${MODAL}/${FILTER}/${EXPENDED_TOKENS}_${MANUAL}_${MANUAL_LENGTH}_lora \
  --index ${OUTPUT_DIR}/${MODEL}/${DATASET}/${MODAL}/${FILTER}/${EXPENDED_TOKENS}_${MANUAL}_${MANUAL_LENGTH}_lora/index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 16 \
  --impact --pretokenized
