#!/bin/bash

OUTPUT_DIR=sparse_output
MODEL=royokong-e5-v
DATASET=flickr
MODAL=text
FILTER=no_filter
EXPENDED_TOKENS=0

python -m pyserini.index.lucene \
  --collection JsonVectorCollection \
  --input ${OUTPUT_DIR}/${MODEL}/${DATASET}/${MODAL}/${FILTER}/${EXPENDED_TOKENS} \
  --index ${OUTPUT_DIR}/${MODEL}/${DATASET}/${MODAL}/${FILTER}/${EXPENDED_TOKENS}/index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 16 \
  --impact --pretokenized
