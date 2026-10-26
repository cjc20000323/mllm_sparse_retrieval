# FreeRet multi-process time estimation

The standalone launcher runs `src/freeret_retrieval.py` directly through
DeepSpeed. It does not call another shell script, the three-stage pipeline,
or any three-stage index builder.

```bash
# All eight datasets; one process per visible GPU.
bash scripts/estimate_freeret_retrieval_time.sh

# Select physical GPUs and a single dataset.
CUDA_VISIBLE_DEVICES=2,3 NUM_PROCESSES=2 DATASETS=webqa \
  bash scripts/estimate_freeret_retrieval_time.sh

# A subset with custom paths and a separate distributed port.
PYTHON_BIN=/opt/conda/envs/freeret/bin/python \
MODEL_PATH=/data/checkpoints/Qwen2.5-VL-7B-Instruct \
DATASET_ROOT=/data DATASETS=coco,flickr NUM_PROCESSES=1 MASTER_PORT=29503 \
OUTPUT_DIR=/data/freeret_estimates CACHE_TARGETS=1 \
  bash scripts/estimate_freeret_retrieval_time.sh
```

## Defaults and configuration

`DATASETS=all` selects `coco`, `flickr`, `CUHK-PEDES`, `ICFG-PEDES`,
`RSTPReid`, `webqa`, `remuq`, and `llava`. `DIRECTION=auto` runs both directions
for COCO/Flickr, text-to-image for person retrieval, text-to-image+text for
WebQA, and image+question-to-text for ReMuQ/LLaVA. This gives ten tasks,
processed sequentially in one distributed job rather than ten competing jobs.
Each task uses the existing FreeRet distributed work partitioning.

Defaults match `scripts/freeret_retrieval.sh`, except for selecting all datasets,
always enabling estimate-only mode, and using a separate output directory:

- `MODEL_PATH=/root/autodl-fs/checkpoints/Qwen2.5-VL-7B-Instruct`,
  `DATASET_ROOT=/root/autodl-tmp`, `SPLIT=test`.
- `DTYPE=bfloat16`, `ATTN_IMPLEMENTATION=sdpa`, `MAX_PIXELS=0`
  (the existing dataset-specific image setting).
- `TARGET_BATCH_SIZE=8`, `QUERY_BATCH_SIZE=4`, `RERANK_BATCH_SIZE=2`,
  `RERANK_NUM=50`, `STAGE1_TOPK=50`.
- `SIMILARITY_CHUNK_SIZE=8192`, `MAX_TARGET_GPU_GB=4`, `MAX_QUERIES=-1`,
  `METRICS_KS="1 5 10"`, `SEED=2026`, `CACHE_TARGETS=0`.

Set these environment variables to override defaults. Like the original
FreeRet launcher, configuration uses environment variables, not positional or
Python CLI arguments. `CACHE_TARGETS=1` enables compatible target-cache reuse
and writes a complete cache on a miss. To reuse an existing cache, select its
existing output base with `OUTPUT_DIR`. `MAX_QUERIES` limits the workload being
estimated; leave it at `-1` to estimate all eligible queries.

`NUM_PROCESSES=auto` selects all visible GPUs; a positive integer selects that
many GPUs from the front of the visible list. `CUDA_VISIBLE_DEVICES=2,3` refers
to physical GPU IDs. Each process loads a complete model on its own local GPU;
`DEVICE_MAP` is retained for compatibility but ignored by distributed workers.
Even one process uses DeepSpeed. `MASTER_PORT=29500` and `NUM_NODES=1` are the
defaults. Multi-node runs, duplicate/non-numeric GPU selectors, invalid process
counts, missing CUDA and missing DeepSpeed are rejected without a CPU fallback.

Use a CUDA/DeepSpeed environment compatible with the selected model family.
This script does not install packages, download datasets or build indexes.
All eight datasets must be available for the default run; otherwise select a
prepared subset with `DATASETS`.

## Existing FreeRet estimation policy

The script always supplies `--estimate_time_only`, even if an inherited
`ESTIMATE_TIME_ONLY=false` is present. Use the original `freeret_retrieval.sh`
for full evaluation instead. Neither the original launcher nor the estimator
implementation is changed.

Targets must be completely encoded, or read from a compatible cache. Query
retrieval and reranking then use the existing FreeRet estimator: 20 warmup
**query batches**, throughput samples at least 5 seconds apart, and early stop
when the latest 5 rates have population CV strictly below 5%. Reranking
microbatches are not separate query batches. Short or unstable shards finish
and report actual duration; stability does not interrupt an incomplete batch.

The total includes target encoding/cache-read time plus the projected query
encoding/search/reranking duration. Model loading, dataset preparation, image
validation, target placement/cache writing, distributed exchanges/control waits,
Recall and final report writing are excluded under the existing FreeRet scope.
It is not an estimate of the entire job including startup. Workload variability
can affect the projection; validate it against a full run on the CUDA machine.

## Output

The default output base is `results/freeret_time_estimates`. FreeRet appends a
sanitized model-name directory and writes:

```text
results/freeret_time_estimates/<model>/
  time_estimate_summary.json
  prompt_manifest.json
  <dataset>/<effective_split>_<direction>/time_estimate.json
```

The summary is updated after each completed task. Existing effective split
labels are preserved (some dataset loaders use `parquet` rather than `test`).
Estimate JSON files at the same paths are replaced on a repeated run; normal
retrieval outputs are not deleted. Optional target/image caches and temporary
distributed exchange files follow the existing FreeRet behavior. No partial
retrieval results, Recall metrics or full-evaluation summaries are exported.
