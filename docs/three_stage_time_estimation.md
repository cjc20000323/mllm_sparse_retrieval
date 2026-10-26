# Three-stage retrieval time estimation

Run the standalone launcher with the existing corpus indexes and model snapshots:

```bash
DATASET=webqa bash scripts/estimate_three_stage_retrieval_time.sh
CUDA_VISIBLE_DEVICES=2,3 NUM_PROCESSES=2 DATASET=remuq \
  bash scripts/estimate_three_stage_retrieval_time.sh
DATASET=coco QUERY_TYPE=text bash scripts/estimate_three_stage_retrieval_time.sh
DATASET=CUHK-PEDES TBPR_TYPE=origin_type bash scripts/estimate_three_stage_retrieval_time.sh
```

Eight datasets are supported, one per invocation: `coco`, `flickr`, `CUHK-PEDES`,
`ICFG-PEDES`, `RSTPReid`, `webqa`, `remuq`, `llava`. COCO/Flickr default to image
queries and caption candidates; `QUERY_TYPE=text` reverses the direction. Person
retrieval and WebQA use text queries. ReMuQ/LLaVA `image` and `composed` queries
both retain the image **and** question text.

Defaults match the multimodal pipeline: WebQA/test, LLaVA-Mistral-7B, FP16,
retrieval/rerank batch size 1, rerank 5 candidates, depths 1000, dense alpha 0.5,
`sparse_concat` Stage 1 and `disassembleeol_separate+concat` Stage 2. Set
`BASE_MODEL`, `ARTIFACT_MODEL_NAME`, `RETRIEVAL_MODEL_PATH`, `RANKER_MODEL_PATH`,
`DATA_ROOT`, or explicit `STAGE1_SPARSE_INDEX`, `STAGE2_DENSE_INDEX`,
`STAGE2_SPARSE_INDEX` to match your existing artifacts. Models/datasets/indexes
are not downloaded or built by this launcher. This does not use native MMRet,
UniME, VLM2Vec, or FreeRet embedding indexes; it profiles the existing three-stage
pipeline and its dense **and Lucene sparse** indexes.

## What executes

1. Complete Stage 1 for all disjoint query shards. Each worker encodes its own
   queries and retains their CPU dense vectors and sparse topics.
2. Complete Stage 2 using those vectors/topics, without a second query encoding
   or a second retrieval model load. Retrieval weights are released before the
   ranker is loaded.
3. Rerank real Stage-2 candidates, using the same shared `StableRateEstimator`
   and `StageTimeObserver` as FreeRet. The first **20 complete queries** execute
   as warmup. Subsequent rate samples cover at least **5 seconds** each. Stop a
   worker when the latest **5 rates** have population CV **strictly below 5%**,
   or when it exhausts its shard. Rerank candidate microbatches are not separate
   queries. Short/unstable shards finish completely and report actual duration.

The estimate is **not** obtained by truncating upstream candidates or multiplying
one GPU's throughput by the process count. Early-finished/empty workers remain
in small control collectives until every worker finishes. Sampling control/wait
time is excluded, and the maximum rank duration is used for each parallel stage.
No partial retrieval files, Recall, normal pipeline metadata or Excel files are
exported, even if all queries happen to finish before stability is established.
The existing full-run launchers retain their original behavior.

## GPU processes and overrides

The script always launches through `PYTHON_BIN -m deepspeed.launcher.runner`,
including one worker. `PYTHON_BIN` must select an environment with the existing
pipeline dependencies, CUDA, DeepSpeed and Lucene/Pyserini. No CPU fallback or
automatic package installation is performed.

`NUM_PROCESSES=auto` (default) uses all visible GPUs; `GPUS` is a process-count
alias when `NUM_PROCESSES` is unset. `CUDA_VISIBLE_DEVICES=2,3` selects physical
GPU IDs, not logical IDs. Duplicate/non-numeric IDs, insufficient visible GPUs
and multi-node settings are rejected. Each worker loads a complete model on its
own local GPU; model sharding/offload across GPUs is not used. `MASTER_PORT`
defaults to 29500 and `NUM_NODES` must be 1.

```bash
PYTHON_BIN=/opt/conda/envs/retrieval/bin/python NUM_PROCESSES=1 MASTER_PORT=29503 \
  DATASET=llava RERANK_NUM=20 RERANK_BATCH_SIZE=2 \
  bash scripts/estimate_three_stage_retrieval_time.sh

bash scripts/estimate_three_stage_retrieval_time.sh \
  --dataset-name RSTPReid --tbpr-type type --rerank-num 20 \
  --stage2-dense-index /data/dense --stage2-sparse-index /data/sparse \
  --stage1-sparse-index /data/sparse --output-dir /data/time-estimate
```

CLI values override environment settings, and the last repeated managed value
wins. Path-affecting labels are resolved before inferred paths and preflight
checks. Prior-cache options remain subject to the existing pipeline's task,
direction and `modify_single`/caption-template restrictions; cache preparation
is auxiliary setup, not estimated reranking compute. Precomputed runs, external
Stage-1 commands, non-`sparse_concat` coarse retrieval and listwise reranking are
rejected by this estimate-only entrypoint.

## Report and timing scope

Default output:

```text
three_stage_time_estimates/<artifact_model>/<dataset>/<direction>/<split>/
  sparse_concat_stage1/[<tbpr_type>/]time_estimate.json
```

An explicit `OUTPUT_DIR` / `--output-dir` is preserved. Existing output
directories require `OVERWRITE=true` / `--overwrite`; this replaces only estimate
files, not other files in that directory. Rank 0 writes JSON atomically after
each completed stage. Failures preserve completed measurements and independent
`failed_rank_<rank>.json` diagnostics; failed/incomplete reports have no total.

The reported total is:

```text
max_rank(Stage 1 actual core compute)
+ max_rank(Stage 2 actual core compute)
+ max_rank(Stage 3 estimated total, or actual if completed)
```

Each stage includes rank reports and the report records world size, per-rank
query counts, configuration, sampling policy, actual/estimated status and seconds.
Core compute includes query preprocessing/encoding, search, sorting, fusion and
reranking; model/index/dataset loading, index construction, prior-cache setup,
query/result transfers, sampling control, Recall, output and cleanup are excluded.
`profiling_job_wall_seconds` separately describes rank zero's elapsed profiling
job after distributed initialization, including setup and incremental report I/O
but excluding the final report write and distributed teardown; it is **not an
estimate of a complete end-to-end job including setup**. Early stopping also
means sampled image sizes/candidate workloads may not perfectly represent later
queries. Validate estimates against a full run on the target CUDA machine.
