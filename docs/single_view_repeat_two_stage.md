# Single-view sparse weights scaled by the selected perspective count

This control experiment supports LLaVA-Llama3-8B and LLaVA-Mistral-7B on
Flickr/COCO, in both retrieval directions. Set
`SINGLE_VIEW_REPEAT_SELECTED_COUNT=true` for encoding, indexing and two-stage
retrieval. It is disabled by default.

The model's existing locked view-selection record supplies `K = len(selected)`.
The encoder constructs only the ordinary whole-input Summary prompt, encodes it
once, and uses the original PromptEOL token selection with
`round(K * log1p(relu(logit)) * 100)` sparse weights. Dense representations are
not scaled. Image/text candidates and queries share the same selection record.
Selection rounds and the size of the candidate perspective pool are not K.

Stage 1 retrieves sparse candidates. Stage 2 reuses the query encoding and
combines dense and sparse scores within those candidates. The existing hybrid
normalization and alpha formula are preserved; a uniform score scale can cancel
under normalization, with quantization still potentially changing results.
There is no Stage 3, ranker loading or prior-cache preparation in retrieval.

## Unified launcher

The unified entry configures all three existing launchers with the same settings:

```bash
bash scripts/logits_x_sparse_two_stage_ir.sh all

# Override the model, dataset, paths or GPU count as needed.
MODEL_FAMILY=llava_llama DATASET=coco GPUS=1 \
  bash scripts/logits_x_sparse_two_stage_ir.sh all

# Resume at one step, using the same configuration as encoding.
bash scripts/logits_x_sparse_two_stage_ir.sh index
bash scripts/logits_x_sparse_two_stage_ir.sh search
```

Omitting the stage runs `all`; `encode`, `index`, `search` and `--help` are also
supported. The default is LLaVA-Mistral/Flickr/test, both candidate modalities and
both query directions, float16, base images, sparse length 30, hierarchical views,
probability-mass threshold 0.9 and automatic examples. Sampling defaults are
20 rounds, 30 texts, 10 images and seed 42. Retrieval depths are 200/200 with
dense weight 0.7. The script always enables selected-count scaling and hybrid
Stage 2, and disables alpha scanning, mean aggregation, overlap, STC, LENS,
full FreeRet prompts, option shuffling and prior-cache preparation.

Edit the configuration block or set environment variables. Default checkpoints
are under `/root/autodl-fs/checkpoints`, data under `/root/autodl-tmp`, and dense,
sparse, encoding logs and results under `/root/autodl-fs`. Set
`CHECKPOINT_ROOT`/`MODEL_PATH`, `DATA_ROOT`, `DENSE_OUTPUT_ROOT`,
`SPARSE_OUTPUT_ROOT`, `ENCODE_LOG_ROOT` and `ADAPTIVE_OUTPUT_ROOT` to your paths.
Use `IR_QUERY_TYPES=text` for text-to-image only, or `IR_QUERY_TYPES=image` for
image-to-text only. The launcher works from any working directory when invoked
with its correct path, and stops at the first failed stage with its exit code.

This entry automatically locates matching selection caches and artifacts. It
clears inherited single-job paths/directions such as `OUTPUT_DIR`, `ENCODE_TYPE`,
`IR_QUERY_TYPE`, `VIEW_SELECTION_FILE` and `TARGET_SPARSE_DIR`; configure the root
and list variables instead. `all`/`encode` re-encode corpora; `search` requires
existing matching representations and indexes. `DRY_RUN=true` delegates the
existing stage preflights, so downstream preflights still require encoded artifacts.

## Run the three existing entries separately

Run from the repository root in the environment containing the model,
Transformers, FAISS and Pyserini. Keep all encoding/selection settings consistent
across these commands. The following uses the two-stage launcher's current
selection policy and retrieval defaults:

```bash
export MODEL_FAMILY=llava_mistral  # or llava_llama
export DATASET=flickr            # or coco
export DATASET_SPLIT=test
export ENCODE_TYPES="image text"
export IR_QUERY_TYPES="image text"
export CHECKPOINT_ROOT=/root/gpufree-share/checkpoints
export DATA_ROOT=/root/gpufree-share
export DENSE_OUTPUT_ROOT=/root/gpufree-share/dense_output
export SPARSE_OUTPUT_ROOT=/root/gpufree-share/sparse_output
export DTYPE=float16
export LLAVA_IMAGE_MODE=base
export SPARSE_LENGTH=30
export TBPR_TYPE=type
export VIEW_SELECTION_MODALITY=both
export VIEW_SELECTION_ROUNDS=20
export VIEW_TEXTS_PER_ROUND=30
export VIEW_IMAGES_PER_ROUND=10
export VIEW_SELECTION_SEED=42
export USE_HIERARCHICAL_VIEWS=true
export USE_VIEW_PROBABILITY_MASS=true
export VIEW_PROBABILITY_MASS_THRESHOLD=0.9
export USE_VIEW_PMI=false
export VIEW_EXAMPLE_MODE=auto
export SINGLE_VIEW_REPEAT_SELECTED_COUNT=true
export USE_RERANK_PRIOR_CACHE=false
export STAGE2_METHOD=hybrid
export STAGE1_DEPTH=200
export STAGE2_DEPTH=200
export HYBRID_ALPHA=0.7
export SELECT_HYBRID_ALPHA=false

bash scripts/encode_adaptive_views_ir.sh
bash scripts/sparse_index_adaptive_ir.sh
bash scripts/two_stage_retrieval_pipeline_adaptive_views_ir.sh
```

The encoder can reuse the existing selection cache for the same experiment.
Generate new candidate representations and indexes: ordinary multi-view and
legacy fixed-x5 artifacts do not describe this representation. New paths contain
`_single_view_repeat_k<K>_<selection fingerprint prefix>` while retaining the
existing image-resolution and FreeRet suffix conventions. Different selections
and counts are isolated in encoding fingerprints, query cache keys and results.

For direct Python encoding, use `--single_view_repeat_selected_count true`
(alias: `--single-view-repeat-selected-count true`) with the existing adaptive
encoding options. Direct two-stage retrieval accepts
`--single-view-repeat-selected-count` together with `--adaptive-views`,
`--stop-after-stage2` and `--stage2-method hybrid`.

Inspect `single_view_repeat` in `adaptive_views_manifest.json`, `batch_job.json`
and `pipeline_metadata.json` to see the resolved K and selection fingerprints.
Missing/empty selections, inconsistent `selected_count`, and incompatible cached
representations fail rather than falling back to five views. Mean aggregation,
the legacy fixed-x5 flag, overlap, STC, LENS and full FreeRet multi-view prompts
are incompatible with this mode. Existing behavior, including
`scripts/logits_x5_sparse.sh`, is preserved when the new switch is off.
