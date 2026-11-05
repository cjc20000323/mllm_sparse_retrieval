# LLaVA-NeXT image processing modes

The ordinary and adaptive encode/three-stage entrypoints support `anyres` and
`base`. The default is `anyres`, preserving checkpoint-native preprocessing.
The switch applies to LLaVA-NeXT (including Llama3 and Mistral). Qwen and other
model families reject `base`; their default processing stays unchanged.

## Encoding

Add to `encode.py` or `encode_adaptive_views.py`:

```bash
--llava_image_mode base
```

Or use `LLAVA_IMAGE_MODE=base` with `scripts/encode.sh` or
`scripts/encode_adaptive_views.sh`. Existing model, dataset, split and sparse
settings still apply. Adaptive view selection uses the same image mode as
adaptive encoding, including image-only and text-then-image selection.

`base` retains only the native global image. It resizes the complete image to
the vision tower's nominal size, with the same interpolation/normalization as
the native global branch. It does not generate or encode local AnyRes crops.
The processor expands image placeholders to match the projected global features
and the native single-image `image_newline` vector. Standard 336/14 CLIP settings
therefore produce 577 visual positions per image (576 patches plus one newline).
The count is calculated from the checkpoint, not hard-coded. This is an
inference ablation; retrieval accuracy should be measured again.

The shared adapter accepts both legacy fast-processor `interpolation` calls
and unified Torchvision-backend `resample` calls (PIL/int or Torchvision enums).
It uses the native backend's global resize. It also preserves the installed
model's feature interface: legacy chunks/packed lists or a newer ModelOutput
with projected image features in `pooler_output`, including `return_dict`.
These compatibility changes apply to both ordinary and adaptive entrypoints;
no mode, artifact suffix or manifest version change is needed.

## Standalone search and rerank

Both `src/search.py` and `src/rerank.py` accept `--llava_image_mode base`.
Their Python default remains `anyres`. The rerank entrypoint uses one model for
query encoding and reranking, so the selected image mode applies to both.
Artifact paths supplied to Python are kept intact and checked against the
encoding sidecars. Image-bearing corpora require matching base encodings;
text-only corpora can reuse existing representations. Validation artifacts
are checked when validation retrieval is enabled, but ignored for fixed alpha.

New independent launchers leave `scripts/search.sh` and `scripts/rerank.sh`
unchanged:

```bash
bash scripts/search_llava_base.sh
USE_VALIDATION=true bash scripts/search_llava_base.sh
QUERY_TYPE=text bash scripts/rerank_llava_base.sh
LLAVA_IMAGE_MODE=anyres USE_FIXED_ALPHA=true bash scripts/search_llava_base.sh
```

These launchers default to base mode, Mistral-7B, Flickr, `prompt_5`,
`disassembleeol_separate+concat` and sparse length 40. Use `--help` for model,
dataset, index and output overrides. They derive the native encode directories,
appending `_llava_base` only for image candidates. Explicit paths are preserved.
Recall spreadsheets retain the existing Python `search_results` hierarchy,
with `_llava_base` appended to the experiment leaf for all base runs. Rerank
leaves receive the same suffix before any `use_fixed_alpha/` subdirectory.
The auxiliary `output_dir` remains explicit; time estimates record the image
mode and resolution metadata there. The launchers print both destinations.

The search launcher defaults to `USE_VALIDATION=false USE_FIXED_ALPHA=false`:
it retrieves the test queries once, then selects alpha from 0.1 to 0.9 using
the cached dense/sparse scores. No validation annotations, representations,
indexes or generated validation data are accessed. Validation environment
settings are ignored in this mode. The existing per-alpha and best XLSX reports
are retained. Set `USE_VALIDATION=true` to use validation indexes, or set
`USE_FIXED_ALPHA=true ALPHA=0.5` to use a fixed weight and skip validation.
Fixed alpha takes precedence over `USE_VALIDATION`. Direct `search.py` calls
also skip validation dataset construction when no val indexes are supplied or
fixed alpha is enabled. The rerank launcher's validation defaults are unchanged.

## Three-stage retrieval

The ordinary and adaptive pipeline Python entrypoints accept independent modes:

```bash
--retrieval-llava-image-mode anyres \
--ranker-llava-image-mode base
```

This example changes only reranking, so existing retrieval artifacts can be
reused. To use base images in stages 1/2 as well, set both flags to `base`.
The mode is recorded in `pipeline_metadata.json`, together with the nominal
image size and expected visual length when that model is loaded.

The ordinary ITR/TBPR/multimodal launchers and the adaptive launcher accept:

```bash
LLAVA_IMAGE_MODE=base bash scripts/three_stage_retrieval_pipeline.sh
```

Set `RETRIEVAL_LLAVA_IMAGE_MODE` or `RANKER_LLAVA_IMAGE_MODE` to override one
stage independently. For example, to keep AnyRes retrieval and test base
reranking, use `RETRIEVAL_LLAVA_IMAGE_MODE=anyres RANKER_LLAVA_IMAGE_MODE=base`.
Launchers keep explicit index/result paths intact; their derived result paths
distinguish runs using a nondefault image mode.

## Artifacts and indexing

Ordinary image-bearing base encodings append `_llava_base` to the existing
experiment leaf, e.g. `..._lower_no_mean_llava_base/corpus_0.pkl`. Pure-text
ordinary encodings retain their original path and may be reused in either mode.
For the standard tasks, image-bearing corpora are IR text-to-image, TBPR and
WebQA; IR image-to-text and ReMuQ/LLaVA have text-only corpora.

Adaptive base outputs append `_llava_base` to the existing view-fingerprint
directory. Their model identity and selection manifest also distinguish the
image mode, even if both modes happen to select the same five views. Existing
default-mode adaptive identities remain compatible. Re-select/re-encode when
switching adaptive retrieval modes; point the adaptive launcher at the printed
selection, dense and sparse paths.

Normal LLaVA encoding writes a JSON sidecar next to each output directory:

```text
experiment_llava_base/
    corpus_0.jsonl
experiment_llava_base.llava_resolution.json
```

Build the sparse index from the experiment directory as before, and keep the
sidecar next to that directory. Do not move it inside the JSONL collection:
Anserini must only read vector documents. The pipeline checks the sidecar for
image-bearing corpora, including paths ending in `collection/` or `index/`.
Legacy AnyRes artifacts without a sidecar remain accepted; base image artifacts
require matching metadata. Adaptive artifacts retain their existing completion
manifest checks.

This change reduces image blocks and sequence length. It does not add visual
feature/KV caching or change prompts, EOL extraction, candidate depths, fusion
weights, caption scoring or prior correction.
