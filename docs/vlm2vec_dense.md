# VLM2Vec-LLaVa-Next native Hugging Face dense retrieval

This adapter supports the complete **TIGER-Lab/VLM2Vec-LLaVa-Next** checkpoint
on all eight project datasets. It uses Transformers' native
`LlavaNextForConditionalGeneration` and `LlavaNextProcessor`, loaded from the
same HF repository or full local snapshot. No project-local VLM2Vec code,
hard-coded Qwen backbone, PEFT adapter, model service or API key is required.
Qwen, Phi and VLM2Vec-V2 are not variants of this new native adapter.
Candidate encoding and online query encoding share the same native backbone,
image preprocessing, last-token pooling and FP32 L2 normalization.

References used for this implementation:

- [Official checkpoint and embedding demo](https://huggingface.co/TIGER-Lab/VLM2Vec-LLaVa-Next).
- [Official V1 pooling and loading](https://github.com/TIGER-AI-Lab/VLM2Vec/blob/v1/src/model.py).
- [Official V1 image preprocessing](https://github.com/TIGER-AI-Lab/VLM2Vec/blob/v1/src/dataset.py).
- [Official evaluation settings](https://github.com/TIGER-AI-Lab/VLM2Vec/blob/v1/examples/llava_next/run_eval.sh).

## Run directly from Hugging Face

The default model identifier is `TIGER-Lab/VLM2Vec-LLaVa-Next`; Transformers
downloads and caches its weights, tokenizer, processor and configuration on the
first call. The launcher does not download any datasets or install packages.
Use the project's existing environment with Transformers **>=4.57.3**, PyTorch,
torchvision, Pillow, accelerate, safetensors, pandas, a Parquet engine and the
project's tevatron package. Multiple GPU processes additionally need DeepSpeed.
Searching additionally needs FAISS (`faiss-cpu` or a compatible GPU build) and
an Excel writer (`openpyxl` or `xlsxwriter`) for the Recall spreadsheet.

Run one dataset per invocation:

```bash
DATASET=coco bash scripts/encode_dense_vlm2vec.sh
DATASET=flickr bash scripts/encode_dense_vlm2vec.sh
DATASET=CUHK-PEDES bash scripts/encode_dense_vlm2vec.sh
DATASET=ICFG-PEDES bash scripts/encode_dense_vlm2vec.sh
DATASET=RSTPReid bash scripts/encode_dense_vlm2vec.sh
DATASET=webqa bash scripts/encode_dense_vlm2vec.sh
DATASET=remuq bash scripts/encode_dense_vlm2vec.sh
DATASET=llava bash scripts/encode_dense_vlm2vec.sh

# COCO/Flickr caption candidates for the opposite retrieval direction:
DATASET=coco ENCODE_TYPE=text bash scripts/encode_dense_vlm2vec.sh
DATASET=flickr QUERY_TYPE=image bash scripts/encode_dense_vlm2vec.sh
```

Defaults are Flickr/test, image candidates, batch size 1, one process, BF16 and
`VLM2VEC_IMAGE_RESOLUTION=high`. `DATA_ROOT` is the parent of `data/`, defaulting
to `/root/autodl-tmp`. The launcher infers task and candidate modality:

| Dataset | Task | Candidate inputs |
|---|---|---|
| COCO / Flickr | `ir` | `image` or `text` |
| CUHK-PEDES / ICFG-PEDES / RSTPReid | `tbpr` | `image` |
| WebQA | `t2it` | `corpus`: image + text |
| ReMuQ / LLaVA | `it2t` | `corpus`: text |

`QUERY_TYPE` is only a direction selector in the encoding script; queries are
encoded by the search script below. Existing split, qrels filtering, generated-validation
selection and original candidate IDs are retained.

```bash
CACHE_DIR=/data/hf-cache DATA_ROOT=/data/root DATASET=webqa \
  bash scripts/encode_dense_vlm2vec.sh
CUDA_VISIBLE_DEVICES=2,3 NUM_PROCESSES=2 DATASET=webqa \
  bash scripts/encode_dense_vlm2vec.sh
DTYPE=float32 VLM2VEC_IMAGE_RESOLUTION=low DATASET=flickr \
  bash scripts/encode_dense_vlm2vec.sh
```

CPU inference requires `DTYPE=float32`. Each GPU process loads a complete
model and encodes disjoint candidate IDs; this is not model sharding. Both
`NUM_PROCESSES` and the `GPUS` alias are accepted. `DTYPE` also accepts
`float16`. CLI overrides environment values; the last repeated value wins.
Other Python arguments are forwarded, including generated-validation settings.

## Local snapshots and direct Python

Download the **whole repository**, not just one of its weight shards:

```bash
hf download TIGER-Lab/VLM2Vec-LLaVa-Next --local-dir ./checkpoints/TIGER-Lab-VLM2Vec-LLaVa-Next
MODEL_NAME_OR_PATH=./checkpoints/TIGER-Lab-VLM2Vec-LLaVa-Next \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DATASET=flickr \
  bash scripts/encode_dense_vlm2vec.sh

python src/encode_dense.py \
  --model_name_or_path TIGER-Lab/VLM2Vec-LLaVa-Next --dense_model_type vlm2vec \
  --dataset_name webqa --dataset_split test --task_type t2it --encode_type corpus \
  --data_root /root/autodl-tmp --dense_output_dir ./dense_output \
  --vlm2vec_image_resolution high --per_device_batch_size 1 --bf16 \
  --report_to none --output_dir ./dense_output/.trainer
```

Canonical repository/directory names are detected automatically. A renamed
complete local checkpoint can use `--dense_model_type vlm2vec`. Its config must
identify LLaVA-NeXT/Mistral and its retrieval parameters must load completely;
the adapter does not silently substitute an unmodified base model.

## Search the matching corpus index

Run encoding first, then search with the **same checkpoint, dataset, split,
direction, path labels and image resolution**. Previously generated native
VLM2Vec candidate indexes remain compatible; their templates have not changed.

```bash
DATASET=webqa bash scripts/encode_dense_vlm2vec.sh
DATASET=webqa bash scripts/search_dense_vlm2vec.sh

# The same pair supports each of the other datasets:
DATASET=coco bash scripts/search_dense_vlm2vec.sh
DATASET=flickr bash scripts/search_dense_vlm2vec.sh
DATASET=CUHK-PEDES bash scripts/search_dense_vlm2vec.sh
DATASET=ICFG-PEDES bash scripts/search_dense_vlm2vec.sh
DATASET=RSTPReid bash scripts/search_dense_vlm2vec.sh
DATASET=remuq bash scripts/search_dense_vlm2vec.sh
DATASET=llava bash scripts/search_dense_vlm2vec.sh

# Image -> caption, with QUERY_TYPE=image on BOTH commands:
DATASET=flickr QUERY_TYPE=image bash scripts/encode_dense_vlm2vec.sh
DATASET=flickr QUERY_TYPE=image bash scripts/search_dense_vlm2vec.sh

# ReMuQ/LLaVA image and composed queries both keep the image AND question text:
DATASET=remuq QUERY_TYPE=composed bash scripts/search_dense_vlm2vec.sh
```

The standalone search launcher defaults to Flickr/test, text queries, batch 1,
one process, BF16 and high resolution. It infers the matching corpus directory,
including `_vlm2vec_<resolution>`, and errors if no index exists. It never falls
back to a legacy index or downloads datasets. `CACHE_DIR`, local full snapshots,
`DTYPE`, `NUM_PROCESSES`/`GPUS` and physical GPU selection work like encoding.
CPU requires `DTYPE=float32`.

```bash
PASSAGE_REPS=/data/custom-index SAVE_DIR=/data/custom-results DATASET=webqa \
  bash scripts/search_dense_vlm2vec.sh
DATASET=webqa VLM2VEC_QUERY_INSTRUCTION='Find an image and text pair answering:' \
  bash scripts/search_dense_vlm2vec.sh

python src/search_dense.py \
  --model_name_or_path TIGER-Lab/VLM2Vec-LLaVa-Next --dense_model_type vlm2vec \
  --dataset_name webqa --dataset_split test --task_type t2it --encode_type corpus \
  --query_type text --data_root /root/autodl-tmp \
  --passage_reps /path/to/matching-index --vlm2vec_image_resolution high \
  --per_device_batch_size 1 --bf16 --depth 1000 \
  --save_dir ./search_results/webqa --report_to none --output_dir ./search_results/.trainer
```

`PASSAGE_REPS` / `--passage_reps` accepts a corpus directory or one `corpus*.pkl`
file; a single file intentionally searches only that shard. `DEPTH` defaults to
1000. `USE_GPU=false` controls FAISS only, not model placement; unavailable GPU
FAISS falls back to CPU. Other search arguments, including `--remove_query`,
`--retrieval_batch_size`, `--quiet` and generated-validation options, are forwarded.
CLI values override environment values; the last repeated value wins.

Query prompts are task-specific rather than copied from candidate prompts:

| Query task | Default instruction |
|---|---|
| COCO/Flickr text -> image | `Retrieve an image matching the following description:` |
| COCO/Flickr image -> text | `Find an image caption describing the given image.` |
| Three pedestrian datasets | `Retrieve a person image matching the following description:` |
| WebQA text -> image+text | `Retrieve an image and text pair answering the following question:` |
| ReMuQ/LLaVA image+text -> text | `Represent the given image with the following question:` |

Text queries use `{instruction} {text}`; image-only queries use
`<image> {instruction}`; joint queries use `<image> {instruction} {text}`.
The joint question form follows the official model-card example; the other
dataset mappings are project defaults, not a claim to reproduce official
benchmark settings. `--vlm2vec_query_instruction` / `VLM2VEC_QUERY_INSTRUCTION`
replaces only the instruction. It must be nonblank and must not include
`<image>` because the encoder inserts the token. `--mmret_query_instruction`
cannot be used with VLM2Vec, nor can the VLM2Vec override be used with other models.

## Embedding conventions

Pure image candidates use `<image> Represent the given image.`. Pure text
candidates use their raw caption or answer text. WebQA candidates use
`<image>\n{corpus_text}` and encode both modalities together, not separately with
a vector sum. This WebQA candidate convention is a **project extension**, not
a claim to reproduce an official WebQA evaluation setting.

Inputs are passed directly to the HF processor; there is no chat wrapper,
`[INST]` summary template or `generate()` call. The last valid token of the
final backbone layer is pooled, then L2-normalized in FP32. The generation head,
KV cache and all-layer language activations are not needed for this path.

`--vlm2vec_image_resolution` / `VLM2VEC_IMAGE_RESOLUTION` controls the resize
before the checkpoint's AnyRes processor: `high` = 1344x1344 (official evaluation
default), `mid` = 672x672, `low` = 128x128, or `original` = no forced resize.
BMP, embedded bytes, Pillow images and image paths are supported; relative
WebQA/ReMuQ/LLaVA image paths are resolved against the dataset directory. Inputs exceeding
the model context limit are rejected instead of truncating image placeholders.

## Outputs and scope

Each process atomically writes `(float32_embeddings, original_candidate_ids)`
to `corpus_<rank>.pkl`. The checkpoint supplies the dimension (4096 for the
official model). Uneven shards never pad/repeat IDs; empty shards retain the
correct two-dimensional shape. Old extra shards from a larger previous process
count cause an error rather than being silently reused.

```text
<DENSE_OUTPUT_DIR>/<model_basename>/<dataset>/<encode_type>/
  <filter>/<calculate_type>/<prompt_type>/<split>/[<tbpr_type>/]
  <existing_parameter_suffix>_vlm2vec_<resolution>/corpus_<rank>.pkl
```

The exact directory is printed before launch and after saving. Resolution
suffixes keep indexes from different preprocessing settings separate. Existing
prompt/eol/sparse flags remain output-path compatibility labels; they do not
change VLM2Vec's candidate templates or pooling.

Search defaults to
`search_results/<model_basename>/<dataset>/<query_type>/<split>/vlm2vec_<resolution>/dense.xlsx`.
An explicit `SAVE_DIR` / `--save_dir` is preserved exactly. Recall@1, 5, 10,
100, 200, 300 and 400 retain the existing dense spreadsheet format. Query shards
are disjoint; actual hit/query counts are reduced across all processes, including
empty local shards, and only rank 0 writes the result. A corpus's IDs and vector
dimension are validated before retrieval. The existing pickle format has no
checkpoint/resolution metadata, so custom index overrides must be selected correctly
by the caller; a same-dimension index from a different model cannot be detected.

`search_dense.py` routes this checkpoint to the native VLM2Vec query encoder,
never to its old Qwen-based branch. The three-stage pipeline is unchanged and
does not use this standalone search adapter. Existing MMRet/UniME outputs are unaffected.

Offline tests use real dataset fixtures and miniature local LLaVA-NeXT weights.
Full-checkpoint GPU runs and real multi-GPU execution require separate testing
on a machine with the complete weights and CUDA devices.
