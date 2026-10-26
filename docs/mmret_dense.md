# MMRet / BGE-VL-MLLM dense candidate encoding

`encode_dense.py` supports the LLaVA-NeXT Mistral-based **S1 and S2** checkpoints,
including the original `JUNJIE99/MMRet-MLLM-S1` / `MMRet-MLLM-S2` and the official
`BAAI/BGE-VL-MLLM-S1` / `BGE-VL-MLLM-S2` repositories. This does **not** adapt
MMRet-base/large (CLIP). Matching online query encoding is supported in
`search_dense.py`; the three-stage pipeline is not adapted.

## Download a complete checkpoint

Recommended for a zero-shot baseline:

```bash
hf download BAAI/BGE-VL-MLLM-S1 --local-dir ./checkpoints/BAAI-BGE-VL-MLLM-S1
```

Alternatively:

```bash
hf download BAAI/BGE-VL-MLLM-S2 --local-dir ./checkpoints/BAAI-BGE-VL-MLLM-S2
```

Download the **whole repository**, including config, tokenizer, image processor
and all four weight shards (approximately 15.1 GB of weights). No separate base
model, LoRA merge, model service, or API key is needed. S1 was trained on
MegaPairs; S2 additionally fine-tuned on the MMEB training set, so record the
chosen checkpoint when comparing zero-shot results.

Sources: [official project](https://github.com/VectorSpaceLab/MegaPairs),
[S1](https://huggingface.co/BAAI/BGE-VL-MLLM-S1),
[S2](https://huggingface.co/BAAI/BGE-VL-MLLM-S2).

Use the project's existing Python environment with PyTorch, torchvision, Pillow,
Transformers **>=4.57.3**, accelerate, pandas, a Parquet engine (e.g. pyarrow), and
the project's tevatron package. Multi-GPU launch additionally needs DeepSpeed.
The implementation uses native HF LLaVA-NeXT checkpoint conversion and does not
execute the old checkpoint's Mistral monkey patch. It does not require upgrading
Sentence Transformers or downgrading to the legacy Transformers 4.45 runtime.
No package installation or environment changes are performed by the launcher.

## Run one dataset per invocation

`DATA_ROOT` is the directory **containing `data/`**, e.g. `/root/autodl-tmp`.
The standalone launcher defaults to S1, Flickr/test, image candidates, FP16,
batch size 1, and one process. Change the environment settings or pass the
corresponding `encode_dense.py` options after the script name.

```bash
# COCO/Flickr: image library for text-to-image retrieval
DATASET=coco ENCODE_TYPE=image bash scripts/encode_dense_mmret.sh
DATASET=flickr ENCODE_TYPE=image bash scripts/encode_dense_mmret.sh

# COCO/Flickr: caption library for image-to-text retrieval
DATASET=coco ENCODE_TYPE=text bash scripts/encode_dense_mmret.sh
DATASET=flickr ENCODE_TYPE=text bash scripts/encode_dense_mmret.sh

# TBPR: image candidates; the task is selected automatically
DATASET=CUHK-PEDES bash scripts/encode_dense_mmret.sh
DATASET=ICFG-PEDES bash scripts/encode_dense_mmret.sh
DATASET=RSTPReid bash scripts/encode_dense_mmret.sh

# WebQA: joint image+text candidate embeddings (not a sum of separate vectors)
DATASET=webqa bash scripts/encode_dense_mmret.sh

# ReMuQ/LLaVA: text candidates
DATASET=remuq bash scripts/encode_dense_mmret.sh
DATASET=llava bash scripts/encode_dense_mmret.sh
```

For a custom model location, larger batch or multiple GPUs:

```bash
MODEL_NAME_OR_PATH=/root/checkpoints/BGE-VL-MLLM-S2 \
DATA_ROOT=/root/autodl-tmp DATASET=flickr ENCODE_TYPE=text BATCH_SIZE=4 \
CUDA_VISIBLE_DEVICES=2,3 NUM_PROCESSES=2 MASTER_PORT=29501 \
bash scripts/encode_dense_mmret.sh
```

`NUM_PROCESSES=1` uses Python directly; larger values use DeepSpeed, with one
complete model on each GPU. It is data parallelism, **not** model sharding: each
GPU must fit the model plus its batch. Start with batch size 1. `DTYPE` accepts
`float16` (default), `bfloat16`, or `float32`. For CPU smoke tests use `float32`.
Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` after downloading to verify
that the local checkpoint is self-contained. A Hugging Face ID instead of a
local path is also supported, but it can trigger a download.

For direct Python launch, add `--dense_model_type mmret` if the directory name
does not contain `MMRet-MLLM` or `BGE-VL-MLLM`:

```bash
python src/encode_dense.py \
  --model_name_or_path /root/checkpoints/my_model --dense_model_type mmret \
  --dataset_name webqa --dataset_split test --task_type t2it --encode_type corpus \
  --data_root /root/autodl-tmp --dense_output_dir ./dense_output \
  --per_device_batch_size 1 --fp16 --output_dir ./dense_output/.trainer --report_to none
```

`--encode_is_query` is rejected in this branch: the legacy flag only changed
the output filename, not the records being encoded. These are **candidate**
embeddings with the official candidate template; online queries need the same
checkpoint and an appropriate query instruction before retrieval is meaningful.

## Dataset layout and behavior

| Dataset | Location below DATA_ROOT | Input / candidate selection |
|---|---|---|
| COCO | `data/coco/` | `<dataset>_<split>.csv`; images in CSV-relative folders; image IDs or all caption IDs |
| Flickr | `data/flickr/` | `flickr_<split>.csv`, `flickr30k-images/`; image IDs or all caption IDs |
| CUHK-PEDES | `data/CUHK-PEDES/` | `reid_raw.json`, `imgs/`; selected split's images |
| ICFG-PEDES | `data/ICFG-PEDES/` | `ICFG-PEDES.json`, `imgs/`; selected split's images |
| RSTPReid | `data/RSTPReid/` | `data_captions.json`, `imgs/`; selected split's images |
| WebQA | `data/WebQA/` | existing query/qrels Parquets plus two corpus shards; candidates selected by the existing dataset's qrels filtering |
| ReMuQ | `data/UMRB-ReMuQ/` | `query.parquet`, `qrels.parquet`, `corpus.parquet`; all text candidates |
| LLaVA | `data/UMRB-LLaVA/` | same three Parquet names; all text candidates |

WebQA filenames are `query-00000-of-00001.parquet`,
`qrels-00000-of-00001.parquet`, `corpus-00000-of-00002.parquet`,
`corpus-00001-of-00002.parquet`. Images can contain embedded bytes or paths
relative to `data/WebQA/`. Pillow decoding supports the BMP images common in
pedestrian datasets. All images follow the official 512x512 preprocessing.
Parquet datasets retain the existing loader's split behavior: setting `val`
does not invent separate files if the data layout has no split-specific files.

## Outputs

Each process writes `(float32_embeddings, original_candidate_ids)` to its own
`corpus_<rank>.pkl`. Embeddings are L2-normalized, usually 4096-dimensional.
Index rows correspond exactly to the IDs in the same pickle. Empty process
shards have shape `(0, 4096)`; sharding never pads with duplicate candidates.

The existing output hierarchy is retained, with the **local model directory
name / HF repository basename** instead of a fixed-length path slice:

```text
<DENSE_OUTPUT_DIR>/<model_basename>/<dataset>/<encode_type>/
  <filter>/<calculate_type>/<prompt_type>/<split>/
  [<tbpr_type>/]<existing_parameter_suffix>/corpus_<rank>.pkl
```

The terminal prints the exact path, vector shape, task, modality and rank.
Generic prompt/eol/sparse settings in this path are compatibility labels; they
do not replace the official MMRet candidate template or pooling. Keep S1/S2 in
separately named directories/output roots. If rerunning with fewer processes,
use a new output directory or move old shards first; stale extra shards are
rejected rather than silently included in a later index.

## Online dense search

Use the **same checkpoint and settings** for both commands. The paired standalone
launchers select MMRet explicitly, including for renamed local checkpoints.
Run one dataset per invocation; the search script automatically locates the
index produced by the corresponding encode command:

```bash
DATASET=coco bash scripts/encode_dense_mmret.sh
DATASET=coco bash scripts/search_dense_mmret.sh
DATASET=flickr bash scripts/encode_dense_mmret.sh
DATASET=flickr bash scripts/search_dense_mmret.sh
DATASET=CUHK-PEDES bash scripts/encode_dense_mmret.sh
DATASET=CUHK-PEDES bash scripts/search_dense_mmret.sh
DATASET=ICFG-PEDES bash scripts/encode_dense_mmret.sh
DATASET=ICFG-PEDES bash scripts/search_dense_mmret.sh
DATASET=RSTPReid bash scripts/encode_dense_mmret.sh
DATASET=RSTPReid bash scripts/search_dense_mmret.sh
DATASET=webqa bash scripts/encode_dense_mmret.sh
DATASET=webqa bash scripts/search_dense_mmret.sh
DATASET=remuq bash scripts/encode_dense_mmret.sh
DATASET=remuq bash scripts/search_dense_mmret.sh
DATASET=llava bash scripts/encode_dense_mmret.sh
DATASET=llava bash scripts/search_dense_mmret.sh

# Reverse direction: image queries against caption candidates.
DATASET=coco QUERY_TYPE=image bash scripts/encode_dense_mmret.sh
DATASET=coco QUERY_TYPE=image bash scripts/search_dense_mmret.sh
DATASET=flickr ENCODE_TYPE=text bash scripts/encode_dense_mmret.sh
DATASET=flickr ENCODE_TYPE=text bash scripts/search_dense_mmret.sh
```

The launchers default to BAAI-BGE-VL-MLLM-S1, Flickr/test, text queries,
one process, FP16 and batch size 1. Override `MODEL_NAME_OR_PATH`, `DATA_ROOT`,
`DATASET_SPLIT`, `DENSE_OUTPUT_DIR`, `BATCH_SIZE`, `NUM_PROCESSES` (or `GPUS`),
`MASTER_PORT` and `DTYPE` as needed. Search also accepts `DEPTH`, `SAVE_DIR` and
`USE_GPU`. CPU inference requires `DTYPE=float32`.

Both scripts infer `ENCODE_TYPE` from `QUERY_TYPE` or the reverse; contradictory
directions fail before loading the model. CLI values override environment
values, and the last repeated option wins. Both `--option=value` and
`--option value` forms work. Other Python CLI arguments are forwarded.

Index paths follow `dense_encoding.output_directory()`, including the model
basename, candidate modality, split, optional `TBPR_TYPE` layer and parameter
suffix. When changing compatibility labels (`PROMPT_TYPE`, `CALCULATE_TYPE`,
`EOL_TYPE`, `NUM_EXPENDED_TOKENS`, `SPARSE_LENGTH`, `SPARSE_VALUE_TYPE`, `REPS_LOC`,
`SPARSE_LOWER_OR_UPPER`, `IS_FILTERED`, `SPARSE_MANUAL`,
`USE_OUTPUT_EMBEDDING_CLUSTER`, `CLUSTER_SUM`, `SPARSE_VALUE_MEAN`), pass the same
values to both commands. The defaults match the Python dataclasses.

Use `PASSAGE_REPS` / `--passage_reps` to override automatic index selection with
an existing index directory or one `corpus_<rank>.pkl` file (which searches only
that shard). The search launcher checks for corpus files before starting Python:

```bash
PASSAGE_REPS=/custom/index SAVE_DIR=./results/mmret bash scripts/search_dense_mmret.sh
CUDA_VISIBLE_DEVICES=2,3 NUM_PROCESSES=2 DATASET=webqa bash scripts/encode_dense_mmret.sh
CUDA_VISIBLE_DEVICES=2,3 NUM_PROCESSES=2 DATASET=webqa bash scripts/search_dense_mmret.sh
```

MMRet queries use the official `q_or_c='q'` layout with task instructions,
`<instruct>` and `<query>` markers. The instruction defaults to the retrieval
direction (caption/image, person, text-to-image+text, or image+text-to-text),
is printed at startup, and can be overridden with
`MMRET_QUERY_INSTRUCTION='your instruction'` or
`--mmret_query_instruction 'your instruction'`. Images follow the same 512x512
preprocessing as candidates; both sides use normalized last-token embeddings.
For ReMuQ/LLaVA, `QUERY_TYPE=image` and `composed` both include image **and text**.

Search preserves dataset IDs, split selection, generated-validation options
and image-path handling. Each process encodes disjoint queries and searches
the complete corpus; Recall@K aggregates actual query counts, including uneven
or empty worker shards. `--use_gpu` uses GPU FAISS on the current worker's GPU
when available, otherwise falls back to CPU search.

Only rank 0 writes `dense.xlsx`, using `SAVE_DIR` / `--save_dir`, or by default
`search_results/<model_basename>/<dataset>/<query_type>/<split>/`. It contains
the existing Recall@K header and dense-recall row. Excel output requires pandas
and an Excel writer (`openpyxl` or `xlsxwriter`). No weights or packages are
downloaded by the launcher. The three-stage pipeline remains separate.
