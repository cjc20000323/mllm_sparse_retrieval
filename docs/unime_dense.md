# UniME-LLaVA-1.6-7B dense candidate encoding

This adapter targets **DeepGlint-AI/UniME-LLaVA-1.6-7B** only. It does not target
UniME-V2, UniME-OneVision or UniME-Phi. It encodes candidate libraries in
`encode_dense.py`, and matching online queries in `search_dense.py`. Reranking
and the three-stage pipeline are separate.

## Download the complete model

Run from the project root:

```bash
hf download DeepGlint-AI/UniME-LLaVA-1.6-7B --local-dir ./checkpoints/DeepGlint-AI-UniME-LLaVA-1.6-7B
```

Download the **whole repository**, including all four safetensors shards,
config, tokenizer and processor files (approximately 15.1 GB of weights).
It is a complete checkpoint: no separate base model, LoRA merge, NV-Embed
teacher model, API key or external model service is needed for inference.

Sources: [official model card](https://huggingface.co/DeepGlint-AI/UniME-LLaVA-1.6-7B),
[official code](https://github.com/deepglint/UniME),
[checkpoint preprocessing configuration](https://huggingface.co/DeepGlint-AI/UniME-LLaVA-1.6-7B/blob/main/preprocessor_config.json).

Use the project's existing environment with Transformers **>=4.57.3**, PyTorch,
torchvision, accelerate, Pillow, pandas, a Parquet engine such as pyarrow, and
the project's tevatron package. DeepSpeed is additionally required for multiple
processes. The launcher does not install packages or change the environment.
The native HF model handles old and new checkpoint key layouts. No remote code
or UniME repository clone is required at runtime.

## Standalone launcher

```bash
# COCO/Flickr image candidates for text-to-image retrieval
DATASET=coco ENCODE_TYPE=image bash scripts/encode_dense_unime.sh
DATASET=flickr ENCODE_TYPE=image bash scripts/encode_dense_unime.sh

# COCO/Flickr text candidates for image-to-text retrieval
DATASET=coco ENCODE_TYPE=text bash scripts/encode_dense_unime.sh
DATASET=flickr ENCODE_TYPE=text bash scripts/encode_dense_unime.sh

# Text-person retrieval: image candidates
DATASET=CUHK-PEDES bash scripts/encode_dense_unime.sh
DATASET=ICFG-PEDES bash scripts/encode_dense_unime.sh
DATASET=RSTPReid bash scripts/encode_dense_unime.sh

# WebQA: joint image+text candidates
DATASET=webqa bash scripts/encode_dense_unime.sh

# ReMuQ/LLaVA: text candidates
DATASET=remuq bash scripts/encode_dense_unime.sh
DATASET=llava bash scripts/encode_dense_unime.sh
```

Defaults: the local checkpoint directory above, Flickr/test, image candidates,
FP16, batch size 1 and one process. The script automatically selects `ir`,
`tbpr`, `t2it` or `it2t` and the appropriate candidate modality for each dataset.
COCO/Flickr accept `ENCODE_TYPE=image` or `text`; TBPR requires `image`;
WebQA/ReMuQ/LLaVA require `corpus`.

`DATA_ROOT` is the directory containing **data/** (default `/root/autodl-tmp`),
with `data/coco`, `data/flickr`, `data/CUHK-PEDES`, `data/ICFG-PEDES`,
`data/RSTPReid`, `data/WebQA`, `data/UMRB-ReMuQ`, `data/UMRB-LLaVA`.
Annotation filenames and split handling are the same as existing dataset.py.
WebQA retains its existing qrels-filtered candidate selection; ReMuQ/LLaVA
encode all text candidates. A split setting does not invent split-specific
Parquet files when the existing dataset layout has none.

For a custom local checkpoint path, larger batch and multiple GPUs:

```bash
MODEL_NAME_OR_PATH=/root/checkpoints/UniME-LLaVA-1.6-7B \
DATA_ROOT=/root/autodl-tmp DATASET=flickr ENCODE_TYPE=text BATCH_SIZE=4 \
CUDA_VISIBLE_DEVICES=2,3 NUM_PROCESSES=2 MASTER_PORT=29501 \
bash scripts/encode_dense_unime.sh
```

`NUM_PROCESSES=1` launches Python directly. Larger values launch DeepSpeed
directly, with **one complete model per GPU**, not model sharding. Each GPU
must fit the model plus its batch. Start with batch size 1. Multi-GPU selection
supports numeric GPU IDs, not UUID/MIG identifiers.

`DTYPE` accepts `float16` (default), `bfloat16`, or `float32`. CPU smoke tests
require `float32`. `DENSE_OUTPUT_DIR` changes the candidate output root.
The script also accepts corresponding Python options such as
`--dataset_name coco --encode_type text --data_root /root/data_root`.

For direct Python launch, use `--dense_model_type unime` when your local
directory has an arbitrary name:

```bash
python src/encode_dense.py \
  --model_name_or_path /root/checkpoints/my_model --dense_model_type unime \
  --dataset_name webqa --dataset_split test --task_type t2it --encode_type corpus \
  --data_root /root/autodl-tmp --dense_output_dir ./dense_output \
  --per_device_batch_size 1 --fp16 --report_to none --output_dir ./dense_output/.trainer
```

The canonical repository/local name is automatically recognized with
`--dense_model_type auto`. Other UniME families are rejected instead of
silently falling back to CLIP. A Hugging Face ID may also be passed, but this
can trigger a download. To test a complete local snapshot without network
access, set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` after downloading it.

## Templates and image preprocessing

Image-only and text-only prompts follow the **official model card**. The model
summarizes the supplied content in one word; no generation call is needed.
The final effective token's last-layer hidden state is pooled and L2-normalized.
The adapter disables KV cache, all-layer output collection, and LM-head logits
to reduce inference memory.

Images are decoded with Pillow (including BMP) and converted to RGB without
forcing a new resolution. The checkpoint's `LlavaNextProcessor` owns AnyRes
grid selection, resizing, crops and normalization. Its base crop size is
336x336; this is **not** a forced 336x336 or 512x512 input resize. MMRet retains
its own 512x512 preprocessing and candidate prompt, unchanged.

WebQA candidates pass both the image and text through the same model call,
**not** separate encodings followed by vector addition. Embedded image bytes
and image paths are supported; relative paths are resolved below the WebQA
dataset directory. Its prompt is a documented **engineering extension**:

```text
[INST] <image>
{text}
Summary above image and text in one word: [/INST]
```

This is not an official published UniME WebQA template or a claim to reproduce
official WebQA scores. Generic `prompt_type`, `eol_type`, sparse and pooling
options in the legacy output hierarchy do not override the UniME templates or
last-token pooling. The inherited `--encode_is_query` flag is rejected because
it previously changed the filename without changing the candidate records.

## Output format and limitations

Each process writes `corpus_<rank>.pkl` containing:

```text
(embeddings: float32[N, 4096], original_candidate_ids: list[N])
```

The model supplies the embedding dimension (4096 for this checkpoint), and
each row corresponds to the ID at the same position. Empty rank shards retain
shape `(0, dimension)`. Shards are disjoint, including when the candidate count
is not divisible by the process count. Files are replaced atomically, and stale
extra shards are rejected when reducing the number of processes.

The existing hierarchy is preserved with the model basename instead of a
hard-coded path slice:

```text
<DENSE_OUTPUT_DIR>/<model_basename>/<dataset>/<encode_type>/
  <filter>/<calculate_type>/<prompt_type>/<split>/
  [<tbpr_type>/]<existing_parameter_suffix>/corpus_<rank>.pkl
```

The terminal prints the exact saved path, vector shape and per-rank progress.
Use a separate output directory when changing checkpoint contents under the
same model directory name. Online retrieval uses matching query embeddings
from the same checkpoint and template convention, as described below.

## Online dense search

```bash
DATASET=coco bash scripts/encode_dense_unime.sh
DATASET=coco bash scripts/search_dense_unime.sh
DATASET=flickr bash scripts/encode_dense_unime.sh
DATASET=flickr bash scripts/search_dense_unime.sh
DATASET=CUHK-PEDES bash scripts/encode_dense_unime.sh
DATASET=CUHK-PEDES bash scripts/search_dense_unime.sh
DATASET=ICFG-PEDES bash scripts/encode_dense_unime.sh
DATASET=ICFG-PEDES bash scripts/search_dense_unime.sh
DATASET=RSTPReid bash scripts/encode_dense_unime.sh
DATASET=RSTPReid bash scripts/search_dense_unime.sh
DATASET=webqa bash scripts/encode_dense_unime.sh
DATASET=webqa bash scripts/search_dense_unime.sh
DATASET=remuq bash scripts/encode_dense_unime.sh
DATASET=remuq bash scripts/search_dense_unime.sh
DATASET=llava bash scripts/encode_dense_unime.sh
DATASET=llava bash scripts/search_dense_unime.sh

# COCO/Flickr reverse direction: image query -> caption candidates.
DATASET=coco QUERY_TYPE=image bash scripts/encode_dense_unime.sh
DATASET=coco QUERY_TYPE=image bash scripts/search_dense_unime.sh
DATASET=flickr ENCODE_TYPE=text bash scripts/encode_dense_unime.sh
DATASET=flickr ENCODE_TYPE=text bash scripts/search_dense_unime.sh
```

Run one dataset per invocation, encoding before searching. Both standalone
scripts explicitly select UniME and default to its existing checkpoint under
`checkpoints/`; renamed local checkpoints only need `MODEL_NAME_OR_PATH`.
Use the **same checkpoint, data settings and path labels** for both commands.
The search script derives the matching index path from `DENSE_OUTPUT_DIR` using
the encoding hierarchy, including the candidate modality and `TBPR_TYPE` layer.
An explicit `PASSAGE_REPS` / `--passage_reps` takes priority and may identify an
index directory or a single corpus shard. Missing indexes fail before Python
starts. The generic `search_dense.sh` remains available for explicit index paths.

Both launchers derive task/query direction from `DATASET`; COCO/Flickr support
`QUERY_TYPE=text` against image candidates, or `image` against text candidates.
The three pedestrian datasets and WebQA use text queries; ReMuQ/LLaVA use both
query image and text (`QUERY_TYPE=image` or `composed`).
Set only `QUERY_TYPE` or `ENCODE_TYPE` to infer the other; incompatible explicit
directions are rejected. CLI overrides environment values; the last repeated
option wins, using either `--option value` or `--option=value`. Other Python
arguments (including generated-validation settings) are forwarded.

Pure text/image queries reuse the existing UniME summary templates and native
AnyRes image preprocessing. Joint queries use the existing `JOINT_PROMPT`;
this is a **project extension**, not an official published multimodal query
evaluation configuration. No MMRet task instruction is injected into UniME.

Set `DATA_ROOT`, `DATASET_SPLIT`, `DENSE_OUTPUT_DIR`, `BATCH_SIZE`, `NUM_PROCESSES`
(or `GPUS`), `MASTER_PORT` and `DTYPE` as needed; search also accepts `DEPTH`,
`SAVE_DIR` and `USE_GPU`. Defaults are Flickr/test, text queries, one process,
batch size 1 and FP16; use `DTYPE=float32` on CPU. Each worker searches the whole
index with disjoint queries. Rank 0 writes `dense.xlsx` to `SAVE_DIR` or
`search_results/<model_basename>/<dataset>/<query_type>/<split>/`.
Excel output requires pandas and `openpyxl` or `xlsxwriter`. The launcher does
not download models or install packages. The three-stage pipeline is unchanged.

Path-label overrides must be identical on both commands: `PROMPT_TYPE`,
`CALCULATE_TYPE`, `EOL_TYPE`, `TBPR_TYPE`, `NUM_EXPENDED_TOKENS`, `SPARSE_LENGTH`,
`SPARSE_VALUE_TYPE`, `REPS_LOC`, `SPARSE_LOWER_OR_UPPER`, `IS_FILTERED`,
`SPARSE_MANUAL`, `USE_OUTPUT_EMBEDDING_CLUSTER`, `CLUSTER_SUM`, `SPARSE_VALUE_MEAN`.
They are compatibility labels and do not change the native UniME templates.

```bash
PASSAGE_REPS=/custom/index SAVE_DIR=./results/unime bash scripts/search_dense_unime.sh
CUDA_VISIBLE_DEVICES=2,3 NUM_PROCESSES=2 DATASET=webqa bash scripts/encode_dense_unime.sh
CUDA_VISIBLE_DEVICES=2,3 NUM_PROCESSES=2 DATASET=webqa bash scripts/search_dense_unime.sh
```

Tests use miniature local LLaVA-NeXT checkpoints and dataset fixtures. They
verify input modalities/IDs, batching, AnyRes images with differing crop counts,
last-token pooling, old/new checkpoint conversion, output shards, CLI routing
and launch arguments. Full checkpoint/GPU and real DeepSpeed multi-GPU runs
must be tested separately on a machine with the weights and CUDA devices.
