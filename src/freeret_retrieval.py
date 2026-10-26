#!/usr/bin/env python3
"""A self-contained FreeRet search + reranking pipeline for this project.

This file intentionally does not import or modify encode.py, search.py, or
rerank.py.  It follows the data layouts used by dataset.py and implements the
two FreeRet stages directly:

1. controlled one-word embeddings from the hidden state immediately before the
   final decoder-layer MLP;
2. A/B multiple-choice reranking using the logits of the next token.

The implementation supports Qwen2.5-VL, LLaVA-NeXT, LLaVA-OneVision (OV),
and LLaVA-OneVision-1.5 (OV-1.5) checkpoints. A forward pre-hook on the last
decoder layer's post-attention layer norm captures the same pre-MLP attention
state exposed by the official FreeRet model. Qwen2.5-VL and OV-1.5 keep the
Qwen-style vision-input path; LLaVA-NeXT and OV use their native processor
and chat template.

Official MSCOCO and WebQA prompts are transcribed from
mmeb_evaluation/input_format_oneword.py.  Datasets absent from the official
prompt table have explicit, separately registered prompt adapters with their
provenance written to every run's prompt_manifest.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from collections import OrderedDict, defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import pandas as pd
import torch
import torch.nn as nn
try:
    from rerank_batching import indexed_batches, last_valid_logits
except ModuleNotFoundError as exc:
    if exc.name != "rerank_batching":
        raise
    from src.rerank_batching import indexed_batches, last_valid_logits
import torch.nn.functional as F
import transformers
from packaging.version import InvalidVersion, Version
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from tqdm.auto import tqdm
except ImportError as exc:  # pragma: no cover - depends on the runtime environment
    raise ImportError(
        "FreeRet progress visualization requires tqdm. Install it with `pip install tqdm`."
    ) from exc

process_vision_info: Any
try:
    from qwen_vl_utils import process_vision_info
except ImportError as exc:  # pragma: no cover - depends on the runtime image
    process_vision_info = None
    _QWEN_VL_UTILS_IMPORT_ERROR = exc

from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

try:
    from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
except ImportError:  # pragma: no cover - depends on the Transformers version
    LlavaNextForConditionalGeneration = None
    LlavaNextProcessor = None

try:
    from transformers import LlavaOnevisionForConditionalGeneration
except ImportError:  # pragma: no cover - depends on the Transformers version
    LlavaOnevisionForConditionalGeneration = None

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:  # pragma: no cover - compatibility with older Transformers
    try:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLForConditionalGeneration,
        )
    except ImportError:
        Qwen2_5_VLForConditionalGeneration = None


QWEN_MODEL_FAMILY = "qwen2_5_vl"
LLAVA_NEXT_MODEL_FAMILY = "llava_next"
LLAVA_ONEVISION_MODEL_FAMILY = "llava_onevision"
LLAVA_ONEVISION_1_5_MODEL_FAMILY = "llava_onevision_1_5"
OV15_TRANSFORMERS_VERSION = Version("4.53.0")
OV15_FORWARD_INPUT_KEYS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "position_ids",
        "past_key_values",
        "inputs_embeds",
        "labels",
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "rope_deltas",
        "cache_position",
    }
)


def _require_ov15_transformers_version(version_string: str) -> None:
    """Fail before model allocation when OV-1.5 uses an unsupported runtime.

    The released OV-1.5 remote implementation targets Transformers 4.53.0.
    Its configuration, RoPE implementation, and checkpoint conversion hooks
    are not compatible with Transformers 5.x.  Keeping this check strict is
    safer than silently constructing a model with missing decoder weights.
    """
    try:
        installed = Version(version_string)
    except InvalidVersion as exc:
        raise RuntimeError(
            "Cannot determine the Transformers version required by "
            f"LLaVA-OV-1.5: {version_string!r}."
        ) from exc

    if installed != OV15_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "LLaVA-OV-1.5 requires the official compatible runtime "
            f"transformers=={OV15_TRANSFORMERS_VERSION}; found "
            f"transformers=={version_string}. Create a dedicated environment "
            "with Transformers 4.53.0 and run this script with "
            "`PYTHON_BIN=/path/to/that/environment/bin/python "
            "bash scripts/freeret_retrieval.sh`."
        )


def _detect_model_family(model_path: str) -> str:
    """Detect one of the model families implemented by this FreeRet runner.

    The explicit path checks are intentional: they also work when a local
    checkpoint is being used without network access. ``AutoConfig`` covers
    arbitrary local aliases whose directory name does not contain the model
    name.
    """
    path_hint = str(model_path).replace("\\", "/").lower()
    # Check OV-1.5 before the broader OneVision patterns.  This is important
    # for local aliases such as ``LLaVA-OV-1.5`` where AutoConfig may not be
    # available offline.
    if (
        "llava-onevision-1.5" in path_hint
        or "llava-one-vision-1.5" in path_hint
        or "llava-ov-1.5" in path_hint
        or "llavaonevision1_5" in path_hint
    ):
        return LLAVA_ONEVISION_1_5_MODEL_FAMILY
    if (
        "llava-onevision" in path_hint
        or "llava-one-vision" in path_hint
        or "llava-ov" in path_hint
    ):
        return LLAVA_ONEVISION_MODEL_FAMILY
    if (
        "llava-v1.6-mistral-7b" in path_hint
        or "llava-mistral-7b" in path_hint
        or "llama3-llava-next-8b" in path_hint
        or "llava-llama-8b" in path_hint
    ):
        return LLAVA_NEXT_MODEL_FAMILY
    if "qwen2.5-vl" in path_hint or "qwen2_5_vl" in path_hint:
        return QWEN_MODEL_FAMILY

    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_type = str(getattr(config, "model_type", "")).lower()
    except Exception:
        model_type = ""

    if model_type in {
        "llavaonevision1_5",
        "llava_onevision1_5",
        "llava_onevision_1_5",
    }:
        return LLAVA_ONEVISION_1_5_MODEL_FAMILY
    if model_type in {"llava_onevision", "llava-onevision"}:
        return LLAVA_ONEVISION_MODEL_FAMILY
    if model_type in {"llava_next", "llava-next"}:
        return LLAVA_NEXT_MODEL_FAMILY
    if model_type in {"qwen2_5_vl", "qwen2.5_vl"}:
        return QWEN_MODEL_FAMILY

    raise ValueError(
        "Unsupported FreeRet model. This runner supports Qwen2.5-VL, "
        "LLaVA-NeXT, LLaVA-OV, and LLaVA-OV-1.5 checkpoints; "
        f"could not identify model family for {model_path!r}."
    )


PROMPT_VERSION = "freeret_official_dataset_prompts_v3"
IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"
CHAT_PREFIX = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
)
CHAT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

SUPPORTED_DATASETS = (
    "coco",
    "flickr",
    "CUHK-PEDES",
    "ICFG-PEDES",
    "RSTPReid",
    "webqa",
    "remuq",
    "llava",
)
IMAGE_DATASETS = {"coco", "flickr"}
TBPR_DATASETS = {"CUHK-PEDES", "ICFG-PEDES", "RSTPReid"}
T2IT_DATASETS = {"webqa"}
IT2T_DATASETS = {"remuq", "llava"}

DATASET_DIR_NAMES = {
    "coco": "coco",
    "flickr": "flickr",
    "CUHK-PEDES": "CUHK-PEDES",
    "ICFG-PEDES": "ICFG-PEDES",
    "RSTPReid": "RSTPReid",
    "webqa": "WebQA",
    "remuq": "UMRB-ReMuQ",
    "llava": "UMRB-LLaVA",
}
PERSON_JSON_NAMES = {
    "CUHK-PEDES": "reid_raw.json",
    "ICFG-PEDES": "ICFG-PEDES.json",
    "RSTPReid": "data_captions.json",
}


def normalize_id(value: Any) -> str:
    """Make parquet/CSV/JSON identifiers comparable across pandas and Python."""

    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def resolve_path(base: Path, value: Any) -> Path:
    path = Path(safe_text(value))
    return path if path.is_absolute() else base / path


@dataclass
class QueryItem:
    query_id: str
    text: str = ""
    image_path: Optional[str] = None
    relevant_ids: set[str] = field(default_factory=set)


@dataclass
class CandidateItem:
    candidate_id: str
    text: str = ""
    image_path: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FreeRetPromptSpec:
    """The six prompt components used by official FreeRet INPUT_FROMAT."""

    task_description: str
    target_input: Callable[[str], str]
    query_input: Callable[[str], str]
    target_instruction: str
    query_instruction: str
    rerank_input: Callable[[str, str], str]
    rerank_image_order: tuple[str, ...]
    max_pixels: int
    source: str


@dataclass
class RetrievalTask:
    dataset: str
    split: str
    direction: str
    prompt_key: str
    queries: list[QueryItem]
    candidates: list[CandidateItem]


class ImageMaterializer:
    """Materialize parquet image bytes for the model-specific processors."""

    def __init__(self, cache_dir: Path, data_root: Path):
        self.cache_dir = cache_dir
        self.data_root = data_root
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def materialize(self, value: Any, key: str) -> Optional[str]:
        if value is None:
            return None

        if isinstance(value, dict):
            for byte_key in ("bytes", "data"):
                if value.get(byte_key) is not None:
                    return self._write_bytes(value[byte_key], key)
            for path_key in ("path", "file_name", "filename"):
                if value.get(path_key):
                    return self._resolve_existing(value[path_key])

        if isinstance(value, Image.Image):
            digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
            target = self.cache_dir / f"{digest}.png"
            if not target.exists():
                value.convert("RGB").save(target, format="PNG")
            return str(target)

        if isinstance(value, (bytes, bytearray, memoryview)):
            return self._write_bytes(bytes(value), key)

        if hasattr(value, "as_py"):
            return self.materialize(value.as_py(), key)

        if isinstance(value, str) or isinstance(value, Path):
            return self._resolve_existing(value)

        return None

    def _resolve_existing(self, value: Any) -> str:
        path = Path(str(value))
        if not path.is_absolute():
            path = self.data_root / path
        return str(path)

    def _write_bytes(self, value: Any, key: str) -> str:
        raw = bytes(value)
        digest = hashlib.sha1(raw + key.encode("utf-8")).hexdigest()
        target = self.cache_dir / f"{digest}.img"
        if not target.exists():
            target.write_bytes(raw)
        return str(target)


def _dataset_dir(data_root: Path, dataset: str) -> Path:
    return data_root / "data" / DATASET_DIR_NAMES[dataset]


def load_csv_task(
    data_root: Path, dataset: str, split: str, direction: str
) -> RetrievalTask:
    base = _dataset_dir(data_root, dataset)
    csv_path = base / f"{dataset}_{split}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Cannot find split file: {csv_path}")

    images: OrderedDict[str, CandidateItem] = OrderedDict()
    texts: OrderedDict[str, CandidateItem] = OrderedDict()
    image_to_text: defaultdict[str, list[str]] = defaultdict(list)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        first_row = next(reader, None)
        if first_row is None:
            raise RuntimeError(f"CSV is empty: {csv_path}")
        rows: Iterable[list[str]]
        if first_row and first_row[0].strip().lower() == "imgid":
            rows = reader
        else:
            rows = _prepend_row(first_row, reader)

        for row_number, row in enumerate(rows):
            if not row:
                continue
            if dataset == "coco":
                if len(row) < 5:
                    raise ValueError(f"COCO row has fewer than 5 columns: {row}")
                image_id = normalize_id(row[0])
                image_dir = row[1]
                image_name = row[2]
                caption = row[3]
                text_id = normalize_id(row[4])
                image_path = resolve_path(base, Path(image_dir) / image_name)
            else:
                if len(row) < 4:
                    raise ValueError(f"Flickr row has fewer than 4 columns: {row}")
                image_id = normalize_id(row[0])
                image_name = row[1]
                caption = row[2]
                text_id = normalize_id(row[3])
                image_path = base / "flickr30k-images" / image_name

            if not text_id:
                text_id = f"{image_id}::caption::{row_number}"
            while text_id in texts:
                text_id = f"{text_id}::{row_number}"

            if image_id not in images:
                images[image_id] = CandidateItem(
                    candidate_id=image_id,
                    image_path=str(image_path),
                    metadata={"image_id": image_id},
                )
            texts[text_id] = CandidateItem(
                candidate_id=text_id,
                text=safe_text(caption),
                metadata={"image_id": image_id, "image_path": str(image_path)},
            )
            image_to_text[image_id].append(text_id)

    if direction == "t2i":
        queries = [
            QueryItem(
                query_id=text_id,
                text=item.text,
                relevant_ids={normalize_id(item.metadata["image_id"])},
            )
            for text_id, item in texts.items()
        ]
        candidates = list(images.values())
        return RetrievalTask(
            dataset, split, direction, f"{dataset}_t2i", queries, candidates
        )

    if direction == "i2t":
        queries = [
            QueryItem(
                query_id=image_id,
                image_path=item.image_path,
                relevant_ids=set(image_to_text[image_id]),
            )
            for image_id, item in images.items()
        ]
        candidates = list(texts.values())
        return RetrievalTask(
            dataset, split, direction, f"{dataset}_i2t", queries, candidates
        )

    raise ValueError(f"Unsupported CSV retrieval direction: {direction}")


def _prepend_row(first: list[str], reader: Iterable[list[str]]) -> Iterable[list[str]]:
    yield first
    yield from reader


def load_tbpr_task(data_root: Path, dataset: str, split: str) -> RetrievalTask:
    base = _dataset_dir(data_root, dataset)
    annotation_path = base / PERSON_JSON_NAMES[dataset]
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Cannot find annotation file: {annotation_path}")

    with annotation_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {annotation_path}")

    candidates: OrderedDict[str, CandidateItem] = OrderedDict()
    person_to_images: defaultdict[str, list[str]] = defaultdict(list)
    usable_records: list[tuple[dict[str, Any], str, str]] = []

    for item in records:
        if not isinstance(item, dict) or safe_text(item.get("split")) != split:
            continue
        relative = item.get("file_path") if dataset != "RSTPReid" else item.get("img_path")
        if not relative:
            continue
        relative = safe_text(relative).replace("\\", "/")
        candidate_id = f"{dataset}::{relative}"
        image_path = base / "imgs" / relative
        person_id = normalize_id(item.get("id"))
        if candidate_id not in candidates:
            candidates[candidate_id] = CandidateItem(
                candidate_id=candidate_id,
                image_path=str(image_path),
                metadata={"person_id": person_id, "relative_path": relative},
            )
            person_to_images[person_id].append(candidate_id)
        usable_records.append((item, candidate_id, person_id))

    queries: list[QueryItem] = []
    for item, candidate_id, person_id in usable_records:
        captions = item.get("captions", [])
        if isinstance(captions, str):
            captions = [captions]
        for caption_index, caption in enumerate(captions):
            query_id = f"{candidate_id}::caption::{caption_index}"
            queries.append(
                QueryItem(
                    query_id=query_id,
                    text=safe_text(caption),
                    relevant_ids=set(person_to_images[person_id]),
                )
            )

    prompt_key = {
        "CUHK-PEDES": "cuhk_pedes_t2i",
        "ICFG-PEDES": "icfg_pedes_t2i",
        "RSTPReid": "rstpreid_t2i",
    }[dataset]
    return RetrievalTask(
        dataset, split, "t2i", prompt_key, queries, list(candidates.values())
    )


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find parquet file: {path}")
    return pd.read_parquet(path)


def load_webqa_task(
    data_root: Path, materializer: ImageMaterializer
) -> RetrievalTask:
    base = _dataset_dir(data_root, "webqa")
    query_df = read_parquet(base / "query-00000-of-00001.parquet")
    rel_df = read_parquet(base / "qrels-00000-of-00001.parquet")
    corpus_paths = [
        base / "corpus-00000-of-00002.parquet",
        base / "corpus-00001-of-00002.parquet",
    ]

    query_to_targets: defaultdict[str, list[str]] = defaultdict(list)
    target_set: set[str] = set()
    for _, row in rel_df.iterrows():
        query_id = normalize_id(row["query-id"])
        corpus_id = normalize_id(row["corpus-id"])
        query_to_targets[query_id].append(corpus_id)
        target_set.add(corpus_id)

    candidates: list[CandidateItem] = []
    seen_candidates: set[str] = set()
    for corpus_path in corpus_paths:
        corpus_df = read_parquet(corpus_path)
        for _, row in corpus_df.iterrows():
            candidate_id = normalize_id(row["id"])
            if candidate_id not in target_set or candidate_id in seen_candidates:
                continue
            seen_candidates.add(candidate_id)
            image_path = materializer.materialize(
                row.get("image"), f"webqa_candidate_{candidate_id}"
            )
            candidates.append(
                CandidateItem(
                    candidate_id=candidate_id,
                    text=safe_text(row.get("text")),
                    image_path=image_path,
                )
            )

    queries: list[QueryItem] = []
    for _, row in query_df.iterrows():
        query_id = normalize_id(row["id"])
        queries.append(
            QueryItem(
                query_id=query_id,
                text=safe_text(row.get("text")),
                relevant_ids=set(query_to_targets.get(query_id, [])),
            )
        )

    return RetrievalTask(
        "webqa", "parquet", "t2it", "webqa_t2it", queries, candidates
    )


def load_it2t_task(
    data_root: Path, dataset: str, materializer: ImageMaterializer
) -> RetrievalTask:
    base = _dataset_dir(data_root, dataset)
    query_df = read_parquet(base / "query.parquet")
    rel_df = read_parquet(base / "qrels.parquet")
    corpus_df = read_parquet(base / "corpus.parquet")

    # dataset.py assigns one corpus-id per query and overwrites if a duplicate
    # qrels row appears.  Keep the same behavior for strict compatibility.
    query_to_target: dict[str, str] = {}
    for _, row in rel_df.iterrows():
        query_to_target[normalize_id(row["query-id"])] = normalize_id(row["corpus-id"])

    candidates: list[CandidateItem] = []
    for _, row in corpus_df.iterrows():
        candidates.append(
            CandidateItem(
                candidate_id=normalize_id(row["id"]),
                text=safe_text(row.get("text")),
            )
        )

    queries: list[QueryItem] = []
    for _, row in query_df.iterrows():
        query_id = normalize_id(row["id"])
        image_path = materializer.materialize(
            row.get("image"), f"{dataset}_query_{query_id}"
        )
        target = query_to_target.get(query_id)
        queries.append(
            QueryItem(
                query_id=query_id,
                text=safe_text(row.get("text")),
                image_path=image_path,
                relevant_ids={target} if target else set(),
            )
        )

    return RetrievalTask(
        dataset, "parquet", "it2t", f"{dataset}_it2t", queries, candidates
    )


def parse_dataset_names(value: str) -> list[str]:
    aliases = {name.lower(): name for name in SUPPORTED_DATASETS}
    aliases.update({"cuhk-pedes": "CUHK-PEDES", "icfg-pedes": "ICFG-PEDES"})
    names = [x.strip() for x in re.split(r"[,\s]+", value) if x.strip()]
    result: list[str] = []
    for name in names:
        if name.lower() == "all":
            for supported in SUPPORTED_DATASETS:
                if supported not in result:
                    result.append(supported)
            continue
        canonical = aliases.get(name.lower())
        if canonical is None:
            raise ValueError(f"Unknown dataset {name!r}; choices: {SUPPORTED_DATASETS}")
        if canonical not in result:
            result.append(canonical)
    if not result:
        raise ValueError("--datasets cannot be empty")
    return result


def directions_for(dataset: str, requested: str) -> list[str]:
    if requested == "auto":
        return ["t2i", "i2t"] if dataset in IMAGE_DATASETS else [
            "t2i" if dataset in TBPR_DATASETS else "t2it" if dataset in T2IT_DATASETS else "it2t"
        ]
    if requested == "both":
        if dataset not in IMAGE_DATASETS:
            raise ValueError(f"--direction both is only valid for coco/flickr, not {dataset}")
        return ["t2i", "i2t"]
    expected = (
        {"t2i", "i2t"}
        if dataset in IMAGE_DATASETS
        else {"t2i"}
        if dataset in TBPR_DATASETS
        else {"t2it"}
        if dataset in T2IT_DATASETS
        else {"it2t"}
    )
    if requested not in expected:
        raise ValueError(f"{dataset} only supports {sorted(expected)}, got {requested}")
    return [requested]


def prompt_key_for(dataset: str, direction: str) -> str:
    if dataset in IMAGE_DATASETS:
        return f"{dataset}_{direction}"
    if dataset == "CUHK-PEDES":
        return "cuhk_pedes_t2i"
    if dataset == "ICFG-PEDES":
        return "icfg_pedes_t2i"
    if dataset == "RSTPReid":
        return "rstpreid_t2i"
    if dataset == "webqa":
        return "webqa_t2it"
    if dataset in IT2T_DATASETS:
        return f"{dataset}_it2t"
    raise ValueError(f"No FreeRet prompt is registered for {dataset}/{direction}")


def load_task(
    data_root: Path,
    dataset: str,
    split: str,
    direction: str,
    materializer: ImageMaterializer,
) -> RetrievalTask:
    if dataset in IMAGE_DATASETS:
        return load_csv_task(data_root, dataset, split, direction)
    if dataset in TBPR_DATASETS:
        return load_tbpr_task(data_root, dataset, split)
    if dataset in T2IT_DATASETS:
        return load_webqa_task(data_root, materializer)
    if dataset in IT2T_DATASETS:
        return load_it2t_task(data_root, dataset, materializer)
    raise ValueError(dataset)


def _mscoco_t2i_spec(source: str) -> FreeRetPromptSpec:
    return FreeRetPromptSpec(
        task_description="You are required to determine if the provided caption matches the given image.",
        target_input=lambda _text: f"<photo>: {IMAGE_TOKEN}\n",
        query_input=lambda text: f"<caption>: {text}\n",
        target_instruction=(
            "Based on the given <image>, output exactly one word that follows these rules:\n"
            " - Output a single lowercase noun that best captures the semantics of the <image>.\n"
            " - You may use world knowledge and paraphrase if needed.\n"
            " - The word must be a complete, standalone word — not a function word, preposition, symbol, or fragment.\n"
            " - Favor the caption of the <image> if possible.\n"
        ),
        query_instruction=(
            "Based on the given <caption>, output exactly one word that follows these rules:\n"
            " - Output a single lowercase noun that best captures the semantics of the <caption>.\n"
            " - You may use world knowledge and paraphrasing if helpful.\n"
            " - The word must be a complete, standalone word — not a function word, preposition, symbol, or fragment.\n"
            " - Favor the word of the <caption> if possible.\n"
        ),
        rerank_input=lambda _target_text, query_text: (
            f"{IMAGE_TOKEN}\n"
            "Question: Does the caption accurately describe the image?\n"
            f"Caption: {query_text}\n"
            "A. Yes\n"
            "B. No"
        ),
        rerank_image_order=("target",),
        max_pixels=256,
        source=source,
    )


def _mscoco_i2t_spec(source: str) -> FreeRetPromptSpec:
    return FreeRetPromptSpec(
        task_description="You are required to determine if the provided caption matches the given image.",
        target_input=lambda text: f"<caption>: {text}\n",
        query_input=lambda _text: f"<photo>: {IMAGE_TOKEN}\n",
        target_instruction=(
            "Based on the given <caption>, output exactly one word that follows these rules:\n"
            " - Output a single lowercase noun that best captures the semantics of the <caption>.\n"
            " - You may use world knowledge and paraphrasing if helpful.\n"
            " - The word must be a complete, standalone word — not a function word, preposition, symbol, or fragment.\n"
            " - Favor the word of the <caption> if possible.\n"
        ),
        query_instruction=(
            "Based on the given <image>, output exactly one word that follows these rules:\n"
            " - Output a single lowercase noun that best captures the semantics of the <image>.\n"
            " - You may use world knowledge and paraphrase if needed.\n"
            " - The word must be a complete, standalone word — not a function word, preposition, symbol, or fragment.\n"
            " - Favor the caption of the <image> if possible.\n"
        ),
        rerank_input=lambda target_text, _query_text: (
            f"{IMAGE_TOKEN}\n"
            f"Caption: {target_text}\n"
            "Question: Does the caption accurately describe the image?\n"
            "A. Yes, the caption matches the image content.\n"
            "B. No, the caption does not match the image content."
        ),
        rerank_image_order=("query",),
        max_pixels=256,
        source=source,
    )


def _tbpr_spec(dataset_name: str) -> FreeRetPromptSpec:
    return FreeRetPromptSpec(
        task_description=(
            "You are required to determine whether the person description matches "
            "the person image."
        ),
        target_input=lambda _text: f"<person image>: {IMAGE_TOKEN}\n",
        query_input=lambda text: f"<person description>: '{text}'\n",
        target_instruction=(
            "Based on the given <person image>, output exactly one word that follows these rules:\n"
            " - Output a single lowercase noun or adjective that best captures the person's visible appearance.\n"
            " - Focus on identity-relevant clothing, color, carried objects, age, or gender cues.\n"
            " - Do not use function words, prepositions, symbols, or incomplete fragments.\n"
            " - Prefer the most distinctive visible attribute.\n"
        ),
        query_instruction=(
            "Based on the given <person description>, output exactly one word that follows these rules:\n"
            " - Output a single lowercase noun or adjective that best captures the described person's appearance.\n"
            " - Focus on identity-relevant clothing, color, carried objects, age, or gender cues.\n"
            " - Do not use function words, prepositions, symbols, or incomplete fragments.\n"
            " - Prefer a distinctive word directly provided in the description.\n"
        ),
        rerank_input=lambda _target_text, query_text: (
            "You are required to determine whether the person image matches the person description.\n"
            f"<person image>: {IMAGE_TOKEN}\n"
            f"<person description>: '{query_text}'\n\n"
            "Multiple Choice Options:\n"
            "A. Yes, the visible person matches the described identity and appearance.\n"
            "B. No, the visible person does not match the description."
        ),
        rerank_image_order=("target",),
        max_pixels=512,
        source=(
            f"Project adaptation for {dataset_name}; the official FreeRet repository "
            "does not provide a TBPR prompt."
        ),
    )


def _webqa_spec() -> FreeRetPromptSpec:
    return FreeRetPromptSpec(
        task_description=(
            "You are required to determine if the provided image and information is "
            "related to the question.\n"
        ),
        target_input=lambda text: (
            f"<image>: {IMAGE_TOKEN}\n<information>: '{text}'\n\n"
        ),
        query_input=lambda text: f"<question>: '{text}'\n\n",
        target_instruction=(
            "Given the image and accompanying information, output exactly one word that follows these rules:\n"
            " - The word should best capture the overall meaning of information.\n"
            " - World knowledge may be used if helpful.\n"
            " - The word must be a complete, standalone word — not a function word, preposition, symbol, or fragment.\n"
            " - Prefer using a word directly provided in the given information."
        ),
        query_instruction=(
            "Given the question, output exactly one word that follows these rules:\n"
            " - The word should best capture the meaning of the main subjects mentioned in the question\n"
            " - Do not answer the question itself.\n"
            " - World knowledge may be used if helpful.\n"
            " - The word must be a complete, standalone word — not a function word, preposition, symbol, or fragment.\n"
            " - Prefer using a word directly provided in the question."
        ),
        rerank_input=lambda target_text, query_text: (
            "You are required to determine if the provided image and the "
            "accompanying information are relevant to the given question.\n"
            f"<image>: {IMAGE_TOKEN}\n"
            f"<information>: '{target_text}'\n\n"
            f"Question:\n{query_text}\n\n"
            "Multiple Choice Options:\n"
            "A. Yes, the image and the information directly relate to the question "
            "and are sufficient to provide an appropriate answer.\n"
            "B. No, the image and the information are not relevant enough, or they "
            "fail to address the question properly."
        ),
        rerank_image_order=("target",),
        max_pixels=256,
        source="FreeRet INPUT_FROMAT['WebQA'] (raw project text fields substituted directly).",
    )


def _remuq_spec() -> FreeRetPromptSpec:
    return FreeRetPromptSpec(
        task_description=(
            "You are required to determine whether the candidate passage contains "
            "knowledge relevant to the question about the image.\n"
        ),
        target_input=lambda text: f"<candidate passage>: '{text}'\n\n",
        query_input=lambda text: (
            f"<image>: {IMAGE_TOKEN}\n<question>: '{text}'\n\n"
        ),
        target_instruction=(
            "Given the candidate passage, output exactly one word that follows these rules:\n"
            " - The word should best capture the main entity or knowledge subject of the passage.\n"
            " - World knowledge and paraphrasing may be used if helpful.\n"
            " - The word must be a complete, standalone word — not a function word, preposition, symbol, or fragment.\n"
            " - Prefer using a word directly provided in the passage."
        ),
        query_instruction=(
            "Given the image and question, output exactly one word that follows these rules:\n"
            " - The word should best capture the main subject whose knowledge is required by the question.\n"
            " - Do not answer the question itself.\n"
            " - World knowledge may be used if helpful.\n"
            " - The word must be a complete, standalone word — not a function word, preposition, symbol, or fragment.\n"
            " - Prefer using a word directly provided in the question when possible."
        ),
        rerank_input=lambda target_text, query_text: (
            "You are required to determine whether a candidate passage provides "
            "relevant knowledge for answering a question about an image.\n"
            f"<image>: {IMAGE_TOKEN}\n"
            f"<question>: '{query_text}'\n\n"
            f"<candidate passage>: '{target_text}'\n\n"
            "Multiple Choice Options:\n"
            "A. Yes, the passage provides directly relevant knowledge for answering the question.\n"
            "B. No, the passage is irrelevant or does not provide the knowledge needed to answer the question."
        ),
        rerank_image_order=("query",),
        max_pixels=512,
        source=(
            "Project adaptation for UMRB-ReMuQ; the official FreeRet repository does "
            "not provide a ReMuQ prompt."
        ),
    )


def _llava_spec() -> FreeRetPromptSpec:
    return FreeRetPromptSpec(
        task_description=(
            "You are required to determine whether the provided answer correctly "
            "responds to the given question, based on the provided image."
        ),
        target_input=lambda text: f"<answer>: {text}\n",
        query_input=lambda text: (
            f"<image>: {IMAGE_TOKEN}\n<question>: {text}\n"
        ),
        target_instruction=(
            "Represent the given <answer> in one word following these rules:\n"
            " - Output a single word that best captures the semantics of the <answer>.\n"
            " - You may use world knowledge and paraphrase if needed.\n"
            " - Do not repeat function words, use prepositions, or symbols.\n"
            " - Favor using original words from <answer>.\n"
        ),
        query_instruction=(
            "Represent the given <image> and <question> in one word following these rules:\n"
            " - Output a single word that best captures the semantics of the <question>.\n"
            " - You may use world knowledge and paraphrasing if helpful.\n"
            " - Do not use function words, prepositions, or symbols.\n"
            " - Favor the answer of the <question>.\n"
        ),
        rerank_input=lambda target_text, query_text: (
            f"{IMAGE_TOKEN}\n"
            "You are given a question and a proposed answer.\n"
            "Your task is to evaluate whether the proposed answer correctly addresses the question.\n"
            f"Choose the best option below.\n\nQuestion:\n{query_text}\n\n"
            f"Proposed Answer:\n{target_text}\n\n"
            "Multiple Choice Options:\n"
            "A. The proposed answer is correct and complete.\n"
            "B. The proposed answer is partially correct, or incorrect, or irrelevant to the question."
        ),
        rerank_image_order=("query",),
        max_pixels=512,
        source=(
            "Project adaptation for UMRB-LLaVA using the official FreeRet "
            "OK-VQA image-question-to-answer prompt family."
        ),
    )


PROMPT_SPECS: dict[str, FreeRetPromptSpec] = {
    "coco_t2i": _mscoco_t2i_spec("FreeRet INPUT_FROMAT['MSCOCO_t2i']."),
    "coco_i2t": _mscoco_i2t_spec("FreeRet INPUT_FROMAT['MSCOCO_i2t']."),
    "flickr_t2i": _mscoco_t2i_spec(
        "Flickr adaptation of FreeRet INPUT_FROMAT['MSCOCO_t2i']; Flickr is absent from the official prompt table."
    ),
    "flickr_i2t": _mscoco_i2t_spec(
        "Flickr adaptation of FreeRet INPUT_FROMAT['MSCOCO_i2t']; Flickr is absent from the official prompt table."
    ),
    "cuhk_pedes_t2i": _tbpr_spec("CUHK-PEDES"),
    "icfg_pedes_t2i": _tbpr_spec("ICFG-PEDES"),
    "rstpreid_t2i": _tbpr_spec("RSTPReid"),
    "webqa_t2it": _webqa_spec(),
    "remuq_it2t": _remuq_spec(),
    "llava_it2t": _llava_spec(),
}


def get_prompt_spec(prompt_key: str) -> FreeRetPromptSpec:
    try:
        return PROMPT_SPECS[prompt_key]
    except KeyError as exc:
        raise ValueError(f"Unknown FreeRet prompt key: {prompt_key}") from exc


def prompt_manifest(prompt_key: str) -> dict[str, Any]:
    spec = get_prompt_spec(prompt_key)
    return {
        "prompt_key": prompt_key,
        "prompt_version": PROMPT_VERSION,
        "source": spec.source,
        "max_pixels": spec.max_pixels,
        "task_description": spec.task_description,
        "stage1_target": spec.target_input("{target_text}")
        + spec.target_instruction,
        "stage1_query": spec.query_input("{query_text}") + spec.query_instruction,
        "stage2_rerank": spec.rerank_input("{target_text}", "{query_text}"),
        "rerank_image_order": list(spec.rerank_image_order),
    }


def one_word_prompt(
    prompt_key: str, role: str, item: QueryItem | CandidateItem
) -> str:
    spec = get_prompt_spec(prompt_key)
    text = safe_text(item.text)
    if role == "query":
        body = spec.query_input(text) + spec.query_instruction
    elif role == "target":
        body = spec.target_input(text) + spec.target_instruction
    else:
        raise ValueError(f"Unknown FreeRet prompt role: {role}")
    expected_images = body.count(IMAGE_TOKEN)
    actual_images = 1 if item.image_path else 0
    if expected_images != actual_images:
        item_id = (
            item.query_id if isinstance(item, QueryItem) else item.candidate_id
        )
        raise ValueError(
            f"Prompt {prompt_key!r} role {role!r} item {item_id!r} requires "
            f"{expected_images} image(s), but received {actual_images}."
        )
    return CHAT_PREFIX + spec.task_description + body + CHAT_SUFFIX


def rerank_prompt(
    prompt_key: str, query: QueryItem, candidate: CandidateItem
) -> tuple[str, list[str]]:
    spec = get_prompt_spec(prompt_key)
    image_paths: list[str] = []
    for image_role in spec.rerank_image_order:
        path = candidate.image_path if image_role == "target" else query.image_path
        if not path:
            raise ValueError(
                f"Prompt {prompt_key!r} requires a {image_role} image for "
                f"reranking query {query.query_id!r} with candidate "
                f"{candidate.candidate_id!r}."
            )
        image_paths.append(path)
    body = spec.rerank_input(safe_text(candidate.text), safe_text(query.text))
    return CHAT_PREFIX + body + CHAT_SUFFIX, image_paths


class FreeRetQwen:
    """FreeRet model wrapper for the supported Qwen and LLaVA checkpoints.

    The class keeps its historical name because the rest of this standalone
    runner uses it as the model interface.  The model family is detected from
    the checkpoint configuration (with explicit support for the Qwen2.5-VL,
    LLaVA-NeXT, LLaVA-OV, and LLaVA-OV-1.5 families used by this project).
    """

    def __init__(
        self,
        model_path: str,
        dtype: str,
        device_map: str,
        attn_implementation: str,
        max_pixels: int,
    ):
        self.model_family = _detect_model_family(model_path)
        self.transformers_version = transformers.__version__
        if self.model_family == LLAVA_ONEVISION_1_5_MODEL_FAMILY:
            _require_ov15_transformers_version(self.transformers_version)

        if (
            self.model_family == LLAVA_NEXT_MODEL_FAMILY
            and (
                LlavaNextForConditionalGeneration is None
                or LlavaNextProcessor is None
            )
        ):
            raise ImportError(
                "LLaVA-NeXT support requires a Transformers version exposing "
                "LlavaNextProcessor and LlavaNextForConditionalGeneration."
            )
        if (
            self.model_family == LLAVA_ONEVISION_MODEL_FAMILY
            and LlavaOnevisionForConditionalGeneration is None
        ):
            raise ImportError(
                "LLaVA-OV support requires a Transformers version exposing "
                "LlavaOnevisionForConditionalGeneration."
            )
        if self.model_family == QWEN_MODEL_FAMILY and process_vision_info is None:
            raise ImportError(
                "Qwen2.5-VL FreeRet runs require qwen-vl-utils. Install it in the "
                "same environment used by the project."
            ) from _QWEN_VL_UTILS_IMPORT_ERROR
        if (
            self.model_family == LLAVA_ONEVISION_1_5_MODEL_FAMILY
            and process_vision_info is None
        ):
            raise ImportError(
                "LLaVA-OV-1.5 FreeRet runs require qwen-vl-utils for the "
                "model's Qwen-style image processor inputs. Install it in the "
                "same environment used by the project."
            ) from _QWEN_VL_UTILS_IMPORT_ERROR
        if self.model_family == QWEN_MODEL_FAMILY and Qwen2_5_VLForConditionalGeneration is None:
            raise ImportError(
                "Qwen2.5-VL support requires a Transformers version exposing "
                "Qwen2_5_VLForConditionalGeneration."
            )

        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": None,
        }[dtype]
        if self.model_family == QWEN_MODEL_FAMILY:
            processor_kwargs = {
                "padding_side": "left",
                "min_pixels": 4 * 28 * 28,
                "max_pixels": max_pixels * 28 * 28,
            }
            self.processor = AutoProcessor.from_pretrained(
                model_path, **processor_kwargs
            )
            model_class = Qwen2_5_VLForConditionalGeneration
        elif self.model_family == LLAVA_NEXT_MODEL_FAMILY:
            self.processor = LlavaNextProcessor.from_pretrained(model_path)
            model_class = LlavaNextForConditionalGeneration
        elif self.model_family == LLAVA_ONEVISION_MODEL_FAMILY:
            self.processor = AutoProcessor.from_pretrained(model_path)
            model_class = LlavaOnevisionForConditionalGeneration
        elif self.model_family == LLAVA_ONEVISION_1_5_MODEL_FAMILY:
            # Follow the checkpoint's official remote-code loading path. Its
            # config registers the same conditional-generation implementation
            # for AutoModel and AutoModelForCausalLM; the latter states the
            # intended interface explicitly and guarantees top-level logits.
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            model_class = AutoModelForCausalLM
        else:  # pragma: no cover - _detect_model_family is exhaustive
            raise ValueError(f"Unsupported model family: {self.model_family}")

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
            if (
                getattr(tokenizer, "pad_token_id", None) is None
                and getattr(tokenizer, "eos_token_id", None) is not None
            ):
                tokenizer.pad_token = tokenizer.eos_token

        self.max_pixels = max_pixels
        self.max_pixels_applied = False
        self._reported_dropped_input_keys: set[str] = set()

        effective_attn_implementation = attn_implementation
        if self.model_family == LLAVA_ONEVISION_1_5_MODEL_FAMILY:
            if attn_implementation != "sdpa":
                print(
                    "[warning] LLaVA-OV-1.5 uses the official SDPA-compatible "
                    f"path; overriding attn_implementation={attn_implementation!r} "
                    "with 'sdpa'.",
                    file=sys.stderr,
                )
            effective_attn_implementation = "sdpa"

        load_kwargs: dict[str, Any] = {}
        if torch_dtype is not None:
            try:
                installed_transformers = Version(self.transformers_version)
            except InvalidVersion:
                installed_transformers = Version("0")
            dtype_key = (
                "dtype" if installed_transformers.major >= 5 else "torch_dtype"
            )
            load_kwargs[dtype_key] = torch_dtype
        if device_map != "none":
            load_kwargs["device_map"] = device_map
        if effective_attn_implementation != "auto":
            load_kwargs["attn_implementation"] = effective_attn_implementation
        if self.model_family == LLAVA_ONEVISION_1_5_MODEL_FAMILY:
            load_kwargs["trust_remote_code"] = True
            # The official remote class owns _checkpoint_conversion_mapping.
            # Loading information is requested only to ensure that mapping
            # actually populated the text decoder before retrieval begins.
            load_kwargs["output_loading_info"] = True

        try:
            self.model = self._load_pretrained_model(
                model_class, model_path, load_kwargs
            )
        except (ImportError, ModuleNotFoundError, ValueError, RuntimeError) as exc:
            if effective_attn_implementation != "flash_attention_2":
                raise
            print(
                f"[warning] flash_attention_2 unavailable ({exc}); retrying with sdpa.",
                file=sys.stderr,
            )
            load_kwargs.pop("attn_implementation", None)
            load_kwargs["attn_implementation"] = "sdpa"
            self.model = self._load_pretrained_model(
                model_class, model_path, load_kwargs
            )
            effective_attn_implementation = "sdpa"

        if self.model_family == LLAVA_NEXT_MODEL_FAMILY:
            self._configure_llava_next_processor()

        self.model.eval()
        self.attn_implementation = effective_attn_implementation
        self.model_class_name = type(self.model).__name__
        self.processor_class_name = type(self.processor).__name__
        self.device = self._model_device()
        self._attn_hidden: Optional[torch.Tensor] = None

        decoder_norm_path, decoder_norm = self._resolve_last_decoder_norm()
        self._decoder_norm_path = decoder_norm_path
        self._attn_hook_handle = decoder_norm.register_forward_pre_hook(
            self._capture_attention_hidden
        )

        self._prune_lm_head_to_ab()

    def _configure_llava_next_processor(self) -> None:
        """Populate vision metadata needed for native LLaVA-NeXT processing."""
        config = getattr(self.model, "config", None)
        vision_config = getattr(config, "vision_config", None)
        patch_size = getattr(vision_config, "patch_size", None)
        vision_feature_select_strategy = getattr(
            config, "vision_feature_select_strategy", None
        )

        if getattr(self.processor, "patch_size", None) is None and patch_size is not None:
            self.processor.patch_size = patch_size
        if (
            getattr(self.processor, "vision_feature_select_strategy", None) is None
            and vision_feature_select_strategy is not None
        ):
            self.processor.vision_feature_select_strategy = (
                vision_feature_select_strategy
            )

    def _load_pretrained_model(
        self,
        model_class: Any,
        model_path: str,
        load_kwargs: dict[str, Any],
    ) -> Any:
        loaded = model_class.from_pretrained(model_path, **load_kwargs)
        if self.model_family != LLAVA_ONEVISION_1_5_MODEL_FAMILY:
            return loaded

        if not isinstance(loaded, tuple) or len(loaded) != 2:
            raise RuntimeError(
                "LLaVA-OV-1.5 did not return loading information. Refusing to "
                "continue without checking whether decoder weights were loaded."
            )

        model, loading_info = loaded
        self._validate_ov15_loading_info(loading_info)
        return model

    @staticmethod
    def _validate_ov15_loading_info(loading_info: Any) -> None:
        if not isinstance(loading_info, dict):
            raise RuntimeError(
                "LLaVA-OV-1.5 returned an invalid loading report; cannot verify "
                "checkpoint key conversion."
            )

        missing_keys = [str(key) for key in loading_info.get("missing_keys", [])]
        unexpected_keys = [
            str(key) for key in loading_info.get("unexpected_keys", [])
        ]
        missing_text_keys = [
            key
            for key in missing_keys
            if key.startswith("model.language_model.layers.")
            or key.startswith("model.language_model.embed_tokens.")
            or key.startswith("model.language_model.norm.")
        ]
        unexpected_checkpoint_text_keys = [
            key
            for key in unexpected_keys
            if key.startswith("model.layers.")
            or key.startswith("model.embed_tokens.")
            or key.startswith("model.norm.")
        ]

        if missing_text_keys or unexpected_checkpoint_text_keys:
            missing_preview = ", ".join(missing_text_keys[:5]) or "none"
            unexpected_preview = (
                ", ".join(unexpected_checkpoint_text_keys[:5]) or "none"
            )
            raise RuntimeError(
                "LLaVA-OV-1.5 decoder checkpoint mapping was not applied. "
                f"missing target keys: {missing_preview}; "
                f"unexpected checkpoint keys: {unexpected_preview}."
            )

    def _resolve_last_decoder_norm(self) -> tuple[str, nn.Module]:
        """Find the norm immediately before the final decoder MLP.

        Qwen2.5-VL, LLaVA-NeXT, and the two OneVision families have slightly
        different nesting depending on the Transformers version and the
        underlying text model. FreeRet needs the input to
        ``post_attention_layernorm`` because it is the attention-residual
        state before the final decoder MLP.
        """
        model_base = getattr(self.model, "model", None)
        model_language = getattr(model_base, "language_model", None)
        top_language = getattr(self.model, "language_model", None)
        candidates = (
            (
                "model.model.language_model.layers[-1].post_attention_layernorm",
                model_language,
            ),
            (
                "model.model.language_model.model.layers[-1].post_attention_layernorm",
                getattr(model_language, "model", None),
            ),
            (
                "model.model.layers[-1].post_attention_layernorm",
                model_base,
            ),
            (
                "model.language_model.layers[-1].post_attention_layernorm",
                top_language,
            ),
            (
                "model.language_model.model.layers[-1].post_attention_layernorm",
                getattr(top_language, "model", None),
            ),
        )

        for path, decoder in candidates:
            layers = getattr(decoder, "layers", None)
            if layers is None:
                continue
            try:
                if len(layers) == 0:
                    continue
                last_layer = layers[-1]
            except (IndexError, TypeError):
                continue

            decoder_norm = getattr(last_layer, "post_attention_layernorm", None)
            if isinstance(decoder_norm, nn.Module):
                return path, decoder_norm

        model_type = type(self.model).__name__
        base_type = type(model_base).__name__ if model_base is not None else "None"
        attempted_paths = ", ".join(path for path, _ in candidates)
        raise RuntimeError(
            f"The loaded {self.model_family} model does not expose a decoder "
            "norm at "
            f"any supported path. model={model_type}, base_model={base_type}; "
            f"tried: {attempted_paths}."
        )

    def set_max_pixels(self, max_pixels: int) -> None:
        """Apply the per-dataset image-pixel setting when the processor supports it.

        Qwen2.5-VL and OV-1.5 expose Qwen-style image preprocessing. If the
        loaded processor exposes ``min_pixels``/``max_pixels``, those controls
        are applied; otherwise (as for the native CLIP-style OV and
        LLaVA-NeXT processors) the model's native preprocessing is retained.
        """
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is None:
            raise RuntimeError("The loaded processor has no image_processor.")
        self.max_pixels_applied = False
        if self.model_family == QWEN_MODEL_FAMILY:
            image_processor.min_pixels = 4 * 28 * 28
            image_processor.max_pixels = max_pixels * 28 * 28
            self.max_pixels_applied = True
        elif hasattr(image_processor, "min_pixels") and hasattr(
            image_processor, "max_pixels"
        ):
            image_processor.min_pixels = 4 * 28 * 28
            image_processor.max_pixels = max_pixels * 28 * 28
            self.max_pixels_applied = True
        self.max_pixels = max_pixels

    def _model_device(self) -> torch.device:
        try:
            return self.model.device
        except AttributeError:
            return next(self.model.parameters()).device

    def _capture_attention_hidden(self, _module: nn.Module, inputs: tuple[Any, ...]) -> None:
        if inputs:
            self._attn_hidden = inputs[0].detach().clone()

    def _single_token_id(self, token: str) -> int:
        encoded = self.processor.tokenizer(token, add_special_tokens=False)
        ids = encoded.input_ids
        if len(ids) != 1:
            raise RuntimeError(
                f"Tokenizer must map {token!r} to exactly one token without "
                f"special tokens, but returned {ids!r}."
            )
        return int(ids[0])

    def _prune_lm_head_to_ab(self) -> None:
        token_a = self._single_token_id("A")
        token_b = self._single_token_id("B")
        if token_a == token_b:
            raise RuntimeError("The tokenizer maps A and B to the same token ID.")
        old_head = self.model.lm_head
        new_head = nn.Linear(old_head.in_features, 2, bias=False)
        new_head = new_head.to(device=old_head.weight.device, dtype=old_head.weight.dtype)
        with torch.no_grad():
            new_head.weight.copy_(old_head.weight[[token_a, token_b]])
        self.model.lm_head = new_head

    def _llava_reserved_image_tokens(self) -> tuple[str, ...]:
        tokens = {"<image>"}
        tokenizer = getattr(self.processor, "tokenizer", None)
        for owner in (self.processor, tokenizer):
            token = getattr(owner, "image_token", None)
            if isinstance(token, str) and token:
                tokens.add(token)
        return tuple(sorted(tokens, key=len, reverse=True))

    def _llava_prompt_body(self, text: str) -> str:
        """Convert this runner's Qwen-shaped prompt into plain LLaVA text."""
        body = text
        if body.startswith(CHAT_PREFIX):
            body = body[len(CHAT_PREFIX) :]
        if body.endswith(CHAT_SUFFIX):
            body = body[: -len(CHAT_SUFFIX)]
        body = body.replace(IMAGE_TOKEN, "")
        for token in self._llava_reserved_image_tokens():
            body = body.replace(token, "image")
        return body.strip()

    def _validate_llava_rendered_image_counts(
        self,
        prompts: list[str],
        image_paths: list[list[str]],
    ) -> None:
        image_tokens = self._llava_reserved_image_tokens()
        for index, (prompt, paths) in enumerate(zip(prompts, image_paths)):
            token_count = sum(prompt.count(token) for token in image_tokens)
            if token_count != len(paths):
                raise RuntimeError(
                    "LLaVA prompt/image mismatch at batch item "
                    f"{index}: rendered prompt contains {token_count} image "
                    f"token(s), but {len(paths)} image path(s) were supplied. "
                    "FreeRet prompt text must not contain reserved LLaVA image "
                    "tokens; actual image tokens are inserted by the chat template."
                )

    @staticmethod
    def _validate_llava_message_image_counts(
        messages: list[list[dict[str, Any]]],
        image_paths: list[list[str]],
    ) -> None:
        for index, (conversation, paths) in enumerate(zip(messages, image_paths)):
            message_image_count = 0
            for message in conversation:
                content = message.get("content", [])
                if not isinstance(content, list):
                    continue
                message_image_count += sum(
                    1
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "image"
                )
            if message_image_count != len(paths):
                raise RuntimeError(
                    "LLaVA message/image mismatch at batch item "
                    f"{index}: the multimodal message contains "
                    f"{message_image_count} image item(s), but {len(paths)} "
                    "image path(s) were supplied."
                )

    def _prepare_inputs(
        self, texts: list[str], image_paths: list[list[str]]
    ) -> Any:
        if len(texts) != len(image_paths):
            raise ValueError(
                "The number of prompt strings must match the number of image-path "
                "lists."
            )

        if self.model_family == QWEN_MODEL_FAMILY:
            return self._prepare_qwen_inputs(texts, image_paths)
        if self.model_family == LLAVA_ONEVISION_MODEL_FAMILY:
            return self._prepare_llava_onevision_inputs(texts, image_paths)
        if self.model_family == LLAVA_ONEVISION_1_5_MODEL_FAMILY:
            return self._prepare_llava_onevision_1_5_inputs(texts, image_paths)
        if self.model_family == LLAVA_NEXT_MODEL_FAMILY:
            return self._prepare_llava_next_inputs(texts, image_paths)
        raise RuntimeError(f"Unsupported model family: {self.model_family}")

    def _prepare_qwen_inputs(
        self, texts: list[str], image_paths: list[list[str]]
    ) -> Any:
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": path} for path in paths
                    ],
                }
            ]
            for paths in image_paths
        ]
        vision_processor: Any = globals().get("process_vision_info")
        if not callable(vision_processor):
            raise RuntimeError("qwen-vl-utils is unavailable in the active environment.")
        # noinspection PyCallingNonCallable
        image_inputs, video_inputs = vision_processor(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )
        return inputs.to(self.device)

    def _llava_messages(
        self,
        texts: list[str],
        image_sources: Sequence[Sequence[Any]],
        image_source_key: Optional[str],
    ) -> list[list[dict[str, Any]]]:
        messages: list[list[dict[str, Any]]] = []
        for text, sources in zip(texts, image_sources):
            content: list[dict[str, Any]] = []
            for source in sources:
                image_item: dict[str, Any] = {"type": "image"}
                if image_source_key is not None:
                    image_item[image_source_key] = source
                content.append(image_item)
            content.append({"type": "text", "text": self._llava_prompt_body(text)})
            messages.append([{"role": "user", "content": content}])
        return messages

    def _chat_template(self) -> Callable[..., Any]:
        apply_chat_template = getattr(self.processor, "apply_chat_template", None)
        if not callable(apply_chat_template):
            raise RuntimeError("The LLaVA-family processor has no chat template.")
        return apply_chat_template

    @staticmethod
    def _chat_template_processing_kwargs(
        apply_chat_template: Callable[..., Any],
    ) -> dict[str, Any]:
        """Return padding kwargs for both legacy and modern ProcessorMixin APIs."""
        try:
            parameters = inspect.signature(apply_chat_template).parameters
        except (TypeError, ValueError):
            parameters = {}

        text_kwargs = {"padding": True, "padding_side": "left"}
        if "processor_kwargs" in parameters:
            return {"processor_kwargs": {"text_kwargs": text_kwargs}}
        return text_kwargs

    def _prepare_llava_onevision_inputs(
        self, texts: list[str], image_paths: list[list[str]]
    ) -> Any:
        """Use native OV multimodal templates without flattening image groups."""
        messages = self._llava_messages(texts, image_paths, image_source_key="path")
        apply_chat_template = self._chat_template()
        inputs = apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **self._chat_template_processing_kwargs(apply_chat_template),
        )
        return inputs.to(self.device)

    def _prepare_llava_onevision_1_5_inputs(
        self, texts: list[str], image_paths: list[list[str]]
    ) -> Any:
        """Follow the OV-1.5 repository's Qwen-style vision-input example."""
        messages = self._llava_messages(texts, image_paths, image_source_key="image")
        apply_chat_template = self._chat_template()
        prompts = [
            apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
            for message in messages
        ]

        vision_processor: Any = globals().get("process_vision_info")
        if not callable(vision_processor):
            raise RuntimeError(
                "qwen-vl-utils is unavailable in the active environment; "
                "it is required for LLaVA-OV-1.5 image inputs."
            )
        # noinspection PyCallingNonCallable
        image_inputs, video_inputs = vision_processor(messages)
        inputs = self.processor(
            text=prompts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )
        self._filter_ov15_forward_inputs(inputs)
        return inputs.to(self.device)

    def _filter_ov15_forward_inputs(self, inputs: Any) -> None:
        """Drop processor metadata absent from the released OV-1.5 forward."""
        dropped = {str(key) for key in inputs.keys()} - OV15_FORWARD_INPUT_KEYS
        for key in dropped:
            inputs.pop(key, None)
        unreported = dropped - self._reported_dropped_input_keys
        if unreported:
            print(
                "[input] Dropped processor fields unsupported by LLaVA-OV-1.5: "
                + ", ".join(sorted(unreported)),
                file=sys.stderr,
            )
            self._reported_dropped_input_keys.update(unreported)

    def _prepare_llava_next_inputs(
        self, texts: list[str], image_paths: list[list[str]]
    ) -> Any:
        image_counts = {len(paths) for paths in image_paths}
        if len(image_counts) > 1:
            raise RuntimeError(
                "A LLaVA-NeXT batch must contain the same number of images per "
                f"sample, but received counts {sorted(image_counts)}. Reduce the "
                "query or target batch size for mixed-image batches."
            )

        # Load local files through Pillow before handing them to the processor.
        # Modern Transformers routes string paths through torchvision.decode_image,
        # which does not support BMP even though Pillow does.  CUHK-PEDES contains
        # BMP candidates, so normalizing every image to an in-memory RGB image also
        # gives all LLaVA-NeXT inputs one consistent mode.
        image_inputs = [
            [self._load_llava_next_image(path, batch_index) for path in paths]
            for batch_index, paths in enumerate(image_paths)
        ]

        # Keep each image attached to its own conversation.  Flattening all PIL
        # images into one global list makes modern ProcessorMixin versions lose
        # the sample-to-image mapping and can exhaust their replacement iterator.
        messages = self._llava_messages(texts, image_inputs, image_source_key="image")
        self._validate_llava_message_image_counts(messages, image_paths)
        apply_chat_template = self._chat_template()
        prompts = [
            apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
            for message in messages
        ]
        self._validate_llava_rendered_image_counts(prompts, image_paths)

        try:
            inputs = apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                **self._chat_template_processing_kwargs(apply_chat_template),
            )
        except StopIteration as exc:
            rendered_counts = [
                sum(prompt.count(token) for token in self._llava_reserved_image_tokens())
                for prompt in prompts
            ]
            path_counts = [len(paths) for paths in image_paths]
            raise RuntimeError(
                "LLaVA-NeXT processor exhausted its image replacements while "
                "tokenizing a multimodal batch. "
                f"rendered_image_counts={rendered_counts}, "
                f"image_path_counts={path_counts}, "
                f"processor={type(self.processor).__name__}, "
                f"transformers={self.transformers_version}."
            ) from exc
        return inputs.to(self.device)

    @staticmethod
    def _load_llava_next_image(image_path: str, batch_index: int) -> Image.Image:
        """Load a local LLaVA-NeXT image as a detached RGB Pillow image."""
        path = Path(image_path)
        try:
            with Image.open(path) as source:
                source.load()
                rgb_image = ImageOps.exif_transpose(source).convert("RGB")
                return rgb_image.copy()
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            suffix = path.suffix.lower() or "<none>"
            raise RuntimeError(
                "Unable to decode a LLaVA-NeXT image at batch item "
                f"{batch_index}: path={path!s}, suffix={suffix!r}, "
                f"error={exc}"
            ) from exc

    @torch.inference_mode()
    def encode_one_word(
        self, texts: list[str], image_paths: list[list[str]]
    ) -> torch.Tensor:
        inputs = self._prepare_inputs(texts, image_paths)
        self._attn_hidden = None
        self.model(
            **inputs,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        if self._attn_hidden is None:
            raise RuntimeError("The last-layer attention hook did not capture a state.")
        features = self._attn_hidden[:, -1].float()
        features = F.normalize(features, dim=-1)
        self._attn_hidden = None
        del inputs
        return features.cpu()

    @torch.inference_mode()
    def rerank(
        self, texts: list[str], image_paths: list[list[str]]
    ) -> torch.Tensor:
        if not texts:
            if image_paths:
                raise ValueError("Prompt and image-path batch sizes differ")
            return torch.empty(0)
        inputs = self._prepare_inputs(texts, image_paths)
        try:
            outputs = self.model(
                **inputs,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            logits = last_valid_logits(outputs.logits, inputs['attention_mask']).float()
            scores = F.softmax(logits, dim=-1)[:, 0]
            return scores.cpu()
        finally:
            # The embedding hook also fires during reranking. Do not keep a
            # full batch's hidden tensor alive after scoring (or after an OOM).
            self._attn_hidden = None


try:
    from time_estimation import StableRateEstimator, StageTimeObserver, TorchStageCollectives
except ModuleNotFoundError as exc:  # importlib-based project-root callers
    if exc.name != "time_estimation":
        raise
    from src.time_estimation import StableRateEstimator, StageTimeObserver, TorchStageCollectives


ACTUAL_TIMING_COMPONENTS = (
    "candidate_encoding_seconds", "query_encoding_seconds",
    "similarity_search_seconds", "reranking_seconds",
    "cache_read_seconds", "cache_write_seconds", "target_transfer_seconds",
    "distributed_communication_seconds",
)
ACTUAL_TIMING_TOTALS = (
    "online_retrieval_seconds", "retrieval_seconds", "total_compute_seconds",
    "online_retrieval_and_rerank_seconds",
)


def _write_timing_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


class ActualTaskTiming:
    """Actual durations only; independent of ETA, tqdm and warmup sampling."""

    def __init__(
        self, metadata: Optional[dict[str, Any]] = None,
        *, clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.clock = clock if clock is not None else time.perf_counter
        self.started_at = self.clock()
        self.finished_at: Optional[float] = None
        self.status = "in_progress"
        self.metadata = dict(metadata or {})
        self.metadata.setdefault("target_cache_hit", False)
        self.metadata.setdefault("processed_queries", 0)
        self.seconds = {name: 0.0 for name in ACTUAL_TIMING_COMPONENTS}
        self.failed_components: set[str] = set()
        self.error: Optional[dict[str, str]] = None

    @contextmanager
    def measure(self, component: str):
        started = self.clock()
        try:
            yield
        except BaseException:
            self.failed_components.add(component)
            raise
        finally:
            self.seconds[component] += max(0.0, self.clock() - started)

    def finish(self, error: Optional[BaseException] = None) -> None:
        if self.finished_at is not None:
            return
        self.finished_at = self.clock()
        self.status = "completed" if error is None else "failed"
        if error is not None:
            self.error = {"type": type(error).__name__, "message": str(error)}

    def report(self) -> dict[str, Any]:
        end = self.finished_at if self.finished_at is not None else self.clock()
        online = self.seconds["query_encoding_seconds"] + self.seconds["similarity_search_seconds"]
        retrieval = self.seconds["candidate_encoding_seconds"] + online
        return {
            **self.metadata,
            "mode": "full_run_actual_timing", "status": self.status, "time_unit": "seconds",
            **self.seconds,
            "online_retrieval_seconds": online,
            "retrieval_seconds": retrieval,
            "total_compute_seconds": retrieval + self.seconds["reranking_seconds"],
            "online_retrieval_and_rerank_seconds": online + self.seconds["reranking_seconds"],
            "task_end_to_end_seconds": max(0.0, end - self.started_at),
            "failed_components": sorted(self.failed_components),
            "error": self.error,
            "timing_scope": {
                "retrieval_seconds": "candidate encoding + query encoding + similarity search",
                "online_retrieval_seconds": "query encoding + similarity search",
                "reranking_seconds": "rerank_candidates call including prompt preparation and model scoring",
                "total_compute_seconds": "retrieval_seconds + reranking_seconds; excludes cache I/O, device placement and result formatting",
                "task_end_to_end_seconds": "task data loading (when called by main) through result/metric/config writes; excludes timing JSON write",
                "distributed_communication_seconds": "embedding/result exchange and waits; excludes core model computation; can overlap peer I/O",
            },
        }

    def save(self, run_dir: Path) -> None:
        path = run_dir / "timing_metrics.json"
        self.metadata["report_path"] = str(path.resolve())
        report = self.report()
        _write_timing_json(path, report)
        print(
            f"[timing] {self.metadata.get('dataset')}/{self.metadata.get('direction')}: "
            f"retrieval={report['retrieval_seconds']:.3f}s "
            f"(online={report['online_retrieval_seconds']:.3f}s), "
            f"rerank={report['reranking_seconds']:.3f}s, "
            f"compute_total={report['total_compute_seconds']:.3f}s, "
            f"task_end_to_end={report['task_end_to_end_seconds']:.3f}s; "
            f"status={self.status}; saved={path}"
        )


class ActualRunTiming:
    """One model load, sequential tasks, and a separate whole-run wall clock."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.clock = time.perf_counter
        self.started_at = self.clock()
        self.finished_at: Optional[float] = None
        self.status = "in_progress"
        self.model_path = args.model_path
        self.model_loading_seconds = 0.0
        self.requested_tasks = 0
        self.tasks: list[ActualTaskTiming] = []
        self.error: Optional[dict[str, str]] = None
        self.distributed_metadata: dict[str, Any] = {}

    @contextmanager
    def model_loading(self):
        started = self.clock()
        try:
            yield
        finally:
            self.model_loading_seconds += max(0.0, self.clock() - started)

    def new_task(self, dataset: str, split: str, direction: str) -> ActualTaskTiming:
        task = ActualTaskTiming(
            {"dataset": dataset, "split": split, "direction": direction}, clock=self.clock,
        )
        self.tasks.append(task)
        return task

    def finish(self, error: Optional[BaseException] = None) -> None:
        self.finished_at = self.clock()
        self.status = "completed" if error is None else "failed"
        if error is not None:
            self.error = {"type": type(error).__name__, "message": str(error)}

    def save(self, output_root: Path) -> None:
        reports = [task.report() for task in self.tasks]
        end = self.finished_at if self.finished_at is not None else self.clock()
        report = {
            "mode": "full_run_actual_timing", "status": self.status, "time_unit": "seconds",
            **self.distributed_metadata,
            "model_path": self.model_path,
            "model_loading_seconds": self.model_loading_seconds,
            "end_to_end_seconds": max(0.0, end - self.started_at),
            "requested_tasks": self.requested_tasks,
            "started_tasks": len(reports),
            "completed_tasks": sum(task.status == "completed" for task in self.tasks),
            "failed_tasks": sum(task.status == "failed" for task in self.tasks),
            "measured_totals": {
                name: sum(task[name] for task in reports)
                for name in ACTUAL_TIMING_COMPONENTS + ACTUAL_TIMING_TOTALS
            },
            "task_end_to_end_seconds_sum": sum(task["task_end_to_end_seconds"] for task in reports),
            "end_to_end_scope": self.distributed_metadata.get(
                "end_to_end_scope",
                "after argument validation through normal summary write; includes shared model load once and task timing writes, excludes final timing_summary.json write",
            ),
            "error": self.error,
            "tasks": reports,
        }
        path = output_root / "timing_summary.json"
        _write_timing_json(path, report)
        if self.status != "in_progress":
            totals = report["measured_totals"]
            print(
                f"[timing][total] retrieval={totals['retrieval_seconds']:.3f}s, "
                f"rerank={totals['reranking_seconds']:.3f}s, "
                f"compute_total={totals['total_compute_seconds']:.3f}s, "
                f"model_loading={self.model_loading_seconds:.3f}s, "
                f"end_to_end={report['end_to_end_seconds']:.3f}s; status={self.status}; saved={path}"
            )


def _actual_scope(timing: Optional[ActualTaskTiming], component: str):
    return timing.measure(component) if timing is not None else nullcontext()


def encode_targets(
    model: FreeRetQwen,
    task: RetrievalTask,
    batch_size: int,
    cache_path: Optional[Path],
    max_pixels: int,
    time_estimate: Optional[dict[str, Any]] = None,
    actual_timing: Optional[ActualTaskTiming] = None,
    progress_options: Optional[dict[str, Any]] = None,
) -> torch.Tensor:
    candidate_ids = [candidate.candidate_id for candidate in task.candidates]
    if cache_path is not None and cache_path.is_file():
        cache_started = time.perf_counter() if time_estimate is not None else None
        with _actual_scope(actual_timing, "cache_read_seconds"):
            payload = torch.load(cache_path, map_location="cpu")
        if (
            payload.get("prompt_version") == PROMPT_VERSION
            and payload.get("prompt_key") == task.prompt_key
            and payload.get("max_pixels") == max_pixels
            and payload.get("candidate_ids") == candidate_ids
        ):
            print(f"[cache] loaded target embeddings: {cache_path}")
            with _actual_scope(actual_timing, "cache_read_seconds"):
                embeddings = payload["embeddings"].float()
            if actual_timing is not None:
                actual_timing.metadata["target_cache_hit"] = True
            if time_estimate is not None:
                elapsed = time.perf_counter() - cache_started
                time_estimate.update(
                    status="cached", cache_hit=True, elapsed_seconds=elapsed,
                    remaining_seconds=0.0, estimated_total_seconds=elapsed,
                    total_items=len(candidate_ids), completed_items=len(candidate_ids),
                    completed_batches=0, stabilized=False, rate_per_second=None,
                    rate_samples=[], rate_cv=None, num_rate_samples=0,
                    stable_estimate=None, rate_unit="candidate/second",
                )
            return embeddings

    chunks: list[torch.Tensor] = []
    estimator = StableRateEstimator(len(task.candidates)) if time_estimate is not None else None
    with tqdm(**{
        "total": len(task.candidates),
        "desc": f"stage1-target {task.dataset}/{task.direction}",
        "unit": "candidate", "dynamic_ncols": True,
        **(progress_options or {}),
    }) as progress:
        for start in range(0, len(task.candidates), batch_size):
            batch = task.candidates[start:start + batch_size]
            with _actual_scope(actual_timing, "candidate_encoding_seconds"):
                texts = [one_word_prompt(task.prompt_key, "target", item) for item in batch]
                paths = [[item.image_path] if item.image_path else [] for item in batch]
                chunks.append(model.encode_one_word(texts, paths))
            progress.update(len(batch))
            if estimator is not None and estimator.observe_batch(start + len(batch)):
                estimate = estimator.stable_estimate
                progress.write(
                    f"[time-estimate] target {task.dataset}/{task.direction}: "
                    f"stable CV={estimate['rate_cv']:.2%}, "
                    f"estimated_total_seconds={estimate['estimated_total_seconds']:.3f}; "
                    "continuing to finish the full candidate pool for real retrieval."
                )

    with _actual_scope(actual_timing, "candidate_encoding_seconds"):
        embeddings = torch.cat(chunks, dim=0) if chunks else torch.empty((0, 0))
    if time_estimate is not None:
        time_estimate.update(estimator.result(completed=True))
        time_estimate.update(cache_hit=False, rate_unit="candidate/second")
    if cache_path is not None:
        with _actual_scope(actual_timing, "cache_write_seconds"):
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "prompt_version": PROMPT_VERSION,
                    "prompt_key": task.prompt_key,
                    "max_pixels": max_pixels,
                    "candidate_ids": candidate_ids,
                    "embeddings": embeddings,
                },
                cache_path,
            )
    return embeddings


def prepare_similarity_embeddings(
    embeddings: torch.Tensor, device: torch.device, max_gpu_gb: float
) -> torch.Tensor:
    embeddings = embeddings.float().contiguous()
    if device.type != "cuda":
        return embeddings.cpu()
    bytes_required = embeddings.numel() * embeddings.element_size()
    if bytes_required <= max_gpu_gb * (1024**3):
        return embeddings.to(device)
    return embeddings.cpu()


def topk_similarity(
    query_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    k: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target_embeddings.numel() == 0:
        raise RuntimeError("The candidate pool is empty.")
    k = min(k, target_embeddings.shape[0])
    query_embeddings = query_embeddings.float()

    if target_embeddings.device.type == "cuda":
        query_embeddings = query_embeddings.to(target_embeddings.device)
        scores = query_embeddings @ target_embeddings.t()
        values, indices = torch.topk(scores, k=k, dim=1)
        return values.cpu(), indices.cpu()

    best_values: Optional[torch.Tensor] = None
    best_indices: Optional[torch.Tensor] = None
    for start in range(0, target_embeddings.shape[0], chunk_size):
        stop = min(start + chunk_size, target_embeddings.shape[0])
        scores = query_embeddings @ target_embeddings[start:stop].t()
        local_k = min(k, stop - start)
        values, indices = torch.topk(scores, k=local_k, dim=1)
        indices = indices + start
        if best_values is None:
            best_values = values
            best_indices = indices
        else:
            assert best_indices is not None
            merged_values = torch.cat([best_values, values], dim=1)
            merged_indices = torch.cat([best_indices, indices], dim=1)
            best_values, order = torch.topk(merged_values, k=k, dim=1)
            best_indices = torch.gather(merged_indices, 1, order)
    assert best_values is not None and best_indices is not None
    return best_values, best_indices


def rerank_candidates(
    model: FreeRetQwen,
    task: RetrievalTask,
    query: QueryItem,
    candidate_indices: list[int],
    batch_size: int,
) -> list[float]:
    samples = [rerank_prompt(task.prompt_key, query, task.candidates[index]) for index in candidate_indices]
    scores = [0.0] * len(samples)
    for batch in indexed_batches(samples, batch_size, key=lambda item: len(item[1])):
        values = model.rerank([item[0] for _, item in batch], [item[1] for _, item in batch])
        if values.ndim != 1 or len(values) != len(batch):
            raise ValueError("FreeRet must return exactly one score per rerank candidate")
        for (index, _), value in zip(batch, values.tolist()):
            scores[index] = float(value)
    return scores


def evaluate_records(records: list[dict[str, Any]], ks: Sequence[int], key: str) -> dict[str, float]:
    if not records:
        return {f"recall@{k}": 0.0 for k in ks} | {"mrr": 0.0}
    result: dict[str, float] = {}
    for k in ks:
        hits = 0
        for record in records:
            relevant = set(record["relevant_ids"])
            ranked = record[key][:k]
            if any(item["candidate_id"] in relevant for item in ranked):
                hits += 1
        result[f"recall@{k}"] = hits / len(records)

    reciprocal_sum = 0.0
    for record in records:
        relevant = set(record["relevant_ids"])
        for rank, item in enumerate(record[key], start=1):
            if item["candidate_id"] in relevant:
                reciprocal_sum += 1.0 / rank
                break
    result["mrr"] = reciprocal_sum / len(records)
    return result


def _safe_model_tag(model_path: str) -> str:
    """Return a filesystem-safe tag for the loaded model checkpoint."""
    model_name = Path(model_path).expanduser().name
    model_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("._")
    return model_tag or "model"


def _atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


class FreeRetCollectives(TorchStageCollectives):
    def broadcast(self, value: Any) -> Any:
        packet = [value if self.rank == 0 else None]
        self.dist.broadcast_object_list(packet, src=0)
        return packet[0]


class FreeRetDistributed:
    """Control messages only; large CPU embeddings/results use shared files."""

    def __init__(self, collectives: Any) -> None:
        self.collectives = collectives
        self.rank = collectives.rank
        self.world_size = collectives.world_size
        self.local_rank = getattr(collectives, "local_rank", self.rank)
        self.model_device = getattr(collectives, "model_device", collectives.device)

    @classmethod
    def initialize(cls, local_rank: Optional[int] = None) -> "FreeRetDistributed":
        import torch.distributed as dist

        world_size = int(os.environ["WORLD_SIZE"])
        environment_rank = os.environ.get("LOCAL_RANK")
        if environment_rank is not None:
            if local_rank is not None and local_rank != int(environment_rank):
                raise ValueError("--local_rank disagrees with the LOCAL_RANK environment variable.")
            local_rank = int(environment_rank)
        if local_rank is None:
            raise ValueError("FreeRet distributed launch requires LOCAL_RANK or --local_rank.")
        visible_gpus = torch.cuda.device_count()
        if world_size < 1 or world_size > visible_gpus or not 0 <= local_rank < world_size:
            raise ValueError("FreeRet distributed mode requires one visible GPU per local process.")
        # torchrun uses LOCAL_WORLD_SIZE; DeepSpeed uses LOCAL_SIZE. Both
        # launchers remap selected physical GPUs into cuda:0..local_size-1.
        for name in ("LOCAL_WORLD_SIZE", "LOCAL_SIZE"):
            if name in os.environ and int(os.environ[name]) != world_size:
                raise ValueError("FreeRet currently supports a single node with a shared output directory.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            "nccl" if dist.is_nccl_available() else "gloo", timeout=timedelta(minutes=30),
        )
        collective = FreeRetCollectives(torch.device("cuda", local_rank))
        collective.local_rank = local_rank
        collective.model_device = torch.device("cuda", local_rank)
        return cls(collective)

    def primary(self, function: Callable[[], Any], *, share: bool = False) -> Any:
        # All ranks enter this operation on successful control flow. A rank-zero
        # data/I/O failure is broadcast before re-raising; never do this in finally.
        result = error = None
        packet = None
        if self.rank == 0:
            try:
                result = function()
                packet = {"result": result if share else None, "error": None}
            except BaseException as exc:
                error = exc
                packet = {"error": f"{type(exc).__name__}: {exc}"}
        packet = self.collectives.broadcast(packet)
        if packet["error"] is not None:
            if error is not None:
                raise error
            raise RuntimeError(f"[rank0] {packet['error']}")
        return packet["result"] if share else result

    def new_session(self, output_root: Path) -> Path:
        def create() -> str:
            parent = output_root / ".distributed"
            parent.mkdir(parents=True, exist_ok=True)
            return tempfile.mkdtemp(prefix="run-", dir=parent)
        return Path(self.primary(create, share=True))

    def cleanup(self, session: Path, output_root: Path) -> None:
        def remove() -> None:
            # Delete only this run's generated exchange directory, never a root.
            target = session.resolve()
            parent = (output_root / ".distributed").resolve()
            if target.parent != parent or not target.name.startswith("run-"):
                raise ValueError(f"Unsafe distributed temporary directory: {target}")
            shutil.rmtree(target)
        self.primary(remove)


def _merge_eta_rank_reports(
    reports: list[dict[str, Any]], total: int, *, cache_hit: bool = False,
) -> dict[str, Any]:
    elapsed = max(report["elapsed_seconds"] for report in reports)
    estimated = max(report["estimated_total_seconds"] for report in reports)
    return {
        "status": "cached" if cache_hit else "estimated" if any(
            report["status"] == "estimated" for report in reports
        ) else "completed",
        "cache_hit": cache_hit,
        "elapsed_seconds": elapsed, "estimated_total_seconds": estimated,
        "remaining_seconds": max(0.0, estimated - elapsed),
        "total_items": total,
        "completed_items": total if cache_hit else sum(report["completed_items"] for report in reports),
        "world_size": len(reports), "aggregation": "maximum_rank_duration",
        "rank_reports": reports,
    }


def _merge_indexed_records(
    directory: Path, world_size: int, query_count: int, *, partial: bool,
) -> list[dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for rank in range(world_size):
        payload = json.loads((directory / f"results-{rank}.json").read_text(encoding="utf-8"))
        expected = list(range(rank, query_count, world_size))
        rows = payload["records"]
        if (not partial and len(rows) != len(expected)) or len(rows) > len(expected):
            raise RuntimeError(f"Rank {rank} returned an incomplete or oversized query shard.")
        for index, row in zip(expected, rows):
            if row["index"] != index or index in indexed:
                raise RuntimeError(f"Duplicate or incorrectly ordered query index: {row['index']}")
            indexed[index] = row["record"]
    if not partial and len(indexed) != query_count:
        raise RuntimeError("Distributed results do not cover the complete judged query set.")
    return [indexed[index] for index in sorted(indexed)]


class FreeRetDistributedTask:
    def __init__(
        self, context: FreeRetDistributed, directory: Path,
        timing: Optional[ActualTaskTiming],
    ) -> None:
        self.context = context
        self.directory = directory
        self.timing = timing
        self.query_indices: list[int] = []
        self.global_queries: list[QueryItem] = []
        self.observer: Optional[StageTimeObserver] = None
        self.rank_reports: list[dict[str, Any]] = []

    def progress_options(self) -> dict[str, Any]:
        return {"position": self.context.rank, "leave": True}

    def label(self) -> str:
        return f"[rank {self.context.rank}/{self.context.world_size} gpu {self.context.local_rank}]"

    def communication(self):
        return _actual_scope(self.timing, "distributed_communication_seconds")

    def prepare_targets(
        self, model: FreeRetQwen, task: RetrievalTask, args: argparse.Namespace,
        cache_path: Optional[Path], max_pixels: int, target_timing: dict[str, Any],
    ) -> torch.Tensor:
        ctx = self.context
        candidate_ids = [item.candidate_id for item in task.candidates]
        if not candidate_ids:
            raise RuntimeError("The candidate pool is empty.")
        cached: Optional[torch.Tensor] = None
        cache_elapsed = 0.0

        def check_cache() -> bool:
            nonlocal cached, cache_elapsed
            if cache_path is None or not cache_path.is_file():
                return False
            started = time.perf_counter()
            with _actual_scope(self.timing, "cache_read_seconds"):
                payload = torch.load(cache_path, map_location="cpu")
                if (payload.get("prompt_version") == PROMPT_VERSION
                        and payload.get("prompt_key") == task.prompt_key
                        and payload.get("max_pixels") == max_pixels
                        and payload.get("candidate_ids") == candidate_ids
                        and isinstance(payload.get("embeddings"), torch.Tensor)
                        and payload["embeddings"].ndim == 2
                        and payload["embeddings"].shape[0] == len(candidate_ids)):
                    cached = payload["embeddings"].float()
            cache_elapsed = time.perf_counter() - started
            return cached is not None

        started = time.perf_counter()
        cache_hit = ctx.primary(check_cache, share=True)
        if self.timing is not None:
            self.timing.seconds["distributed_communication_seconds"] += max(
                0.0, time.perf_counter() - started - (cache_elapsed if ctx.rank == 0 else 0.0),
            )
        if self.timing is not None:
            self.timing.metadata["target_cache_hit"] = cache_hit
        if cache_hit:
            if ctx.rank != 0:
                started = time.perf_counter()
                with _actual_scope(self.timing, "cache_read_seconds"):
                    cached = torch.load(cache_path, map_location="cpu")["embeddings"].float()
                cache_elapsed = time.perf_counter() - started
            local_report = {
                "rank": ctx.rank, "status": "cached", "elapsed_seconds": cache_elapsed,
                "estimated_total_seconds": cache_elapsed, "completed_items": len(candidate_ids),
            }
            if args.estimate_time_only:
                target_timing.update(_merge_eta_rank_reports(
                    ctx.collectives.gather(local_report), len(candidate_ids), cache_hit=True,
                ))
            assert cached is not None
            return cached

        indices = list(range(ctx.rank, len(candidate_ids), ctx.world_size))
        shard = replace(task, candidates=[task.candidates[index] for index in indices])
        local_timing: dict[str, Any] = {}
        options = self.progress_options()
        options["desc"] = f"{self.label()} target {task.dataset}/{task.direction}"
        embeddings = encode_targets(
            model, shard, args.target_batch_size, None, max_pixels,
            time_estimate=local_timing if args.estimate_time_only else None,
            actual_timing=self.timing, progress_options=options,
        )
        with self.communication():
            _atomic_torch_save(
                {"indices": indices, "embeddings": embeddings}, self.directory / f"targets-{ctx.rank}.pt",
            )
            del embeddings
            ctx.collectives.barrier()

        full: Optional[torch.Tensor] = None
        full_path = cache_path if cache_path is not None else self.directory / "targets-full.pt"

        def merge() -> None:
            nonlocal full
            with self.communication():
                covered = 0
                for rank in range(ctx.world_size):
                    part = torch.load(self.directory / f"targets-{rank}.pt", map_location="cpu")
                    expected = list(range(rank, len(candidate_ids), ctx.world_size))
                    values = part["embeddings"]
                    if part["indices"] != expected or values.ndim != 2 or values.shape[0] != len(expected):
                        raise RuntimeError(f"Invalid candidate embedding shard from rank {rank}.")
                    if expected:
                        if full is None:
                            full = torch.empty((len(candidate_ids), values.shape[1]), dtype=values.dtype)
                        if values.shape[1] != full.shape[1]:
                            raise RuntimeError("Candidate embedding dimensions differ between ranks.")
                        full[expected] = values
                        covered += len(expected)
                    del part, values
                if full is None or covered != len(candidate_ids):
                    raise RuntimeError("Distributed candidate embeddings have missing rows.")
            with _actual_scope(self.timing, "cache_write_seconds") if cache_path is not None else self.communication():
                _atomic_torch_save(
                    {"prompt_version": PROMPT_VERSION, "prompt_key": task.prompt_key,
                     "max_pixels": max_pixels, "candidate_ids": candidate_ids, "embeddings": full},
                    full_path,
                )

        # Do not wrap rank zero's merge a second time in the communication timer.
        started = time.perf_counter()
        ctx.primary(merge)
        if ctx.rank != 0:
            full = torch.load(full_path, map_location="cpu")["embeddings"].float()
            if self.timing is not None:
                self.timing.seconds["distributed_communication_seconds"] += max(0.0, time.perf_counter() - started)
        if args.estimate_time_only:
            local_timing["rank"] = ctx.rank
            target_timing.update(_merge_eta_rank_reports(
                ctx.collectives.gather(local_timing), len(candidate_ids),
            ))
        assert full is not None
        return full

    def shard_queries(self, queries: list[QueryItem]) -> list[QueryItem]:
        self.global_queries = queries
        self.query_indices = list(range(self.context.rank, len(queries), self.context.world_size))
        return [queries[index] for index in self.query_indices]

    def batches(self, queries: list[QueryItem], args: argparse.Namespace):
        batches = (queries[start:start + args.query_batch_size] for start in range(0, len(queries), args.query_batch_size))
        if not args.estimate_time_only:
            return batches
        self.observer = StageTimeObserver(
            "query_retrieval_rerank", early_stop=True, collectives=self.context.collectives,
        )
        return self.observer.iterate(batches, total_items=len(queries), item_count=len)

    def merge_records(self, records: list[dict[str, Any]], *, partial: bool) -> list[dict[str, Any]]:
        with self.communication():
            _write_timing_json(self.directory / f"results-{self.context.rank}.json", {
                "records": [{"index": index, "record": record} for index, record in zip(self.query_indices, records)],
            })
            self.context.collectives.barrier()
        with self.communication():
            return self.context.primary(lambda: _merge_indexed_records(
                self.directory, self.context.world_size, len(self.global_queries), partial=partial,
            ))

    def merge_timing(self) -> None:
        assert self.timing is not None
        self.timing.finish()
        local = self.timing.report()
        local.update(rank=self.context.rank, local_rank=self.context.local_rank)
        _write_timing_json(self.directory / f"timing-{self.context.rank}.json", local)
        reports = self.context.collectives.gather(local)
        if self.context.rank == 0:
            for key in ACTUAL_TIMING_COMPONENTS:
                self.timing.seconds[key] = max(report[key] for report in reports)
            self.timing.metadata.update(
                processed_queries=sum(report["processed_queries"] for report in reports),
                world_size=self.context.world_size, rank_reports=reports,
                aggregation="maximum_per_component; derived sums are not exact parallel wall time",
            )
            self.timing.finished_at = self.timing.clock()


def run_task(
    model: FreeRetQwen,
    task: RetrievalTask,
    args: argparse.Namespace,
    run_dir: Path,
    actual_timing: Optional[ActualTaskTiming] = None,
    distributed: Optional[FreeRetDistributedTask] = None,
) -> dict[str, Any]:
    # ETA mode remains a separate, opt-in workflow with its original outputs.
    if getattr(args, "estimate_time_only", False):
        return _run_task(model, task, args, run_dir, distributed=distributed)
    timing = actual_timing if actual_timing is not None else ActualTaskTiming()
    if distributed is not None:
        distributed.timing = timing
    timing.metadata.update(
        model_path=args.model_path, model_family=model.model_family,
        dataset=task.dataset, split=task.split, direction=task.direction,
        prompt_key=task.prompt_key, num_candidates=len(task.candidates),
        target_batch_size=args.target_batch_size, query_batch_size=args.query_batch_size,
        rerank_batch_size=args.rerank_batch_size, rerank_num=args.rerank_num,
        similarity_chunk_size=args.similarity_chunk_size,
        max_target_gpu_gb=args.max_target_gpu_gb,
    )
    try:
        result = _run_task(model, task, args, run_dir, actual_timing=timing, distributed=distributed)
        if distributed is not None:
            distributed.merge_timing()
    except BaseException as exc:
        timing.finish(exc)
        try:
            if distributed is not None:
                _write_timing_json(distributed.directory / f"failed-{distributed.context.rank}.json", {
                    "status": "failed", "error": {"type": type(exc).__name__, "message": str(exc)},
                    "timing": timing.report(),
                })
            if distributed is None or distributed.context.rank == 0:
                timing.save(run_dir)
        except Exception as timing_exc:
            print(f"[timing][warning] Failed to save task timing: {timing_exc}", file=sys.stderr)
        raise
    timing.finish()
    if distributed is None or distributed.context.rank == 0:
        timing.save(run_dir)
    return result


def _run_task(
    model: FreeRetQwen,
    task: RetrievalTask,
    args: argparse.Namespace,
    run_dir: Path,
    actual_timing: Optional[ActualTaskTiming] = None,
    distributed: Optional[FreeRetDistributedTask] = None,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    if distributed is None:
        _validate_images(task)
    spec = get_prompt_spec(task.prompt_key)
    effective_max_pixels = args.max_pixels if args.max_pixels > 0 else spec.max_pixels
    model.set_max_pixels(effective_max_pixels)
    print(
        f"[prompt] {task.prompt_key}; max_pixels={effective_max_pixels}; "
        f"source={spec.source}"
    )

    queries = [query for query in task.queries if query.relevant_ids]
    skipped = len(task.queries) - len(queries)
    if args.max_queries > 0:
        queries = queries[: args.max_queries]
    if not queries:
        raise RuntimeError(f"No judged queries are available for {task.dataset}.")
    if actual_timing is not None:
        actual_timing.metadata.update(
            num_queries=len(queries), skipped_unjudged_queries=skipped,
            max_pixels=effective_max_pixels,
        )

    cache_path = None
    if args.cache_targets:
        model_tag = _safe_model_tag(args.model_path)
        cache_path = run_dir / f"target_embeddings_{model_tag}.pt"
    estimate_time_only = getattr(args, "estimate_time_only", False)
    target_timing: dict[str, Any] = {}
    if distributed is not None:
        target_embeddings = distributed.prepare_targets(
            model, task, args, cache_path, effective_max_pixels, target_timing,
        )
        queries = distributed.shard_queries(queries)
    elif estimate_time_only:
        target_embeddings = encode_targets(
            model, task, args.target_batch_size, cache_path, effective_max_pixels,
            time_estimate=target_timing,
        )
    else:
        target_embeddings = encode_targets(
            model, task, args.target_batch_size, cache_path, effective_max_pixels,
            actual_timing=actual_timing,
        )
    with _actual_scope(actual_timing, "target_transfer_seconds"):
        target_embeddings = prepare_similarity_embeddings(
            target_embeddings, model.device, args.max_target_gpu_gb
        )

    metrics_ks = sorted({int(k) for k in args.metrics_ks})
    max_metric_k = max(metrics_ks)
    requested_search_k = max(
        int(args.stage1_topk), int(args.rerank_num), max_metric_k
    )
    search_k = min(int(target_embeddings.shape[0]), requested_search_k)
    if actual_timing is not None:
        actual_timing.metadata["search_k"] = search_k
    records: list[dict[str, Any]] = []

    with tqdm(
        total=len(queries),
        desc=f"{distributed.label() + ' ' if distributed is not None else ''}stage1+stage2 {task.dataset}/{task.direction}",
        unit="query",
        dynamic_ncols=True,
        **(distributed.progress_options() if distributed is not None else {}),
    ) as query_progress:
        query_estimator = StableRateEstimator(len(queries)) if estimate_time_only and distributed is None else None
        batches = distributed.batches(queries, args) if distributed is not None else (
            queries[start:start + args.query_batch_size] for start in range(0, len(queries), args.query_batch_size)
        )
        for query_batch in batches:
            with _actual_scope(actual_timing, "query_encoding_seconds"):
                query_texts = [one_word_prompt(task.prompt_key, "query", item) for item in query_batch]
                query_paths = [[item.image_path] if item.image_path else [] for item in query_batch]
                query_embeddings = model.encode_one_word(query_texts, query_paths)
            with _actual_scope(actual_timing, "similarity_search_seconds"):
                stage1_values, stage1_indices = topk_similarity(
                    query_embeddings,
                    target_embeddings,
                    search_k,
                    args.similarity_chunk_size,
                )

            for batch_index, query in enumerate(query_batch):
                indices = stage1_indices[batch_index].tolist()
                values = stage1_values[batch_index].tolist()
                rerank_count = min(args.rerank_num, len(indices))
                rerank_indices = indices[:rerank_count]
                with _actual_scope(actual_timing, "reranking_seconds"):
                    rerank_scores = rerank_candidates(
                        model, task, query, rerank_indices, args.rerank_batch_size
                    )
                order = sorted(
                    range(len(rerank_indices)),
                    key=lambda index: (-rerank_scores[index], index),
                )
                final_indices = [rerank_indices[index] for index in order]
                reranked_set = set(rerank_indices)
                final_indices.extend(index for index in indices if index not in reranked_set)

                stage1_rank = [
                    {
                        "candidate_id": task.candidates[index].candidate_id,
                        "score": float(score),
                    }
                    for index, score in zip(indices, values)
                ]
                final_rank = []
                rerank_score_by_index = {
                    index: score for index, score in zip(rerank_indices, rerank_scores)
                }
                for index in final_indices:
                    final_rank.append(
                        {
                            "candidate_id": task.candidates[index].candidate_id,
                            "score": float(
                                rerank_score_by_index.get(
                                    index, values[indices.index(index)]
                                )
                            ),
                        }
                    )

                records.append(
                    {
                        "query_id": query.query_id,
                        "relevant_ids": sorted(query.relevant_ids),
                        "stage1": stage1_rank,
                        "final": final_rank,
                    }
                )
                if actual_timing is not None:
                    actual_timing.metadata["processed_queries"] = len(records)
                query_progress.update(1)

            # Count query batches, not tqdm's individual query updates or the
            # rerank microbatches. Never stop in the middle of a query batch.
            if query_estimator is not None and query_estimator.observe_batch(len(records)):
                break
        if query_estimator is not None:
            query_timing = query_estimator.result(completed=len(records) == len(queries))

    if distributed is not None:
        if estimate_time_only:
            query_estimator = distributed.observer.estimator
            query_timing = _merge_eta_rank_reports(
                distributed.observer.report["rank_reports"], len(distributed.global_queries),
            )
        records = distributed.merge_records(records, partial=estimate_time_only)
        if distributed.context.rank != 0:
            return {}
        queries = distributed.global_queries

    if estimate_time_only:
        query_timing["rate_unit"] = "query/second"
        report_path = run_dir / "time_estimate.json"
        estimated_total = (
            target_timing["elapsed_seconds"] + query_timing["estimated_total_seconds"]
        )
        report = {
            "mode": "estimate_time_only",
            "status": query_timing["status"],
            "time_unit": "seconds",
            "dataset": task.dataset,
            "split": task.split,
            "direction": task.direction,
            "model_path": args.model_path,
            "model_family": model.model_family,
            "prompt_key": task.prompt_key,
            "max_pixels": effective_max_pixels,
            "num_candidates": len(task.candidates),
            "num_queries": len(queries),
            "processed_queries": len(records),
            "skipped_unjudged_queries": skipped,
            "target_batch_size": args.target_batch_size,
            "query_batch_size": args.query_batch_size,
            "rerank_batch_size": args.rerank_batch_size,
            "rerank_num": args.rerank_num,
            "search_k": search_k,
            "similarity_chunk_size": args.similarity_chunk_size,
            "max_target_gpu_gb": args.max_target_gpu_gb,
            "target_cache_hit": target_timing["cache_hit"],
            "target_elapsed_seconds": target_timing["elapsed_seconds"],
            "query_elapsed_seconds": query_timing["elapsed_seconds"],
            "query_remaining_seconds": query_timing["remaining_seconds"],
            "query_estimated_total_seconds": query_timing["estimated_total_seconds"],
            "estimated_total_seconds": estimated_total,
            "estimation_policy": {
                "warmup_batches": query_estimator.warmup_batches,
                "sample_interval_seconds": query_estimator.sample_interval_seconds,
                "stable_samples": query_estimator.stable_samples,
                "max_cv_exclusive": query_estimator.max_cv,
                "cv_definition": "population_standard_deviation / mean",
            },
            "stages": {"target_encoding": target_timing, "query_retrieval_rerank": query_timing},
            "timing_scope": "target encoding/cache read + query encoding/search/reranking",
            "excluded_time": [
                "model_loading", "dataset_preparation", "image_validation",
                "target_embedding_device_placement", "target_cache_write",
                "recall_evaluation", "final_report_write",
            ],
            "report_path": str(report_path.resolve()),
        }
        if distributed is not None:
            report["world_size"] = distributed.context.world_size
            report["excluded_time"].append("distributed_embedding_exchange_and_control_waits")
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        print(
            f"[time-estimate] {task.dataset}/{task.direction}: status={report['status']}; "
            f"estimated_total_seconds={estimated_total:.3f} "
            f"(target={target_timing['elapsed_seconds']:.3f}s, "
            f"query+rerank={query_timing['estimated_total_seconds']:.3f}s); "
            f"processed_queries={len(records)}/{len(queries)}; saved={report_path}"
        )
        if report["status"] == "completed":
            print("[time-estimate] Query stage completed; saving actual duration, not an early-exit estimate.")
        return report

    metrics: dict[str, Any] = {
        "dataset": task.dataset,
        "split": task.split,
        "direction": task.direction,
        "prompt_key": task.prompt_key,
        "prompt_source": spec.source,
        "model_family": model.model_family,
        "model_class": model.model_class_name,
        "processor_class": model.processor_class_name,
        "transformers_version": model.transformers_version,
        "attention_implementation": model.attn_implementation,
        "decoder_norm": model._decoder_norm_path,
        "max_pixels": effective_max_pixels,
        "max_pixels_applied": model.max_pixels_applied,
        "num_queries": len(records),
        "skipped_unjudged_queries": skipped,
        "num_candidates": len(task.candidates),
        "rerank_num": args.rerank_num,
        "stage1": evaluate_records(records, metrics_ks, "stage1"),
        "final": evaluate_records(records, metrics_ks, "final"),
    }

    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_path": args.model_path,
                "data_root": str(args.data_root),
                "prompt_version": PROMPT_VERSION,
                "prompt": prompt_manifest(task.prompt_key),
                "model_family": model.model_family,
                "model_class": model.model_class_name,
                "processor_class": model.processor_class_name,
                "transformers_version": model.transformers_version,
                "attention_implementation": model.attn_implementation,
                "decoder_norm": model._decoder_norm_path,
                "effective_max_pixels": effective_max_pixels,
                "max_pixels_applied": model.max_pixels_applied,
                "stage1_topk": search_k,
                "rerank_num": args.rerank_num,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    return metrics


def _validate_images(task: RetrievalTask) -> None:
    missing: list[str] = []
    invalid: list[str] = []
    for item in list(task.queries) + list(task.candidates):
        if not item.image_path:
            continue
        item_id = item.query_id if isinstance(item, QueryItem) else item.candidate_id
        image_path = Path(item.image_path)
        if not image_path.is_file():
            if len(missing) < 10:
                missing.append(f"{item_id}: {image_path}")
            continue
        image_format = "unknown"
        try:
            with Image.open(image_path) as image:
                image_format = image.format or "unknown"
                image.verify()
        except (UnidentifiedImageError, OSError) as exc:
            if len(invalid) < 10:
                invalid.append(
                    f"{item_id}: {image_path} "
                    f"(format={image_format}, error={exc})"
                )
    if missing:
        raise FileNotFoundError(
            f"{task.dataset} requires image files that are missing; examples: {missing}"
        )
    if invalid:
        raise RuntimeError(
            f"{task.dataset} contains image files that Pillow cannot decode; "
            f"examples: {invalid}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FreeRet two-stage retrieval for the project's dataset.py formats."
    )
    parser.add_argument("--datasets", default="coco")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--direction",
        choices=("auto", "t2i", "i2t", "t2it", "it2t", "both"),
        default="auto",
    )
    parser.add_argument(
        "--data_root",
        default=os.environ.get("DATASET_ROOT", "/root/autodl-tmp"),
        help="root containing data/coco, data/WebQA, data/UMRB-LLaVA, etc.",
    )
    parser.add_argument(
        "--output_dir",
        default="./results/freeret",
        help=(
            "base output directory; a sanitized model-name subdirectory is "
            "added before metrics, results, and target caches"
        ),
    )
    parser.add_argument(
        "--model_path",
        default=os.environ.get(
            "MODEL_PATH", "/root/autodl-fs/checkpoints/Qwen2.5-VL-7B-Instruct"
        ),
    )
    parser.add_argument("--device_map", default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--attn_implementation",
        choices=("auto", "flash_attention_2", "sdpa", "eager"),
        default="flash_attention_2",
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=0,
        help="0 uses the per-dataset FreeRet prompt setting; a positive value overrides it.",
    )
    parser.add_argument("--target_batch_size", type=int, default=8)
    parser.add_argument("--query_batch_size", type=int, default=4)
    parser.add_argument("--rerank_batch_size", type=int, default=2)
    parser.add_argument("--rerank_num", type=int, default=50)
    parser.add_argument("--stage1_topk", type=int, default=50)
    parser.add_argument("--similarity_chunk_size", type=int, default=8192)
    parser.add_argument("--max_target_gpu_gb", type=float, default=4.0)
    parser.add_argument("--metrics_ks", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--max_queries", type=int, default=-1)
    parser.add_argument("--cache_targets", action="store_true")
    parser.add_argument(
        "--estimate_time_only", action="store_true",
        help=(
            "Estimate seconds for each dataset/direction without writing retrieval metrics. "
            "Finish the full target pool, then stop querying after 20 warmup batches "
            "and five >=5s throughput samples with population CV <5%."
        ),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None, help=argparse.SUPPRESS)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "target_batch_size",
        "query_batch_size",
        "rerank_batch_size",
        "rerank_num",
        "stage1_topk",
        "similarity_chunk_size",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.max_pixels < 0:
        raise ValueError("--max_pixels cannot be negative")
    if args.max_target_gpu_gb < 0:
        raise ValueError("--max_target_gpu_gb cannot be negative")
    if any(k <= 0 for k in args.metrics_ks):
        raise ValueError("--metrics_ks values must be positive")


def _save_time_estimate_summary(
    reports: list[dict[str, Any]], requested_tasks: int, output_root: Path
) -> None:
    all_done = len(reports) == requested_tasks
    summary = {
        "mode": "estimate_time_only",
        "status": (
            "in_progress" if not all_done
            else "estimated" if any(report["status"] == "estimated" for report in reports)
            else "completed"
        ),
        "time_unit": "seconds",
        "requested_tasks": requested_tasks,
        "completed_tasks": len(reports),
        "all_tasks_processed": all_done,
        "estimated_total_seconds": sum(report["estimated_total_seconds"] for report in reports),
        "tasks": reports,
    }
    path = output_root / "time_estimate_summary.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _distributed_main(args: argparse.Namespace, context: Optional[FreeRetDistributed] = None) -> None:
    actual = ActualRunTiming(args) if not args.estimate_time_only else None
    output_root: Optional[Path] = None
    session: Optional[Path] = None
    active: Optional[ActualTaskTiming] = None
    run_dir: Optional[Path] = None
    owned_group = context is None
    try:
        context = context if context is not None else FreeRetDistributed.initialize(
            getattr(args, "local_rank", None)
        )
        rank = context.rank
        torch.manual_seed(args.seed)
        data_root = Path(args.data_root).expanduser().resolve()
        args.data_root = data_root
        output_root = Path(args.output_dir).expanduser().resolve() / _safe_model_tag(args.model_path)
        context.primary(lambda: output_root.mkdir(parents=True, exist_ok=True))
        session = context.new_session(output_root)
        datasets = parse_dataset_names(args.datasets)
        specs = [(dataset, direction) for dataset in datasets for direction in directions_for(dataset, args.direction)]
        if actual is not None:
            actual.requested_tasks = len(specs)
            actual.distributed_metadata.update(
                world_size=context.world_size,
                aggregation="maximum_per_component; end_to_end_seconds is rank-zero wall time",
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                end_to_end_scope="after argument validation through merged summary write and successful exchange-file cleanup; includes parallel model loading and task report writes, excludes final timing_summary.json write",
            )
            if rank == 0:
                actual.save(output_root)

        def save_manifest() -> None:
            manifests = {}
            for dataset, direction in specs:
                manifest = prompt_manifest(prompt_key_for(dataset, direction))
                manifest["effective_max_pixels"] = args.max_pixels if args.max_pixels > 0 else manifest["max_pixels"]
                manifests[f"{dataset}/{direction}"] = manifest
            _write_timing_json(output_root / "prompt_manifest.json", manifests)
        context.primary(save_manifest)
        device_map = str(context.model_device)
        print(f"[rank {rank}/{context.world_size}] model_device={device_map}; one complete model per GPU; --device_map={args.device_map!r} is ignored; output_root={output_root}")
        with actual.model_loading() if actual is not None else nullcontext():
            model = FreeRetQwen(
                model_path=args.model_path, dtype=args.dtype, device_map=device_map,
                attn_implementation=args.attn_implementation,
                max_pixels=args.max_pixels if args.max_pixels > 0 else 512,
            )
        if actual is not None:
            loading = context.collectives.gather({"rank": rank, "seconds": actual.model_loading_seconds})
            actual.model_loading_seconds = max(item["seconds"] for item in loading)
            actual.distributed_metadata["model_loading_rank_reports"] = loading

        materializer = ImageMaterializer(output_root / "image_cache", data_root) if rank == 0 else None
        summary: list[dict[str, Any]] = []
        for task_index, (dataset, direction) in enumerate(specs):
            run_dir = None
            active = actual.new_task(dataset, args.split, direction) if actual is not None else None
            work_dir = session / f"task-{task_index}"
            context.primary(lambda: work_dir.mkdir())

            def prepare_task() -> RetrievalTask:
                task = load_task(data_root, dataset, args.split, direction, materializer)
                _validate_images(task)
                return task
            task = context.primary(prepare_task, share=True)
            run_dir = output_root / dataset / f"{task.split}_{direction}"
            if rank == 0:
                print(f"[start] {dataset}/{direction}: queries={len(task.queries)}, candidates={len(task.candidates)}, processes={context.world_size}")
            execution = FreeRetDistributedTask(context, work_dir, active)
            result = run_task(model, task, args, run_dir, actual_timing=active, distributed=execution)
            if rank == 0:
                summary.append(result)
                if args.estimate_time_only:
                    _save_time_estimate_summary(summary, len(specs), output_root)
                else:
                    actual.save(output_root)
                print(f"[done] {run_dir}")

        if rank == 0 and not args.estimate_time_only:
            _write_timing_json(output_root / "summary.json", summary)
        context.cleanup(session, output_root)
        if rank == 0:
            if actual is not None:
                actual.finish()
                actual.save(output_root)
            else:
                print(f"[time-estimate] all tasks finished; saved={output_root / 'time_estimate_summary.json'}")
    except BaseException as exc:
        # No exception-path collective: the launcher terminates surviving
        # workers when a peer exits. Keep diagnostics without masking errors.
        rank = context.rank if context is not None else int(os.environ.get("RANK", "0"))
        if active is not None:
            active.finish(exc)
        try:
            if session is not None:
                _write_timing_json(session / f"failed-rank-{rank}.json", {
                    "status": "failed", "rank": rank,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "timing": active.report() if active is not None else None,
                })
            if actual is not None and rank == 0 and output_root is not None:
                if active is not None and "rank_reports" not in active.metadata:
                    active.metadata["aggregation"] = "partial_rank_zero_only_due_to_failure; see retained rank diagnostics"
                if active is not None and run_dir is not None:
                    active.save(run_dir)
                actual.finish(exc)
                actual.save(output_root)
        except Exception as timing_exc:
            print(f"[rank {rank}][timing][warning] {timing_exc}", file=sys.stderr)
        print(f"[rank {rank}][error] {exc}; diagnostics={session}", file=sys.stderr)
        raise
    finally:
        if owned_group:
            import torch.distributed as dist
            if dist.is_initialized():
                try:
                    dist.destroy_process_group()
                except Exception as cleanup_exc:
                    print(f"[distributed][warning] Process-group cleanup failed: {cleanup_exc}", file=sys.stderr)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    # DeepSpeed also supplies these rank variables for a single worker. It must
    # use the same GPU binding and orchestration as a multi-worker launch.
    launched = args.local_rank is not None or any(
        name in os.environ for name in ("RANK", "LOCAL_RANK")
    )
    if launched or int(os.environ.get("WORLD_SIZE", "1")) > 1:
        validate_args(args)
        _distributed_main(args)
        return
    actual_run_timing: Optional[ActualRunTiming] = None
    active_task_timing: Optional[ActualTaskTiming] = None
    output_root: Optional[Path] = None
    try:
        validate_args(args)
        if not args.estimate_time_only:
            actual_run_timing = ActualRunTiming(args)
        torch.manual_seed(args.seed)
        data_root = Path(args.data_root).expanduser().resolve()
        args.data_root = data_root
        output_base = Path(args.output_dir).expanduser().resolve()
        model_tag = _safe_model_tag(args.model_path)
        output_root = output_base / model_tag
        output_root.mkdir(parents=True, exist_ok=True)
        materializer = ImageMaterializer(output_root / "image_cache", data_root)

        datasets = parse_dataset_names(args.datasets)
        run_specs = [
            (dataset, direction)
            for dataset in datasets
            for direction in directions_for(dataset, args.direction)
        ]
        if actual_run_timing is not None:
            actual_run_timing.requested_tasks = len(run_specs)
            actual_run_timing.save(output_root)
        print("FreeRet runs:")
        for dataset, direction in run_specs:
            print(f"  - {dataset} / {direction}")
        print(f"data_root={data_root}")
        print(f"model_path={args.model_path}")
        print(f"output_root={output_root}")
        if args.estimate_time_only:
            print(
                "[time-estimate] enabled: 20 warmup batches, >=5s sample intervals, "
                "5 consecutive rates with CV <5%; targets remain complete. "
                "Each configured task is profiled; no retrieval metrics will be written."
            )

        selected_prompts: dict[str, dict[str, Any]] = {}
        for dataset, direction in run_specs:
            prompt_key = prompt_key_for(dataset, direction)
            manifest = prompt_manifest(prompt_key)
            manifest["effective_max_pixels"] = (
                args.max_pixels if args.max_pixels > 0 else manifest["max_pixels"]
            )
            selected_prompts[f"{dataset}/{direction}"] = manifest
        with (output_root / "prompt_manifest.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(selected_prompts, handle, ensure_ascii=False, indent=2)

        with actual_run_timing.model_loading() if actual_run_timing is not None else nullcontext():
            model = FreeRetQwen(
                model_path=args.model_path,
                dtype=args.dtype,
                device_map=args.device_map,
                attn_implementation=args.attn_implementation,
                max_pixels=args.max_pixels if args.max_pixels > 0 else 512,
            )
        print(f"model_family={model.model_family}")
        print(f"model_class={model.model_class_name}")
        print(f"processor_class={model.processor_class_name}")
        print(f"transformers_version={model.transformers_version}")
        print(f"attention_implementation={model.attn_implementation}")
        print(f"decoder_norm={model._decoder_norm_path}")

        summary: list[dict[str, Any]] = []
        for dataset, direction in run_specs:
            if actual_run_timing is not None:
                # Include loading/materialization in task wall time, not core compute.
                active_task_timing = actual_run_timing.new_task(dataset, args.split, direction)
            task = load_task(data_root, dataset, args.split, direction, materializer)
            run_dir = output_root / dataset / f"{task.split}_{direction}"
            print(
                f"[start] {dataset}/{direction}: "
                f"queries={len(task.queries)}, candidates={len(task.candidates)}"
            )
            if actual_run_timing is None:
                summary.append(run_task(model, task, args, run_dir))
            else:
                summary.append(run_task(model, task, args, run_dir, actual_timing=active_task_timing))
                actual_run_timing.save(output_root)
            if args.estimate_time_only:
                # Persist incrementally so completed tasks survive a later error.
                _save_time_estimate_summary(summary, len(run_specs), output_root)
            print(f"[done] {run_dir}")

        if args.estimate_time_only:
            print(f"[time-estimate] all tasks finished; saved={output_root / 'time_estimate_summary.json'}")
        else:
            with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2)
            actual_run_timing.finish()
            actual_run_timing.save(output_root)
    except BaseException as exc:
        if actual_run_timing is not None:
            # A task can fail during loading, before run_task can save its report.
            if active_task_timing is not None and active_task_timing.status == "in_progress":
                active_task_timing.finish(exc)
            actual_run_timing.finish(exc)
            if output_root is not None:
                try:
                    actual_run_timing.save(output_root)
                except Exception as timing_exc:
                    print(f"[timing][warning] Failed to save run timing: {timing_exc}", file=sys.stderr)
        print(f"[error] {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
