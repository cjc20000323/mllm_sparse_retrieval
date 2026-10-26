"""Shared candidate IO, sharding and native LLaVA-NeXT dense inference.

Model-specific adapters supply their own prompts and image preprocessing.
The same native inference is reused by dense_search for online queries.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from io import BytesIO
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any, Optional

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


@dataclass
class DenseEncoderArguments:
    dense_model_type: str = field(
        default="auto", metadata={"choices": ["auto", "mmret", "unime", "vlm2vec"],
                                  "help": "Force mmret, unime or vlm2vec for a renamed local checkpoint."})
    data_root: Optional[str] = field(
        default=None, metadata={"help": "Native dense data root containing data/<dataset>/; defaults to arguments.py."})
    vlm2vec_image_resolution: str = field(
        default="high", metadata={"choices": ["high", "mid", "low", "original"],
                                  "help": "VLM2Vec image resize before HF AnyRes processing; high=1344, mid=672, low=128."})


DATASET_PROFILES = {
    "coco": ("ir", "coco", "coco_file_path"),
    "flickr": ("ir", "flickr", "flickr_file_path"),
    "CUHK-PEDES": ("tbpr", "CUHK-PEDES", "cuhk_pedes_file_path"),
    "ICFG-PEDES": ("tbpr", "ICFG-PEDES", "icfg_pedes_flie_path"),
    "RSTPReid": ("tbpr", "RSTPReid", "rstpreid_file_path"),
    "webqa": ("t2it", "WebQA", "webqa_file_path"),
    "remuq": ("it2t", "UMRB-ReMuQ", "remuq_file_path"),
    "llava": ("it2t", "UMRB-LLaVA", "llava_file_path"),
}


def validate_task(dataset_name: str, task_type: str, encode_type: str,
                  encode_is_query: bool = False, model_label: str = "MMRet") -> None:
    if encode_is_query:
        raise ValueError(f"{model_label} encode_dense supports candidates only; --encode_is_query is not supported. "
                         f"Query-time encoding must use the same {model_label} model and its query template.")
    if dataset_name not in DATASET_PROFILES:
        raise ValueError(f"Unsupported {model_label} dataset {dataset_name!r}; choose {', '.join(DATASET_PROFILES)}.")
    expected = DATASET_PROFILES[dataset_name][0]
    if task_type != expected:
        raise ValueError(f"{dataset_name} requires --task_type {expected}, got {task_type!r}.")
    allowed = ("text", "image") if expected == "ir" else (("image",) if expected == "tbpr" else ("corpus",))
    if encode_type not in allowed:
        raise ValueError(f"{dataset_name} requires --encode_type {' or '.join(allowed)}, got {encode_type!r}.")


@contextmanager
def dataset_paths(dataset_module, data_root: Optional[str]):
    """Temporarily supply constructor paths; do not change arguments.py or dataset.py."""
    if data_root is None:
        yield Path(dataset_module.dataset_path_prefix).expanduser().resolve()
        return
    root = Path(data_root).expanduser().resolve()
    replacements = {"dataset_path_prefix": str(root) + os.sep}
    for _, directory, attr in DATASET_PROFILES.values():
        replacements[attr] = str(root / "data" / directory) + os.sep
    original = {key: getattr(dataset_module, key) for key in replacements}
    try:
        for key, value in replacements.items():
            setattr(dataset_module, key, value)
        yield root
    finally:
        for key, value in original.items():
            setattr(dataset_module, key, value)


@dataclass(frozen=True)
class Candidate:
    identifier: Any
    text: Optional[str] = None
    image: Any = None


class DenseCandidates(Dataset):
    """Preserve dataset IDs/selection without WebQA's bytes-only __getitem__."""

    def __init__(self, source, dataset_name: str, encode_type: str, original_image_dir: Path):
        self.source = source
        self.name = dataset_name
        self.encode_type = encode_type
        self.original_image_dir = original_image_dir
        self.data_dir = Path(source.data_path)
        if dataset_name in ("coco", "flickr"):
            self.ids = source.text_id_list if encode_type == "text" else source.img_id_list
        elif DATASET_PROFILES[dataset_name][0] == "tbpr":
            self.ids = source.img_id_list
        else:
            self.ids = source.candidate_id_list
        if len(set(self.ids)) != len(self.ids):
            raise ValueError(f"{dataset_name}: duplicate candidate IDs; cannot write an unambiguous dense index.")

    def __len__(self):
        return len(self.ids)

    def _image_path(self, relative: Path) -> str:
        # Generated-validation images are optional; otherwise fall back to the
        # original image tree captured before dataset_paths restores globals.
        local = self.data_dir / relative
        return str(local if local.is_file() else self.original_image_dir / relative)

    def __getitem__(self, index):
        identifier = self.ids[index]
        if self.name in ("coco", "flickr"):
            if self.encode_type == "text":
                return Candidate(identifier, text=self.source.text_dict[identifier])
            filename = self.source.img_dict[identifier]
            if self.name == "coco":
                relative = Path(str(self.source.img2filepath[identifier]).strip("/\\")) / filename
            else:
                relative = Path("flickr30k-images") / filename
            return Candidate(identifier, image=self._image_path(relative))
        if DATASET_PROFILES[self.name][0] == "tbpr":
            return Candidate(identifier, image=self._image_path(Path("imgs") / self.source.img_dict[identifier]))
        value = self.source.id2candidate[identifier]
        if self.name == "webqa":
            image = value["image"]
            if isinstance(image, dict) and image.get("path") and not image.get("bytes"):
                image = dict(image)
                path = Path(image["path"])
                image["path"] = str(path if path.is_absolute() else self.data_dir / path)
            elif isinstance(image, (str, os.PathLike)):
                path = Path(image)
                image = str(path if path.is_absolute() else self.data_dir / path)
            if image is None or not isinstance(value["text"], str):
                raise ValueError(f"WebQA candidate {identifier!r} must contain both text and image.")
            return Candidate(identifier, text=value["text"], image=image)
        return Candidate(identifier, text=value)


def build_candidates(data_args, training_args, data_root=None, dataset_module=None, model_label="MMRet"):
    if dataset_module is None:
        import dataset as dataset_module
    name = data_args.dataset_name
    validate_task(name, training_args.task_type, training_args.encode_type, data_args.encode_is_query, model_label)
    kwargs = dict(data_args=data_args,
                  use_generated_validation_data=getattr(data_args, "use_generated_validation_data", False),
                  generated_validation_data_root=getattr(data_args, "generated_validation_data_root", None))
    with dataset_paths(dataset_module, data_root) as root:
        args = (name, None, data_args.dataset_split)
        task = DATASET_PROFILES[name][0]
        if task == "ir":
            mode = "full" if training_args.encode_type == "text" else "single"
            source = dataset_module.CrossModalRetrievalDataset(*args, mode, **kwargs)
        elif task == "tbpr":
            source = dataset_module.TextPersonRetrievalDataset(*args, "single", **kwargs)
        elif task == "t2it":
            source = dataset_module.Text2ImagetextRetrievalDataset(*args, "corpus", **kwargs)
        else:
            source = dataset_module.Imagetext2TextRetrievalDataset(*args, "corpus", **kwargs)
        original_image_dir = root / "data" / DATASET_PROFILES[name][1]
    return DenseCandidates(source, name, training_args.encode_type, original_image_dir)


def decode_image(value) -> Image.Image:
    """Return a detached Pillow image with original size/mode; caller owns it.

    Deliberately no resize or RGB conversion here: MMRet resizes before RGB
    conversion, while UniME gives an unresized RGB image to its AnyRes processor.
    """
    if isinstance(value, Image.Image):
        return value.copy()
    if isinstance(value, dict):
        raw = value.get("bytes")
        value = raw if raw is not None and len(raw) else value.get("path")
    if isinstance(value, (bytes, bytearray, memoryview)):
        value = BytesIO(bytes(value))
    if value is None:
        raise ValueError("Missing candidate image bytes/path.")
    with Image.open(value) as image:
        return image.copy()


def last_token_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if hidden.shape[:2] != mask.shape or not torch.all(mask.any(dim=1)):
        raise ValueError("Dense hidden states and nonempty attention masks must align.")
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    last = positions.masked_fill(~mask.bool(), -1).max(dim=1).values
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


class LlavaNextDenseEncoder:
    model_label = "Dense"

    def format_prompt(self, candidate: Candidate) -> str:
        raise NotImplementedError

    def prepare_image(self, value) -> Image.Image:
        raise NotImplementedError

    def __init__(self, backbone, processor, device, dtype):
        self.backbone = backbone.eval()
        self.processor = processor
        self.device = torch.device(device)
        self.dtype = dtype
        self.dimension = backbone.config.text_config.hidden_size

    @classmethod
    def from_pretrained(cls, model_path, device, dtype, cache_dir=None):
        from packaging.version import Version
        from transformers import __version__, LlavaNextForConditionalGeneration, LlavaNextProcessor
        if Version(__version__) < Version("4.57.3"):
            raise RuntimeError(f"{cls.model_label} native encoding requires transformers>=4.57.3; "
                               "legacy remote-code wrappers are not used.")
        model, info = LlavaNextForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype, device_map={"": str(device)},
            low_cpu_mem_usage=True, attn_implementation="sdpa", cache_dir=cache_dir,
            output_loading_info=True)
        # A missing unused LM head is harmless; a missing retrieval parameter is not.
        missing = [key for key in info.get("missing_keys", []) if key != "lm_head.weight"]
        if missing or info.get("mismatched_keys") or info.get("unexpected_keys"):
            raise ValueError(f"{cls.model_label} checkpoint did not load completely: {info}")
        if model.config.text_config.model_type != "mistral":
            raise ValueError(f"{cls.model_label} requires the Mistral LLaVA-NeXT architecture.")
        processor = LlavaNextProcessor.from_pretrained(model_path, cache_dir=cache_dir, use_fast=False)
        # Some old snapshots omit these processor fields; CLIP has one CLS token.
        processor.patch_size = model.config.vision_config.patch_size
        processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy
        processor.num_additional_image_tokens = 1
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        # Keep the backbone only; neither logits nor every layer's activations are needed.
        return cls(model.model, processor, device, dtype)

    @torch.inference_mode()
    def encode_batch(self, candidates: list[Candidate]) -> np.ndarray:
        if not candidates:
            return np.empty((0, self.dimension), dtype=np.float32)
        images = []
        prompts = []
        has_images = [item.image is not None for item in candidates]
        if any(has_images) and not all(has_images):
            raise ValueError(f"A {self.model_label} batch must have a consistent image modality.")
        try:
            for item in candidates:
                if item.text is None and item.image is None:
                    raise ValueError(f"Candidate {item.identifier!r} has no input.")
                prompts.append(self.format_prompt(item))
                if item.image is not None:
                    images.append(self.prepare_image(item.image))
            inputs = self.processor(text=prompts, images=images or None, padding=True,
                                    truncation=False, return_tensors="pt")
        except Exception as exc:
            raise ValueError(f"{self.model_label} preprocessing failed for IDs {[item.identifier for item in candidates]}: {exc}") from exc
        finally:
            for image in images:
                image.close()
        max_length = self.backbone.config.text_config.max_position_embeddings
        if inputs["input_ids"].shape[1] > max_length:
            raise ValueError(f"{self.model_label} input exceeds context length {max_length}; "
                             "image placeholders must not be truncated. Candidate IDs: "
                             f"{[item.identifier for item in candidates]}")
        inputs = {key: value.to(self.device, dtype=self.dtype) if value.is_floating_point()
                  else value.to(self.device) for key, value in inputs.items()}
        mask = inputs["attention_mask"]
        inputs["position_ids"] = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 0)
        outputs = self.backbone(**inputs, use_cache=False, output_hidden_states=False, return_dict=True)
        reps = F.normalize(last_token_pool(outputs.last_hidden_state, mask).float(), dim=-1)
        if reps.shape != (len(candidates), self.dimension) or not torch.isfinite(reps).all():
            raise ValueError(f"{self.model_label} returned invalid candidate embeddings.")
        return reps.cpu().numpy()


@dataclass
class EncodingContext:
    rank: int
    world_size: int
    device: torch.device
    owns_process_group: bool = False


def setup_context() -> EncodingContext:
    initialized = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if initialized else int(os.environ.get("WORLD_SIZE", "1"))
    rank = dist.get_rank() if initialized else int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("Invalid RANK, LOCAL_RANK or WORLD_SIZE for dense encoding.")
    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise ValueError(f"LOCAL_RANK={local_rank} exceeds the visible GPU count.")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    owns = world_size > 1 and not initialized
    if owns:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return EncodingContext(rank, world_size, device, owns)


def shard_indices(length: int, rank: int, world_size: int) -> range:
    if length < 0 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid candidate sharding parameters.")
    # Unlike DistributedSampler, this never pads/repeats IDs on uneven shards.
    return range(rank, length, world_size)


def output_directory(model_args, data_args, training_args) -> Path:
    model_name = str(model_args.model_name_or_path).replace("\\", "/").rstrip("/").split("/")[-1]
    if not model_name or model_name in (".", ".."):
        raise ValueError("Dense model path must identify a model directory or HF repository.")
    cluster = f"cluster_{model_args.cluster_sum}" if model_args.use_output_embedding_cluster else "no_cluster"
    suffix = (f"{data_args.num_expended_tokens}_{'manual' if data_args.sparse_manual else 'no_manual'}_"
              f"{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_"
              f"{model_args.eol_type}_{data_args.sparse_lower_or_upper}_"
              f"{'mean' if data_args.sparse_value_mean else 'no_mean'}")
    result = (Path(data_args.dense_output_dir) / model_name / data_args.dataset_name /
              training_args.encode_type / ("filter" if data_args.is_filtered else "no_filter") /
              model_args.calculate_type / data_args.prompt_type / data_args.dataset_split)
    if training_args.task_type == "tbpr":
        result /= data_args.tbpr_type
    return result / suffix


def check_existing_shards(directory: Path, world_size: int) -> None:
    for path in directory.glob("corpus_*.pkl"):
        rank = path.stem.removeprefix("corpus_")
        if not rank.isdigit() or int(rank) >= world_size:
            raise ValueError(f"Stale shard {path}; choose a new --dense_output_dir or move old shards "
                             "before running with a different process count.")


def save_shard(directory: Path, rank: int, embeddings: np.ndarray, ids: list) -> Path:
    if embeddings.ndim != 2 or embeddings.shape[0] != len(ids):
        raise ValueError("Dense embeddings and candidate IDs are misaligned.")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"corpus_{rank}.pkl"
    with tempfile.NamedTemporaryFile(dir=directory, prefix=f".corpus_{rank}_", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            pickle.dump((np.asarray(embeddings, dtype=np.float32), ids), handle)
        except BaseException:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def encode_shard(encoder, candidates, indices, batch_size, description, rank=0):
    if batch_size < 1:
        raise ValueError("--per_device_batch_size must be positive.")
    loader = DataLoader(Subset(candidates, indices), batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=list)
    chunks, ids = [], []
    for batch in tqdm(loader, desc=description, unit="batch", position=rank):
        reps = encoder.encode_batch(batch)
        if reps.shape != (len(batch), encoder.dimension):
            raise ValueError("Dense encoder output shape does not match the candidate batch.")
        chunks.append(reps)
        ids.extend(item.identifier for item in batch)
    embeddings = np.concatenate(chunks) if chunks else np.empty((0, encoder.dimension), dtype=np.float32)
    return embeddings, ids


def run_dense_encoding(model_args, data_args, training_args, dense_args, encoder_type,
                       context_factory=setup_context, candidate_builder=build_candidates,
                       *, encoder_kwargs=None, directory_builder=None):
    label = encoder_type.model_label
    validate_task(data_args.dataset_name, training_args.task_type, training_args.encode_type,
                  data_args.encode_is_query, label)
    if data_args.per_device_batch_size < 1:
        raise ValueError("--per_device_batch_size must be positive.")
    context = context_factory()
    try:
        candidates = candidate_builder(data_args, training_args, dense_args.data_root, model_label=label)
        if not len(candidates):
            raise ValueError(f"{label} candidate dataset is empty; check the requested split and data root.")
        directory = (directory_builder or output_directory)(model_args, data_args, training_args)
        check_existing_shards(directory, context.world_size)
        dtype = torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else torch.float32)
        if context.device.type == "cpu" and dtype != torch.float32:
            raise ValueError(f"CPU {label} encoding requires FP32 (omit --fp16/--bf16).")
        encoder = encoder_type.from_pretrained(model_args.model_name_or_path, context.device, dtype,
                                               cache_dir=model_args.cache_dir, **(encoder_kwargs or {}))
        indices = shard_indices(len(candidates), context.rank, context.world_size)
        description = (f"{label.lower()}: {data_args.dataset_name}/{data_args.dataset_split}/"
                       f"{training_args.encode_type} rank={context.rank}")
        print(f"[{description}] candidates={len(candidates)}, local={len(indices)}, model={model_args.model_name_or_path}")
        embeddings, ids = encode_shard(encoder, candidates, indices, data_args.per_device_batch_size,
                                      description, context.rank)
        target = save_shard(directory, context.rank, embeddings, ids)
        print(f"[{label.lower()}] saved {target} shape={embeddings.shape} dtype={embeddings.dtype}")
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    finally:
        if context.owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
