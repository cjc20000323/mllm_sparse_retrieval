"""Native MMRet/UniME/VLM2Vec query encoding, dense search and CPU-safe Recall@K."""

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import pickle
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dense_encoding import (Candidate, DATASET_PROFILES, DenseCandidates, dataset_paths,
                            setup_context, shard_indices)
from mmret_dense import MMRetEncoder, query_prompt
from unime_dense import UniMEEncoder
from vlm2vec_dense import (QUERY_INSTRUCTIONS as VLM2VEC_INSTRUCTIONS, VLM2VecQueryEncoder,
                           query_prompt as vlm2vec_query_prompt, validate_resolution)

logger = logging.getLogger(__name__)
RECALL_K = (1, 5, 10, 100, 200, 300, 400)
MMRET_INSTRUCTIONS = {
    "ir:text": "Retrieve an image matching the given description.",
    "ir:image": "Retrieve a caption describing the given image.",
    "tbpr": "Retrieve a person image matching the given description.",
    "t2it": "Retrieve an image and text pair answering the given question.",
    "it2t": "Retrieve text answering the given question about the image.",
}


@dataclass
class DenseSearchArguments:
    mmret_query_instruction: Optional[str] = field(
        default=None, metadata={"help": "Override the task-specific MMRet query instruction (MMRet only)."})
    vlm2vec_query_instruction: Optional[str] = field(
        default=None, metadata={"help": "Override the VLM2Vec query instruction without <image> (VLM2Vec only)."})


class MMRetQueryEncoder(MMRetEncoder):
    instruction = ""

    def format_prompt(self, candidate: Candidate) -> str:
        return query_prompt(candidate.text, candidate.image is not None, self.instruction)


def validate_query_task(dataset_name, task_type, query_type):
    if dataset_name not in DATASET_PROFILES:
        raise ValueError(f"Unsupported dense search dataset: {dataset_name!r}.")
    expected = DATASET_PROFILES[dataset_name][0]
    if task_type != expected:
        raise ValueError(f"{dataset_name} requires --task_type {expected}, got {task_type!r}.")
    allowed = ("text", "image") if expected == "ir" else (
        ("image", "composed") if expected == "it2t" else ("text",))
    if query_type not in allowed:
        raise ValueError(f"{dataset_name} requires --query_type {' or '.join(allowed)}; "
                         "image/composed queries in it2t always include their text.")


def _relative_image(value, directory):
    if isinstance(value, dict):
        raw = value.get("bytes")
        if raw is not None and len(raw):
            return value
        if value.get("path"):
            return {**value, "path": _relative_image(value["path"], directory)}
    elif isinstance(value, (str, os.PathLike)):
        path = Path(value)
        return str(path if path.is_absolute() else directory / path)
    return value


class DenseQueries(DenseCandidates):
    """Read source dictionaries directly, retaining raw query IDs and image paths."""

    def __init__(self, source, name, query_type, original_image_dir):
        self.source = source
        self.name = name
        self.query_type = query_type
        self.encode_type = "image"  # Only used by the inherited IR image accessor.
        self.data_dir = Path(source.data_path)
        self.original_image_dir = Path(original_image_dir)
        if name in ("coco", "flickr"):
            self.ids = source.text_id_list if query_type == "text" else source.img_id_list
        elif DATASET_PROFILES[name][0] == "tbpr":
            self.ids = source.text_id_list
        else:
            self.ids = source.query_id_list
        if len({str(identifier) for identifier in self.ids}) != len(self.ids):
            raise ValueError(f"{name}: duplicate query IDs.")

    def __getitem__(self, index):
        identifier = self.ids[index]
        if self.name in ("coco", "flickr"):
            if self.query_type == "image":
                return super().__getitem__(index)
            return Candidate(identifier, text=self.source.text_dict[identifier])
        if DATASET_PROFILES[self.name][0] == "tbpr":
            return Candidate(identifier, text=self.source.text_dict[identifier])
        value = self.source.id2query[identifier]
        if self.name == "webqa":
            return Candidate(identifier, text=value)
        if not isinstance(value["text"], str) or value["image"] is None:
            raise ValueError(f"Query {identifier!r} requires both text and image.")
        return Candidate(identifier, text=value["text"], image=_relative_image(value["image"], self.data_dir))


def build_queries(data_args, training_args, search_args, data_root=None, dataset_module=None):
    if dataset_module is None:
        import dataset as dataset_module
    name = data_args.dataset_name
    validate_query_task(name, training_args.task_type, search_args.query_type)
    kwargs = dict(data_args=data_args,
                  use_generated_validation_data=getattr(data_args, "use_generated_validation_data", False),
                  generated_validation_data_root=getattr(data_args, "generated_validation_data_root", None))
    task = DATASET_PROFILES[name][0]
    with dataset_paths(dataset_module, data_root) as root:
        args = (name, None, data_args.dataset_split)
        if task == "ir":
            mode = "full" if search_args.query_type == "text" else "single"
            source = dataset_module.CrossModalRetrievalDataset(*args, mode, **kwargs)
        elif task == "tbpr":
            source = dataset_module.TextPersonRetrievalDataset(*args, "full", **kwargs)
        elif task == "t2it":
            source = dataset_module.Text2ImagetextRetrievalDataset(*args, "query", **kwargs)
        else:
            source = dataset_module.Imagetext2TextRetrievalDataset(*args, "query", **kwargs)
        image_dir = root / "data" / DATASET_PROFILES[name][1]
    return DenseQueries(source, name, search_args.query_type, image_dir)


def load_dense_index(path, dimension, use_gpu=False, device=None):
    import faiss

    path = Path(path)
    files = [path] if path.is_file() else sorted(path.glob("corpus*.pkl"))
    if not files:
        raise ValueError(f"No corpus*.pkl dense shards found in {path}.")
    index = faiss.IndexFlatIP(dimension)
    lookup, seen = [], set()
    for shard in files:
        with shard.open("rb") as handle:
            reps, ids = pickle.load(handle)
        reps = np.asarray(reps, dtype=np.float32)
        ids = list(ids)
        if reps.ndim != 2 or reps.shape != (len(ids), dimension):
            raise ValueError(f"Dense shard {shard} has misaligned IDs or dimension; "
                             f"expected ({len(ids)}, {dimension}), got {reps.shape}.")
        if not np.isfinite(reps).all():
            raise ValueError(f"Dense shard {shard} contains nonfinite embeddings.")
        normalized_ids = [str(identifier) for identifier in ids]
        if len(set(normalized_ids)) != len(ids) or seen.intersection(normalized_ids):
            raise ValueError(f"Duplicate candidate IDs in {shard}; check for stale or overlapping shards.")
        seen.update(normalized_ids)
        lookup.extend(ids)
        index.add(np.ascontiguousarray(reps))
    if not lookup:
        raise ValueError("Dense corpus index is empty.")
    resources = None
    if use_gpu:
        if device is None or torch.device(device).type != "cuda" or not hasattr(faiss, "StandardGpuResources"):
            logger.warning("GPU FAISS is unavailable; searching the dense index on CPU.")
        else:
            resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(resources, torch.device(device).index or 0, index)
    # Retain GPU resources for the full lifetime of the index.
    return index, lookup, resources


def search_batch(index, embeddings, lookup, query_ids, depth, remove_query=False, retrieval_batch_size=0):
    if depth < 1 or retrieval_batch_size < 0:
        raise ValueError("--depth must be positive and --retrieval_batch_size nonnegative.")
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape != (len(query_ids), index.d) or not np.isfinite(embeddings).all():
        raise ValueError("Query embeddings must be finite and align with query IDs and corpus dimension.")
    count = min(len(lookup), depth + int(remove_query))
    batch_size = retrieval_batch_size or max(1, len(query_ids))
    run = {}
    for start in range(0, len(query_ids), batch_size):
        batch_ids = query_ids[start:start + batch_size]
        scores, positions = index.search(np.ascontiguousarray(embeddings[start:start + batch_size]), count)
        for qid, row_scores, row_positions in zip(batch_ids, scores, positions):
            docs = {}
            for score, position in zip(row_scores, row_positions):
                if position < 0 or position >= len(lookup):
                    continue
                doc = str(lookup[position])
                if remove_query and doc == str(qid):
                    continue
                docs[doc] = float(score)
                if len(docs) == depth:
                    break
            run[qid] = docs
    return run


def recall_counts(source, name, query_type, run):
    counts = np.zeros(len(RECALL_K) + 1, dtype=np.int64)
    for qid, docs in run.items():
        if name in ("coco", "flickr"):
            targets = source.get_target(qid, query_type)
        elif DATASET_PROFILES[name][0] == "tbpr":
            targets = source.get_target_from_text(qid)
        else:
            targets = source.get_target(qid)
        targets = targets if isinstance(targets, (list, tuple, set)) else [targets]
        targets = {str(target) for target in targets}
        ranking = sorted(docs, key=docs.get, reverse=True)
        if DATASET_PROFILES[name][0] == "tbpr":
            ranking = [str(source.get_target_from_image(identifier)) for identifier in ranking]
        for index, k in enumerate(RECALL_K):
            counts[index] += bool(targets.intersection(ranking[:k]))
        counts[-1] += 1
    return counts


def reduce_recall_counts(counts, context):
    counts = torch.as_tensor(counts, dtype=torch.int64, device=context.device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    return counts.cpu().numpy()


def result_directory(model_args, data_args, search_args, *, model_type=None, dense_args=None):
    if search_args.save_dir:
        return Path(search_args.save_dir)
    model_name = str(model_args.model_name_or_path).replace("\\", "/").rstrip("/").split("/")[-1]
    directory = Path("search_results") / model_name / data_args.dataset_name / search_args.query_type / data_args.dataset_split
    if model_type == "vlm2vec":
        resolution = dense_args.vlm2vec_image_resolution
        validate_resolution(resolution)
        directory /= f"vlm2vec_{resolution}"
    return directory


def write_recall(counts, directory):
    import pandas as pd

    if not counts[-1]:
        raise ValueError("Cannot evaluate an empty query set.")
    recalls = counts[:-1] / counts[-1]
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "dense.xlsx"
    # Match the existing dense.xlsx layout: K header, followed by dense Recall@K.
    pd.DataFrame([recalls], columns=list(RECALL_K)).to_excel(target, index=False)
    print(f"[dense] queries={counts[-1]} " + " ".join(f"R@{k}={value:.6f}" for k, value in zip(RECALL_K, recalls)))
    print(f"[dense] saved {target}")
    return target


def run_dense_search(model_args, data_args, search_args, training_args, dense_args, native_args, model_type,
                     *, context_factory=setup_context, query_builder=build_queries, encoder_type=None):
    validate_query_task(data_args.dataset_name, training_args.task_type, search_args.query_type)
    if not search_args.passage_reps:
        raise ValueError("Native dense search requires --passage_reps pointing to the matching corpus index.")
    if data_args.per_device_batch_size < 1 or search_args.depth < 1 or search_args.retrieval_batch_size < 0:
        raise ValueError("Batch size and depth must be positive; retrieval_batch_size must be nonnegative.")
    if model_type not in ("mmret", "unime", "vlm2vec"):
        raise ValueError(f"Unsupported native dense model: {model_type}.")
    if model_type != "mmret" and native_args.mmret_query_instruction is not None:
        raise ValueError("--mmret_query_instruction is only supported by MMRet.")
    if model_type != "vlm2vec" and native_args.vlm2vec_query_instruction is not None:
        raise ValueError("--vlm2vec_query_instruction is only supported by VLM2Vec.")
    instruction = None
    encoder_kwargs = {}
    if model_type == "mmret":
        key = f"ir:{search_args.query_type}" if training_args.task_type == "ir" else training_args.task_type
        instruction = native_args.mmret_query_instruction
        if instruction is None:
            instruction = MMRET_INSTRUCTIONS[key]
        query_prompt("", False, instruction)  # Validate before loading model weights.
    elif model_type == "vlm2vec":
        validate_resolution(dense_args.vlm2vec_image_resolution)
        encoder_kwargs["image_resolution"] = dense_args.vlm2vec_image_resolution
        key = f"ir:{search_args.query_type}" if training_args.task_type == "ir" else training_args.task_type
        instruction = native_args.vlm2vec_query_instruction
        if instruction is None:
            instruction = VLM2VEC_INSTRUCTIONS[key]
        vlm2vec_query_prompt("", False, instruction)
    context = context_factory()
    try:
        queries = query_builder(data_args, training_args, search_args, dense_args.data_root)
        if not len(queries):
            raise ValueError("Dense query dataset is empty; check the split and data root.")
        dtype = torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else torch.float32)
        if context.device.type == "cpu" and dtype != torch.float32:
            raise ValueError("CPU dense query encoding requires FP32 (omit --fp16/--bf16).")
        encoder_type = encoder_type or {"mmret": MMRetQueryEncoder, "unime": UniMEEncoder,
                                        "vlm2vec": VLM2VecQueryEncoder}[model_type]
        encoder = encoder_type.from_pretrained(model_args.model_name_or_path, context.device, dtype,
                                               cache_dir=model_args.cache_dir, **encoder_kwargs)
        if instruction is not None:
            encoder.instruction = instruction
            print(f"[{model_type}] query instruction: {instruction}")
        index, lookup, gpu_resources = load_dense_index(search_args.passage_reps, encoder.dimension,
                                                       search_args.use_gpu, context.device)
        indices = shard_indices(len(queries), context.rank, context.world_size)
        loader = DataLoader(Subset(queries, indices), batch_size=data_args.per_device_batch_size,
                            shuffle=False, num_workers=0, collate_fn=list)
        counts = np.zeros(len(RECALL_K) + 1, dtype=np.int64)
        print(f"[{model_type}] {data_args.dataset_name}/{data_args.dataset_split}, "
              f"query={search_args.query_type}, queries={len(queries)}, local={len(indices)}, corpus={len(lookup)}, "
              f"rank={context.rank}/{context.world_size}")
        for batch in tqdm(loader, desc=f"{model_type} queries rank={context.rank}", disable=search_args.quiet):
            reps = encoder.encode_batch(batch)
            run = search_batch(index, reps, lookup, [item.identifier for item in batch], search_args.depth,
                               search_args.remove_query, search_args.retrieval_batch_size)
            counts += recall_counts(queries.source, queries.name, search_args.query_type, run)
        counts = reduce_recall_counts(counts, context)
        if context.rank == 0:
            write_recall(counts, result_directory(model_args, data_args, search_args,
                                                model_type=model_type, dense_args=dense_args))
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return counts
    finally:
        if context.owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
