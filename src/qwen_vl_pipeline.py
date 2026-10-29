"""Shared Qwen2/Qwen2.5/Qwen3-VL path for encode.py, search.py and rerank.py.

Dataset annotations, corpus pickle/impact formats and fusion equations match
the existing program. Non-Qwen entrypoints continue through their original path.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from qwen_vl_encoding import NativeQwenEncoder, SparseEncoder
from qwen_vl_paths import artifact_root, resolve_sparse_index
from qwen_vl_prompts import (METHODS, PROMPT_VERSION, QWEN2_PROMPT_VERSION, QWEN3_PROMPT_VERSION,
                            PROMPT_VERSIONS,
                            MODEL_TYPES, TURN_END, model_family, render_user, retrieval_prompt)

MANIFEST = "qwen25_prompt_manifest.json"
QWEN2_MANIFEST = "qwen2_prompt_manifest.json"
QWEN3_MANIFEST = "qwen3_prompt_manifest.json"


def manifest_name(metadata):
    return {PROMPT_VERSION: MANIFEST, QWEN2_PROMPT_VERSION: QWEN2_MANIFEST,
            QWEN3_PROMPT_VERSION: QWEN3_MANIFEST}[metadata["config"]["version"]]


def model_class(config):
    """Use the complete native multimodal model, including Qwen3 DeepStack."""
    import transformers
    names = {"qwen2_vl": "Qwen2VLForConditionalGeneration",
             "qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration",
             "qwen3_vl": "Qwen3VLForConditionalGeneration"}
    family = model_family(config)
    if family not in MODEL_TYPES:
        raise ValueError(f"Unsupported Qwen model_type={family}; use Qwen2-VL, Qwen2.5-VL or Qwen3-VL Dense Instruct")
    return getattr(transformers, names[family])


def configure_pixels(processor, data):
    """Align effective fast/slow image processor pixel-area limits."""
    image = processor.image_processor
    size = dict(getattr(image, "size", {}) or {})
    low = data.qwen_min_pixels or getattr(image, "min_pixels", None) or size.get("shortest_edge")
    high = data.qwen_max_pixels or getattr(image, "max_pixels", None) or size.get("longest_edge")
    if low is None or high is None or low > high:
        raise ValueError("Invalid effective Qwen image pixel limits; min_pixels must not exceed max_pixels")
    image.size = dict(size, shortest_edge=low, longest_edge=high)
    # Slow processors still read these attributes; fast ones accept both forms.
    image.min_pixels, image.max_pixels = low, high


def init_process():
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0"))) if torch.cuda.is_available() else torch.device("cpu")
    return rank, world, device


def load_qwen_processor(model_path, *, min_pixels=0, max_pixels=0):
    """Load/validate native conversation and image settings without GPU weights."""
    from transformers import AutoConfig, AutoProcessor
    if min(min_pixels, max_pixels) < 0 or (min_pixels and max_pixels and min_pixels > max_pixels):
        raise ValueError("Qwen pixel limits must be nonnegative and min_pixels must not exceed max_pixels")
    config = AutoConfig.from_pretrained(model_path)
    model_class(config)
    family = model_family(config)
    if family == "qwen3_vl" and any("thinking" in str(value).lower() for value in (
            model_path, getattr(config, "_name_or_path", ""))):
        raise ValueError("Qwen3-VL retrieval supports Instruct checkpoints only; Thinking is unsupported")
    pixels = {name: value for name, value in (("min_pixels", min_pixels), ("max_pixels", max_pixels)) if value > 0}
    processor = AutoProcessor.from_pretrained(model_path, **pixels)
    if family in {"qwen2_vl", "qwen3_vl"}:
        from types import SimpleNamespace
        configure_pixels(processor, SimpleNamespace(qwen_min_pixels=min_pixels, qwen_max_pixels=max_pixels))
        if not render_user(processor, "").endswith(TURN_END):
            detail = "; Thinking templates are unsupported" if family == "qwen3_vl" else ""
            raise ValueError(f"{family} requires the native Instruct assistant prefix{detail}")
    return config, processor


def load_native_qwen(model_path, *, dtype, device_map=None, device=None, min_pixels=0,
                     max_pixels=0, lora_model_path=None, config=None, processor=None):
    """Load a complete Qwen model; callers own devices and process groups."""
    if config is None or processor is None:
        config, processor = load_qwen_processor(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
    kwargs = dict(config=config, torch_dtype=dtype, attn_implementation="sdpa", low_cpu_mem_usage=True)
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = model_class(config).from_pretrained(model_path, **kwargs).eval()
    if device_map is None and device is not None:
        model = model.to(device)
    if lora_model_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_model_path).merge_and_unload().eval()
    return model, processor


def load_runtime(model_args, data, training):
    rank, world, device = init_process()
    if (model_args.eol_type, model_args.calculate_type) not in METHODS:
        raise ValueError(f"Qwen supports only {sorted(METHODS)}")
    if model_args.use_output_embedding_cluster or data.sparse_type != "single" or data.sparse_logits_multiply_view_count:
        raise ValueError("Qwen adaptation supports single sparse representations without output clustering or the LLaVA-only x5 ablation")
    if data.use_few_shot_examples or data.use_source_dataset_subset or data.prompt_generation:
        raise ValueError("Qwen native prompts do not support the LLaVA few-shot/subset/generated-prompt experiments")
    if data.per_device_batch_size <= 0 or min(data.qwen_min_pixels, data.qwen_max_pixels) < 0:
        raise ValueError("Batch size must be positive and Qwen pixel limits must be nonnegative")
    if data.sparse_value_type not in {"replace", "sum", "max"} or data.sparse_lower_or_upper not in {"lower", "upper"}:
        raise ValueError("Unsupported sparse aggregation or letter-case policy")
    if min(data.sparse_length, data.text_sparse_length, data.image_sparse_length) <= 0 or data.num_expended_tokens < 0:
        raise ValueError("Sparse lengths must be positive and expansion count must be nonnegative")
    if data.qwen_min_pixels and data.qwen_max_pixels and data.qwen_min_pixels > data.qwen_max_pixels:
        raise ValueError("qwen_min_pixels must not exceed qwen_max_pixels")
    dtype = torch.bfloat16 if training.bf16 else torch.float16 if training.fp16 else torch.float32
    model, processor = load_native_qwen(
        model_args.model_name_or_path, dtype=dtype, device_map={"": device},
        min_pixels=data.qwen_min_pixels, max_pixels=data.qwen_max_pixels,
        lora_model_path=model_args.lora_model_path if getattr(model_args, "lora", False) else None,
    )
    return model, processor, rank, world


def build_dataset(data, task, processor, *, corpus=False, query_type="text", split=None, name=None, generated=False):
    from dataset import (CrossModalRetrievalDataset, TextPersonRetrievalDataset,
                         Text2ImagetextRetrievalDataset, Imagetext2TextRetrievalDataset,
                         ComposedTextImageRetrievalDataset)
    name, split = name or data.dataset_name, split or data.dataset_split
    kwargs = dict(use_generated_validation_data=generated, generated_validation_data_root=data.generated_validation_data_root)
    if task in {"ir", "tbpr"}:
        cls = CrossModalRetrievalDataset if task == "ir" else TextPersonRetrievalDataset
        mode = "single" if query_type == "image" else "full"
    elif task in {"t2it", "it2t"}:
        cls = Text2ImagetextRetrievalDataset if task == "t2it" else Imagetext2TextRetrievalDataset
        mode = "corpus" if corpus else "query"
    elif task == "cir":
        cls, mode = ComposedTextImageRetrievalDataset, "image" if corpus else "composed"
    else:
        raise ValueError(f"Unsupported retrieval task: {task}")
    return cls(name, processor, split, mode, **kwargs)


def dataset_records(dataset, task, kind):
    """Keep native annotation IDs for evaluation; serialize/search their strings."""
    records = []
    if task in {"ir", "tbpr"}:
        image = kind == "image"
        ids = dataset.img_id_list if image else dataset.text_id_list
        # Resolve image paths using the dataset's own path policy, including
        # generated validation images. No hard-coded checkpoint/data prefix.
        if image:
            dataset.mode = "single"
            for index, identifier in enumerate(ids):
                records.append(dict(id=str(identifier), native_id=identifier, image=dataset[index][1], text=""))
        else:
            records = [dict(id=str(identifier), native_id=identifier, text=dataset.text_dict[identifier]) for identifier in ids]
    elif task in {"t2it", "it2t"}:
        corpus = kind == "corpus"
        ids = dataset.candidate_id_list if corpus else dataset.query_id_list
        values = dataset.id2candidate if corpus else dataset.id2query
        for identifier in ids:
            value = values[identifier]
            record = dict(value) if isinstance(value, dict) else {"text": value}
            records.append(dict(record, id=str(identifier), native_id=identifier))
    elif task == "cir":
        corpus = kind == "image"
        dataset.mode = "image" if corpus else "composed"
        for index in range(len(dataset)):
            text, image, _, _, image_id, composed_id, dress = dataset[index]
            if isinstance(image, (str, Path)) and not Path(image).is_file() and hasattr(dataset, "data_path"):
                # The legacy Fashion-IQ iterator emits a project-relative path;
                # annotations may instead live under arguments.fashion_iq_file_path.
                image = Path(dataset.data_path) / "images" / "images" / dataset.img_dict[image_id]
            identifier = image_id if corpus else composed_id
            records.append(dict(id=str(identifier), native_id=identifier, text=text, image=image,
                                composed=not corpus, reference_id=str(image_id), dress_type=dress))
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("Dataset contains duplicate retrieval IDs")
    return records


def corpus_kind(task, query_type):
    if task == "ir":
        return "text" if query_type == "image" else "image"
    return "image" if task in {"tbpr", "cir"} else "corpus"


def query_kind(task, query_type):
    return query_type if task == "ir" else "text" if task == "tbpr" else "composed" if task == "cir" else "query"


def prompt_metadata(model_args, data, training, processor, kind, *, family=None):
    # Corpus/query prompts differ by design. Store BOTH role specifications in
    # the fingerprint so a query can verify exactly the corpus it is searching.
    task = training.task_type
    examples = [dict(text="<sent>"), dict(text="", image="<image>"), dict(text="<sent>", image="<image>", composed=True, dress_type="{}")]
    specs = [list(retrieval_prompt(record, task, data, processor, include_views=model_args.calculate_type == "concat")) for record in examples if task != "cir" or record.get("image")]
    config = {key: getattr(data, key, None) for key in (
        "prompt_type", "tbpr_type", "cir_type", "random_view_seed", "reps_loc",
        "sparse_length", "text_sparse_length", "image_sparse_length", "sparse_value_type",
        "sparse_lower_or_upper", "sparse_value_mean", "sparse_manual", "is_filtered", "num_expended_tokens",
    )}
    tokenizer_state = json.dumps(processor.tokenizer.get_vocab(), sort_keys=True, ensure_ascii=False)
    image_options = {key: getattr(processor.image_processor, key, None) for key in (
        "min_pixels", "max_pixels", "patch_size", "temporal_patch_size", "merge_size",
        "image_mean", "image_std", "do_resize", "do_rescale", "rescale_factor", "do_normalize",
    )}
    # File identity detects a checkpoint/adapter replaced in place without
    # reading multi-GB weight files. Hub revisions use their resolved config.
    weights = []
    for identity in (model_args.model_name_or_path,
                     getattr(model_args, "lora_model_path", None) if getattr(model_args, "lora", False) else None):
        if identity and Path(identity).is_dir():
            for path in sorted(Path(identity).iterdir()):
                if path.is_file() and (path.suffix in {".safetensors", ".bin"} or path.name == "config.json"):
                    stat = path.stat()
                    weights.append([str(path), stat.st_size, stat.st_mtime_ns])
    family = family or {"Qwen2VLProcessor": "qwen2_vl", "Qwen3VLProcessor": "qwen3_vl"}.get(
        type(processor).__name__, "qwen2_5_vl")
    if family not in MODEL_TYPES:
        raise ValueError(f"Unsupported Qwen family: {family}")
    # Preserve existing Qwen2.5/Qwen3 hashes. Qwen2 also fingerprints the
    # effective fast-processor size under its own version/manifest namespace.
    if family in {"qwen2_vl", "qwen3_vl"}:
        image_options["size"] = getattr(processor.image_processor, "size", None)
        image_options["processor_class"] = type(processor.image_processor).__name__
        config["model_type"] = family
    config.update(version=PROMPT_VERSIONS[family],
                  model=str(model_args.model_name_or_path), task=task,
                  dataset=data.dataset_name, split=data.dataset_split,
                  method=[model_args.eol_type, model_args.calculate_type], prompts=specs,
                  lora=getattr(model_args, "lora_model_path", None) if getattr(model_args, "lora", False) else None,
                  tokenizer_class=type(processor.tokenizer).__name__,
                  tokenizer_sha256=hashlib.sha256(tokenizer_state.encode()).hexdigest(),
                  chat_template=getattr(processor, "chat_template", None) or getattr(processor.tokenizer, "chat_template", None),
                  vision_processor=image_options, checkpoint_files=weights,
                  min_pixels=getattr(processor.image_processor, "min_pixels", None),
                  max_pixels=getattr(processor.image_processor, "max_pixels", None))
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return dict(fingerprint=digest, config=config, kind=kind)


def validate_manifest(path, expected):
    root = artifact_root(path)
    manifest = root / manifest_name(expected)
    if not manifest.is_file():
        raise ValueError(f"Missing {manifest}; rebuild the old Qwen corpus with the native prompt adapter")
    actual = json.loads(manifest.read_text(encoding="utf-8"))
    if actual["fingerprint"] != expected["fingerprint"] or actual["kind"] != expected["kind"]:
        raise ValueError(f"Qwen prompt/configuration mismatch at {root}; encode, index and search must use the same settings")
    return actual


def artifact_dirs(model_args, data, training):
    manual = "manual" if data.sparse_manual else "no_manual"
    lengths = f"{data.text_sparse_length}_{data.image_sparse_length}" if data.sparse_manual else str(data.sparse_length)
    leaf = f"{data.num_expended_tokens}_{manual}_{lengths}_{data.sparse_value_type}_no_cluster_{data.reps_loc}_{model_args.eol_type}_{data.sparse_lower_or_upper}_{'mean' if data.sparse_value_mean else 'no_mean'}"
    relative = Path(Path(model_args.model_name_or_path).name) / data.dataset_name / training.encode_type / ("filter" if data.is_filtered else "no_filter") / model_args.calculate_type / data.prompt_type / data.dataset_split
    if training.task_type in {"tbpr", "cir"}:
        relative /= data.tbpr_type if training.task_type == "tbpr" else data.cir_type
    return [Path(base) / relative / leaf for base in (data.dense_output_dir, data.sparse_output_dir)]


def encode_batches(encoder, records, data):
    sparse = SparseEncoder(encoder.processor.tokenizer, data, concat=encoder.concat)
    for start in tqdm(range(0, len(records), data.per_device_batch_size), desc="Qwen query/corpus encoding"):
        batch = records[start:start + data.per_device_batch_size]
        logits, dense = encoder.encode(batch)
        vectors = [sparse.vector(row, record) for row, record in zip(logits, batch)]
        yield batch, dense.float().cpu().numpy(), vectors


def prepare_encode_outputs(roots, metadata, *, query=False):
    """Reset this encoding's artifacts once, on rank zero before any writes.

    Resolve the explicitly configured leaf directories before selecting files;
    linked artifacts must not redirect cleanup to another dataset or directory.
    Old indexes are kept as backups, but cannot be used as the current index.
    """
    dense, sparse = (Path(root).resolve() for root in roots)
    for root in (dense, sparse):
        root.mkdir(parents=True, exist_ok=True)
    collection = sparse / "collection"
    manifests = list(dict.fromkeys(root / manifest_name(metadata) for root in (dense, sparse)))
    if query:
        artifacts = [dense / "query.pkl", sparse / "query.tsv"]
    else:
        artifacts = [*dense.glob("corpus*.pkl"), *collection.glob("corpus*.jsonl")]
    indexes = (collection / "index", sparse / "index")
    # Validate all selected paths before invalidating manifests or removing data.
    checked = [*manifests, *artifacts]
    if not query:
        checked.extend((collection, *indexes))
    for path in checked:
        if path.resolve() != path:
            raise ValueError(f"Refusing to overwrite a linked Qwen artifact: {path}")
        if path not in (collection, *indexes) and path.exists() and not path.is_file():
            raise ValueError(f"Expected a Qwen artifact file: {path}")
    if not query and collection.exists() and not collection.is_dir():
        raise ValueError(f"Expected a Qwen collection directory: {collection}")
    # A failed restart must not retain the previous run's completion marker.
    for path in manifests:
        path.unlink(missing_ok=True)
    if not query:
        for index in indexes:
            if index.exists():
                backup = sparse / f"index.stale_{uuid4().hex}"
                while backup.exists():
                    backup = sparse / f"index.stale_{uuid4().hex}"
                # Both checked index paths are within the sparse leaf; backups
                # stay outside collection so Anserini cannot ingest their files.
                index.rename(backup)
                print(f"Previous Qwen index backed up to {backup}; rebuild index after encoding")
    for path in artifacts:
        path.unlink(missing_ok=True)


def run_encode(model_args, data, training):
    if data.use_generated_validation_data:
        data = replace(data, dataset_split="val")
    model, processor, rank, world = load_runtime(model_args, data, training)
    task, kind = training.task_type, training.encode_type
    valid_kinds = {"ir": {"image", "text"}, "tbpr": {"image", "text"},
                   "cir": {"image", "composed"}, "t2it": {"corpus", "query"}, "it2t": {"corpus", "query"}}
    if kind not in valid_kinds.get(task, set()):
        raise ValueError(f"Invalid encode_type={kind} for task={task}; choose {valid_kinds.get(task)}")
    if data.encode_is_query and world != 1:
        raise ValueError("Query artifact encoding requires one worker; online search shards queries automatically")
    dataset = build_dataset(data, task, processor, corpus=not data.encode_is_query,
                            query_type=kind, generated=data.use_generated_validation_data)
    records = dataset_records(dataset, task, kind)[rank::world]
    metadata = prompt_metadata(model_args, data, training, processor, kind, family=model_family(model.config))
    roots = artifact_dirs(model_args, data, training)
    # Only rank zero resets old files. The collective also keeps other ranks
    # from opening new shards until cleanup has finished successfully.
    error = None
    if rank == 0:
        try:
            prepare_encode_outputs(roots, metadata, query=data.encode_is_query)
        except Exception as exc:
            error = str(exc)
    if dist.is_initialized():
        packet = [error]
        dist.broadcast_object_list(packet, src=0)
        error = packet[0]
    if error:
        raise ValueError(error)
    encoder = NativeQwenEncoder(model, processor, model_args, data, task)
    rows, ids = [], []
    sparse_name = "query.tsv" if data.encode_is_query else f"corpus_{rank}.jsonl"
    collection = roots[1] if data.encode_is_query else roots[1] / "collection"
    collection.mkdir(exist_ok=True)
    with (collection / sparse_name).open("w", encoding="utf-8") as stream:
        for batch, dense, vectors in encode_batches(encoder, records, data):
            rows.append(dense)
            ids.extend(record["id"] for record in batch)
            for record, vector in zip(batch, vectors):
                if data.encode_is_query:
                    topic = " ".join(" ".join([term] * value) for term, value in vector.items())
                    stream.write(f"{record['id']}\t{topic}\n")
                else:
                    stream.write(json.dumps(dict(id=record["id"], contents="", vector=vector), ensure_ascii=False) + "\n")
    dense = np.concatenate(rows) if rows else np.empty((0, getattr(model.config, "text_config", model.config).hidden_size), dtype="float32")
    with (roots[0] / ("query.pkl" if data.encode_is_query else f"corpus_{rank}.pkl")).open("wb") as stream:
        pickle.dump((dense, ids), stream)
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        for root in roots:
            (root / manifest_name(metadata)).write_text(json.dumps(dict(metadata, num_shards=world, examples=encoder.last_prompts[:2]), indent=2, ensure_ascii=False), encoding="utf-8")
        print("Qwen corpus artifacts:", *(str(root) for root in roots), sep="\n")


def _run(scores):
    values = list(scores.values())
    return dict(docs=scores, min_score=min(values, default=0.0), max_score=max(values, default=0.0))


class CorpusSearch:
    def __init__(self, dense_path, sparse_path, expected, search, rank=0):
        self.dense, self.sparse, self.lookup = None, None, []
        self.search = search
        if dense_path:
            metadata = validate_manifest(dense_path, expected)
            files = [Path(dense_path)] if Path(dense_path).is_file() else sorted(Path(dense_path).glob("corpus*.pkl"))
            if len(files) != metadata["num_shards"]:
                raise ValueError("Dense corpus shard count differs from its Qwen manifest")
            from tevatron.retriever.searcher import FaissFlatSearcher
            for path in files:
                with path.open("rb") as stream:
                    reps, ids = pickle.load(stream)
                if len(reps) != len(ids):
                    raise ValueError(f"Dense rows and document IDs differ in {path}")
                if self.dense is None:
                    self.dense = FaissFlatSearcher(reps)
                self.dense.add(np.asarray(reps, dtype="float32"))
                self.lookup.extend(str(identifier) for identifier in ids)
            if len(set(self.lookup)) != len(self.lookup):
                raise ValueError("Duplicate document IDs across corpus shards")
            if not self.lookup:
                raise ValueError("Dense corpus has no documents")
            if search.use_gpu and torch.cuda.is_available():
                import faiss
                self.gpu_resources = faiss.StandardGpuResources()
                self.dense.index = faiss.index_cpu_to_gpu(self.gpu_resources, int(os.environ.get("LOCAL_RANK", "0")), self.dense.index)
        if sparse_path:
            validate_manifest(sparse_path, expected)
            index = resolve_sparse_index(sparse_path)
            from pyserini.search.lucene import LuceneImpactSearcher
            from pyserini.analysis import JWhiteSpaceAnalyzer
            self.sparse = LuceneImpactSearcher(str(index), None)
            self.analyzer = JWhiteSpaceAnalyzer()
            self.sparse.set_analyzer(self.analyzer)
        if self.dense is None and self.sparse is None:
            raise ValueError("Provide --passage_reps and/or --sparse_index")

    def close(self):
        for resource in (self.sparse, getattr(self, "analyzer", None)):
            close = getattr(resource, "close", None)
            if callable(close):
                close()
        self.sparse, self.dense = None, None
        self.analyzer = None
        self.gpu_resources = None

    def retrieve(self, encoder, records, data):
        dense_run, sparse_run = {}, {}
        for batch, reps, vectors in encode_batches(encoder, records, data):
            identifiers = [record["id"] for record in batch]
            if self.dense is not None:
                depth = min(self.search.depth, len(self.lookup))
                scores, indices = self.dense.search(reps, depth)
                for identifier, values, hits in zip(identifiers, scores, indices):
                    dense_run[identifier] = _run({self.lookup[int(i)]: float(value) for i, value in zip(hits, values) if i >= 0})
            if self.sparse is not None:
                topics = [" ".join(" ".join([term] * value) for term, value in vector.items()) for vector in vectors]
                results = self.sparse.batch_search(topics, identifiers, self.search.depth, threads=self.search.threads)
                for identifier in identifiers:
                    sparse_run[identifier] = _run({hit.docid: float(hit.score) for hit in results[identifier]})
            for record in batch:
                for run in (dense_run, sparse_run):
                    run.setdefault(record["id"], _run({}))
                    # Text/image IDs belong to different namespaces. The legacy
                    # cross-modal search deliberately does NOT remove equal IDs;
                    # only CIR has an actual reference image to exclude.
                    if self.search.remove_query and "reference_id" in record:
                        excluded = record["reference_id"]
                        run[record["id"]] = _run({key: value for key, value in run[record["id"]]["docs"].items() if key != excluded})
        return dense_run, sparse_run


def recall(dataset, records, run, task, query_type):
    from metrices import get_recall_k_settings
    cutoffs = get_recall_k_settings(dataset.data_name)
    hits = {k: 0 for k in cutoffs}
    categories = {}
    native_images = {str(key): key for key in dataset.img_id_list} if task == "tbpr" else {}
    for record in records:
        native = record["native_id"]
        if task == "ir":
            target = dataset.get_target(native, query_type)
        elif task == "tbpr":
            target = dataset.get_target_from_text(native)
        else:
            target = dataset.get_target(native)
        targets = {str(value) for value in target} if isinstance(target, (list, tuple, set, np.ndarray)) else {str(target)}
        docs = sorted(run.get(record["id"], {}), key=run.get(record["id"], {}).get, reverse=True)
        if task == "tbpr":
            docs = [str(dataset.get_target_from_image(native_images[key])) for key in docs]
        group = record.get("dress_type") if task == "cir" else None
        if group:
            categories.setdefault(group, {"count": 0, **{k: 0 for k in cutoffs}})["count"] += 1
        for k in cutoffs:
            hit = bool(targets.intersection(docs[:k]))
            hits[k] += hit
            if group:
                categories[group][k] += hit
    result = {f"Recall@{k}": 100 * value / max(len(records), 1) for k, value in hits.items()}
    if categories:
        result["by_category"] = {group: {f"Recall@{k}": 100 * values[k] / values["count"] for k in cutoffs} for group, values in categories.items()}
    return result


def _gather(run):
    if not dist.is_initialized():
        return run
    parts = [None] * dist.get_world_size()
    dist.all_gather_object(parts, run)
    return {key: value for part in parts for key, value in part.items()}


def _baseline_recalls(dataset, records, dense, sparse, task, query_type):
    return {name: recall(dataset, records, {key: value["docs"] for key, value in run.items()}, task, query_type)
            for name, run in (("dense", dense), ("sparse", sparse))}


def _alpha_quality(metric, task):
    if task == "cir":
        values = [group[f"Recall@{k}"] for group in metric.get("by_category", {}).values() for k in (10, 50)]
        return float(np.mean(values)) if values else 0.0
    return sum(metric[f"Recall@{k}"] for k in (1, 5, 10))


def _select_alpha(dataset, records, dense, sparse, task, query_type):
    """Keep nine small metric records, but only the current best candidate run."""
    from hybrid import fuse
    best, candidates = None, []
    for alpha in (n / 10 for n in range(1, 10)):
        fused = fuse([dense, sparse], [alpha, 1 - alpha])
        metric = recall(dataset, records, fused, task, query_type)
        score = _alpha_quality(metric, task)
        candidates.append(dict(alpha=alpha, metric=metric, score=score))
        if best is None or score > best[3]:
            best = alpha, fused, metric, score
    return best, candidates


def _metric_rows(metrics, data, source, alpha, *, score=None, selected=False):
    rows = []
    for method, metric in metrics.items():
        groups = [("all", metric), *sorted(metric.get("by_category", {}).items())]
        for category, values in groups:
            rows.append(dict(dataset=data.dataset_name, split=data.dataset_split, selection_source=source,
                             alpha=alpha, beta=round(1 - alpha, 12), method=method, category=category,
                             selection_score=score if method == "hybrid" else None,
                             selected=selected and method == "hybrid",
                             **{key: value for key, value in values.items() if key.startswith("Recall@")}))
    return rows


def _write_metrics_csv(path, rows):
    recalls = sorted({key for row in rows for key in row if key.startswith("Recall@")},
                     key=lambda key: int(key.split("@")[1]))
    fields = ["dataset", "split", "selection_source", "alpha", "beta", "method", "category",
              "selection_score", "selected", *recalls]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _search_status(status):
    """Broadcast small control results, including root-side failures, to every rank."""
    packet = [status]
    if dist.is_initialized():
        dist.broadcast_object_list(packet, src=0)
    if packet[0].get("error"):
        raise RuntimeError(f"Qwen search failed on rank 0: {packet[0]['error']}")
    return packet[0]


def _scatter_rerank_run(run, query_ids, rank, world):
    if not dist.is_initialized():
        return run
    parts = [{key: run[key] for key in query_ids[worker::world]} for worker in range(world)] if rank == 0 else None
    received = [None]
    dist.scatter_object_list(received, parts, src=0)
    return received[0]


def run_search(model_args, data, search, training, *, rerank=False, time_options=None):
    from hybrid import fuse, write_trec_run
    if time_options is not None and time_options.estimate_time_only:
        raise ValueError("The standalone Qwen path currently supports complete retrieval/reranking, not the LLaVA time-estimation experiment")
    task = training.task_type
    val_paths = (search.val_passage_reps, search.val_sparse_index)
    use_validation = not search.use_fixed_alpha and any(path is not None for path in val_paths)
    if use_validation and not all(val_paths):
        raise ValueError("Qwen validation alpha selection requires both --val_passage_reps and --val_sparse_index; omit both to select on the target split")
    if rerank and search.rerank_type != "pointwise":
        raise ValueError("Qwen listwise reranking has no implemented scoring definition")
    if not 0 <= search.alpha <= 1 or search.depth <= 0 or data.per_device_batch_size <= 0:
        raise ValueError("alpha must be in [0,1] and retrieval depth/batch size must be positive")
    valid_queries = {"ir": {"image", "text"}, "tbpr": {"text"}, "t2it": {"text"}, "it2t": {"image", "composed"}, "cir": {"composed"}}
    if search.query_type not in valid_queries.get(task, set()):
        raise ValueError("Query direction does not match the retrieval task")
    if rerank:
        from rerank_prior_cache import validate_prior_cache_options
        if search.modify_type not in {"no", "modify_single"}:
            raise ValueError("Qwen caption correction supports modify_type=no or modify_single")
        validate_prior_cache_options(search.use_rerank_prior_cache, search.modify_type, search.rerank_template,
                                     task, search.query_type, search.rerank_type)
    model, processor, rank, world = load_runtime(model_args, data, training)
    if rerank:
        from qwen_vl_prompts import rerank_template
        rerank_template(processor, task, search.query_type, search.rerank_template, caption="caption_generation" in search.rerank_template)
    encoder = NativeQwenEncoder(model, processor, model_args, data, task)
    target_kind = corpus_kind(task, search.query_type)
    dataset = build_dataset(data, task, processor, query_type=search.query_type)
    all_records = dataset_records(dataset, task, query_kind(task, search.query_type))
    expected = prompt_metadata(model_args, data, training, processor, target_kind, family=model_family(model.config))
    index = CorpusSearch(search.passage_reps, search.sparse_index, expected, search, rank)
    try:
        dense, sparse = index.retrieve(encoder, all_records[rank::world], data)
    finally:
        index.close()
    dense, sparse = _gather(dense), _gather(sparse)
    selection_data, selection_dataset, selection_records = data, dataset, all_records
    selection_dense, selection_sparse = dense, sparse
    if use_validation:
        name = search.validation_dataset_name or ({"ir": "flickr", "tbpr": "RSTPReid", "t2it": "webqa", "it2t": "llava"}.get(task, data.dataset_name) if search.use_fixed_validation_dataset else data.dataset_name)
        selection_data = replace(data, dataset_name=name, dataset_split="val")
        selection_dataset = build_dataset(selection_data, task, processor, query_type=search.query_type, generated=data.use_generated_validation_data)
        selection_records = dataset_records(selection_dataset, task, query_kind(task, search.query_type))
        val_encoder = NativeQwenEncoder(model, processor, model_args, selection_data, task)
        val_index = CorpusSearch(search.val_passage_reps, search.val_sparse_index, prompt_metadata(model_args, selection_data, training, processor, target_kind, family=model_family(model.config)), search, rank)
        try:
            selection_dense, selection_sparse = val_index.retrieve(val_encoder, selection_records[rank::world], selection_data)
        finally:
            val_index.close()
        selection_dense, selection_sparse = _gather(selection_dense), _gather(selection_sparse)

    output = Path(training.output_dir)
    status, fused, metrics, selection_score = None, None, None, None
    if rank == 0:
        try:
            output.mkdir(parents=True, exist_ok=True)
            metrics = _baseline_recalls(dataset, all_records, dense, sparse, task, search.query_type)
            alpha_selection = dict(source="fixed", dataset=None, split=None)
            alpha = search.alpha
            if search.use_fixed_alpha:
                fused = fuse([dense, sparse], [alpha, 1 - alpha])
                metrics["hybrid"] = recall(dataset, all_records, fused, task, search.query_type)
            else:
                best, candidates = _select_alpha(selection_dataset, selection_records,
                                                 selection_dense, selection_sparse, task, search.query_type)
                alpha, best_run, best_metric, selection_score = best
                alpha_selection = dict(source="validation" if use_validation else "target",
                                       dataset=selection_data.dataset_name, split=selection_data.dataset_split)
                baselines = (_baseline_recalls(selection_dataset, selection_records, selection_dense,
                                               selection_sparse, task, search.query_type) if use_validation else dict(metrics))
                if use_validation:
                    fused = fuse([dense, sparse], [alpha, 1 - alpha])
                    metrics["hybrid"] = recall(dataset, all_records, fused, task, search.query_type)
                else:
                    fused, metrics["hybrid"] = best_run, best_metric
                rows = []
                for candidate in candidates:
                    rows.extend(_metric_rows({**baselines, "hybrid": candidate["metric"]}, selection_data,
                                             alpha_selection["source"], candidate["alpha"], score=candidate["score"],
                                             selected=candidate["alpha"] == alpha))
                _write_metrics_csv(output / "alpha_sweep.csv", rows)
            status = dict(alpha=alpha, alpha_selection=alpha_selection)
        except Exception as exc:
            status = dict(error=f"{type(exc).__name__}: {exc}")
    status = _search_status(status)
    alpha, alpha_selection = status["alpha"], status["alpha_selection"]

    if rerank:
        local_run = _scatter_rerank_run(fused, [record["id"] for record in all_records], rank, world)
        ranked = rerank_records(model, processor, dataset, all_records, local_run, task, model_args, data,
                                search, training, rank, world, already_sharded=True)
        ranked = _gather(ranked)
    status = None
    if rank == 0:
        try:
            for name, run in (("dense", dense), ("sparse", sparse), ("hybrid", fused)):
                write_trec_run(run, output / f"{name}_run.trec", name)
            if rerank:
                metrics["rerank"] = recall(dataset, all_records, ranked, task, search.query_type)
                write_trec_run(ranked, output / "rerank_run.trec", "rerank")
            _write_metrics_csv(output / "best.csv", _metric_rows(metrics, data, alpha_selection["source"], alpha,
                                                               score=selection_score, selected=True))
            report = dict(alpha=alpha, alpha_selection=alpha_selection, metrics=metrics)
            (output / "recall_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            (output / manifest_name(expected)).write_text(json.dumps(dict(expected, query_examples=encoder.last_prompts[:2]), indent=2, ensure_ascii=False), encoding="utf-8")
            print(json.dumps(report, indent=2))
            status = {}
        except Exception as exc:
            status = dict(error=f"{type(exc).__name__}: {exc}")
    _search_status(status)


def rerank_records(model, processor, dataset, queries, run, task, model_args, data, search, training, rank, world, *, already_sharded=False):
    from reranker import Reranker
    from rerank_batching import RerankRequest
    from qwen_vl_prompts import rerank_template
    if search.rerank_type != "pointwise":
        raise ValueError("Qwen reranking supports pointwise only; listwise has no implemented scoring definition")
    caption = "caption_generation" in search.rerank_template
    template = rerank_template(processor, task, search.query_type, search.rerank_template, caption=caption)
    candidate_kind = corpus_kind(task, search.query_type)
    corpus = dataset_records(dataset, task, candidate_kind)
    query_map = {record["id"]: record for record in queries}
    corpus_map = {record["id"]: record for record in corpus}

    class RecordReranker(Reranker):
        def _requests(self, query_id, candidate_ids, task_type, template, prompt_type, *, caption):
            query = query_map[query_id]
            requests = []
            for candidate_id in candidate_ids:
                candidate = corpus_map[candidate_id]
                text, prefix, baseline = template, None, None
                if task_type == "cir":
                    dress = query["dress_type"]
                    text = template.replace("{}", "shirt" if dress == "toptee" else dress).replace("<sent>", query["text"])
                    images = (candidate["image"], query["image"]) if prompt_type == "origin_old_relevant" else (query["image"], candidate["image"])
                elif task_type == "t2it":
                    images = (candidate["image"],)
                    if caption:
                        prefix = template.replace("<sent>", candidate["text"])
                        text = prefix + query["text"]
                    else:
                        text = template.replace("<sent>", candidate["text"], 1).replace("<sent>", query["text"], 1)
                elif task_type == "it2t":
                    images = (query["image"],)
                    if caption:
                        prefix = template.replace("<sent>", query["text"])
                        text, baseline = prefix + candidate["text"], template
                    else:
                        text = template.replace("<sent>", query["text"], 1).replace("<sent>", candidate["text"], 1)
                else:
                    image_query = task_type == "ir" and search.query_type == "image"
                    images = (query["image"] if image_query else candidate["image"],)
                    target = candidate["text"] if image_query else query["text"]
                    if caption:
                        prefix, text = template, template + target
                        baseline = template if image_query else None
                    else:
                        text = template.replace("<sent>", target)
                requests.append(RerankRequest(candidate_id, text, images, prefix, baseline))
            return requests

    text_map = {key: value["text"] for key, value in (corpus_map if task == "it2t" or task == "ir" and search.query_type == "image" else query_map).items()}
    ranker = RecordReranker(model, processor, data.dataset_name, search.query_type, text_map, None, {}, processor.tokenizer.get_vocab(), dataset)
    local = run if already_sharded else {key: run[key] for key in list(run)[rank::world]}
    if caption:
        return ranker.caption_generation_rerank(local, search.rerank_type, search.rerank_num, data, training, model_args,
                                               search, rerank_batch_size=search.rerank_batch_size, rerank_prompt_type=search.rerank_template)
    return ranker.rerank(local, search.rerank_type, search.rerank_num, data, training, model_args,
                         rerank_batch_size=search.rerank_batch_size, rerank_prompt_type=search.rerank_template)
