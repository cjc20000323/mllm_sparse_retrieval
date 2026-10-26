"""Persistent, query-independent mean-NLL priors for caption reranking.

Only CPU scalar scores are exchanged/stored. No prompts or scoring definitions
are supplied here: the Reranker callback uses the existing text-only forward.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import numbers
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any


# Increment when the definition of the text-only inputs or NLL changes.
CACHE_VERSION = 1
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "cache" / "rerank_prior"


def validate_prior_cache_options(enabled, modify_type, template, task_type, query_type,
                                 rerank_type="pointwise"):
    if not enabled:
        return
    if modify_type != "modify_single":
        raise ValueError("--use_rerank_prior_cache requires --modify_type=modify_single")
    if "caption_generation" not in template or rerank_type != "pointwise":
        raise ValueError("Prior caching requires pointwise caption_generation reranking")
    if task_type != "it2t" and not (task_type == "ir" and query_type == "image"):
        raise ValueError("Prior caching supports only the existing image-to-text and image-text-to-text correction branches")


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      default=str, allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def candidate_key(candidate_id):
    # Do not conflate an integer ID with a textual ID of the same spelling.
    if isinstance(candidate_id, str):
        return _json(["str", str(candidate_id)])
    if isinstance(candidate_id, numbers.Integral) and not isinstance(candidate_id, bool):
        return _json(["int", int(candidate_id)])
    return _json([type(candidate_id).__name__, candidate_id])


def _configuration(value):
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return {key: _configuration(item) for key, item in vars(value).items()
                if not key.startswith("_")}
    if isinstance(value, dict):
        return {key: _configuration(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_configuration(item) for item in value]
    return value


def _checkpoint(path, revision=None):
    path = str(path or "")
    resolved = Path(path).expanduser()
    if not path or not resolved.exists():
        return {"name": path, "revision": revision}
    resolved = resolved.resolve()
    files = [resolved] if resolved.is_file() else sorted(
        item for item in resolved.rglob("*")
        if item.is_file() and item.suffix.lower() in
        {".safetensors", ".bin", ".pt", ".pth", ".json", ".model", ".txt", ".tiktoken"}
    )
    manifest = []
    for item in files:
        stat = item.stat()
        entry = {"name": item.name if resolved.is_file() else item.relative_to(resolved).as_posix(),
                 "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if item.suffix.lower() in {".json", ".model", ".txt", ".tiktoken"}:
            entry["sha256"] = hashlib.sha256(item.read_bytes()).hexdigest()
        manifest.append(entry)
    return {"path": str(resolved), "revision": revision, "files": manifest}


def prior_metadata(model, processor, model_args, training_args, data_args, *, dataset,
                   query_type, template_name, prefix, candidates):
    tokenizer = getattr(processor, "tokenizer", processor)
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_state = json.loads(backend.to_str()) if backend is not None else None
    if backend_state is not None:
        # Fast tokenizers retain the most recent batch's padding/truncation
        # settings. These are not model identity and must not key the cache.
        backend_state.pop("padding", None)
        backend_state.pop("truncation", None)
    tokenizer_state = {
        "class": type(tokenizer).__qualname__,
        "backend": backend_state,
        "settings": {name: getattr(tokenizer, name, None) for name in (
            "bos_token_id", "eos_token_id", "pad_token_id", "padding_side", "truncation_side",
            "model_max_length", "chat_template", "init_kwargs",
        )},
    }
    if backend is None and callable(getattr(tokenizer, "get_vocab", None)):
        tokenizer_state["vocab"] = tokenizer.get_vocab()
    config = getattr(model, "config", None)
    lora = bool(getattr(model_args, "lora", False))
    return {
        "version": CACHE_VERSION,
        "scoring": "existing_text_only_mean_target_nll_v1",
        "model": _checkpoint(getattr(model_args, "model_name_or_path", ""),
                             getattr(config, "_commit_hash", None)),
        "model_class": type(model).__qualname__,
        "model_config": _configuration(config),
        "lora": _checkpoint(getattr(model_args, "lora_model_path", None)) if lora else None,
        "output_head": {"clustered": bool(getattr(model_args, "use_output_embedding_cluster", False)),
                        "cluster_sum": getattr(model_args, "cluster_sum", None)
                        if getattr(model_args, "use_output_embedding_cluster", False) else None},
        "dtype": str(getattr(model, "dtype", None)),
        "attention_implementation": getattr(config, "_attn_implementation", None),
        "fp16_autocast": bool(getattr(training_args, "fp16", False)),
        "bf16": bool(getattr(training_args, "bf16", False)),
        "processor_class": type(processor).__qualname__,
        "tokenizer": fingerprint(tokenizer_state),
        "transformers_version": importlib.metadata.version("transformers"),
        "dataset": dataset,
        "split": getattr(data_args, "dataset_split", "unspecified"),
        "task_type": training_args.task_type,
        "query_type": query_type,
        "template_name": template_name,
        "prefix": prefix,
        "candidate_count": len(candidates),
        "candidates_sha256": fingerprint(candidates),
    }


class PriorCacheCollectives:
    def __init__(self):
        import torch.distributed as dist
        self.dist = dist
        self.distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0
        self.world_size = dist.get_world_size() if self.distributed else 1

    def broadcast(self, value):
        packet = [value if self.rank == 0 else None]
        if self.distributed:
            self.dist.broadcast_object_list(packet, src=0)
        return packet[0]

    def gather(self, value):
        if not self.distributed:
            return [value]
        values = [None] * self.world_size
        self.dist.all_gather_object(values, value)
        return values


def _slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "unknown"


def _validated_scores(entries, candidates):
    expected = {key: hashlib.sha256(text.encode("utf-8")).hexdigest() for key, text in candidates}
    if len(expected) != len(candidates) or len(entries) != len(expected):
        raise ValueError("Prior cache has duplicate or missing candidate IDs")
    scores = {}
    for entry in entries:
        key, score = entry["key"], entry["mean_nll"]
        if key in scores or key not in expected or entry["text_sha256"] != expected[key]:
            raise ValueError("Prior cache candidate IDs/texts do not match the current corpus")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or score < 0:
            raise ValueError("Prior cache contains an invalid mean NLL")
        scores[key] = float(score)
    return scores


def _atomic_save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_or_build_prior_cache(metadata, candidates, score_batch, batch_size, *, cache_dir=None,
                             collectives=None, progress_factory=None):
    """All ranks enter, including ranks with zero online queries or prior items."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("Prior cache batch size must be a positive integer")
    started = time.perf_counter()
    group = collectives if collectives is not None else PriorCacheCollectives()
    digest = fingerprint(metadata)
    digests = group.gather(digest)
    if any(item != digest for item in digests):
        raise ValueError("Ranks disagree about the model/template/candidate pool for the prior cache")
    model = metadata["model"]
    model_name = Path(model.get("path") or model.get("name") or "model").name
    root = Path(cache_dir).expanduser() if cache_dir else DEFAULT_CACHE_DIR
    path = (root / _slug(model_name) / _slug(metadata["dataset"]) /
            _slug(metadata["split"]) / f"{digest}.json").resolve()
    packet = None
    if group.rank == 0:
        read_started = time.perf_counter()
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload["fingerprint"] != digest or _json(payload["metadata"]) != _json(metadata):
                raise ValueError("Prior cache metadata mismatch")
            if payload["entries_sha256"] != fingerprint(payload["entries"]):
                raise ValueError("Prior cache score checksum mismatch")
            scores = _validated_scores(payload["entries"], candidates)
            packet = {"hit": True, "scores": scores, "load_seconds": time.perf_counter() - read_started}
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
            packet = {"hit": False, "reason": f"{type(exc).__name__}: {exc}", "load_seconds": 0.0}
    packet = group.broadcast(packet)
    precompute_seconds = 0.0
    if packet["hit"]:
        scores = packet["scores"]
    else:
        local = candidates[group.rank::group.world_size]
        entries, error = [], None
        build_started = time.perf_counter()
        try:
            progress = progress_factory(
                total=len(local), desc=f"[prior-cache][rank={group.rank}] {metadata['dataset']}/{metadata['split']}",
                unit="candidate", dynamic_ncols=True, position=group.rank,
            ) if progress_factory is not None else None
            try:
                for start in range(0, len(local), batch_size):
                    batch = local[start:start + batch_size]
                    values = score_batch(batch)
                    if len(values) != len(batch):
                        raise ValueError("Prior model must return one NLL per candidate")
                    for (key, text), score in zip(batch, values):
                        if isinstance(score, bool):
                            raise ValueError("Prior model returned a boolean instead of an NLL")
                        entries.append({"key": key, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                                        "mean_nll": float(score)})
                    if progress is not None:
                        progress.update(len(batch))
            finally:
                if progress is not None:
                    progress.close()
        except Exception as exc:
            error = f"rank {group.rank}: {type(exc).__name__}: {exc}"
        reports = group.gather({"entries": entries, "error": error,
                                "seconds": time.perf_counter() - build_started})
        failures = [item["error"] for item in reports if item["error"]]
        if failures:
            raise RuntimeError("Prior precomputation failed: " + "; ".join(failures))
        precompute_seconds = max(item["seconds"] for item in reports)
        completed = None
        if group.rank == 0:
            try:
                merged = [entry for item in reports for entry in item["entries"]]
                scores = _validated_scores(merged, candidates)
                _atomic_save(path, {"fingerprint": digest, "metadata": metadata, "entries": merged,
                                    "entries_sha256": fingerprint(merged),
                                    "precompute_seconds": precompute_seconds})
                completed = {"scores": scores, "error": None}
            except Exception as exc:
                completed = {"error": f"{type(exc).__name__}: {exc}"}
        completed = group.broadcast(completed)
        if completed["error"] is not None:
            raise RuntimeError("Could not finalize prior cache: " + completed["error"])
        scores = completed["scores"]
    report = {
        "enabled": True, "cache_hit": packet["hit"], "cache_path": str(path),
        "fingerprint": digest, "candidate_count": len(candidates), "world_size": group.world_size,
        "precompute_seconds": precompute_seconds, "load_seconds": packet["load_seconds"],
        "prepare_seconds": time.perf_counter() - started,
        "time_unit": "seconds", "scoring": metadata["scoring"],
        "timing_scope": "offline cache preparation; excluded from online reranking/ETA compute, included in whole-run wall time",
    }
    if group.rank == 0:
        print(f"[prior-cache] {'hit' if packet['hit'] else 'built'}: candidates={len(candidates)}, "
              f"precompute={precompute_seconds:.3f}s, load={packet['load_seconds']:.3f}s; {path}")
    return scores, report
