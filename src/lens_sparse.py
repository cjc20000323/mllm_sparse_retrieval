"""LENS-inspired centroid LM head for adaptive views (no training or attention changes)."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import lru_cache
import gc
import hashlib
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

VERSION = 1
FIELD = "lens_sparse"
BUNDLE = "lens_head"
METHOD = "centroid_head_log1p_relu_cluster_topk_x100"
DEFAULTS = dict(USE_LENS_SPARSE="false", LENS_ALGORITHM="kmeans", LENS_N_CLUSTERS="8000",
                LENS_SEED="42", LENS_N_INIT="1", LENS_MAX_ITER="300", LENS_BATCH_SIZE="4096",
                LENS_CACHE_ROOT="")
NAMES = tuple(DEFAULTS)
OPTIONS = {name.lower(): value for name, value in DEFAULTS.items()}


def digest(value):
    from adaptive_views import digest as hash_json
    return hash_json(value)


def enabled(args):
    from dense_representation import boolean
    return boolean(getattr(args, "use_lens_sparse", False))


def validate(args):
    if not enabled(args):
        return
    from dense_representation import boolean
    if boolean(getattr(args, "use_stc_sparse", False)):
        raise ValueError("USE_LENS_SPARSE and USE_STC_SPARSE are mutually exclusive")
    if boolean(getattr(args, "view_overlap_selection", False)):
        raise ValueError("LENS sparse mode does not support overlap selections")
    if getattr(args, "lens_algorithm", "kmeans") not in {"kmeans", "minibatch_kmeans"}:
        raise ValueError("LENS_ALGORITHM must be kmeans or minibatch_kmeans")
    for key in ("lens_n_clusters", "lens_n_init", "lens_max_iter", "lens_batch_size"):
        if int(getattr(args, key, OPTIONS[key])) < 1:
            raise ValueError(f"{key} must be positive")
    if not 0 <= int(getattr(args, "lens_seed", 42)) < 2**32:
        raise ValueError("LENS_SEED must be in [0, 2**32)")


def environment_controls(env):
    from dense_representation import boolean
    values = {name: env.get(name) or value for name, value in DEFAULTS.items()}
    values["USE_LENS_SPARSE"] = str(boolean(values["USE_LENS_SPARSE"])).lower()
    validate(SimpleNamespace(**{key.lower(): value for key, value in values.items()},
        use_stc_sparse=boolean(env.get("USE_STC_SPARSE", False)),
        view_overlap_selection=boolean(env.get("USE_VIEW_OVERLAP_SELECTION", False))))
    return values


def add_parser(parser, *, inherit=False):
    from dense_representation import boolean
    for name, default in DEFAULTS.items():
        key = name.lower()
        if key == "use_lens_sparse":
            kwargs = dict(default=False, nargs="?", const=True, type=boolean)
        else:
            string = key in {"lens_algorithm", "lens_cache_root"}
            kwargs = dict(default=default if string else int(default), type=str if string else int)
        if inherit:
            kwargs["default"] = None
        parser.add_argument("--" + key, "--" + key.replace("_", "-"), **kwargs)


def copy_options(source, target, *, dtype=None, root=None, data=None):
    validate(source)
    if not enabled(source):
        return target
    for key, default in OPTIONS.items():
        setattr(target, key, getattr(source, key, default))
    target.use_lens_sparse = True
    target.lens_dtype = dtype or getattr(source, "lens_dtype", "float16")
    target.lens_cache_root = str(getattr(source, "lens_cache_root", "") or root or "./sparse_output/lens_heads")
    filters = data or source
    target.lens_is_filtered = getattr(filters, "is_filtered", getattr(source, "lens_is_filtered", False))
    target.lens_case_policy = getattr(filters, "sparse_lower_or_upper", getattr(source, "lens_case_policy", "lower"))
    if getattr(source, "lens_head_path", None):
        target.lens_head_path = str(source.lens_head_path)
    return target


@lru_cache(maxsize=16)
def _identity(path, checkpoint_hash):
    from transformers import AutoConfig, AutoTokenizer
    from adaptive_views import family_of
    return AutoTokenizer.from_pretrained(path), family_of(AutoConfig.from_pretrained(path))


def request(args, *, tokenizer=None, family=None):
    from adaptive_views import checkpoint_identity
    from dense_representation import boolean
    checkpoint = checkpoint_identity(args.model_name_or_path)
    if tokenizer is None or family is None:
        loaded, detected = _identity(str(args.model_name_or_path), digest(checkpoint))
        tokenizer, family = tokenizer or loaded, family or detected
    adapter = getattr(args, "lora_model_path", None) if getattr(args, "lora", False) else None
    config = dict(version=VERSION, checkpoint=checkpoint,
        adapter=checkpoint_identity(adapter) if adapter else None, family=family,
        tokenizer=digest(tokenizer.get_vocab()), special_ids=sorted(tokenizer.all_special_ids),
        dtype=getattr(args, "lens_dtype", "float16"), embedding="raw_output_head_only",
        metric="squared_euclidean", normalization="none", init="k-means++",
        algorithm=args.lens_algorithm, clusters=int(args.lens_n_clusters), seed=int(args.lens_seed),
        n_init=int(args.lens_n_init), max_iter=int(args.lens_max_iter),
        filters=dict(special="exclude", blank="exclude", output_zero="exclude", numeric="cluster",
            is_filtered=boolean(getattr(args, "lens_is_filtered", getattr(args, "is_filtered", False))),
            case=getattr(args, "lens_case_policy", getattr(args, "sparse_lower_or_upper", "lower"))))
    if args.lens_algorithm == "minibatch_kmeans":
        config.update(batch_size=int(args.lens_batch_size), reassignment_ratio=0)
    else:
        config["solver"] = "lloyd"
    return config, tokenizer


def cache_path(args, config=None):
    if getattr(args, "lens_head_path", None):
        return Path(args.lens_head_path)
    config = config or request(args)[0]
    return Path(args.lens_cache_root).expanduser() / Path(args.model_name_or_path).name / digest(config)[:16]


def arrays_digest(arrays):
    result = hashlib.sha256()
    for name in ("token_ids", "cluster_ids", "centroids", "bias"):
        array = np.ascontiguousarray(arrays[name], dtype="<i4" if name.endswith("ids") else "<f4")
        result.update(name.encode())
        result.update(str(array.shape).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def _save_arrays(path, arrays):
    path = Path(path)
    temporary = path.with_name(".lens-" + uuid4().hex + ".npz")
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class Head:
    def __init__(self, directory, *, expected=None):
        self.directory = Path(directory)
        try:
            self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
            with np.load(self.directory / "head.npz", allow_pickle=False) as archive:
                self.arrays = {name: archive[name].copy() for name in ("token_ids", "cluster_ids", "centroids", "bias")}
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"Missing or corrupt LENS head: {directory}; run lens_sparse.py build before encoding") from exc
        m, a = self.manifest, self.arrays
        if (m.get("version") != VERSION or m.get("arrays_sha256") != arrays_digest(a)
                or m.get("fingerprint") != digest({k: v for k, v in m.items() if k != "fingerprint"})):
            raise ValueError(f"LENS head checksum/version mismatch: {directory}")
        k, d = m.get("cluster_count"), m.get("hidden_size")
        if (type(k) is not int or type(d) is not int or min(k, d) < 1
                or a["centroids"].shape != (k, d) or a["bias"].shape not in {(0,), (k,)}
                or not np.isfinite(a["centroids"]).all() or not np.isfinite(a["bias"]).all()
                or a["centroids"].dtype != np.float32 or a["bias"].dtype != np.float32):
            raise ValueError(f"Invalid LENS centroid dimensions/values: {directory}")
        ids, labels = a["token_ids"], a["cluster_ids"]
        if (ids.ndim != 1 or labels.shape != ids.shape or not len(ids)
                or ids.dtype.kind not in "iu" or labels.dtype.kind not in "iu"
                or np.any(ids < 0) or np.any(np.diff(ids.astype(np.int64)) <= 0)
                or set(labels.tolist()) != set(range(k)) or m["request"].get("clusters") != k):
            raise ValueError(f"Invalid LENS member mapping: {directory}")
        if expected is not None and m["request"] != expected:
            raise ValueError(f"LENS model/tokenizer/clustering configuration mismatch: {directory}")

    @property
    def fingerprint(self):
        return self.manifest["fingerprint"]


def load(args, *, tokenizer=None, family=None):
    validate(args)
    config, _ = request(args, tokenizer=tokenizer, family=family)
    return Head(cache_path(args, config), expected=config)


def configuration(args):
    if not enabled(args):
        return None
    head = load(args)
    return dict(version=VERSION, method=METHOD, head_fingerprint=head.fingerprint,
        request=head.manifest["request"], cluster_count=head.manifest["cluster_count"],
        weight_scale=100, pooling="existing_adaptive_view_positions", attention="unchanged",
        quantization="round_per_view_then_merge_floor_mean_drop_nonpositive")


def mark_path(path, config):
    if config is None:
        return Path(path)
    path = Path(path)
    match = re.search(r"(_llava_base)?(_dense_freeret_pre_mlp(?:_freeret_(?:constraints|full_prompt)_v\d+)?)?$", path.name)
    suffix = match.group(0)
    base = path.name[:-len(suffix)] if suffix else path.name
    marker = f"_lens_{config['request']['algorithm']}_k{config['cluster_count']}_{digest(config)[:12]}"
    return path if marker in path.name else path.with_name(base + marker + suffix)


def publish_bundle(root, head):
    from adaptive_views import atomic_json
    target = Path(root) / BUNDLE
    target.mkdir(parents=True, exist_ok=True)
    if (target / "manifest.json").exists() and Head(target).fingerprint != head.fingerprint:
        raise ValueError(f"Refusing to overwrite a different LENS head: {target}")
    _save_arrays(target / "head.npz", head.arrays)
    atomic_json(target / "manifest.json", head.manifest)


def validate_bundle(root, config):
    if config is None:
        return
    if config.get("version") != VERSION or config.get("method") != METHOD or config.get("weight_scale") != 100:
        raise ValueError(f"LENS sparse method/version mismatch at {root}; re-encode and rebuild the index")
    head = Head(Path(root) / BUNDLE, expected=config["request"])
    if head.fingerprint != config.get("head_fingerprint") or head.manifest["cluster_count"] != config.get("cluster_count"):
        raise ValueError(f"LENS artifact/head fingerprint mismatch: {root}")


def bind_search(args, root, config):
    from adaptive_views import require_equal
    from dense_representation import boolean
    requested = getattr(args, "use_lens_sparse", None)
    if requested is not None and boolean(requested) != (config is not None):
        raise ValueError(f"LENS mode mismatch at {root}: requested USE_LENS_SPARSE={requested}")
    args.use_lens_sparse = config is not None
    vars(args).pop("lens_head_path", None)
    if config is None:
        return
    validate_bundle(root, config)
    saved = {key.lower(): config["request"].get(key[5:].lower(), default) for key, default in DEFAULTS.items()}
    saved.update(lens_n_clusters=config["request"]["clusters"], lens_seed=config["request"]["seed"],
                 lens_algorithm=config["request"]["algorithm"])
    for key, default in OPTIONS.items():
        if key == "use_lens_sparse":
            continue
        string = key in {"lens_algorithm", "lens_cache_root"}
        value = saved.get(key, default)
        value = str(value) if string else int(value)
        actual = getattr(args, key, None)
        compare = key != "lens_cache_root" and (key != "lens_batch_size" or config["request"]["algorithm"] == "minibatch_kmeans")
        if actual is not None and compare:
            require_equal({key: value}, {key: str(actual) if string else int(actual)}, f"LENS search configuration at {root}")
        if actual is None:
            setattr(args, key, value)
    args.lens_head_path = str(Path(root) / BUNDLE)
    validate(args)


def bind_corpus(args):
    if not enabled(args):
        return
    validate(args)
    from adaptive_views import artifact_root, MANIFEST
    root = artifact_root(args.stage1_sparse_index)
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    config = manifest["configuration"].get(FIELD)
    if config is None:
        raise ValueError("USE_LENS_SPARSE=true requires a LENS-encoded corpus")
    validate_bundle(root, config)
    args.lens_head_path = str(root / BUNDLE)


class CentroidLayer(nn.Module):
    def __init__(self, head, reference):
        super().__init__()
        if not reference.weight.is_floating_point() or reference.weight.shape[1] != head.arrays["centroids"].shape[1]:
            raise ValueError("LENS requires a floating-point linear LM head with matching hidden size")
        self.register_buffer("weight", torch.from_numpy(head.arrays["centroids"].copy()).to(reference.weight))
        bias = head.arrays["bias"]
        self.register_buffer("bias", torch.from_numpy(bias.copy()).to(reference.weight) if len(bias) else None)

    def forward(self, hidden):
        return F.linear(hidden, self.weight, self.bias)


class SparseEncoder:
    def __init__(self, tokenizer, data, args, *, family):
        self.head = load(args, tokenizer=tokenizer, family=family)
        self.data, self.layer = data, None

    @contextmanager
    def activate(self, model):
        original, inputs = model.get_output_embeddings(), model.get_input_embeddings()
        input_weight = inputs.weight
        if self.layer is None or self.layer.weight.device != original.weight.device or self.layer.weight.dtype != original.weight.dtype:
            self.layer = CentroidLayer(self.head, original)
        try:
            model.set_output_embeddings(self.layer)
            if model.get_input_embeddings() is not inputs or model.get_input_embeddings().weight is not input_weight:
                raise ValueError("LENS output-head replacement unexpectedly changed the input embeddings")
            yield
        finally:
            model.set_output_embeddings(original)
            if model.get_input_embeddings() is not inputs or model.get_input_embeddings().weight is not input_weight:
                model.set_input_embeddings(inputs)
                inputs.weight = input_weight

    def vector(self, values, record=None):
        if values.ndim != 2 or not len(values) or values.shape[1] != self.head.manifest["cluster_count"]:
            raise ValueError("LENS expects nonempty [views, clusters] activated values")
        if not torch.isfinite(values).all():
            raise ValueError("Non-finite LENS sparse values")
        merged = {}
        for row in values:
            weights, ids = row.topk(min(int(self.data.sparse_length), len(row)))
            for index, weight in zip(ids.tolist(), torch.round(weights.float() * 100).long().tolist()):
                if weight <= 0:
                    continue
                if index not in merged or self.data.sparse_value_type == "replace":
                    merged[index] = weight
                elif self.data.sparse_value_type == "sum":
                    merged[index] += weight
                elif self.data.sparse_value_type == "max":
                    merged[index] = max(merged[index], weight)
                else:
                    raise ValueError("Unsupported LENS sparse_value_type")
        if self.data.sparse_value_mean:
            merged = {key: value // len(values) for key, value in merged.items() if value // len(values) > 0}
        if merged and max(merged.values()) >= 2**31:
            raise ValueError("LENS integer weight exceeds Lucene's signed integer range")
        return {f"lens_{key:06d}": value for key, value in sorted(merged.items())}


def _accepted(token, filters, family):
    if filters["case"] == "lower":
        token = token.lower()
    if filters["is_filtered"]:
        if family.startswith("qwen"):
            token = token.lstrip("Ġġ▁ ")
            if not token or not token.isalnum() or not ("a" <= token[0].lower() <= "z"):
                return False
        elif filters["case"] == "lower" and token.startswith(("Ġ", "ġ")):
            token = token[1:]
    return bool(token) and not any(char.isspace() for char in token)


def build_head(output_embedding, tokenizer, args, directory, config, *, bias=None):
    """Small CPU fixtures can exercise the exact same centroid construction."""
    from adaptive_views import atomic_json
    validate(args)
    vocab = tokenizer.get_vocab()
    raw = {int(index): token for token, index in vocab.items()}
    ids = sorted(raw)
    if not ids or ids[0] < 0 or ids[-1] >= len(output_embedding):
        raise ValueError("LENS tokenizer IDs exceed output embedding rows")
    special = set(tokenizer.all_special_ids)
    decoded = tokenizer.batch_decode([[i] for i in ids], skip_special_tokens=True)
    eligible = [i for i, text in zip(ids, decoded) if i not in special and text.strip()
                and _accepted(raw[i], config["filters"], config["family"])]
    retained, zero = [], []
    matrix = output_embedding.detach().cpu()
    for start in range(0, len(eligible), 4096):
        indices = eligible[start:start + 4096]
        block = matrix[indices].float()
        norms = block.norm(dim=1)
        finite = torch.isfinite(block).all(dim=1) & torch.isfinite(norms)
        if not finite.all():
            invalid = [i for i, ok in zip(indices, finite.tolist()) if not ok]
            raise ValueError(f"Non-finite LENS output embeddings/norms; token IDs={invalid[:20]}")
        for i, norm in zip(indices, norms.tolist()):
            (retained if norm > 0 else zero).append(i)
    k = int(args.lens_n_clusters)
    if not 1 <= k <= len(retained):
        raise ValueError(f"LENS_N_CLUSTERS={k} exceeds {len(retained)} eligible output vectors")
    if bias is not None and (len(bias) != len(matrix) or not torch.isfinite(bias[ids]).all()):
        raise ValueError("Invalid LENS original LM-head bias")
    print(f"[LENS] {args.lens_algorithm}: tokens={len(retained)} clusters={k}; raw output embeddings only; zero_excluded={len(zero)}", flush=True)
    features = matrix[retained].float().numpy()
    if k == len(retained):
        labels = np.arange(k)
    else:
        from sklearn.cluster import KMeans, MiniBatchKMeans
        kwargs = dict(n_clusters=k, init="k-means++", n_init=int(args.lens_n_init),
                      random_state=int(args.lens_seed), max_iter=int(args.lens_max_iter))
        estimator = (KMeans(**kwargs, algorithm="lloyd") if args.lens_algorithm == "kmeans" else
                     MiniBatchKMeans(**kwargs, batch_size=int(args.lens_batch_size), reassignment_ratio=0))
        labels = estimator.fit_predict(features)
    groups = {}
    for index, label in zip(retained, labels):
        groups.setdefault(int(label), []).append(index)
    groups = sorted(groups.values(), key=min)
    if len(groups) != k:
        raise ValueError(f"LENS produced {len(groups)} nonempty clusters, expected {k}; lower LENS_N_CLUSTERS or increase initialization/iterations")
    # Arithmetic means of raw rows: h @ center == mean(original member logits).
    centers = np.stack([matrix[group].float().mean(0).numpy() for group in groups]).astype(np.float32)
    biases = (np.array([float(bias[group].float().mean()) for group in groups], dtype=np.float32)
              if bias is not None else np.empty(0, dtype=np.float32))
    mapping = {i: label for label, members in enumerate(groups) for i in members}
    arrays = dict(token_ids=np.array(retained, dtype=np.int32),
                  cluster_ids=np.array([mapping[i] for i in retained], dtype=np.int32), centroids=centers, bias=biases)
    manifest = dict(version=VERSION, request=config, cluster_count=k, hidden_size=centers.shape[1],
                    excluded_zero_token_ids=zero, arrays_sha256=arrays_digest(arrays))
    manifest["fingerprint"] = digest(manifest)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists():
        existing = Head(directory, expected=config)
        if existing.fingerprint != manifest["fingerprint"]:
            raise ValueError(f"Refusing to overwrite a different LENS head: {directory}")
        return existing
    _save_arrays(directory / "head.npz", arrays)
    atomic_json(directory / "cluster_members.json", [dict(cluster_id=i, size=len(group),
                tokens=[dict(id=j, token=raw[j]) for j in group]) for i, group in enumerate(groups)])
    atomic_json(directory / "manifest.json", manifest)
    return Head(directory, expected=config)


def _build_new(args, tokenizer, directory, config):
    import transformers
    classes = dict(qwen2_vl="Qwen2VLForConditionalGeneration", qwen2_5_vl="Qwen2_5_VLForConditionalGeneration",
                   qwen3_vl="Qwen3VLForConditionalGeneration", llava_llama="LlavaNextForConditionalGeneration",
                   llava_mistral="LlavaNextForConditionalGeneration")
    model = getattr(transformers, classes[config["family"]]).from_pretrained(args.model_name_or_path,
        torch_dtype=getattr(torch, args.lens_dtype), low_cpu_mem_usage=True, attn_implementation="eager", device_map="cpu")
    if getattr(args, "lora_model_path", None):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_model_path).merge_and_unload()
    layer = model.get_output_embeddings()
    weight = layer.weight.detach().cpu().clone()
    bias = layer.bias.detach().cpu().clone() if getattr(layer, "bias", None) is not None else None
    del layer, model
    gc.collect()
    return build_head(weight, tokenizer, args, directory, config, bias=bias)


def build(args):
    validate(args)
    config, tokenizer = request(args)
    directory = cache_path(args, config)
    if (directory / "manifest.json").exists():
        head = Head(directory, expected=config)
        print(f"[LENS] Reuse head: {directory}; fingerprint={head.fingerprint}", flush=True)
        return head
    if getattr(args, "dry_run", False):
        print(f"[LENS] Would build {args.lens_algorithm} head: {directory}", flush=True)
        return None
    directory.parent.mkdir(parents=True, exist_ok=True)
    lock = directory.with_name("." + directory.name + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError(f"LENS precomputation locked: {lock}; wait for the builder or remove a confirmed stale lock") from exc
    try:
        return _build_new(args, tokenizer, directory, config)
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)


def main(argv=None):
    from dense_representation import boolean
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build"])
    add_parser(parser)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--dtype", dest="lens_dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--lora-model-path")
    parser.add_argument("--is-filtered", type=boolean, default=False)
    parser.add_argument("--sparse-lower-or-upper", default="lower", choices=["lower", "upper"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.use_lens_sparse, args.lora = True, bool(args.lora_model_path)
    args.lens_cache_root = args.lens_cache_root or "./sparse_output/lens_heads"
    try:
        build(args)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
