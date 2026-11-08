"""Fixed STC-inspired vocabulary compression for adaptive sparse retrieval.

This is embedding clustering + probability-mass aggregation, not STC's
generation-dependent uncertainty score/prefix matching. No output head is replaced.
"""
from __future__ import annotations

import argparse
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
import torch.nn.functional as F

VERSION = 1
FIELD = "stc_sparse"
BUNDLE = "stc_vocab"
DEFAULTS = dict(USE_STC_SPARSE="false", STC_ALGORITHM="agglomerative", STC_N_CLUSTERS="16000",
                STC_SEED="42", STC_BATCH_SIZE="4096", STC_MAX_ITER="100",
                STC_WEIGHT_SCALE="10000", STC_CACHE_ROOT="")
NAMES = tuple(DEFAULTS)
OPTIONS = {name.lower(): value for name, value in DEFAULTS.items()}


def digest(value):
    from adaptive_views import digest as hash_json
    return hash_json(value)


def enabled(args):
    from dense_representation import boolean
    return boolean(getattr(args, "use_stc_sparse", False))


def validate(args):
    if not enabled(args):
        return
    if getattr(args, "stc_algorithm", "agglomerative") not in {"agglomerative", "minibatch_kmeans"}:
        raise ValueError("STC_ALGORITHM must be agglomerative or minibatch_kmeans")
    for name in ("stc_n_clusters", "stc_batch_size", "stc_max_iter", "stc_weight_scale"):
        number = int(getattr(args, name, OPTIONS[name]))
        if number < 1 or (name == "stc_weight_scale" and number > 1000000):
            raise ValueError(f"{name} must be positive (weight scale <= 1000000)")
    if not 0 <= int(getattr(args, "stc_seed", 42)) < 2**32:
        raise ValueError("STC_SEED must be in [0, 2**32)")
    if getattr(args, "view_overlap_selection", False):
        raise ValueError("STC sparse mode currently supports ordinary adaptive views, not overlap")


def environment_controls(env):
    values = {name: env.get(name) or default for name, default in DEFAULTS.items()}
    from dense_representation import boolean
    values["USE_STC_SPARSE"] = str(boolean(values["USE_STC_SPARSE"])).lower()
    validate(SimpleNamespace(**{k.lower(): v for k, v in values.items()},
                             view_overlap_selection=boolean(env.get("USE_VIEW_OVERLAP_SELECTION", "false"))))
    return values


def add_parser(parser, *, inherit=False):
    from dense_representation import boolean
    for name, default in DEFAULTS.items():
        key = name.lower()
        kwargs = dict(default=False, nargs="?", const=True, type=boolean) if key == "use_stc_sparse" else dict(
            default=default if key in {"stc_algorithm", "stc_cache_root"} else int(default),
            type=str if key in {"stc_algorithm", "stc_cache_root"} else int)
        if inherit:
            kwargs["default"] = None
        parser.add_argument("--" + key.replace("_", "-"), "--" + key, **kwargs)


def copy_options(source, target, *, dtype=None, root=None):
    """Do not add even false fields to legacy namespace/cache fingerprints."""
    validate(source)
    if enabled(source):
        for name, default in OPTIONS.items():
            setattr(target, name, getattr(source, name, default))
        target.use_stc_sparse = True
        target.stc_dtype = dtype or getattr(source, "stc_dtype", "float16")
        target.stc_cache_root = str(getattr(source, "stc_cache_root", "") or root or "./sparse_output/stc_vocab")
        if getattr(source, "stc_vocab_path", None):
            target.stc_vocab_path = str(source.stc_vocab_path)
    return target


@lru_cache(maxsize=16)
def _tokenizer(path, checkpoint_hash):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path)


def request(args, *, tokenizer=None):
    from adaptive_views import checkpoint_identity
    checkpoint = checkpoint_identity(args.model_name_or_path)
    tokenizer = tokenizer or _tokenizer(str(args.model_name_or_path), digest(checkpoint))
    adapter_path = getattr(args, "lora_model_path", None) if getattr(args, "lora", False) else None
    config = dict(version=VERSION, checkpoint=checkpoint,
                  adapter=checkpoint_identity(adapter_path) if adapter_path else None,
                  tokenizer=digest(tokenizer.get_vocab()), special_ids=sorted(tokenizer.all_special_ids),
                  dtype=getattr(args, "stc_dtype", "float16"),
                  embedding="separately_l2_normalized_input_output_concat", numeric="singleton",
                  algorithm=args.stc_algorithm, semantic_clusters=int(args.stc_n_clusters))
    if args.stc_algorithm == "agglomerative":
        config.update(metric="cosine", linkage="complete")
    else:
        config.update(seed=int(args.stc_seed), batch_size=int(args.stc_batch_size),
                      max_iter=int(args.stc_max_iter), init="random", n_init=3)
    return config, tokenizer


def cache_path(args, config=None):
    if getattr(args, "stc_vocab_path", None):
        return Path(args.stc_vocab_path)
    config = config or request(args)[0]
    return Path(args.stc_cache_root).expanduser() / Path(args.model_name_or_path).name / digest(config)[:16]


def arrays_digest(arrays):
    result = hashlib.sha256()
    for name in ("vocab_ids", "token_ids", "cluster_ids"):
        array = np.asarray(arrays[name], dtype="<i4")
        result.update(name.encode())
        result.update(str(array.shape).encode())
        result.update(array.tobytes())
    return result.hexdigest()


class Vocabulary:
    def __init__(self, directory, *, expected=None):
        self.directory = Path(directory)
        filename = self.directory / "manifest.json"
        try:
            self.manifest = json.loads(filename.read_text(encoding="utf-8"))
            with np.load(self.directory / "token_to_cluster.npz", allow_pickle=False) as archive:
                self.arrays = {name: archive[name].copy() for name in ("vocab_ids", "token_ids", "cluster_ids")}
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"Missing or corrupt STC vocabulary: {self.directory}; run the encoding script or stc_sparse.py build") from exc
        m, a = self.manifest, self.arrays
        if (m.get("version") != VERSION or m.get("arrays_sha256") != arrays_digest(a)
                or m.get("fingerprint") != digest({k: v for k, v in m.items() if k != "fingerprint"})):
            raise ValueError(f"STC vocabulary checksum mismatch: {self.directory}")
        for name, array in a.items():
            if array.ndim != 1 or array.dtype.kind not in "iu" or np.any(array < 0):
                raise ValueError(f"Invalid STC {name}: {self.directory}")
        if (len(a["token_ids"]) != len(a["cluster_ids"]) or not len(a["vocab_ids"])
                or np.any(np.diff(a["vocab_ids"].astype(np.int64)) <= 0)
                or np.any(np.diff(a["token_ids"].astype(np.int64)) <= 0)
                or not np.isin(a["token_ids"], a["vocab_ids"]).all()
                or set(a["cluster_ids"].tolist()) != set(range(int(m["cluster_count"])))):
            raise ValueError(f"Invalid STC mapping: {self.directory}")
        if expected is not None and m["request"] != expected:
            raise ValueError(f"STC model/tokenizer/clustering configuration mismatch: {self.directory}")

    @property
    def fingerprint(self):
        return self.manifest["fingerprint"]


def load(args, *, tokenizer=None):
    config, _ = request(args, tokenizer=tokenizer)
    return Vocabulary(cache_path(args, config), expected=config)


def configuration(args):
    if not enabled(args):
        return None
    vocabulary = load(args)
    return dict(version=VERSION, method="cluster_probability_sum_then_topk",
                mapping_fingerprint=vocabulary.fingerprint, request=vocabulary.manifest["request"],
                cluster_count=vocabulary.manifest["cluster_count"], weight_scale=int(args.stc_weight_scale),
                quantization="round_after_view_merge_min_one", prefix_matching=False)


def mark_path(path, config):
    if config is None:
        return Path(path)
    path = Path(path)
    # Keep existing resolution/FreeRet suffixes at the end for their validators.
    suffix_re = r"(_llava_base)?(_dense_freeret_pre_mlp(?:_freeret_(?:constraints|full_prompt)_v\d+)?)?$"
    match = re.search(suffix_re, path.name)
    suffix = match.group(0)
    base = path.name[:-len(suffix)] if suffix else path.name
    marker = f"_stc_{config['request']['algorithm']}_k{config['request']['semantic_clusters']}_{digest(config)[:12]}"
    if marker in path.name:
        return path
    if not base.endswith(marker):
        base += marker
    return path.with_name(base + suffix)


def publish_bundle(root, vocabulary):
    from adaptive_views import atomic_json
    target = Path(root) / BUNDLE
    target.mkdir(parents=True, exist_ok=True)
    _save_arrays(target / "token_to_cluster.npz", vocabulary.arrays)
    atomic_json(target / "manifest.json", vocabulary.manifest)


def validate_bundle(root, config):
    if config is None:
        return
    vocabulary = Vocabulary(Path(root) / BUNDLE, expected=config["request"])
    if vocabulary.fingerprint != config["mapping_fingerprint"] or vocabulary.manifest["cluster_count"] != config["cluster_count"]:
        raise ValueError(f"STC artifact/mapping fingerprint mismatch: {root}")


def bind_search(args, root, config):
    """Inherit omitted search options and bind the candidate's immutable map.

    A cache root is only a lookup location. It must never replace the mapping
    locked by an explicitly supplied corpus, including validation corpora.
    """
    from adaptive_views import require_equal
    from dense_representation import boolean
    active = config is not None
    requested = getattr(args, "use_stc_sparse", None)
    if requested is not None and boolean(requested) != active:
        raise ValueError(f"STC mode mismatch at {root}: stored USE_STC_SPARSE={active}, requested={requested}")
    args.use_stc_sparse = active
    # A copied validation namespace must not retain the target's bundle path.
    vars(args).pop("stc_vocab_path", None)
    if not active:
        return
    validate_bundle(root, config)
    request_config = config["request"]
    saved = dict(stc_algorithm=request_config["algorithm"], stc_n_clusters=request_config["semantic_clusters"],
                 stc_weight_scale=config["weight_scale"])
    if request_config["algorithm"] == "minibatch_kmeans":
        saved.update(stc_seed=request_config["seed"], stc_batch_size=request_config["batch_size"],
                     stc_max_iter=request_config["max_iter"])
    for name, value in saved.items():
        actual = getattr(args, name, None)
        if actual is not None:
            actual = str(actual) if name == "stc_algorithm" else int(actual)
            require_equal({name: value}, {name: actual}, f"STC search configuration at {root}")
    for name, default in OPTIONS.items():
        if name == "use_stc_sparse":
            continue
        value = saved.get(name, default)
        if getattr(args, name, None) is None:
            setattr(args, name, value if name in {"stc_algorithm", "stc_cache_root"} else int(value))
    args.stc_vocab_path = str(Path(root) / BUNDLE)
    validate(args)


def bind_corpus(args):
    """Use the immutable corpus copy; never build a new map while retrieving."""
    validate(args)
    if not enabled(args):
        return
    from adaptive_views import artifact_root, MANIFEST
    root = artifact_root(args.stage1_sparse_index)
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    config = manifest["configuration"].get(FIELD)
    if config is None:
        raise ValueError("USE_STC_SPARSE=true requires an STC-encoded corpus")
    validate_bundle(root, config)
    args.stc_vocab_path = str(root / BUNDLE)


class SparseEncoder:
    """Aggregate all eligible token probabilities before selecting sparse clusters."""
    def __init__(self, tokenizer, data, args):
        self.vocabulary = load(args, tokenizer=tokenizer)
        self.data, self.scale = data, int(args.stc_weight_scale)
        from qwen_vl_encoding import SparseEncoder as LegacyFilter
        term_filter = LegacyFilter(tokenizer, data, concat=True)
        raw = {value: key for key, value in tokenizer.get_vocab().items()}
        a = self.vocabulary.arrays
        self.vocab_ids = torch.tensor(a["vocab_ids"].astype(np.int64))
        keep = [i for i, token_id in enumerate(a["token_ids"]) if term_filter._term(raw[int(token_id)]) is not None]
        self.ids = torch.tensor(a["token_ids"][keep].astype(np.int64))
        self.clusters = torch.tensor(a["cluster_ids"][keep].astype(np.int64))
        self.vocab_positions = torch.searchsorted(self.vocab_ids, self.ids)
        self.count = int(self.vocabulary.manifest["cluster_count"])
        if not len(keep):
            raise ValueError("STC vocabulary has no tokens accepted by the sparse filters")

    def vector(self, raw_logits, record=None):
        if raw_logits.ndim != 2 or not len(raw_logits):
            raise ValueError("STC expects nonempty [views, vocabulary] raw logits")
        if int(self.vocab_ids[-1]) >= raw_logits.shape[-1]:
            raise ValueError("STC tokenizer IDs exceed the output-head vocabulary")
        # CPU scatter avoids nondeterministic CUDA atomic sums. Only K floats per
        # view are retained; no V-by-V matrix is constructed.
        logits = raw_logits.detach().float().cpu()[:, self.vocab_ids]
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite logits in STC sparse encoding")
        probabilities = logits.softmax(dim=-1)[:, self.vocab_positions]
        merged = {}
        for row in probabilities:
            mass = torch.zeros(self.count, dtype=torch.float32)
            mass.scatter_add_(0, self.clusters, row)
            order = torch.argsort(mass, descending=True, stable=True)[:min(self.data.sparse_length, self.count)]
            for index in order.tolist():
                value = float(mass[index])
                if value <= 0:
                    continue
                if index not in merged or self.data.sparse_value_type == "replace":
                    merged[index] = value
                elif self.data.sparse_value_type == "sum":
                    merged[index] += value
                elif self.data.sparse_value_type == "max":
                    merged[index] = max(merged[index], value)
                else:
                    raise ValueError("Unsupported STC view aggregation")
        divisor = len(raw_logits) if self.data.sparse_value_mean else 1
        result = {f"stc_{index:06d}": max(1, round(value * self.scale / divisor))
                  for index, value in sorted(merged.items())}
        if result and max(result.values()) >= 2**31:
            raise ValueError("STC integer weight exceeds Lucene's signed integer range")
        return result


def _save_arrays(path, arrays):
    path = Path(path)
    temp = path.with_name(".stc-" + uuid4().hex + ".npz")
    try:
        np.savez_compressed(temp, **arrays)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def build_mapping(input_embedding, output_embedding, tokenizer, args, directory, config):
    """Also usable with small CPU fixtures, independently of Transformers loaders."""
    from adaptive_views import atomic_json
    validate(args)
    vocab = tokenizer.get_vocab()
    ids = sorted(set(vocab.values()))
    if not ids or ids[0] < 0 or ids[-1] >= min(len(input_embedding), len(output_embedding)):
        raise ValueError("Tokenizer vocabulary is outside the input/output embedding rows")
    special = set(tokenizer.all_special_ids)
    decoded = tokenizer.batch_decode([[i] for i in ids], skip_special_tokens=True)
    eligible = [i for i, text in zip(ids, decoded) if i not in special and text.strip()]
    numeric = {i for i, text in zip(ids, decoded) if any(c.isdigit() for c in text)}
    semantic = [i for i in eligible if i not in numeric]
    count = int(args.stc_n_clusters)
    if not 1 <= count <= len(semantic):
        raise ValueError(f"STC_N_CLUSTERS={count} exceeds {len(semantic)} eligible nonnumeric tokens")
    print(f"[STC] {args.stc_algorithm}: semantic_tokens={len(semantic)} clusters={count}; "
          f"CPU pairwise float32 full-matrix equivalent={len(semantic)**2 * 4 / 1e9:.2f} GB "
          "(hierarchical peak may be higher); MiniBatchKMeans avoids pairwise storage", flush=True)
    selected = torch.tensor(semantic, dtype=torch.long)
    pieces = []
    for weight in (input_embedding, output_embedding):
        value = weight.detach().cpu()[selected].float()
        if not torch.isfinite(value).all() or (value.norm(dim=1) == 0).any():
            raise ValueError("STC cannot cluster non-finite or zero-norm token embeddings")
        pieces.append(F.normalize(value, dim=1))
    features = F.normalize(torch.cat(pieces, dim=1), dim=1).numpy()
    del pieces
    if count == len(semantic):
        labels = np.arange(count)
    elif args.stc_algorithm == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        labels = AgglomerativeClustering(n_clusters=count, metric="cosine", linkage="complete").fit_predict(features)
    else:
        from sklearn.cluster import MiniBatchKMeans
        labels = MiniBatchKMeans(n_clusters=count, random_state=int(args.stc_seed),
                                batch_size=int(args.stc_batch_size), max_iter=int(args.stc_max_iter),
                                init="random", n_init=3, reassignment_ratio=0).fit_predict(features)
    # Canonical labels by smallest member ID, independent of sklearn's label numbering.
    groups = {}
    for token_id, label in zip(semantic, labels):
        groups.setdefault(int(label), []).append(token_id)
    groups = sorted(groups.values(), key=min) + [[i] for i in eligible if i in numeric]
    mapping = {i: label for label, members in enumerate(groups) for i in members}
    arrays = dict(vocab_ids=np.array(ids, dtype=np.int32), token_ids=np.array(eligible, dtype=np.int32),
                  cluster_ids=np.array([mapping[i] for i in eligible], dtype=np.int32))
    manifest = dict(version=VERSION, request=config, arrays_sha256=arrays_digest(arrays),
                    cluster_count=len(groups), semantic_cluster_count=len(groups) - len(eligible) + len(semantic),
                    numeric_dimensions=len(eligible) - len(semantic))
    manifest["fingerprint"] = digest(manifest)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists():
        existing = Vocabulary(directory, expected=config)
        if existing.fingerprint != manifest["fingerprint"]:
            raise ValueError(f"Refusing to overwrite a different STC mapping: {directory}")
        return existing
    reverse = {value: key for key, value in vocab.items()}
    _save_arrays(directory / "token_to_cluster.npz", arrays)
    atomic_json(directory / "cluster_members.json", [dict(cluster_id=i, size=len(group),
                tokens=[dict(id=j, token=reverse[j]) for j in group]) for i, group in enumerate(groups)])
    # The manifest is the completion marker and is published last.
    atomic_json(directory / "manifest.json", manifest)
    return Vocabulary(directory, expected=config)


def build(args):
    """CPU-only precomputation, before any distributed/NCCL process is started."""
    validate(args)
    config, tokenizer = request(args)
    directory = cache_path(args, config)
    if (directory / "manifest.json").exists():
        vocabulary = Vocabulary(directory, expected=config)
        print(f"[STC] Reuse vocabulary: {directory}; fingerprint={vocabulary.fingerprint}", flush=True)
        return vocabulary
    if args.dry_run:
        print(f"[STC] Would build {args.stc_algorithm} vocabulary: {directory}", flush=True)
        return None
    directory.parent.mkdir(parents=True, exist_ok=True)
    lock = directory.with_name("." + directory.name + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError(f"STC precomputation is already locked: {lock}; wait for the other builder, or remove a stale lock after confirming it stopped") from exc
    try:
        if (directory / "manifest.json").exists():
            return Vocabulary(directory, expected=config)
        return _build_new(args, tokenizer, directory, config)
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)


def _build_new(args, tokenizer, directory, config):
    import transformers
    from adaptive_views import family_of
    family = family_of(transformers.AutoConfig.from_pretrained(args.model_name_or_path))
    classes = dict(qwen2_vl="Qwen2VLForConditionalGeneration", qwen2_5_vl="Qwen2_5_VLForConditionalGeneration",
                   qwen3_vl="Qwen3VLForConditionalGeneration", llava_llama="LlavaNextForConditionalGeneration",
                   llava_mistral="LlavaNextForConditionalGeneration")
    model_class = getattr(transformers, classes[family])
    print(f"[STC] Loading embeddings on CPU: {args.model_name_or_path}", flush=True)
    model = model_class.from_pretrained(args.model_name_or_path, torch_dtype=getattr(torch, args.stc_dtype),
                                       low_cpu_mem_usage=True, attn_implementation="eager", device_map="cpu")
    if args.lora_model_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_model_path).merge_and_unload()
    inputs = model.get_input_embeddings().weight.detach().cpu().clone()
    outputs = model.get_output_embeddings().weight.detach().cpu().clone()
    del model
    gc.collect()
    vocabulary = build_mapping(inputs, outputs, tokenizer, args, directory, config)
    print(f"[STC] Vocabulary: {directory}; dimensions={vocabulary.manifest['cluster_count']}; "
          f"fingerprint={vocabulary.fingerprint}", flush=True)
    return vocabulary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build"])
    add_parser(parser)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--dtype", dest="stc_dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--lora-model-path")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.use_stc_sparse, args.lora = True, bool(args.lora_model_path)
    args.stc_cache_root = args.stc_cache_root or "./sparse_output/stc_vocab"
    try:
        build(args)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
