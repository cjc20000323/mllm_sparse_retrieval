"""FreeRet pre-MLP dense ablation; leave prompts and vocabulary logits intact."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
import shutil
from uuid import uuid4

SOURCE = "freeret_pre_mlp"
SUFFIX = "_dense_freeret_pre_mlp"
SIDECAR_SUFFIX = ".dense_representation.json"
CONFIGURATION = {"source": SOURCE, "version": 1}
PROMPT_SUFFIX = "_freeret_constraints_v2"
LEGACY_PROMPT_SUFFIXES = ("_freeret_constraints_v1",)
FULL_PROMPT_SUFFIX = "_freeret_full_prompt_v1"


def boolean(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"true", "1", "yes"}:
        return True
    if str(value).lower() in {"false", "0", "no"}:
        return False
    raise ValueError(f"USE_FREERET_DENSE must be true or false, got {value!r}")


def enabled(args):
    return boolean(getattr(args, "use_freeret_dense", False))


def configuration(config, use_freeret_dense, use_freeret_prompt_constraints=False, use_freeret_full_prompt=False):
    # Omitted/false must retain the original fingerprint payload exactly.
    if use_freeret_full_prompt and not use_freeret_dense:
        raise ValueError("USE_FREERET_FULL_PROMPT=true requires USE_FREERET_DENSE=true")
    if use_freeret_prompt_constraints and not use_freeret_dense:
        raise ValueError("USE_FREERET_PROMPT_CONSTRAINTS=true requires USE_FREERET_DENSE=true")
    result = {**config, "dense_representation": dict(CONFIGURATION)} if use_freeret_dense else config
    if use_freeret_full_prompt:
        from freeret_full_prompt import CONFIGURATION as full_configuration
        result = {**result, "freeret_full_prompt": dict(full_configuration)}
    elif use_freeret_prompt_constraints:
        from freeret_prompt_constraints import CONFIGURATION as prompt_configuration
        result = {**result, "freeret_prompt_constraints": dict(prompt_configuration)}
    return result


def _supported(config, model_path=""):
    kind = getattr(type(config), "model_type", None) or getattr(config, "model_type", None)
    if kind in {"qwen2_vl", "qwen2_5_vl", "qwen3_vl"}:
        return True
    text = getattr(config, "text_config", None)
    name = str(model_path).lower()
    if kind == "llava_next" and not any(tag in name for tag in ("vicuna", "34b")):
        return getattr(text, "model_type", None) == "mistral" or (
            getattr(text, "model_type", None) == "llama" and
            (getattr(text, "vocab_size", 0) >= 128000 or "llama3" in name))
    return False


def validate_checkpoint(model_path, use_freeret_dense=False):
    if not use_freeret_dense:
        return
    from transformers import AutoConfig
    if not _supported(AutoConfig.from_pretrained(model_path), model_path):
        raise ValueError("FreeRet dense supports llava_llama, llava_mistral, qwen2, qwen25 and qwen3 only: " + str(model_path))


def _last_norm(model):
    owners = [model]
    get_base = getattr(model, "get_base_model", None)
    if callable(get_base):
        base = get_base()
        if base is not model:
            owners.append(base)
    # Same node choices as freeret_retrieval.FreeRetQwen._resolve_last_decoder_norm.
    for owner in owners:
        for path in ("model.language_model.layers", "model.language_model.model.layers", "model.layers",
                     "language_model.layers", "language_model.model.layers"):
            layers = owner
            for part in path.split("."):
                layers = getattr(layers, part, None)
            if layers is not None and len(layers):
                norm = getattr(layers[-1], "post_attention_layernorm", None)
                if norm is not None and hasattr(norm, "register_forward_pre_hook"):
                    return path + "[-1].post_attention_layernorm", norm
    raise ValueError("Cannot locate the final decoder post_attention_layernorm for FreeRet dense; refusing another hidden-state source")


def configure_model(model, use_freeret_dense=False, model_path=""):
    use_freeret_dense = boolean(use_freeret_dense)
    handles = getattr(model, "_freeret_dense_hooks", None)
    if handles is not None:
        if use_freeret_dense:
            return model
        for handle in handles:
            handle.remove()
        delattr(model, "_freeret_dense_hooks")
    if not use_freeret_dense:
        return model
    if not _supported(model.config, model_path):
        raise ValueError("Unsupported model for FreeRet dense; supported families: llava_llama, llava_mistral, qwen2, qwen25, qwen3")
    path, norm = _last_norm(model)
    frames = []

    def begin(_module, _args, kwargs):
        frames.append(dict(requested=bool(kwargs.get("output_hidden_states", False) or
                                         getattr(model, "_freeret_dense_force_capture", False)), hidden=None))

    def capture(_module, inputs):
        if frames and frames[-1]["requested"] and inputs:
            frames[-1]["hidden"] = inputs[0].detach().clone()

    def finish(_module, _args, _kwargs, output):
        if not frames:
            return output
        frame = frames.pop()
        if output is None or not frame["requested"]:
            return output
        hidden = frame["hidden"]
        if hidden is None or hidden.ndim != 3:
            raise ValueError("FreeRet dense hook did not capture a batch/sequence/hidden tensor")
        # The full original forward (including LM head) has already completed.
        # Only the returned dense state changes. Existing EOL indexing and
        # normalization keep operating at exactly the same token positions.
        object.__setattr__(output, "_freeret_dense_hidden", hidden)
        states = getattr(output, "hidden_states", None)
        if states is not None:
            if states[-1].shape != hidden.shape:
                raise ValueError("FreeRet dense state shape differs from the existing EOL hidden state")
            output.hidden_states = (*states[:-1], hidden)
        return output

    handles = []
    try:
        handles.append(model.register_forward_pre_hook(begin, with_kwargs=True))
        handles.append(norm.register_forward_pre_hook(capture))
        handles.append(model.register_forward_hook(finish, with_kwargs=True, always_call=True))
    except Exception:
        for handle in handles:
            handle.remove()
        raise
    model._freeret_dense_hooks = handles
    print(f"[dense representation] source={SOURCE}; node={path}; token positions/pooling unchanged", flush=True)
    return model


@contextmanager
def force_capture(model):
    previous = getattr(model, "_freeret_dense_force_capture", False)
    model._freeret_dense_force_capture = True
    try:
        yield
    finally:
        model._freeret_dense_force_capture = previous


def forward(model, **inputs):
    # Memory-efficient DSPy encoding does not request all layer hidden states.
    with force_capture(model) if getattr(model, "_freeret_dense_hooks", None) else nullcontext():
        return model(**inputs)


def artifact_root(path):
    path = Path(path)
    if path.is_file() or path.suffix in {".pkl", ".pickle", ".trec", ".tsv"}:
        path = path.parent
    while path.name in {"index", "collection"}:
        path = path.parent
    return path


def marked(path):
    return artifact_root(path).name.endswith(SUFFIX) or prompts_marked(path) or full_prompts_marked(path)


def full_prompts_marked(path):
    return artifact_root(path).name.endswith(SUFFIX + FULL_PROMPT_SUFFIX)


def prompts_marked(path):
    return artifact_root(path).name.endswith(tuple(SUFFIX + item for item in (PROMPT_SUFFIX, *LEGACY_PROMPT_SUFFIXES)))


def validate_prompt_version(path):
    if artifact_root(path).name.endswith(tuple(SUFFIX + item for item in LEGACY_PROMPT_SUFFIXES)):
        raise ValueError(f"FreeRet prompt version mismatch at {path}: v1 lacks task_description; "
                         "re-encode and index with the current _freeret_constraints_v2 mode")


def mode_path(path, use_freeret_dense=False, use_freeret_prompt_constraints=False, use_freeret_full_prompt=False):
    if use_freeret_full_prompt and not use_freeret_dense:
        raise ValueError("USE_FREERET_FULL_PROMPT=true requires USE_FREERET_DENSE=true")
    if use_freeret_prompt_constraints and not use_freeret_dense:
        raise ValueError("USE_FREERET_PROMPT_CONSTRAINTS=true requires USE_FREERET_DENSE=true")
    if not use_freeret_dense or path is None:
        return path
    validate_prompt_version(path)
    path = Path(path)
    root = artifact_root(path)
    if full_prompts_marked(root):
        if not use_freeret_full_prompt:
            raise ValueError(f"Complete FreeRet prompt mode mismatch at {path}")
        return path
    if use_freeret_full_prompt:
        if prompts_marked(root):
            raise ValueError(f"Complete FreeRet prompt mode requires a new artifact; existing constraints directory: {path}")
        name = root.name[:-len(SUFFIX)] if root.name.endswith(SUFFIX) else root.name
        return root.with_name(name + SUFFIX + FULL_PROMPT_SUFFIX) / path.relative_to(root)
    if marked(root) and (not use_freeret_prompt_constraints or prompts_marked(root)):
        return path
    name = root.name[:-len(SUFFIX)] if root.name.endswith(SUFFIX) else root.name
    suffix = SUFFIX + (PROMPT_SUFFIX if use_freeret_prompt_constraints else "")
    return root.with_name(name + suffix) / path.relative_to(root)


def sidecar(path):
    root = artifact_root(path)
    return root.with_name(root.name + SIDECAR_SUFFIX)


def write_manifest(path, use_freeret_dense=False, use_freeret_prompt_constraints=False, use_freeret_full_prompt=False):
    if not use_freeret_dense:
        return
    validate_prompt_version(path)
    target = sidecar(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".dm-{uuid4().hex[:12]}.tmp")
    try:
        metadata = configuration({}, True, use_freeret_prompt_constraints, use_freeret_full_prompt)
        value = dict(CONFIGURATION)
        if use_freeret_full_prompt:
            value["freeret_full_prompt"] = metadata["freeret_full_prompt"]
        elif use_freeret_prompt_constraints:
            value["freeret_prompt_constraints"] = metadata["freeret_prompt_constraints"]
        temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def validate_artifact(path, use_freeret_dense=False, use_freeret_prompt_constraints=False, use_freeret_full_prompt=False):
    validate_prompt_version(path)
    expected = dict(CONFIGURATION)
    metadata = configuration({}, use_freeret_dense, use_freeret_prompt_constraints, use_freeret_full_prompt)
    use_freeret_prompt_constraints = use_freeret_prompt_constraints and not use_freeret_full_prompt
    if use_freeret_full_prompt:
        expected["freeret_full_prompt"] = metadata["freeret_full_prompt"]
    elif use_freeret_prompt_constraints:
        expected["freeret_prompt_constraints"] = metadata["freeret_prompt_constraints"]
    marker = sidecar(path)
    if marker.is_file():
        actual = json.loads(marker.read_text(encoding="utf-8"))
        if (not use_freeret_dense or actual != expected or prompts_marked(path) != use_freeret_prompt_constraints
                or full_prompts_marked(path) != use_freeret_full_prompt):
            raise ValueError(f"Dense representation mode mismatch at {path}: {actual}")
    elif use_freeret_dense or marked(path):
        raise ValueError(f"Missing dense representation metadata at {path}; re-encode with USE_FREERET_DENSE=true")


def prepare_entry(model_args, data, training, search=None):
    from freeret_full_prompt import enabled as full_enabled
    full = full_enabled(model_args)
    from freeret_prompt_constraints import enabled as prompt_enabled, validate
    validate(model_args, data.dataset_name)
    prompts = prompt_enabled(model_args)
    use = enabled(model_args)
    validate_checkpoint(model_args.model_name_or_path, use)
    if use:
        training.output_dir = str(mode_path(training.output_dir, True, prompts, full))
    if search is not None:
        for name in ("passage_reps", "sparse_index", "val_passage_reps", "val_sparse_index"):
            path = getattr(search, name, None)
            if path:
                path = mode_path(path, use, prompts, full)
                if use:
                    setattr(search, name, str(path))
                validate_artifact(path, use, prompts, full)


def prepare_pipeline(args):
    from freeret_full_prompt import enabled as full_enabled
    full = full_enabled(args)
    from freeret_prompt_constraints import enabled as prompt_enabled, validate
    validate(args, getattr(args, "dataset_name", None))
    prompts = prompt_enabled(args)
    use = enabled(args)
    model_path = getattr(args, "retrieval_model_name_or_path", None)
    validate_checkpoint(model_path, use and bool(model_path))
    if use and args.output_dir is not None:
        args.output_dir = str(mode_path(args.output_dir, True, prompts, full))
    for name in ("stage1_sparse_index", "stage2_dense_index", "stage2_sparse_index",
                 "val_passage_reps", "val_sparse_index", "val_stage1_sparse_index"):
        path = getattr(args, name, None)
        if path:
            path = mode_path(path, use, prompts, full)
            if use:
                setattr(args, name, str(path))
            validate_artifact(path, use, prompts, full)
    for name in ("stage2_run", "ranker_run"):
        path = getattr(args, name, None)
        if path:
            validate_artifact(path, use, prompts, full)


def original_path(path):
    path = Path(path)
    for suffix in (PROMPT_SUFFIX, FULL_PROMPT_SUFFIX, *LEGACY_PROMPT_SUFFIXES):
        if path.name.endswith(suffix):
            path = path.with_name(path.name[:-len(suffix)])
            break
    if not path.name.endswith(SUFFIX):
        raise ValueError(f"Expected a marked FreeRet dense artifact: {path}")
    return path.with_name(path.name[:-len(SUFFIX)])


def reuse_sparse_index(target, *, manifest_name=None):
    """Copy a completed index only after validating all current sparse documents.

    Dense encoding still requires its model forward; sparse conversion is cheap.
    Compare the resulting documents as well, including legacy corpora without
    encoding manifests. Copies have independent writable inodes.
    """
    target = Path(target)
    if prompts_marked(target) or full_prompts_marked(target):
        # Constraints change sparse logits. Never borrow the unaugmented index.
        return False
    source = original_path(target)
    if not source.is_dir():
        return False
    if source.resolve() != source or target.resolve() != target:
        raise ValueError("Refusing linked sparse reuse directories")
    if manifest_name:
        old_path, new_path = source / manifest_name, target / manifest_name
        if not old_path.is_file():
            raise ValueError(f"Incomplete original sparse artifacts: {old_path}")
        old = json.loads(old_path.read_text(encoding="utf-8"))
        new = json.loads(new_path.read_text(encoding="utf-8"))
        key = "configuration" if "configuration" in new else "config"
        expected = dict(new[key])
        expected.pop("dense_representation", None)
        if old[key] != expected:
            raise ValueError(f"Cannot reuse sparse artifacts with different encoding configuration: {source}")
        options = dict(sort_keys=True, ensure_ascii=False)
        if key == "configuration":
            options["separators"] = (",", ":")
        for root, value in ((source, old), (target, new)):
            digest = hashlib.sha256(json.dumps(value[key], **options).encode()).hexdigest()
            if value.get("fingerprint") != digest:
                raise ValueError(f"Corrupt sparse completion fingerprint at {root}")

    def documents(root):
        collection = root / "collection" if (root / "collection").is_dir() else root
        files, result = sorted(collection.glob("corpus_*.jsonl")), {}
        for file in files:
            if file.resolve() != file:
                raise ValueError(f"Refusing linked sparse shard: {file}")
            with file.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    identifier = str(row["id"])
                    if identifier in result:
                        raise ValueError(f"Duplicate sparse document {identifier} at {root}")
                    row["id"] = identifier
                    result[identifier] = hashlib.sha256(json.dumps(
                        row, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).digest()
        return files, result

    old_files, old_docs = documents(source)
    new_files, new_docs = documents(target)
    if not old_files:
        return False
    if old_docs != new_docs:
        raise ValueError(f"Sparse documents differ between dense modes at {target}; refusing index reuse")

    def file_digest(path):
        result = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                result.update(block)
        return result.digest()

    if len(old_files) == len(new_files):
        for old_file, new_file in zip(old_files, new_files):
            if file_digest(old_file) == file_digest(new_file):
                shutil.copyfile(old_file, new_file)
    index = source / "collection/index"
    if not index.exists():
        index = source / "index"
    if not index.exists():
        return False
    if index.resolve() != index or not index.is_dir() or not any(p.is_file() for p in index.glob("segments_*")):
        raise ValueError(f"Incomplete or linked source Lucene index: {index}")
    destination = target / ("collection/index" if (target / "collection").is_dir() else "index")
    if destination.exists():
        if destination.resolve() != destination or not destination.is_dir() or not any(p.is_file() for p in destination.glob("segments_*")):
            raise ValueError(f"Incomplete or linked destination Lucene index: {destination}")
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    for file in index.rglob("*"):
        if file.resolve() != file:
            raise ValueError(f"Refusing linked Lucene index entry: {file}")
    temporary = target / f".i-{uuid4().hex[:6]}"
    try:
        shutil.copytree(index, temporary, ignore=shutil.ignore_patterns("write.lock"))
        temporary.replace(destination)
    finally:
        if temporary.exists():
            if temporary.resolve().parent != target.resolve() or temporary.is_symlink():
                raise ValueError(f"Refusing to remove a relocated index staging directory: {temporary}")
            shutil.rmtree(temporary)
    print(f"[dense representation] reused unchanged sparse index: {source} -> {target}", flush=True)
    return True
