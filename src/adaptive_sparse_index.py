"""Adaptive-only shard layout compatibility and canonical collection/index builds."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile

from adaptive_views import MANIFEST
from qwen_vl_paths import artifact_root


def _plain(path, *, directory=False):
    if path.is_symlink() or path.resolve() != path or not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"Expected an unlinked {'directory' if directory else 'file'}: {path}")


def _same_content(left, right):
    # Do not use filecmp's size/mtime cache: a resumed copy must compare bytes.
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as first, right.open("rb") as second:
        while True:
            chunk = first.read(1024 * 1024)
            if chunk != second.read(1024 * 1024):
                return False
            if not chunk:
                return True


@dataclass(frozen=True)
class CollectionPlan:
    root: Path
    documents: tuple[Path, ...]
    copies: tuple[Path, ...]

    @property
    def collection(self):
        return self.root / "collection"

    @property
    def target(self):
        return self.collection / "index"


def collection_plan(sparse_root, manifest=None):
    """Inspect without writing; prefer a complete collection over leftover root shards."""
    root = artifact_root(sparse_root).resolve()
    completion = root / MANIFEST
    if manifest is None:
        _plain(completion)
        manifest = json.loads(completion.read_text(encoding="utf-8"))
    elif completion.exists() or completion.is_symlink():
        # check_artifact also accepts an already loaded in-memory manifest.
        # Actual normalization always reloads the on-disk Adaptive completion.
        _plain(completion)
    count = manifest.get("num_shards")
    if type(count) is not int or count < 1:
        raise ValueError(f"Invalid num_shards in {completion}")
    collection = root / "collection"
    if collection.is_symlink() or collection.resolve() != collection:
        raise ValueError(f"Refusing linked adaptive collection/index at {root}")
    if collection.exists():
        _plain(collection, directory=True)
    documents = tuple(sorted(collection.glob("corpus_*.jsonl")))
    for document in documents:
        _plain(document)
    if len(documents) == count:
        return CollectionPlan(root, documents, ())
    legacy = tuple(sorted(root.glob("corpus_*.jsonl")))
    if len(documents) > count or len(legacy) != count:
        raise ValueError(f"Sparse shard count mismatch at {collection}: found {len(documents)}, expected {count}; "
                         f"root layout has {len(legacy)} shards (also requires {count})")
    for document in legacy:
        _plain(document)
    by_name = {document.name: document for document in legacy}
    for document in documents:
        source = by_name.get(document.name)
        if source is None or not _same_content(source, document):
            raise ValueError(f"Conflicting adaptive corpus shard: {document}; root/collection copies must be identical; "
                             "existing files are not overwritten")
    present = {document.name for document in documents}
    return CollectionPlan(root, legacy, tuple(document for document in legacy if document.name not in present))


def normalize_collection(sparse_root):
    """Copy legacy JSONL shards atomically, preserving originals and matching partial copies."""
    plan = collection_plan(sparse_root)
    if not plan.copies:
        return plan
    plan.collection.mkdir(exist_ok=True)
    _plain(plan.collection, directory=True)
    for source in plan.copies:
        _plain(source)
        target = plan.collection / source.name
        # A temporary file stays outside the corpus_*.jsonl namespace. Publishing
        # with link is atomic and refuses to replace even a concurrently created file.
        with tempfile.NamedTemporaryFile(prefix=".adaptive-shard-", suffix=".tmp", dir=plan.collection,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            try:
                with source.open("rb") as original:
                    shutil.copyfileobj(original, stream)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                stream.close()
                temporary.unlink()
                raise
        try:
            if not _same_content(source, temporary):
                raise ValueError(f"Adaptive corpus changed while copying: {source}")
            _plain(plan.collection, directory=True)
            try:
                os.link(temporary, target)
            except FileExistsError:
                _plain(target)
                if not _same_content(temporary, target):
                    raise ValueError(f"Conflicting adaptive corpus shard: {target}; existing files are not overwritten")
        finally:
            temporary.unlink()
        print(f"Adaptive shard: {source} -> {target} (original preserved)", flush=True)
    return collection_plan(plan.root)


def _existing_index(index):
    if not index.exists() and not index.is_symlink():
        return None
    _plain(index, directory=True)
    segments = list(index.glob("segments_*"))
    if not segments:
        raise ValueError(f"Incomplete existing index (no segments_*) at {index}; existing indexes are not overwritten")
    for segment in segments:
        _plain(segment)
    return index


def canonical_index(sparse_root):
    """Only collection/index completes a new Adaptive indexing job."""
    root = artifact_root(sparse_root).resolve()
    return _existing_index(root / "collection/index")


def resolve_sparse_index(path):
    """Read-only legacy compatibility; an incomplete canonical index remains an error."""
    supplied = Path(os.path.abspath(Path(path).expanduser()))
    root = artifact_root(supplied).resolve()
    if supplied.name == "index":
        index = _existing_index(supplied)
    else:
        index = canonical_index(root) or _existing_index(root / "index")
    if index is None:
        raise ValueError(f"Missing adaptive Lucene index (no segments_*) at {root / 'collection/index'}; "
                         f'run python -B src/adaptive_views.py index --sparse-root "{root}" first')
    return index


def build_sparse_index(sparse_root, *, threads=16, builder=None):
    """Called after Adaptive manifest/selection validation by both index entry points."""
    if threads < 1:
        raise ValueError("threads must be positive")
    # Fail on an incomplete destination before copying any shards.
    existing = canonical_index(sparse_root)
    plan = normalize_collection(sparse_root)
    if existing is not None:
        print(f"Skip: existing Adaptive Lucene index: {existing}", flush=True)
        return existing
    if builder is None:
        from qwen_sparse_index import build_sparse_index as builder
    return builder(plan.root, threads=threads)
