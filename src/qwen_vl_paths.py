"""Filesystem contract shared by Qwen encoding, indexing and retrieval."""
from pathlib import Path


MANIFEST_NAMES = ("qwen2_prompt_manifest.json", "qwen25_prompt_manifest.json", "qwen3_prompt_manifest.json")


def artifact_root(path):
    root = Path(path).expanduser()
    if root.is_file():
        root = root.parent
    if root.name == "index":
        root = root.parent
    if root.name == "collection":
        root = root.parent
    return root


def has_qwen_manifest(path):
    return any((artifact_root(path) / name).is_file() for name in MANIFEST_NAMES)


def resolve_sparse_index(path, *, require=True):
    """Prefer collection/index; an incomplete new index blocks legacy fallback.

    An explicitly supplied index directory selects that exact index. Root and
    collection aliases fall back to the legacy sibling only if the new path is absent.
    """
    path = Path(path).expanduser()
    if path.name == "index":
        index = path
    else:
        root = artifact_root(path)
        nested, legacy = root / "collection/index", root / "index"
        index = nested if nested.exists() or nested.is_symlink() else legacy
        if not index.exists() and not index.is_symlink():
            index = nested
    if require and not any(item.is_file() for item in index.glob("segments_*")):
        raise ValueError(f"Missing Qwen Lucene index (no segments_*) at {index}; run sparse_index_qwen.sh first")
    return index
