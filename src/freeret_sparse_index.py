"""Index/check ordinary LLaVA or Qwen artifacts in the requested FreeRet namespace."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from dense_representation import artifact_root, boolean, mode_path, validate_artifact


def checked_root(path, use_dense=False, use_constraints=False, use_full=False):
    requested = Path(os.path.abspath(Path(mode_path(path, use_dense, use_constraints, use_full)).expanduser()))
    root = artifact_root(requested)
    if not root.is_dir():
        raise ValueError(f"Corpus directory not found: {root}")
    if root.resolve() != root or requested.resolve() != requested:
        raise ValueError(f"Refusing linked artifact path: {requested}")
    validate_artifact(root, use_dense, use_constraints, use_full)
    if (root / "adaptive_views_manifest.json").exists():
        raise ValueError("These launchers use ordinary encode.py artifacts; use the adaptive index/retrieval launchers for adaptive views")
    return root, requested


def sparse_layout(path, family, use_dense=False, use_constraints=False, use_full=False):
    root, requested = checked_root(path, use_dense, use_constraints, use_full)
    source = root / "collection" if family.startswith("qwen") else root
    if not source.is_dir() or source.resolve() != source:
        raise ValueError(f"Sparse shard directory not found or linked: {source}")
    documents = sorted(source.glob("corpus_*.jsonl"))
    if not documents:
        raise ValueError(f"No corpus_*.jsonl found: {source}; run the matching encode_freeret script")
    if any(not item.is_file() or item.resolve() != item for item in documents):
        raise ValueError(f"Invalid/linked corpus shard: {source}")
    target = requested if requested.name == "index" else source / "index"
    if target.resolve() != target:
        raise ValueError(f"Refusing linked Lucene index: {target}")
    return root, documents, target


def build_index(path, family, *, threads=16, use_dense=False, use_constraints=False, use_full=False):
    if threads < 1:
        raise ValueError("threads must be positive")
    root, documents, target = sparse_layout(path, family, use_dense, use_constraints, use_full)
    if target.exists() or target.is_symlink():
        raise ValueError(f"Index output already exists: {target}; existing indexes are not overwritten")
    with tempfile.TemporaryDirectory(prefix=".freeret-index-", dir=root) as temporary:
        work = Path(temporary)
        source, output = work / "input", work / "index"
        source.mkdir()
        for document in documents:
            staged = source / document.name
            try:
                os.link(document, staged)
            except OSError:
                shutil.copy2(document, staged)
        subprocess.run([
            sys.executable, "-B", "-m", "pyserini.index.lucene",
            "--collection", "JsonVectorCollection", "--input", str(source), "--index", str(output),
            "--generator", "DefaultLuceneDocumentGenerator", "--threads", str(threads), "--impact", "--pretokenized",
        ], check=True)
        if not output.is_dir() or output.resolve() != output or not any(item.is_file() for item in output.glob("segments_*")):
            raise ValueError("Index build produced no valid segments_*")
        if target.exists() or target.is_symlink() or target.resolve() != target:
            raise ValueError(f"Index destination appeared or changed during the build: {target}")
        output.rename(target)
    print(f"Lucene index: {target}")
    return target


def check_dense(path, use_dense=False, use_constraints=False, use_full=False):
    root, requested = checked_root(path, use_dense, use_constraints, use_full)
    files = [requested] if requested.is_file() else sorted(root.glob("corpus*.pkl"))
    if not files or any(not item.is_file() or item.resolve() != item or not item.name.startswith("corpus") or item.suffix != ".pkl" for item in files):
        raise ValueError(f"No valid corpus*.pkl at {requested}")


def check_sparse(path, family, use_dense=False, use_constraints=False, use_full=False):
    _, _, target = sparse_layout(path, family, use_dense, use_constraints, use_full)
    if not target.is_dir() or not any(item.is_file() for item in target.glob("segments_*")):
        raise ValueError(f"Missing Lucene segments_* at {target}; run the matching sparse_index_freeret script")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "check"))
    parser.add_argument("--family", required=True, choices=("llava_llama", "llava_mistral", "qwen2", "qwen25", "qwen3"))
    parser.add_argument("--sparse-root", action="append", default=[])
    parser.add_argument("--dense-root", action="append", default=[])
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--use-freeret-dense", "--use_freeret_dense", type=boolean, default=False)
    parser.add_argument("--use-freeret-prompt-constraints", "--use_freeret_prompt_constraints", type=boolean, default=False)
    parser.add_argument("--use-freeret-full-prompt", "--use_freeret_full_prompt", type=boolean, default=False)
    args = parser.parse_args(argv)
    try:
        if args.action == "build":
            if len(args.sparse_root) != 1 or args.dense_root:
                raise ValueError("build requires one --sparse-root and no --dense-root")
            build_index(args.sparse_root[0], args.family, threads=args.threads,
                        use_dense=args.use_freeret_dense, use_constraints=args.use_freeret_prompt_constraints, use_full=args.use_freeret_full_prompt)
        else:
            if not args.sparse_root:
                raise ValueError("check requires --sparse-root")
            for path in args.dense_root:
                check_dense(path, args.use_freeret_dense, args.use_freeret_prompt_constraints, args.use_freeret_full_prompt)
            for path in args.sparse_root:
                check_sparse(path, args.family, args.use_freeret_dense, args.use_freeret_prompt_constraints, args.use_freeret_full_prompt)
    except subprocess.CalledProcessError as exc:
        return exc.returncode if exc.returncode >= 0 else 128 - exc.returncode
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
