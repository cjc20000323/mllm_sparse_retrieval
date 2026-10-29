"""Build a Qwen impact index without exposing non-corpus files to Anserini."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from qwen_vl_paths import artifact_root


def build_sparse_index(sparse_root, *, threads=16):
    if threads < 1:
        raise ValueError("threads must be positive")
    root = artifact_root(sparse_root).resolve()
    collection = root / "collection"
    target = collection / "index"
    if collection.resolve() != collection or target.resolve() != target:
        raise ValueError(f"Refusing to build through a linked Qwen artifact: {collection}")
    if target.exists() or target.is_symlink():
        raise ValueError(f"Index output already exists: {target}; existing indexes are not overwritten")
    documents = sorted(collection.glob("corpus_*.jsonl"))
    if not documents:
        raise ValueError(f"No corpus_*.jsonl found in {collection}; run encode_qwen.sh first")
    for document in documents:
        if not document.is_file() or document.resolve() != document:
            raise ValueError(f"Expected an unlinked Qwen corpus file: {document}")
    # This scoped directory is outside collection, on the same filesystem as
    # the final index. Anserini recursively scans only the staged JSONL files.
    with tempfile.TemporaryDirectory(prefix=".qwen-index-", dir=root) as temporary:
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
            "--collection", "JsonVectorCollection", "--input", str(source),
            "--index", str(output), "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", str(threads), "--impact", "--pretokenized",
        ], check=True)
        if not any(item.is_file() for item in output.glob("segments_*")):
            raise ValueError(f"Index build produced no segments_*: {output}")
        # Recheck the checked destination immediately before publishing. Never
        # replace an existing index, even if another process created it meanwhile.
        if collection.resolve() != collection or target.resolve() != target:
            raise ValueError(f"Refusing to publish through a linked Qwen artifact: {target}")
        if target.exists() or target.is_symlink():
            raise ValueError(f"Index output already exists: {target}")
        output.rename(target)
    print(f"Qwen Lucene index: {target}")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse-root", required=True)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    try:
        build_sparse_index(args.sparse_root, threads=args.threads)
    except subprocess.CalledProcessError as exc:
        # Preserve the indexer's status for the Bash batch's stop-on-failure logic.
        raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    main()
