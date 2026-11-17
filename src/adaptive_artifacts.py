"""Small, model-free handoff between adaptive encoding, indexing and retrieval."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from uuid import uuid4

MANIFEST = "adaptive_views_manifest.json"
PATH_FIELDS = ("dense_root", "sparse_root", "selection_file")


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def validate(record):
    if record.get("version") != 1:
        raise ValueError("Unsupported adaptive artifact-path record")
    configuration = record["configuration"]
    if _digest(configuration) != record["fingerprint"]:
        raise ValueError("Corrupt adaptive artifact-path fingerprint")
    for field in ("dense_root", "sparse_root"):
        root = Path(record[field])
        completion = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
        if (completion.get("configuration") != configuration
                or completion.get("fingerprint") != record["fingerprint"]
                or completion.get("selection_fingerprint") != record["selection_fingerprint"]):
            raise ValueError(f"Adaptive artifact-path record does not match {root / MANIFEST}")
    selection = json.loads(Path(record["selection_file"]).read_text(encoding="utf-8"))
    fingerprint = selection.pop("fingerprint")
    if (fingerprint != record["selection_fingerprint"] or _digest(selection) != fingerprint
            or selection.get("views_fingerprint") != configuration["views_fingerprint"]):
        raise ValueError("Adaptive artifact-path record does not match its view selection")
    return record


def publish(filename, roots, selection_file, metadata, selection_fingerprint):
    """Publish only after both completion manifests have been written successfully."""
    record = dict(version=1, dense_root=str(Path(roots[0]).resolve()),
                  sparse_root=str(Path(roots[1]).resolve()),
                  selection_file=str(Path(selection_file).resolve()),
                  configuration=metadata["configuration"], fingerprint=metadata["fingerprint"],
                  selection_fingerprint=selection_fingerprint)
    validate(record)
    path = Path(filename).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".aa-{uuid4().hex[:12]}.tmp")
    try:
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print("Adaptive artifact paths:", path.resolve())


def read(filename):
    return validate(json.loads(Path(filename).expanduser().read_text(encoding="utf-8")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="Path record published by --adaptive_artifacts_file")
    parser.add_argument("field", choices=PATH_FIELDS)
    args = parser.parse_args()
    try:
        print(read(args.file)[args.field])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
