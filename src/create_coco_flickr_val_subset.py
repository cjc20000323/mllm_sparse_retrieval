#!/usr/bin/env python3
"""Create reproducible COCO/Flickr validation subsets.

Each selected example is one unique image with exactly one source caption.
The output is written as a generated-validation tree so it can be consumed by
dataset.py with --use_generated_validation_data and
--generated_validation_data_root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


DATASET_SCHEMAS: dict[str, tuple[str, ...]] = {
    "coco": ("imgid", "filepath", "filename", "caption", "sentid"),
    "flickr": ("imgid", "filename", "caption", "sentid"),
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _stable_dataset_seed(seed: int, dataset: str) -> int:
    digest = hashlib.sha256(f"{seed}:{dataset}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _input_path(data_root: Path, dataset: str) -> Path:
    return data_root / dataset / f"{dataset}_val.csv"


def _output_path(output_root: Path, dataset: str) -> Path:
    return output_root / "data" / dataset / f"{dataset}_val.csv"


def _load_rows(path: Path, dataset: str) -> tuple[list[str], dict[str, list[dict[str, str]]]]:
    if not path.is_file():
        raise FileNotFoundError(f"{dataset} validation CSV not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        required = set(DATASET_SCHEMAS[dataset])
        missing = sorted(required.difference(fieldnames))
        if missing:
            raise ValueError(
                f"{path} is missing required columns for {dataset}: {missing}"
            )

        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row_number, row in enumerate(reader, start=2):
            image_id = str(row.get("imgid", "")).strip()
            if not image_id:
                raise ValueError(f"{path}:{row_number} has an empty imgid")
            grouped[image_id].append(
                {name: str(row.get(name, "")) for name in fieldnames}
            )

    if not grouped:
        raise ValueError(f"{path} contains no data rows")
    return fieldnames, grouped


def _sample_rows(
    grouped: dict[str, list[dict[str, str]]],
    dataset: str,
    num_samples: int,
    seed: int,
) -> tuple[list[dict[str, str]], list[str]]:
    image_ids = sorted(grouped)
    if len(image_ids) < num_samples:
        raise ValueError(
            f"{dataset} validation set has only {len(image_ids)} unique images; "
            f"cannot sample {num_samples}."
        )

    rng = random.Random(_stable_dataset_seed(seed, dataset))
    selected_ids = rng.sample(image_ids, num_samples)
    selected_rows = [rng.choice(grouped[image_id]) for image_id in selected_ids]
    return selected_rows, selected_ids


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    project_root = _project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample unique validation images from COCO/Flickr and "
            "write dataset.py-compatible CSV subsets."
        )
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=project_root / "data",
        help="root containing coco/coco_val.csv and flickr/flickr_val.csv",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help=(
            "generated-validation output root; defaults to "
            "generated_val_subsets/coco_flickr_val_<num>_seed_<seed>"
        ),
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="number of unique images sampled independently per dataset",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_SCHEMAS),
        default=["coco", "flickr"],
        help="datasets to process; defaults to coco flickr",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing existing subset CSV files and manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.num_samples <= 0:
        raise SystemExit("--num_samples must be a positive integer")

    data_root = args.data_root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else (
            _project_root()
            / "generated_val_subsets"
            / f"coco_flickr_val_{args.num_samples}_seed_{args.seed}"
        ).resolve()
    )

    loaded: dict[str, tuple[list[str], list[dict[str, str]], list[str]]] = {}
    for dataset in args.datasets:
        fieldnames, grouped = _load_rows(_input_path(data_root, dataset), dataset)
        rows, selected_ids = _sample_rows(
            grouped,
            dataset,
            args.num_samples,
            args.seed,
        )
        loaded[dataset] = (fieldnames, rows, selected_ids)

    output_paths = {dataset: _output_path(output_root, dataset) for dataset in args.datasets}
    manifest_path = output_root / "manifest.json"
    existing = [
        path
        for path in [*output_paths.values(), manifest_path]
        if path.exists()
    ]
    if existing and not args.overwrite:
        formatted = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Output already exists; pass --overwrite to replace:\n" + formatted
        )

    manifest: dict[str, Any] = {
        "source_root": str(data_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "num_samples_per_dataset": args.num_samples,
        "sampling_unit": "unique_image_with_one_caption",
        "datasets": {},
    }

    for dataset, (fieldnames, rows, selected_ids) in loaded.items():
        output_path = output_paths[dataset]
        _write_csv(output_path, fieldnames, rows)
        manifest["datasets"][dataset] = {
            "source_file": str(_input_path(data_root, dataset)),
            "output_file": str(output_path),
            "num_rows": len(rows),
            "num_unique_images": len(set(selected_ids)),
            "image_ids": selected_ids,
        }
        print(
            f"{dataset}: sampled {len(selected_ids)} unique images, "
            f"wrote {output_path}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
