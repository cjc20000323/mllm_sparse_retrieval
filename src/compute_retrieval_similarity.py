"""Compute dense/sparse retrieval agreement on saved test runs.

The search script writes one pickle for each sparse_manual setting.  Once both
pickles are available, this module computes query-level Overlap, Spearman rank
correlation, and RBO, then writes their averages to one Excel workbook.
"""

from __future__ import annotations

import argparse
import os
import pickle
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


TOP_KS = (10, 20, 50, 100)
DEFAULT_RBO_P = 0.9


def _score_map(record: Any) -> dict[str, float] | None:
    """Return a normalized doc-to-score mapping from a get_run_dict record."""
    if not isinstance(record, Mapping) or not isinstance(record.get("docs"), Mapping):
        return None

    scores: dict[str, float] = {}
    for doc_id, score in record["docs"].items():
        try:
            scores[str(doc_id)] = float(score)
        except (TypeError, ValueError):
            continue
    return scores


def _collect_score_maps(value: Any, query_id: str) -> list[dict[str, float]]:
    """Collect records from both the normal and nested run layouts."""
    direct = _score_map(value)
    if direct is not None:
        return [direct]

    if isinstance(value, Mapping):
        if query_id in value:
            return _collect_score_maps(value[query_id], query_id)
        records: list[dict[str, float]] = []
        for child in value.values():
            records.extend(_collect_score_maps(child, query_id))
        return records

    if isinstance(value, (list, tuple)):
        records = []
        for child in value:
            records.extend(_collect_score_maps(child, query_id))
        return records

    return []


def normalize_run(run: Mapping[Any, Any]) -> dict[str, dict[str, float]]:
    """Normalize a saved run to ``query_id -> doc_id -> score``.

    The normal search path already has one record per query.  If a task stores
    multiple records for one query, duplicate document scores are averaged so
    that the statistics script can still produce one ranking for that query.
    """
    normalized: dict[str, dict[str, float]] = {}
    for query_id, value in run.items():
        query_id = str(query_id)
        records = _collect_score_maps(value, query_id)
        if not records:
            continue

        merged: dict[str, list[float]] = {}
        for record in records:
            for doc_id, score in record.items():
                merged.setdefault(doc_id, []).append(score)
        normalized[query_id] = {
            doc_id: float(np.mean(scores)) for doc_id, scores in merged.items()
        }
    return normalized


def load_saved_runs(
        path: str | os.PathLike[str],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, Any]]:
    with open(path, "rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, Mapping):
        raise ValueError(f"Unsupported retrieval run file: {path}")

    dense_run = payload.get("dense_run")
    sparse_run = payload.get("sparse_run")
    if not isinstance(dense_run, Mapping) or not isinstance(sparse_run, Mapping):
        raise ValueError(
            f"{path} must contain 'dense_run' and 'sparse_run' mappings."
        )

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    return normalize_run(dense_run), normalize_run(sparse_run), dict(metadata)


def _top_docs(scores: Mapping[str, float], depth: int) -> list[str]:
    return [
        doc_id
        for doc_id, _ in sorted(
            scores.items(),
            key=lambda item: (-float(item[1]), str(item[0])),
        )[:depth]
    ]


def _rank_map(scores: Mapping[str, float]) -> dict[str, float]:
    """Rank documents with rank 1 as the highest score; average ties."""
    series = pd.Series(
        {str(doc_id): float(score) for doc_id, score in scores.items()},
        dtype=float,
    )
    return series.rank(method="average", ascending=False).to_dict()


def _spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if np.allclose(x_array, x_array[0]) or np.allclose(y_array, y_array[0]):
        return float("nan")
    return float(np.corrcoef(x_array, y_array)[0, 1])


def rank_biased_overlap(
        ranking_a: list[str],
        ranking_b: list[str],
        depth: int,
        p: float = DEFAULT_RBO_P,
) -> float:
    """Compute the finite-depth, extrapolated Rank-Biased Overlap."""
    if not 0.0 < p < 1.0:
        raise ValueError("RBO persistence p must be strictly between 0 and 1.")

    depth = min(depth, len(ranking_a), len(ranking_b))
    if depth == 0:
        return float("nan")

    seen_a: set[str] = set()
    seen_b: set[str] = set()
    weighted_agreement = 0.0
    agreement_at_depth = 0.0

    for current_depth in range(1, depth + 1):
        seen_a.add(ranking_a[current_depth - 1])
        seen_b.add(ranking_b[current_depth - 1])
        agreement_at_depth = len(seen_a & seen_b) / current_depth
        weighted_agreement += (
            (1.0 - p) * (p ** (current_depth - 1)) * agreement_at_depth
        )

    return float(weighted_agreement + (p ** depth) * agreement_at_depth)


def _query_metrics(
        dense_scores: Mapping[str, float],
        sparse_scores: Mapping[str, float],
        depth: int,
        rbo_p: float,
) -> dict[str, float]:
    dense_top = _top_docs(dense_scores, depth)
    sparse_top = _top_docs(sparse_scores, depth)
    sparse_top_set = set(sparse_top)
    common_docs = [doc_id for doc_id in dense_top if doc_id in sparse_top_set]

    denominator = min(depth, len(dense_top), len(sparse_top))
    overlap = (
        len(common_docs) / denominator if denominator > 0 else float("nan")
    )

    dense_top_scores = {doc_id: dense_scores[doc_id] for doc_id in dense_top}
    sparse_top_scores = {doc_id: sparse_scores[doc_id] for doc_id in sparse_top}
    dense_ranks = _rank_map(dense_top_scores)
    sparse_ranks = _rank_map(sparse_top_scores)
    spearman = _spearman(
        [dense_ranks[doc_id] for doc_id in common_docs],
        [sparse_ranks[doc_id] for doc_id in common_docs],
    )

    return {
        "overlap": float(overlap),
        "spearman": spearman,
        "rbo": rank_biased_overlap(dense_top, sparse_top, depth, rbo_p),
        "common_docs": float(len(common_docs)),
        "dense_docs": float(len(dense_top)),
        "sparse_docs": float(len(sparse_top)),
    }


def _nanmean(values: list[float]) -> float:
    valid = [value for value in values if np.isfinite(value)]
    return float(np.mean(valid)) if valid else float("nan")


def _calculate_condition(
        condition: str,
        dense_run: Mapping[str, Mapping[str, float]],
        sparse_run: Mapping[str, Mapping[str, float]],
        rbo_p: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    query_ids = sorted(set(dense_run) & set(sparse_run))
    per_query: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "condition": condition,
        "queries_with_dense_and_sparse": len(query_ids),
    }

    for query_id in query_ids:
        row: dict[str, Any] = {"condition": condition, "query_id": query_id}
        for depth in TOP_KS:
            values = _query_metrics(
                dense_run[query_id], sparse_run[query_id], depth, rbo_p
            )
            row[f"Overlap@{depth}"] = values["overlap"]
            row[f"Spearman@{depth}"] = values["spearman"]
            row[f"RBO@{depth}"] = values["rbo"]
            row[f"CommonDocs@{depth}"] = values["common_docs"]
        per_query.append(row)

    for depth in TOP_KS:
        for metric_name in ("Overlap", "Spearman", "RBO", "CommonDocs"):
            column = f"{metric_name}@{depth}"
            values = [row[column] for row in per_query]
            summary[column] = _nanmean(values)
        spearman_column = f"Spearman@{depth}"
        summary[f"ValidSpearmanQueries@{depth}"] = sum(
            np.isfinite(row[spearman_column]) for row in per_query
        )

    return summary, per_query


def compute_similarity(
        manual_path: str | os.PathLike[str],
        no_manual_path: str | os.PathLike[str],
        output_path: str | os.PathLike[str],
        rbo_p: float = DEFAULT_RBO_P,
) -> str:
    manual_dense, manual_sparse, manual_metadata = load_saved_runs(manual_path)
    no_manual_dense, no_manual_sparse, no_manual_metadata = load_saved_runs(
        no_manual_path
    )

    manual_summary, manual_rows = _calculate_condition(
        "manual", manual_dense, manual_sparse, rbo_p
    )
    no_manual_summary, no_manual_rows = _calculate_condition(
        "no_manual", no_manual_dense, no_manual_sparse, rbo_p
    )

    delta: dict[str, Any] = {"condition": "manual - no_manual"}
    for column in manual_summary:
        if column == "condition":
            continue
        manual_value = manual_summary.get(column)
        no_manual_value = no_manual_summary.get(column)
        if isinstance(manual_value, (int, float)) and isinstance(
                no_manual_value, (int, float)
        ):
            delta[column] = manual_value - no_manual_value

    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    metadata_rows = [
        {"item": "manual_run_file", "value": str(manual_path)},
        {"item": "no_manual_run_file", "value": str(no_manual_path)},
        {"item": "rbo_p", "value": rbo_p},
        {"item": "top_k", "value": ",".join(map(str, TOP_KS))},
    ]
    for key, value in sorted(manual_metadata.items()):
        metadata_rows.append({"item": f"manual_{key}", "value": value})
    for key, value in sorted(no_manual_metadata.items()):
        metadata_rows.append({"item": f"no_manual_{key}", "value": value})

    with pd.ExcelWriter(output_path) as writer:
        pd.DataFrame([manual_summary, no_manual_summary, delta]).to_excel(
            writer, sheet_name="summary", index=False
        )
        pd.DataFrame(manual_rows + no_manual_rows).to_excel(
            writer, sheet_name="per_query", index=False
        )
        pd.DataFrame(metadata_rows).to_excel(
            writer, sheet_name="metadata", index=False
        )

    return output_path


def compute_from_directory(
        stats_directory: str | os.PathLike[str],
        output_path: str | os.PathLike[str] | None = None,
        rbo_p: float = DEFAULT_RBO_P,
) -> str | None:
    stats_directory = str(stats_directory)
    manual_path = os.path.join(stats_directory, "test_runs_manual.pkl")
    no_manual_path = os.path.join(stats_directory, "test_runs_no_manual.pkl")
    if not (os.path.isfile(manual_path) and os.path.isfile(no_manual_path)):
        return None

    if output_path is None:
        output_path = os.path.join(
            stats_directory, "dense_sparse_similarity_test.xlsx"
        )
    return compute_similarity(manual_path, no_manual_path, output_path, rbo_p)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute test-set dense/sparse retrieval similarity."
    )
    parser.add_argument(
        "--stats_directory",
        required=True,
        help="Directory containing test_runs_manual.pkl and test_runs_no_manual.pkl.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output Excel path; defaults to the stats directory.",
    )
    parser.add_argument(
        "--rbo_p",
        type=float,
        default=DEFAULT_RBO_P,
        help="RBO persistence parameter, strictly between 0 and 1.",
    )
    args = parser.parse_args()

    output_path = compute_from_directory(
        args.stats_directory, args.output, args.rbo_p
    )
    if output_path is None:
        raise FileNotFoundError(
            "Both test_runs_manual.pkl and test_runs_no_manual.pkl are required."
        )
    print(f"Saved retrieval similarity statistics to: {output_path}")


if __name__ == "__main__":
    main()
