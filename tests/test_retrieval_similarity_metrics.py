"""Agreement-report regressions using explicit rankings, without model imports."""
import csv
import importlib.util
import math
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def similarity():
    path = Path(__file__).resolve().parents[1] / "src/compute_retrieval_similarity.py"
    spec = importlib.util.spec_from_file_location("similarity_metrics_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize("dense,sparse,overlap,spearman,rbo", [
    ({"a": 3., "b": 2., "c": 1.}, {"a": 3., "b": 2., "c": 1.}, 1., 1., 1.),
    ({"a": 3., "b": 2., "c": 1.}, {"a": 1., "b": 2., "c": 3.}, 1., -1., .855),
    ({"a": 1., "b": 1., "c": 1.}, {"a": 1., "b": 1., "c": 1.}, 1., None, 1.),
    ({"a": 2., "b": 1.}, {"c": 2., "d": 1.}, 0., None, 0.),
    ({"a": 2., "b": 1.}, {"a": 2., "c": 1.}, .5, None, .55),
    ({}, {"a": 1.}, None, None, None),
])
def test_agreement_metrics_and_undefined_ranks(similarity, tmp_path, dense, sparse, overlap, spearman, rbo):
    path = similarity.compute_run_similarity({"q": dense}, {"q": sparse}, tmp_path / "metrics.csv",
                                             condition="adaptive", metadata={"selection_fingerprint": "locked"})
    rows = report_rows(path)
    summary, = [row for row in rows if row["table_type"] == "summary"]
    query, = [row for row in rows if row["table_type"] == "per_query"]
    assert query["query_id"] == "q"
    for depth in (10, 20, 50, 100):
        for metric, expected in (("Overlap", overlap), ("Spearman", spearman), ("RBO", rbo)):
            for row in (summary, query):
                value = float(row[f"{metric}@{depth}"] or "nan")
                if expected is None:
                    assert math.isnan(value)
                else:
                    assert value == pytest.approx(expected)
        assert int(summary[f"ValidSpearmanQueries@{depth}"]) == (0 if spearman is None else 1)
    metadata = {row["item"]: row["value"] for row in rows if row["table_type"] == "metadata"}
    assert metadata["adaptive_selection_fingerprint"] == "locked"


def test_summary_ignores_undefined_metrics_and_keeps_empty_queries(similarity, tmp_path):
    scores = {"a": 3., "b": 2., "c": 1.}
    dense = {"same": scores, "reverse": scores, "ties": dict.fromkeys(scores, 1.), "empty": {}}
    sparse = {"same": scores, "reverse": {"a": 1., "b": 2., "c": 3.},
              "ties": dict.fromkeys(scores, 1.), "empty": scores}
    path = similarity.compute_run_similarity(dense, sparse, tmp_path / "metrics.csv")
    rows = report_rows(path)
    summary, = [row for row in rows if row["table_type"] == "summary"]
    assert int(summary["queries_with_dense_and_sparse"]) == 4
    assert len([row for row in rows if row["table_type"] == "per_query"]) == 4
    for depth in (10, 20, 50, 100):
        assert float(summary[f"Overlap@{depth}"]) == 1.
        assert float(summary[f"Spearman@{depth}"]) == pytest.approx(0.)
        assert int(summary[f"ValidSpearmanQueries@{depth}"]) == 2
