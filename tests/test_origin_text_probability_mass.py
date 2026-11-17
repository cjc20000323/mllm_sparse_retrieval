"""Verify the existing cumulative selector and its saved-probability handoff."""
import ast
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def mass():
    spec = importlib.util.spec_from_file_location("mass_probe", ROOT / "src/adaptive_view_probability_mass.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pool(count):
    return [{"label": chr(65 + i), "aspect": f"aspect {i}"} for i in range(count)]


@pytest.mark.parametrize("scores,threshold,labels,total", [
    ([.6, .3, .1], .9, ["A", "B"], .9),
    ([.6, .3, .1], .8, ["A"], .6),
    ([.95, .03, .02], .9, ["A"], .95),
    ([.6, .3, .1], 0, ["A"], .6),
    ([.6, .3, .1], 1, ["A", "B", "C"], 1),
    ([.1, .6, .3], .9, ["B", "C"], .9),
])
def test_current_prefix_rule(mass, scores, threshold, labels, total):
    selected, ranking, cumulative = mass.select(pool(len(scores)), scores, threshold)
    assert [row["label"] for row in selected] == labels
    assert cumulative == pytest.approx(total)
    assert sum(row["selected"] for row in ranking) == len(labels)


@pytest.mark.parametrize("threshold", [-.1, 1.1, float("nan"), float("inf"), True, "invalid"])
def test_invalid_threshold_rejected(mass, threshold):
    with pytest.raises(ValueError, match="finite number in"):
        mass.configuration(threshold)


def test_derive_all_scores_without_mutating_saved_source(mass, monkeypatch, tmp_path):
    # Load the real ranking helpers without importing any model runtime.
    namespace = {"SELECTION_MODALITIES": ("text", "image", "both"), "TOP_K": 5}
    names = {"digest", "rank_probabilities", "resolve_modality", "enabled_modalities", "modality_weights"}
    tree = ast.parse((ROOT / "src/adaptive_views.py").read_text(encoding="utf-8"))
    import hashlib
    namespace.update(hashlib=hashlib, json=json,
                     views_fingerprint=lambda selected, mode, **kwargs: namespace["digest"]([selected, mode]))
    exec(compile(ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names],
                            type_ignores=[]), "adaptive_views.py", "exec"), namespace)
    views = SimpleNamespace(**namespace)
    views.VERSION, views.PMI_VERSION = 12, 14
    views.VARIANT_VERSION, views.VARIANT_PMI_VERSION = 20, 21
    views.PREFIX_VERSION, views.PREFIX_PMI_VERSION = 26, 27
    views.is_overlap_selection = views.is_pmi_selection = views.uses_token_variants = lambda _: False
    views.uses_prefix_tokens = views.uses_choice_scoring = lambda _: False
    views.selection_version = lambda probability_mass=False, **kwargs: 18 if probability_mass else 12
    views.resolve_pmi_prior_prompt_mode = lambda **kwargs: "false"
    views.pool_mode = lambda _: None
    views.add_pool_mode = lambda value, _: value
    views.examples = SimpleNamespace(FIELD="examples", add_configuration=lambda value, _: value)
    views.option_shuffle = SimpleNamespace(FIELD="shuffle", add_metadata=lambda value, _: value)
    monkeypatch.setitem(sys.modules, "adaptive_views", views)
    monkeypatch.setitem(sys.modules, "dense_representation", SimpleNamespace(boolean=lambda value: value))
    scores = [.35, .25, .15, .1, .08, .04, .03]
    candidates = pool(len(scores))
    source = dict(version=12, pool=candidates, selection_modality="text",
                  rounds=[{"text_probabilities": scores}],
                  **views.rank_probabilities([scores], None, candidates, selection_modality="text"))
    source["fingerprint"] = views.digest(source)
    original = copy.deepcopy(source)
    file = tmp_path / "source.json"
    file.write_text(json.dumps(source), encoding="utf-8")
    original_bytes = file.read_bytes()
    derived = mass.derive(source, .9)
    other_threshold = mass.derive(derived, .99)
    same_count = mass.derive(source, .91)
    assert source == original and file.read_bytes() == original_bytes
    assert len(source["selected"]) == 5
    assert derived["selected_count"] == same_count["selected_count"] == 4
    assert other_threshold["selected_count"] == 6  # Uses all scores, including those outside the old Top-5.
    assert derived["views_fingerprint"] != same_count["views_fingerprint"]
    assert derived["fingerprint"] != other_threshold["fingerprint"]
    assert derived["probability_source_fingerprint"] == source["fingerprint"]
    assert other_threshold["probability_source_fingerprint"] == source["fingerprint"]
    assert derived["scores"] == other_threshold["scores"] == scores
    # Omitted retrieval flags inherit either mode; explicit conflicts still fail.
    mass.validate_requested(source, None, None)
    mass.validate_requested(derived, None, None)
    with pytest.raises(ValueError, match="mode differs"):
        mass.validate_requested(derived, False)
    with pytest.raises(ValueError, match="threshold differs"):
        mass.validate_requested(derived, True, .95)
