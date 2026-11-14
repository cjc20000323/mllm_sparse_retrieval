"""Select a probability-ranked prefix without changing the underlying scores."""
from __future__ import annotations

import copy
import math

STRATEGY = "cumulative_probability"
VERSION = 18
PMI_VERSION = 19
VARIANT_VERSION = 24
VARIANT_PMI_VERSION = 25
PREFIX_VERSION = 30
PREFIX_PMI_VERSION = 31
DEFAULT_THRESHOLD = 0.9
TOLERANCE = 1e-12
FIELD = "probability_mass_configuration"
DERIVED_FIELDS = ("selection_strategy", FIELD, "probability_source_fingerprint",
                  "probability_mass_ranking", "selected_probability_mass", "selected_count", "minimum_one_fallback")


def enabled(options):
    from dense_representation import boolean
    use = boolean(getattr(options, "view_probability_mass", False) or False)
    if use and boolean(getattr(options, "view_overlap_selection", False) or False):
        raise ValueError("Cumulative probability selection and overlap are mutually exclusive; disable overlap")
    if use:
        threshold = getattr(options, "view_probability_mass_threshold", None)
        configuration(DEFAULT_THRESHOLD if threshold is None else threshold)
    return use


def configuration(threshold=DEFAULT_THRESHOLD):
    if isinstance(threshold, bool):
        raise ValueError("VIEW_PROBABILITY_MASS_THRESHOLD must be a finite number in [0, 1]")
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("VIEW_PROBABILITY_MASS_THRESHOLD must be a finite number in [0, 1]") from exc
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("VIEW_PROBABILITY_MASS_THRESHOLD must be a finite number in [0, 1]")
    return dict(strategy=STRATEGY, version=2, threshold=threshold, comparison="le", tolerance=TOLERANCE,
                minimum_selected_count=1)


def is_selection(selection):
    return selection.get("version") in {VERSION, PMI_VERSION, VARIANT_VERSION, VARIANT_PMI_VERSION, PREFIX_VERSION, PREFIX_PMI_VERSION}


def _exceeds(value, threshold):
    return value > threshold and not math.isclose(value, threshold, rel_tol=0, abs_tol=TOLERANCE)


def _legacy_configuration(threshold):
    return dict(strategy=STRATEGY, version=1, threshold=configuration(threshold)["threshold"],
                comparison="gt", tolerance=TOLERANCE)


def is_legacy_selection(selection):
    return is_selection(selection) and selection.get(FIELD, {}).get("version") == 1


def require_current_rule(selection):
    if is_legacy_selection(selection):
        import adaptive_views as views
        raise views.IncompatibleViewSelectionError(
            "Legacy cumulative probability rule v1 retains the crossing view; current v2 stops before it "
            "(minimum one view). Re-encode the corpus and rebuild its index. To reuse probabilities only, "
            "pass the old JSON as VIEW_SELECTION_FILE with USE_VIEW_PROBABILITY_MASS=true when encoding; "
            "the corrected selection is saved separately. Do not rename old artifacts.")


def _select(pool, scores, config):
    """The v1 branch is used solely to validate explicitly reused probability sources."""
    if (not pool or len(scores) != len(pool)
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value < 0 for value in scores)
            or not math.isclose(math.fsum(scores), 1, rel_tol=0, abs_tol=1e-6)):
        raise ValueError("Cumulative selection requires finite, normalized final perspective probabilities")
    order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))
    chosen, rows, stop = [], [], False
    for rank, index in enumerate(order, 1):
        cumulative = math.fsum(scores[i] for i in order[:rank])
        exceeds = config["threshold"] < 1 and _exceeds(cumulative, config["threshold"])
        if config["version"] == 2 and chosen and exceeds:
            stop = True
        if not stop:
            chosen.append(index)
        rows.append(dict(rank=rank, label=pool[index]["label"], aspect=pool[index]["aspect"],
                         probability=scores[index], cumulative_probability=cumulative, selected=not stop))
        if config["version"] == 1 and exceeds:
            stop = True
    return ([dict(label=pool[i]["label"], aspect=pool[i]["aspect"]) for i in chosen], rows,
            math.fsum(scores[i] for i in chosen))


def select(pool, scores, threshold):
    return _select(pool, scores, configuration(threshold))


def probability_source(selection):
    """Recover the immutable ordinary probability record from a derived prefix."""
    import adaptive_views as views
    if views.is_overlap_selection(selection) or selection.get("overlap_configuration") is not None:
        raise ValueError("Overlap selections cannot supply cumulative probability views")
    source = copy.deepcopy(selection)
    if not is_selection(source):
        if source.get("version") not in {views.VERSION, views.PMI_VERSION, views.VARIANT_VERSION, views.VARIANT_PMI_VERSION, views.PREFIX_VERSION, views.PREFIX_PMI_VERSION}:
            raise ValueError("Expected an ordinary/PMI probability selection")
        return source
    pmi = views.is_pmi_selection(source)
    for key in DERIVED_FIELDS:
        source.pop(key, None)
    order = sorted(range(len(source["pool"])), key=lambda i: (-source["scores"][i], i))
    source["selected"] = [dict(label=source["pool"][i]["label"], aspect=source["pool"][i]["aspect"])
                          for i in order[:views.TOP_K]]
    source["version"] = views.selection_version(pmi=pmi, token_variants=views.uses_token_variants(selection), prefix_tokens=views.uses_prefix_tokens(selection))
    source["views_fingerprint"] = views.views_fingerprint(source["selected"], source["selection_modality"],
        token_variants=views.uses_token_variants(source), prefix_tokens=views.uses_prefix_tokens(source),
        pmi=pmi, prior_prompt_mode=views.resolve_pmi_prior_prompt_mode(selection=source), shuffle=source.get(views.option_shuffle.FIELD), pool_mode=views.pool_mode(source),
        example_configuration=source.get(views.examples.FIELD))
    source["fingerprint"] = views.digest({key: value for key, value in source.items() if key != "fingerprint"})
    return source


def fingerprint(selection):
    import adaptive_views as views
    payload = dict(version=selection["version"], selection_strategy=STRATEGY,
        selection_modality=selection["selection_modality"], selected=selection["selected"],
        probability_mass_configuration=selection[FIELD],
        probability_source_fingerprint=selection["probability_source_fingerprint"])
    if views.uses_choice_scoring(selection):
        payload["choice_scoring"] = views.choice_configuration(prefix_tokens=views.uses_prefix_tokens(selection))
    return views.digest(views.examples.add_configuration(views.add_pool_mode(views.option_shuffle.add_metadata(
        payload, selection.get(views.option_shuffle.FIELD)), views.pool_mode(selection)), selection.get(views.examples.FIELD)))


def _derive(source, config):
    import adaptive_views as views
    chosen, ranking, cumulative = _select(source["pool"], source["scores"], config)
    result = copy.deepcopy(source)
    result.update(version=views.selection_version(pmi=views.is_pmi_selection(source), probability_mass=True,
                  token_variants=views.uses_token_variants(source), prefix_tokens=views.uses_prefix_tokens(source)),
        selection_strategy=STRATEGY, probability_source_fingerprint=source["fingerprint"],
        probability_mass_configuration=config, selected=chosen,
        probability_mass_ranking=ranking, selected_probability_mass=cumulative, selected_count=len(chosen))
    if config["version"] == 2:
        result["minimum_one_fallback"] = (len(chosen) == 1 and config["threshold"] < 1
                                          and _exceeds(cumulative, config["threshold"]))
    result["views_fingerprint"] = fingerprint(result)
    result["fingerprint"] = views.digest({key: value for key, value in result.items() if key != "fingerprint"})
    return result


def derive(selection, threshold=DEFAULT_THRESHOLD):
    source = probability_source(selection)
    validate_probabilities(source)
    return _derive(source, configuration(threshold))


def validate_probabilities(selection):
    import numpy as np
    import adaptive_views as views
    if views.uses_choice_scoring(selection):
        views.validate_choice_selection(selection)
    matrices = {name: [row[name + "_probabilities"] for row in selection["rounds"]]
                if name in views.enabled_modalities(selection["selection_modality"]) else None
                for name in ("text", "image")}
    ranking = views.rank_probabilities(matrices["text"], matrices["image"], selection["pool"],
                                      selection_modality=selection["selection_modality"])
    for key in ("scores", "text_mean", "image_mean"):
        actual, expected = selection.get(key), ranking[key]
        if ((actual is None) != (expected is None)
                or (expected is not None and (np.shape(actual) != np.shape(expected)
                    or not np.allclose(actual, expected, atol=1e-7, rtol=0)))):
            raise ValueError(f"Cumulative selection {key} differs from its probability rounds")
    if views.is_pmi_selection(selection):
        views.validate_pmi_selection(selection, check_selected=False)


def validate_selection(selection, *, allow_legacy_source=False):
    import adaptive_views as views
    if not is_selection(selection) or selection.get("selection_strategy") != STRATEGY:
        raise ValueError("Expected a cumulative probability selection")
    config = selection.get(FIELD, {})
    legacy = is_legacy_selection(selection)
    if legacy and not allow_legacy_source:
        require_current_rule(selection)
    expected_config = _legacy_configuration(config.get("threshold")) if legacy else configuration(config.get("threshold"))
    views.require_equal(config, expected_config, "Cumulative probability strategy")
    validate_probabilities(selection)
    source = probability_source(selection)
    if source["fingerprint"] != selection.get("probability_source_fingerprint"):
        raise ValueError("Cumulative probability source fingerprint mismatch")
    expected = _derive(source, expected_config)
    for key in (*DERIVED_FIELDS, "selected", "views_fingerprint"):
        if legacy and key == "minimum_one_fallback":
            if key in selection:
                raise ValueError("Legacy cumulative probability source contains a v2 fallback field")
            continue
        if selection.get(key) != expected[key]:
            raise ValueError(f"Invalid cumulative probability {key}")
    return len(expected["selected"])


def validate_requested(selection, requested=None, threshold=None, *, allow_legacy_source=False):
    if not (allow_legacy_source and requested is None and threshold is None):
        require_current_rule(selection)
    if requested is None:
        if threshold is not None and is_selection(selection) and configuration(threshold) != selection.get(FIELD):
            raise ValueError("Cumulative probability threshold differs from encoding; match VIEW_PROBABILITY_MASS_THRESHOLD")
        return
    from dense_representation import boolean
    requested = boolean(requested)
    if requested != is_selection(selection):
        raise ValueError("Cumulative probability mode differs from the encoded selection; match USE_VIEW_PROBABILITY_MASS")
    if requested and configuration(DEFAULT_THRESHOLD if threshold is None else threshold) != selection.get(FIELD):
        raise ValueError("Cumulative probability threshold differs from encoding; match VIEW_PROBABILITY_MASS_THRESHOLD")


def add_metadata(value, selection):
    return {**value, "selection_strategy": STRATEGY, FIELD: dict(selection[FIELD])} if is_selection(selection) else value


def mark_output(path, selection):
    """Keep the FreeRet namespace suffix last, including when called repeatedly."""
    from pathlib import Path
    from dense_representation import SUFFIX, PROMPT_SUFFIX, FULL_PROMPT_SUFFIX
    path = Path(path)
    name, tail = path.name, ""
    for marker in (SUFFIX + FULL_PROMPT_SUFFIX, SUFFIX + PROMPT_SUFFIX, SUFFIX):
        if name.endswith(marker):
            name, tail = name[:-len(marker)], marker
            break
    suffix = "_pm_" + selection["views_fingerprint"][:16]
    return path.with_name((name if name.endswith(suffix) else name + suffix) + tail)


def log_selection(selection):
    if not is_selection(selection):
        return
    print(f"[adaptive probability mass] threshold={selection[FIELD]['threshold']} "
          f"rule_version={selection[FIELD]['version']} comparison={selection[FIELD]['comparison']} "
          f"selected_count={len(selection['selected'])} cumulative_probability={selection['selected_probability_mass']:.12f} "
          f"minimum_one_fallback={selection.get('minimum_one_fallback', False)}", flush=True)
    rejected = next((row for row in selection["probability_mass_ranking"] if not row["selected"]), None)
    if rejected is not None:
        print(f"[adaptive probability mass] stopped_before_rank={rejected['rank']} view={rejected['label']} "
              f"would_be_probability={rejected['cumulative_probability']:.12f}", flush=True)
    for row in selection["probability_mass_ranking"]:
        print(f"[adaptive probability mass] rank={row['rank']} view={row['label']} "
              f"probability={row['probability']:.12f} cumulative={row['cumulative_probability']:.12f} "
              f"selected={row['selected']}", flush=True)


class ProbabilityMassStrategy:
    def __init__(self, options, model_args, data, task=None):
        enabled(options)
        self.options, self.model_args, self.data = options, model_args, data
        threshold = getattr(options, "view_probability_mass_threshold", None)
        self.configuration = configuration(DEFAULT_THRESHOLD if threshold is None else threshold)

    def validate(self, selection, *, inherit=False):
        validate_requested(selection, True, self.configuration["threshold"])
        validate_selection(selection)

    def report(self, selection, *, reused=False):
        log_selection(selection)

    def choose(self, model, processor, identity, dataset, source_split, texts, images, options, *, report_path=None):
        import adaptive_views as views
        mode = views.resolve_modality(options.view_selection_modality)
        pool_mode = views.options_pool_mode(options)
        shuffle = views.option_shuffle.for_options(options, identity, dataset, views.view_pool(pool_mode))
        sampling = views.selection_sampling(options, mode, model_config=model.config)
        key = views.selection_cache_key(identity, dataset, source_split, mode, sampling,
            options.adaptive_data_root, prior_prompt_mode=getattr(options, "view_pmi_prior_prompt_mode", None), pmi=views.resolve_pmi(options.view_selection_pmi), shuffle=shuffle, pool_mode=pool_mode, token_variants=getattr(options, "view_selection_token_variants", False), prefix_tokens=getattr(options, "view_selection_prefix_tokens", False))
        path = views.selection_cache_path(self.data.sparse_output_dir, self.model_args.model_name_or_path, dataset, key)
        if path.is_file():
            source = views.load_selection(path, identity=identity, dataset=dataset, selection_modality=mode,
                view_selection_pmi=views.resolve_pmi(options.view_selection_pmi), view_pmi_prior_prompt_mode=getattr(options, "view_pmi_prior_prompt_mode", None), view_selection_hierarchical=pool_mode is not None,
                view_example_mode=options.view_example_mode, view_example_max_tokens=options.view_example_max_tokens,
                view_probability_mass=False, view_selection_token_variants=getattr(options, "view_selection_token_variants", False), view_selection_prefix_tokens=getattr(options, "view_selection_prefix_tokens", False))
            views.require_equal(source["sampling"], sampling, "Probability source sampling")
            views.require_equal(dict(mapping=source.get(views.option_shuffle.FIELD)), dict(mapping=shuffle), "Probability source shuffle")
            print(f"[adaptive probability mass] reused probability source: {path}", flush=True)
        else:
            source = views.choose_views(model, processor, identity, dataset, source_split, texts, images, options)
            views.atomic_json(path, source)
            views.atomic_json(path.with_name("view_pool.json"), source["pool"])
            if shuffle is not None:
                views.atomic_json(path.with_name("option_mapping.json"), shuffle)
        result = derive(source, self.configuration["threshold"])
        if report_path is not None:
            views.atomic_json(report_path, dict(configuration=self.configuration, source_file=str(path),
                source_fingerprint=source["fingerprint"], ranking=result["probability_mass_ranking"],
                selected_count=result["selected_count"], selected_probability_mass=result["selected_probability_mass"],
                minimum_one_fallback=result["minimum_one_fallback"]))
        log_selection(result)
        return result
