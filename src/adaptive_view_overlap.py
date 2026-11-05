"""Sparse-overlap selection used only by the new copied encode entrypoint."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
import math
import random
from typing import Optional

import adaptive_views as views


STRATEGY = "sparse_overlap_frequency"
RULE = "candidate_fraction_lt"
PROBE_LENGTH = 30
DEFAULTS = dict(selected_count=5, threshold=0.1, samples_per_modality=20, seed=42)
OPTION_NAMES = dict(selected_count="view_selected_count", threshold="view_overlap_threshold",
                    samples_per_modality="view_overlap_samples", seed="view_overlap_seed")
PROBE_FIELDS = ("reps_loc", "tbpr_type", "cir_type", "sparse_value_type", "sparse_lower_or_upper",
                "sparse_value_mean", "is_filtered", "num_expended_tokens")


@dataclass
class OverlapViewArguments(views.AdaptiveViewArguments):
    view_overlap_selection: bool = field(default=False, metadata={"help": "Enable sparse-overlap frequency selection."})
    # None distinguishes omitted options from explicit values when reusing a file.
    view_selected_count: Optional[int] = field(default=None, metadata={"help": "Selected views, default 5; inherit a reused file."})
    view_overlap_threshold: Optional[float] = field(default=None, metadata={"help": "Accept candidate overlap strictly below this value, default 0.1."})
    view_overlap_samples: Optional[int] = field(default=None, metadata={"help": "New samples per enabled modality, default 20."})
    view_overlap_seed: Optional[int] = field(default=None, metadata={"help": "Independent resampling seed, default 42."})


def validate_configuration(configuration, *, pool_mode=None):
    if "freeret_full_prompt" in configuration:
        from freeret_full_prompt import CONFIGURATION
        views.require_equal(configuration["freeret_full_prompt"], CONFIGURATION, "Overlap complete FreeRet prompt")
    if configuration.get("rule") != RULE:
        raise ValueError("Overlap requires rule=candidate_fraction_lt; reselect views, encode and rebuild the index")
    count, samples, seed = (configuration.get(name) for name in ("selected_count", "samples_per_modality", "seed"))
    size = len(views.view_pool(pool_mode))
    if type(count) is not int or not 1 <= count <= size:
        raise ValueError(f"view_selected_count must be an integer in [1,{size}]")
    if type(samples) is not int or samples < 1:
        raise ValueError("view_overlap_samples must be a positive integer")
    if type(seed) is not int:
        raise ValueError("view_overlap_seed must be an integer")
    threshold = configuration.get("threshold")
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("view_overlap_threshold must be finite and in [0,1]")
    if configuration.get("probe_sparse_length") != PROBE_LENGTH:
        raise ValueError("Overlap probes must use sparse top-30")
    if "freeret_prompt_constraints" in configuration:
        from freeret_prompt_constraints import CONFIGURATION
        views.require_equal(configuration["freeret_prompt_constraints"], CONFIGURATION, "Overlap prompt constraints")
    if set(configuration.get("probe_encoding", {})) != set(PROBE_FIELDS):
        raise ValueError("Missing overlap probe encoding configuration")


def overlap_fingerprint(selection):
    """Keep legacy view hashes unchanged; segregate new heuristic artifacts."""
    pmi = views.is_pmi_selection(selection)
    configuration = dict(
        version=views.PMI_OVERLAP_VERSION if pmi else views.OVERLAP_VERSION, selection_strategy=STRATEGY,
        selection_modality=selection["selection_modality"], selected=selection["selected"],
        overlap_configuration=selection["overlap_configuration"],
        sample_ids=[dict(modality=row["modality"], id=row["id"]) for row in selection["overlap_samples"]])
    if pmi:
        configuration["calibration"] = views.pmi_configuration(selection.get(views.option_shuffle.FIELD), pool_mode=views.pool_mode(selection))
    configuration = views.add_pool_mode(views.option_shuffle.add_metadata(
        configuration, selection.get(views.option_shuffle.FIELD)), views.pool_mode(selection))
    return views.digest(views.examples.add_configuration(configuration, selection.get(views.examples.FIELD)))


def frequency_order(pool, counts, scores):
    return sorted(range(len(pool)), key=lambda i: (-counts[pool[i]["label"]], -scores[i], i))


def validate_selection(selection):
    if selection.get("selection_strategy") != STRATEGY:
        raise ValueError("Expected a sparse-overlap frequency selection file")
    if views.is_pmi_selection(selection):
        # PMI validates probabilities; overlap votes determine the final (possibly non-top-five) views.
        views.validate_pmi_selection(selection, check_selected=False)
        pool, scores = selection["pool"], selection["scores"]
        order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))
        expected = [dict(label=pool[i]["label"], aspect=pool[i]["aspect"], probability=scores[i]) for i in order]
        views.require_equal({"ranking": selection.get("probability_ranking")}, {"ranking": expected},
                            "PMI overlap probability ranking")
    configuration = selection.get("overlap_configuration", {})
    validate_configuration(configuration, pool_mode=views.pool_mode(selection))
    pool, scores = selection["pool"], selection.get("scores", [])
    counts = selection.get("acceptance_counts", {})
    if (len(scores) != len(pool) or any(not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in scores)
            or not math.isclose(sum(scores), 1, abs_tol=1e-6)
            or set(counts) != {row["label"] for row in pool}
            or any(type(value) is not int or value < 0 for value in counts.values())):
        raise ValueError("Invalid overlap probabilities or acceptance counts")
    samples = selection.get("overlap_samples", [])
    actual_counts = dict.fromkeys(counts, 0)
    if not samples:
        raise ValueError("Missing overlap sample diagnostics")
    for sample in samples:
        if sample.get("modality") not in views.enabled_modalities(selection["selection_modality"]):
            raise ValueError("Overlap sample modality differs from the selection mode")
        labels = sample.get("selected_labels", [])
        if len(labels) != len(set(labels)) or any(label not in counts for label in labels):
            raise ValueError("Invalid per-sample overlap selection")
        if sample.get("valid"):
            for label in labels:
                actual_counts[label] += 1
        elif labels:
            raise ValueError("Invalid overlap samples cannot contribute votes")
    if actual_counts != counts:
        raise ValueError("Overlap acceptance counts differ from sample diagnostics")
    count = configuration["selected_count"]
    eligible = [i for i in frequency_order(pool, counts, scores) if counts[pool[i]["label"]] > 0]
    expected = [pool[i]["label"] for i in eligible[:count]]
    if len(eligible) < count or [row.get("label") for row in selection.get("selected", [])] != expected:
        raise ValueError("Final overlap perspectives do not match the acceptance-frequency ranking")
    return count


def sample_items(items, count, seed, modality):
    if not items:
        raise ValueError(f"No {modality} samples available for overlap selection")
    # Separate namespace from adaptive_views.sampled_groups; not a disjoint split.
    return random.Random(f"adaptive-views-overlap/{seed}/{modality}").sample(items, min(count, len(items)))


def log_decision(perspective, decision, selected, token_union, *, prefix, threshold, aspects):
    def sparse_text(vector):
        ordered = dict(sorted(vector.items(), key=lambda item: (-item[1], item[0])))
        return json.dumps(ordered, ensure_ascii=False)

    ratio = ("N/A (seed)" if decision["seed"] else "N/A (empty candidate)" if not decision["vector"] else
             f"{decision['intersection']}/{decision['denominator']}={decision['overlap']:.6f}")
    outcome = "accepted" if decision["accepted"] else "rejected"
    reason = (" reason=empty_seed_vector" if decision["seed"] else " reason=empty_candidate_vector") if not decision["vector"] else ""
    selected_views = [dict(label=label, aspect=aspects[label]) for label in selected]
    print("\n".join([
        f"{prefix} view={perspective['label']} ({perspective['aspect']}) "
        f"probability={perspective['probability']:.8f} decision={outcome}{reason}",
        f"{prefix} sparse_tokens={sparse_text(decision['vector'])}",
        f"{prefix} overlap={ratio} denominator=candidate_unique_tokens threshold={threshold} (strict <)",
        f"{prefix} selected_views={json.dumps(selected_views, ensure_ascii=False)}",
        f"{prefix} token_union(size={len(token_union)})={sparse_text(token_union)}",
    ]), flush=True)


def greedy_select(ranking, vector_for, *, count, threshold, merge="sum", on_decision=None):
    """Accept candidates whose token fraction shared with the accepted union is below the threshold."""
    if merge not in {"sum", "max", "replace"}:
        raise ValueError(f"Unsupported sparse merge: {merge}")
    selected, tokens, tested = [], {}, []
    for perspective in ranking:
        vector = dict(vector_for(perspective))
        if any(not token or any(c.isspace() for c in token) or type(weight) is not int or weight <= 0
               for token, weight in vector.items()):
            raise ValueError("Overlap probes require valid sparse terms and positive integer weights")
        seed = not tested
        denominator = len(vector)
        intersection = len(tokens.keys() & vector.keys())
        overlap = intersection / denominator if denominator and not seed else None
        accepted = bool(vector) and (seed or overlap < threshold)
        tested.append(dict(label=perspective["label"], probability=perspective["probability"],
                           vector=vector, intersection=intersection, denominator=denominator,
                           overlap=overlap, accepted=accepted, seed=seed))
        if accepted:
            selected.append(perspective["label"])
            for token, weight in vector.items():
                if token not in tokens or merge == "replace":
                    tokens[token] = weight
                elif merge == "sum":
                    tokens[token] += weight
                else:
                    tokens[token] = max(tokens[token], weight)
        if on_decision is not None:
            # Observers receive snapshots; logging cannot mutate the saved trace.
            on_decision(dict(perspective), copy.deepcopy(tested[-1]), tuple(selected), dict(tokens))
        if seed and not vector:
            return dict(valid=False, reason="empty_seed_vector", selected_labels=[], tested=tested, token_union={})
        if len(selected) == count:
            break
    return dict(valid=True, selected_labels=selected, tested=tested, token_union=tokens)


class SparseViewProbe:
    """Single-view forward with the same masked EOL and sparse conversion as encode."""
    def __init__(self, model, processor, model_args, data, task, first_view):
        from adaptive_view_encoding import AdaptiveEncoder
        probe_data = copy.deepcopy(data)
        probe_data.sparse_length = probe_data.text_sparse_length = probe_data.image_sparse_length = PROBE_LENGTH
        self.encoder = AdaptiveEncoder(model, processor, model_args, probe_data, task, {"selected": [first_view]})

    def __call__(self, record, perspective):
        from adaptive_view_encoding import sparse_vectors
        from freeret_full_prompt import enabled
        if enabled(self.encoder.model_args):
            record = dict(record, _freeret_projection=True)
        self.encoder.selected = [dict(label=perspective["label"], aspect=perspective["aspect"])]
        logits, _ = self.encoder.encode([record])
        return sparse_vectors(self.encoder, [record], logits)[0]


class OverlapSelectionStrategy:
    """Hook for the shared encoder; ordinary adaptive entrypoints use no hook."""
    def __init__(self, options, model_args, data, task):
        self.model_args, self.data, self.task = model_args, data, task
        self.pool_mode = views.options_pool_mode(options)
        self.requested = {key: getattr(options, name) for key, name in OPTION_NAMES.items()}
        self.configuration = {key: DEFAULTS[key] if value is None else value for key, value in self.requested.items()}
        self.configuration.update(rule=RULE, probe_sparse_length=PROBE_LENGTH,
                                  probe_encoding={name: getattr(data, name) for name in PROBE_FIELDS})
        from freeret_prompt_constraints import configuration
        self.configuration = configuration(self.configuration, model_args)
        validate_configuration(self.configuration, pool_mode=self.pool_mode)

    def validate(self, selection, *, inherit=False):
        if views.pool_mode(selection) != self.pool_mode:
            raise ValueError("Overlap view pool mode mismatch")
        if not views.is_overlap_selection(selection) or selection.get("selection_strategy") != STRATEGY:
            raise ValueError("Overlap mode needs an overlap selection file; choose a new file to generate one")
        stored = selection["overlap_configuration"]
        if inherit:
            for key, requested in self.requested.items():
                if requested is None:
                    self.configuration[key] = stored[key]
        views.require_equal(stored, self.configuration, "Overlap selection configuration")

    def report(self, selection, *, reused=False):
        scores = {row["label"]: score for row, score in zip(selection["pool"], selection["scores"])}
        status = ("reused locked selection" if reused else
                  "incomplete selection" if selection.get("status") == "insufficient_views" else "final selection")
        calibration = "; calibration=pmi (corrected probabilities)" if views.is_pmi_selection(selection) else ""
        print(f"[adaptive overlap] {status}; modality={selection['selection_modality']}{calibration} "
              f"selected_count={len(selection['selected'])}", flush=True)
        for row in selection["selected"]:
            label = row["label"]
            print(f"[adaptive overlap] selected view={label} ({row['aspect']}) "
                  f"acceptance_count={selection['acceptance_counts'][label]} "
                  f"probability={scores[label]:.8f}", flush=True)
        selected = {row["label"] for row in selection["selected"]}
        order = frequency_order(selection["pool"], selection["acceptance_counts"], selection["scores"])
        lines = []
        for rank, index in enumerate(order, 1):
            row = selection["pool"][index]
            label = row["label"]
            lines.append(f"[adaptive overlap] acceptance rank={rank} view={label} ({row['aspect']}) "
                         f"acceptance_count={selection['acceptance_counts'][label]} probability={scores[label]:.8f} "
                         f"selected={str(label in selected).lower()}")
        print("\n".join(lines), flush=True)

    def choose(self, model, processor, identity, dataset, source_split, texts, images, options, *, report_path):
        selection = views.choose_views(model, processor, identity, dataset, source_split, texts, images, options)
        version = views.PMI_OVERLAP_VERSION if views.is_pmi_selection(selection) else views.OVERLAP_VERSION
        selection.pop("fingerprint")
        pool, scores = selection["pool"], selection["scores"]
        order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))
        ranking = [dict(label=pool[i]["label"], aspect=pool[i]["aspect"], probability=scores[i]) for i in order]
        selection.update(version=version, selection_strategy=STRATEGY,
                         overlap_configuration=copy.deepcopy(self.configuration), probability_ranking=ranking,
                         acceptance_counts={row["label"]: 0 for row in pool}, overlap_samples=[])
        selection.pop("views_fingerprint")
        selection["selected"] = []
        mode = selection["selection_modality"]
        active = views.enabled_modalities(mode)
        probe = SparseViewProbe(model, processor, self.model_args, self.data, self.task, ranking[0])
        count = self.configuration["selected_count"]
        aspects = {row["label"]: row["aspect"] for row in pool}
        for modality in active:
            items = sample_items(texts if modality == "text" else images,
                                 self.configuration["samples_per_modality"], self.configuration["seed"], modality)
            for index, record in enumerate(items):
                # Pass pure modalities even if a source record has extra fields.
                sample = {"id": record["id"], modality: record[modality]}
                prefix = (f"[adaptive overlap] {modality} sample={index + 1}/{len(items)} "
                          f"id={json.dumps(record['id'], ensure_ascii=False)}")
                print(f"{prefix} begin; target={count}", flush=True)
                def report_decision(perspective, decision, selected, token_union):
                    log_decision(perspective, decision, selected, token_union, prefix=prefix,
                                 threshold=self.configuration["threshold"], aspects=aspects)
                result = greedy_select(ranking, lambda perspective: probe(sample, perspective), count=count,
                                       threshold=self.configuration["threshold"], merge=self.data.sparse_value_type,
                                       on_decision=report_decision)
                result.update(id=record["id"], modality=modality)
                selection["overlap_samples"].append(result)
                if result["valid"]:
                    for label in result["selected_labels"]:
                        selection["acceptance_counts"][label] += 1
                print(f"{prefix} complete; valid={result['valid']} "
                      f"selected={result['selected_labels']} tested={len(result['tested'])}", flush=True)
                print(f"{prefix} acceptance_counts={json.dumps(selection['acceptance_counts'], ensure_ascii=False)}",
                      flush=True)
        counts = selection["acceptance_counts"]
        final_order = [i for i in frequency_order(pool, counts, scores) if counts[pool[i]["label"]] > 0]
        selection["selected"] = [dict(label=pool[i]["label"], aspect=pool[i]["aspect"]) for i in final_order[:count]]
        selection["overlap_sample_counts"] = {
            name: dict(total=sum(row["modality"] == name for row in selection["overlap_samples"]),
                       valid=sum(row["modality"] == name and row["valid"] for row in selection["overlap_samples"]))
            for name in active}
        if len(final_order) < count:
            selection["status"] = "insufficient_views"
            self.report(selection)
            selection["fingerprint"] = views.digest(selection)
            views.atomic_json(report_path, selection)
            if not any(row["valid"] for row in selection["overlap_samples"]):
                raise ValueError(f"All overlap samples produced empty seed sparse vectors. "
                                 f"Check model logits and sparse filtering. Diagnostics: {report_path}")
            raise ValueError(f"Only {len(final_order)} views received acceptance votes; requested {count}. "
                             f"Increase view_overlap_threshold and rerun. Diagnostics: {report_path}")
        selection["views_fingerprint"] = overlap_fingerprint(selection)
        selection["fingerprint"] = views.digest(selection)
        validate_selection(selection)
        self.report(selection)
        return selection
