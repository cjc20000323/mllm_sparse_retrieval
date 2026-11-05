"""Context-budget packing for dataset-level perspective-selection examples."""
from __future__ import annotations

import random

FIELD = "example_sampling"


def validate_options(options):
    mode = getattr(options, "view_example_mode", "auto")
    maximum = getattr(options, "view_example_max_tokens", 0)
    if mode not in {"auto", "fixed"}:
        raise ValueError("view_example_mode must be auto or fixed")
    if type(maximum) is not int or maximum < 0:
        raise ValueError("view_example_max_tokens must be a nonnegative integer")
    return mode


def context_limit(config):
    for source in (getattr(config, "text_config", None), config):
        value = getattr(source, "max_position_embeddings", None)
        if type(value) is int and value > 0:
            return value
    raise ValueError("Cannot determine the model context limit: max_position_embeddings must be a positive integer")


def configuration(options, config):
    if validate_options(options) == "fixed":
        return None
    limit = context_limit(config)
    requested = getattr(options, "view_example_max_tokens", 0)
    effective = min(requested, limit) if requested else limit
    if effective < 2:
        raise ValueError("View-example budget must leave room for the input and one answer token")
    return dict(version=1, mode="auto", max_tokens=requested, context_limit=limit,
                effective_max_tokens=effective, answer_tokens=1, input_limit=effective - 1)


def validate_configuration(value):
    if value is None:
        return
    if not isinstance(value, dict) or value.get("version") != 1 or value.get("mode") != "auto":
        raise ValueError("Unsupported automatic view-example sampling configuration")
    from types import SimpleNamespace
    try:
        expected = configuration(SimpleNamespace(view_example_mode="auto", view_example_max_tokens=value["max_tokens"]),
                                 SimpleNamespace(max_position_embeddings=value["context_limit"]))
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid automatic view-example token budget") from exc
    if value != expected:
        raise ValueError("Inconsistent automatic view-example token budget")


def add_configuration(value, policy):
    return {**value, FIELD: policy} if policy is not None else value


def validate_selection(selection, *, mode=None, maximum=None):
    policy = selection.get(FIELD)
    validate_configuration(policy)
    stored_mode = "auto" if policy is not None else "fixed"
    if mode is not None and mode != stored_mode:
        raise ValueError(f"View-example mode mismatch: stored={stored_mode}, requested={mode}; "
                         "use VIEW_EXAMPLE_MODE=fixed for legacy selections or choose a new selection file")
    if maximum is not None and policy is not None and maximum != policy["max_tokens"]:
        raise ValueError(f"View-example budget mismatch: stored={policy['max_tokens']}, requested={maximum}")
    sampling_policy = selection.get("sampling", {}).get(FIELD)
    if sampling_policy != policy:
        raise ValueError("Selection sampling/example policy mismatch")
    if policy is not None:
        for row in selection.get("rounds", []):
            for modality in ("text", "image"):
                tokens = row.get(modality + "_tokens")
                if tokens is None:
                    continue
                details = row.get(modality + "_examples", {})
                if (type(tokens) is not int or not 0 < tokens <= policy["input_limit"]
                        or not row.get(modality + "_ids")
                        or details.get("sample_count") != len(row[modality + "_ids"])
                        or details.get("input_tokens") != tokens
                        or details.get("input_limit") != policy["input_limit"]):
                    raise ValueError("Invalid automatic view-example round diagnostics")
                next_tokens = details.get("next_input_tokens")
                reason = details.get("stop_reason")
                if (reason not in {"context_limit", "samples_exhausted"}
                        or (reason == "context_limit" and (type(next_tokens) is not int or next_tokens <= policy["input_limit"]))
                        or (reason == "samples_exhausted" and next_tokens is not None)):
                    raise ValueError("Invalid automatic view-example stopping condition")
    return policy


def configuration_for_path(options, model_path):
    if validate_options(options) == "fixed":
        return None
    from pathlib import Path
    if not (Path(model_path) / "config.json").is_file():
        raise ValueError(f"Automatic view examples require the local checkpoint config.json: {model_path}")
    from transformers import AutoConfig
    return configuration(options, AutoConfig.from_pretrained(model_path, local_files_only=True))


def ordered_groups(items, rounds, seed, modality):
    if not items:
        raise ValueError(f"No {modality} samples available for adaptive view selection")
    for index in range(rounds):
        generator = random.Random(f"adaptive-views/{seed}/{modality}/{index}")
        yield generator.sample(items, len(items))


def pack(items, prepare, policy, modality):
    """Find a fitting deterministic prefix without running the language model."""
    validate_configuration(policy)
    limit = policy["input_limit"]
    base = prepare([])
    if base["tokens"] > limit:
        raise ValueError(f"View-selection {modality} base prompt has {base['tokens']} tokens, input budget={limit}")
    if not items:
        raise ValueError(f"No {modality} samples available for adaptive view selection")
    lengths = {}
    lower, upper, best = 0, None, None
    count = 1
    while True:
        prepared = prepare(items[:count])
        lengths[count] = prepared["tokens"]
        if prepared["tokens"] > limit:
            if count == 1:
                raise ValueError(f"First {modality} example id={items[0]['id']!r} needs {prepared['tokens']} tokens, "
                                 f"input budget={limit}; examples are never truncated or skipped")
            upper = count
            del prepared
            break
        lower, best = count, prepared
        if count == len(items):
            break
        count = min(count * 2, len(items))
    if upper is not None:
        while upper - lower > 1:
            middle = (lower + upper) // 2
            prepared = prepare(items[:middle])
            lengths[middle] = prepared["tokens"]
            if prepared["tokens"] <= limit:
                lower, best = middle, prepared
            else:
                upper = middle
            del prepared
    details = dict(sample_count=lower, input_tokens=best["tokens"], input_limit=limit,
                   effective_max_tokens=policy["effective_max_tokens"], context_limit=policy["context_limit"],
                   stop_reason="samples_exhausted" if lower == len(items) else "context_limit",
                   next_sample_id=items[lower]["id"] if lower < len(items) else None,
                   next_input_tokens=lengths[lower + 1] if lower < len(items) else None)
    return items[:lower], best, details


def log_round(details, *, modality, index, rounds):
    print(f"[adaptive examples] {modality} round={index + 1}/{rounds} mode=auto "
          f"samples={details['sample_count']} tokens={details['input_tokens']} "
          f"input_limit={details['input_limit']} context_limit={details['context_limit']} "
          f"stop={details['stop_reason']} next_tokens={details['next_input_tokens']} "
          f"next_sample_id={details['next_sample_id']!r}", flush=True)
