"""Sum single-token vocabulary variants of option letters before normalization."""
from __future__ import annotations


def configuration():
    return dict(method="vocabulary_variant_probability_sum", version=1,
                matching="decoded_strip_ascii_case", normalization="within_options",
                pmi_input="vocabulary_log_probability")


PREFIXES = {"llava_llama": "Ġ", "llava_mistral": "▁"}


def prefix_configuration():
    return dict(method="model_prefixed_single_token", version=1, prefixes=dict(PREFIXES),
                normalization="within_options", pmi_input="vocabulary_log_probability")


def prefix_for_family(family):
    if family not in PREFIXES:
        raise ValueError("USE_VIEW_PREFIX_TOKENS only supports llava_llama and llava_mistral")
    return PREFIXES[family]


def validate_prefix_tokens(mapping, labels, family):
    labels = list(labels)
    if not labels or len(set(labels)) != len(labels) or any(not isinstance(label, str) or len(label) != 1 or not "A" <= label <= "Z" for label in labels):
        raise ValueError("Prefixed option labels must be distinct uppercase ASCII letters")
    prefix = prefix_for_family(family)
    validate_variants(mapping, labels)
    for row in mapping:
        if (len(row["variants"]) != 1 or row["variants"][0]["token"] != prefix + row["label"]
                or row["variants"][0]["decoded"].strip() != row["label"]):
            raise ValueError(f"Expected exactly the prefixed uppercase token {prefix + row['label']!r}")


def prefixed_tokens(tokenizer, labels, family):
    labels = list(labels)
    if not labels or len(set(labels)) != len(labels) or any(not isinstance(label, str) or len(label) != 1 or not "A" <= label <= "Z" for label in labels):
        raise ValueError("Prefixed option labels must be distinct uppercase ASCII letters")
    prefix = prefix_for_family(family)
    vocabulary = tokenizer.get_vocab()
    special = set(tokenizer.all_special_ids)
    mapping = []
    for label in labels:
        token = prefix + label
        token_id = vocabulary.get(token)
        if type(token_id) is not int or token_id < 0 or token_id in special:
            raise ValueError(f"Missing or invalid prefixed option token {token!r} for {family}; no fallback is allowed")
        decoded = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        mapping.append(dict(label=label, variants=[dict(token_id=token_id, token=token, decoded=decoded)]))
    validate_prefix_tokens(mapping, labels, family)
    return mapping


def token_variants(tokenizer, labels):
    labels = list(labels)
    if not labels or len(set(labels)) != len(labels) or any(len(label) != 1 or not "A" <= label <= "Z" for label in labels):
        raise ValueError("Perspective labels must be distinct uppercase ASCII letters")
    grouped = {label: [] for label in labels}
    special = set(tokenizer.all_special_ids)
    # Aliases for the same vocabulary ID must not contribute probability twice.
    vocabulary = {}
    for token, token_id in sorted(tokenizer.get_vocab().items()):
        vocabulary.setdefault(int(token_id), token)
    for token_id, token in sorted(vocabulary.items()):
        if token_id in special:
            continue
        decoded = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        letter = decoded.strip()
        if len(letter) == 1 and ("A" <= letter <= "Z" or "a" <= letter <= "z") and letter.upper() in grouped:
            grouped[letter.upper()].append(dict(token_id=token_id, token=token, decoded=decoded))
    result = [dict(label=label, variants=grouped[label]) for label in labels]
    validate_variants(result, labels)
    return result


def validate_variants(mapping, labels):
    if (not isinstance(mapping, list) or not mapping or any(not isinstance(row, dict) for row in mapping)
            or [row.get("label") for row in mapping] != list(labels)):
        raise ValueError("Option-token variants differ from the displayed option labels")
    seen = set()
    for row in mapping:
        variants = row.get("variants")
        if not isinstance(variants, list) or not variants:
            raise ValueError(f"No vocabulary token variants for perspective {row['label']}")
        for variant in variants:
            token_id = variant.get("token_id")
            decoded = variant.get("decoded")
            if (type(token_id) is not int or token_id < 0 or token_id in seen
                    or not isinstance(variant.get("token"), str) or not isinstance(decoded, str)
                    or decoded.strip() not in {row["label"], row["label"].lower()}):
                raise ValueError(f"Invalid vocabulary token variant for perspective {row['label']}")
            seen.add(token_id)


def token_probability_details(logits, mapping, *, vocabulary_normalizer=None):
    """Describe only the tokens actually scored, without changing option scores."""
    import json
    import torch
    values = logits.double()
    entries = [(row["label"], entry) for row in mapping for entry in row["variants"]]
    ids = [entry["token_id"] for _, entry in entries]
    if (values.ndim != 1 or not ids or len(set(ids)) != len(ids)
            or min(ids) < 0 or max(ids) >= values.numel() or not torch.isfinite(values).all()):
        raise ValueError("Invalid logits/token mapping for token probability diagnostics")
    selected = values[ids]
    if vocabulary_normalizer is None:
        vocabulary_normalizer = torch.logsumexp(values, dim=0)
    vocabulary_logs = selected - vocabulary_normalizer
    candidate_logs = selected - torch.logsumexp(selected, dim=0)
    if not torch.isfinite(vocabulary_logs).all() or not torch.isfinite(candidate_logs).all():
        raise ValueError("Non-finite token probability diagnostics")
    vocabulary_p = vocabulary_logs.exp().cpu().tolist()
    candidate_p = candidate_logs.exp().cpu().tolist()
    vocabulary_logs, candidate_logs = vocabulary_logs.cpu().tolist(), candidate_logs.cpu().tolist()
    return [dict(display_label=label, token_id=entry["token_id"], token=entry["token"],
                 decoded_repr=json.dumps(entry["decoded"], ensure_ascii=True),
                 vocabulary_token_probability=vocabulary_p[i], candidate_token_probability=candidate_p[i],
                 vocabulary_token_log_probability=vocabulary_logs[i], candidate_token_log_probability=candidate_logs[i])
            for i, (label, entry) in enumerate(entries)]


def score_logits(logits, mapping, *, return_token_details=False):
    import torch
    if logits.ndim != 1 or not logits.numel() or not torch.isfinite(logits).all():
        raise ValueError("View selection requires finite vocabulary logits")
    groups = [[entry["token_id"] for entry in row["variants"]] for row in mapping]
    if any(not ids or min(ids) < 0 or max(ids) >= logits.numel() for ids in groups):
        raise ValueError("Option token variants fall outside the model output vocabulary")
    # Double-precision log masses retain tiny full-vocabulary probabilities and
    # avoid subtracting a large vocabulary normalizer before option softmax.
    values = logits.double()
    grouped_logits = torch.stack([torch.logsumexp(values[ids], dim=0) for ids in groups])
    vocabulary_normalizer = torch.logsumexp(values, dim=0)
    vocabulary_logs = grouped_logits - vocabulary_normalizer
    option_logs = grouped_logits.log_softmax(dim=0)
    if not torch.isfinite(vocabulary_logs).all() or not torch.isfinite(option_logs).all():
        raise ValueError("Non-finite vocabulary/option log probabilities")
    result = dict(probabilities=option_logs.exp().cpu().tolist(),
                log_probabilities=option_logs.cpu().tolist(),
                vocabulary_probabilities=vocabulary_logs.exp().cpu().tolist(),
                vocabulary_log_probabilities=vocabulary_logs.cpu().tolist())
    if return_token_details:
        result["token_probabilities"] = token_probability_details(values, mapping, vocabulary_normalizer=vocabulary_normalizer)
    return result


def validate_probabilities(probabilities, vocabulary_probabilities, vocabulary_logs, count, *, label):
    import numpy as np
    option = np.asarray(probabilities, dtype="float64")
    mass = np.asarray(vocabulary_probabilities, dtype="float64")
    logs = np.asarray(vocabulary_logs, dtype="float64")
    if (any(value.shape != (count,) or not np.isfinite(value).all() for value in (option, mass, logs))
            or (option < 0).any() or (mass < 0).any() or (logs > 1e-8).any()
            or not np.isclose(option.sum(), 1, atol=1e-6) or mass.sum() > 1 + 1e-6
            or not np.allclose(mass, np.exp(logs), atol=0, rtol=1e-6)):
        raise ValueError(f"Invalid {label} vocabulary/option probabilities")
    weights = np.exp(logs - logs.max())
    if not np.allclose(option, weights / weights.sum(), atol=1e-7, rtol=1e-6):
        raise ValueError(f"Inconsistent {label} vocabulary/option normalization")
