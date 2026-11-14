"""TopK-only token probability sidecars, independent of selection fingerprints."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path


VERSION = 1
CACHE_DIRECTORY = "token_probabilities_v1"
FILENAME = "view_token_probabilities.json"
CSV_COLUMNS = (
    "phase", "modality", "round", "canonical_label", "display_label", "aspect",
    "token_id", "token", "decoded_repr", "vocabulary_token_probability",
    "candidate_token_probability", "vocabulary_token_log_probability",
    "candidate_token_log_probability",
)
_RESELECT = "Clear VIEW_SELECTION_FILE and rerun TopK to reselect views with token diagnostics."


def _fail(message):
    raise ValueError("Invalid token probability diagnostics: " + message)


def _logsumexp(values):
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))


def _close(actual, expected, label, *, logs=False, legacy=False):
    # The original single-token selector stores float32 log_softmax results.
    relative = 1e-6 if legacy else 1e-7
    absolute = 1e-6 if logs and legacy else (1e-7 if logs else 1e-9 if legacy else 0)
    if not math.isclose(actual, expected, rel_tol=relative, abs_tol=absolute):
        _fail(f"{label} differs: actual={actual!r}, expected={expected!r}")


def _record(record):
    if not isinstance(record, dict) or set(record) != set(CSV_COLUMNS):
        _fail("unexpected token record fields")
    phase, modality, index = record["phase"], record["modality"], record["round"]
    if phase == "prior":
        if modality != "prior" or type(index) is not int or index != 0:
            _fail("prior records require modality=prior and round=0")
    elif phase == "selection":
        if modality not in ("text", "image") or type(index) is not int or index < 1:
            _fail("selection records require text/image and a 1-based round")
    else:
        _fail("unknown phase")
    for field in ("canonical_label", "display_label"):
        value = record[field]
        if not isinstance(value, str) or len(value) != 1 or not "A" <= value <= "Z":
            _fail(f"invalid {field}")
    if (not isinstance(record["aspect"], str) or not record["aspect"]
            or type(record["token_id"]) is not int or record["token_id"] < 0
            or not isinstance(record["token"], str) or not record["token"]):
        _fail("invalid token identity/aspect")
    try:
        decoded = json.loads(record["decoded_repr"])
    except (ValueError, TypeError):
        _fail("decoded_repr must be an escaped JSON string")
    if not isinstance(decoded, str) or json.dumps(decoded, ensure_ascii=True) != record["decoded_repr"]:
        _fail("decoded_repr must use json.dumps(decoded, ensure_ascii=True)")
    if decoded.strip() not in (record["display_label"], record["display_label"].lower()):
        _fail("decoded token does not match its displayed option")
    for kind in ("vocabulary", "candidate"):
        probability, log_probability = (record[f"{kind}_token_probability"], record[f"{kind}_token_log_probability"])
        if (isinstance(probability, bool) or not isinstance(probability, (int, float))
                or not math.isfinite(probability) or not 0 <= probability <= 1
                or isinstance(log_probability, bool) or not isinstance(log_probability, (int, float))
                or not math.isfinite(log_probability) or log_probability > 1e-7):
            _fail(f"invalid {kind} token probability/log probability")
        # exp may legitimately underflow to zero, but finite log values are retained.
        expected = math.exp(log_probability)
        if not math.isclose(probability, expected, rel_tol=1e-6, abs_tol=0):
            _fail(f"{kind} token probability differs from its log probability")
    return record


def log_records(records, cached=False):
    """Print unambiguous token identities without hiding very small probabilities."""
    lines = []
    for record in records:
        _record(record)
        lines.append(
            "[adaptive token] " + ("cached=true " if cached else "")
            + f"phase={record['phase']} modality={record['modality']} round={record['round']} "
            + f"view={record['canonical_label']} display={record['display_label']} "
            + f"token_id={record['token_id']} token={json.dumps(record['token'], ensure_ascii=True)} "
            + f"decoded_repr={record['decoded_repr']} "
            + f"vocabulary_token_probability={record['vocabulary_token_probability']:.12e} "
            + f"candidate_token_probability={record['candidate_token_probability']:.12e} "
            + f"vocabulary_token_log_probability={record['vocabulary_token_log_probability']:.12e} "
            + f"candidate_token_log_probability={record['candidate_token_log_probability']:.12e}"
        )
    if lines:
        print("\n".join(lines), flush=True)


def _vector(value, count, label, *, logs=False):
    if (not isinstance(value, list) or len(value) != count
            or any(isinstance(item, bool) or not isinstance(item, (int, float))
                   or not math.isfinite(item) or (item > 1e-7 if logs else not 0 <= item <= 1)
                   for item in value)):
        _fail(f"invalid selection {label}")
    return value


def validate(payload, selection):
    """Validate sidecar identity, every round and each token's aggregate contribution."""
    import adaptive_views as views

    if not isinstance(payload, dict) or set(payload) != {"version", "selection_fingerprint", "records", "fingerprint"}:
        _fail("unexpected sidecar fields")
    if payload["version"] != VERSION or type(payload["version"]) is not int:
        _fail("unsupported sidecar version")
    body = {key: value for key, value in payload.items() if key != "fingerprint"}
    try:
        fingerprint = views.digest(body)
    except (TypeError, ValueError):
        _fail("sidecar contains invalid JSON values")
    if payload["fingerprint"] != fingerprint:
        _fail("sidecar fingerprint mismatch")
    if payload["selection_fingerprint"] != selection.get("fingerprint") or not isinstance(selection.get("fingerprint"), str):
        _fail("selection fingerprint mismatch")
    if not isinstance(payload["records"], list) or not payload["records"]:
        _fail("missing token records")

    pool = selection["pool"]
    labels = [row["label"] for row in pool]
    indices = {label: index for index, label in enumerate(labels)}
    mapping = selection.get(views.option_shuffle.FIELD)
    views.option_shuffle.validate(mapping, pool)
    display = {row["canonical_label"]: row["display_label"] for row in mapping["mapping"]} if mapping else {label: label for label in labels}
    aspects = {row["label"]: row["aspect"] for row in pool}
    variant = views.uses_choice_scoring(selection)
    variants = None
    if variant:
        variants = selection.get("option_token_variants")
        views.option_scoring.validate_variants(variants, [row["label"] for row in views.option_shuffle.display_pool(pool, mapping)])
        if views.uses_prefix_tokens(selection):
            views.option_scoring.validate_prefix_tokens(variants, [row["label"] for row in views.option_shuffle.display_pool(pool, mapping)], selection["model"]["family"])
        variants = {row["label"]: row["variants"] for row in variants}

    pmi = views.is_pmi_selection(selection)
    expected = {("selection", modality, index + 1): row
                for modality in views.enabled_modalities(selection["selection_modality"])
                for index, row in enumerate(selection["rounds"])}
    if pmi:
        expected[("prior", "prior", 0)] = selection["calibration"]
    groups = {}
    for record in payload["records"]:
        _record(record)
        key = record["phase"], record["modality"], record["round"]
        if key not in expected:
            _fail(f"unexpected phase/modality/round {key}")
        label = record["canonical_label"]
        if label not in indices or record["display_label"] != display[label] or record["aspect"] != aspects[label]:
            _fail("canonical/display option mapping mismatch")
        groups.setdefault(key, {}).setdefault(label, []).append(record)
    if set(groups) != set(expected):
        _fail("missing selection rounds or PMI prior")

    legacy_identities = None
    for key, source in expected.items():
        grouped = groups[key]
        if set(grouped) != set(labels):
            _fail(f"missing options in {key}")
        identities, seen, vocabulary_logs, candidate_logs = {}, set(), [], []
        for label in labels:
            records = grouped[label]
            actual = {row["token_id"]: (row["token"], row["decoded_repr"]) for row in records}
            if len(actual) != len(records) or seen.intersection(actual):
                _fail(f"duplicate token IDs in {key}")
            seen.update(actual)
            if variant:
                target = {entry["token_id"]: (entry["token"], json.dumps(entry["decoded"], ensure_ascii=True))
                          for entry in variants[display[label]]}
                if actual != target:
                    _fail(f"token variants differ from the saved mapping for {label}")
            elif len(records) != 1:
                _fail("legacy scoring requires exactly one token per option")
            identities[label] = actual
            vocabulary_logs.append(_logsumexp([row["vocabulary_token_log_probability"] for row in records]))
            candidate_logs.append(_logsumexp([row["candidate_token_log_probability"] for row in records]))
        if not variant:
            if legacy_identities is not None and identities != legacy_identities:
                _fail("legacy token identity differs between rounds/prior")
            legacy_identities = identities

        normalizer = _logsumexp(vocabulary_logs)
        if normalizer > 1e-7:
            _fail("traced tokens exceed the full vocabulary probability mass")
        _close(_logsumexp(candidate_logs), 0, "candidate token probability sum", logs=True, legacy=not variant)
        for index in range(len(labels)):
            _close(candidate_logs[index], vocabulary_logs[index] - normalizer,
                   "candidate token normalization", logs=True, legacy=not variant)
        for records in grouped.values():
            for record in records:
                _close(record["candidate_token_log_probability"], record["vocabulary_token_log_probability"] - normalizer,
                       "individual candidate token normalization", logs=True, legacy=not variant)

        prior = key[0] == "prior"
        prefix = "prior" if prior else key[1]
        candidate_key = "prior_probabilities" if prior else prefix + ("_option_probabilities" if variant else "_raw_probabilities" if pmi else "_probabilities")
        candidate = _vector(source.get(candidate_key), len(pool), candidate_key)
        candidate_log_key = "prior_log_probabilities" if prior else None
        target_candidate_logs = (_vector(source.get(candidate_log_key), len(pool), candidate_log_key, logs=True)
                                 if prior else None)
        if pmi and not prior and not variant:
            scores = source.get(prefix + "_pmi_scores")
            baseline = selection["calibration"].get("prior_log_probabilities")
            _vector(baseline, len(pool), "prior_log_probabilities", logs=True)
            if (not isinstance(scores, list) or len(scores) != len(pool)
                    or any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for value in scores)):
                _fail("invalid selection PMI scores")
            target_candidate_logs = [score + prior_log for score, prior_log in zip(scores, baseline)]
        for index, value in enumerate(candidate):
            _close(math.exp(candidate_logs[index]), value, f"{key} option {labels[index]} probability", legacy=not variant)
            if target_candidate_logs is not None:
                _close(candidate_logs[index], target_candidate_logs[index], f"{key} option log probability", logs=True, legacy=not variant)
        if variant:
            values = _vector(source.get(prefix + "_vocabulary_probabilities"), len(pool), prefix + "_vocabulary_probabilities")
            logs = _vector(source.get(prefix + "_vocabulary_log_probabilities"), len(pool), prefix + "_vocabulary_log_probabilities", logs=True)
            for index, (value, log_value) in enumerate(zip(values, logs)):
                _close(vocabulary_logs[index], log_value, f"{key} vocabulary token log sum", logs=True)
                _close(math.exp(vocabulary_logs[index]), value, f"{key} vocabulary token probability sum")
    return payload


class Collector:
    """Collect forward-pass diagnostics without altering the selection object."""
    def __init__(self):
        self.records = []

    def __call__(self, records):
        if not isinstance(records, list) or not records:
            _fail("callback requires a nonempty list of token records")
        snapshot = copy.deepcopy(records)
        log_records(snapshot)
        if len({(row["phase"], row["modality"], row["round"]) for row in snapshot}) != 1:
            _fail("one callback must describe exactly one round or PMI prior")
        self.records.extend(snapshot)

    def finish(self, selection):
        import adaptive_views as views
        payload = dict(version=VERSION, selection_fingerprint=selection["fingerprint"], records=copy.deepcopy(self.records))
        payload["fingerprint"] = views.digest(payload)
        return validate(payload, selection)


def load(selection_path, selection):
    """A cached selection is usable only with its matching original sidecar."""
    path = Path(selection_path).expanduser().with_name(FILENAME)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return validate(payload, selection)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Cannot reuse token probability diagnostics at {path}: {exc}. {_RESELECT}") from exc
