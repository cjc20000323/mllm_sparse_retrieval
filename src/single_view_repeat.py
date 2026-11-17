"""Single whole-input view with sparse weights scaled by a locked selection's K."""
from __future__ import annotations

import copy
from pathlib import Path
import re

FLAG = "single_view_repeat_selected_count"
ENV = "SINGLE_VIEW_REPEAT_SELECTED_COUNT"
FIELD = "single_view_repeat"
FAMILIES = {"llava_llama", "llava_mistral"}


def enabled(options):
    value = getattr(options, FLAG, False)
    if isinstance(value, str):
        if value.lower() not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError(f"{ENV} must be true or false")
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def selection_configuration(selection):
    selected = selection.get("selected")
    if not isinstance(selected, list) or not selected:
        raise ValueError("Single-view repetition requires a nonempty locked selected list")
    count = len(selected)
    if "selected_count" in selection and (
            type(selection["selected_count"]) is not int or selection["selected_count"] != count):
        raise ValueError("Single-view repetition selected_count differs from len(selected)")
    if selection.get("pool") is not None and count > len(selection["pool"]):
        raise ValueError("Single-view repetition count exceeds the perspective pool")
    for key in ("fingerprint", "views_fingerprint"):
        if not isinstance(selection.get(key), str) or not selection[key]:
            raise ValueError(f"Single-view repetition requires a locked {key}")
    if selection.get("overlap_configuration") is not None or selection.get("selection_strategy") == "overlap":
        raise ValueError("Single-view repetition does not support overlap selections")
    return dict(version=1, count=count, count_source="selected",
                selection_fingerprint=selection["fingerprint"],
                views_fingerprint=selection["views_fingerprint"])


def validate(options, data, task, *, family=None):
    if not enabled(options):
        return
    if task != "ir" or data.dataset_name not in {"flickr", "coco"}:
        raise ValueError("Single-view repetition supports Flickr/COCO IR only")
    if family is not None and family not in FAMILIES:
        raise ValueError("Single-view repetition supports LLaVA-Llama3-8B and LLaVA-Mistral-7B only")
    if getattr(options, "eol_type", "disassembleeol_separate") != "disassembleeol_separate":
        raise ValueError("Single-view repetition requires the standard adaptive whole-input prompt")
    for source, keys in ((data, ("sparse_value_mean", "sparse_logits_multiply_view_count")),
                         (options, ("use_output_embedding_cluster", "use_stc_sparse", "use_lens_sparse",
                                    "use_freeret_full_prompt", "view_overlap_selection"))):
        for key in keys:
            if getattr(source, key, False):
                raise ValueError(f"Single-view repetition is incompatible with {key}")


def configuration(options, data, task, selection, *, family=None):
    if not enabled(options):
        return None
    validate(options, data, task, family=family)
    return selection_configuration(selection)


def multiplier(data):
    if getattr(data, "sparse_logits_multiply_view_count", False):
        raise ValueError("Selected-count repetition and the fixed x5 flag are mutually exclusive")
    count = getattr(data, "selected_view_count", None)
    if type(count) is not int or count < 1:
        raise ValueError("Single-view repetition requires a validated positive selected_view_count")
    return count


def validate_saved(configuration, selection):
    saved = configuration.get(FIELD)
    if saved is None:
        return
    if saved != selection_configuration(selection):
        raise ValueError("Single-view repetition metadata differs from the locked selection/count")
    if (configuration.get("task") != "ir" or configuration.get("dataset") not in {"flickr", "coco"}
            or configuration.get("model", {}).get("family") not in FAMILIES):
        raise ValueError("Single-view repetition manifest must describe LLaVA Flickr/COCO IR")


def mark_path(path, config):
    path = Path(path)
    if config is None:
        return path
    count = config.get("count")
    if type(count) is not int or count < 1:
        raise ValueError("Invalid single-view repetition path count")
    fingerprint = config.get("selection_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("Single-view repetition path requires its selection fingerprint")
    marker = f"_single_view_repeat_k{count}_{fingerprint[:12]}"
    existing = re.search(r"_single_view_repeat_k\d+_[^_]+", path.name)
    if existing:
        if existing.group() != marker:
            raise ValueError(f"Single-view repetition count/path mismatch: {path}")
        return path
    suffix = re.search(r"(_llava_base)?(_dense_freeret_pre_mlp(?:_freeret_(?:constraints|full_prompt)_v\d+)?)?$", path.name).group()
    name = path.name[:-len(suffix)] if suffix else path.name
    return path.with_name(name + marker + suffix)


def sparse_vectors(encoder, records, logits):
    """Preserve prompteol token selection and round(K * weight * 100)."""
    from encode import get_img_valid_tokens_values, get_text_valid_tokens_values
    model = copy.copy(encoder.model_args)
    model.eol_type, model.calculate_type = "prompteol", "separate"
    data = copy.copy(getattr(encoder, "sparse_data", encoder.data))
    setattr(data, FLAG, True)
    data.selected_view_count = selection_configuration(encoder.selection)["count"]
    vocabulary = {value: key for key, value in encoder.processor.tokenizer.get_vocab().items()}
    filtered_ids = None
    if data.num_expended_tokens > 0:
        from encode import get_filtered_ids
        filtered_ids = get_filtered_ids(encoder.processor.tokenizer)
    result = []
    for row, record in zip(logits, records):
        if len(row) != 1:
            raise ValueError("Single-view repetition expects exactly one vocabulary-logit vector per record")
        if record.get("image") is None:
            tokens, weights = get_text_valid_tokens_values(
                record.get("text", ""), encoder.processor.tokenizer, row[0], vocabulary,
                data, filtered_ids, model)
        else:
            tokens, weights = get_img_valid_tokens_values(
                encoder.processor.tokenizer, row[0], vocabulary, data, filtered_ids, model)
        vector = {}
        for token, weight in zip(tokens, weights):
            weight = int(weight)
            if not token or any(char.isspace() for char in token) or weight <= 0:
                continue
            if token not in vector or data.sparse_value_type == "replace":
                vector[token] = weight
            elif data.sparse_value_type == "sum":
                vector[token] += weight
            else:
                vector[token] = max(vector[token], weight)
        result.append(vector)
    return result
