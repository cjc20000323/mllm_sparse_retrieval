"""Optional FreeRet task context and constraints for representation prompts."""
from __future__ import annotations

import hashlib
import json
import re

from freeret_prompt_catalog import PROMPT_SPECS, PROMPT_VERSION

SUFFIX = "_freeret_constraints_v2"
DATASET_KEYS = {"coco": "coco_t2i", "flickr": "flickr_t2i", "CUHK-PEDES": "cuhk_pedes_t2i",
                "ICFG-PEDES": "icfg_pedes_t2i", "RSTPReid": "rstpreid_t2i", "webqa": "webqa_t2it",
                "remuq": "remuq_it2t", "llava": "llava_it2t"}


def plain_instruction(value):
    # These are textual references in FreeRet, not extra processor image tokens.
    return re.sub(r"<([^<>]+)>", r"\1", value).rstrip("\n")


CONFIGURATION = dict(version=2, source="freeret_task_instructions", catalog_version=PROMPT_VERSION,
    description_placement="shared_user_context_before_input_once_newline_v1",
    instruction_fingerprint=hashlib.sha256(json.dumps({key: {
        "task_description": spec.task_description.rstrip("\n") + "\n",
        "query": plain_instruction(spec.query_instruction), "target": plain_instruction(spec.target_instruction)}
        for key, spec in PROMPT_SPECS.items()}, sort_keys=True, ensure_ascii=False).encode()).hexdigest())
FINGERPRINT = hashlib.sha256(json.dumps(CONFIGURATION, sort_keys=True).encode()).hexdigest()


def enabled(args):
    from freeret_full_prompt import enabled as full_enabled
    if full_enabled(args):
        return False
    from dense_representation import boolean, enabled as dense_enabled
    use = boolean(getattr(args, "use_freeret_prompt_constraints", False))
    if use and not dense_enabled(args):
        raise ValueError("USE_FREERET_PROMPT_CONSTRAINTS=true requires USE_FREERET_DENSE=true")
    return use


def validate(args, dataset=None):
    from freeret_full_prompt import validate as validate_full
    validate_full(args, dataset)
    if enabled(args) and dataset is not None and dataset not in DATASET_KEYS:
        raise ValueError(f"No FreeRet prompt constraints registered for dataset {dataset!r}")


def configuration(value, args):
    from freeret_full_prompt import enabled as full_enabled, CONFIGURATION as full_configuration
    if full_enabled(args):
        return {**value, "freeret_full_prompt": dict(full_configuration)}
    return {**value, "freeret_prompt_constraints": dict(CONFIGURATION)} if enabled(args) else value


def active(args):
    from freeret_full_prompt import enabled as full_enabled
    return full_enabled(args) or enabled(args)


def validate_selection(selection, args):
    if "overlap_configuration" in selection:
        from freeret_full_prompt import enabled as full_enabled, CONFIGURATION as full_configuration
        expected_full = full_configuration if full_enabled(args) else None
        if selection["overlap_configuration"].get("freeret_full_prompt") != expected_full:
            raise ValueError("Overlap selection full FreeRet prompt mode differs; reselect views with matching prompt mode")
        expected = CONFIGURATION if enabled(args) else None
        if selection["overlap_configuration"].get("freeret_prompt_constraints") != expected:
            raise ValueError("Overlap selection prompt constraints differ; reselect views with matching prompt constraints")


def instruction(dataset, record=None, *, role=None, input_type=None):
    if dataset not in DATASET_KEYS:
        raise ValueError(f"No FreeRet prompt constraints registered for dataset {dataset!r}")
    record = record or {}
    role = role or record.get("_freeret_role")
    image = record.get("image") is not None or input_type == "image"
    if dataset in {"coco", "flickr", "CUHK-PEDES", "ICFG-PEDES", "RSTPReid"}:
        # The caption/image instructions coincide in both IR directions.
        role = "target" if image else "query"
    elif role is None:
        if input_type in {"query", "corpus", "target"}:
            role = input_type
        elif dataset == "webqa":
            role = "target" if image else "query"
        else:
            role = "query" if image else "target"
    role = "target" if role == "corpus" else role
    if role not in {"query", "target"}:
        raise ValueError(f"Unknown FreeRet representation role: {role!r}")
    spec = PROMPT_SPECS[DATASET_KEYS[dataset]]
    return plain_instruction(getattr(spec, role + "_instruction"))


def augment(value, args, dataset, record=None, *, role=None, input_type=None):
    if not enabled(args):
        return value
    prefix = instruction(dataset, record, role=role, input_type=input_type) + "\n"
    return value if value.startswith(prefix) else prefix + value


def task_description(dataset):
    if dataset not in DATASET_KEYS:
        raise ValueError(f"No FreeRet prompt constraints registered for dataset {dataset!r}")
    return PROMPT_SPECS[DATASET_KEYS[dataset]].task_description.rstrip("\n") + "\n"


def augment_context(value, args, dataset):
    """Add once to the common context, before image/text and isolated view blocks."""
    return task_description(dataset) + value if enabled(args) else value


def _description_template(value, description):
    # Called on project templates before user text substitution. View-only
    # content elements have no user header and must not repeat the description.
    if isinstance(value, str):
        pattern = r"\[INST\]|<\|start_header_id\|>user<\|end_header_id\|>\n\n|<\|im_start\|>user\n|USER: |<s>user\n\n"
        return re.sub(pattern, lambda match: match[0] + description, value, count=1)
    if isinstance(value, dict):
        if value.get("role") == "user":
            content = value["content"]
            if isinstance(content, str):
                content = description + content
            else:
                content = [{"type": "text", "text": description}, *content]
            return {**value, "content": content}
        return {key: _description_template(item, description) for key, item in value.items()}
    if isinstance(value, list):
        return [_description_template(item, description) for item in value]
    if isinstance(value, tuple):
        return tuple(_description_template(item, description) for item in value)
    return value


def _template(value, prefix):
    if isinstance(value, str):
        # Only project template strings reach here, before user text substitution.
        # Preserve every original character, including the final Summary space.
        return re.sub(r"(?<![A-Za-z])(?:Summary|Describe)(?=\s)", lambda match: prefix + match[0], value)
    if isinstance(value, list):
        return [_template(item, prefix) for item in value]
    if isinstance(value, tuple):
        return tuple(_template(item, prefix) for item in value)
    if isinstance(value, dict):
        return {key: _template(item, prefix) for key, item in value.items()}
    return value


class Templates:
    """Per-call view of legacy templates; never mutate module globals or processors."""
    def __init__(self, args, data, *, input_type=None):
        self.use = enabled(args)
        self.prefix = instruction(data.dataset_name, input_type=input_type) + "\n" if self.use else ""
        self.description = task_description(data.dataset_name) if self.use else ""

    def __getattr__(self, name):
        import template
        value = getattr(template, name)
        if not self.use:
            return value
        if name == "build_random_view_prompts":
            return lambda *args, **kwargs: _template(value(*args, **kwargs), self.prefix)
        return _description_template(_template(value, self.prefix), self.description)


def augment_template(value, args, data, *, input_type=None):
    if not enabled(args):
        return value
    return _description_template(_template(value, instruction(data.dataset_name, input_type=input_type) + "\n"),
                                 task_description(data.dataset_name))
