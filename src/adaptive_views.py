"""Dataset-level closed-pool view selection. No model service or DSPy is used."""
from __future__ import annotations

import freeret_full_prompt as full_prompt

import argparse
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import re
from typing import Optional
from types import SimpleNamespace
from uuid import uuid4

import adaptive_view_shuffle as option_shuffle
import adaptive_view_examples as examples

VERSION = 12
OVERLAP_VERSION = 16
PMI_VERSION = 14
PMI_OVERLAP_VERSION = 17
TOP_K = 5
SELECTION_MODALITIES = ("text", "image", "both")
MANIFEST = "adaptive_views_manifest.json"
POOL_MODE_FIELD = "view_pool_mode"
HIERARCHICAL_POOL = "hierarchical_v1"
# Primary dimensions and their secondary dimensions, in the supplied table order.
HIERARCHICAL_DIMENSIONS = (
    ("scene", "location, indoor or outdoor setting and terrain"),
    ("natural environment", "weather, lighting, season and time"),
    ("people", "number of people, appearance and pose"),
    ("clothing", "category, color, material and style"),
    ("objects", "category, number, appearance and state"),
    ("actions", "behavior, participants, tools and targets"),
    ("relations", "spatial relations, interactions and ownership"),
    ("visual attributes", "color, shape, size and material"),
    ("spatial layout", "foreground and background, distance and direction"),
    ("high-level semantics", "events, activities and functions"),
)
SOURCES = (
    "retrieval_disassemble_text_prompts_for_concat",
    "retrieval_disassemble_image_prompts_for_concat",
    "retrieval_disassemble_text_origin_prompts_person_retrieval_for_concat",
    "retrieval_disassemble_image_origin_prompts_person_retrieval_for_concat",
    "retrieval_disassemble_text_prompts_person_retrieval_for_concat",
    "retrieval_disassemble_image_prompts_person_retrieval_for_concat",
    "retrieval_disassemble_query_prompts_t2it_retrieval_for_concat",
    "retrieval_disassemble_corpus_prompts_t2it_retrieval_for_concat",
    "retrieval_disassemble_query_prompts_it2t_retrieval_for_concat",
    "retrieval_disassemble_corpus_prompts_it2t_retrieval_for_concat",
    "retrieval_disassemble_query_prompts_llava_it2t_retrieval_for_concat",
    "retrieval_disassemble_corpus_prompts_llava_it2t_retrieval_for_concat",
)
# In the original source catalog, append only these individual prompts after
# A-R; paired text/image descriptions deduplicate to the old S/T labels.
EXTRA_VIEW_SOURCES = (
    ("retrieval_disassemble_text_prompts_6_for_concat", 5),
    ("retrieval_disassemble_image_prompts_6_for_concat", 5),
    ("retrieval_disassemble_text_prompts_for_concat_mistral_generation", 6),
    ("retrieval_disassemble_image_prompts_for_concat_mistral_generation", 6),
)
# Freeze the version-2 source labels so edits to template.py cannot silently
# attach a perspective to a different merge group. Baseline templates stay intact.
SOURCE_ASPECTS = (
    "people or objects",
    "relations, such as belongings or spatial position, between main people or objects",
    "environment, weather or places",
    "actions or movements of main people or objects",
    "appearance, such as color, material, decoration and so on, of main people or objects",
    "gender", "wearing of people or objects", "gender of person",
    "actions or movements of person", "objects", "wearing of person",
    "appearance and decoration details of person, such as color, pattern and so on,",
    "events and actions",
    "environment and geographical locations, such as architecture, buildings or structures,",
    "time",
    "appearance, such as color, material, decoration, shape and so on, of main people or objects",
    "main person or objects", "environment, geographical locations, weather or places",
    "reason why main people or objects might be in this position and doing this thing",
    "animals and plants",
)
# Keep the frozen source catalog for validation after retiring the independent
# objects candidate. Reference overrides do not add aliases to the candidate pool.
RETIRED_SOURCE_ASPECTS = frozenset({"objects"})
TASK_VIEW_REFERENCE_OVERRIDES = {
    f"retrieval_disassemble_{kind}_prompts_person_retrieval_for_concat": {"objects": "A"}
    for kind in ("text", "image")
}
# (new label, canonical descriptor, version-2 source labels).
# C merges the former C/N environment options; M keeps the architectural descriptor.
# J and N are manually defined perspectives, with no historical aliases or sources.
VIEW_GROUPS = (
    ("A", "main people or objects", "AQ"),
    ("B", "relations, such as belongings or spatial position, between main people or objects", "B"),
    ("C", "environment, geographical locations, weather or places", "CR"),
    ("D", "actions or movements of main people or objects", "DI"),
    ("E", "appearance of main people or objects, including color, material, decoration, pattern and shape", "ELP"),
    ("F", "gender of people", "FH"),
    ("G", "wearing of people or objects, including clothing and accessories", "GK"),
    ("H", "time", "O"),
    ("I", "events and actions", "M"),
    ("J", "written words and symbols", ""),
    ("K", "reason why main people or objects might be in this position and doing this thing", "S"),
    ("L", "animals and plants", "T"),
    ("M", "environment and geographical locations, such as architecture, buildings or structures", "N"),
    ("N", "emotions and facial expressions", ""),
)
# (label, model-visible descriptor, exact source descriptor, paired source-name
# pattern, zero-based prompt index). Source wording is kept for validation and
# provenance even when the candidate descriptor has been shortened.
ADDITIONAL_VIEWS = (
    ("O", "group", "group", "retrieval_disassemble_{}_prompts_for_concat_llama_generation", 6),
    ("P", "age and ethnicity", "gender, age and ethnicity", "retrieval_disassemble_{}_prompts_for_concat_mistral_generation", 0),
    ("Q", "poses", "actions, movements and poses", "retrieval_disassemble_{}_prompts_for_concat_mistral_generation", 1),
    ("R", "tools and equipment", "objects, tools, clothing and accessories", "retrieval_disassemble_{}_prompts_for_concat_mistral_generation", 2),
    ("S", "professions and activities", "professions and activities", "retrieval_disassemble_{}_prompts_for_concat_mistral_generation", 5),
    ("T", "body type, height of person", "body type, height of person", "retrieval_disassemble_{}_prompts_person_retrieval_for_concat_mistral_generation", 3),
)
CHOICE_INSTRUCTION = (
    "Select a semantic perspective for cross-modal retrieval. The following are "
    "unordered examples from one dataset, all in the same modality. Which perspective "
    "best captures distinctive information useful for representing and matching these "
    "examples? Choose from the following closed list. Answer with exactly one capital "
    "option letter, without explanation.\n"
)


@dataclass
class AdaptiveViewArguments:
    adaptive_views: bool = False
    view_selection_hierarchical: bool = field(default=False, metadata={
        "help": "Use the ten primary/secondary dimension perspectives (A-J) instead of the original A-T pool"})
    view_selection_shuffle: bool = False
    view_selection_shuffle_run_id: Optional[str] = field(default=None, metadata={
        "help": "Internal batch run identifier; omit for a fresh standalone shuffled experiment"})
    view_selection_file: Optional[str] = None
    view_selection_modality: Optional[str] = field(default=None, metadata={
        "choices": SELECTION_MODALITIES,
        "help": "text/image/both; new selections default to both, existing files inherit their mode when omitted",
    })
    view_selection_pmi: Optional[bool] = field(default=None, metadata={
        "help": "PMI prior correction; new selections default false, omitted values inherit a reused file",
    })
    view_selection_rounds: int = 20
    view_example_mode: str = field(default="auto", metadata={"help": "auto fills the model context budget; fixed uses per-round counts"})
    view_example_max_tokens: int = field(default=0, metadata={"help": "Total example-prompt budget including one answer token; 0 uses the model context limit"})
    view_texts_per_round: int = field(default=8, metadata={"help": "Text examples per round in fixed mode only"})
    view_images_per_round: int = field(default=2, metadata={"help": "Image examples per round in fixed mode only"})
    view_selection_seed: int = 42
    view_selection_split: Optional[str] = None
    adaptive_data_root: str = "/root/autodl-tmp"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the temporary basename short for deeply nested Windows artifact paths.
    temporary = path.with_name(f".av-{uuid4().hex[:12]}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _extract_aspect(source, prompt):
    match = re.fullmatch(r"Summary the (.+) in above (?:image and sentence|sentence|image) in one word:\s*", prompt)
    if not match:
        raise ValueError(f"Cannot extract a perspective from {source}: {prompt!r}")
    return " ".join(match[1].split())


def _source_view_pool():
    """Extract the original 20 descriptors and their exact template origins."""
    import template
    rows, by_key = [], {}

    def add_prompt(source, prompt):
        aspect = _extract_aspect(source, prompt)
        key = aspect.casefold()
        if key not in by_key:
            if len(rows) == 26:
                raise ValueError("The perspective pool exceeds single-letter A-Z labels")
            entry = dict(label=chr(65 + len(rows)), aspect=aspect, sources=[])
            rows.append(entry)
            by_key[key] = entry
        if source not in by_key[key]["sources"]:
            by_key[key]["sources"].append(source)

    for source in SOURCES:
        prompts = getattr(template, source)
        if len(prompts) != 5:
            raise ValueError(f"Expected five default views in {source}")
        for prompt in prompts:
            add_prompt(source, prompt)
    for source, index in EXTRA_VIEW_SOURCES:
        add_prompt(source, getattr(template, source)[index])
    if len(rows) != 20:
        raise ValueError(f"Expected 20 source perspectives (A-T), got {len(rows)}; review the adaptive merge groups")
    if tuple(row["aspect"] for row in rows) != SOURCE_ASPECTS:
        raise ValueError("Source perspective descriptions/order changed; review the adaptive merge groups")
    return rows


def resolve_pool_mode(mode=None):
    if mode is not None and mode != HIERARCHICAL_POOL:
        raise ValueError(f"Unknown adaptive view_pool_mode: {mode!r}")
    return mode


def pool_mode(selection):
    return resolve_pool_mode(selection.get(POOL_MODE_FIELD))


def options_pool_mode(options):
    from dense_representation import boolean
    return HIERARCHICAL_POOL if boolean(getattr(options, "view_selection_hierarchical", False)) else None


def add_pool_mode(value, mode):
    mode = resolve_pool_mode(mode)
    return {**value, POOL_MODE_FIELD: mode} if mode is not None else value


def view_pool(mode=None):
    """Build A-T, keeping canonical wording separate from historical source aliases."""
    if resolve_pool_mode(mode) == HIERARCHICAL_POOL:
        return [dict(label=chr(65 + index), aspect=f"{primary}, such as {secondary}", sources=[], aliases=[])
                for index, (primary, secondary) in enumerate(HIERARCHICAL_DIMENSIONS)]
    import template
    originals = {row["label"]: row for row in _source_view_pool()}
    active = {label for label, row in originals.items() if row["aspect"] not in RETIRED_SOURCE_ASPECTS}
    members = "".join(labels for _, _, labels in VIEW_GROUPS)
    if len(members) != len(active) or set(members) != active:
        raise ValueError("Every retained source perspective must belong to exactly one adaptive merge group")
    rows = []
    for label, aspect, labels in VIEW_GROUPS:
        aliases = [originals[old] for old in labels]
        sources = list(dict.fromkeys(source for alias in aliases for source in alias["sources"]))
        rows.append(dict(label=label, aspect=aspect, sources=sources, aliases=aliases))
    seen = {row["aspect"].casefold() for row in rows}
    seen.update(alias["aspect"].casefold() for row in rows for alias in row["aliases"])
    for label, aspect, source_aspect, pattern, index in ADDITIONAL_VIEWS:
        if aspect.casefold() in seen:
            raise ValueError(f"Additional perspective duplicates an existing descriptor: {aspect}")
        sources = [pattern.format(kind) for kind in ("text", "image")]
        for source in sources:
            actual = _extract_aspect(source, getattr(template, source)[index])
            if actual != source_aspect:
                raise ValueError(f"Additional perspective changed at {source}[{index}]: expected {source_aspect!r}, got {actual!r}")
        aliases = [dict(label=None, aspect=source_aspect, sources=sources)]
        rows.append(dict(label=label, aspect=aspect, sources=sources, aliases=aliases))
        seen.add(aspect.casefold())
    if [row["label"] for row in rows] != list("ABCDEFGHIJKLMNOPQRST"):
        raise ValueError("Expected exactly 20 adaptive perspectives with labels A-T")
    return rows


def task_view_mappings(pool=None):
    """Map five source positions; retired items without a reference override remain None."""
    import template
    pool = view_pool() if pool is None else pool
    lookup = {alias["aspect"]: row["label"] for row in pool for alias in row["aliases"]}
    available = {row["label"] for row in pool}
    result = {}
    for source in SOURCES:
        labels = []
        overrides = TASK_VIEW_REFERENCE_OVERRIDES.get(source, {})
        for prompt in getattr(template, source):
            aspect = _extract_aspect(source, prompt)
            if aspect in overrides:
                label = overrides[aspect]
                if label not in available:
                    raise ValueError(f"Missing reference perspective {label!r} for {source}")
                labels.append(label)
            elif aspect in RETIRED_SOURCE_ASPECTS:
                labels.append(None)
            elif aspect in lookup:
                labels.append(lookup[aspect])
            else:
                raise ValueError(f"No adaptive mapping for retained source perspective {aspect!r} in {source}")
        mapped = [label for label in labels if label is not None]
        if len(labels) != TOP_K or len(set(mapped)) != len(mapped):
            raise ValueError(f"Original task {source} must retain five source positions with distinct mapped perspectives")
        result[source] = labels
    return result


def resolve_modality(value=None):
    mode = "both" if value is None else value
    if mode not in SELECTION_MODALITIES:
        raise ValueError(f"view_selection_modality must be text, image or both, got {value!r}")
    return mode


def enabled_modalities(value=None):
    mode = resolve_modality(value)
    return ("text", "image") if mode == "both" else (mode,)


def modality_weights(value=None):
    active = enabled_modalities(value)
    return {name: 1 / len(active) if name in active else 0.0 for name in ("text", "image")}


def selection_sampling(options, mode=None, *, model_config=None, example_configuration=None):
    mode = resolve_modality(getattr(options, "view_selection_modality", None) if mode is None else mode)
    active = enabled_modalities(mode)
    if options.view_selection_rounds < 1:
        raise ValueError("View selection rounds must be positive")
    if examples.validate_options(options) == "auto":
        policy = examples.configuration(options, model_config) if example_configuration is None else example_configuration
        examples.validate_configuration(policy)
        return dict(rounds=options.view_selection_rounds, seed=options.view_selection_seed,
                    texts_per_round=None, images_per_round=None, **{examples.FIELD: policy})
    result = dict(rounds=options.view_selection_rounds, seed=options.view_selection_seed,
                  texts_per_round=options.view_texts_per_round if "text" in active else None,
                  images_per_round=options.view_images_per_round if "image" in active else None)
    if result["rounds"] < 1 or any(result[name] is not None and result[name] < 1
                                   for name in ("texts_per_round", "images_per_round")):
        raise ValueError("View selection rounds and enabled modality sample counts must be positive")
    return result


def pmi_configuration(shuffle=None, *, pool_mode=None):
    import template
    pool = option_shuffle.display_pool(view_pool(pool_mode), shuffle)
    return dict(method="pmi", version=1, normalization="per_round_softmax",
                prior_prompt_fingerprint=digest(dict(
                    instruction=CHOICE_INSTRUCTION, answer="Answer with one letter:",
                    candidates=[(row["label"], row["aspect"]) for row in pool],
                    llama=template.llama3_template, mistral=template.llava_mistral_template)))


def selection_cache_key(identity, dataset, source_split, mode, sampling, data_root, *, pmi=False, strategy=None, shuffle=None, pool_mode=None):
    """The original encoding cache contract; shared with read-only retrieval lookup."""
    key = dict(model=identity, dataset=dataset, split=source_split,
               pool=view_pool(pool_mode), selection_modality=mode, sampling=sampling,
               data_root=str(Path(data_root).resolve()))
    if pmi:
        key["calibration"] = pmi_configuration(shuffle, pool_mode=pool_mode)
    if strategy is not None:
        key["selection_strategy"] = strategy
    return add_pool_mode(option_shuffle.add_metadata(key, shuffle), pool_mode)


def selection_cache_path(sparse_root, model_path, dataset, key):
    return (Path(sparse_root) / "view_selection" / Path(model_path).name / dataset /
            option_shuffle.mark_path(Path(digest(key)[:16]), key.get(option_shuffle.FIELD)) / "view_selection.json")


def is_pmi_selection(selection):
    return selection.get("version") in {PMI_VERSION, PMI_OVERLAP_VERSION}


def is_overlap_selection(selection):
    return selection.get("version") in {OVERLAP_VERSION, PMI_OVERLAP_VERSION}


def resolve_pmi(requested=None, *, selection=None):
    if requested is not None and type(requested) is not bool:
        raise ValueError("view_selection_pmi must be true or false")
    stored = selection is not None and is_pmi_selection(selection)
    if selection is not None and requested is not None and requested != stored:
        raise ValueError(f"View selection PMI mismatch: stored={stored}, requested={requested}; "
                         "omit the switch to inherit the file or choose a new selection file")
    return stored if selection is not None else bool(requested)


def views_fingerprint(selected, selection_modality="both", *, pmi=False, shuffle=None, pool_mode=None, example_configuration=None):
    configuration = dict(version=PMI_VERSION if pmi else VERSION,
                         selection_modality=resolve_modality(selection_modality), selected=selected)
    if pmi:
        configuration["calibration"] = pmi_configuration(shuffle, pool_mode=pool_mode)
    return digest(examples.add_configuration(add_pool_mode(option_shuffle.add_metadata(configuration, shuffle), pool_mode), example_configuration))


def pmi_probabilities(log_probabilities, prior_log_probabilities):
    """Normalize a log-probability ratio without dividing underflowed probabilities."""
    import numpy as np
    conditional = np.asarray(log_probabilities, dtype="float64")
    prior = np.asarray(prior_log_probabilities, dtype="float64")
    if (conditional.ndim != 1 or not conditional.size or conditional.shape != prior.shape
            or not np.isfinite(conditional).all() or not np.isfinite(prior).all()):
        raise ValueError("PMI requires matching finite candidate log-probability vectors")
    scores = conditional - prior
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite PMI scores")
    weights = np.exp(scores - scores.max())
    return (weights / weights.sum()).tolist(), scores.tolist()


def validate_pmi_selection(selection, *, check_selected=True):
    import numpy as np
    calibration = selection.get("calibration", {})
    require_equal(calibration.get("configuration", {}),
                  pmi_configuration(selection.get(option_shuffle.FIELD), pool_mode=pool_mode(selection)), "PMI configuration")
    prior = np.asarray(calibration.get("prior_probabilities"), dtype="float64")
    log_prior = np.asarray(calibration.get("prior_log_probabilities"), dtype="float64")
    count = len(selection["pool"])
    if (prior.shape != (count,) or log_prior.shape != (count,)
            or not np.isfinite(prior).all() or not np.isfinite(log_prior).all()
            or (prior < 0).any() or (log_prior > 0).any()
            or not np.isclose(prior.sum(), 1, atol=1e-6)
            or not np.allclose(prior, np.exp(log_prior), atol=1e-7, rtol=1e-6)
            or type(calibration.get("prior_tokens")) is not int or calibration["prior_tokens"] < 1
            or not isinstance(calibration.get("prior_prompt"), str) or not calibration["prior_prompt"]):
        raise ValueError("Invalid PMI prior diagnostics")
    matrices = {}
    for modality in enabled_modalities(selection["selection_modality"]):
        matrices[modality] = []
        for row in selection.get("rounds", []):
            raw = np.asarray(row.get(modality + "_raw_probabilities"), dtype="float64")
            scores = np.asarray(row.get(modality + "_pmi_scores"), dtype="float64")
            if (raw.shape != (count,) or scores.shape != (count,)
                    or not np.isfinite(scores).all() or (scores + log_prior > 1e-6).any()):
                raise ValueError(f"Invalid {modality} PMI round diagnostics")
            corrected, _ = pmi_probabilities(scores + log_prior, log_prior)
            if (not np.allclose(raw, np.exp(scores + log_prior), atol=1e-7, rtol=1e-5)
                    or not np.allclose(row[modality + "_probabilities"], corrected, atol=1e-7)):
                raise ValueError(f"Inconsistent {modality} PMI round diagnostics")
            matrices[modality].append(corrected)
    ranking = rank_probabilities(matrices.get("text"), matrices.get("image"), selection["pool"],
                                 selection_modality=selection["selection_modality"])
    if ((check_selected and ranking["selected"] != selection["selected"])
            or not np.allclose(ranking["scores"], selection["scores"], atol=1e-7)):
        raise ValueError("PMI selection differs from its corrected probability ranking")
    if not check_selected:
        for name in ("text_mean", "image_mean"):
            actual, expected = selection.get(name), ranking[name]
            if ((actual is None) != (expected is None)
                    or (expected is not None and not np.allclose(actual, expected, atol=1e-7))):
                raise ValueError(f"PMI {name} differs from its corrected probability ranking")


def family_of(config):
    from qwen_vl_prompts import model_family, MODEL_TYPES
    family = model_family(config)
    if family in MODEL_TYPES:
        return family
    text = getattr(config, "text_config", config)
    if family == "llava_next" and getattr(text, "model_type", None) in {"llama", "mistral"}:
        return "llava_" + text.model_type
    raise ValueError("Adaptive views support LLaVA-NeXT Llama3/Mistral and Qwen2/2.5/3-VL Instruct only")


def checkpoint_identity(path):
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Adaptive views require a local checkpoint directory: {root}")
    return dict(path=str(root), files=[
        dict(name=item.name, size=item.stat().st_size, mtime_ns=item.stat().st_mtime_ns)
        for item in sorted(root.iterdir()) if item.is_file()
        and (item.suffix in {".safetensors", ".bin"} or item.name == "config.json")
    ])


def _json_metadata(value):
    """Copy processor metadata into JSON types without altering preprocessing."""
    from transformers.image_utils import SizeDict

    if isinstance(value, SizeDict):
        # Unset SizeDict fields are absent from equivalent legacy size dicts.
        return {item.name: _json_metadata(getattr(value, item.name))
                for item in fields(value) if getattr(value, item.name) is not None}
    if isinstance(value, Mapping):
        return {key: _json_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_metadata(item) for item in value]
    return value


def model_identity(path, config, processor, *, dtype, lora=None):
    image = processor.image_processor
    vision = {name: getattr(image, name, None) for name in (
        "size", "min_pixels", "max_pixels", "patch_size", "temporal_patch_size", "merge_size",
        "image_grid_pinpoints", "crop_size", "image_mean", "image_std", "do_resize", "do_rescale",
        "rescale_factor", "do_normalize",
    )}
    if getattr(processor, "llava_image_mode", None) == "base":
        from llava_resolution import resolution_metadata
        vision["llava_resolution"] = resolution_metadata(config, "base")
    return _json_metadata(dict(
        checkpoint=checkpoint_identity(path), adapter=checkpoint_identity(lora) if lora else None,
        family=family_of(config), dtype=str(dtype).removeprefix("torch."),
        tokenizer=type(processor.tokenizer).__name__, vocab=digest(processor.tokenizer.get_vocab()),
        chat_template=getattr(processor, "chat_template", None) or getattr(processor.tokenizer, "chat_template", None),
        image_processor=type(image).__name__, vision=vision,
        processor_options={name: getattr(processor, name, None) for name in (
            "patch_size", "num_additional_image_tokens", "vision_feature_select_strategy", "image_token",
        )},
    ))


def differences(actual, expected, prefix=""):
    result = []
    for key in sorted(set(actual) | set(expected)):
        a, e = actual.get(key), expected.get(key)
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(a, dict) and isinstance(e, dict):
            result.extend(differences(a, e, name))
        elif a != e:
            result.append(f"{name}: stored={repr(a)[:160]}, requested={repr(e)[:160]}")
    return result


def require_equal(actual, expected, label):
    problems = differences(actual, expected)
    if problems:
        raise ValueError(f"{label} mismatch:\n" + "\n".join(problems[:20]))


class IncompatibleViewSelectionError(ValueError):
    """A self-consistent selection whose version or candidate pool cannot be reused."""


class LegacyOverlapSelectionError(IncompatibleViewSelectionError):
    """A selection made with the retired union-denominator, high-overlap rule."""


def selection_compatibility_details(selection, expected_pool):
    stored_pool = selection.get("pool")
    if isinstance(stored_pool, list):
        count = len(stored_pool)
        labels = [row["label"] for index, row in enumerate(expected_pool)
                  if index >= count or stored_pool[index] != row]
        labels.extend(str(row.get("label", f"entry[{index}]")) if isinstance(row, dict) else f"entry[{index}]"
                      for index, row in enumerate(stored_pool[len(expected_pool):], start=len(expected_pool)))
        differing = ",".join(labels[:20]) or "none"
        if len(labels) > 20:
            differing += f",... (+{len(labels) - 20})"
    else:
        count, differing = "missing/invalid", "unknown (pool is not a list)"
    supported = sorted({VERSION, OVERLAP_VERSION, PMI_VERSION, PMI_OVERLAP_VERSION})
    return (f"stored version={selection.get('version')!r}; supported versions={supported}; "
            f"stored candidates={count}, expected candidates={len(expected_pool)}; "
            f"differing candidate labels={differing}. ")


def load_selection(path, *, identity=None, dataset=None, selection_modality=None, view_selection_pmi=None,
                   view_selection_hierarchical=None, view_example_mode=None, view_example_max_tokens=None):
    path = Path(path).expanduser()
    value = json.loads(path.read_text(encoding="utf-8"))
    stored = value.get("fingerprint")
    if stored != digest({key: item for key, item in value.items() if key != "fingerprint"}):
        raise ValueError(f"Corrupt adaptive view selection fingerprint: {path}")
    example_policy = examples.validate_selection(value, mode=view_example_mode, maximum=view_example_max_tokens)
    candidate_mode = pool_mode(value)
    if view_selection_hierarchical is not None:
        requested = options_pool_mode(SimpleNamespace(view_selection_hierarchical=view_selection_hierarchical))
        if requested != candidate_mode:
            raise ValueError(f"View pool mode mismatch: stored={candidate_mode or 'original'}, requested={requested or 'original'}; "
                             "match USE_HIERARCHICAL_VIEWS to the file or choose a new selection file")
    expected_pool = view_pool(candidate_mode)
    shuffle = value.get(option_shuffle.FIELD)
    option_shuffle.validate(shuffle, expected_pool, identity=value.get("model"), dataset=value.get("dataset"))
    option_shuffle.validate_diagnostics(value)
    if value.get("version") in {13, 15}:
        details = selection_compatibility_details(value, expected_pool)
        raise LegacyOverlapSelectionError(
            f"Legacy overlap selection version {value['version']} is incompatible: {path}; "
            f"{details}"
            "reselect views with candidate_fraction_lt, re-encode the corpus and rebuild its index. "
            "Omit --view_selection_file or choose a new file.")
    if value.get("version") not in {VERSION, OVERLAP_VERSION, PMI_VERSION, PMI_OVERLAP_VERSION} or value.get("pool") != expected_pool:
        details = selection_compatibility_details(value, expected_pool)
        raise IncompatibleViewSelectionError(
            f"Adaptive view pool/version mismatch: {path}; expected version {VERSION}, {OVERLAP_VERSION}, {PMI_VERSION} or {PMI_OVERLAP_VERSION} with {len(expected_pool)} views (A-{expected_pool[-1]['label']}). "
            f"{details}"
            "Rerun encode_adaptive_views.py without --view_selection_file, or choose a new selection file, "
            "then encode and index the new artifacts.")
    mode = value.get("selection_modality")
    if mode not in SELECTION_MODALITIES:
        raise ValueError(f"Missing or invalid selection_modality in {path}")
    if selection_modality is not None and resolve_modality(selection_modality) != mode:
        raise ValueError(f"View selection modality mismatch: stored={mode}, requested={selection_modality}; "
                         "omit the mode to inherit the file or choose a new selection file")
    selected = value.get("selected", [])
    pmi = resolve_pmi(view_selection_pmi, selection=value)
    lookup = {row["label"]: row["aspect"] for row in value["pool"]}
    count = TOP_K
    fingerprint = views_fingerprint(selected, mode, pmi=pmi, shuffle=shuffle, pool_mode=candidate_mode,
                                    example_configuration=example_policy)
    if is_overlap_selection(value):
        from adaptive_view_overlap import validate_selection, overlap_fingerprint
        count = validate_selection(value)
        fingerprint = overlap_fingerprint(value)
    elif pmi:
        validate_pmi_selection(value)
    if (len(selected) != count or len({row.get("label") for row in selected}) != count
            or any(lookup.get(row.get("label")) != row.get("aspect") for row in selected)
            or value.get("views_fingerprint") != fingerprint):
        raise ValueError(f"Invalid adaptive selected perspectives (expected {count}): {path}")
    if identity is not None:
        require_equal(value["model"], identity, "View selection model/processor")
    if dataset is not None and value.get("dataset") != dataset:
        raise ValueError(f"View selection dataset mismatch: stored={value.get('dataset')}, requested={dataset}")
    return value


def rank_probabilities(text_probabilities=None, image_probabilities=None, pool=None, *, selection_modality="both"):
    import numpy as np
    pool = pool or view_pool()
    mode = resolve_modality(selection_modality)
    active = enabled_modalities(mode)
    weights = modality_weights(mode)
    means = dict(text=None, image=None)
    scores = np.zeros(len(pool), dtype="float64")
    for name, probabilities in (("text", text_probabilities), ("image", image_probabilities)):
        if name not in active:
            if probabilities is not None:
                raise ValueError(f"Inactive {name} probabilities must be None in {mode} mode")
            continue
        matrix = np.asarray(probabilities, dtype="float64")
        if (matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] != len(pool)
                or not np.isfinite(matrix).all() or (matrix < 0).any()
                or not np.allclose(matrix.sum(axis=1), 1, atol=1e-6)):
            raise ValueError(f"Expected finite candidate-normalized perspective probabilities for enabled {name} modality")
        means[name] = matrix.mean(axis=0).tolist()
        scores += weights[name] * matrix.mean(axis=0)
    order = sorted(range(len(pool)), key=lambda index: (-scores[index], index))
    selected = [dict(label=pool[index]["label"], aspect=pool[index]["aspect"]) for index in order[:TOP_K]]
    return dict(selected=selected, text_mean=means["text"], image_mean=means["image"],
                scores=scores.tolist(), views_fingerprint=views_fingerprint(selected, mode))


def sampled_groups(items, count, rounds, seed, modality):
    if not items:
        raise ValueError(f"No {modality} samples available for adaptive view selection")
    if count < 1 or rounds < 1:
        raise ValueError("View sampling counts and rounds must be positive")
    for round_index in range(rounds):
        generator = random.Random(f"adaptive-views/{seed}/{modality}/{round_index}")
        yield generator.sample(items, min(count, len(items)))


def render_choice(processor, family, pool, *, texts=None, image_count=0, prior=False):
    if prior and (texts is not None or image_count):
        raise ValueError("The PMI prior cannot contain examples or images")
    options = "\n".join(f"{row['label']}. {row['aspect']}" for row in pool)
    examples = "\n".join(f"Example {index + 1}: {text}" for index, text in enumerate(texts or []))
    body = CHOICE_INSTRUCTION
    if not prior:
        body += ("\nText examples:\n" + examples if texts is not None else
                 f"\nImage examples: {image_count} unordered images.\n")
    body += "\nCandidate perspectives:\n" + options + "\nAnswer with one letter:"
    if family.startswith("qwen"):
        content = [dict(type="image") for _ in range(image_count)] + [dict(type="text", text=body)]
        return processor.apply_chat_template([dict(role="user", content=content)], tokenize=False, add_generation_prompt=True)
    import template
    body = "<image>\n" * image_count + body
    if family == "llava_mistral":
        return template.llava_mistral_template.format(body)
    if family == "llava_llama":
        return template.llama3_template.format(body)
    raise ValueError(f"Unsupported choice family: {family}")


def prepare_choice_group(processor, family, pool, items, modality, *, prior=False):
    from qwen_vl_encoding import read_image
    if prior and items:
        raise ValueError("The PMI prior requires an empty sample group")
    images = [read_image(item["image"]) for item in items] if modality == "image" else []
    prompt = render_choice(processor, family, pool,
                           texts=None if prior or modality == "image" else [item["text"] for item in items],
                           image_count=len(images), prior=prior)
    kwargs = dict(text=[prompt], padding=True, truncation=False, return_tensors="pt", add_special_tokens=family == "llava_mistral")
    if images:
        kwargs["images"] = images
    inputs = dict(processor(**kwargs))
    length = int(inputs["attention_mask"].sum())
    return dict(inputs=inputs, tokens=length, prompt=prompt)


def score_group(model, processor, family, pool, items, modality, *, prior=False, return_details=False,
                prepared=None, input_limit=None):
    import torch
    from qwen_vl_encoding import _forward_inputs
    labels = [row["label"] for row in pool]
    ids = [processor.tokenizer.encode(label, add_special_tokens=False) for label in labels]
    if any(len(row) != 1 for row in ids) or len({row[0] for row in ids}) != len(labels):
        raise ValueError("Every perspective letter must be one distinct tokenizer token")
    if any(row[0] in processor.tokenizer.all_special_ids for row in ids):
        raise ValueError("Perspective letters cannot be special tokens")
    if prior and items:
        raise ValueError("The PMI prior requires an empty sample group")
    prepared = prepare_choice_group(processor, family, pool, items, modality, prior=prior) if prepared is None else prepared
    inputs, length, prompt = prepared["inputs"], prepared["tokens"], prepared["prompt"]
    text_config = getattr(model.config, "text_config", model.config)
    limit = getattr(text_config, "max_position_embeddings", None)
    if input_limit is not None:
        limit = min(limit, input_limit) if limit else input_limit
    if limit and length > limit:
        raise ValueError(f"View-selection {modality} group has {length} tokens, context limit={limit}; "
                         "reduce samples per round or image pixels; inputs are never silently truncated")
    device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    if family.startswith("qwen"):
        inputs = _forward_inputs(model, inputs)
    mask = inputs["attention_mask"].bool()
    last = mask[0].nonzero(as_tuple=True)[0][-1].item()
    forward = dict(use_cache=False, output_hidden_states=False, return_dict=True)
    # All supported wrappers accept this in recent Transformers. Older releases
    # return the full sequence; still select its last valid position explicitly.
    if "logits_to_keep" in inspect.signature(model.forward).parameters:
        forward["logits_to_keep"] = torch.tensor([last], device=device)
    with torch.inference_mode():
        output = model(**inputs, **forward)
        row = output.logits[0, 0 if "logits_to_keep" in forward else last]
        candidate_logits = row[[value[0] for value in ids]].float()
        probabilities = candidate_logits.softmax(dim=-1)
    if not torch.isfinite(probabilities).all():
        raise ValueError("Non-finite last-token view probabilities")
    if return_details:
        log_probabilities = candidate_logits.log_softmax(dim=-1)
        if not torch.isfinite(log_probabilities).all():
            raise ValueError("Non-finite candidate log-probabilities")
        return dict(probabilities=probabilities.cpu().tolist(), log_probabilities=log_probabilities.cpu().tolist(),
                    tokens=length, prompt=prompt)
    return probabilities.cpu().tolist(), length


def log_round_result(pool, probabilities, *, modality, round_index, rounds, sample_ids, tokens,
                     raw_probabilities=None, prior_probabilities=None, pmi_scores=None):
    """Report completed scoring without changing probabilities or candidate order."""
    order = sorted(range(len(pool)), key=lambda index: (-probabilities[index], index))
    prefix = f"[adaptive views] {modality} round {round_index + 1}/{rounds}"
    best = pool[order[0]]
    lines = [
        f"{prefix}; samples={len(sample_ids)} tokens={tokens}",
        f"{prefix} sample_ids={json.dumps(sample_ids, ensure_ascii=False)}",
        f"{prefix} top1={best['label']} ({best['aspect']}) probability={probabilities[order[0]]:.8f}",
    ]
    for rank, index in enumerate(order, 1):
        row = pool[index]
        detail = (f" raw_probability={raw_probabilities[index]:.8f} prior_probability={prior_probabilities[index]:.8f} "
                  f"pmi={pmi_scores[index]:.8f} corrected_probability={probabilities[index]:.8f}"
                  if raw_probabilities is not None else "")
        lines.append(f"{prefix} rank={rank} view={row['label']} ({row['aspect']}) "
                     f"probability={probabilities[index]:.8f}{detail}")
    print("\n".join(lines), flush=True)


def log_probability_ranking(selection):
    pool, scores = selection["pool"], selection["scores"]
    order = sorted(range(len(pool)), key=lambda index: (-scores[index], index))
    prefix = "[adaptive views] probability ranking"
    lines = [f"{prefix}; modality={selection['selection_modality']}"]
    if is_pmi_selection(selection):
        lines[0] += "; calibration=pmi (corrected probabilities)"
    for rank, index in enumerate(order, 1):
        row = pool[index]
        means = {name: "N/A" if selection[name + "_mean"] is None else f"{selection[name + '_mean'][index]:.8f}"
                 for name in ("text", "image")}
        lines.append(f"{prefix} rank={rank} view={row['label']} ({row['aspect']}) "
                     f"text_mean={means['text']} image_mean={means['image']} score={scores[index]:.8f}")
    print("\n".join(lines), flush=True)


def log_selection(selection, path, *, reused=False):
    option_shuffle.log_mapping(selection.get(option_shuffle.FIELD))
    status = "reused locked selection" if reused else "saved selection"
    print(f"[adaptive views] {status}; path={Path(path).resolve()} "
          f"modality={selection['selection_modality']} selected_count={len(selection['selected'])}", flush=True)
    policy = selection.get(examples.FIELD)
    if policy is not None:
        print(f"[adaptive views] example_mode=auto requested_max_tokens={policy['max_tokens']} "
              f"effective_max_tokens={policy['effective_max_tokens']} input_limit={policy['input_limit']}", flush=True)
    if reused:
        if is_pmi_selection(selection):
            log_prior(selection["pool"], selection["calibration"])
        log_probability_ranking(selection)
    print(f"[adaptive views] selected_views={json.dumps(selection['selected'], ensure_ascii=False)}", flush=True)
    if pool_mode(selection) is not None:
        print(f"[adaptive views] view_pool_mode={pool_mode(selection)} candidates={len(selection['pool'])}", flush=True)


def log_prior(pool, calibration):
    probabilities = calibration["prior_probabilities"]
    order = sorted(range(len(pool)), key=lambda index: (-probabilities[index], index))
    lines = [f"[adaptive PMI prior] no examples; tokens={calibration['prior_tokens']}"]
    for rank, index in enumerate(order, 1):
        row = pool[index]
        lines.append(f"[adaptive PMI prior] rank={rank} view={row['label']} ({row['aspect']}) "
                     f"probability={probabilities[index]:.8f} "
                     f"log_probability={calibration['prior_log_probabilities'][index]:.8f}")
    print("\n".join(lines), flush=True)


def choose_views(model, processor, identity, dataset, source_split, texts, images, options):
    candidate_mode = options_pool_mode(options)
    pool = view_pool(candidate_mode)
    shuffle = option_shuffle.for_options(options, identity, dataset, pool)
    displayed_pool = option_shuffle.display_pool(pool, shuffle)
    option_shuffle.log_mapping(shuffle)
    family = family_of(model.config)
    mode = resolve_modality(options.view_selection_modality)
    active = enabled_modalities(mode)
    sampling = selection_sampling(options, mode, model_config=model.config)
    example_policy = sampling.get(examples.FIELD)
    score_options = dict(input_limit=example_policy["input_limit"]) if example_policy else {}
    pmi = resolve_pmi(getattr(options, "view_selection_pmi", None))
    pools = dict(text=texts, image=images)
    for name in active:
        if not pools[name]:
            raise ValueError(f"No {name} samples available for adaptive view selection")
    rows = [dict(round=index, text_ids=[], image_ids=[], text_probabilities=None,
                 image_probabilities=None, text_tokens=None, image_tokens=None)
            for index in range(sampling["rounds"])]
    calibration = None
    if pmi:
        prior = score_group(model, processor, family, displayed_pool, [], "text", prior=True, return_details=True, **score_options)
        if shuffle is not None:
            display_prior = dict(display_prior_probabilities=prior["probabilities"], display_prior_log_probabilities=prior["log_probabilities"])
            for key in ("probabilities", "log_probabilities"):
                prior[key] = option_shuffle.canonical_values(prior[key],pool,shuffle)
        calibration = dict(configuration=pmi_configuration(shuffle, pool_mode=candidate_mode), prior_prompt=prior["prompt"],
                           prior_tokens=prior["tokens"], prior_probabilities=prior["probabilities"],
                           prior_log_probabilities=prior["log_probabilities"])
        if shuffle is not None:
            calibration.update(display_prior)
        log_prior(pool, calibration)
    # Complete all text rounds before starting image rounds. Each modality is
    # scored independently over the full pool with its own deterministic seed stream.
    for name in active:
        count = sampling["texts_per_round" if name == "text" else "images_per_round"]
        groups = (examples.ordered_groups(pools[name], sampling["rounds"], sampling["seed"], name) if example_policy
                  else sampled_groups(pools[name], count, sampling["rounds"], sampling["seed"], name))
        for index, group in enumerate(groups):
            log_details = {}
            group_score_options = dict(score_options)
            if example_policy:
                group, prepared, diagnostics = examples.pack(
                    group, lambda items: prepare_choice_group(processor, family, displayed_pool, items, name), example_policy, name)
                group_score_options["prepared"] = prepared
                rows[index][name + "_examples"] = diagnostics
                examples.log_round(diagnostics, modality=name, index=index, rounds=sampling["rounds"])
            if pmi:
                details = score_group(model, processor, family, displayed_pool, group, name, return_details=True, **group_score_options)
                if shuffle is not None:
                    rows[index][name + "_display_raw_probabilities"] = details["probabilities"]
                    for key in ("probabilities", "log_probabilities"):
                        details[key] = option_shuffle.canonical_values(details[key],pool,shuffle)
                probabilities, pmi_scores = pmi_probabilities(details["log_probabilities"],
                                                             calibration["prior_log_probabilities"])
                length = details["tokens"]
                rows[index][name + "_raw_probabilities"] = details["probabilities"]
                rows[index][name + "_pmi_scores"] = pmi_scores
                log_details = dict(raw_probabilities=details["probabilities"],
                                   prior_probabilities=calibration["prior_probabilities"], pmi_scores=pmi_scores)
            else:
                probabilities, length = score_group(model, processor, family, displayed_pool, group, name, **group_score_options)
                probabilities = option_shuffle.canonical_values(probabilities,pool,shuffle)
            if example_policy:
                group_score_options.pop("prepared")
                del prepared
            rows[index][name + "_ids"] = [item["id"] for item in group]
            rows[index][name + "_probabilities"] = probabilities
            rows[index][name + "_tokens"] = length
            option_shuffle.log_round(shuffle,pool,probabilities,modality=name,round_index=index,row=rows[index])
            log_round_result(pool, probabilities, modality=name, round_index=index, rounds=sampling["rounds"],
                             sample_ids=rows[index][name + "_ids"], tokens=length, **log_details)
    text_p = [row["text_probabilities"] for row in rows] if "text" in active else None
    image_p = [row["image_probabilities"] for row in rows] if "image" in active else None
    ranking = rank_probabilities(text_p, image_p, pool, selection_modality=mode)
    result = dict(version=PMI_VERSION if pmi else VERSION, pool=pool, task_views={} if candidate_mode else task_view_mappings(pool),
                  selection_modality=mode,
                  model=identity, dataset=dataset, selection_split=source_split,
                  sampling=sampling, modality_weights=modality_weights(mode), rounds=rows, **ranking)
    if pmi:
        result["calibration"] = calibration
    if pmi or shuffle is not None or candidate_mode is not None:
        result["views_fingerprint"] = views_fingerprint(result["selected"], mode, pmi=pmi, shuffle=shuffle, pool_mode=candidate_mode,
                                                       example_configuration=example_policy)
    elif example_policy:
        result["views_fingerprint"] = views_fingerprint(result["selected"], mode, example_configuration=example_policy)
    result = add_pool_mode(option_shuffle.add_metadata(result,shuffle), candidate_mode)
    result = examples.add_configuration(result, example_policy)
    result["fingerprint"] = digest(result)
    log_probability_ranking(result)
    return result


def artifact_root(path):
    from qwen_vl_paths import artifact_root as root
    return root(path)


def validate_artifact(path, expected):
    manifest = artifact_root(path) / MANIFEST
    if not manifest.is_file():
        raise ValueError(f"Missing adaptive completion manifest: {manifest}; use encode_adaptive_views.py")
    actual = json.loads(manifest.read_text(encoding="utf-8"))
    from dense_representation import validate_artifact as validate_dense_mode
    validate_dense_mode(path, bool(expected["configuration"].get("dense_representation")),
                        bool(expected["configuration"].get("freeret_prompt_constraints")), bool(expected["configuration"].get("freeret_full_prompt")))
    require_equal(actual["configuration"], expected["configuration"], f"Adaptive corpus at {manifest.parent}")
    if actual.get("fingerprint") != digest(actual["configuration"]):
        raise ValueError(f"Corrupt adaptive corpus manifest: {manifest}")
    return actual


class SelectionFileResolver:
    """Find exact copies of missing locked selections, caching each search directory once."""
    def __init__(self, selection_root=None, *, sparse_root=None):
        self.selection_root = Path(selection_root).expanduser().resolve() if selection_root else None
        self.sparse_root = Path(sparse_root).expanduser().resolve() if sparse_root else None
        self._indexes = {}

    def search_root(self, artifact):
        if self.selection_root is not None:
            return self.selection_root
        preferred = self.sparse_root / "view_selection" if self.sparse_root is not None else None
        if preferred is not None and preferred.is_dir():
            return preferred
        artifact = Path(artifact).resolve()
        for ancestor in (artifact, *artifact.parents):
            if ancestor == ancestor.parent:
                break
            candidate = ancestor / "view_selection"
            if candidate.is_dir():
                return candidate
        return preferred if preferred is not None else artifact / "view_selection"

    def recover(self, original, artifact, manifest):
        expected = manifest.get("selection_fingerprint")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"Cannot recover missing selection {original}: missing/invalid full selection_fingerprint")
        directory = self.search_root(artifact)
        guidance = (f"Missing selection file: {original}; expected selection_fingerprint={expected}; "
                    f"expected views_fingerprint={manifest['configuration'].get('views_fingerprint')}; "
                    f"recovery directory={directory}. ")
        if not directory.is_dir():
            raise FileNotFoundError(guidance + "Recovery directory does not exist. Set VIEW_SELECTION_ROOT, restore the original JSON or re-encode the corpus.")
        if directory not in self._indexes:
            index, unreadable = {}, 0
            for path in sorted(directory.rglob("*.json")):
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    fingerprint = value.get("fingerprint") if isinstance(value, dict) else None
                    if isinstance(fingerprint, str) and re.fullmatch(r"[0-9a-f]{64}", fingerprint):
                        index.setdefault(fingerprint, []).append(path)
                except (OSError, ValueError):
                    unreadable += 1
            self._indexes[directory] = index, unreadable
        index, unreadable = self._indexes[directory]
        invalid_matches = 0
        for path in index.get(expected, []):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if (not isinstance(value, dict) or value.get("fingerprint") != expected
                        or digest({key: item for key, item in value.items() if key != "fingerprint"}) != expected
                        or value.get("views_fingerprint") != manifest["configuration"].get("views_fingerprint")):
                    invalid_matches += 1
                    continue
            except (OSError, ValueError, TypeError):
                invalid_matches += 1
                continue
            print(f"[adaptive selection] recovered missing file: {original} -> {path} "
                  f"(selection_fingerprint={expected}; ignored_unreadable={unreadable}; invalid_matches={invalid_matches})", flush=True)
            return path
        raise FileNotFoundError(guidance + f"No exact full-fingerprint match; ignored_unreadable={unreadable}, "
                                f"invalid_matches={invalid_matches}. Restore the original JSON or re-encode the corpus.")


def validate_index_selection(root, manifest, *, resolver=None):
    """Check the locked selection before building or reusing an adaptive index."""
    root = Path(root)
    configuration = manifest["configuration"]
    from dense_representation import validate_artifact as validate_dense_mode
    validate_dense_mode(root, bool(configuration.get("dense_representation")), bool(configuration.get("freeret_prompt_constraints")), bool(configuration.get("freeret_full_prompt")))
    if manifest.get("fingerprint") != digest(configuration):
        raise ValueError(f"Corrupt adaptive configuration fingerprint: {root / MANIFEST}")
    filename = manifest.get("selection_file")
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"Missing selection_file in {root / MANIFEST}; cannot validate the encoding rule")
    selection_path = Path(filename).expanduser()
    if not selection_path.is_absolute():
        selection_path = root / selection_path
    try:
        selection = load_selection(selection_path, identity=configuration.get("model"), dataset=configuration.get("dataset"))
    except FileNotFoundError:
        resolver = resolver if resolver is not None else SelectionFileResolver()
        recovered = resolver.recover(selection_path, root, manifest)
        selection = load_selection(recovered, identity=configuration.get("model"), dataset=configuration.get("dataset"))
    if (selection["fingerprint"] != manifest.get("selection_fingerprint")
            or selection["views_fingerprint"] != configuration.get("views_fingerprint")
            or selection["selected"] != configuration.get("selected")
            or selection["selection_modality"] != configuration.get("selection_modality")
            or pool_mode(selection) != pool_mode(configuration)
            or is_pmi_selection(selection) != (configuration.get("selection_calibration", {}).get("method") == "pmi")):
        raise ValueError(f"Adaptive manifest/selection fingerprint or configuration mismatch: {root}")
    require_equal({"calibration": configuration.get("selection_calibration")},
                  {"calibration": selection.get("calibration", {}).get("configuration")}, "Adaptive index calibration")
    if configuration.get(option_shuffle.FIELD) != selection.get(option_shuffle.FIELD):
        raise ValueError("Adaptive manifest/selection option shuffle mismatch")
    require_equal(dict(examples=configuration.get(examples.FIELD)), dict(examples=selection.get(examples.FIELD)),
                  "Adaptive index example sampling")
    if is_overlap_selection(selection):
        require_equal(dict(full=selection["overlap_configuration"].get("freeret_full_prompt")),
                      dict(full=configuration.get("freeret_full_prompt")), "Adaptive index overlap complete FreeRet prompt")
        require_equal(dict(constraints=selection["overlap_configuration"].get("freeret_prompt_constraints")),
                      dict(constraints=configuration.get("freeret_prompt_constraints")), "Adaptive index overlap prompt constraints")
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pool_parser = commands.add_parser("pool", help="Print the exact letter-to-perspective catalog")
    pool_parser.add_argument("--output")
    pool_parser.add_argument("--hierarchical", action="store_true", help="Print the ten primary/secondary dimensions")
    index_parser = commands.add_parser("index", help="Build collection/index from completed adaptive JSONL shards")
    index_parser.add_argument("--sparse-root", required=True)
    index_parser.add_argument("--threads", type=int, default=16)
    index_parser.add_argument("--view-selection-root", type=Path, default=os.environ.get("VIEW_SELECTION_ROOT") or None,
                              help="Find exact copies of missing selections here; defaults to the nearest view_selection directory")
    args = parser.parse_args()
    if args.command == "pool":
        pool = view_pool(HIERARCHICAL_POOL if args.hierarchical else None)
        if args.output:
            atomic_json(args.output, pool)
        print(json.dumps(pool, indent=2, ensure_ascii=False))
    else:
        root = artifact_root(args.sparse_root)
        if not (root / MANIFEST).is_file():
            parser.error(f"Missing adaptive completion manifest: {root / MANIFEST}")
        try:
            validate_index_selection(root, json.loads((root / MANIFEST).read_text(encoding="utf-8")),
                                     resolver=SelectionFileResolver(args.view_selection_root))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(str(exc))
        from qwen_sparse_index import build_sparse_index
        import subprocess
        try:
            build_sparse_index(root, threads=args.threads)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    main()
