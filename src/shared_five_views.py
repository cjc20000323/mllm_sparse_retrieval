"""The original Flickr/COCO five views, shared by concat retrieval tasks."""
from __future__ import annotations

import re

PROMPT_TYPE = "prompt_shared_5"
TASKS = {"ir", "tbpr", "t2it", "it2t"}
METHOD = ("disassembleeol_separate", "concat")


def enabled(data):
    return getattr(data, "prompt_type", None) == PROMPT_TYPE


def validate(prompt_type, task, eol_type, calculate_type):
    if prompt_type != PROMPT_TYPE:
        return
    if task not in TASKS:
        raise ValueError(f"{PROMPT_TYPE} supports ir/tbpr/t2it/it2t, got {task!r}")
    if (eol_type, calculate_type) != METHOD:
        raise ValueError(f"{PROMPT_TYPE} requires disassembleeol_separate + concat, got {eol_type} + {calculate_type}")


def validate_entry(model, data, task):
    if enabled(data):
        validate(data.prompt_type, task, model.eol_type, model.calculate_type)


def instructions(modality):
    import template
    if modality == "fusion":
        return [value.replace("above sentence", "above image and sentence")
                for value in template.retrieval_disassemble_text_prompts_for_concat]
    if modality not in {"text", "image"}:
        raise ValueError(f"Unknown shared-five input modality: {modality}")
    return list(getattr(template, f"retrieval_disassemble_{modality}_prompts_for_concat"))


def whole_instruction(modality):
    import template
    return getattr(template, {"text": "text_prompt_for_concat", "image": "img_prompt_for_concat",
                              "fusion": "fusion_prompt_for_concat"}[modality])


def record_modality(record, task):
    if task not in TASKS:
        raise ValueError(f"{PROMPT_TYPE} supports ir/tbpr/t2it/it2t, got {task!r}")
    if record.get("image") is not None:
        return "fusion" if task in {"t2it", "it2t"} else "image"
    return "text"


def template_override(name, dataset, input_type):
    """Resolve legacy template names without changing their module globals."""
    if name.startswith("retrieval_disassemble_") and "for_concat" in name:
        for kind in ("text", "image"):
            if name.startswith(f"retrieval_disassemble_{kind}_"):
                return instructions(kind)
        match = re.fullmatch(r"retrieval_disassemble_(query|corpus)_prompts_(?:llava_)?(t2it|it2t)_retrieval_for_concat", name)
        if match:
            role, task = match.groups()
            fusion = (task, role) in {("t2it", "corpus"), ("it2t", "query")}
            return instructions("fusion" if fusion else "text")
    match = re.fullmatch(r"person_retrieval_(img|text)_prompt_for_concat(?:_[12])?", name)
    if match:
        return whole_instruction("image" if match[1] == "img" else "text")
    if name in {"img_prompt_for_concat", "text_prompt_for_concat", "fusion_prompt_for_concat"}:
        if dataset == "webqa" and input_type in {"image", "corpus"}:
            return whole_instruction("fusion")
        if dataset in {"remuq", "llava"} and input_type in {"image", "query"}:
            return whole_instruction("fusion")
        return whole_instruction({"img_prompt_for_concat": "image", "text_prompt_for_concat": "text",
                                  "fusion_prompt_for_concat": "fusion"}[name])
    return None
