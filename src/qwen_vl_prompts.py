"""Shared native Qwen2/Qwen2.5/Qwen3-VL prompts with Mistral task semantics.

Only the conversation/vision syntax changes. English instructions, input order,
and valid task/template combinations come from this project's Mistral baseline.
"""
from __future__ import annotations

import json
from pathlib import Path

import template as baseline

PROMPT_VERSION = "qwen25-prompteol-v1"
QWEN2_PROMPT_VERSION = "qwen2-prompteol-v1"
QWEN3_PROMPT_VERSION = "qwen3-prompteol-v1"
PROMPT_VERSIONS = {"qwen2_vl": QWEN2_PROMPT_VERSION, "qwen2_5_vl": PROMPT_VERSION,
                   "qwen3_vl": QWEN3_PROMPT_VERSION}
MODEL_TYPES = set(PROMPT_VERSIONS)
IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"
ASSISTANT = "<|im_start|>assistant\n"
TURN_END = "<|im_end|>\n" + ASSISTANT
METHODS = {("prompteol", "separate"), ("disassembleeol_separate", "concat")}


def model_family(config):
    """Read the outer multimodal type, despite Qwen2's text-config delegation.

    After a save/load roundtrip Qwen2VLConfig.model_type can resolve to the
    nested qwen2_vl_text instance field. Its config class retains qwen2_vl.
    """
    return getattr(type(config), "model_type", None) or getattr(config, "model_type", None)


def is_qwen25_model(model) -> bool:
    return model_family(getattr(model, "config", None)) == "qwen2_5_vl"


def is_native_qwen_model(model) -> bool:
    return model_family(getattr(model, "config", None)) in MODEL_TYPES


def is_native_qwen_checkpoint(path) -> bool:
    """Route by config, including renamed local checkpoints; exclude Qwen3 MoE."""
    config = Path(path) / "config.json"
    if config.is_file():
        model_type = json.loads(config.read_text(encoding="utf-8")).get("model_type", "")
    else:
        # Do not trigger a config download for existing non-Qwen launchers.
        if not any(name in str(path).lower() for name in ("qwen2-vl", "qwen2.5", "qwen3")):
            return False
        from transformers import AutoConfig
        model_type = model_family(AutoConfig.from_pretrained(path))
    if model_type.startswith("qwen3_vl") and model_type != "qwen3_vl":
        raise ValueError("Qwen3-VL retrieval supports Dense Instruct checkpoints only; MoE is unsupported")
    return model_type in MODEL_TYPES


def is_qwen25_checkpoint(path) -> bool:
    """Read local config first, including checkpoints whose directory was renamed."""
    config = Path(path) / "config.json"
    if config.is_file():
        return json.loads(config.read_text(encoding="utf-8")).get("model_type") == "qwen2_5_vl"
    # Existing non-Qwen launchers need no extra config download. Local renamed
    # checkpoints were already detected above; canonical Qwen Hub IDs are
    # verified through their configuration, regardless of parameter count.
    if "qwen2.5" not in str(path).lower():
        return False
    from transformers import AutoConfig
    return model_family(AutoConfig.from_pretrained(path)) == "qwen2_5_vl"


def render_user(processor, body: str) -> str:
    """Render native content in its existing order, without a second BOS/EOS pass."""
    parts = body.replace(IMAGE, "<image>").split("<image>")
    content = []
    for index, text in enumerate(parts):
        if index:
            content.append({"type": "image"})
        if text:
            content.append({"type": "text", "text": text})
    return processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True,
    )


def mistral_body(name: str) -> str:
    value = getattr(baseline, name)
    if not value.startswith("[INST]") or not value.endswith("[/INST]"):
        raise ValueError(f"Expected a Mistral template: {name}")
    return value[len("[INST]"):-len("[/INST]")]


def _views(modality, prompt_type):
    if prompt_type == baseline.RANDOM_VIEW_PROMPT_TYPE:
        return []  # resolved once with the tokenizer by retrieval_prompt
    if prompt_type == "prompt_5":
        name = f"retrieval_disassemble_{modality}_prompts_for_concat"
    elif prompt_type in {f"prompt_{n}" for n in range(1, 8)}:
        name = f"retrieval_disassemble_{modality}_prompts_{prompt_type[-1]}_for_concat"
    else:
        raise ValueError(f"Unsupported Qwen retrieval prompt_type={prompt_type!r}")
    return list(getattr(baseline, name))


def retrieval_prompt(record, task, data, processor, *, include_views=True):
    """Return shared context, whole-input instruction, and ordered aspect instructions."""
    image = record.get("image") is not None
    text = record.get("text", "")
    kind = "image" if image else "text"
    context = "<image>\n" if image else ""
    if text:
        context += text + "\n"
    whole = baseline.img_prompt_for_concat if image else baseline.text_prompt_for_concat
    if task == "ir":
        if not include_views:
            return context, whole, []
        views = _views(kind, data.prompt_type)
    elif task == "tbpr":
        tbpr_type = data.tbpr_type
        if tbpr_type == "origin_type":
            if not include_views:
                return context, whole, []
            views = list(getattr(baseline, f"retrieval_disassemble_{kind}_origin_prompts_person_retrieval_for_concat"))
        elif tbpr_type in {"type", "type_1", "type_2"}:
            suffix = "" if tbpr_type == "type" else "_" + tbpr_type[-1]
            whole = getattr(baseline, f"person_retrieval_{'img' if image else 'text'}_prompt_for_concat{suffix}")
            if not include_views:
                return context, whole, []
            if suffix:
                name = f"retrieval_disassemble_{kind}_prompts_person_retrieval_for_concat{suffix}"
            elif data.prompt_type == baseline.RANDOM_VIEW_PROMPT_TYPE:
                name = f"retrieval_disassemble_{kind}_prompts_person_retrieval_for_concat"
            else:
                number = "" if data.prompt_type == "prompt_5" else "_" + data.prompt_type.removeprefix("prompt_")
                name = f"retrieval_disassemble_{kind}_prompts_person_retrieval{number}_for_concat"
            if not hasattr(baseline, name):
                raise ValueError(f"No Mistral counterpart for TBPR {tbpr_type}/{data.prompt_type}")
            views = list(getattr(baseline, name))
        else:
            raise ValueError(f"Unsupported TBPR view type: {tbpr_type}")
    elif task in {"t2it", "it2t"}:
        multimodal = (task == "t2it" and image) or (task == "it2t" and image)
        whole = baseline.fusion_prompt_for_concat if multimodal else baseline.text_prompt_for_concat
        if not include_views:
            return context, whole, []
        role = "corpus" if (task == "t2it" and image) or (task == "it2t" and not image) else "query"
        dataset_tag = "llava_" if task == "it2t" and data.dataset_name == "llava" else ""
        views = list(getattr(baseline, f"retrieval_disassemble_{role}_prompts_{dataset_tag}{task}_retrieval_for_concat"))
    elif task == "cir":
        if include_views and data.cir_type == "classify_type":
            raise ValueError("CIR classify_type has no defined whole-input dense representation for disassembleeol_separate; use cir_type=type or type_1")
        dress = record.get("dress_type", "shirt")
        dress = "shirt" if dress == "toptee" else dress
        composed = bool(record.get("composed"))
        context = "<image>" + (f" change the style of this {dress} to {text}" if composed else "") + "\n"
        whole = (baseline.fashion_iq_composed_image_for_concat if composed else baseline.fashion_iq_img_prompt_for_concat).format(dress)
        if not include_views:
            return context, whole, []
        name = "retrieval_disassemble_" + ("composed_image" if composed else "image") + "_prompts_fashion_iq_for_concat"
        if data.cir_type != "type":
            name += "_1"
        views = [value.format(dress) for value in getattr(baseline, name)]
    else:
        raise ValueError(f"Unsupported Qwen retrieval task: {task}")
    if data.prompt_type == baseline.RANDOM_VIEW_PROMPT_TYPE:
        if task not in {"ir", "tbpr"}:
            raise ValueError("Random views have no Mistral counterpart for this task")
        _, text_views, image_views = baseline.build_random_view_prompts(
            processor.tokenizer, data.random_view_seed,
            person_retrieval=task == "tbpr" and data.tbpr_type == "type",
        )
        views = list(image_views if image else text_views)
    return context, whole, views


IR_POINTWISE = {
    "relevant": "mistral_relevant_prompt",
    "in_one_word_relevant": "mistral_in_one_word_relevant_prompt",
    "please_relevant": "mistral_please_relevant_prompt",
    "precise_caption": "mistral_precise_caption_prompt",
    "first_precise_caption": "mistral_first_precise_caption_prompt",
    "role_relevant": "mistral_role_relevant_prompt",
    "role_precise_caption": "mistral_role_precise_caption_prompt",
}
DIRECTED = {
    "old_relevant": "old_{direction}_query_relevant_prompt",
    "reverse_old_relevant": "old_{direction}_reverse_query_relevant_prompt",
    "query_relevant": "{direction}_query_relevant_prompt",
    "reverse_query_relevant": "{direction}_reverse_query_relevant_prompt",
    "origin_old_relevant": "origin_old_{direction}_query_relevant_prompt",
    "reverse_origin_old_relevant": "origin_old_{direction}_reverse_query_relevant_prompt",
    "role_old_relevant": "role_old_{direction}_query_relevant_prompt",
    "freeret": "freeret_{direction}_rerank_prompt",
}
CAPTION_SUFFIXES = {
    "caption_generation": "", "what_caption_generation": "_1",
    **{f"caption_generation_{n}": f"_{n}" for n in range(2, 6)},
}


def rerank_template(processor, task, query_type, mode, *, caption=False):
    """Select explicitly; never fall back to a different experiment's prompt."""
    if mode in {"caption_generation_6", "caption_generation_7"}:
        raise ValueError(f"{mode} inserts LLaVA BOS <s>; Qwen VL has no equivalent BOS experiment")
    name = None
    if caption:
        if task == "cir":
            raise ValueError("CIR caption-generation reranking has no implemented scoring definition")
        if task == "ir":
            if mode in CAPTION_SUFFIXES:
                name = "mistral_query_generation_paradigm_prompt" + CAPTION_SUFFIXES[mode]
            elif mode in {"detailed_caption_generation", "detailed_caption_generation_1"}:
                name = "detailed_mistral_query_generation_paradigm_prompt" + ("_1" if mode.endswith("_1") else "")
        elif task == "tbpr":
            suffix = {"caption_generation": "", "what_caption_generation": "_1", "describe_caption_generation": "_2"}.get(mode)
            if suffix is not None:
                name = "person_retrieval_mistral_query_generation_paradigm_prompt" + suffix
        elif task in {"t2it", "it2t"} and mode in {"caption_generation", "what_caption_generation"}:
            name = f"{task}_retrieval_mistral_query_generation_paradigm_prompt" + ("_1" if mode.startswith("what_") else "")
    elif task == "ir":
        name = IR_POINTWISE.get(mode)
        if mode in DIRECTED:
            name = "mistral_" + DIRECTED[mode].format(direction="image" if query_type == "image" else "text")
    elif task == "tbpr":
        suffix = {
            "relevant": "relevant", "old_relevant": "old_query_relevant", "query_relevant": "query_relevant",
            "origin_old_relevant": "origin_old_query_relevant", "reverse_old_relevant": "reverse_old_query_relevant",
            "reverse_query_relevant": "reverse_query_relevant", "reverse_origin_old_relevant": "reverse_origin_old_query_relevant",
            "please_relevant": "please_relevant", "in_one_word_relevant": "in_one_word_relevant",
            "precise_caption": "precise_caption", "first_precise_caption": "first_precise_caption",
            "role_relevant": "role_relevant", "role_precise_caption": "role_precise_caption", "role_old_relevant": "role_old_query_relevant",
        }.get(mode)
        if suffix:
            name = "mistral_person_retrieval_" + suffix + "_prompt"
        if mode == "freeret":
            name = "mistral_freeret_text_rerank_prompt"
    elif task in {"t2it", "it2t"}:
        suffix = {"relevant": "old_query_relevant", "old_relevant": "old_query_relevant", "query_relevant": "query_relevant", "origin_old_relevant": "origin_old_query_relevant"}.get(mode)
        if mode == "answer_relevant" and task == "it2t":
            suffix = "answer_relevant"
        if suffix:
            name = f"mistral_{task}_retrieval_{suffix}_prompt"
    elif task == "cir":
        suffix = {"relevant": "relevant", "old_relevant": "old_query_relevant", "query_relevant": "query_relevant", "origin_old_relevant": "origin_old_query_relevant", "please_relevant": "please_relevant", "in_one_word_relevant": "in_one_word_relevant", "role_relevant": "role_relevant", "role_old_relevant": "role_old_query_relevant"}.get(mode)
        if suffix:
            name = "mistral_fashion_iq_" + suffix + "_prompt"
    if name is None or not hasattr(baseline, name):
        raise ValueError(f"No Mistral/Qwen counterpart for task={task}, query_type={query_type}, rerank_template={mode}")
    body = mistral_body(name)
    if "<s>" in body or "<|begin_of_text|>" in body:
        raise ValueError(f"LLaVA-only special token in {mode}")
    return render_user(processor, body)
