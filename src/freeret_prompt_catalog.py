"""Dependency-light FreeRet prompt catalog, shared without loading model or dataset code."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

PROMPT_VERSION = "freeret_official_dataset_prompts_v3"

IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"


@dataclass(frozen=True)
class FreeRetPromptSpec:
    """The six prompt components used by official FreeRet INPUT_FROMAT."""

    task_description: str
    target_input: Callable[[str], str]
    query_input: Callable[[str], str]
    target_instruction: str
    query_instruction: str
    rerank_input: Callable[[str, str], str]
    rerank_image_order: tuple[str, ...]
    max_pixels: int
    source: str


def _mscoco_t2i_spec(source: str) -> FreeRetPromptSpec:
    return FreeRetPromptSpec(
        task_description="You are required to determine if the provided caption matches the given image.",
        target_input=lambda _text: f"<photo>: {IMAGE_TOKEN}\n",
        query_input=lambda text: f"<caption>: {text}\n",
        target_instruction=(
            "Based on the given <image>, output exactly one word that follows these rules:\n"
            " - Output a single lowercase noun that best captures the semantics of the <image>.\n"
            " - You may use world knowledge and paraphrase if needed.\n"
            " - The word must be a complete, standalone word — not a function word, preposition, symbol, or fragment.\n"
            " - Favor the caption of the <image> if possible.\n"
        ),
        query_instruction=(
            "Based on the given <caption>, output exactly one word that follows these rules:\n"
            " - Output a single lowercase noun that best captures the semantics of the <caption>.\n"
            " - You may use world knowledge and paraphrasing if helpful.\n"
            " - The word must be a complete, standalone word — not a function word, preposition, symbol, or fragment.\n"
            " - Favor the word of the <caption> if possible.\n"
        ),
        rerank_input=lambda _target_text, query_text: (
            f"{IMAGE_TOKEN}\n"
            "Question: Does the caption accurately describe the image?\n"
            f"Caption: {query_text}\n"
            "A. Yes\n"
            "B. No"
        ),
        rerank_image_order=("target",),
        max_pixels=256,
        source=source,
    )


def _mscoco_i2t_spec(source: str) -> FreeRetPromptSpec:
    return FreeRetPromptSpec(
        task_description="You are required to determine if the provided caption matches the given image.",
        target_input=lambda text: f"<caption>: {text}\n",
        query_input=lambda _text: f"<photo>: {IMAGE_TOKEN}\n",
        target_instruction=(
            "Based on the given <caption>, output exactly one word that follows these rules:\n"
            " - Output a single lowercase noun that best captures the semantics of the <caption>.\n"
            " - You may use world knowledge and paraphrasing if helpful.\n"
            " - The word must be a complete, standalone word — not a function word, preposition, symbol, or fragment.\n"
            " - Favor the word of the <caption> if possible.\n"
        ),
        query_instruction=(
            "Based on the given <image>, output exactly one word that follows these rules:\n"
            " - Output a single lowercase noun that best captures the semantics of the <image>.\n"
            " - You may use world knowledge and paraphrase if needed.\n"
            " - The word must be a complete, standalone word — not a function word, preposition, symbol, or fragment.\n"
            " - Favor the caption of the <image> if possible.\n"
        ),
        rerank_input=lambda target_text, _query_text: (
            f"{IMAGE_TOKEN}\n"
            f"Caption: {target_text}\n"
            "Question: Does the caption accurately describe the image?\n"
            "A. Yes, the caption matches the image content.\n"
            "B. No, the caption does not match the image content."
        ),
        rerank_image_order=("query",),
        max_pixels=256,
        source=source,
    )


def _tbpr_spec(dataset_name: str) -> FreeRetPromptSpec:
    return FreeRetPromptSpec(
        task_description=(
            "You are required to determine whether the person description matches "
            "the person image."
        ),
        target_input=lambda _text: f"<person image>: {IMAGE_TOKEN}\n",
        query_input=lambda text: f"<person description>: '{text}'\n",
        target_instruction=(
            "Based on the given <person image>, output exactly one word that follows these rules:\n"
            " - Output a single lowercase noun or adjective that best captures the person's visible appearance.\n"
            " - Focus on identity-relevant clothing, color, carried objects, age, or gender cues.\n"
            " - Do not use function words, prepositions, symbols, or incomplete fragments.\n"
            " - Prefer the most distinctive visible attribute.\n"
        ),
        query_instruction=(
            "Based on the given <person description>, output exactly one word that follows these rules:\n"
            " - Output a single lowercase noun or adjective that best captures the described person's appearance.\n"
            " - Focus on identity-relevant clothing, color, carried objects, age, or gender cues.\n"
            " - Do not use function words, prepositions, symbols, or incomplete fragments.\n"
            " - Prefer a distinctive word directly provided in the description.\n"
        ),
        rerank_input=lambda _target_text, query_text: (
            "You are required to determine whether the person image matches the person description.\n"
            f"<person image>: {IMAGE_TOKEN}\n"
            f"<person description>: '{query_text}'\n\n"
            "Multiple Choice Options:\n"
            "A. Yes, the visible person matches the described identity and appearance.\n"
            "B. No, the visible person does not match the description."
        ),
        rerank_image_order=("target",),
        max_pixels=512,
        source=(
            f"Project adaptation for {dataset_name}; the official FreeRet repository "
            "does not provide a TBPR prompt."
        ),
    )


def _webqa_spec() -> FreeRetPromptSpec:
    return FreeRetPromptSpec(
        task_description=(
            "You are required to determine if the provided image and information is "
            "related to the question.\n"
        ),
        target_input=lambda text: (
            f"<image>: {IMAGE_TOKEN}\n<information>: '{text}'\n\n"
        ),
        query_input=lambda text: f"<question>: '{text}'\n\n",
        target_instruction=(
            "Given the image and accompanying information, output exactly one word that follows these rules:\n"
            " - The word should best capture the overall meaning of information.\n"
            " - World knowledge may be used if helpful.\n"
            " - The word must be a complete, standalone word — not a function word, preposition, symbol, or fragment.\n"
            " - Prefer using a word directly provided in the given information."
        ),
        query_instruction=(
            "Given the question, output exactly one word that follows these rules:\n"
            " - The word should best capture the meaning of the main subjects mentioned in the question\n"
            " - Do not answer the question itself.\n"
            " - World knowledge may be used if helpful.\n"
            " - The word must be a complete, standalone word — not a function word, preposition, symbol, or fragment.\n"
            " - Prefer using a word directly provided in the question."
        ),
        rerank_input=lambda target_text, query_text: (
            "You are required to determine if the provided image and the "
            "accompanying information are relevant to the given question.\n"
            f"<image>: {IMAGE_TOKEN}\n"
            f"<information>: '{target_text}'\n\n"
            f"Question:\n{query_text}\n\n"
            "Multiple Choice Options:\n"
            "A. Yes, the image and the information directly relate to the question "
            "and are sufficient to provide an appropriate answer.\n"
            "B. No, the image and the information are not relevant enough, or they "
            "fail to address the question properly."
        ),
        rerank_image_order=("target",),
        max_pixels=256,
        source="FreeRet INPUT_FROMAT['WebQA'] (raw project text fields substituted directly).",
    )


def _remuq_spec() -> FreeRetPromptSpec:
    return FreeRetPromptSpec(
        task_description=(
            "You are required to determine whether the candidate passage contains "
            "knowledge relevant to the question about the image.\n"
        ),
        target_input=lambda text: f"<candidate passage>: '{text}'\n\n",
        query_input=lambda text: (
            f"<image>: {IMAGE_TOKEN}\n<question>: '{text}'\n\n"
        ),
        target_instruction=(
            "Given the candidate passage, output exactly one word that follows these rules:\n"
            " - The word should best capture the main entity or knowledge subject of the passage.\n"
            " - World knowledge and paraphrasing may be used if helpful.\n"
            " - The word must be a complete, standalone word — not a function word, preposition, symbol, or fragment.\n"
            " - Prefer using a word directly provided in the passage."
        ),
        query_instruction=(
            "Given the image and question, output exactly one word that follows these rules:\n"
            " - The word should best capture the main subject whose knowledge is required by the question.\n"
            " - Do not answer the question itself.\n"
            " - World knowledge may be used if helpful.\n"
            " - The word must be a complete, standalone word — not a function word, preposition, symbol, or fragment.\n"
            " - Prefer using a word directly provided in the question when possible."
        ),
        rerank_input=lambda target_text, query_text: (
            "You are required to determine whether a candidate passage provides "
            "relevant knowledge for answering a question about an image.\n"
            f"<image>: {IMAGE_TOKEN}\n"
            f"<question>: '{query_text}'\n\n"
            f"<candidate passage>: '{target_text}'\n\n"
            "Multiple Choice Options:\n"
            "A. Yes, the passage provides directly relevant knowledge for answering the question.\n"
            "B. No, the passage is irrelevant or does not provide the knowledge needed to answer the question."
        ),
        rerank_image_order=("query",),
        max_pixels=512,
        source=(
            "Project adaptation for UMRB-ReMuQ; the official FreeRet repository does "
            "not provide a ReMuQ prompt."
        ),
    )


def _llava_spec() -> FreeRetPromptSpec:
    return FreeRetPromptSpec(
        task_description=(
            "You are required to determine whether the provided answer correctly "
            "responds to the given question, based on the provided image."
        ),
        target_input=lambda text: f"<answer>: {text}\n",
        query_input=lambda text: (
            f"<image>: {IMAGE_TOKEN}\n<question>: {text}\n"
        ),
        target_instruction=(
            "Represent the given <answer> in one word following these rules:\n"
            " - Output a single word that best captures the semantics of the <answer>.\n"
            " - You may use world knowledge and paraphrase if needed.\n"
            " - Do not repeat function words, use prepositions, or symbols.\n"
            " - Favor using original words from <answer>.\n"
        ),
        query_instruction=(
            "Represent the given <image> and <question> in one word following these rules:\n"
            " - Output a single word that best captures the semantics of the <question>.\n"
            " - You may use world knowledge and paraphrasing if helpful.\n"
            " - Do not use function words, prepositions, or symbols.\n"
            " - Favor the answer of the <question>.\n"
        ),
        rerank_input=lambda target_text, query_text: (
            f"{IMAGE_TOKEN}\n"
            "You are given a question and a proposed answer.\n"
            "Your task is to evaluate whether the proposed answer correctly addresses the question.\n"
            f"Choose the best option below.\n\nQuestion:\n{query_text}\n\n"
            f"Proposed Answer:\n{target_text}\n\n"
            "Multiple Choice Options:\n"
            "A. The proposed answer is correct and complete.\n"
            "B. The proposed answer is partially correct, or incorrect, or irrelevant to the question."
        ),
        rerank_image_order=("query",),
        max_pixels=512,
        source=(
            "Project adaptation for UMRB-LLaVA using the official FreeRet "
            "OK-VQA image-question-to-answer prompt family."
        ),
    )


PROMPT_SPECS: dict[str, FreeRetPromptSpec] = {
    "coco_t2i": _mscoco_t2i_spec("FreeRet INPUT_FROMAT['MSCOCO_t2i']."),
    "coco_i2t": _mscoco_i2t_spec("FreeRet INPUT_FROMAT['MSCOCO_i2t']."),
    "flickr_t2i": _mscoco_t2i_spec(
        "Flickr adaptation of FreeRet INPUT_FROMAT['MSCOCO_t2i']; Flickr is absent from the official prompt table."
    ),
    "flickr_i2t": _mscoco_i2t_spec(
        "Flickr adaptation of FreeRet INPUT_FROMAT['MSCOCO_i2t']; Flickr is absent from the official prompt table."
    ),
    "cuhk_pedes_t2i": _tbpr_spec("CUHK-PEDES"),
    "icfg_pedes_t2i": _tbpr_spec("ICFG-PEDES"),
    "rstpreid_t2i": _tbpr_spec("RSTPReid"),
    "webqa_t2it": _webqa_spec(),
    "remuq_it2t": _remuq_spec(),
    "llava_it2t": _llava_spec(),
}


def get_prompt_spec(prompt_key: str) -> FreeRetPromptSpec:
    try:
        return PROMPT_SPECS[prompt_key]
    except KeyError as exc:
        raise ValueError(f"Unknown FreeRet prompt key: {prompt_key}") from exc
