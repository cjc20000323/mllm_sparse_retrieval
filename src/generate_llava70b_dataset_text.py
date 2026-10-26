"""Generate LLaVA-70B descriptions or QA pairs in the formats used by dataset.py.

This is intentionally a standalone program.  It does not import or modify the
project's existing dataset/argument files, because the generated data is
written to a separate directory that mirrors the layouts expected by
``src/dataset.py``.

The program supports the datasets currently handled by dataset.py:

* image -> text: coco, flickr, fashion-iq, CUHK-PEDES, ICFG-PEDES, RSTPReid
* text -> image+text: webqa, edis
* image+text -> text: remuq, llava

For the mixed-modality datasets, the model is asked for a question and an
answer.  For the other datasets, it is asked for an image description.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm


IMAGE_DATASETS = (
    "coco",
    "flickr",
    "fashion-iq",
    "CUHK-PEDES",
    "ICFG-PEDES",
    "RSTPReid",
)
T2IT_DATASETS = ("webqa", "edis")
ALL_DATASETS = IMAGE_DATASETS + T2IT_DATASETS + (
    "remuq",
    "llava",
)

DATASET_DIR_NAMES = {
    "coco": "coco",
    "flickr": "flickr",
    "fashion-iq": "fashion-iq",
    "CUHK-PEDES": "CUHK-PEDES",
    "ICFG-PEDES": "ICFG-PEDES",
    "RSTPReid": "RSTPReid",
    "webqa": "WebQA",
    "edis": "EDIS",
    "remuq": "UMRB-ReMuQ",
    "llava": "UMRB-LLaVA",
}

PERSON_JSON_NAMES = {
    "CUHK-PEDES": "reid_raw.json",
    "ICFG-PEDES": "ICFG-PEDES.json",
    "RSTPReid": "data_captions.json",
}

T2IT_CORPUS_PARTS = {
    "webqa": 2,
    "edis": 4,
}


@dataclass
class ImageSample:
    """One sampled image and the metadata needed to write it back."""

    image_id: str
    image_path: Optional[Path] = None
    image_bytes: Optional[bytes] = None
    meta: dict[str, Any] = field(default_factory=dict)
    category: Optional[str] = None


@dataclass(frozen=True)
class FewShotExample:
    """One source image and its ground-truth assistant response."""

    sample: ImageSample
    response: str


@dataclass
class GenerationResult:
    """Parsed model output plus the unparsed text for auditing."""

    description: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    raw_output: str = ""
    qa_status: Optional[str] = None


@dataclass(frozen=True)
class QAPromptSpec:
    """Dataset-specific style and length targets for generated QA records."""

    prompt: str
    min_question_words: int
    max_question_words: int
    min_answer_words: int
    max_answer_words: int
    max_new_tokens: int


DEFAULT_QA_PROMPT_SPEC = QAPromptSpec(
    prompt=(
        "Inspect the image carefully and create one useful visual question and "
        "its answer for a cross-modal retrieval dataset. The question must be "
        "answerable from visible image content. Write a specific question of "
        "15-35 English words and an informative factual answer of 40-100 English "
        "words. Include the important people, objects, actions, relations, and "
        "scene details that are visible, but do not invent details. Return ONLY "
        "valid JSON in exactly this form: "
        '{"question": "...", "answer": "..."}'
    ),
    min_question_words=15,
    max_question_words=35,
    min_answer_words=40,
    max_answer_words=100,
    max_new_tokens=384,
)


QA_PROMPT_SPECS: dict[str, QAPromptSpec] = {
    "webqa": QAPromptSpec(
        prompt=(
            "Inspect the image carefully and create one WebQA-style multimodal "
            "question and answer. The question should ask about concrete visual "
            "content and contain 10-25 English words. The answer should read like "
            "a useful image-associated WebQA corpus entry rather than a one-word "
            "reply: write 30-80 English words in 1-4 factual sentences, covering "
            "the main subject, visible objects, actions, attributes, spatial "
            "relations, and scene context when they are clear. Keep the answer "
            "grounded in the image and do not invent names, text, or events. Return "
            "ONLY valid JSON in exactly this form: "
            '{"question": "...", "answer": "..."}'
        ),
        min_question_words=10,
        max_question_words=25,
        min_answer_words=30,
        max_answer_words=80,
        max_new_tokens=256,
    ),
    "remuq": QAPromptSpec(
        prompt=(
            "Inspect the image carefully and create one ReMuQ-style multimodal "
            "question and answer. The question should be a specific, image-grounded "
            "information need of 12-30 English words, combining the visible scene "
            "with a meaningful request for information. The answer should be a "
            "self-contained factual knowledge passage of 40-90 English words in "
            "1-4 sentences, suitable as a text retrieval corpus item. Explain the "
            "relevant subject, attributes, actions, or context rather than replying "
            "with a short label. Stay faithful to visible evidence and do not invent "
            "unsupported facts. Return ONLY valid JSON in exactly this form: "
            '{"question": "...", "answer": "..."}'
        ),
        min_question_words=12,
        max_question_words=30,
        min_answer_words=40,
        max_answer_words=90,
        max_new_tokens=320,
    ),
    "llava": QAPromptSpec(
        prompt=(
            "Inspect the image carefully and create one LLaVA-style visual question "
            "and answer. The question should be a natural, specific image question "
            "of 10-25 English words. The answer must be a detailed, explanatory "
            "response of 50-110 English words in 2-5 sentences. Describe the visible "
            "evidence, relevant objects and people, actions, relationships, and the "
            "reasoning needed to answer the question. Answer directly, remain "
            "grounded in the image, and do not hallucinate unreadable text or hidden "
            "facts. Return ONLY valid JSON in exactly this form: "
            '{"question": "...", "answer": "..."}'
        ),
        min_question_words=10,
        max_question_words=25,
        min_answer_words=50,
        max_answer_words=110,
        max_new_tokens=384,
    ),
}


LLAVA_NEXT_DETAILED_CAPTION_PROMPT = (
    "Describe this image in detail with GPT-T-COCO format."
)
LLAVA_NEXT_ONE_SENTENCE_CAPTION_PROMPT = (
    "Provide a one-sentence caption for the provided image."
)
LLAVA_NEXT_DESCRIPTION_PROMPTS: dict[str, str] = {
    "coco": LLAVA_NEXT_DETAILED_CAPTION_PROMPT,
    "flickr": LLAVA_NEXT_ONE_SENTENCE_CAPTION_PROMPT,
    "fashion-iq": LLAVA_NEXT_ONE_SENTENCE_CAPTION_PROMPT,
    "CUHK-PEDES": LLAVA_NEXT_DETAILED_CAPTION_PROMPT,
    "ICFG-PEDES": LLAVA_NEXT_DETAILED_CAPTION_PROMPT,
    "RSTPReid": LLAVA_NEXT_DETAILED_CAPTION_PROMPT,
}


LLAVA_NEXT_QA_PROMPT_SPECS: dict[str, QAPromptSpec] = {
    "webqa": QAPromptSpec(
        prompt=(
            "Create one visual question-and-answer pair in the style of the "
            "natural COCO visual conversations used to train LLaVA-NeXT. Ask "
            "one specific question about a directly visible object, attribute, "
            "count, action, or spatial relation in 10-25 English words. Do not "
            "include meta-instructions such as 'Answer the question with "
            "GPT-T-COCO format.' in the generated question. Give a direct, "
            "image-grounded answer of 30-80 English words in 1-4 factual "
            "sentences. Do not invent names, unreadable text, or events. Return "
            "ONLY valid JSON in exactly this form: "
            '{"question": "...", "answer": "..."}'
        ),
        min_question_words=10,
        max_question_words=25,
        min_answer_words=30,
        max_answer_words=80,
        max_new_tokens=256,
    ),
    "remuq": QAPromptSpec(
        prompt=(
            "Create one image-grounded question-and-answer pair in the style of "
            "LLaVA-NeXT visual instruction data, adapted to a ReMuQ retrieval "
            "sample. Write a natural information-seeking question of 12-30 "
            "English words that starts from a clearly visible subject, action, "
            "attribute, or scene. Do not include meta-instructions such as "
            "'Answer the question with GPT-T-COCO format.' in the generated "
            "question. Write a self-contained answer of 40-90 English words in "
            "1-4 sentences that is useful as a retrieval passage while remaining "
            "faithful to the visible evidence. Do not invent unsupported facts. "
            "Return ONLY valid JSON in exactly this form: "
            '{"question": "...", "answer": "..."}'
        ),
        min_question_words=12,
        max_question_words=30,
        min_answer_words=40,
        max_answer_words=90,
        max_new_tokens=320,
    ),
    "llava": QAPromptSpec(
        prompt=(
            "Create one LLaVA-NeXT-style visual instruction and assistant "
            "response. The user question should naturally request detailed "
            "visual understanding or visual reasoning about the image and contain "
            "10-25 English words. Do not include meta-instructions such as "
            "'Answer the question with GPT-T-COCO format.' in the generated "
            "question. Give a helpful, detailed, and direct answer of 50-110 "
            "English words in 2-5 sentences, grounded in visible people, objects, "
            "actions, relationships, and scene evidence. Do not hallucinate "
            "unreadable text or hidden facts. Return ONLY valid JSON in exactly "
            "this form: "
            '{"question": "...", "answer": "..."}'
        ),
        min_question_words=10,
        max_question_words=25,
        min_answer_words=50,
        max_answer_words=110,
        max_new_tokens=384,
    ),
}


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _normalise_id(value: Any) -> str:
    """Keep IDs stable when pandas/pyarrow returns integer-like values."""

    text = _safe_text(value).strip()
    if re.fullmatch(r"[-+]?\d+\.0", text):
        return text[:-2]
    return text


def _safe_filename_part(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value[:160] or "image"


def _json_ready(value: Any) -> Any:
    """Convert common numpy/pyarrow scalar values to JSON-compatible values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": "bytes", "length": len(value)}
    if hasattr(value, "as_py"):
        try:
            return _json_ready(value.as_py())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return str(value)


def _unpack_arrow_value(value: Any) -> Any:
    if hasattr(value, "as_py"):
        try:
            return value.as_py()
        except Exception:
            return value
    return value


def _path_from_value(value: Any, roots: Iterable[Path]) -> Optional[Path]:
    value = _unpack_arrow_value(value)
    if isinstance(value, Path):
        candidates = [value]
    elif isinstance(value, str) and value.strip():
        candidates = [Path(value.strip())]
    else:
        return None

    for candidate in candidates:
        if candidate.is_file():
            return candidate
        if not candidate.is_absolute():
            for root in roots:
                resolved = root / candidate
                if resolved.is_file():
                    return resolved
    return None


def _image_value_to_bytes(
    value: Any,
    roots: Iterable[Path],
) -> tuple[Optional[bytes], Optional[Path]]:
    """Extract image bytes from a parquet value or resolve a path value."""

    value = _unpack_arrow_value(value)

    if value is None:
        return None, None
    if isinstance(value, float) and math.isnan(value):
        return None, None

    if isinstance(value, dict):
        for key in ("bytes", "data", "contents", "content"):
            if key in value and value[key] is not None:
                raw = value[key]
                raw = _unpack_arrow_value(raw)
                if isinstance(raw, (bytes, bytearray, memoryview)):
                    return bytes(raw), None
        for key in ("path", "filename", "file_name", "url"):
            if key in value:
                path = _path_from_value(value[key], roots)
                if path is not None:
                    return path.read_bytes(), path
        return None, None

    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value), None

    if isinstance(value, Image.Image):
        buffer = io.BytesIO()
        value.convert("RGB").save(buffer, format="PNG")
        return buffer.getvalue(), None

    path = _path_from_value(value, roots)
    if path is not None:
        return path.read_bytes(), path

    # A few dataset versions return an object with .bytes or .path attributes.
    raw = getattr(value, "bytes", None)
    if isinstance(raw, (bytes, bytearray, memoryview)):
        return bytes(raw), None
    path = _path_from_value(getattr(value, "path", None), roots)
    if path is not None:
        return path.read_bytes(), path

    return None, None


def _image_from_sample(sample: ImageSample) -> Image.Image:
    if sample.image_bytes is not None:
        return Image.open(io.BytesIO(sample.image_bytes)).convert("RGB")
    if sample.image_path is None or not sample.image_path.is_file():
        raise FileNotFoundError(
            f"找不到样本 {sample.image_id} 的图像文件: {sample.image_path}。"
            "请检查 --source_root，或保留 parquet 中的 image.bytes。"
        )
    return Image.open(sample.image_path).convert("RGB")


class Llava70BGenerator:
    """Small compatibility wrapper for several HuggingFace LLaVA variants."""

    DESCRIPTION_PROMPT = (
        "Inspect the image carefully. Write one concise, factual English "
        "description of the visible content. Mention the main people, objects, "
        "actions, and scene when they are clear. Do not invent details. "
        "Return only the description, with no heading or quotation marks."
    )

    def __init__(self, args: argparse.Namespace):
        try:
            import transformers
        except ImportError as exc:  # pragma: no cover - depends on runtime env
            raise RuntimeError(
                "生成程序需要 transformers，请在运行脚本的环境中安装项目依赖。"
            ) from exc

        self.args = args
        processor_classes: list[Any] = []
        for class_name in (
            "AutoProcessor",
            "LlavaNextProcessor",
            "LlavaProcessor",
        ):
            candidate = getattr(transformers, class_name, None)
            if candidate is not None and candidate not in processor_classes:
                processor_classes.append(candidate)

        processor_errors: list[str] = []
        for processor_class in processor_classes:
            try:
                self.processor = processor_class.from_pretrained(
                    args.model_name_or_path,
                    trust_remote_code=True,
                )
                break
            except Exception as exc:  # try the next compatible processor
                processor_errors.append(f"{processor_class.__name__}: {exc}")
        else:
            joined = "\n".join(processor_errors[-3:])
            raise RuntimeError(
                "无法加载 LLaVA processor，请确认模型目录完整。尝试结果：\n" + joined
            )

        self.model = self._load_model()
        self.model.eval()
        self._cache: dict[tuple[str, str], GenerationResult] = {}

    def _torch_dtype(self) -> torch.dtype:
        if self.args.dtype == "float32":
            return torch.float32
        if self.args.dtype == "bfloat16":
            return torch.bfloat16
        return torch.float16

    def _load_model(self) -> Any:
        """Try the standard HF vision-to-sequence class, then LLaVA fallbacks."""

        try:
            import transformers
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("未安装 transformers。") from exc

        classes: list[Any] = []
        for class_name in (
            "AutoModelForVision2Seq",
            "LlavaNextForConditionalGeneration",
            "LlavaForConditionalGeneration",
            "AutoModelForCausalLM",
        ):
            candidate = getattr(transformers, class_name, None)
            if candidate is not None and candidate not in classes:
                classes.append(candidate)

        load_kwargs: dict[str, Any] = {
            "torch_dtype": self._torch_dtype(),
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if self.args.device_map and self.args.device_map.lower() != "none":
            load_kwargs["device_map"] = self.args.device_map

        errors: list[str] = []
        for model_class in classes:
            try:
                return model_class.from_pretrained(
                    self.args.model_name_or_path,
                    **load_kwargs,
                )
            except Exception as exc:  # try the next compatible HF class
                errors.append(f"{model_class.__name__}: {exc}")

        joined = "\n".join(errors[-4:])
        raise RuntimeError(
            "无法加载 LLaVA 模型。请确认 --model_name_or_path 指向实际的 "
            f"LLaVA-70B 检查点，并检查 transformers 版本。尝试结果：\n{joined}"
        )

    def _model_device(self) -> torch.device:
        device = getattr(self.model, "device", None)
        if isinstance(device, torch.device) and device.type != "meta":
            return device
        for parameter in self.model.parameters():
            if parameter.device.type != "meta":
                return parameter.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _user_turn(prompt: str) -> dict[str, Any]:
        return {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }

    @staticmethod
    def _text_user_turn(prompt: str) -> dict[str, Any]:
        """Create a user turn without an image placeholder."""
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }

    def _processor_inputs(
        self,
        image: Image.Image,
        prompt: str,
        few_shot_examples: Optional[list[FewShotExample]] = None,
        few_shot_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        examples = few_shot_examples or []
        example_prompt = few_shot_prompt or prompt
        few_shot_text_only = bool(
            examples and getattr(self.args, "few_shot_text_only", False)
        )
        conversation: list[dict[str, Any]] = []
        images: list[Image.Image] = []
        for example in examples:
            conversation.append(
                self._text_user_turn(example_prompt)
                if few_shot_text_only
                else self._user_turn(example_prompt)
            )
            conversation.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": example.response},
                    ],
                }
            )
            if not few_shot_text_only:
                images.append(_image_from_sample(example.sample))
        conversation.append(self._user_turn(prompt))
        images.append(image)

        # In text-only few-shot mode the target image is the only image sent to
        # the processor.  Otherwise preserve the original multimodal few-shot
        # behavior and the original single-image path when disabled.
        if few_shot_text_only:
            processor_images: Any = image
        else:
            processor_images = images if examples else image

        apply_chat_template = getattr(self.processor, "apply_chat_template", None)
        if apply_chat_template is not None:
            try:
                text = apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return dict(
                    self.processor(
                        text=text,
                        images=processor_images,
                        return_tensors="pt",
                    )
                )
            except Exception:
                # Older LlavaProcessor versions do not implement the same
                # chat-template signature.  The conventional prompt below is
                # accepted by those versions.
                pass

        fallback_parts: list[str] = []
        for example in examples:
            if few_shot_text_only:
                fallback_parts.append(
                    f"USER: {example_prompt}\nASSISTANT: {example.response}"
                )
            else:
                fallback_parts.append(
                    f"USER: <image>\n{example_prompt}\nASSISTANT: {example.response}"
                )
        fallback_parts.append(f"USER: <image>\n{prompt}\nASSISTANT:")
        fallback_prompt = "\n".join(fallback_parts)
        return dict(
            self.processor(
                text=fallback_prompt,
                images=processor_images,
                return_tensors="pt",
            )
        )

    def _generate_raw(
        self,
        sample: ImageSample,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        few_shot_examples: Optional[list[FewShotExample]] = None,
        few_shot_prompt: Optional[str] = None,
    ) -> str:
        image = _image_from_sample(sample)
        inputs = self._processor_inputs(
            image,
            prompt,
            few_shot_examples,
            few_shot_prompt,
        )
        device = self._model_device()
        for key, value in list(inputs.items()):
            if torch.is_tensor(value):
                inputs[key] = value.to(device)

        generation_limit = self.args.max_new_tokens
        if max_new_tokens is not None:
            generation_limit = min(generation_limit, max_new_tokens)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": generation_limit,
            "do_sample": self.args.temperature > 0,
        }
        if self.args.temperature > 0:
            generation_kwargs["temperature"] = self.args.temperature
            generation_kwargs["top_p"] = self.args.top_p

        tokenizer = getattr(self.processor, "tokenizer", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            generation_kwargs["pad_token_id"] = eos_token_id

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        input_ids = inputs.get("input_ids")
        if torch.is_tensor(input_ids) and output_ids.ndim == 2:
            prompt_length = input_ids.shape[-1]
            output_ids = output_ids[:, prompt_length:]

        if hasattr(self.processor, "batch_decode"):
            text = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
            )[0]
        else:
            if tokenizer is None:
                raise RuntimeError("processor 没有 batch_decode，也没有 tokenizer。")
            text = tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
            )[0]
        return text.strip()

    @staticmethod
    def _clean_description(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^description\s*:\s*", "", text, flags=re.IGNORECASE)
        if len(text) >= 2 and text[0] == text[-1] == '"':
            text = text[1:-1].strip()
        return text

    @classmethod
    def _parse_qa_candidate(cls, text: str) -> Optional[tuple[str, str]]:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        # First try every possible opening brace instead of relying on the
        # last closing brace.  This tolerates leading assistant text and
        # trailing explanations after an otherwise valid JSON object.
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", cleaned):
            try:
                candidate, _ = decoder.raw_decode(cleaned[match.start():])
            except json.JSONDecodeError:
                continue
            if not isinstance(candidate, dict):
                continue
            question = _safe_text(
                candidate.get("question") or candidate.get("Question")
            ).strip()
            answer = _safe_text(
                candidate.get("answer") or candidate.get("Answer")
            ).strip()
            if question and answer:
                return question, answer

        # Models sometimes emit a Python-like object, a truncated JSON object,
        # or labelled multi-line text.  Extract fields independently as a
        # second pass so a missing final quote/brace does not discard all text.
        field_matches = list(
            re.finditer(
                r"(?is)[\"']?\b(question|answer)\b[\"']?\s*[:：]\s*",
                cleaned,
            )
        )
        fields: dict[str, str] = {}
        for index, match in enumerate(field_matches):
            key = match.group(1).lower()
            end = field_matches[index + 1].start() if index + 1 < len(field_matches) else len(cleaned)
            value = cls._clean_qa_field(cleaned[match.end():end])
            if value and (key not in fields or len(value) > len(fields[key])):
                fields[key] = value
        if fields.get("question") and fields.get("answer"):
            return fields["question"], fields["answer"]
        return None

    @staticmethod
    def _clean_qa_field(value: str) -> str:
        value = value.strip()
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
            value = value[1:-1].strip()
        value = re.sub(r"[\"']?\s*[,}]\s*$", "", value).strip()
        return value

    @classmethod
    def _fallback_qa(cls, text: str) -> tuple[str, str]:
        cleaned = cls._clean_qa_field(text)
        cleaned = re.sub(r"^assistant\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()
        question = (
            "What visual content is shown in the image, and what are the main "
            "objects, people, actions, and scene details?"
        )
        answer = cleaned or "The model did not return a structured answer for this image."
        return question, answer

    @classmethod
    def _parse_qa(cls, text: str) -> tuple[str, str]:
        parsed = cls._parse_qa_candidate(text)
        if parsed is not None:
            return parsed
        return cls._fallback_qa(text)

    @staticmethod
    def _word_count(text: str) -> int:
        """Count English words without treating punctuation as content."""

        return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text))

    @classmethod
    def _qa_meets_minimum(
        cls,
        qa: Optional[tuple[str, str]],
        spec: QAPromptSpec,
    ) -> bool:
        if qa is None:
            return False
        question, answer = qa
        return (
            cls._word_count(question) >= spec.min_question_words
            and cls._word_count(answer) >= spec.min_answer_words
        )

    @staticmethod
    def _expansion_prompt(
        spec: QAPromptSpec,
        qa: Optional[tuple[str, str]],
    ) -> str:
        draft = {
            "question": qa[0] if qa is not None else "",
            "answer": qa[1] if qa is not None else "",
        }
        draft_json = json.dumps(draft, ensure_ascii=False)
        return (
            "Inspect the image again and repair or expand the draft visual QA "
            "below. Preserve all factual information that is supported by the "
            "image. Expand only the short or vague fields, and do not add details "
            "that cannot be seen. The question should contain at least "
            f"{spec.min_question_words} English words and the answer should contain "
            f"at least {spec.min_answer_words} English words. Use the following "
            "dataset-specific style requirements:\n"
            f"{spec.prompt}\n"
            f"Draft JSON: {draft_json}\n"
            "Return ONLY valid JSON in exactly this form: "
            '{"question": "...", "answer": "..."}'
        )

    @staticmethod
    def _qa_log_value(qa: Optional[tuple[str, str]]) -> str:
        if qa is None:
            return "invalid JSON"
        return f"question={len(qa[0].split())} words, answer={len(qa[1].split())} words"

    def _use_llava_next_training_prompts(self) -> bool:
        return bool(
            getattr(self.args, "use_llava_next_training_prompts", False)
        )

    def _description_prompt(self, dataset: str) -> str:
        if not self._use_llava_next_training_prompts():
            return self.DESCRIPTION_PROMPT
        return LLAVA_NEXT_DESCRIPTION_PROMPTS.get(
            dataset,
            self.DESCRIPTION_PROMPT,
        )

    def _qa_spec(self, dataset: str) -> QAPromptSpec:
        if self._use_llava_next_training_prompts():
            return LLAVA_NEXT_QA_PROMPT_SPECS.get(
                dataset,
                QA_PROMPT_SPECS.get(dataset, DEFAULT_QA_PROMPT_SPEC),
            )
        return QA_PROMPT_SPECS.get(dataset, DEFAULT_QA_PROMPT_SPEC)

    def _finalize_qa(
        self,
        dataset: str,
        sample: ImageSample,
        raw: str,
        qa: Optional[tuple[str, str]],
        spec: QAPromptSpec,
        qa_status: str,
    ) -> GenerationResult:
        if qa is None:
            qa = self._fallback_qa(raw)
            qa_status = "fallback"
        if not self._qa_meets_minimum(qa, spec):
            tqdm.write(
                f"[warning] {dataset} sample {sample.image_id} remains below "
                f"the target length after fallback: {self._qa_log_value(qa)}",
                file=sys.stderr,
            )
        return GenerationResult(
            question=qa[0],
            answer=qa[1],
            raw_output=raw,
            qa_status=qa_status,
        )

    @staticmethod
    def _few_shot_signature(
        few_shot_examples: Optional[list[FewShotExample]],
    ) -> str:
        examples = few_shot_examples or []
        if not examples:
            return "zero_shot"
        digest = hashlib.sha1()
        for example in examples:
            digest.update(example.sample.image_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(example.response.encode("utf-8"))
            digest.update(b"\0")
        return f"few_shot_{len(examples)}_{digest.hexdigest()}"

    def make_qa(
        self,
        sample: ImageSample,
        dataset: str,
        few_shot_examples: Optional[list[FewShotExample]] = None,
    ) -> GenerationResult:
        spec = self._qa_spec(dataset)
        prompt_mode = (
            "llava_next_training"
            if self._use_llava_next_training_prompts()
            else "existing"
        )
        few_shot_signature = self._few_shot_signature(few_shot_examples)
        key = self._cache_key(
            sample,
            f"qa:{prompt_mode}:{dataset}:{few_shot_signature}",
        )
        if key in self._cache:
            return self._cache[key]

        raw = self._generate_raw(
            sample,
            spec.prompt,
            spec.max_new_tokens,
            few_shot_examples,
        )
        qa = self._parse_qa_candidate(raw)
        if self._qa_meets_minimum(qa, spec):
            result = self._finalize_qa(
                dataset, sample, raw, qa, spec, "success"
            )
            self._cache[key] = result
            return result

        tqdm.write(
            f"[qa-length] {dataset} sample {sample.image_id}: first attempt "
            f"{self._qa_log_value(qa)}; asking the model to expand",
            file=sys.stderr,
        )
        expansion_source = qa
        expansion_raw = self._generate_raw(
            sample,
            self._expansion_prompt(spec, expansion_source),
            spec.max_new_tokens,
            few_shot_examples,
            spec.prompt,
        )
        expansion_qa = self._parse_qa_candidate(expansion_raw)
        if expansion_qa is not None:
            expansion_status = (
                "expansion_success"
                if self._qa_meets_minimum(expansion_qa, spec)
                else "expansion_short"
            )
            result = self._finalize_qa(
                dataset,
                sample,
                expansion_raw,
                expansion_qa,
                spec,
                expansion_status,
            )
        elif qa is not None:
            result = self._finalize_qa(
                dataset, sample, raw, qa, spec, "initial_short"
            )
        else:
            fallback_raw = expansion_raw or raw
            fallback_qa = self._fallback_qa(fallback_raw)
            tqdm.write(
                f"[qa-fallback] {dataset} sample {sample.image_id}: "
                "both attempts were not parseable; using fallback QA and continuing",
                file=sys.stderr,
            )
            result = self._finalize_qa(
                dataset,
                sample,
                fallback_raw,
                fallback_qa,
                spec,
                "fallback",
            )
        self._cache[key] = result
        return result

    def _cache_key(self, sample: ImageSample, task: str) -> tuple[str, str]:
        if sample.image_bytes is not None:
            identity = hashlib.sha1(sample.image_bytes).hexdigest()
        elif sample.image_path is not None:
            identity = str(sample.image_path.resolve())
        else:
            identity = sample.image_id
        return identity, task

    def describe(
        self,
        sample: ImageSample,
        dataset: str,
        few_shot_examples: Optional[list[FewShotExample]] = None,
    ) -> GenerationResult:
        prompt_mode = (
            "llava_next_training"
            if self._use_llava_next_training_prompts()
            else "existing"
        )
        few_shot_signature = self._few_shot_signature(few_shot_examples)
        key = self._cache_key(
            sample,
            f"description:{prompt_mode}:{dataset}:{few_shot_signature}",
        )
        if key not in self._cache:
            raw = self._generate_raw(
                sample,
                self._description_prompt(dataset),
                few_shot_examples=few_shot_examples,
            )
            self._cache[key] = GenerationResult(
                description=self._clean_description(raw),
                raw_output=raw,
            )
        return self._cache[key]


class DatasetGenerator:
    def __init__(self, args: argparse.Namespace, model: Llava70BGenerator):
        self.args = args
        self.model = model
        self.source_prefix = Path(args.source_root).expanduser().resolve()
        self.output_root = Path(args.output_root).expanduser().resolve()
        self.rng = random.Random(args.seed)
        self.use_few_shot_examples = bool(
            getattr(args, "use_few_shot_examples", False)
        )
        self.few_shot_num = int(getattr(args, "few_shot_num", 2))
        self.few_shot_split = _safe_text(
            getattr(args, "few_shot_split", None) or args.split
        ).lower()
        self._few_shot_by_dataset: dict[str, list[FewShotExample]] = {}
        self.manifest: dict[str, Any] = {
            "model_name_or_path": args.model_name_or_path,
            "split": args.split,
            "num_images_requested_per_dataset": args.num_images,
            "seed": args.seed,
            "datasets": {},
        }
        if self.use_few_shot_examples:
            self.manifest.update(
                {
                    "use_few_shot_examples": True,
                    "few_shot_num": self.few_shot_num,
                    "few_shot_split": self.few_shot_split,
                    "few_shot_text_only": bool(
                        getattr(args, "few_shot_text_only", False)
                    ),
                }
            )

    def _data_dir(self, dataset: str) -> Path:
        return self.source_prefix / "data" / DATASET_DIR_NAMES[dataset]

    def _output_data_dir(self, dataset: str) -> Path:
        return self.output_root / "data" / DATASET_DIR_NAMES[dataset]

    def _sample(self, samples: list[ImageSample]) -> list[ImageSample]:
        if not samples:
            raise RuntimeError("没有找到可以抽样的图像记录。")
        unique: dict[str, ImageSample] = {}
        for sample in samples:
            unique.setdefault(sample.image_id, sample)
        values = list(unique.values())
        count = min(self.args.num_images, len(values))
        if count < self.args.num_images:
            print(
                f"[warning] {len(values)} 个唯一图像少于请求的 "
                f"{self.args.num_images} 个，将全部使用。",
                file=sys.stderr,
            )
        return self.rng.sample(values, count)

    def _csv_samples(
        self,
        dataset: str,
        split: Optional[str] = None,
    ) -> tuple[list[ImageSample], Optional[list[str]]]:
        base = self._data_dir(dataset)
        source_split = split or self.args.split
        path = base / f"{dataset}_{source_split}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"找不到 {dataset} 的划分文件: {path}")

        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        if not rows:
            raise RuntimeError(f"文件为空: {path}")

        header: Optional[list[str]] = None
        start = 0
        if rows[0] and rows[0][0].strip().lower() == "imgid":
            header = rows[0]
            start = 1

        samples: list[ImageSample] = []
        for row in rows[start:]:
            if not row:
                continue
            if dataset == "coco":
                if len(row) < 5:
                    raise ValueError(f"COCO 行字段不足 5 列: {row}")
                image_dir, image_name = row[1], row[2]
                image_path = Path(image_dir) / image_name
                if not image_path.is_absolute():
                    image_path = base / image_path
            else:
                if len(row) < 4:
                    raise ValueError(f"Flickr 行字段不足 4 列: {row}")
                image_name = row[1]
                image_path = Path(image_name)
                if not image_path.is_absolute():
                    image_path = base / "flickr30k-images" / image_path

            image_id = _normalise_id(row[0])
            samples.append(
                ImageSample(
                    image_id=image_id,
                    image_path=image_path,
                    meta={
                        "row": row,
                        "header": header,
                        "image_dir": row[1] if dataset == "coco" else None,
                        "image_name": image_name,
                    },
                )
            )
        return samples, header

    def _person_samples(
        self,
        dataset: str,
        split: Optional[str] = None,
    ) -> list[ImageSample]:
        base = self._data_dir(dataset)
        path = base / PERSON_JSON_NAMES[dataset]
        if not path.is_file():
            raise FileNotFoundError(f"找不到 {dataset} 的标注文件: {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            data = data.get("data", data.get("annotations", []))
        if not isinstance(data, list):
            raise ValueError(f"无法识别 {path} 中的 JSON 列表格式。")

        requested_split = (split or self.args.split).lower()
        source_split = requested_split
        if dataset == "ICFG-PEDES" and requested_split in {"val", "valid", "validation", "dev"}:
            has_requested_split = any(
                isinstance(item, dict)
                and _safe_text(item.get("split")).lower() == requested_split
                for item in data
            )
            if not has_requested_split:
                # ICFG-PEDES commonly provides only train/test.  The existing
                # retrieval code uses train as its original validation source;
                # sample train here but write the generated records as val.
                source_split = "train"
                print(
                    "[warning] ICFG-PEDES has no validation split; sampling "
                    "its train split for generated validation data.",
                    file=sys.stderr,
                )

        samples: list[ImageSample] = []
        seen_paths: set[str] = set()
        for item in data:
            if not isinstance(item, dict):
                continue
            item_split = _safe_text(item.get("split")).lower()
            if item_split and item_split != source_split:
                continue
            relative = item.get("file_path") if dataset != "RSTPReid" else item.get("img_path")
            if not relative:
                continue
            relative = _safe_text(relative).replace("\\", "/")
            if relative in seen_paths:
                continue
            seen_paths.add(relative)
            samples.append(
                ImageSample(
                    image_id=_normalise_id(item.get("id", relative)),
                    image_path=base / "imgs" / relative,
                    meta=copy.deepcopy(item),
                )
            )
        return samples

    def _fashion_samples(self) -> list[ImageSample]:
        base = self._data_dir("fashion-iq")
        samples: list[ImageSample] = []
        for category in ("dress", "shirt", "toptee"):
            split_path = (
                base
                / "image_splits-20220326T130551Z-001"
                / "image_splits"
                / f"split.{category}.{self.args.split}.json"
            )
            if not split_path.is_file():
                raise FileNotFoundError(f"找不到 Fashion-IQ 划分文件: {split_path}")
            with split_path.open("r", encoding="utf-8") as handle:
                image_ids = json.load(handle)
            for raw_id in image_ids:
                image_id = _normalise_id(raw_id)
                image_name = image_id if image_id.lower().endswith(".png") else f"{image_id}.png"
                samples.append(
                    ImageSample(
                        image_id=image_id,
                        image_path=base / "images" / "images" / image_name,
                        category=category,
                    )
                )
        return samples

    @staticmethod
    def _read_parquet(path: Path) -> pd.DataFrame:
        if not path.is_file():
            raise FileNotFoundError(f"找不到 parquet 文件: {path}")
        return pd.read_parquet(path)

    def _filter_split(
        self,
        frame: pd.DataFrame,
        split: Optional[str] = None,
    ) -> pd.DataFrame:
        if "split" not in frame.columns:
            return frame
        source_split = (split or self.args.split).lower()
        mask = frame["split"].astype(str).str.lower() == source_split
        if mask.any():
            return frame.loc[mask].copy()
        return frame

    def _t2it_samples(
        self,
        dataset: str,
        split: Optional[str] = None,
        include_reference: bool = False,
    ) -> list[ImageSample]:
        base = self._data_dir(dataset)
        qrels = self._read_parquet(base / "qrels-00000-of-00001.parquet")
        question_by_corpus: dict[str, str] = {}
        if include_reference:
            query = self._filter_split(
                self._read_parquet(base / "query-00000-of-00001.parquet"),
                split,
            )
            query_text = {
                _normalise_id(row.get("id")): _safe_text(row.get("text")).strip()
                for _, row in query.iterrows()
            }
            for _, row in qrels.iterrows():
                query_id = _normalise_id(row.get("query-id"))
                corpus_id = _normalise_id(row.get("corpus-id"))
                question = query_text.get(query_id, "")
                if question:
                    question_by_corpus.setdefault(corpus_id, question)
            target_ids = set(question_by_corpus)
        else:
            target_ids = {
                _normalise_id(value) for value in qrels["corpus-id"].tolist()
            }
        parts = T2IT_CORPUS_PARTS[dataset]
        frames = [
            self._read_parquet(
                base / f"corpus-{part:05d}-of-{parts:05d}.parquet"
            )
            for part in range(parts)
        ]
        corpus = pd.concat(frames, ignore_index=True)
        corpus = self._filter_split(corpus, split)
        samples: list[ImageSample] = []
        seen: set[str] = set()
        for _, row in corpus.iterrows():
            image_id = _normalise_id(row.get("id"))
            if image_id not in target_ids or image_id in seen:
                continue
            image_bytes, image_path = _image_value_to_bytes(
                row.get("image"),
                (base, self.source_prefix),
            )
            if image_bytes is None:
                continue
            seen.add(image_id)
            meta: dict[str, Any] = {
                "source_row": _json_ready(row.to_dict()),
            }
            if include_reference:
                meta["few_shot_question"] = question_by_corpus[image_id]
                meta["few_shot_answer"] = _safe_text(row.get("text")).strip()
            samples.append(
                ImageSample(
                    image_id=image_id,
                    image_path=image_path,
                    image_bytes=image_bytes,
                    meta=meta,
                )
            )
        return samples

    def _it2t_samples(
        self,
        dataset: str,
        split: Optional[str] = None,
        include_reference: bool = False,
    ) -> list[ImageSample]:
        base = self._data_dir(dataset)
        query = self._filter_split(
            self._read_parquet(base / "query.parquet"),
            split,
        )
        answer_by_query: dict[str, str] = {}
        if include_reference:
            qrels = self._read_parquet(base / "qrels.parquet")
            corpus = self._filter_split(
                self._read_parquet(base / "corpus.parquet"),
                split,
            )
            corpus_text = {
                _normalise_id(row.get("id")): _safe_text(row.get("text")).strip()
                for _, row in corpus.iterrows()
            }
            for _, row in qrels.iterrows():
                query_id = _normalise_id(row.get("query-id"))
                corpus_id = _normalise_id(row.get("corpus-id"))
                answer = corpus_text.get(corpus_id, "")
                if answer:
                    answer_by_query.setdefault(query_id, answer)
        samples: list[ImageSample] = []
        seen: set[str] = set()
        for _, row in query.iterrows():
            image_id = _normalise_id(row.get("id"))
            if image_id in seen:
                continue
            image_bytes, image_path = _image_value_to_bytes(
                row.get("image"),
                (base, self.source_prefix),
            )
            if image_bytes is None:
                continue
            seen.add(image_id)
            meta: dict[str, Any] = {
                "source_row": _json_ready(row.to_dict()),
            }
            if include_reference:
                meta["few_shot_question"] = _safe_text(row.get("text")).strip()
                meta["few_shot_answer"] = answer_by_query.get(image_id, "")
            samples.append(
                ImageSample(
                    image_id=image_id,
                    image_path=image_path,
                    image_bytes=image_bytes,
                    meta=meta,
                )
            )
        return samples

    @staticmethod
    def _first_nonempty_text(values: Any) -> str:
        if isinstance(values, (list, tuple)):
            for value in values:
                text = _safe_text(value).strip()
                if text:
                    return text
            return ""
        return _safe_text(values).strip()

    def _few_shot_response(self, dataset: str, sample: ImageSample) -> str:
        if dataset in ("coco", "flickr"):
            row = sample.meta.get("row", [])
            caption_index = 3 if dataset == "coco" else 2
            if len(row) > caption_index:
                return _safe_text(row[caption_index]).strip()
            return ""
        if dataset in PERSON_JSON_NAMES:
            return self._first_nonempty_text(sample.meta.get("captions"))
        if dataset in T2IT_DATASETS or dataset in ("remuq", "llava"):
            question = _safe_text(
                sample.meta.get("few_shot_question")
            ).strip()
            answer = _safe_text(sample.meta.get("few_shot_answer")).strip()
            if question and answer:
                return json.dumps(
                    {"question": question, "answer": answer},
                    ensure_ascii=False,
                )
        return ""

    def _few_shot_source_samples(self, dataset: str) -> list[ImageSample]:
        if dataset in ("coco", "flickr"):
            samples, _ = self._csv_samples(dataset, self.few_shot_split)
            return samples
        if dataset in PERSON_JSON_NAMES:
            return self._person_samples(dataset, self.few_shot_split)
        if dataset in T2IT_DATASETS:
            return self._t2it_samples(
                dataset,
                self.few_shot_split,
                include_reference=True,
            )
        if dataset in ("remuq", "llava"):
            return self._it2t_samples(
                dataset,
                self.few_shot_split,
                include_reference=True,
            )
        return []

    def _prepare_few_shot_examples(
        self,
        dataset: str,
        target_samples: list[ImageSample],
    ) -> list[FewShotExample]:
        if not self.use_few_shot_examples:
            return []
        if dataset == "fashion-iq":
            print(
                "[warning] Fashion-IQ has no standalone source captions for "
                "the sampled images; few-shot examples are disabled for this "
                "dataset and the original generation path is used.",
                file=sys.stderr,
            )
            return []

        target_ids = {sample.image_id for sample in target_samples}
        unique: dict[str, FewShotExample] = {}
        for sample in self._few_shot_source_samples(dataset):
            if sample.image_id in target_ids:
                continue
            response = self._few_shot_response(dataset, sample)
            if response:
                unique.setdefault(
                    sample.image_id,
                    FewShotExample(sample=sample, response=response),
                )

        available = list(unique.values())
        if len(available) < self.few_shot_num:
            raise RuntimeError(
                f"{dataset} 的 {self.few_shot_split} 划分中仅有 "
                f"{len(available)} 个可用且不与目标重叠的图文示例，"
                f"少于请求的 {self.few_shot_num} 个。"
            )

        seed_text = (
            f"{self.args.seed}:{dataset}:{self.few_shot_split}:"
            f"{self.few_shot_num}"
        )
        seed_value = int.from_bytes(
            hashlib.sha256(seed_text.encode("utf-8")).digest()[:8],
            "big",
        )
        return random.Random(seed_value).sample(available, self.few_shot_num)

    def _record_few_shot_manifest(self, dataset: str) -> None:
        if not self.use_few_shot_examples:
            return
        examples = self._few_shot_by_dataset.get(dataset, [])
        entry = self.manifest["datasets"][dataset]
        entry["few_shot"] = {
            "enabled": bool(examples),
            "requested": self.few_shot_num,
            "split": self.few_shot_split,
            "text_only": bool(getattr(self.args, "few_shot_text_only", False)),
            "examples": [
                {
                    "image_id": example.sample.image_id,
                    "response": example.response,
                }
                for example in examples
            ],
        }
        if dataset == "fashion-iq":
            entry["few_shot"]["reason"] = (
                "unsupported: source annotations are relative modifications, "
                "not standalone image descriptions"
            )

    def _ensure_output(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not self.args.overwrite:
            raise FileExistsError(
                f"输出文件已存在: {path}。如需覆盖，请在命令行增加 --overwrite。"
            )

    def _copy_sample_image(self, sample: ImageSample, destination: Path) -> None:
        if not self.args.copy_images:
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        if sample.image_path is not None and sample.image_path.is_file():
            shutil.copy2(sample.image_path, destination)
        elif sample.image_bytes is not None:
            destination.write_bytes(sample.image_bytes)
        else:
            raise FileNotFoundError(f"无法复制样本图像: {sample.image_id}")

    def _write_audit(
        self,
        output_dir: Path,
        dataset: str,
        samples: list[ImageSample],
        results: list[GenerationResult],
    ) -> list[dict[str, Any]]:
        audit: list[dict[str, Any]] = []
        few_shot_examples = self._few_shot_by_dataset.get(dataset, [])
        for sample, result in zip(samples, results):
            item: dict[str, Any] = {
                "image_id": sample.image_id,
                "source_image": str(sample.image_path) if sample.image_path else "embedded_bytes",
                "category": sample.category,
                "raw_model_output": result.raw_output,
            }
            if self.use_few_shot_examples:
                item["few_shot_example_ids"] = [
                    example.sample.image_id for example in few_shot_examples
                ]
            if result.description is not None:
                item["description"] = result.description
            if result.question is not None:
                item["question"] = result.question
            if result.answer is not None:
                item["answer"] = result.answer
            if result.qa_status is not None:
                item["qa_status"] = result.qa_status
            audit.append(item)

        audit_path = output_dir / "generation_audit.jsonl"
        self._ensure_output(audit_path)
        with audit_path.open("w", encoding="utf-8") as handle:
            for item in audit:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        return audit

    def _write_csv(
        self,
        dataset: str,
        samples: list[ImageSample],
        results: list[GenerationResult],
        source_header: Optional[list[str]],
    ) -> list[str]:
        output_dir = self._output_data_dir(dataset)
        output_path = output_dir / f"{dataset}_{self.args.split}.csv"
        self._ensure_output(output_path)
        if dataset == "coco":
            header = source_header or ["imgid", "filepath", "filename", "caption", "sentid"]
        else:
            header = source_header or ["imgid", "filename", "caption", "sentid"]

        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for index, (sample, result) in enumerate(zip(samples, results)):
                image_name = _safe_text(sample.meta["image_name"])
                text_id = f"generated_{dataset}_{self.args.split}_{index:05d}"
                if dataset == "coco":
                    row = [
                        sample.image_id,
                        _safe_text(sample.meta["image_dir"]),
                        image_name,
                        result.description or "",
                        text_id,
                    ]
                else:
                    row = [sample.image_id, image_name, result.description or "", text_id]
                writer.writerow(row)

                if dataset == "coco":
                    target = output_dir / _safe_text(sample.meta["image_dir"]) / image_name
                else:
                    target = output_dir / "flickr30k-images" / image_name
                self._copy_sample_image(sample, target)

        audit = self._write_audit(output_dir, dataset, samples, results)
        self.manifest["datasets"][dataset] = {
            "kind": "image_to_text",
            "format": "csv",
            "split": self.args.split,
            "num_images_written": len(samples),
            "files": [str(output_path.relative_to(self.output_root)),
                      str((output_dir / "generation_audit.jsonl").relative_to(self.output_root))],
            "records": audit,
        }
        return [str(output_path)]

    def _write_person(
        self,
        dataset: str,
        samples: list[ImageSample],
        results: list[GenerationResult],
    ) -> list[str]:
        output_dir = self._output_data_dir(dataset)
        output_path = output_dir / PERSON_JSON_NAMES[dataset]
        self._ensure_output(output_path)
        records: list[dict[str, Any]] = []
        for sample, result in zip(samples, results):
            item = copy.deepcopy(sample.meta)
            item["captions"] = [result.description or ""]
            item["split"] = self.args.split
            records.append(item)
            relative = (
                item.get("file_path")
                if dataset != "RSTPReid"
                else item.get("img_path")
            )
            relative = _safe_text(relative).replace("\\", "/")
            self._copy_sample_image(sample, output_dir / "imgs" / relative)

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)

        audit = self._write_audit(output_dir, dataset, samples, results)
        self.manifest["datasets"][dataset] = {
            "kind": "image_to_text",
            "format": "json",
            "split": self.args.split,
            "num_images_written": len(samples),
            "files": [str(output_path.relative_to(self.output_root)),
                      str((output_dir / "generation_audit.jsonl").relative_to(self.output_root))],
            "records": audit,
        }
        return [str(output_path)]

    def _write_fashion(
        self,
        samples: list[ImageSample],
        results: list[GenerationResult],
    ) -> list[str]:
        output_dir = self._output_data_dir("fashion-iq")
        caption_dir = output_dir / "captions-20220326T130604Z-001" / "captions"
        split_dir = output_dir / "image_splits-20220326T130551Z-001" / "image_splits"
        caption_dir.mkdir(parents=True, exist_ok=True)
        split_dir.mkdir(parents=True, exist_ok=True)

        files: list[str] = []
        for category in ("dress", "shirt", "toptee"):
            selected = [
                (sample, result)
                for sample, result in zip(samples, results)
                if sample.category == category
            ]
            image_ids = [sample.image_id for sample, _ in selected]
            split_path = split_dir / f"split.{category}.{self.args.split}.json"
            caption_path = caption_dir / f"cap.{category}.{self.args.split}.json"
            self._ensure_output(split_path)
            self._ensure_output(caption_path)
            with split_path.open("w", encoding="utf-8") as handle:
                json.dump(image_ids, handle, ensure_ascii=False, indent=2)

            caption_records = [
                {
                    "candidate": sample.image_id,
                    "target": sample.image_id,
                    "captions": [result.description or ""],
                }
                for sample, result in selected
            ]
            with caption_path.open("w", encoding="utf-8") as handle:
                json.dump(caption_records, handle, ensure_ascii=False, indent=2)

            for sample, _ in selected:
                image_name = sample.image_id if sample.image_id.endswith(".png") else f"{sample.image_id}.png"
                self._copy_sample_image(
                    sample,
                    output_dir / "images" / "images" / image_name,
                )
            files.extend(
                [
                    str(split_path.relative_to(self.output_root)),
                    str(caption_path.relative_to(self.output_root)),
                ]
            )

        audit = self._write_audit(output_dir, "fashion-iq", samples, results)
        files.append(str((output_dir / "generation_audit.jsonl").relative_to(self.output_root)))
        self.manifest["datasets"]["fashion-iq"] = {
            "kind": "image_to_text",
            "format": "fashion_iq_json",
            "split": self.args.split,
            "num_images_written": len(samples),
            "files": files,
            "records": audit,
        }
        return files

    @staticmethod
    def _write_parquet(path: Path, frame: pd.DataFrame, overwrite: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(f"输出文件已存在: {path}")
        frame.to_parquet(path, index=False)

    def _write_t2it(
        self,
        dataset: str,
        samples: list[ImageSample],
        results: list[GenerationResult],
    ) -> list[str]:
        output_dir = self._output_data_dir(dataset)
        output_dir.mkdir(parents=True, exist_ok=True)

        query_rows: list[dict[str, Any]] = []
        qrels_rows: list[dict[str, Any]] = []
        corpus_rows: list[dict[str, Any]] = []
        for index, (sample, result) in enumerate(zip(samples, results)):
            query_id = f"generated-{dataset}-{self.args.split}-{index:05d}"
            corpus_id = f"generated-{dataset}-image-{index:05d}"
            query_rows.append({"id": query_id, "text": result.question or ""})
            qrels_rows.append({"query-id": query_id, "corpus-id": corpus_id})
            if sample.image_bytes is None:
                if sample.image_path is None:
                    raise FileNotFoundError(f"没有图像字节: {sample.image_id}")
                image_bytes = sample.image_path.read_bytes()
            else:
                image_bytes = sample.image_bytes
            corpus_rows.append(
                {
                    "id": corpus_id,
                    "text": result.answer or "",
                    "image": {"bytes": image_bytes},
                }
            )

        query_path = output_dir / "query-00000-of-00001.parquet"
        qrels_path = output_dir / "qrels-00000-of-00001.parquet"
        self._write_parquet(
            query_path,
            pd.DataFrame(query_rows, columns=["id", "text"]),
            self.args.overwrite,
        )
        self._write_parquet(
            qrels_path,
            pd.DataFrame(qrels_rows, columns=["query-id", "corpus-id"]),
            self.args.overwrite,
        )

        parts = T2IT_CORPUS_PARTS[dataset]
        corpus_paths: list[Path] = []
        for part in range(parts):
            part_rows = corpus_rows if part == 0 else []
            corpus_path = output_dir / f"corpus-{part:05d}-of-{parts:05d}.parquet"
            self._write_parquet(
                corpus_path,
                pd.DataFrame(part_rows, columns=["id", "text", "image"]),
                self.args.overwrite,
            )
            corpus_paths.append(corpus_path)

        audit = self._write_audit(output_dir, dataset, samples, results)
        files = [
            str(query_path.relative_to(self.output_root)),
            str(qrels_path.relative_to(self.output_root)),
            *[str(path.relative_to(self.output_root)) for path in corpus_paths],
            str((output_dir / "generation_audit.jsonl").relative_to(self.output_root)),
        ]
        self.manifest["datasets"][dataset] = {
            "kind": "text_to_image_text",
            "format": "parquet",
            "split": self.args.split,
            "num_images_written": len(samples),
            "files": files,
            "records": audit,
            "dataset_path_for_dataset_py": str(output_dir),
        }
        return files

    def _write_it2t(
        self,
        dataset: str,
        samples: list[ImageSample],
        results: list[GenerationResult],
    ) -> list[str]:
        output_dir = self._output_data_dir(dataset)
        output_dir.mkdir(parents=True, exist_ok=True)

        query_rows: list[dict[str, Any]] = []
        qrels_rows: list[dict[str, Any]] = []
        corpus_rows: list[dict[str, Any]] = []
        for index, (sample, result) in enumerate(zip(samples, results)):
            query_id = f"generated-{dataset}-{self.args.split}-{index:05d}"
            corpus_id = f"generated-{dataset}-answer-{index:05d}"
            if sample.image_bytes is None:
                if sample.image_path is None:
                    raise FileNotFoundError(f"没有图像字节: {sample.image_id}")
                image_bytes = sample.image_path.read_bytes()
            else:
                image_bytes = sample.image_bytes
            query_rows.append(
                {
                    "id": query_id,
                    "text": result.question or "",
                    "image": {"bytes": image_bytes},
                }
            )
            qrels_rows.append({"query-id": query_id, "corpus-id": corpus_id})
            corpus_rows.append({"id": corpus_id, "text": result.answer or ""})

        query_path = output_dir / "query.parquet"
        qrels_path = output_dir / "qrels.parquet"
        corpus_path = output_dir / "corpus.parquet"
        self._write_parquet(
            query_path,
            pd.DataFrame(query_rows, columns=["id", "text", "image"]),
            self.args.overwrite,
        )
        self._write_parquet(
            qrels_path,
            pd.DataFrame(qrels_rows, columns=["query-id", "corpus-id"]),
            self.args.overwrite,
        )
        self._write_parquet(
            corpus_path,
            pd.DataFrame(corpus_rows, columns=["id", "text"]),
            self.args.overwrite,
        )

        audit = self._write_audit(output_dir, dataset, samples, results)
        files = [
            str(query_path.relative_to(self.output_root)),
            str(qrels_path.relative_to(self.output_root)),
            str(corpus_path.relative_to(self.output_root)),
            str((output_dir / "generation_audit.jsonl").relative_to(self.output_root)),
        ]
        self.manifest["datasets"][dataset] = {
            "kind": "image_text_to_text",
            "format": "parquet",
            "split": self.args.split,
            "num_images_written": len(samples),
            "files": files,
            "records": audit,
            "dataset_path_for_dataset_py": str(output_dir),
        }
        return files

    @staticmethod
    def _progress_description(dataset: str, samples: list[ImageSample]) -> Iterable[ImageSample]:
        return tqdm(
            samples,
            total=len(samples),
            desc=f"{dataset} | generating descriptions",
            unit="image",
            dynamic_ncols=True,
        )

    @staticmethod
    def _progress_qa(dataset: str, samples: list[ImageSample]) -> Iterable[ImageSample]:
        return tqdm(
            samples,
            total=len(samples),
            desc=f"{dataset} | generating QA",
            unit="image",
            dynamic_ncols=True,
        )

    def _generate_descriptions(
        self,
        dataset: str,
        samples: list[ImageSample],
        few_shot_examples: Optional[list[FewShotExample]] = None,
    ) -> list[GenerationResult]:
        return [
            self.model.describe(sample, dataset, few_shot_examples)
            for sample in self._progress_description(dataset, samples)
        ]

    def _generate_qa(
        self,
        dataset: str,
        samples: list[ImageSample],
        few_shot_examples: Optional[list[FewShotExample]] = None,
    ) -> list[GenerationResult]:
        return [
            self.model.make_qa(sample, dataset, few_shot_examples)
            for sample in self._progress_qa(dataset, samples)
        ]

    def _activate_few_shot(
        self,
        dataset: str,
        samples: list[ImageSample],
    ) -> list[FewShotExample]:
        examples = self._prepare_few_shot_examples(dataset, samples)
        self._few_shot_by_dataset[dataset] = examples
        if examples:
            print(
                f"[few-shot] {dataset}: using {len(examples)} fixed examples "
                f"from split={self.few_shot_split}; ids="
                + ",".join(example.sample.image_id for example in examples)
            )
        return examples

    def run_dataset(self, dataset: str) -> None:
        if dataset in ("coco", "flickr"):
            source_samples, header = self._csv_samples(dataset)
            samples = self._sample(source_samples)
            examples = self._activate_few_shot(dataset, samples)
            results = self._generate_descriptions(dataset, samples, examples)
            self._write_csv(dataset, samples, results, header)
        elif dataset == "fashion-iq":
            samples = self._sample(self._fashion_samples())
            examples = self._activate_few_shot(dataset, samples)
            results = self._generate_descriptions(dataset, samples, examples)
            self._write_fashion(samples, results)
        elif dataset in PERSON_JSON_NAMES:
            samples = self._sample(self._person_samples(dataset))
            examples = self._activate_few_shot(dataset, samples)
            results = self._generate_descriptions(dataset, samples, examples)
            self._write_person(dataset, samples, results)
        elif dataset in T2IT_DATASETS:
            samples = self._sample(self._t2it_samples(dataset))
            examples = self._activate_few_shot(dataset, samples)
            results = self._generate_qa(dataset, samples, examples)
            self._write_t2it(dataset, samples, results)
        elif dataset in ("remuq", "llava"):
            samples = self._sample(self._it2t_samples(dataset))
            examples = self._activate_few_shot(dataset, samples)
            results = self._generate_qa(dataset, samples, examples)
            self._write_it2t(dataset, samples, results)
        else:
            raise ValueError(f"不支持的数据集: {dataset}")
        self._record_few_shot_manifest(dataset)

    def save_manifest(self) -> Path:
        self.output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.output_root / "manifest.json"
        self._ensure_output(manifest_path)
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(self.manifest, handle, ensure_ascii=False, indent=2)
        return manifest_path


def _expand_datasets(value: str) -> list[str]:
    names = [item.strip() for item in re.split(r"[,\s]+", value) if item.strip()]
    expanded: list[str] = []
    for name in names:
        if name == "all":
            for item in ALL_DATASETS:
                if item not in expanded:
                    expanded.append(item)
        elif name in ALL_DATASETS and name not in expanded:
            expanded.append(name)
        else:
            valid = ", ".join(ALL_DATASETS)
            raise ValueError(f"未知数据集 {name}，可选值为: {valid}")
    if not expanded:
        raise ValueError("--datasets 不能为空。")
    return expanded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use LLaVA-70B to generate descriptions/QA and write dataset.py-compatible files."
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="comma/space separated names, or all; llava is handled only as it2t",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="source split for CSV/JSON/Fashion-IQ; parquet datasets use it when a split column exists",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=10,
        help="number of unique images sampled per dataset",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--source_root",
        default="/root/autodl-tmp",
        help="directory containing data/<dataset>, matching dataset_path_prefix in arguments.py",
    )
    parser.add_argument(
        "--output_root",
        default="./generated_llava70b_data",
        help="new root that will contain a data/ tree and manifest.json",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=os.environ.get("MODEL_NAME_OR_PATH", "/root/autodl-fs/checkpoints/llava-70b"),
        help="local/HuggingFace LLaVA-70B checkpoint",
    )
    parser.add_argument("--device_map", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=384,
        help="global upper bound; each QA dataset also has its own generation cap",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--use_llava_next_training_prompts",
        action="store_true",
        help=(
            "use dataset-specific prompts adapted from LLaVA-NeXT training; "
            "when omitted, preserve the existing prompts"
        ),
    )
    parser.add_argument(
        "--use_few_shot_examples",
        action="store_true",
        help=(
            "prepend a fixed set of source image/text demonstrations before "
            "each target; when omitted, preserve the original generation path"
        ),
    )
    parser.add_argument(
        "--few_shot_num",
        type=int,
        default=2,
        help="number of fixed demonstrations sampled independently per dataset",
    )
    parser.add_argument(
        "--few_shot_split",
        choices=("val", "test"),
        default=None,
        help="source split for demonstrations; defaults to --split",
    )
    parser.add_argument(
        "--few_shot_text_only",
        action="store_true",
        help=(
            "use few-shot demonstrations as text-only writing examples; "
            "the target image remains the only image passed to the model"
        ),
    )
    parser.add_argument(
        "--copy_images",
        action="store_true",
        help="copy sampled file-backed images into the generated data/ tree",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing files under output_root",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.num_images <= 0:
        parser.error("--num_images 必须是正整数。")
    if args.max_new_tokens <= 0:
        parser.error("--max_new_tokens 必须是正整数。")
    if args.few_shot_split is None:
        args.few_shot_split = args.split
    if args.use_few_shot_examples:
        if args.few_shot_num <= 0:
            parser.error("启用 --use_few_shot_examples 时，--few_shot_num 必须是正整数。")
        if args.few_shot_split not in {"val", "test"}:
            parser.error("--few_shot_split 仅支持 val 或 test。")

    datasets = _expand_datasets(args.datasets)
    print(f"将处理数据集: {', '.join(datasets)}")
    print(f"每个数据集随机抽取 {args.num_images} 个图像，split={args.split}")
    print(f"模型: {args.model_name_or_path}")
    prompt_mode = (
        "llava_next_training"
        if args.use_llava_next_training_prompts
        else "existing"
    )
    print(f"提示词模式: {prompt_mode}")
    if args.use_few_shot_examples:
        print(
            f"Few-shot 模式: enabled, num={args.few_shot_num}, "
            f"split={args.few_shot_split}"
        )
    else:
        print("Few-shot 模式: disabled")

    model = Llava70BGenerator(args)
    generator = DatasetGenerator(args, model)
    for dataset in datasets:
        print(f"[start] {dataset}")
        generator.run_dataset(dataset)
        print(f"[done]  {dataset}")
    manifest_path = generator.save_manifest()
    print(f"生成完成，清单文件: {manifest_path}")


if __name__ == "__main__":
    main()
