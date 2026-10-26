"""VLM2Vec-LLaVa-Next embeddings using native Hugging Face model/processor APIs.

Reference: TIGER-Lab/VLM2Vec-LLaVa-Next model card and official VLM2Vec v1
model.py, dataset.py and examples/llava_next/run_eval.sh. No local VLM2Vec
wrapper, chat/summary template, base checkpoint or PEFT adapter is loaded.
"""

from typing import Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from dense_encoding import (Candidate, build_candidates, decode_image, last_token_pool,
                            output_directory, run_dense_encoding, setup_context)

MODEL_ID = "TIGER-Lab/VLM2Vec-LLaVa-Next"
IMAGE_PROMPT = "<image> Represent the given image."
IMAGE_RESOLUTIONS = {"high": 1344, "mid": 672, "low": 128, "original": None}
QUERY_INSTRUCTIONS = {
    "ir:text": "Retrieve an image matching the following description:",
    "ir:image": "Find an image caption describing the given image.",
    "tbpr": "Retrieve a person image matching the following description:",
    "t2it": "Retrieve an image and text pair answering the following question:",
    "it2t": "Represent the given image with the following question:",
}


def is_vlm2vec_model(model_path: str, model_type: str = "auto") -> bool:
    if model_type not in ("auto", "mmret", "unime", "vlm2vec"):
        raise ValueError("--dense_model_type must be auto, mmret, unime or vlm2vec.")
    if model_type in ("mmret", "unime"):
        return False
    name = str(model_path).replace("\\", "/").rstrip("/").split("/")[-1].lower().replace("_", "-")
    supported = "vlm2vec-llava-next" in name
    if model_type == "vlm2vec":
        if "vlm2vec" in name and not supported:
            raise ValueError(f"The native VLM2Vec encoder supports {MODEL_ID} only, not Qwen, Phi or V2. "
                             "For a renamed full LLaVA-NeXT snapshot, pass --dense_model_type vlm2vec.")
        return True
    return supported


def validate_resolution(resolution):
    if resolution not in IMAGE_RESOLUTIONS:
        raise ValueError("--vlm2vec_image_resolution must be high, mid, low or original.")


def candidate_prompt(text: Optional[str], has_image: bool) -> str:
    if text is not None and not isinstance(text, str):
        raise TypeError("VLM2Vec candidate text must be a string.")
    if has_image:
        # WebQA image+text candidates are a project extension; retain both inputs.
        return IMAGE_PROMPT if text is None else f"<image>\n{text}"
    if text is None:
        raise ValueError("A VLM2Vec candidate must contain text or an image.")
    return text


def query_prompt(text: Optional[str], has_image: bool, instruction: str) -> str:
    # The model-card VQA form is reused for it2t; other tasks are project mappings.
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("VLM2Vec query instruction must be a nonempty string.")
    if "<image>" in instruction:
        raise ValueError("VLM2Vec query instruction must not contain <image>; it is added automatically.")
    if text is not None and not isinstance(text, str):
        raise TypeError("VLM2Vec query text must be a string.")
    if text is None and not has_image:
        raise ValueError("A VLM2Vec query must contain text or an image.")
    prompt = f"<image> {instruction}" if has_image else instruction
    return prompt if text is None else f"{prompt} {text}"


def load_image(value, resolution="high") -> Image.Image:
    validate_resolution(resolution)
    with decode_image(value) as image:
        size = IMAGE_RESOLUTIONS[resolution]
        if size is None:
            return image.convert("RGB")
        # Match the official v1 process_image(): square resize before HF processing.
        with image.resize((size, size)) as resized:
            return resized.convert("RGB")


def vlm2vec_output_directory(model_args, data_args, training_args, resolution="high"):
    validate_resolution(resolution)
    directory = output_directory(model_args, data_args, training_args)
    return directory.with_name(f"{directory.name}_vlm2vec_{resolution}")


class VLM2VecEncoder:
    model_label = "VLM2Vec"

    def __init__(self, backbone, processor, device, dtype, image_resolution="high"):
        validate_resolution(image_resolution)
        self.backbone = backbone.eval()
        self.processor = processor
        self.device = torch.device(device)
        self.dtype = dtype
        self.image_resolution = image_resolution
        self.dimension = int(backbone.config.text_config.hidden_size)

    @classmethod
    def from_pretrained(cls, model_path, device, dtype, cache_dir=None, image_resolution="high"):
        from packaging.version import Version
        from transformers import (__version__, AutoConfig, LlavaNextForConditionalGeneration,
                                  LlavaNextProcessor)

        validate_resolution(image_resolution)
        if Version(__version__) < Version("4.57.3"):
            raise RuntimeError("Native VLM2Vec encoding requires transformers>=4.57.3 for LLaVA-NeXT checkpoint conversion.")
        config = AutoConfig.from_pretrained(model_path, cache_dir=cache_dir, trust_remote_code=False)
        if config.model_type != "llava_next" or getattr(config.text_config, "model_type", None) != "mistral":
            raise ValueError(f"{MODEL_ID} requires a full LLaVA-NeXT/Mistral checkpoint; "
                             "Qwen/Phi/V2 backbones and adapter-only repositories are not supported.")
        model, info = LlavaNextForConditionalGeneration.from_pretrained(
            model_path, config=config, torch_dtype=dtype, device_map={"": str(device)},
            attn_implementation="sdpa", low_cpu_mem_usage=True, cache_dir=cache_dir,
            use_safetensors=True, trust_remote_code=False, output_loading_info=True)
        missing = [key for key in info.get("missing_keys", []) if key != "lm_head.weight"]
        if missing or info.get("mismatched_keys") or info.get("unexpected_keys"):
            raise ValueError(f"VLM2Vec checkpoint did not load completely: {info}")
        processor = LlavaNextProcessor.from_pretrained(model_path, cache_dir=cache_dir,
                                                      use_fast=False, trust_remote_code=False)
        processor.patch_size = config.vision_config.patch_size
        processor.vision_feature_select_strategy = config.vision_feature_select_strategy
        processor.num_additional_image_tokens = 1  # CLIP's CLS token.
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        # Drop the unused language generation head; the backbone returns the final layer directly.
        return cls(model.model, processor, device, dtype, image_resolution)

    def format_prompt(self, candidate: Candidate) -> str:
        return candidate_prompt(candidate.text, candidate.image is not None)

    @torch.inference_mode()
    def encode_batch(self, candidates: list[Candidate]) -> np.ndarray:
        if not candidates:
            return np.empty((0, self.dimension), dtype=np.float32)
        has_images = [item.image is not None for item in candidates]
        if any(has_images) and not all(has_images):
            raise ValueError("A VLM2Vec batch must have a consistent image modality.")
        images = []
        identifiers = [item.identifier for item in candidates]
        try:
            prompts = [self.format_prompt(item) for item in candidates]
            for item in candidates:
                if item.image is not None:
                    images.append(load_image(item.image, self.image_resolution))
            inputs = self.processor(text=prompts, images=images or None, padding=True,
                                    truncation=False, return_tensors="pt")
        except Exception as exc:
            raise ValueError(f"VLM2Vec preprocessing failed for IDs {identifiers}: {exc}") from exc
        finally:
            for image in images:
                image.close()
        limit = self.backbone.config.text_config.max_position_embeddings
        if inputs["input_ids"].shape[1] > limit:
            raise ValueError(f"VLM2Vec input exceeds context length {limit}; image tokens must not be truncated. "
                             f"Candidate IDs: {identifiers}")
        inputs = {key: value.to(self.device, dtype=self.dtype) if value.is_floating_point()
                  else value.to(self.device) for key, value in inputs.items()}
        mask = inputs["attention_mask"]
        inputs["position_ids"] = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 0)
        outputs = self.backbone(**inputs, use_cache=False, output_hidden_states=False, return_dict=True)
        reps = F.normalize(last_token_pool(outputs.last_hidden_state, mask).float(), dim=-1)
        if reps.shape != (len(candidates), self.dimension) or not torch.isfinite(reps).all():
            raise ValueError(f"VLM2Vec returned invalid embeddings for IDs {identifiers}.")
        return reps.cpu().numpy()


class VLM2VecQueryEncoder(VLM2VecEncoder):
    instruction = ""

    def format_prompt(self, candidate: Candidate) -> str:
        return query_prompt(candidate.text, candidate.image is not None, self.instruction)


def run_vlm2vec_encoding(model_args, data_args, training_args, dense_args):
    resolution = dense_args.vlm2vec_image_resolution
    validate_resolution(resolution)
    return run_dense_encoding(model_args, data_args, training_args, dense_args, VLM2VecEncoder,
        context_factory=setup_context, candidate_builder=build_candidates,
        encoder_kwargs={"image_resolution": resolution},
        directory_builder=lambda model, data, train: vlm2vec_output_directory(model, data, train, resolution))
