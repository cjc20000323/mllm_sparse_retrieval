"""UniME-LLaVA-1.6-7B candidate embeddings using the official native weights.

Pure-modality templates follow the model card. The WebQA joint template is a
project extension, not an official published WebQA evaluation configuration.
"""

from typing import Optional

from PIL import Image

from dense_encoding import (
    Candidate, DenseEncoderArguments, LlavaNextDenseEncoder, build_candidates,
    decode_image, run_dense_encoding, setup_context,
)

MODEL_ID = "DeepGlint-AI/UniME-LLaVA-1.6-7B"
IMAGE_PROMPT = "[INST] <image>\nSummary above image in one word: [/INST]"
TEXT_PROMPT = "[INST] {text}\nSummary above sentence in one word: [/INST]"
# Engineering extension for joint candidate images+text, not a vector sum.
JOINT_PROMPT = "[INST] <image>\n{text}\nSummary above image and text in one word: [/INST]"


def is_unime_model(model_path: str, model_type: str = "auto") -> bool:
    if model_type not in ("auto", "mmret", "unime"):
        raise ValueError("--dense_model_type must be auto, mmret or unime.")
    if model_type == "mmret":
        return False
    name = str(model_path).replace("\\", "/").rstrip("/").split("/")[-1].lower().replace("_", "-")
    supported = "unime-llava-1.6-7b" in name
    if model_type == "unime":
        unsupported = ("unime-v2", "unime-v-2", "unime-phi", "unime-llava-ov", "unime-llava-onevision", "unime-r1")
        if not any(part in name for part in unsupported):
            return True
    elif supported:
        return True
    if "unime" in name or model_type == "unime":
        raise ValueError(f"Only {MODEL_ID} is supported by --dense_model_type unime. "
                         "UniME-V2, OneVision and Phi checkpoints are not interchangeable. "
                         "For an arbitrarily named local LLaVA-1.6 checkpoint, pass --dense_model_type unime.")
    return False


def candidate_prompt(text: Optional[str], has_image: bool) -> str:
    if text is not None and not isinstance(text, str):
        raise TypeError(f"Candidate text must be a string, got {type(text).__name__}.")
    if has_image:
        return IMAGE_PROMPT if text is None else JOINT_PROMPT.format(text=text)
    if text is None:
        raise ValueError("A UniME candidate must contain text or an image.")
    return TEXT_PROMPT.format(text=text)


def load_image(value) -> Image.Image:
    # Keep native width/height. The checkpoint's processor owns AnyRes resize,
    # crops, normalization and grid selection; MMRet's 512x512 rule does not apply.
    with decode_image(value) as image:
        return image.convert("RGB")


class UniMEEncoder(LlavaNextDenseEncoder):
    model_label = "UniME"

    def format_prompt(self, candidate: Candidate) -> str:
        return candidate_prompt(candidate.text, candidate.image is not None)

    def prepare_image(self, value) -> Image.Image:
        return load_image(value)


def run_unime_encoding(model_args, data_args, training_args, dense_args):
    return run_dense_encoding(model_args, data_args, training_args, dense_args, UniMEEncoder,
                              context_factory=setup_context, candidate_builder=build_candidates)
