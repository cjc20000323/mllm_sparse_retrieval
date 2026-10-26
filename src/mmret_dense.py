"""MMRet/BGE-VL-MLLM S1/S2 adapter and backward-compatible helper exports."""

from typing import Optional

from PIL import Image
import torch
import torch.distributed as dist

from dense_encoding import (
    DATASET_PROFILES, Candidate, DenseCandidates as MMRetCandidates, DenseEncoderArguments,
    EncodingContext, LlavaNextDenseEncoder, build_candidates, check_existing_shards,
    dataset_paths, decode_image, encode_shard, last_token_pool, output_directory,
    run_dense_encoding, save_shard, setup_context, shard_indices, validate_task,
)


def is_mmret_model(model_path: str, model_type: str = "auto") -> bool:
    if model_type not in ("auto", "mmret"):
        raise ValueError("--dense_model_type must be auto or mmret.")
    name = str(model_path).lower().replace("_", "-")
    return model_type == "mmret" or "mmret-mllm" in name or "bge-vl-mllm" in name


def load_image(value) -> Image.Image:
    # Preserve MMRet's original order: resize first, then convert to RGB.
    with decode_image(value) as image:
        return image.resize((512, 512)).convert("RGB")


def candidate_prompt(text: Optional[str], has_image: bool) -> str:
    # Match MMRet's prepare_text_input(q_or_c='c'), including whitespace.
    prompt = "[INST] "
    if text is not None:
        if not isinstance(text, str):
            raise TypeError(f"Candidate text must be a string, got {type(text).__name__}.")
        prompt = f"{prompt} {text} \n"
    if has_image:
        prompt = f"{prompt} <image>"
    return f"{prompt} [/INST]"


def query_prompt(text: Optional[str], has_image: bool, instruction: str) -> str:
    """Match prepare_text_input(q_or_c='q'), including its whitespace."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("MMRet query instruction must be a nonempty string.")
    if text is not None and not isinstance(text, str):
        raise TypeError("MMRet query text must be a string.")
    if text is None and not has_image:
        raise ValueError("An MMRet query must contain text or an image.")
    prompt = f"[INST] \n <instruct> {instruction} <query> "
    if text is not None:
        prompt = f"{prompt} {text} \n"
    if has_image:
        prompt = f"{prompt} <image>"
    return f"{prompt} [/INST]"


class MMRetEncoder(LlavaNextDenseEncoder):
    model_label = "MMRet"

    def format_prompt(self, candidate: Candidate) -> str:
        return candidate_prompt(candidate.text, candidate.image is not None)

    def prepare_image(self, value) -> Image.Image:
        return load_image(value)


def run_mmret_encoding(model_args, data_args, training_args, dense_args):
    return run_dense_encoding(model_args, data_args, training_args, dense_args, MMRetEncoder,
                              context_factory=setup_context, candidate_builder=build_candidates)
