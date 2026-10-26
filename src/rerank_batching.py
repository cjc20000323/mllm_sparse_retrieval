"""Batch scheduling and token scoring shared by the two reranking pipelines."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import torch
import torch.nn.functional as F


@dataclass
class RerankRequest:
    candidate_id: Any
    text: str
    images: tuple[Any, ...]
    prefix: Optional[str] = None
    text_only_prefix: Optional[str] = None


def validate_rerank_batch_size(batch_size: int) -> None:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("rerank_batch_size must be a positive integer")


def indexed_batches(
    items: Sequence[Any], batch_size: int, *, key: Optional[Callable[[Any], Any]] = None,
):
    """Group compatible samples without losing their original result positions."""
    validate_rerank_batch_size(batch_size)
    groups: dict[Any, list[tuple[int, Any]]] = OrderedDict()
    for index, item in enumerate(items):
        groups.setdefault(key(item) if key is not None else None, []).append((index, item))
    for group in groups.values():
        for start in range(0, len(group), batch_size):
            yield group[start:start + batch_size]


def last_valid_logits(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or logits.shape[:2] != attention_mask.shape:
        raise ValueError(
            "Model logits and attention_mask sequence sizes differ. Configure the "
            "multimodal processor's image-token expansion metadata before batch scoring."
        )
    mask = attention_mask.to(device=logits.device, dtype=torch.bool)
    positions = torch.arange(mask.shape[1], device=logits.device).expand_as(mask)
    ends = positions.masked_fill(~mask, -1).max(dim=1).values
    if (ends < 0).any():
        raise ValueError("Cannot score a sample with an empty attention mask")
    return logits[torch.arange(logits.shape[0], device=logits.device), ends]


def target_labels(
    input_ids: torch.Tensor, attention_mask: torch.Tensor,
    prefix_ids: torch.Tensor, prefix_mask: torch.Tensor,
) -> torch.Tensor:
    """Align actual (image-expanded) prefix IDs; never guess labels from text length.

    A tokenizer may merge across the prefix/answer boundary. The first changed
    token then belongs to the scored continuation, instead of fabricating labels
    from separately tokenized answer IDs. BOS/EOS and either padding side are
    taken from the processor's real sequences.
    """
    if input_ids.shape != attention_mask.shape or prefix_ids.shape != prefix_mask.shape:
        raise ValueError("Token IDs and masks must have matching shapes")
    if input_ids.shape[0] != prefix_ids.shape[0]:
        raise ValueError("Full inputs and prompt prefixes must have the same batch size")
    labels = torch.full_like(input_ids, -100)
    for row in range(input_ids.shape[0]):
        positions = attention_mask[row].bool().nonzero(as_tuple=True)[0]
        full = input_ids[row, positions].tolist()
        prefix = prefix_ids[row, prefix_mask[row].bool()].tolist()
        common = 0
        for full_id, prefix_id in zip(full, prefix):
            if full_id != prefix_id:
                break
            common += 1
        if not common or common == len(full):
            raise ValueError("Caption target has no scoreable continuation or its prompt prefix cannot be aligned")
        labels[row, positions[common:]] = input_ids[row, positions[common:]]
    return labels


def mean_target_nll(
    logits: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 3 or logits.shape[:2] != labels.shape or labels.shape != attention_mask.shape:
        raise ValueError("Logits, target labels and attention_mask must describe the same sequences")
    labels = labels.to(logits.device)
    mask = attention_mask.to(device=logits.device, dtype=torch.bool)
    valid = (labels[:, 1:] != -100) & mask[:, 1:] & mask[:, :-1]
    rows, columns = valid.nonzero(as_tuple=True)
    counts = torch.bincount(rows, minlength=logits.shape[0])
    if (counts == 0).any():
        raise ValueError("Every caption must contain at least one valid predicted target token")
    # Gather only answer positions before float conversion/cross entropy. Do not
    # copy the enormous image/prompt portion of [batch, sequence, vocabulary].
    losses = F.cross_entropy(
        logits[rows, columns].float(), labels[rows, columns + 1], reduction="none",
    )
    totals = torch.zeros(logits.shape[0], device=logits.device, dtype=losses.dtype)
    return totals.scatter_add_(0, rows, losses) / counts


def configure_rerank_processor(model: Any, processor: Any, reps_loc: str) -> None:
    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.padding_side = "left" if reps_loc == "after_pad" else "right"
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos = getattr(tokenizer, "eos_token", None)
        if eos is not None:
            tokenizer.pad_token = eos
        elif getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    config = getattr(model, "config", None)
    vision = getattr(config, "vision_config", None)
    # LLaVA processors must expand image placeholders before model.forward so
    # token masks and returned logits retain the same positions.
    if hasattr(processor, "patch_size") and processor.patch_size is None:
        processor.patch_size = getattr(vision, "patch_size", None)
    if hasattr(processor, "vision_feature_select_strategy") and processor.vision_feature_select_strategy is None:
        processor.vision_feature_select_strategy = getattr(config, "vision_feature_select_strategy", "default")
    if (getattr(vision, "model_type", None) == "clip_vision_model"
            and hasattr(processor, "num_additional_image_tokens")
            and not processor.num_additional_image_tokens):
        # CLIP emits a CLS feature before selection; SigLIP does not. Avoid
        # losing one placeholder per image with older LLaVA processor configs.
        processor.num_additional_image_tokens = 1


def move_batch(inputs: Any, device: Any) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
