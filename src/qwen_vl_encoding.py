"""Shared Qwen VL PromptEOL and one-forward, isolated multi-view encoding."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import inspect

import torch
import torch.nn.functional as F
from PIL import Image

from qwen_vl_prompts import ASSISTANT, METHODS, TURN_END, render_user, retrieval_prompt


def read_image(source):
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, dict):
        source = source.get("bytes") if source.get("bytes") is not None else source.get("path")
    if isinstance(source, (bytes, bytearray, memoryview)):
        source = BytesIO(bytes(source))
    with Image.open(source) as image:
        return image.convert("RGB").copy()


@dataclass
class ViewBatch:
    inputs: dict
    # Per sample, inclusive starts / exclusive ends, AFTER image expansion/padding.
    spans: list[list[tuple[int, int]]]
    prompts: list[str]


def _occurrences(sequence, needle):
    return [i for i in range(len(sequence) - len(needle) + 1) if sequence[i:i + len(needle)] == needle]


def prepare_views(processor, contexts, instructions, images, *, padding_side="left"):
    """Build shared-context blocks using the checkpoint's own chat template.

    In the packed text each block ends at a native assistant generation prefix.
    The attention mask makes each block see only the shared user context and its
    own instruction. There are no synthetic/tool-call tokens in the sequence.
    """
    if not contexts or len(contexts) != len(instructions) or len(contexts) != len(images):
        raise ValueError("Contexts, views and image groups must have the same nonzero batch size")
    processor.tokenizer.padding_side = padding_side
    prompts, block_ids = [], []
    for context, views, group in zip(contexts, instructions, images):
        if not views:
            raise ValueError("Every input must have at least one view")
        prefix = render_user(processor, context)
        if not prefix.endswith(TURN_END):
            raise ValueError("Unsupported Qwen checkpoint chat template: missing native assistant prefix")
        blocks = [view + TURN_END for view in views]
        prompt = prefix[:-len(TURN_END)] + "".join(blocks)
        if prompt.count("<|image_pad|>") != len(group):
            raise ValueError("Image count differs from Qwen visual placeholders")
        prompts.append(prompt)
        block_ids.append([processor.tokenizer.encode(block, add_special_tokens=False) for block in blocks])
    kwargs = dict(text=prompts, padding=True, return_tensors="pt", add_special_tokens=False)
    flat_images = [read_image(image) for group in images for image in group]
    if flat_images:
        kwargs["images"] = flat_images
    inputs = dict(processor(**kwargs))
    header = processor.tokenizer.encode(ASSISTANT, add_special_tokens=False)
    spans = []
    for row, expected in enumerate(block_ids):
        sequence = inputs["input_ids"][row].tolist()
        ends = [start + len(header) for start in _occurrences(sequence, header)][-len(expected):]
        if len(ends) != len(expected):
            raise ValueError("Processor changed the number of Qwen view boundaries")
        starts = [ends[0] - len(expected[0])] + ends[:-1]
        row_spans = list(zip(starts, ends))
        for (start, end), ids in zip(row_spans, expected):
            if start < 0 or sequence[start:end] != ids or not inputs["attention_mask"][row, start:end].all():
                raise ValueError("Qwen view tokenization crosses a block boundary; refusing incorrect view positions")
        spans.append(row_spans)
    return ViewBatch(inputs, spans, prompts)


def _mm_token_type_ids(model, inputs):
    """Use processor modality IDs, or reconstruct them after visual expansion."""
    ids = inputs["input_ids"]
    types = inputs.get("mm_token_type_ids")
    if types is not None:
        if not torch.is_tensor(types) or types.shape != ids.shape:
            raise ValueError("Qwen mm_token_type_ids must be a tensor with the same shape as input_ids")
        return types.to(device=ids.device)
    types = torch.zeros_like(ids)
    for modality, value in (("image", 1), ("video", 2)):
        token_id = getattr(model.config, f"{modality}_token_id", None)
        if token_id is None:
            if inputs.get(f"{modality}_grid_thw") is not None:
                raise ValueError(f"Cannot build Qwen mm_token_type_ids: config lacks {modality}_token_id")
        else:
            types.masked_fill_(ids == token_id, value)
    mask = inputs.get("attention_mask")
    if mask is not None and mask.ndim == 2:
        types.masked_fill_(mask == 0, 0)
    return types


def _forward_inputs(model, inputs):
    """Do not leak new processor metadata into older model implementations."""
    parameters = inspect.signature(model.forward).parameters
    supports_types = "mm_token_type_ids" in parameters
    # Some wrappers forward **kwargs to the native multimodal base model.
    if not supports_types and any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        base = getattr(model, "model", None)
        if base is not None and hasattr(base, "forward"):
            supports_types = "mm_token_type_ids" in inspect.signature(base.forward).parameters
    result = dict(inputs)
    if supports_types:
        result["mm_token_type_ids"] = _mm_token_type_ids(model, inputs)
    else:
        result.pop("mm_token_type_ids", None)
    return result


def rope_positions(model, inputs):
    owner = model if hasattr(model, "get_rope_index") else model.model
    parameters = inspect.signature(owner.get_rope_index).parameters
    kwargs = dict(input_ids=inputs["input_ids"], image_grid_thw=inputs.get("image_grid_thw"),
                  video_grid_thw=inputs.get("video_grid_thw"), attention_mask=inputs["attention_mask"])
    if "mm_token_type_ids" in parameters:
        kwargs["mm_token_type_ids"] = _mm_token_type_ids(model, inputs)
    if "second_per_grid_ts" in parameters and "second_per_grid_ts" in inputs:
        kwargs["second_per_grid_ts"] = inputs["second_per_grid_ts"]
    return owner.get_rope_index(**kwargs)[0]


def isolate_views(model, batch):
    """Keep native visual MRoPE, reset text positions per independent view."""
    inputs = batch.inputs
    mask2 = inputs["attention_mask"]
    positions = rope_positions(model, inputs).clone()
    size = mask2.shape[-1]
    dtype = model.get_input_embeddings().weight.dtype
    minimum = torch.finfo(dtype).min
    # Start with a causal mask; unlike the legacy image path, future input tokens
    # must never be visible. Each row has its own boundaries and padding offset.
    mask = torch.full((len(batch.spans), 1, size, size), minimum, dtype=dtype, device=mask2.device)
    causal = torch.ones(size, size, dtype=torch.bool, device=mask2.device).tril()
    mask.masked_fill_(causal[None, None] & mask2[:, None, None, :].bool(), 0)
    for row, spans in enumerate(batch.spans):
        common_end = spans[0][0]
        origin = positions[:, row, common_end].clone()
        for start, end in spans:
            mask[row, 0, start:end, common_end:start] = minimum
            positions[:, row, start:end] = origin[:, None] + torch.arange(end - start, device=mask2.device)[None, :]
    return {**inputs, "attention_mask": mask, "position_ids": positions}


@torch.no_grad()
def forward_views(model, batch, *, isolated):
    device = model.get_input_embeddings().weight.device
    batch.inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.inputs.items()}
    batch.inputs = _forward_inputs(model, batch.inputs)
    inputs = isolate_views(model, batch) if isolated else batch.inputs
    output = model(**inputs, use_cache=False, output_hidden_states=True, return_dict=True)
    logits, hidden = [], []
    for row, spans in enumerate(batch.spans):
        ends = torch.tensor([end - 1 for _, end in spans], device=output.logits.device)
        logits.append(output.logits[row, ends])
        hidden.append(output.hidden_states[-1][row, ends.to(output.hidden_states[-1].device)])
    return logits, hidden


class NativeQwenEncoder:
    def __init__(self, model, processor, model_args, data_args, task):
        method = (model_args.eol_type, model_args.calculate_type)
        if method not in METHODS:
            raise ValueError(f"Qwen adapter supports {sorted(METHODS)}, got {method}")
        self.model, self.processor = model, processor
        self.model_args, self.data, self.task = model_args, data_args, task
        self.concat = method[1] == "concat"
        self.last_prompts = []

    def prepare(self, records):
        contexts, instructions, groups = [], [], []
        for record in records:
            context, whole, views = retrieval_prompt(record, self.task, self.data, self.processor, include_views=self.concat)
            contexts.append(context)
            instructions.append([whole] + views if self.concat else [whole])
            groups.append([record["image"]] if record.get("image") is not None else [])
        return prepare_views(self.processor, contexts, instructions, groups,
                             padding_side="left" if self.data.reps_loc == "after_pad" else "right")

    def encode(self, records):
        batch = self.prepare(records)
        self.last_prompts = batch.prompts
        logits, hidden = forward_views(self.model, batch, isolated=self.concat)
        dense = F.normalize(torch.stack([row[0] for row in hidden]), dim=-1)
        # Ragged per-sample lists also handle task-specific view counts safely.
        sparse = [torch.log1p(torch.relu(row[1:] if self.concat else row)) for row in logits]
        return sparse, dense


class SparseEncoder:
    """Keep PromptReps weighting while filtering IDs and values together."""
    def __init__(self, tokenizer, data, *, concat):
        self.tokenizer, self.data, self.concat = tokenizer, data, concat
        self.vocab = {value: key for key, value in tokenizer.get_vocab().items()}
        self._valid_ids = {}

    def _term(self, token):
        if self.data.sparse_lower_or_upper == "lower":
            token = token.lower()
        if self.data.is_filtered:
            token = token.lstrip("Ġġ▁ ")
            if not token or not token.isalnum() or not ("a" <= token[0].lower() <= "z"):
                return None
        # Lucene's whitespace analyzer cannot represent a term containing spaces.
        return token if token and not any(char.isspace() for char in token) else None

    def vector(self, logits, record):
        vector = {}
        text_input = record.get("image") is None
        length = self.data.sparse_length
        if not self.concat and self.data.sparse_manual:
            length = self.data.text_sparse_length if text_input else self.data.image_sparse_length
        size = logits.shape[-1]
        if size not in self._valid_ids:
            self._valid_ids[size] = sorted(key for key, term in self.vocab.items() if key < size and self._term(term) is not None)
        valid = self._valid_ids[size]
        if not valid:
            raise ValueError("Tokenizer has no valid sparse terms")
        # separate PromptEOL without manual expansion restricts text terms to the
        # input caption, as in get_text_valid_tokens_values; images use top-K.
        source_ids = set()
        if text_input and not self.concat and (not self.data.sparse_manual or self.data.num_expended_tokens > 0):
            import string
            from nltk import word_tokenize
            from nltk.corpus import stopwords
            excluded = set(stopwords.words("english")) | set(string.punctuation)
            words = [word for word in word_tokenize(record.get("text", "").lower()) if word not in excluded]
            source_ids = {i for word in words for i in self.tokenizer.encode(word, add_special_tokens=False)}
            selected = sorted(source_ids.intersection(valid))
            if selected and not self.data.sparse_manual:
                valid = selected
                length = min(len(valid), 128)
            elif not self.data.sparse_manual:
                length = 10
        ids = torch.tensor(valid, device=logits.device)
        for view in logits:
            values, selected = view[ids].topk(min(length, len(valid)))
            selected_ids = ids[selected]
            if not self.concat and self.data.num_expended_tokens > 0:
                excluded = source_ids if text_input else set(selected_ids.tolist())
                expansion = [index for index in self._valid_ids[size] if index not in excluded and self.vocab[index].lstrip("Ġġ▁ ").isalpha()]
                if expansion:
                    expansion_ids = torch.tensor(expansion, device=logits.device)
                    extra_values, extra_positions = view[expansion_ids].topk(min(self.data.num_expended_tokens, len(expansion)))
                    selected_ids = torch.cat([selected_ids, expansion_ids[extra_positions]])
                    values = torch.cat([values, extra_values])
            weights = torch.round(values.float() * 100).long().tolist()
            for index, weight in zip(selected_ids.tolist(), weights):
                term = self._term(self.vocab[index])
                if weight <= 0:
                    continue
                if term not in vector or self.data.sparse_value_type == "replace":
                    vector[term] = weight
                elif self.data.sparse_value_type == "sum":
                    vector[term] += weight
                else:
                    vector[term] = max(vector[term], weight)
        if self.concat and self.data.sparse_value_mean:
            vector = {key: value // len(logits) for key, value in vector.items() if value // len(logits) > 0}
        return vector
