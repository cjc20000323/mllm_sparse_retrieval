"""Complete FreeRet prompts in one forward with isolated shared-context views."""
from __future__ import annotations

import hashlib
import json
import math
import re
from uuid import uuid4

from freeret_prompt_catalog import IMAGE_TOKEN, PROMPT_SPECS, PROMPT_VERSION

SUFFIX = "_freeret_full_prompt_v2"
MANIFEST = "freeret_full_prompt_manifest.json"
CHAT_PREFIX = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
CHAT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
DATASETS = {"coco": "coco_t2i", "flickr": "flickr_t2i", "CUHK-PEDES": "cuhk_pedes_t2i",
            "ICFG-PEDES": "icfg_pedes_t2i", "RSTPReid": "rstpreid_t2i", "webqa": "webqa_t2it",
            "remuq": "remuq_it2t", "llava": "llava_it2t"}
SEMANTIC_RULES = {
    "person": " - Output a single lowercase noun or adjective that best captures the {aspect} in the {reference}.",
    "noun": " - Output a single lowercase noun that best captures the {aspect} of the {reference}.",
    "word": " - Output a single word that best captures the {aspect} of the {reference}.",
    "default": " - The word should best capture the {aspect} of the {reference}.",
}
CONFIGURATION = dict(version=2, catalog_version=PROMPT_VERSION,
    chat_prefix=CHAT_PREFIX, chat_suffix=CHAT_SUFFIX, semantic_rules=SEMANTIC_RULES,
    source="freeret_complete_prompt", perspective_rule="replace_semantic_object_v1",
    overlap_projection="single_modality_fields_and_rules_v1", forwarding="packed_shared_context_v1",
    attention="causal_shared_prefix_isolated_blocks_v1", positions="per_block_reset_native_visual_rope_v1",
    mistral_window="native_window_on_reset_positions_v1",
    token_packing="standalone_token_suffixes_after_shared_prefix_v1",
    fingerprint=hashlib.sha256(json.dumps({key: dict(description=spec.task_description,
        query_input=spec.query_input("{text}"), target_input=spec.target_input("{text}"),
        query=spec.query_instruction, target=spec.target_instruction) for key,spec in PROMPT_SPECS.items()},
        sort_keys=True,ensure_ascii=False).encode()).hexdigest())


def enabled(args):
    from dense_representation import boolean, enabled as dense_enabled
    use = boolean(getattr(args,"use_freeret_full_prompt",False))
    if use and not dense_enabled(args):
        raise ValueError("USE_FREERET_FULL_PROMPT=true requires USE_FREERET_DENSE=true")
    return use


def validate(args, dataset=None):
    if enabled(args) and dataset is not None and dataset not in DATASETS:
        raise ValueError(f"No complete FreeRet prompt registered for dataset {dataset!r}")


def canonical_prompt(key, role, text):
    spec = PROMPT_SPECS[key]
    return CHAT_PREFIX + spec.task_description + getattr(spec,role+"_input")(text) + getattr(spec,role+"_instruction") + CHAT_SUFFIX


def safe_text(value):
    return "" if value is None or isinstance(value,float) and math.isnan(value) else str(value)


def llava_body(text, reserved=("<image>",)):
    body = text.removeprefix(CHAT_PREFIX).removesuffix(CHAT_SUFFIX).replace(IMAGE_TOKEN,"")
    for token in reserved:
        body = body.replace(token,"image")
    return body.strip()


def role_for(dataset, record, input_type=None):
    image = record.get("image") is not None or input_type == "image"
    if dataset in {"coco","flickr","CUHK-PEDES","ICFG-PEDES","RSTPReid"}:
        return "target" if image else "query"
    role = record.get("_freeret_role") or input_type
    if role in {"query","target","corpus"}:
        return "target" if role == "corpus" else role
    return ("target" if image else "query") if dataset == "webqa" else ("query" if image else "target")


def _semantic_rule(dataset, role, aspect, reference):
    if dataset in {"CUHK-PEDES","ICFG-PEDES","RSTPReid"}:
        rule = "person"
    elif dataset in {"coco","flickr"}:
        rule = "noun"
    elif dataset == "llava":
        rule = "word"
    else:
        rule = "default"
    return SEMANTIC_RULES[rule].format(aspect=aspect,reference=reference)


def _prompt_parts(dataset, record, aspect=None):
    if dataset not in DATASETS:
        raise ValueError(f"No complete FreeRet prompt registered for dataset {dataset!r}")
    role = role_for(dataset,record)
    spec = PROMPT_SPECS[DATASETS[dataset]]
    text = safe_text(record.get("text",""))
    image = record.get("image") is not None
    inputs = getattr(spec,role+"_input")(text)
    instruction = getattr(spec,role+"_instruction")
    reference = ({"webqa": "information" if role=="target" else "question",
                  "remuq": "passage" if role=="target" else "main subject whose knowledge is required by the question",
                  "llava": "<answer>" if role=="target" else "<question>"}.get(dataset)
                 or ("<image>" if image else "<caption>"))
    if dataset in {"CUHK-PEDES","ICFG-PEDES","RSTPReid"}:
        reference = "<person image>" if image else "<person description>"
    expected_images = inputs.count(IMAGE_TOKEN)
    if record.get("_freeret_projection") and dataset in {"webqa","remuq","llava"}:
        # Overlap deliberately probes one modality, rather than restoring a pair.
        text_label = {"webqa": "information" if role=="target" else "question",
                      "remuq": "candidate passage" if role=="target" else "question",
                      "llava": "answer" if role=="target" else "question"}[dataset]
        inputs = (f"<image>: {IMAGE_TOKEN}\n" if image else "") + (f"<{text_label}>: {text}\n" if text else "")
        reference = "<image>" if image else f"<{text_label}>"
        instruction = (f"Given the {reference}, output exactly one word that follows these rules:\n"
                       " - The word should best capture the overall meaning of the given input.\n"
                       " - World knowledge may be used if helpful.\n"
                       " - The word must be a complete, standalone word — not a function word, preposition, symbol, or fragment.\n")
    elif expected_images != int(image):
        raise ValueError(f"Complete FreeRet {dataset}/{role} requires {expected_images} image(s), received {int(image)}")
    if aspect is not None:
        lines = instruction.splitlines(keepends=True)
        matches = [i for i,line in enumerate(lines) if re.search(r"\bbest captures?\b",line)]
        if len(matches)!=1:
            raise ValueError(f"No unique FreeRet semantic rule for {dataset}/{role}")
        i = matches[0]
        lines[i] = _semantic_rule(dataset,role,aspect,reference) + ("\n" if lines[i].endswith("\n") else "")
        instruction = "".join(lines)
    return CHAT_PREFIX + spec.task_description + inputs, instruction


def build_prompt(dataset, record, aspect=None):
    context, instruction = _prompt_parts(dataset, record, aspect)
    return context + instruction + CHAT_SUFFIX


def aspect_from_instruction(value):
    # Parse only project-owned instruction strings, never rendered user input.
    match = re.fullmatch(r"Summary the (.+) in above (?:image and (?:question )?sentence|image|sentence|question sentence|person sentence|text|image and question) in one word: ?",value)
    if match:
        return match[1]
    raise ValueError(f"Cannot determine a structured perspective for complete FreeRet mode: {value!r}")


def fixed_aspects(record, task, data, processor, *, concat=True):
    if not concat:
        return []
    from qwen_vl_prompts import retrieval_prompt
    return [aspect_from_instruction(value) for value in retrieval_prompt(record,task,data,processor)[2]]


def _render_prompts(processor, family, raw, groups):
    if family.startswith("qwen"):
        return raw
    reserved = {"<image>"}
    for owner in (processor, processor.tokenizer):
        token = getattr(owner, "image_token", None)
        if token:
            reserved.add(token)
    messages = [[dict(role="user", content=[*[dict(type="image") for _ in group],
        dict(type="text", text=llava_body(text, tuple(sorted(reserved, key=len, reverse=True))))])]
        for text, group in zip(raw, groups)]
    return [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]


def prepare(processor, family, records, dataset, aspect=None, *, padding_side="left", per_record_aspects=None):
    """Render one standalone full prompt (also the independent-forward reference)."""
    import torch
    from qwen_vl_encoding import ViewBatch, read_image
    processor.tokenizer.padding_side = padding_side
    raw = [build_prompt(dataset,record,aspect if per_record_aspects is None else per_record_aspects[i]) for i,record in enumerate(records)]
    groups = [[read_image(record["image"])] if record.get("image") is not None else [] for record in records]
    prompts = _render_prompts(processor, family, raw, groups)
    kwargs = dict(text=prompts,padding=True,return_tensors="pt")
    if not family.startswith("qwen"):
        # Same images-before-text native conversation used by FreeRet LLaVA.
        kwargs["add_special_tokens"] = False
    flat = [image for group in groups for image in group]
    if flat: kwargs["images"] = flat
    inputs = dict(processor(**kwargs))
    masks = inputs["attention_mask"]
    positions = torch.arange(masks.shape[-1],device=masks.device).expand_as(masks)
    ends = positions.masked_fill(~masks.bool(),-1).max(-1).values.tolist()
    return ViewBatch(inputs,[[(0,end+1)] for end in ends],prompts)


def prepare_packed(processor, family, records, dataset, aspects, *, padding_side="left"):
    """Pack native token suffixes; process the shared images only once.

    Each visible common-prefix + block token sequence equals the corresponding
    standalone prompt. Taking token suffixes handles SentencePiece/BPE merges
    at the context/instruction boundary without adding artificial delimiters.
    """
    import torch
    from qwen_vl_encoding import ViewBatch
    if not records or len(records) != len(aspects):
        raise ValueError("FreeRet records and perspective lists must have the same nonzero length")
    if padding_side not in {"left", "right"}:
        raise ValueError("FreeRet padding_side must be left or right")
    if any(len(row) != len(aspects[0]) for row in aspects):
        raise ValueError("Complete FreeRet batches require the same view count per sample")
    base = prepare(processor, family, records, dataset, padding_side=padding_side)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        raise ValueError("FreeRet packed encoding requires a tokenizer pad token")
    add_special = family.startswith("qwen")
    packed, boundaries, common_types, prompts = [], [], [], []
    image_token = getattr(processor, "image_token", "<image>")
    visual_ids = {tokenizer.convert_tokens_to_ids(token) for token in (
        "<|image_pad|>", "<|video_pad|>", image_token) if token in tokenizer.get_vocab()}
    for index, (record, per_record) in enumerate(zip(records, aspects)):
        context, _ = _prompt_parts(dataset, record)
        raw = [build_prompt(dataset, record, aspect) for aspect in [None, *per_record]]
        groups = [[None] if record.get("image") is not None else []] * len(raw)
        rendered = _render_prompts(processor, family, raw, groups)
        if family.startswith("qwen"):
            prefix = context
        else:
            # Used only while rendering a template, never tokenized or forwarded.
            marker = "FREERET_INSTRUCTION_" + uuid4().hex
            while marker in context:
                marker = "FREERET_INSTRUCTION_" + uuid4().hex
            marked = _render_prompts(processor, family, [context + marker + CHAT_SUFFIX], groups[:1])[0]
            if marked.count(marker) != 1:
                raise ValueError("Native chat template changed the FreeRet instruction boundary")
            prefix = marked.split(marker)[0]
        if any(not text.startswith(prefix) for text in rendered):
            raise ValueError("Native chat template changed the FreeRet shared context between views")
        reference = [tokenizer.encode(text, add_special_tokens=add_special) for text in rendered]
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=add_special)
        common = len(prefix_ids)
        for ids in reference:
            limit = min(common, len(ids))
            position = 0
            while position < limit and prefix_ids[position] == ids[position]:
                position += 1
            common = position
        blocks = [ids[common:] for ids in reference]
        if not common or any(not block for block in blocks):
            raise ValueError("Cannot determine the native FreeRet shared-prefix/view token boundaries")
        if any(token in visual_ids for block in blocks for token in block):
            raise ValueError("FreeRet shared prefix must contain every image/video placeholder")
        visible = base.inputs["attention_mask"][index].bool()
        actual = base.inputs["input_ids"][index][visible].tolist()
        first = blocks[0]
        if len(actual) <= len(first) or actual[-len(first):] != first:
            raise ValueError("Processor changed FreeRet view token boundaries; refusing incorrect representation positions")
        shared = actual[:-len(first)]
        sequence, spans = list(shared), []
        for block in blocks:
            start = len(sequence)
            sequence.extend(block)
            spans.append((start, len(sequence)))
        packed.append(sequence)
        boundaries.append(spans)
        if "mm_token_type_ids" in base.inputs:
            common_types.append(base.inputs["mm_token_type_ids"][index][visible][:len(shared)].tolist())
        # Logical native prompt, with the image placeholder unexpanded for logs.
        prompts.append(prefix + "".join(text[len(prefix):] for text in rendered))
    inputs = dict(base.inputs)
    size = max(map(len, packed))
    ids = inputs["input_ids"].new_full((len(records), size), tokenizer.pad_token_id)
    mask = inputs["attention_mask"].new_zeros((len(records), size))
    modality = ids.new_zeros(ids.shape) if common_types else None
    spans = []
    for index, (sequence, row_spans) in enumerate(zip(packed, boundaries)):
        offset = size - len(sequence) if padding_side == "left" else 0
        ids[index, offset:offset + len(sequence)] = torch.tensor(sequence, dtype=ids.dtype, device=ids.device)
        mask[index, offset:offset + len(sequence)] = 1
        if modality is not None:
            values = common_types[index]
            modality[index, offset:offset + len(values)] = torch.tensor(values, dtype=modality.dtype, device=modality.device)
        spans.append([(start + offset, end + offset) for start, end in row_spans])
    inputs.update(input_ids=ids, attention_mask=mask)
    if modality is not None:
        inputs["mm_token_type_ids"] = modality
    if "token_type_ids" in inputs:
        if inputs["token_type_ids"].any():
            raise ValueError("Unsupported nonzero token_type_ids for FreeRet packed views")
        inputs["token_type_ids"] = torch.zeros_like(ids)
    inputs.pop("position_ids", None)  # Recomputed below from native visual positions.
    return ViewBatch(inputs, spans, prompts)


def masked_inputs(model, batch, family):
    """Preserve causal/shared attention and reset each instruction's positions."""
    import torch
    from qwen_vl_encoding import _forward_inputs, isolate_views
    device = model.get_input_embeddings().weight.device
    batch.inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.inputs.items()}
    if family.startswith("qwen"):
        batch.inputs = _forward_inputs(model, batch.inputs)
        return isolate_views(model, batch)
    padding = batch.inputs["attention_mask"]
    size = padding.shape[-1]
    minimum = torch.finfo(model.get_input_embeddings().weight.dtype).min
    mask = torch.full((len(batch.spans), 1, size, size), minimum,
                      dtype=model.get_input_embeddings().weight.dtype, device=device)
    causal = torch.ones(size, size, dtype=torch.bool, device=device).tril()
    mask.masked_fill_(causal[None, None] & padding[:, None, None, :].bool(), 0)
    positions = padding.long().cumsum(-1) - 1
    positions.masked_fill_(~padding.bool(), 0)
    text_config = getattr(model.config, "text_config", model.config)
    window = getattr(text_config, "sliding_window", None) if getattr(text_config, "model_type", None) == "mistral" else None
    for index, spans in enumerate(batch.spans):
        common_end = spans[0][0]
        origin = positions[index, common_end].clone()
        if window is not None:
            if not isinstance(window, int) or window < 1:
                raise ValueError("Mistral sliding_window must be a positive integer or None")
            # HF accepts a supplied 4D mask verbatim, so preserve its native
            # window explicitly. Prefix positions remain contiguous.
            old_keys = torch.ones((common_end, common_end), dtype=torch.bool, device=device).tril(-window)
            mask[index, 0, :common_end, :common_end].masked_fill_(old_keys, minimum)
        for start, end in spans:
            mask[index, 0, start:end, common_end:start] = minimum
            positions[index, start:end] = origin + torch.arange(end - start, device=device)
            if window is not None:
                old_keys = positions[index, start:end, None] - positions[index, None, :end] >= window
                mask[index, 0, start:end, :end].masked_fill_(old_keys, minimum)
    return {**batch.inputs, "attention_mask": mask, "position_ids": positions}


def encode(encoder, records, aspects, *, raw_sparse_logits=False):
    import torch
    import torch.nn.functional as F
    from dense_representation import forward
    family = getattr(encoder,"family",None)
    if family is None:
        from adaptive_views import family_of
        family = family_of(encoder.model.config)
    for config in (encoder.model.config, getattr(encoder.model.config, "text_config", None)):
        if getattr(config, "_attn_implementation", None) == "flash_attention_2":
            raise ValueError("FreeRet packed views require SDPA/eager attention with a 4D mask; reload with SDPA")
    batch = prepare_packed(encoder.processor, family, records, encoder.data.dataset_name, aspects,
                           padding_side="left" if encoder.data.reps_loc == "after_pad" else "right")
    encoder.last_prompts = batch.prompts
    inputs = masked_inputs(encoder.model, batch, family)
    with torch.inference_mode():
        output = forward(encoder.model, **inputs, use_cache=False, output_hidden_states=True, return_dict=True)
    sparse, dense = [], []
    for index, spans in enumerate(batch.spans):
        ends = torch.tensor([end - 1 for _, end in spans], device=output.logits.device)
        dense.append(output.hidden_states[-1][index, ends[0].to(output.hidden_states[-1].device)])
        values = output.logits[index, ends[1:] if len(spans) > 1 else ends]
        sparse.append(values if raw_sparse_logits else torch.log1p(torch.relu(values)))
    return sparse, F.normalize(torch.stack(dense), dim=-1)


def metadata(model_args, data, training, processor, kind, *, family=None):
    from pathlib import Path
    from dense_representation import configuration
    family = family or "qwen2_5_vl"
    fields = {name:getattr(data,name,None) for name in (
        "prompt_type","random_view_seed","tbpr_type","cir_type","reps_loc","sparse_length",
        "text_sparse_length","image_sparse_length","sparse_value_type","sparse_lower_or_upper",
        "sparse_value_mean","sparse_manual","is_filtered","num_expended_tokens","llava_image_mode",
        "qwen_min_pixels","qwen_max_pixels")}
    files = []
    for root in (model_args.model_name_or_path,getattr(model_args,"lora_model_path",None) if getattr(model_args,"lora",False) else None):
        if root and Path(root).is_dir():
            for file in sorted(Path(root).iterdir()):
                if file.is_file() and (file.suffix in {".bin",".safetensors"} or file.name=="config.json"):
                    stat = file.stat(); files.append([str(file),stat.st_size,stat.st_mtime_ns])
    image = processor.image_processor
    config = dict(fields,version="freeret_full_prompt_v2",full_prompt_family=family,
        dtype=getattr(model_args,"full_prompt_dtype", "bfloat16" if getattr(training,"bf16",False) else "float16" if getattr(training,"fp16",False) else "float32"),
        model=str(model_args.model_name_or_path),dataset=data.dataset_name,split=data.dataset_split,
        task=training.task_type,method=[model_args.eol_type,model_args.calculate_type],
        checkpoint_files=files,tokenizer_class=type(processor.tokenizer).__name__,
        tokenizer_sha256=hashlib.sha256(json.dumps(processor.tokenizer.get_vocab(),sort_keys=True).encode()).hexdigest(),
        chat_template=getattr(processor,"chat_template",None) or getattr(processor.tokenizer,"chat_template",None),
        vision_processor={key:getattr(image,key,None) for key in ("size","min_pixels","max_pixels","crop_size",
            "patch_size","merge_size","image_grid_pinpoints","image_mean","image_std")})
    config = json.loads(json.dumps(config,ensure_ascii=False,default=lambda value: dict(value) if hasattr(value,"keys") else vars(value) if hasattr(value,"__dict__") else str(value)))
    config = configuration(config,True,False,True)
    digest = hashlib.sha256(json.dumps(config,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()
    return dict(config=config,fingerprint=digest,kind=kind)
