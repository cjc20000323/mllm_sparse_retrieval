"""Encoding and pipeline integration for the copied adaptive-view entrypoints."""
from __future__ import annotations

import freeret_full_prompt as full_prompt

import freeret_prompt_constraints as prompt_constraints


import copy
import importlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

import adaptive_views as views
import adaptive_view_shuffle as option_shuffle


def pipeline_module():
    return importlib.import_module("three_stage_retrieval_pipeline_adaptive_views")


def configure_processor(config, processor, reps_loc, mode=None):
    from llava_resolution import configure_processor as configure_llava_processor
    from rerank_batching import configure_rerank_processor
    processor = configure_llava_processor(config, processor, mode or getattr(processor, "llava_image_mode", "anyres"))
    configure_rerank_processor(SimpleNamespace(config=config), processor, reps_loc)
    return processor


def load_model_identity(model_path, *, dtype, reps_loc="after_pad", min_pixels=0, max_pixels=0,
                        image_mode="anyres", lora=None):
    """Reproduce encoding identity with configs/processors only, without loading weights."""
    from transformers import AutoConfig, LlavaNextProcessor
    from qwen_vl_prompts import is_native_qwen_checkpoint
    from qwen_vl_pipeline import load_qwen_processor
    if is_native_qwen_checkpoint(model_path):
        if image_mode != "anyres":
            raise ValueError("LLaVA base retrieval mode cannot be used with Qwen")
        config, processor = load_qwen_processor(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
    else:
        config = AutoConfig.from_pretrained(model_path)
        processor = LlavaNextProcessor.from_pretrained(model_path)
    processor = configure_processor(config, processor, reps_loc, image_mode)
    return views.model_identity(model_path, config, processor, dtype=dtype, lora=lora)


def instructions(selected, record):
    modality = "image and sentence" if record.get("image") is not None and record.get("text") else (
        "image" if record.get("image") is not None else "sentence")
    return [f"Summary the {row['aspect']} in above {modality} in one word: " for row in selected]


def context_and_whole(record, task, data, family=None):
    import template
    image, text = record.get("image") is not None, record.get("text", "")
    context = ("<image>\n" if image else "") + (text + "\n" if text else "")
    whole = template.fusion_prompt_for_concat if image and text else (
        template.img_prompt_for_concat if image else template.text_prompt_for_concat)
    if image and text and family == "llava_llama":
        # Preserve the existing non-Mistral fused-modal whole-input prompt.
        whole = template.img_prompt_for_concat
    if task == "tbpr" and data.tbpr_type == "type":
        whole = template.person_retrieval_img_prompt_for_concat if image else template.person_retrieval_text_prompt_for_concat
    return context, whole


class AdaptiveEncoder:
    """One packed forward: one whole-input dense view and K isolated sparse views."""
    def __init__(self, model, processor, model_args, data, task, selection):
        self.model, self.processor, self.model_args, self.data, self.task = model, processor, model_args, data, task
        from dense_representation import configure_model, enabled
        configure_model(model, enabled(model_args), getattr(model_args, "model_name_or_path", ""))
        self.family = views.family_of(model.config)
        self.selected = selection["selected"]
        count = len(views.view_pool(views.pool_mode(selection)))
        if not 1 <= len(self.selected) <= count:
            raise ValueError(f"Adaptive encoding requires 1..{count} selected views")
        self.last_prompts = []
        configure_processor(model.config, processor, data.reps_loc)

    def prepare(self, records):
        if full_prompt.enabled(self.model_args):
            return full_prompt.prepare(self.processor,self.family,records,self.data.dataset_name,
                padding_side="left" if self.data.reps_loc=="after_pad" else "right")
        from qwen_vl_encoding import prepare_views, ViewBatch, read_image
        contexts, blocks = [], []
        for record in records:
            context, whole = context_and_whole(record, self.task, self.data, self.family)
            from freeret_prompt_constraints import augment
            row = [augment(block, self.model_args, self.data.dataset_name, record)
                   for block in [whole, *instructions(self.selected, record)]]
            contexts.append(prompt_constraints.augment_context(context, self.model_args, self.data.dataset_name))
            blocks.append(row)
        if self.family.startswith("qwen"):
            images = [[record["image"]] if record.get("image") is not None else [] for record in records]
            return prepare_views(self.processor, contexts, blocks, images,
                                 padding_side="left" if self.data.reps_loc == "after_pad" else "right")
        import template
        mistral = self.family == "llava_mistral"
        wrapper = template.llava_mistral_template_content_element if mistral else template.llama3_template_content_element
        prefix = "[INST]" if mistral else "<|start_header_id|>user<|end_header_id|>\n\n"
        prompts = [prefix + context + "".join(wrapper.format(block) for block in row)
                   for context, row in zip(contexts, blocks)]
        kwargs = dict(text=prompts, padding=True, return_tensors="pt", add_special_tokens=mistral)
        images = [read_image(record["image"]) for record in records if record.get("image") is not None]
        if images:
            kwargs["images"] = images
        inputs = dict(self.processor(**kwargs))
        start_token, end_token = ("<s>", "</s>") if mistral else ("<|begin_of_text|>", "<|end_of_text|>")
        vocab = self.processor.tokenizer.get_vocab()
        if start_token not in vocab or end_token not in vocab:
            raise ValueError(f"Missing LLaVA {self.family} view delimiters in the checkpoint tokenizer")
        spans = []
        block_count = len(self.selected) + 1
        for row in inputs["input_ids"]:
            starts = (row == vocab[start_token]).nonzero(as_tuple=True)[0].tolist()[-block_count:]
            ends = (row == vocab[end_token]).nonzero(as_tuple=True)[0].tolist()[-block_count:]
            if len(starts) != block_count or len(ends) != block_count or any(start >= end for start, end in zip(starts, ends)):
                raise ValueError("LLaVA processor changed adaptive view boundaries; refusing incorrect EOL positions")
            spans.append([(start, end + 1) for start, end in zip(starts, ends)])
        return ViewBatch(inputs, spans, prompts)

    @torch.inference_mode()
    def encode(self, records):
        import freeret_full_prompt as full_prompt
        if full_prompt.enabled(self.model_args):
            return full_prompt.encode(self, records, [[view["aspect"] for view in self.selected] for _ in records])
        from qwen_vl_encoding import forward_views
        batch = self.prepare(records)
        self.last_prompts = batch.prompts
        if self.family.startswith("qwen"):
            logits, hidden = forward_views(self.model, batch, isolated=True)
        else:
            device = self.model.get_input_embeddings().weight.device
            inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.inputs.items()}
            padding = inputs["attention_mask"]
            length = padding.shape[1]
            dtype = self.model.get_input_embeddings().weight.dtype
            allowed = torch.ones(length, length, dtype=torch.bool, device=device).tril()
            mask = torch.full((len(records), 1, length, length), torch.finfo(dtype).min, dtype=dtype, device=device)
            mask.masked_fill_(allowed[None, None] & padding[:, None, None, :].bool(), 0)
            for row, spans in enumerate(batch.spans):
                shared_end = spans[0][0]
                for start, end in spans:
                    mask[row, 0, start:end, shared_end:start] = torch.finfo(dtype).min
            positions = padding.long().cumsum(-1) - 1
            positions.masked_fill_(~padding.bool(), 0)
            output = self.model(**{**inputs, "attention_mask": mask, "position_ids": positions},
                                use_cache=False, output_hidden_states=True, return_dict=True)
            logits, hidden = [], []
            for row, spans in enumerate(batch.spans):
                # The boundary token is EOS; the representation immediately
                # before it matches the existing LLaVA concat EOL definition.
                ends = torch.tensor([end - 2 for _, end in spans], device=output.logits.device)
                logits.append(output.logits[row, ends])
                hidden.append(output.hidden_states[-1][row, ends.to(output.hidden_states[-1].device)])
        sparse = [torch.log1p(torch.relu(row[1:])) for row in logits]
        dense = F.normalize(torch.stack([row[0] for row in hidden]), dim=-1)
        return sparse, dense


def sparse_vectors(encoder, records, logits):
    if encoder.family.startswith("qwen"):
        from qwen_vl_encoding import SparseEncoder
        if not hasattr(encoder, "sparse_encoder"):
            encoder.sparse_encoder = SparseEncoder(encoder.processor.tokenizer, encoder.data, concat=True)
        return [encoder.sparse_encoder.vector(row, record) for row, record in zip(logits, records)]
    from encode import get_text_valid_disassemble_tokens_values, get_img_valid_disassemble_tokens_values
    vocabulary = {value: key for key, value in encoder.processor.tokenizer.get_vocab().items()}
    result = []
    for row, record in zip(logits, records):
        if record.get("image") is None:
            tokens, weights = get_text_valid_disassemble_tokens_values(
                record.get("text", ""), encoder.processor.tokenizer, row, vocabulary,
                encoder.data, None, None, encoder.model_args)
        else:
            tokens, weights = get_img_valid_disassemble_tokens_values(
                encoder.processor, row, vocabulary, encoder.data, None, None, encoder.model_args)
        vector = {}
        for token, weight in zip(tokens, weights):
            weight = int(weight)
            if not token or any(char.isspace() for char in token) or weight <= 0:
                continue
            if token not in vector or encoder.data.sparse_value_type == "replace":
                vector[token] = weight
            elif encoder.data.sparse_value_type == "sum":
                vector[token] += weight
            else:
                vector[token] = max(vector[token], weight)
        if encoder.data.sparse_value_mean:
            count = len(row)
            vector = {key: value // count for key, value in vector.items() if value // count > 0}
        result.append(vector)
    return result


def encoding_configuration(model_args, data, task, kind, identity, selection):
    prompt_constraints.validate_selection(selection, model_args)
    import template
    fields = {name: getattr(data, name) for name in (
        "reps_loc", "sparse_length", "text_sparse_length", "image_sparse_length", "sparse_value_type",
        "sparse_lower_or_upper", "sparse_value_mean", "sparse_manual", "is_filtered", "num_expended_tokens",
        "tbpr_type", "cir_type",
    )}
    renderers = {name: getattr(template, name) for name in (
        "img_prompt_for_concat", "text_prompt_for_concat", "fusion_prompt_for_concat",
        "person_retrieval_img_prompt_for_concat", "person_retrieval_text_prompt_for_concat",
        "llama3_template_text_prefix", "llama3_template_content_element",
        "llava_mistral_template_text_prefix", "llava_mistral_template_content_element",
    )}
    configuration = dict(version=views.VERSION, model=identity, dataset=data.dataset_name, split=data.dataset_split,
                         task=task, kind=kind, method=[model_args.eol_type, model_args.calculate_type],
                         selection_modality=selection["selection_modality"],
                         selected=selection["selected"], views_fingerprint=selection["views_fingerprint"],
                         renderer=views.digest(renderers), encoding=fields)
    if views.is_pmi_selection(selection):
        configuration["selection_calibration"] = selection["calibration"]["configuration"]
    from dense_representation import configuration as dense_configuration, enabled
    return views.examples.add_configuration(views.add_pool_mode(option_shuffle.add_metadata(
        dense_configuration(configuration, enabled(model_args), prompt_constraints.enabled(model_args), full_prompt.enabled(model_args)),
        selection.get(option_shuffle.FIELD)), views.pool_mode(selection)), selection.get(views.examples.FIELD))


def metadata(model_args, data, task, kind, identity, selection):
    configuration = encoding_configuration(model_args, data, task, kind, identity, selection)
    return dict(configuration=configuration, fingerprint=views.digest(configuration))


def artifact_dirs(model_args, data, training, selection):
    from qwen_vl_pipeline import artifact_dirs as base_dirs
    from llava_resolution import artifact_dir
    from dense_representation import mode_path, enabled, original_path
    bases = base_dirs(model_args, data, training)
    if enabled(model_args):
        bases = [original_path(path) for path in bases]
        if full_prompt.enabled(model_args) and getattr(data,"llava_image_mode","anyres") == "base":
            bases = [path.with_name(path.name.removesuffix("_llava_base")) for path in bases]
    return [mode_path(artifact_dir(option_shuffle.mark_path(path / "adaptive_views" / selection["views_fingerprint"][:16], selection.get(option_shuffle.FIELD)),
                                  getattr(data, "llava_image_mode", "anyres")), enabled(model_args), prompt_constraints.enabled(model_args), full_prompt.enabled(model_args))
            for path in bases]


def make_args(model_args, data, training, options):
    api = pipeline_module()
    task, kind = training.task_type, training.encode_type
    query = "image" if kind == "text" else "text"
    if task == "tbpr":
        query = "text"
    elif task == "t2it":
        query = "text"
    elif task == "it2t":
        query = "image"
    args = api._parser().parse_args(["--dataset-name", data.dataset_name, "--task-type", task, "--query-type", query])
    args.adaptive_views = True
    args.use_freeret_dense = getattr(model_args, "use_freeret_dense", False)
    args.use_freeret_full_prompt = getattr(model_args, "use_freeret_full_prompt", False)
    args.use_freeret_prompt_constraints = getattr(model_args, "use_freeret_prompt_constraints", False)
    args.data_root = options.adaptive_data_root
    args.retrieval_model_name_or_path = model_args.model_name_or_path
    args.retrieval_lora_model_path = model_args.lora_model_path if getattr(model_args, "lora", False) else None
    args.retrieval_dtype = "bfloat16" if training.bf16 else "float16" if training.fp16 else "float32"
    args.retrieval_device_map = "none"
    args.retrieval_reps_loc = data.reps_loc
    args.retrieval_batch_size = data.per_device_batch_size
    args.retrieval_qwen_min_pixels = getattr(data, "qwen_min_pixels", 0)
    args.retrieval_qwen_max_pixels = getattr(data, "qwen_max_pixels", 0)
    args.retrieval_llava_image_mode = getattr(data, "llava_image_mode", "anyres")
    for name in ("dataset_split", "prompt_type", "random_view_seed", "sparse_length", "text_sparse_length",
                 "image_sparse_length", "num_expended_tokens", "sparse_value_type", "sparse_lower_or_upper",
                 "sparse_value_mean", "is_filtered", "tbpr_type", "cir_type"):
        setattr(args, name, getattr(data, name))
    return args


def source_pools(args, api, selection_modality="both"):
    from qwen_vl_pipeline import dataset_records
    active = views.enabled_modalities(selection_modality)
    dataset = api.build_dataset(args)
    if args.task_type in {"ir", "tbpr"}:
        text_records = ([dict(id=f"text:{key}", text=api._plain_text(value)) for key, value in dataset.text_dict.items()]
                        if "text" in active else [])
        image_records = dataset_records(dataset, args.task_type, "image") if "image" in active else []
    else:
        query = dataset_records(dataset, args.task_type, "query")
        corpus = dataset_records(dataset, args.task_type, "corpus")
        all_records = [dict(record, id=f"{role}:{record['id']}") for role, records in (("query", query), ("corpus", corpus)) for record in records]
        if prompt_constraints.active(args):
            for record in all_records:
                record["_freeret_role"] = "query" if record["id"].startswith("query:") else "target"
        text_records = ([dict(id=record["id"], text=api._plain_text(record["text"]),
                              **({"_freeret_role": record["_freeret_role"]} if "_freeret_role" in record else {}))
                         for record in all_records if record.get("text")] if "text" in active else [])
        image_records = [record for record in all_records if record.get("image") is not None] if "image" in active else []
    # Preserve source IDs and stable input ordering. Text duplicates do not get
    # extra votes simply because several records repeat the same sentence.
    unique_texts = {}
    for record in sorted(text_records, key=lambda record: str(record["id"])):
        if record["text"].strip():
            unique_texts.setdefault(record["text"], record)
    unique_images = {}
    for record in sorted(image_records, key=lambda record: str(record["id"])):
        image = record["image"]
        key = ("path", str(image)) if isinstance(image, (str, Path)) else ("id", str(record["id"]))
        unique_images.setdefault(key, dict(id=str(record["id"]), image=image,
            **({"_freeret_role": record["_freeret_role"]} if "_freeret_role" in record else {})))
    for name, pool in (("text", unique_texts), ("image", unique_images)):
        if name in active and not pool:
            raise ValueError(f"Adaptive view selection requires a nonempty {name} sample pool")
    return list(unique_texts.values()), list(unique_images.values())


def root_action(rank, action):
    packet = [None]
    if rank == 0:
        try:
            packet[0] = dict(value=action(), error=None)
        except Exception as exc:
            packet[0] = dict(value=None, error=f"{type(exc).__name__}: {exc}")
    if dist.is_initialized():
        dist.broadcast_object_list(packet, src=0)
    if packet[0]["error"]:
        raise ValueError(packet[0]["error"])
    return packet[0]["value"]


def prepare_outputs(roots):
    roots = [path.resolve() for path in roots]
    dense, sparse = roots
    collection = sparse / "collection"
    indexes = [sparse / "index", collection / "index"]
    files = [*(dense.glob("corpus_*.pkl")), *(collection.glob("corpus_*.jsonl")),
             dense / views.MANIFEST, sparse / views.MANIFEST]
    for path in [*files, collection, *indexes]:
        if path.resolve() != path:
            raise ValueError(f"Refusing linked adaptive artifact: {path}")
        if path in files and path.exists() and not path.is_file():
            raise ValueError(f"Expected an adaptive artifact file: {path}")
        if path not in files and path.exists() and not path.is_dir():
            raise ValueError(f"Expected an adaptive artifact directory: {path}")
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
    for path in files:
        path.unlink(missing_ok=True)
    for path in indexes:
        if path.exists():
            # Checked absolute targets are inside this explicit adaptive leaf;
            # backups are always outside collection's indexer input directory.
            path.rename(sparse / f"index.stale_{uuid4().hex}")
    collection.mkdir(exist_ok=True)


def worker_errors(error):
    errors = [error]
    if dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
    if any(errors):
        raise ValueError("Adaptive corpus encoding failed: " + "; ".join(
            f"rank {rank}: {message}" for rank, message in enumerate(errors) if message))


def write_shard(encoder, records, data, roots, rank):
    dense_parts, identifiers = [], []
    from tqdm import tqdm
    with (roots[1] / "collection" / f"corpus_{rank}.jsonl").open("w", encoding="utf-8") as stream:
        for start in tqdm(range(0, len(records), data.per_device_batch_size), desc="Adaptive corpus encoding"):
            batch = records[start:start + data.per_device_batch_size]
            logits, dense = encoder.encode(batch)
            vectors = sparse_vectors(encoder, batch, logits)
            dense_parts.append(dense.float().cpu().numpy())
            identifiers.extend(str(record["id"]) for record in batch)
            for record, vector in zip(batch, vectors):
                stream.write(json.dumps(dict(id=str(record["id"]), contents="", vector=vector), ensure_ascii=False) + "\n")
    hidden_size = getattr(encoder.model.config, "text_config", encoder.model.config).hidden_size
    dense = np.concatenate(dense_parts) if dense_parts else np.empty((0, hidden_size), dtype="float32")
    with (roots[0] / f"corpus_{rank}.pkl").open("wb") as stream:
        pickle.dump((dense, identifiers), stream)


def run_encode(model_args, data, training, options, *, selection_strategy=None):
    from qwen_vl_pipeline import init_process, dataset_records
    if training.task_type not in {"ir", "tbpr", "t2it", "it2t"}:
        raise ValueError("Adaptive views support ITR/ir, TBPR, t2it and it2t only")
    if (model_args.eol_type, model_args.calculate_type) != ("disassembleeol_separate", "concat") or data.sparse_manual:
        raise ValueError("Adaptive views require disassembleeol_separate+concat and sparse_manual=false")
    if (getattr(model_args, "use_output_embedding_cluster", False)
            or getattr(data, "sparse_type", "single") != "single"
            or getattr(data, "sparse_logits_multiply_view_count", False)):
        raise ValueError("Adaptive views require single sparse representations without output clustering or x5 ablation")
    if data.sparse_value_type not in {"replace", "sum", "max"} or data.sparse_lower_or_upper not in {"lower", "upper"}:
        raise ValueError("Unsupported adaptive sparse aggregation or letter-case policy")
    if data.reps_loc not in {"before_pad", "after_pad"} or data.num_expended_tokens < 0:
        raise ValueError("Invalid adaptive representation location or expansion count")
    if getattr(data, "encode_is_query", False) or any(getattr(data, name, False) for name in (
            "use_generated_validation_data", "use_few_shot_examples", "use_source_dataset_subset", "prompt_generation")):
        raise ValueError("Adaptive encode selects views from original datasets and encodes corpus artifacts only")
    kinds = {"ir": {"image", "text"}, "tbpr": {"image", "text"}, "t2it": {"corpus"}, "it2t": {"corpus"}}
    if training.encode_type not in kinds[training.task_type]:
        raise ValueError(f"Invalid adaptive corpus encode_type={training.encode_type}")
    if min(data.per_device_batch_size, data.sparse_length, data.text_sparse_length, data.image_sparse_length) < 1:
        raise ValueError("Encoding batch/sparse lengths must be positive")
    requested_mode = options.view_selection_modality
    views.resolve_modality(requested_mode)
    views.examples.validate_options(options)
    requested_pmi = getattr(options, "view_selection_pmi", None)
    views.resolve_pmi(requested_pmi)
    candidate_mode = views.options_pool_mode(options)
    pool = views.view_pool(candidate_mode)
    rank, world, device = init_process()
    if option_shuffle.enabled(options):
        options.view_selection_shuffle_run_id = root_action(rank, lambda: option_shuffle.validate_run_id(
            options.view_selection_shuffle_run_id) if options.view_selection_shuffle_run_id else option_shuffle.new_run_id())
        training.output_dir = str(option_shuffle.mark_path(training.output_dir, {"run_id":options.view_selection_shuffle_run_id}))
    explicit_path = Path(options.view_selection_file).expanduser() if options.view_selection_file else None
    def existing_selection():
        if explicit_path is not None and explicit_path.exists():
            if option_shuffle.enabled(options):
                raise ValueError("Shuffled encoding starts a new experiment; omit --view_selection_file or choose a nonexistent file")
            if not explicit_path.is_file():
                raise ValueError(f"Expected a view selection file: {explicit_path}")
            selection = views.load_selection(explicit_path, dataset=data.dataset_name,
                                             selection_modality=requested_mode, view_selection_pmi=requested_pmi,
                                             view_selection_hierarchical=candidate_mode is not None,
                                             view_example_mode=options.view_example_mode,
                                             view_example_max_tokens=options.view_example_max_tokens)
            if selection.get(option_shuffle.FIELD) is not None:
                raise ValueError("A shuffled selection file requires shuffled encoding; choose a new file for a new shuffled experiment")
            if selection_strategy is not None:
                selection_strategy.validate(selection, inherit=True)
            return selection
        return None
    # Inherit an explicitly reused file's mode and reject conflicts before loading weights.
    preselected = root_action(rank, existing_selection) if explicit_path is not None else None
    mode = preselected["selection_modality"] if preselected is not None else views.resolve_modality(requested_mode)
    pmi = views.resolve_pmi(requested_pmi, selection=preselected)
    api = pipeline_module()
    args = make_args(model_args, data, training, options)
    args.retrieval_device = str(device)
    context = api.DistributedContext(rank, device.index or 0, world)
    runtime, error = None, None
    try:
        runtime = api._load_retrieval_runtime(args, context)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    try:
        worker_errors(error)
        configure_processor(runtime.encoder.config, runtime.processor, data.reps_loc)
        identity = views.model_identity(model_args.model_name_or_path, runtime.encoder.config, runtime.processor,
                                        dtype=args.retrieval_dtype, lora=args.retrieval_lora_model_path)
        sampling = views.selection_sampling(options, mode, model_config=runtime.encoder.config) if preselected is None else preselected["sampling"]
        if preselected is not None:
            views.require_equal(dict(examples=preselected.get(views.examples.FIELD)),
                                dict(examples=views.examples.configuration(options, runtime.encoder.config)), "View-example configuration")
        source_split = options.view_selection_split or data.dataset_split
        shuffle = option_shuffle.for_options(options,identity,data.dataset_name,pool)
        selection_key = views.selection_cache_key(
            identity, data.dataset_name, source_split, mode, sampling, args.data_root, pmi=pmi,
            strategy=selection_strategy.configuration if selection_strategy is not None else None, shuffle=shuffle,
            pool_mode=candidate_mode)
        selection_path = explicit_path if explicit_path is not None else (
            views.selection_cache_path(data.sparse_output_dir, model_args.model_name_or_path,
                                       data.dataset_name, selection_key))
        def select():
            if preselected is not None:
                views.require_equal(preselected["model"], identity, "View selection model/processor")
                views.log_selection(preselected, selection_path, reused=True)
                if selection_strategy is not None:
                    selection_strategy.report(preselected, reused=True)
                return preselected
            if selection_path.is_file():
                selection = views.load_selection(selection_path, identity=identity, dataset=data.dataset_name,
                                                 selection_modality=mode, view_selection_pmi=pmi,
                                                 view_selection_hierarchical=candidate_mode is not None,
                                                 view_example_mode=options.view_example_mode,
                                                 view_example_max_tokens=options.view_example_max_tokens)
                views.require_equal(selection["sampling"], sampling, "Selection sampling")
                if selection.get(option_shuffle.FIELD) != shuffle:
                    raise ValueError("Selection cache belongs to a different option shuffle experiment")
                if selection_strategy is not None:
                    selection_strategy.validate(selection)
                views.log_selection(selection, selection_path, reused=True)
                if selection_strategy is not None:
                    selection_strategy.report(selection, reused=True)
                return selection
            selection_args = copy.deepcopy(args)
            selection_args.dataset_split = source_split
            if shuffle is not None:
                mapping_path = selection_path.with_name("option_mapping.json")
                if mapping_path.exists() and json.loads(mapping_path.read_text(encoding="utf-8")) != shuffle:
                    raise ValueError(f"Refusing to overwrite a different option mapping: {mapping_path}")
                views.atomic_json(mapping_path,shuffle)
            texts, images = source_pools(selection_args, api, mode)
            if selection_strategy is None:
                selection = views.choose_views(runtime.encoder, runtime.processor, identity, data.dataset_name,
                                               source_split, texts, images, options)
            else:
                selection = selection_strategy.choose(
                    runtime.encoder, runtime.processor, identity, data.dataset_name, source_split, texts, images,
                    options, report_path=selection_path.with_name("view_selection_diagnostics.json"))
            views.atomic_json(selection_path, selection)
            views.atomic_json(selection_path.with_name("view_pool.json"), selection["pool"])
            views.log_selection(selection, selection_path)
            return selection
        selection = root_action(rank, select)
        roots = artifact_dirs(model_args, data, training, selection)
        root_action(rank, lambda: prepare_outputs(roots))
        error = None
        try:
            encoder = AdaptiveEncoder(runtime.encoder, runtime.processor, model_args, data, training.task_type, selection)
            dataset = api.build_dataset(args)
            records = dataset_records(dataset, training.task_type, training.encode_type, prompt_constraints_mode=prompt_constraints.enabled(model_args))[rank::world]
            expected = metadata(model_args, data, training.task_type, training.encode_type, identity, selection)
            write_shard(encoder, records, data, roots, rank)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        worker_errors(error)
        if dist.is_initialized():
            dist.barrier()
        def complete():
            completion = dict(expected, num_shards=world, selection_file=str(selection_path.resolve()),
                              selection_fingerprint=selection["fingerprint"], examples=encoder.last_prompts[:1])
            for root in roots:
                views.atomic_json(root / views.MANIFEST, completion)
                from dense_representation import write_manifest, enabled
                write_manifest(root, enabled(model_args), prompt_constraints.enabled(model_args), full_prompt.enabled(model_args))
            if enabled(model_args):
                from dense_representation import reuse_sparse_index
                reuse_sparse_index(roots[1], manifest_name=views.MANIFEST)
            print("View selection:", selection_path.resolve())
            print("View selection mode:", selection["selection_modality"])
            print("Selected views:", [(row["label"], row["aspect"]) for row in selection["selected"]])
            print("Adaptive dense:", roots[0])
            print("Adaptive sparse:", roots[1])
            print(f'Build index: python -B src/adaptive_views.py index --sparse-root "{roots[1]}"')
        root_action(rank, complete)
        return roots, selection
    finally:
        api._release_retrieval_runtime(runtime)


def encode_queries(args, runtime, api, *, eol_type, calculate_type, sparse_manual, progress_prefix,
                   requested_query_ids=None, timing=None):
    if (eol_type, calculate_type, sparse_manual) != ("disassembleeol_separate", "concat", False):
        raise ValueError("Adaptive queries require sparse_concat")
    configure_processor(runtime.encoder.config, runtime.processor, args.retrieval_reps_loc)
    identity = views.model_identity(args.retrieval_model_name_or_path, runtime.encoder.config, runtime.processor,
                                    dtype=args.retrieval_dtype, lora=args.retrieval_lora_model_path)
    selection = views.load_selection(args.view_selection_file, identity=identity, dataset=args.dataset_name)
    settings = api._qwen_encoding_settings(args, eol_type, calculate_type, sparse_manual)
    key = ("adaptive", views.digest(settings), tuple(requested_query_ids) if requested_query_ids is not None else None)
    if key in runtime.query_cache:
        return runtime.query_cache[key]
    model_args, data = api._retrieval_namespaces(args, eol_type=eol_type, calculate_type=calculate_type, sparse_manual=False)
    from qwen_vl_pipeline import corpus_kind
    expected = metadata(model_args, data, args.task_type, corpus_kind(args.task_type, args.query_type), identity, selection)
    expected["pipeline_settings"] = settings
    encoder = AdaptiveEncoder(runtime.encoder, runtime.processor, model_args, data, args.task_type, selection)
    dataset = api.build_dataset(args)
    records = api._qwen_query_records(args, api._select_query_records(api._query_records(args, dataset), requested_query_ids))
    ids, dense_parts, topics = [], [], {}
    with api._compute_scope(timing, "query_encoding", model=runtime.encoder, device=runtime.device):
        for start in api._progress(range(0, len(records), args.retrieval_batch_size), desc=f"{progress_prefix} adaptive queries", unit="batch"):
            batch = records[start:start + args.retrieval_batch_size]
            logits, dense = encoder.encode(batch)
            vectors = sparse_vectors(encoder, batch, logits)
            ids.extend(record["id"] for record in batch)
            dense_parts.append(dense.float().cpu().numpy())
            topics.update({record["id"]: api._vector_to_topic(vector) for record, vector in zip(batch, vectors)})
    if not dense_parts:
        raise ValueError("No adaptive queries to encode")
    expected["query_examples"] = encoder.last_prompts[:1]
    encoded = api.EncodedQueries(ids, np.concatenate(dense_parts), topics, expected)
    runtime.query_cache[key] = encoded
    return encoded


def validate_encoded(args, encoded, paths, api, *, eol_type, calculate_type, sparse_manual):
    if not encoded.query_ids:
        return
    expected = encoded.qwen_metadata
    if expected is None:
        raise ValueError("Missing adaptive query configuration")
    views.require_equal(expected["pipeline_settings"], api._qwen_encoding_settings(
        args, eol_type, calculate_type, sparse_manual), "Adaptive query reuse")
    for path, dense in paths:
        actual = views.validate_artifact(path, expected)
        if dense and len(api._find_corpus_pickles(Path(path))) != actual["num_shards"]:
            raise ValueError("Adaptive dense shard count differs from its completion manifest")


def preflight(args, api, *, mark_output=True):
    """Validate candidate artifacts; only the main experiment marks the result path."""
    if args.task_type not in {"ir", "tbpr", "t2it", "it2t"} or args.stage1_method != "sparse_concat":
        raise ValueError("Adaptive views require ITR/ir, TBPR or fused-modal sparse_concat retrieval")
    if args.sparse_logits_multiply_view_count:
        raise ValueError("Adaptive views do not support the x5 sparse ablation")
    if any(getattr(args, key) for key in ("stage1_run", "stage1_command", "stage2_run", "ranker_run", "val_stage1_run")):
        raise ValueError("Adaptive mode executes all three stages; precomputed runs are supported with the mode disabled")
    if not args.view_selection_file:
        raise ValueError("--adaptive-views requires --view-selection-file from encode_adaptive_views.py")
    if not args.stage1_sparse_index or not args.stage2_dense_index or not args.stage2_sparse_index:
        raise ValueError("Adaptive views require Stage-1 sparse and Stage-2 dense/sparse artifacts")
    if args.rerank_template not in {"relevant", "what_caption_generation"}:
        raise ValueError("Adaptive mode uses what_caption_generation reranking")
    args.rerank_template = "what_caption_generation"
    from qwen_vl_pipeline import corpus_kind
    identity = load_model_identity(args.retrieval_model_name_or_path, dtype=args.retrieval_dtype,
                                   min_pixels=args.retrieval_qwen_min_pixels, max_pixels=args.retrieval_qwen_max_pixels,
                                   reps_loc=args.retrieval_reps_loc,
                                   image_mode=getattr(args, "retrieval_llava_image_mode", "anyres"),
                                   lora=args.retrieval_lora_model_path)
    selection = views.load_selection(args.view_selection_file, identity=identity, dataset=args.dataset_name)
    shuffle = selection.get(option_shuffle.FIELD)
    requested_run = getattr(args,"view_selection_run_id",None)
    if requested_run and (shuffle is None or shuffle["run_id"] != option_shuffle.validate_run_id(requested_run)):
        raise ValueError("Selection file does not belong to the requested VIEW_SELECTION_RUN_ID")
    if mark_output and shuffle is not None and args.output_dir is not None:
        args.output_dir = str(option_shuffle.mark_path(args.output_dir,shuffle))
    model_args, data = api._retrieval_namespaces(args, eol_type="disassembleeol_separate", calculate_type="concat", sparse_manual=False)
    expected = metadata(model_args, data, args.task_type, corpus_kind(args.task_type, args.query_type), identity, selection)
    for path, dense in ((args.stage1_sparse_index, False), (args.stage2_dense_index, True), (args.stage2_sparse_index, False)):
        actual = views.validate_artifact(path, expected)
        if dense:
            if len(api._find_corpus_pickles(Path(path))) != actual["num_shards"]:
                raise ValueError("Adaptive dense shard count mismatch")
        else:
            from qwen_vl_paths import resolve_sparse_index
            resolve_sparse_index(path)
    return selection
