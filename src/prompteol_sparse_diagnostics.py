"""Opt-in token diagnostics for the ordinary LLaVA PromptEOL encoder."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Optional


@dataclass
class PrompteolSparsePrintArguments:
    print_sparse_diagnostics: bool = field(default=False, metadata={
        "help": "Opt in to PromptEOL diagnostics; non-manual text reports its retained original tokens"})
    print_sparse_top_k: int = field(default=10)
    print_sparse_format: str = field(default="json", metadata={"choices": ("text", "json")})
    print_sparse_timing: str = field(default="immediate", metadata={"choices": ("immediate", "end")})
    print_sparse_sample_id: Optional[str] = None
    print_sparse_log_dir: Optional[str] = None
    diagnostic_data_root: Optional[str] = field(default=None, metadata={
        "help": "Dataset root containing data/flickr or data/coco; only used in diagnostic mode"})


def validate_configuration(model_args, data, training, options):
    if not options.print_sparse_diagnostics:
        return
    if (model_args.eol_type != "prompteol" or model_args.calculate_type != "separate"
            or data.llava_image_mode != "base"):
        raise ValueError("--print_sparse_diagnostics requires LLaVA base, prompteol and separate encoding")
    if training.task_type != "ir" or training.encode_type not in {"text", "image"}:
        raise ValueError("PromptEOL diagnostics support IR text/image corpus encoding")
    if data.dataset_name not in {"flickr", "coco"}:
        raise ValueError("PromptEOL diagnostics support Flickr and COCO datasets")
    if (getattr(model_args, "use_output_embedding_cluster", False)
            or getattr(data, "sparse_type", "single") != "single"
            or getattr(model_args, "use_freeret_full_prompt", False)):
        raise ValueError("PromptEOL diagnostics require an unclustered single vocabulary and the ordinary prompt")
    if getattr(data, "print_sparse", False):
        if type(options.print_sparse_top_k) is not int or options.print_sparse_top_k < 1:
            raise ValueError("--print_sparse_top_k must be a positive integer")
        if options.print_sparse_format not in {"text", "json"}:
            raise ValueError("--print_sparse_format must be text or json")
        if options.print_sparse_timing not in {"immediate", "end"}:
            raise ValueError("--print_sparse_timing must be immediate or end")


def configure_data_root(options):
    if not options.print_sparse_diagnostics or not options.diagnostic_data_root:
        return
    import dataset
    prefix = str(Path(options.diagnostic_data_root).expanduser().resolve()).rstrip("/\\") + "/"
    dataset.dataset_path_prefix = prefix
    dataset.flickr_file_path = prefix + "data/flickr/"
    dataset.coco_file_path = prefix + "data/coco/"


def quiet_legacy_printing(data, options):
    """Keep sparse extraction identical while the new printer owns its output."""
    if not options.print_sparse_diagnostics or not getattr(data, "print_sparse", False):
        return data
    quiet = copy.copy(data)
    quiet.print_sparse = False
    return quiet


class PrompteolSparsePrinter:
    def __init__(self, processor, model_args, data, training, options, dataset):
        # Import the existing diagnostics only for the explicitly enabled mode.
        from adaptive_view_encoding import print_sparse_views, sparse_print_log, finish_sparse_print, worker_errors
        self._print = print_sparse_views
        self._log = sparse_print_log
        self._finish = finish_sparse_print
        self.kind = training.encode_type
        identifiers = dataset.text_id_list if self.kind == "text" else dataset.img_id_list
        target = options.print_sparse_sample_id
        error = None
        if target:
            count = sum(str(identifier) == str(target) for identifier in identifiers)
            if count != 1:
                error = (f"--print_sparse_sample_id={target!r} matched {count} samples; expected exactly one. "
                         "Check the dataset, split and encode_type.")
        worker_errors(error)
        self.adapter = SimpleNamespace(
            processor=processor, model_args=model_args, data=copy.copy(data),
            selected=[dict(label="whole", aspect="prompteol")],
            print_sparse_top_k=options.print_sparse_top_k, print_sparse_format=options.print_sparse_format,
            print_sparse_timing=options.print_sparse_timing,
            print_sparse_sample_id=target, print_sparse_log_dir=options.print_sparse_log_dir,
            sparse_print_prefix="[prompteol sparse]", sparse_print_log_prefix="[prompteol sparse log]",
        )
        self._context = None

    def __enter__(self):
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        self._context = self._log(self.adapter, rank)
        self._context.__enter__()
        return self

    def __exit__(self, *error):
        return self._context.__exit__(*error)

    def finish(self):
        self._finish(self.adapter)

    @property
    def uses_retained_text(self):
        return self.kind == "text" and not getattr(self.adapter.data, "sparse_manual", False)

    def record_retained_text(self, identifier, text, logits, retained_ids):
        if not self.uses_retained_text:
            raise ValueError("Retained-text diagnostics require text encoding with sparse_manual=false")
        if logits.ndim != 1:
            raise ValueError("Retained-text diagnostics require one vocabulary row")
        record = dict(id=str(identifier), text=text, _sparse_retained_token_ids=list(retained_ids))
        self._print(self.adapter, [record], logits.reshape(1, 1, -1))

    def record_batch(self, identifiers, texts, image_paths, logits):
        if self.uses_retained_text:
            raise ValueError("Non-manual text diagnostics require retained token IDs from sparse extraction")
        # encode_data already applies log1p(relu); never activate or normalize again.
        count = len(identifiers)
        if logits.ndim != 2 or logits.shape[0] != count:
            raise ValueError("PromptEOL diagnostics require one vocabulary row per sample")
        values = texts if self.kind == "text" else image_paths
        if len(values) != count:
            raise ValueError("PromptEOL diagnostic metadata does not match its samples")
        records = []
        for identifier, value in zip(identifiers, values):
            if self.kind == "text":
                records.append(dict(id=str(identifier), text=value))
            else:
                records.append(dict(id=str(identifier), image=value, text=""))
        self._print(self.adapter, records, logits.unsqueeze(1))


def build_printer(processor, model_args, data, training, options, dataset):
    if not options.print_sparse_diagnostics or not getattr(data, "print_sparse", False):
        return None
    return PrompteolSparsePrinter(processor, model_args, data, training, options, dataset)
