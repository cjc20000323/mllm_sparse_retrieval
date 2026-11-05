"""Project-local LLaVA-NeXT image modes; no Transformers installation is patched.

``base`` uses exactly the native global image, not a one-cell AnyRes grid
(which would still encode both a global image and a local crop).
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import MethodType
from uuid import uuid4

from transformers import LlavaNextProcessor
from transformers.image_transforms import resize
from transformers.models.llava_next.image_processing_llava_next import LlavaNextImageProcessor
try:
    from transformers.models.llava_next.image_processing_llava_next_fast import LlavaNextImageProcessorFast
except ImportError:  # Older installations without the fast processor.
    LlavaNextImageProcessorFast = None

MODES = ("anyres", "base")
VERSION = 1
SUFFIX = "_llava_base"
MANIFEST_SUFFIX = ".llava_resolution.json"


def check_mode(mode):
    if mode not in MODES:
        raise ValueError(f"llava image mode must be anyres or base, got {mode!r}")
    return mode


def validate_checkpoint(model_path, mode="anyres"):
    """Reject an unsupported explicit mode before allocating model weights."""
    check_mode(mode)
    if mode == "base":
        from transformers import AutoConfig
        resolution_metadata(AutoConfig.from_pretrained(model_path), mode)


def resolution_metadata(config, mode="anyres"):
    check_mode(mode)
    if getattr(config, "model_type", None) != "llava_next":
        if mode != "anyres":
            raise ValueError("llava image mode 'base' requires a LLaVA-NeXT checkpoint")
        return None
    vision = config.vision_config
    size, patch = int(vision.image_size), int(vision.patch_size)
    strategy = config.vision_feature_select_strategy
    if mode == "base" and (getattr(vision, "model_type", None) != "clip_vision_model"
                           or strategy not in {"default", "full"} or size % patch):
        raise ValueError("LLaVA base mode requires a CLIP vision tower with an integral patch grid")
    return dict(version=VERSION, mode=mode, model_type="llava_next", image_size=size,
                patch_size=patch, vision_feature_select_strategy=strategy,
                visual_tokens_per_image=((size // patch) ** 2 + (strategy == "full") + 1)
                if mode == "base" else None)


class _BaseOnlyTorchvisionPatches:
    def _get_image_patches(self, image, grid_pinpoints, size, patch_size,
                           interpolation=None, *, resample=None):
        # Legacy fast processors pass interpolation (also positionally); the
        # unified Torchvision processor passes resample. Keep the native global
        # resize for each backend, including its interpolation conversion.
        from torchvision.transforms.v2 import functional as F
        from transformers.image_utils import SizeDict, pil_torch_interpolation_mapping

        sampling = resample if resample is not None else interpolation
        if sampling is None:
            sampling = self.resample
        if "resample" in inspect.signature(self.resize).parameters:
            height, width = size
            return [self.resize(image=image, size=SizeDict(height=height, width=width),
                                resample=sampling)]
        if not isinstance(sampling, F.InterpolationMode):
            sampling = pil_torch_interpolation_mapping[sampling]
        return [F.resize(image, size=size, interpolation=sampling)]


class BaseOnlyLlavaNextImageProcessor(_BaseOnlyTorchvisionPatches, LlavaNextImageProcessor):
    def get_image_patches(self, image, grid_pinpoints, size, patch_size, resample,
                          data_format, input_data_format):
        # This is the same resize used for resized_original_image in the native
        # processor. Retain the complete image rather than center-cropping it.
        return [resize(image, size=size, resample=resample, data_format=data_format,
                       input_data_format=input_data_format)]


if LlavaNextImageProcessorFast is not None:
    class BaseOnlyLlavaNextImageProcessorFast(_BaseOnlyTorchvisionPatches, LlavaNextImageProcessorFast):
        pass


class BaseOnlyLlavaNextProcessor(LlavaNextProcessor):
    llava_image_mode = "base"

    def _get_number_of_features(self, orig_height, orig_width, height, width):
        # The caller removes CLS for strategy=default. The model's native
        # single-image pack branch appends one image_newline feature.
        return ((height // self.patch_size) * (width // self.patch_size)
                + self.num_additional_image_tokens + 1)


def configure_processor(config, processor, mode="anyres"):
    """Configure before processing images, including config-only preflight."""
    report = resolution_metadata(config, mode)
    if mode == "anyres":
        if getattr(processor, "llava_image_mode", None) == "base":
            raise ValueError("Reload the LLaVA processor before changing base to anyres")
        return processor
    if isinstance(processor, BaseOnlyLlavaNextProcessor):
        return processor
    if not isinstance(processor, LlavaNextProcessor):
        raise ValueError("LLaVA base mode requires LlavaNextProcessor")
    options = processor.image_processor.to_dict()
    # Keep the original slow/fast backend so a resolution comparison does not
    # silently change interpolation, normalization, or CPU processing backend.
    options.update(size={"shortest_edge": report["image_size"]},
                   crop_size={"height": report["image_size"], "width": report["image_size"]})
    image_class = (BaseOnlyLlavaNextImageProcessorFast
                   if LlavaNextImageProcessorFast is not None
                   and isinstance(processor.image_processor, LlavaNextImageProcessorFast)
                   else BaseOnlyLlavaNextImageProcessor)
    image_processor = image_class.from_dict(options)
    return BaseOnlyLlavaNextProcessor(
        image_processor=image_processor, tokenizer=processor.tokenizer,
        patch_size=report["patch_size"], num_additional_image_tokens=1,
        vision_feature_select_strategy=report["vision_feature_select_strategy"],
        chat_template=getattr(processor, "chat_template", None),
    )


def _base_image_features(self, pixel_values, image_sizes, vision_feature_layer=None,
                         vision_feature_select_strategy=None, output_hidden_states=None,
                         return_dict=None, **kwargs):
    import torch
    count = len(image_sizes)
    if pixel_values.ndim == 5:
        if pixel_values.shape[:2] != (count, 1):
            raise ValueError("LLaVA base mode expects exactly one global image per image_sizes entry")
        pixel_values = pixel_values[:, 0]
    if pixel_values.ndim != 4 or pixel_values.shape[0] != count:
        raise ValueError("LLaVA base image batch and image_sizes do not match")
    layer = self.config.vision_feature_layer if vision_feature_layer is None else vision_feature_layer
    strategy = (self.config.vision_feature_select_strategy if vision_feature_select_strategy is None
                else vision_feature_select_strategy)
    # Hidden states are required for layer selection even when the caller did
    # not request them. Force a ModelOutput internally, independently of the
    # vision config's return_dict setting.
    output = self.vision_tower(pixel_values, output_hidden_states=True, return_dict=True, **kwargs)
    selected = (output.hidden_states[layer] if isinstance(layer, int) else
                torch.cat([output.hidden_states[index] for index in layer], dim=-1))
    if strategy == "default":
        selected = selected[:, 1:]
    elif strategy != "full":
        raise ValueError(f"Unsupported LLaVA feature selection: {strategy}")
    projected = self.multi_modal_projector(selected)
    features = tuple(projected[index:index + 1] for index in range(count))
    if not self._llava_pack_in_features:
        # Older flat wrappers pack the returned chunks in forward().
        return features
    features, _ = self.pack_image_features(features, image_sizes, strategy, image_newline=self.image_newline)
    if self._llava_features_as_model_output:
        # Newer HF forwards read get_image_features(...).pooler_output; older
        # forwards consume the packed list directly. Retain the native vision
        # output fields and the caller/config return_dict convention.
        output.pooler_output = features
        if return_dict is None:
            return_dict = getattr(self.config, "return_dict", True)
        return output if return_dict else output.to_tuple()
    return features


def configure_model(model, mode="anyres"):
    report = resolution_metadata(model.config, mode)
    previous = getattr(model, "_llava_resolution_report", None)
    if previous is not None and previous != report:
        raise ValueError("Reload the LLaVA model before changing its image mode")
    if mode == "base" and previous is None:
        # PEFT exposes get_base_model; recent HF versions put the real visual
        # forward on model.model, while older releases used the outer wrapper.
        owner = model.get_base_model() if hasattr(model, "get_base_model") else model
        inner = getattr(owner, "model", None)
        if inner is not None and hasattr(inner, "vision_tower") and hasattr(inner, "get_image_features"):
            owner = inner
        original = getattr(owner, "get_image_features", None)
        function = inspect.unwrap(original) if original is not None else None
        code = getattr(function, "__code__", None)
        if code is None or not hasattr(owner, "pack_image_features"):
            raise ValueError("Unsupported Transformers LLaVA-NeXT feature interface for base mode")
        owner._llava_pack_in_features = "pack_image_features" in code.co_names
        owner._llava_features_as_model_output = "pooler_output" in code.co_names
        owner.get_image_features = MethodType(_base_image_features, owner)
    if report is not None:
        model._llava_resolution_report = report
    return report


def configure(model, processor, mode="anyres"):
    processor = configure_processor(model.config, processor, mode)
    report = configure_model(model, mode)
    if report is not None:
        print(f"[llava-image] mode={mode}; image_size={report['image_size']}; "
              f"visual_tokens={report['visual_tokens_per_image'] or 'dynamic'}")
    return processor


def configure_search(model, processor, data_args, search_args, task):
    """Configure standalone search/rerank before datasets or image processing.

    Explicit artifact paths are never rewritten. Only indexes actually used
    by the run are checked; pure-text corpora can be reused in either mode.
    """
    processor = configure(model, processor, data_args.llava_image_mode)
    report = getattr(model, "_llava_resolution_report", None)
    images = target_has_images(task, search_args.query_type)
    paths = [search_args.passage_reps, search_args.sparse_index]
    if not search_args.use_fixed_alpha:
        paths.extend([search_args.val_passage_reps, search_args.val_sparse_index])
    for path in paths:
        if path is not None:
            validate_artifact(path, report, has_images=images)
    return processor


def search_result_dir(path, mode="anyres"):
    """Separate all derived result directories, including text-corpus runs."""
    check_mode(mode)
    return str(artifact_dir(path, mode)) if mode == "base" else path


def lucene_index_dir(path):
    """Accept a collection, corpus root or an explicit Lucene index directory."""
    root = Path(path)
    if root.name == "index" or any(root.glob("segments_*")):
        return str(root)
    if (root / "index").is_dir():
        return str(root / "index")
    if (root / "collection" / "index").is_dir():
        return str(root / "collection" / "index")
    return str(root / "index")


def corpus_has_images(task, kind):
    if kind == "text":
        return False
    if task == "it2t" and kind == "corpus":
        return False
    if task == "t2it":
        return kind == "corpus"
    return True


def target_has_images(task, query_type):
    return task in {"tbpr", "t2it", "cir"} or (task == "ir" and query_type == "text")


def artifact_dir(path, mode="anyres", *, has_images=True):
    check_mode(mode)
    path = Path(path)
    return path.with_name(path.name + SUFFIX) if mode == "base" and has_images else path


def sidecar_path(root):
    root = Path(root)
    return root.with_name(root.name + MANIFEST_SUFFIX)


def write_manifest(root, report):
    if report is None:
        return
    path = sidecar_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".lr-{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_artifact(path, expected, *, has_images=True):
    """Accept legacy AnyRes artifacts, but require evidence for image base mode.

    Sidecars are siblings of corpus directories, never input documents for
    Anserini. Accept a corpus pickle, corpus directory, collection/, or index/.
    """
    if expected is None or not has_images:
        return
    path = Path(path)
    root = path.parent if path.is_file() else path
    candidates = [root]
    while root.name in {"index", "collection"}:
        root = root.parent
        candidates.append(root)
    for candidate in candidates:
        manifest = sidecar_path(candidate)
        if manifest.is_file():
            actual = json.loads(manifest.read_text(encoding="utf-8"))
            if actual != expected:
                raise ValueError(f"LLaVA resolution mismatch at {candidate}: stored={actual}, requested={expected}")
            return
    if expected["mode"] == "base" or root.name.endswith(SUFFIX):
        raise ValueError(f"Missing matching LLaVA resolution sidecar for {path}; encode and index with the requested image mode")
