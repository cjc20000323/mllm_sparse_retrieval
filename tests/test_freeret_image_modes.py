"""FreeRet image modes, cache compatibility and real tiny LLaVA-NeXT forwards."""
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import freeret_retrieval as freeret


def bash_executable():
    bash = shutil.which("bash")
    if bash:
        return bash
    git = shutil.which("git")
    if git:
        candidate = Path(git).resolve().parent.parent / "bin" / "bash.exe"
        if candidate.is_file():
            return str(candidate)
    return None


def task_fixture(image_path="fixture.png"):
    return freeret.RetrievalTask(
        "CUHK-PEDES", "test", "t2i", freeret.prompt_key_for("CUHK-PEDES", "t2i"),
        [freeret.QueryItem("q", "a person", relevant_ids={"c"})],
        [freeret.CandidateItem("c", "a person", image_path=image_path)],
    )


class FakeEncoder:
    model_family = "llava_next"
    model_class_name = "fixture"
    processor_class_name = "fixture"
    transformers_version = "fixture"
    attn_implementation = "eager"
    _decoder_norm_path = "fixture"
    device = torch.device("cpu")
    max_pixels_applied = False

    def __init__(self, mode):
        self.requested_llava_image_mode = "auto" if mode == "base" else mode
        self.llava_image_mode = mode
        self.llava_resolution = {"mode": mode, "image_size": 28}
        self.calls = 0

    def encode_one_word(self, texts, image_paths):
        self.calls += 1
        return torch.tensor([[1., 0.]] * len(texts))

    def rerank(self, texts, image_paths):
        return torch.ones(len(texts))

    def set_max_pixels(self, value):
        pass


class ModeTests(unittest.TestCase):
    def test_defaults_overrides_and_unsupported_base_fail_before_loading(self):
        self.assertEqual(freeret.build_parser().parse_args([]).llava_image_mode, "auto")
        self.assertEqual(freeret._resolve_llava_image_mode("llava_next"), "base")
        self.assertEqual(freeret._resolve_llava_image_mode("llava_next", "anyres"), "anyres")
        for family in ("qwen2_5_vl", "llava_onevision", "llava_onevision_1_5"):
            for mode in ("auto", "anyres"):
                self.assertEqual(freeret._resolve_llava_image_mode(family, mode), "native")
            with patch.object(freeret, "_detect_model_family", return_value=family), \
                 patch.object(freeret.AutoProcessor, "from_pretrained") as processor, \
                 patch.object(freeret.FreeRetQwen, "_load_pretrained_model") as load:
                with self.assertRaisesRegex(ValueError, "requires a LLaVA-NeXT"):
                    freeret.FreeRetQwen("fixture", "float32", "none", "eager", 512, "base")
                processor.assert_not_called()
                load.assert_not_called()

    def test_model_output_directories_keep_anyres_and_native_paths(self):
        args = NS(model_path="llava-v1.6-mistral-7b-hf", output_dir="results/freeret", llava_image_mode="auto")
        self.assertEqual(freeret._model_output_root(args).name, args.model_path + "_llava_base")
        args.llava_image_mode = "anyres"
        self.assertEqual(freeret._model_output_root(args).name, args.model_path)
        args.model_path = "Qwen2.5-VL-7B-Instruct"
        args.llava_image_mode = "auto"
        self.assertEqual(freeret._model_output_root(args).name, args.model_path)

    def test_cache_metadata_distinguishes_resolution_but_not_auto_vs_explicit_base(self):
        task = task_fixture()
        model = FakeEncoder("base")
        payload = dict(freeret._target_cache_metadata(model), prompt_version=freeret.PROMPT_VERSION,
                       prompt_key=task.prompt_key, max_pixels=512, candidate_ids=["c"],
                       embeddings=torch.tensor([[1., 0.]]))
        self.assertTrue(freeret._target_cache_matches(payload, model, task, ["c"], 512))
        model.requested_llava_image_mode = "base"
        self.assertTrue(freeret._target_cache_matches(payload, model, task, ["c"], 512))
        changed = copy.deepcopy(payload)
        changed["llava_resolution"]["image_size"] = 56
        self.assertFalse(freeret._target_cache_matches(changed, model, task, ["c"], 512))
        self.assertFalse(freeret._target_cache_matches(payload, FakeEncoder("anyres"), task, ["c"], 512))

    def test_single_and_distributed_cache_reuse_and_reencoding(self):
        for distributed in (False, True):
            with self.subTest(distributed=distributed), tempfile.TemporaryDirectory() as directory, \
                 redirect_stdout(io.StringIO()):
                root = Path(directory)
                task = task_fixture()
                cache = root / "targets.pt"
                torch.save(dict(prompt_version=freeret.PROMPT_VERSION, prompt_key=task.prompt_key,
                                max_pixels=512, candidate_ids=["c"], embeddings=torch.tensor([[0., 1.]])), cache)
                if distributed:
                    context = NS(rank=0, local_rank=0, world_size=1,
                                 primary=lambda action, share=False: action(),
                                 collectives=NS(barrier=lambda: None))
                    execution = freeret.FreeRetDistributedTask(context, root, None)
                    def encode(model):
                        return execution.prepare_targets(
                            model, task, NS(estimate_time_only=False, target_batch_size=2), cache, 512, {},
                        )
                else:
                    def encode(model):
                        return freeret.encode_targets(model, task, 2, cache, 512)
                anyres = FakeEncoder("anyres")
                self.assertEqual(encode(anyres).tolist(), [[0., 1.]])
                self.assertEqual(anyres.calls, 0)
                base = FakeEncoder("base")
                self.assertEqual(encode(base).tolist(), [[1., 0.]])
                self.assertEqual(base.calls, 1)
                stored = torch.load(cache, map_location="cpu")
                self.assertEqual(stored["llava_image_mode"], "base")
                self.assertEqual(stored["llava_resolution"], base.llava_resolution)
                base.calls = 0
                encode(base)
                self.assertEqual(base.calls, 0)
                encode(anyres)
                self.assertEqual(anyres.calls, 1)

    def test_reports_and_timing_include_effective_resolution(self):
        args = freeret.build_parser().parse_args([])
        args.model_path = "llava-v1.6-mistral-7b-hf"
        freeret.validate_args(args)
        model = FakeEncoder("base")
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            from PIL import Image

            root = Path(directory)
            image_path = root / "image.png"
            Image.new("RGB", (28, 28)).save(image_path)
            task = task_fixture(str(image_path))
            timing = freeret.ActualRunTiming(args)
            timing.distributed_metadata.update(freeret._image_mode_metadata(model))
            result = freeret.run_task(model, task, args, root)
            for name in ("config.json", "metrics.json", "timing_metrics.json"):
                data = json.loads((root / name).read_text(encoding="utf-8"))
                self.assertEqual(data["requested_llava_image_mode"], "auto")
                self.assertEqual(data["llava_image_mode"], "base")
                self.assertEqual(data["llava_resolution"], model.llava_resolution)
            self.assertEqual(result["final"]["recall@20"], 1.)
            timing.save(root)
            self.assertEqual(json.loads((root / "timing_summary.json").read_text())["llava_image_mode"], "base")
            args.estimate_time_only = True
            estimate = freeret.run_task(model, task, args, root)
            freeret._save_time_estimate_summary([estimate], 1, root)
            for name in ("time_estimate.json", "time_estimate_summary.json"):
                data = json.loads((root / name).read_text(encoding="utf-8"))
                self.assertEqual(data["llava_image_mode"], "base")


@unittest.skipUnless(bash_executable(), "Bash is required for launcher tests")
class LauncherTests(unittest.TestCase):
    def test_both_launchers_forward_default_and_explicit_modes(self):
        for name in ("freeret_retrieval.sh", "estimate_freeret_retrieval_time.sh"):
            for mode in (None, "base", "anyres", "invalid"):
                with self.subTest(script=name, mode=mode), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    capture = root / "args.txt"
                    python = root / "python.sh"
                    python.write_text(
                        '#!/usr/bin/env bash\n'
                        'if [[ "$1" == -c ]]; then\n'
                        '  [[ "$2" != *torch.cuda.device_count* ]] || echo 1\n'
                        '  exit 0\n'
                        'fi\n'
                        'printf "%s\\n" "$@" > "$CAPTURE_ARGS_FILE"\n', encoding="utf-8",
                    )
                    python.chmod(0o755)
                    env = dict(os.environ, PYTHON_BIN=python.as_posix(), CAPTURE_ARGS_FILE=capture.as_posix(),
                               NUM_PROCESSES="1", NUM_NODES="1", CUDA_VISIBLE_DEVICES="0")
                    env.pop("LLAVA_IMAGE_MODE", None)
                    if mode is not None:
                        env["LLAVA_IMAGE_MODE"] = mode
                    result = subprocess.run([bash_executable(), f"scripts/{name}"], cwd=ROOT, env=env,
                                            capture_output=True, text=True, timeout=30)
                    if mode == "invalid":
                        self.assertEqual(result.returncode, 2)
                        self.assertIn("LLAVA_IMAGE_MODE", result.stderr)
                        self.assertFalse(capture.exists())
                    else:
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        argv = capture.read_text(encoding="utf-8").splitlines()
                        self.assertEqual(argv[argv.index("--llava_image_mode") + 1], mode or "auto")


class TinyLlavaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_llama_and_mistral_encode_and_rerank_real_image_features(self):
        from PIL import Image
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import (LlavaNextConfig, LlavaNextForConditionalGeneration,
                                  LlavaNextImageProcessor, LlavaNextProcessor, LlamaConfig,
                                  MistralConfig, PreTrainedTokenizerFast)
        words = ["[UNK]", "[PAD]", "<s>", "</s>", "<image>", "A", "B", "user", "assistant", "a", "b"]
        words += [f"v{i}" for i in range(64 - len(words))]
        template = ("{% for message in messages %}{{ message['role'] }}: "
                    "{% for content in message['content'] %}"
                    "{% if content['type'] == 'image' %}<image>"
                    "{% elif content['type'] == 'text' %}{{ content['text'] }}{% endif %}"
                    "{% endfor %}{% endfor %}{% if add_generation_prompt %} assistant:{% endif %}")
        for text_class in (LlamaConfig, MistralConfig):
            for mode in ("auto", "base", "anyres"):
                with self.subTest(text=text_class.__name__, mode=mode), \
                     tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
                    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
                    backend.pre_tokenizer = Whitespace()
                    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                                        pad_token="[PAD]", additional_special_tokens=["<image>"])
                    text = text_class(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                      num_attention_heads=2, num_key_value_heads=2, pad_token_id=1)
                    grids = [[28, 28], [28, 56], [56, 28]]
                    config = LlavaNextConfig(text_config=text.to_dict(), vision_config=dict(
                        hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                        image_size=28, patch_size=14, projection_dim=16), image_token_index=4,
                        image_grid_pinpoints=grids, vision_feature_layer=-1)
                    config._attn_implementation = "eager"
                    model = LlavaNextForConditionalGeneration(config)
                    processor = LlavaNextProcessor(
                        LlavaNextImageProcessor(size={"shortest_edge": 28}, crop_size={"height": 28, "width": 28},
                                                image_grid_pinpoints=grids),
                        tokenizer, patch_size=14, num_additional_image_tokens=1,
                        vision_feature_select_strategy="default", chat_template=template,
                    )
                    with patch.object(freeret.AutoConfig, "from_pretrained", return_value=config), \
                         patch.object(freeret.LlavaNextProcessor, "from_pretrained", return_value=processor), \
                         patch.object(freeret.FreeRetQwen, "_load_pretrained_model", return_value=model):
                        encoder = freeret.FreeRetQwen("llava-v1.6-mistral-7b-hf", "float32", "none", "eager", 512,
                                                     llava_image_mode=mode)
                    image_paths = []
                    for index, size in enumerate(((56, 28), (28, 56))):
                        path = Path(directory) / f"image-{index}.png"
                        Image.new("RGB", size, "red").save(path)
                        image_paths.append(str(path))
                    paths = [[image_paths[0]], [image_paths[1]]]
                    prepared = encoder._prepare_inputs(["a", "b"], paths)
                    if mode in {"auto", "base"}:
                        self.assertEqual(encoder.llava_image_mode, "base")
                        self.assertEqual(prepared["pixel_values"].shape[:2], (2, 1))
                        self.assertTrue(torch.all((prepared["input_ids"] == 4).sum(-1) == 5))
                    else:
                        self.assertGreater(prepared["pixel_values"].shape[1], 1)
                    for prompts, images in ((["a", "b"], paths), (["a"], [image_paths]), (["a"], [[]])):
                        features = encoder.encode_one_word(prompts, images)
                        self.assertEqual(features.shape, (len(prompts), 16))
                        self.assertTrue(torch.allclose(features.norm(dim=-1), torch.ones(len(prompts)), atol=1e-5))
                        scores = encoder.rerank(prompts, images)
                        self.assertEqual(scores.shape, (len(prompts),))
                        self.assertTrue(torch.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()
