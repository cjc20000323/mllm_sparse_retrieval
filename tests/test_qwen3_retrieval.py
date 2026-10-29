"""Native tiny Qwen3-VL tests: full visual model, DeepStack and shared pipeline."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

import test_qwen25_retrieval as shared

prompts, pipeline = shared.prompts, shared.pipeline

# Qwen3 Instruct does not implicitly prepend the Qwen2.5 default system turn.
CHAT_TEMPLATE = """{% for message in messages %}<|im_start|>{{ message['role'] }}
{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for item in message['content'] %}{% if item['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{% else %}{{ item['text'] }}{% endif %}{% endfor %}{% endif %}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""


def processor_fixture():
    from transformers import Qwen2VLImageProcessorFast, Qwen3VLProcessor, Qwen3VLVideoProcessor
    tokenizer = shared.processor_fixture().tokenizer
    tokenizer.chat_template = CHAT_TEMPLATE
    return Qwen3VLProcessor(
        tokenizer=tokenizer, chat_template=CHAT_TEMPLATE,
        image_processor=Qwen2VLImageProcessorFast(
            patch_size=2, temporal_patch_size=1, merge_size=2,
            size={"shortest_edge": 64, "longest_edge": 144},
            image_mean=[0.5] * 3, image_std=[0.5] * 3),
        video_processor=Qwen3VLVideoProcessor(),
    )


def model_fixture(processor):
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    tokenizer = processor.tokenizer
    config = Qwen3VLConfig(
        text_config=dict(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, rope_scaling=dict(rope_type="default", mrope_section=[2, 1, 1])),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=4,
                           out_hidden_size=32, patch_size=2, temporal_patch_size=1,
                           spatial_merge_size=2, num_position_embeddings=64,
                           deepstack_visual_indexes=[0, 1]),
        image_token_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        video_token_id=tokenizer.convert_tokens_to_ids("<|video_pad|>"),
        vision_start_token_id=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        vision_end_token_id=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(23)
    return Qwen3VLForConditionalGeneration(config).eval()


class Qwen3NativeTests(shared.QwenNativeTests):
    processor_factory = staticmethod(processor_fixture)
    model_factory = staticmethod(model_fixture)

    def test_deepstack_reaches_text_layers_in_one_native_forward(self):
        encoder = self.encoder()
        language = self.model.model.language_model
        with patch.object(language, "_deepstack_process", wraps=language._deepstack_process) as deepstack, \
                patch.object(self.model, "forward", wraps=self.model.forward) as forward:
            sparse, dense = encoder.encode([dict(text="caption", image=Image.new("RGB", (12, 8), "red"))])
        self.assertEqual(forward.call_count, 1)
        self.assertEqual(deepstack.call_count, 2)
        self.assertTrue(all(call.args[2].shape[-1] == 32 for call in deepstack.call_args_list))
        self.assertTrue(torch.isfinite(dense).all())
        self.assertTrue(torch.isfinite(sparse[0]).all())

    def test_visual_views_stay_isolated_and_keep_native_mrope(self):
        from qwen_vl_encoding import rope_positions
        views = ["Summary above image in one word: ", "Summary the people: ", "Summary the environment: "]
        image = Image.new("RGB", (12, 8), "blue")
        original = shared.prepare_views(self.processor, ["<image>\ncaption\n"], [views], [[image]])
        positions = rope_positions(self.model, original.inputs)
        isolated = shared.isolate_views(self.model, original)
        common_end = original.spans[0][0][0]
        torch.testing.assert_close(isolated["position_ids"][:, :, :common_end], positions[:, :, :common_end])
        self.assertFalse(torch.equal(positions[1, 0, :common_end], positions[2, 0, :common_end]))
        expected, _ = shared.forward_views(self.model, original, isolated=True)
        changed = [views[0], "A longer and entirely different appearance instruction: ", views[2]]
        actual, _ = shared.forward_views(self.model, shared.prepare_views(
            self.processor, ["<image>\ncaption\n"], [changed], [[image]]), isolated=True)
        torch.testing.assert_close(actual[0][[0, 2]], expected[0][[0, 2]], atol=2e-5, rtol=2e-5)

    def test_native_instruct_template_has_no_synthetic_boundaries(self):
        encoder = self.encoder()
        encoder.encode([dict(text="caption", image=Image.new("RGB", (8, 8)))])
        text = encoder.last_prompts[0]
        self.assertTrue(text.startswith("<|im_start|>user\n"))
        for token in ("<think>", "<|box_start|>", "<|box_end|>", "[INST]", "<s>"):
            self.assertNotIn(token, text)
        self.assertTrue(text.endswith(prompts.TURN_END))
        self.assertTrue(prompts.is_native_qwen_model(self.model))
        self.assertFalse(prompts.is_qwen25_model(self.model))

    def test_config_routing_qwen3_renamed_dense_and_moe(self):
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
        self.assertIs(pipeline.model_class(Qwen3VLConfig()), Qwen3VLForConditionalGeneration)
        with tempfile.TemporaryDirectory(prefix="custom-weights-") as root:
            config = Path(root) / "config.json"
            config.write_text('{"model_type":"qwen3_vl"}', encoding="utf-8")
            self.assertTrue(prompts.is_native_qwen_checkpoint(root))
            self.assertFalse(prompts.is_qwen25_checkpoint(root))
            config.write_text('{"model_type":"qwen3_vl_moe"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Dense Instruct"):
                prompts.is_native_qwen_checkpoint(root)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            pipeline.model_class(SimpleNamespace(model_type="qwen3_vl_moe"))

    def test_family_manifest_separation_and_effective_fast_size(self):
        args = self.encoder().model_args
        training = SimpleNamespace(task_type="ir")
        processor = copy.deepcopy(self.processor)
        metadata = pipeline.prompt_metadata(args, shared.Data(), training, processor, "text")
        self.assertEqual(pipeline.manifest_name(metadata), pipeline.QWEN3_MANIFEST)
        self.assertEqual(metadata["config"]["version"], prompts.QWEN3_PROMPT_VERSION)
        self.assertEqual(metadata["config"]["vision_processor"]["size"], {"shortest_edge": 64, "longest_edge": 144})
        processor.image_processor.size = {"shortest_edge": 64, "longest_edge": 256}
        changed = pipeline.prompt_metadata(args, shared.Data(), training, processor, "text")
        self.assertNotEqual(metadata["fingerprint"], changed["fingerprint"])
        qwen25 = pipeline.prompt_metadata(args, shared.Data(), training, shared.processor_fixture(), "text")
        self.assertEqual(pipeline.manifest_name(qwen25), pipeline.MANIFEST)
        self.assertNotIn("model_type", qwen25["config"])
        self.assertNotIn("size", qwen25["config"]["vision_processor"])
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / pipeline.MANIFEST).write_text(json.dumps(qwen25), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "qwen3_prompt_manifest.*rebuild"):
                pipeline.validate_manifest(root, metadata)
            (Path(root) / pipeline.QWEN3_MANIFEST).write_text(json.dumps(metadata), encoding="utf-8")
            self.assertEqual(pipeline.validate_manifest(root, metadata), metadata)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                pipeline.validate_manifest(root, changed)

    def test_pixels_use_effective_defaults_and_validate_one_sided_limits(self):
        processor = copy.deepcopy(self.processor)
        pipeline.configure_pixels(processor, SimpleNamespace(qwen_min_pixels=0, qwen_max_pixels=256))
        self.assertEqual(processor.image_processor.size, {"shortest_edge": 64, "longest_edge": 256})
        self.assertEqual(processor.image_processor.min_pixels, 64)
        self.assertEqual(processor.image_processor.max_pixels, 256)
        processed = processor.image_processor(images=[Image.new("RGB", (40, 40))], return_tensors="pt")
        # The 2x2 patch fixture can fit at most 64 raw patches in 256 pixels.
        self.assertLessEqual(processed["image_grid_thw"][0].prod().item(), 64)
        with self.assertRaisesRegex(ValueError, "effective"):
            pipeline.configure_pixels(processor, SimpleNamespace(qwen_min_pixels=512, qwen_max_pixels=0))

    def test_real_local_runtime_load_uses_config_and_native_processor(self):
        from arguments import ModelArguments, PromptRepsLLMDataArguments
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
        with tempfile.TemporaryDirectory(prefix="renamed-dense-") as root:
            self.model.save_pretrained(root)
            self.processor.save_pretrained(root)
            args = ModelArguments(model_name_or_path=root, eol_type="prompteol", calculate_type="separate")
            data = PromptRepsLLMDataArguments(qwen_min_pixels=64, qwen_max_pixels=144)
            training = SimpleNamespace(fp16=False, bf16=False)
            with patch.object(pipeline, "init_process", return_value=(0, 1, torch.device("cpu"))):
                model, processor, rank, world = pipeline.load_runtime(args, data, training)
            self.assertIsInstance(model, Qwen3VLForConditionalGeneration)
            self.assertIsInstance(processor, Qwen3VLProcessor)
            self.assertEqual((rank, world), (0, 1))
            encoder = shared.NativeQwenEncoder(model, processor, args, shared.Data(), "ir")
            _, dense = encoder.encode([dict(text="caption", image=Image.new("RGB", (8, 8)))])
            self.assertTrue(torch.isfinite(dense).all())

    def test_thinking_checkpoints_fail_before_loading_weights(self):
        from arguments import ModelArguments, PromptRepsLLMDataArguments
        training = SimpleNamespace(fp16=False, bf16=False)
        args = ModelArguments(model_name_or_path="Qwen/Qwen3-VL-8B-Thinking",
                              eol_type="prompteol", calculate_type="separate")
        with patch("transformers.AutoConfig.from_pretrained", return_value=self.model.config), \
                patch.object(pipeline, "init_process", return_value=(0, 1, torch.device("cpu"))), \
                patch.object(type(self.model), "from_pretrained") as load:
            with self.assertRaisesRegex(ValueError, "Thinking"):
                pipeline.load_runtime(args, PromptRepsLLMDataArguments(), training)
            load.assert_not_called()

        # A renamed checkpoint still fails if its native template opens reasoning.
        args.model_name_or_path = "renamed-checkpoint"
        processor = copy.deepcopy(self.processor)
        processor.chat_template += "<think>\n"
        with patch("transformers.AutoConfig.from_pretrained", return_value=self.model.config), \
                patch("transformers.AutoProcessor.from_pretrained", return_value=processor), \
                patch.object(pipeline, "init_process", return_value=(0, 1, torch.device("cpu"))), \
                patch.object(type(self.model), "from_pretrained") as load:
            with self.assertRaisesRegex(ValueError, "Thinking"):
                pipeline.load_runtime(args, PromptRepsLLMDataArguments(), training)
            load.assert_not_called()

if __name__ == "__main__":
    unittest.main()
