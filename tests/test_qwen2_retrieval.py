"""Native Qwen2-VL coverage for both PromptEOL retrieval methods."""
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


def processor_fixture(*, fast=True):
    from transformers import (Qwen2VLProcessor, Qwen2VLVideoProcessor,
                              Qwen2VLImageProcessor, Qwen2VLImageProcessorFast)
    tokenizer = shared.processor_fixture().tokenizer
    options = dict(patch_size=2, temporal_patch_size=1, merge_size=2)
    if fast:
        image = Qwen2VLImageProcessorFast(size={"shortest_edge": 64, "longest_edge": 144}, **options)
    else:
        image = Qwen2VLImageProcessor(min_pixels=64, max_pixels=144, **options)
    return Qwen2VLProcessor(tokenizer=tokenizer, image_processor=image,
                            video_processor=Qwen2VLVideoProcessor(), chat_template=shared.CHAT_TEMPLATE)


def model_fixture(processor):
    from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration
    tokenizer = processor.tokenizer
    config = Qwen2VLConfig(
        text_config=dict(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         rope_scaling=dict(rope_type="default", mrope_section=[1, 1, 2])),
        vision_config=dict(depth=1, embed_dim=32, hidden_size=32, mlp_ratio=2,
                           num_heads=4, patch_size=2, temporal_patch_size=1, spatial_merge_size=2),
        image_token_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        video_token_id=tokenizer.convert_tokens_to_ids("<|video_pad|>"),
        vision_start_token_id=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        vision_end_token_id=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(23)
    return Qwen2VLForConditionalGeneration(config).eval()


class Qwen2NativeTests(shared.QwenNativeTests):
    processor_factory = staticmethod(processor_fixture)
    model_factory = staticmethod(model_fixture)

    def test_config_routes_renamed_and_hub_qwen2_without_size_matching(self):
        from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration
        self.assertIs(pipeline.model_class(self.model.config), Qwen2VLForConditionalGeneration)
        self.assertTrue(prompts.is_native_qwen_model(self.model))
        self.assertFalse(prompts.is_qwen25_model(self.model))
        with tempfile.TemporaryDirectory(prefix="custom-weights-") as root:
            path = Path(root) / "config.json"
            path.write_text('{"model_type":"qwen2_vl"}', encoding="utf-8")
            self.assertTrue(prompts.is_native_qwen_checkpoint(root))
            self.assertFalse(prompts.is_qwen25_checkpoint(root))
            path.write_text('{"model_type":"qwen2"}', encoding="utf-8")
            self.assertFalse(prompts.is_native_qwen_checkpoint(root))
        with patch("transformers.AutoConfig.from_pretrained", return_value=self.model.config) as load:
            self.assertTrue(prompts.is_native_qwen_checkpoint("Qwen/Qwen2-VL-72B-Instruct"))
            load.assert_called_once_with("Qwen/Qwen2-VL-72B-Instruct")
        with patch("transformers.AutoConfig.from_pretrained") as load:
            self.assertFalse(prompts.is_native_qwen_checkpoint("llava-checkpoint"))
            load.assert_not_called()
        restored = Qwen2VLConfig.from_dict(self.model.config.to_dict())
        self.assertEqual(restored.text_config.model_type, "qwen2_vl_text")
        self.assertEqual(prompts.model_family(restored), "qwen2_vl")
        self.assertIs(pipeline.model_class(restored), Qwen2VLForConditionalGeneration)
        self.assertTrue(prompts.is_native_qwen_model(SimpleNamespace(config=restored)))
        with patch("transformers.AutoConfig.from_pretrained", return_value=restored):
            self.assertTrue(prompts.is_native_qwen_checkpoint("Qwen/Qwen2-VL-7B-Instruct"))

    def test_qwen2_manifest_is_separate_and_tracks_effective_size(self):
        args, data = self.encoder().model_args, shared.Data()
        training = SimpleNamespace(task_type="ir")
        processor = copy.deepcopy(self.processor)
        expected = pipeline.prompt_metadata(args, data, training, processor, "text")
        self.assertEqual(expected["config"]["model_type"], "qwen2_vl")
        self.assertEqual(expected["config"]["version"], prompts.QWEN2_PROMPT_VERSION)
        self.assertEqual(pipeline.manifest_name(expected), pipeline.QWEN2_MANIFEST)
        self.assertEqual(expected["config"]["vision_processor"]["size"], {"shortest_edge": 64, "longest_edge": 144})
        self.assertEqual(expected["fingerprint"], pipeline.prompt_metadata(
            args, data, training, processor, "text", family="qwen2_vl")["fingerprint"])
        processor.image_processor.size = {"shortest_edge": 64, "longest_edge": 256}
        changed = pipeline.prompt_metadata(args, data, training, processor, "text")
        self.assertNotEqual(expected["fingerprint"], changed["fingerprint"])
        with tempfile.TemporaryDirectory() as root:
            for family in ("qwen2_5_vl", "qwen3_vl"):
                other = pipeline.prompt_metadata(args, data, training, self.processor, "text", family=family)
                (Path(root) / pipeline.manifest_name(other)).write_text(json.dumps(other), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "qwen2_prompt_manifest.*rebuild"):
                pipeline.validate_manifest(root, expected)
            (Path(root) / pipeline.QWEN2_MANIFEST).write_text(json.dumps(expected), encoding="utf-8")
            self.assertEqual(pipeline.validate_manifest(root, expected), expected)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                pipeline.validate_manifest(root, changed)

    def test_fast_and_slow_pixels_apply_one_sided_limits(self):
        for fast in (False, True):
            with self.subTest(fast=fast):
                processor = processor_fixture(fast=fast)
                pipeline.configure_pixels(processor, SimpleNamespace(qwen_min_pixels=0, qwen_max_pixels=256))
                self.assertEqual(processor.image_processor.size, {"shortest_edge": 64, "longest_edge": 256})
                self.assertEqual(processor.image_processor.min_pixels, 64)
                self.assertEqual(processor.image_processor.max_pixels, 256)
                inputs = processor.image_processor(images=[Image.new("RGB", (40, 40))], return_tensors="pt")
                self.assertLessEqual(inputs["image_grid_thw"][0].prod().item(), 64)
                with self.assertRaisesRegex(ValueError, "effective"):
                    pipeline.configure_pixels(processor, SimpleNamespace(qwen_min_pixels=512, qwen_max_pixels=0))

    def test_mixed_image_and_text_batch_keeps_sample_boundaries(self):
        records = [dict(text="short caption"), dict(text="caption", image=Image.new("RGB", (12, 8), "red")),
                   dict(text="longer caption about people", image=Image.new("RGB", (8, 8), "blue"))]
        for concat in (False, True):
            for side in ("before_pad", "after_pad"):
                encoder = self.encoder(concat=concat, reps_loc=side)
                sparse, dense = encoder.encode(records)
                for index, record in enumerate(records):
                    expected, expected_dense = encoder.encode([record])
                    torch.testing.assert_close(sparse[index], expected[0], atol=3e-5, rtol=3e-5)
                    torch.testing.assert_close(dense[index], expected_dense[0], atol=3e-5, rtol=3e-5)

    def test_visual_views_keep_native_positions_and_cannot_leak(self):
        from qwen_vl_encoding import rope_positions
        context, image = "<image>\ncaption\n", Image.new("RGB", (12, 8), "red")
        views = ["Summary above image: ", "Summary the people: ", "Summary the environment: "]
        batch = shared.prepare_views(self.processor, [context], [views], [[image]])
        original = rope_positions(self.model, batch.inputs)
        isolated = shared.isolate_views(self.model, batch)
        boundary = batch.spans[0][0][0]
        torch.testing.assert_close(isolated["position_ids"][:, :, :boundary], original[:, :, :boundary])
        self.assertFalse(torch.equal(original[1, 0, :boundary], original[2, 0, :boundary]))
        expected, _ = shared.forward_views(self.model, batch, isolated=True)
        views[1] = "A much longer and entirely different appearance instruction: "
        actual, _ = shared.forward_views(self.model, shared.prepare_views(
            self.processor, [context], [views], [[image]]), isolated=True)
        torch.testing.assert_close(actual[0][[0, 2]], expected[0][[0, 2]], atol=2e-5, rtol=2e-5)

    def test_rerank_english_body_exactly_matches_mistral_templates(self):
        import template as baseline
        cases = [
            ("ir", "text", "query_relevant", "mistral_text_query_relevant_prompt", False),
            ("ir", "text", "reverse_query_relevant", "mistral_text_reverse_query_relevant_prompt", False),
            ("ir", "image", "freeret", "mistral_freeret_image_rerank_prompt", False),
            ("tbpr", "text", "role_relevant", "mistral_person_retrieval_role_relevant_prompt", False),
            ("t2it", "text", "query_relevant", "mistral_t2it_retrieval_query_relevant_prompt", False),
            ("it2t", "composed", "answer_relevant", "mistral_it2t_retrieval_answer_relevant_prompt", False),
            ("cir", "composed", "old_relevant", "mistral_fashion_iq_old_query_relevant_prompt", False),
            ("ir", "image", "what_caption_generation", "mistral_query_generation_paradigm_prompt_1", True),
        ]
        for task, direction, mode, name, caption in cases:
            with self.subTest(task=task, mode=mode):
                actual = prompts.rerank_template(self.processor, task, direction, mode, caption=caption)
                body = actual.split("<|im_start|>user\n", 1)[1][:-len(prompts.TURN_END)]
                expected = getattr(baseline, name)[len("[INST]"):-len("[/INST]")]
                self.assertEqual(body.replace(prompts.IMAGE, "<image>"), expected)
        encoder = self.encoder()
        encoder.encode([dict(text="caption", image=Image.new("RGB", (8, 8)))])
        for marker in ("<think>", "<|box_start|>", "<|box_end|>", "[INST]", "<s>"):
            self.assertNotIn(marker, encoder.last_prompts[0])

    def test_real_local_runtime_loads_native_qwen2_after_directory_rename(self):
        from arguments import ModelArguments, PromptRepsLLMDataArguments
        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
        with tempfile.TemporaryDirectory(prefix="custom-weights-") as root:
            self.model.save_pretrained(root)
            self.processor.save_pretrained(root)
            args = ModelArguments(model_name_or_path=root, eol_type="prompteol", calculate_type="separate")
            data = PromptRepsLLMDataArguments(qwen_min_pixels=64, qwen_max_pixels=144)
            with patch.object(pipeline, "init_process", return_value=(0, 1, torch.device("cpu"))):
                model, processor, rank, world = pipeline.load_runtime(args, data, SimpleNamespace(fp16=False, bf16=False))
            self.assertIsInstance(model, Qwen2VLForConditionalGeneration)
            self.assertIsInstance(processor, Qwen2VLProcessor)
            self.assertEqual((rank, world), (0, 1))
            self.assertEqual(prompts.model_family(model.config), "qwen2_vl")
            self.assertTrue(prompts.is_native_qwen_model(model))
            for eol, calc in (("prompteol", "separate"), ("disassembleeol_separate", "concat")):
                args.eol_type, args.calculate_type = eol, calc
                encoder = shared.NativeQwenEncoder(model, processor, args, shared.Data(), "ir")
                sparse, dense = encoder.encode([dict(text="caption", image=Image.new("RGB", (8, 8)))])
                self.assertTrue(torch.isfinite(dense).all())
                self.assertTrue(torch.isfinite(sparse[0]).all())
            metadata = pipeline.prompt_metadata(args, shared.Data(), SimpleNamespace(task_type="ir"),
                                                processor, "text", family=prompts.model_family(model.config))
            self.assertEqual(pipeline.manifest_name(metadata), pipeline.QWEN2_MANIFEST)
            candidates = [dict(id="c", native_id="c", text="caption")]
            query = [dict(id="q", native_id="q", text="", image=Image.new("RGB", (8, 8)))]
            search = SimpleNamespace(query_type="image", rerank_type="pointwise", rerank_template="relevant",
                                     rerank_num=1, rerank_batch_size=1, modify_type="no", use_rerank_prior_cache=False)
            with patch.object(pipeline, "dataset_records", return_value=candidates):
                scores = pipeline.rerank_records(model, processor, SimpleNamespace(data_name="flickr"), query,
                                                 {"q": {"c": 1.0}}, "ir", args, shared.Data(), search,
                                                 SimpleNamespace(task_type="ir", fp16=False, bf16=False), 0, 1)
            self.assertTrue(torch.isfinite(torch.tensor(scores["q"]["c"])))

    def test_invalid_qwen2_processor_fails_before_weights_load(self):
        from arguments import ModelArguments, PromptRepsLLMDataArguments
        processor = copy.deepcopy(self.processor)
        processor.chat_template += "unexpected suffix"
        args = ModelArguments(model_name_or_path="renamed-checkpoint", eol_type="prompteol", calculate_type="separate")
        data, training = PromptRepsLLMDataArguments(), SimpleNamespace(fp16=False, bf16=False)
        with patch("transformers.AutoConfig.from_pretrained", return_value=self.model.config), \
                patch("transformers.AutoProcessor.from_pretrained", return_value=processor), \
                patch.object(pipeline, "init_process", return_value=(0, 1, torch.device("cpu"))), \
                patch.object(type(self.model), "from_pretrained") as load:
            with self.assertRaisesRegex(ValueError, "native Instruct assistant prefix"):
                pipeline.load_runtime(args, data, training)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
