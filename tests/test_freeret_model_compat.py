from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "src" / "freeret_retrieval.py"
MODULE_SPEC = importlib.util.spec_from_file_location("freeret_retrieval", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Cannot load {MODULE_PATH}")
freeret: Any = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = freeret
MODULE_SPEC.loader.exec_module(freeret)


class FakeInputs(dict):
    def to(self, device: torch.device) -> "FakeInputs":
        self.moved_to = device
        return self


class FakeOneVisionProcessor:
    def __init__(self) -> None:
        self.template_calls: list[tuple[object, dict[str, object]]] = []
        self.image_token = "<image>"

    def apply_chat_template(
        self,
        messages: object,
        processor_kwargs: object = None,
        **kwargs: object,
    ) -> FakeInputs:
        kwargs["processor_kwargs"] = processor_kwargs
        self.template_calls.append((messages, kwargs))
        return FakeInputs(input_ids=torch.ones((2, 3), dtype=torch.long))


class FakeOneVision15Processor:
    def __init__(self) -> None:
        self.template_calls: list[object] = []
        self.processor_kwargs: dict[str, object] = {}

    def apply_chat_template(self, messages: object, **_: object) -> str:
        self.template_calls.append(messages)
        return f"prompt-{len(self.template_calls)}"

    def __call__(self, **kwargs: object) -> FakeInputs:
        self.processor_kwargs = kwargs
        return FakeInputs(
            input_ids=torch.ones((2, 3), dtype=torch.long),
            attention_mask=torch.ones((2, 3), dtype=torch.long),
            pixel_values=torch.ones((2, 3)),
            image_grid_thw=torch.ones((2, 3), dtype=torch.long),
            mm_token_type_ids=torch.zeros((2, 3), dtype=torch.long),
            second_per_grid_ts=torch.ones(2),
        )


class FakeLlavaNextProcessor:
    def __init__(self) -> None:
        self.image_token = "<image>"
        self.patch_size = None
        self.vision_feature_select_strategy = None
        self.template_calls: list[tuple[object, dict[str, object]]] = []

    def apply_chat_template(
        self,
        messages: object,
        processor_kwargs: object = None,
        **kwargs: object,
    ) -> object:
        kwargs["processor_kwargs"] = processor_kwargs
        self.template_calls.append((messages, kwargs))
        if kwargs.get("tokenize"):
            return FakeInputs(
                input_ids=torch.ones((2, 3), dtype=torch.long),
                attention_mask=torch.ones((2, 3), dtype=torch.long),
                pixel_values=torch.ones((2, 3)),
            )

        conversations = messages
        if not isinstance(conversations, list):
            raise AssertionError("messages must be a list")
        image_count = 0
        for message in conversations:
            for item in message["content"]:
                if item["type"] == "image":
                    image_count += 1
        return ("<image>" * image_count) + " prompt"


class FreeRetModelCompatibilityTest(unittest.TestCase):
    def test_model_family_detection_prefers_ov15(self) -> None:
        self.assertEqual(
            freeret._detect_model_family(
                "/checkpoints/lmms-lab-LLaVA-OneVision-1.5-8B-Instruct"
            ),
            freeret.LLAVA_ONEVISION_1_5_MODEL_FAMILY,
        )
        self.assertEqual(
            freeret._detect_model_family(
                "/checkpoints/llava-hf-llava-onevision-qwen2-7b-ov-hf"
            ),
            freeret.LLAVA_ONEVISION_MODEL_FAMILY,
        )

    def test_ov15_requires_official_transformers_version(self) -> None:
        freeret._require_ov15_transformers_version("4.53.0")
        with self.assertRaisesRegex(RuntimeError, "transformers==4.53.0"):
            freeret._require_ov15_transformers_version("5.0.0")

    def test_onevision_keeps_images_grouped_by_sample(self) -> None:
        wrapper = freeret.FreeRetQwen.__new__(freeret.FreeRetQwen)
        wrapper.processor = FakeOneVisionProcessor()
        wrapper.device = torch.device("cpu")

        inputs = wrapper._prepare_llava_onevision_inputs(
            ["text only", "two images"],
            [[], ["/images/a.jpg", "/images/b.jpg"]],
        )

        messages, kwargs = wrapper.processor.template_calls[0]
        self.assertEqual(len(messages), 2)
        first_content = messages[0][0]["content"]
        second_content = messages[1][0]["content"]
        self.assertEqual([item["type"] for item in first_content], ["text"])
        self.assertEqual(
            [item.get("path") for item in second_content if item["type"] == "image"],
            ["/images/a.jpg", "/images/b.jpg"],
        )
        self.assertTrue(kwargs["tokenize"])
        self.assertEqual(
            kwargs["processor_kwargs"],
            {"text_kwargs": {"padding": True, "padding_side": "left"}},
        )
        self.assertNotIn("padding", kwargs)
        self.assertNotIn("padding_side", kwargs)
        self.assertIs(inputs.moved_to, wrapper.device)

    def test_llava_prompt_body_removes_reserved_image_tokens(self) -> None:
        wrapper = freeret.FreeRetQwen.__new__(freeret.FreeRetQwen)
        wrapper.processor = FakeOneVisionProcessor()
        candidate = freeret.CandidateItem(
            candidate_id="webqa-candidate-1",
            text="visible information",
            image_path="/images/a.jpg",
        )
        prompt = freeret.one_word_prompt("webqa_t2it", "target", candidate)

        messages = wrapper._llava_messages(
            [prompt],
            [[candidate.image_path]],
            image_source_key=None,
        )
        content = messages[0][0]["content"]
        text_body = content[-1]["text"]

        self.assertEqual([item["type"] for item in content], ["image", "text"])
        self.assertNotIn(freeret.IMAGE_TOKEN, text_body)
        self.assertNotIn("<image>", text_body)
        self.assertIn("image:", text_body)

    def test_llava_prompt_body_sanitizes_repeated_coco_image_labels(self) -> None:
        wrapper = freeret.FreeRetQwen.__new__(freeret.FreeRetQwen)
        wrapper.processor = FakeOneVisionProcessor()
        candidate = freeret.CandidateItem(
            candidate_id="coco-image-1",
            image_path="/images/a.jpg",
        )
        prompt = freeret.one_word_prompt("coco_t2i", "target", candidate)

        messages = wrapper._llava_messages(
            [prompt],
            [[candidate.image_path]],
            image_source_key=None,
        )
        text_body = messages[0][0]["content"][-1]["text"]

        self.assertNotIn("<image>", text_body)
        self.assertIn("given image", text_body)

    def test_stage1_prompt_reports_missing_required_image(self) -> None:
        candidate = freeret.CandidateItem(
            candidate_id="webqa-candidate-missing-image",
            text="visible information",
        )

        with self.assertRaisesRegex(
            ValueError,
            "webqa-candidate-missing-image.*requires 1 image",
        ):
            freeret.one_word_prompt("webqa_t2it", "target", candidate)

    def test_llava_rendered_prompt_rejects_image_count_mismatch(self) -> None:
        wrapper = freeret.FreeRetQwen.__new__(freeret.FreeRetQwen)
        wrapper.processor = FakeOneVisionProcessor()

        with self.assertRaisesRegex(RuntimeError, "contains 2 image token"):
            wrapper._validate_llava_rendered_image_counts(
                ["USER: <image> <image>"],
                [["/images/a.jpg"]],
            )

    def test_llava_next_keeps_images_attached_to_each_conversation(self) -> None:
        wrapper = freeret.FreeRetQwen.__new__(freeret.FreeRetQwen)
        wrapper.processor = FakeLlavaNextProcessor()
        wrapper.device = torch.device("cpu")
        wrapper.transformers_version = "test"

        with tempfile.TemporaryDirectory() as temporary_directory:
            image_paths = [
                Path(temporary_directory) / "a.bmp",
                Path(temporary_directory) / "b.png",
            ]
            Image.new("L", (3, 4), color=64).save(image_paths[0])
            Image.new("RGBA", (5, 6), color=(1, 2, 3, 128)).save(image_paths[1])

            inputs = wrapper._prepare_llava_next_inputs(
                ["first", "second"],
                [[str(image_paths[0])], [str(image_paths[1])]],
            )

        messages, kwargs = wrapper.processor.template_calls[-1]
        self.assertTrue(kwargs["tokenize"])
        self.assertTrue(kwargs["return_dict"])
        self.assertEqual(
            kwargs["processor_kwargs"],
            {"text_kwargs": {"padding": True, "padding_side": "left"}},
        )
        self.assertNotIn("padding", kwargs)
        self.assertNotIn("padding_side", kwargs)
        self.assertEqual(len(messages), 2)
        for conversation, expected_size in zip(messages, [(3, 4), (5, 6)]):
            content = conversation[0]["content"]
            self.assertEqual([item["type"] for item in content], ["image", "text"])
            image = content[0]["image"]
            self.assertIsInstance(image, Image.Image)
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, expected_size)
        self.assertIs(inputs.moved_to, wrapper.device)

    def test_llava_next_image_error_reports_path_and_batch_item(self) -> None:
        wrapper = freeret.FreeRetQwen.__new__(freeret.FreeRetQwen)

        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "broken.bmp"
            image_path.write_bytes(b"not an image")

            with self.assertRaisesRegex(
                RuntimeError,
                r"batch item 7: path=.*broken\.bmp.*suffix='\.bmp'",
            ):
                wrapper._load_llava_next_image(str(image_path), batch_index=7)

    def test_validate_images_accepts_pillow_supported_bmp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "candidate.bmp"
            Image.new("RGB", (4, 5), color=(1, 2, 3)).save(image_path)
            task = freeret.RetrievalTask(
                dataset="CUHK-PEDES",
                split="test",
                direction="t2i",
                prompt_key="cuhk_pedes_t2i",
                queries=[],
                candidates=[
                    freeret.CandidateItem(
                        candidate_id="candidate-bmp",
                        image_path=str(image_path),
                    )
                ],
            )

            freeret._validate_images(task)

    def test_validate_images_reports_invalid_candidate_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "broken.bmp"
            image_path.write_bytes(b"not an image")
            task = freeret.RetrievalTask(
                dataset="CUHK-PEDES",
                split="test",
                direction="t2i",
                prompt_key="cuhk_pedes_t2i",
                queries=[],
                candidates=[
                    freeret.CandidateItem(
                        candidate_id="candidate-broken",
                        image_path=str(image_path),
                    )
                ],
            )

            with self.assertRaisesRegex(
                RuntimeError,
                r"candidate-broken: .*broken\.bmp",
            ):
                freeret._validate_images(task)

    def test_chat_template_processing_kwargs_supports_legacy_api(self) -> None:
        def legacy_apply_chat_template(
            messages: object, **kwargs: object
        ) -> object:
            return messages, kwargs

        self.assertEqual(
            freeret.FreeRetQwen._chat_template_processing_kwargs(
                legacy_apply_chat_template
            ),
            {"padding": True, "padding_side": "left"},
        )

    def test_llava_next_supports_text_only_batches(self) -> None:
        wrapper = freeret.FreeRetQwen.__new__(freeret.FreeRetQwen)
        wrapper.processor = FakeLlavaNextProcessor()
        wrapper.device = torch.device("cpu")
        wrapper.transformers_version = "test"

        wrapper._prepare_llava_next_inputs(
            ["first text", "second text"],
            [[], []],
        )

        messages, kwargs = wrapper.processor.template_calls[-1]
        self.assertTrue(kwargs["tokenize"])
        for conversation in messages:
            content = conversation[0]["content"]
            self.assertEqual([item["type"] for item in content], ["text"])

    def test_llava_next_processor_metadata_comes_from_model_config(self) -> None:
        wrapper = freeret.FreeRetQwen.__new__(freeret.FreeRetQwen)
        wrapper.processor = FakeLlavaNextProcessor()
        wrapper.model = SimpleNamespace(
            config=SimpleNamespace(
                vision_config=SimpleNamespace(patch_size=14),
                vision_feature_select_strategy="default",
            )
        )

        wrapper._configure_llava_next_processor()

        self.assertEqual(wrapper.processor.patch_size, 14)
        self.assertEqual(
            wrapper.processor.vision_feature_select_strategy, "default"
        )

    def test_ov15_uses_qwen_vision_inputs_and_filters_metadata(self) -> None:
        wrapper = freeret.FreeRetQwen.__new__(freeret.FreeRetQwen)
        wrapper.processor = FakeOneVision15Processor()
        wrapper.device = torch.device("cpu")
        wrapper._reported_dropped_input_keys = set()
        captured_messages: list[object] = []

        def fake_process_vision_info(messages: object) -> tuple[list[str], list[object]]:
            captured_messages.append(messages)
            return ["image-a", "image-b"], []

        with patch.dict(
            freeret.__dict__, {"process_vision_info": fake_process_vision_info}
        ):
            inputs = wrapper._prepare_llava_onevision_1_5_inputs(
                ["first", "second"],
                [["/images/a.jpg"], ["/images/b.jpg"]],
            )

        self.assertEqual(len(captured_messages), 1)
        messages = captured_messages[0]
        self.assertEqual(
            messages[0][0]["content"][0],
            {"type": "image", "image": "/images/a.jpg"},
        )
        self.assertNotIn("mm_token_type_ids", inputs)
        self.assertNotIn("second_per_grid_ts", inputs)
        self.assertIn("image_grid_thw", inputs)
        self.assertEqual(
            wrapper.processor.processor_kwargs["images"], ["image-a", "image-b"]
        )

    def test_ov15_rejects_missing_decoder_weights(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "decoder checkpoint mapping"):
            freeret.FreeRetQwen._validate_ov15_loading_info(
                {
                    "missing_keys": [
                        "model.language_model.layers.0.self_attn.q_proj.weight"
                    ],
                    "unexpected_keys": ["model.layers.0.self_attn.q_proj.weight"],
                }
            )


if __name__ == "__main__":
    unittest.main()
