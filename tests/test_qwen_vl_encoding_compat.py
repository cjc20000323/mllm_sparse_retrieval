"""Regression coverage for Transformers' old and new Qwen MRoPE APIs."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qwen_vl_encoding import _forward_inputs, rope_positions


class QwenRopeCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.inputs = {
            "input_ids": torch.tensor([[0, 5, 11, 11, 6], [5, 12, 12, 6, 0]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 0]]),
            "image_grid_thw": torch.tensor([[1, 2, 4]]),
            "video_grid_thw": torch.tensor([[1, 2, 4]]),
            "second_per_grid_ts": torch.tensor([0.5]),
        }
        self.config = SimpleNamespace(image_token_id=11, video_token_id=12)
        self.expected_types = torch.tensor([[0, 0, 1, 1, 0], [0, 2, 2, 0, 0]])
        self.positions = torch.arange(5).expand(3, 2, -1)

    def new_model(self, *, nested=False):
        def get_rope_index(input_ids, mm_token_type_ids, image_grid_thw=None,
                           video_grid_thw=None, second_per_grid_ts=None, attention_mask=None):
            torch.testing.assert_close(mm_token_type_ids, self.expected_types)
            self.assertEqual(mm_token_type_ids.device, input_ids.device)
            self.assertIs(image_grid_thw, self.inputs["image_grid_thw"])
            self.assertIs(video_grid_thw, self.inputs["video_grid_thw"])
            self.assertIs(second_per_grid_ts, self.inputs["second_per_grid_ts"])
            self.assertIs(attention_mask, self.inputs["attention_mask"])
            return self.positions, None
        owner = SimpleNamespace(get_rope_index=get_rope_index)
        return SimpleNamespace(config=self.config, model=owner) if nested else SimpleNamespace(
            config=self.config, get_rope_index=get_rope_index)

    def test_required_modality_ids_for_outer_and_nested_rope_owners(self):
        for nested in (False, True):
            with self.subTest(nested=nested):
                self.assertIs(rope_positions(self.new_model(nested=nested), self.inputs), self.positions)

    def test_old_rope_does_not_receive_new_metadata(self):
        def old_rope(input_ids, image_grid_thw=None, video_grid_thw=None, attention_mask=None):
            return self.positions, None
        inputs = {**self.inputs, "mm_token_type_ids": self.expected_types}
        model = SimpleNamespace(get_rope_index=old_rope)
        self.assertIs(rope_positions(model, inputs), self.positions)

    def test_processor_tensor_is_preserved(self):
        types = self.expected_types.to(torch.int32)
        def get_rope_index(input_ids, mm_token_type_ids, **kwargs):
            self.assertIs(mm_token_type_ids, types)
            return self.positions, None
        model = SimpleNamespace(get_rope_index=get_rope_index)
        rope_positions(model, {**self.inputs, "mm_token_type_ids": types})

    def test_invalid_processor_shape_fails_before_rope(self):
        with self.assertRaisesRegex(ValueError, "same shape as input_ids"):
            rope_positions(self.new_model(), {**self.inputs, "mm_token_type_ids": torch.zeros(2, 1)})

    def test_missing_visual_token_config_fails_instead_of_using_text_positions(self):
        model = self.new_model()
        model.config = SimpleNamespace(video_token_id=12)
        with self.assertRaisesRegex(ValueError, "config lacks image_token_id"):
            rope_positions(model, self.inputs)

    def test_internal_typeerror_is_not_retried_as_an_old_interface(self):
        calls = []
        def get_rope_index(input_ids, mm_token_type_ids, **kwargs):
            calls.append(mm_token_type_ids)
            raise TypeError("internal RoPE failure")
        model = SimpleNamespace(config=self.config, get_rope_index=get_rope_index)
        with self.assertRaisesRegex(TypeError, "internal RoPE failure"):
            rope_positions(model, self.inputs)
        self.assertEqual(len(calls), 1)

    def test_forward_dispatch_preserves_supported_metadata_without_mutating_input(self):
        def new_forward(input_ids, mm_token_type_ids, **kwargs):
            pass
        model = SimpleNamespace(config=self.config, forward=new_forward)
        torch.testing.assert_close(_forward_inputs(model, self.inputs)["mm_token_type_ids"], self.expected_types)
        self.assertNotIn("mm_token_type_ids", self.inputs)
        provided = {**self.inputs, "mm_token_type_ids": self.expected_types}
        self.assertIs(_forward_inputs(model, provided)["mm_token_type_ids"], self.expected_types)
        def old_forward(input_ids, **kwargs):
            pass
        model.forward = old_forward
        self.assertNotIn("mm_token_type_ids", _forward_inputs(model, provided))
        self.assertIs(provided["mm_token_type_ids"], self.expected_types)

    def test_kwargs_wrapper_uses_native_base_forward_capabilities(self):
        def base_forward(input_ids, mm_token_type_ids, **kwargs):
            pass
        model = SimpleNamespace(config=self.config, forward=Mock(), model=SimpleNamespace(forward=base_forward))
        torch.testing.assert_close(_forward_inputs(model, self.inputs)["mm_token_type_ids"], self.expected_types)


if __name__ == "__main__":
    unittest.main()
