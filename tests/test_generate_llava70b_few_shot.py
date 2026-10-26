import argparse
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from PIL import Image

from src.generate_llava70b_dataset_text import (
    DatasetGenerator,
    FewShotExample,
    ImageSample,
    Llava70BGenerator,
)


def _png_bytes(color: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


class _FakeProcessor:
    def __init__(self) -> None:
        self.conversation = None
        self.call_kwargs = None

    def apply_chat_template(self, conversation, **kwargs):
        self.conversation = conversation
        return "rendered"

    def __call__(self, **kwargs):
        self.call_kwargs = kwargs
        return {"input_ids": "stub"}


def _args(root: str, enabled: bool = True) -> argparse.Namespace:
    return argparse.Namespace(
        source_root=root,
        output_root=root,
        seed=2026,
        split="test",
        num_images=2,
        model_name_or_path="stub",
        use_few_shot_examples=enabled,
        few_shot_num=2,
        few_shot_split="val",
    )


class FewShotProcessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = object.__new__(Llava70BGenerator)
        self.generator.processor = _FakeProcessor()

    def test_disabled_path_keeps_single_image_input(self) -> None:
        target = Image.new("RGB", (2, 2), color="white")

        self.generator._processor_inputs(target, "target prompt")

        self.assertIs(
            self.generator.processor.call_kwargs["images"],
            target,
        )
        self.assertEqual(
            [turn["role"] for turn in self.generator.processor.conversation],
            ["user"],
        )

    def test_examples_precede_target_and_use_their_base_prompt(self) -> None:
        examples = [
            FewShotExample(
                ImageSample("one", image_bytes=_png_bytes("red")),
                "first answer",
            ),
            FewShotExample(
                ImageSample("two", image_bytes=_png_bytes("blue")),
                "second answer",
            ),
        ]
        target = Image.new("RGB", (2, 2), color="white")

        self.generator._processor_inputs(
            target,
            "repair target",
            examples,
            "base prompt",
        )

        conversation = self.generator.processor.conversation
        self.assertEqual(
            [turn["role"] for turn in conversation],
            ["user", "assistant", "user", "assistant", "user"],
        )
        self.assertEqual(conversation[0]["content"][1]["text"], "base prompt")
        self.assertEqual(conversation[1]["content"][0]["text"], "first answer")
        self.assertEqual(conversation[-1]["content"][1]["text"], "repair target")
        self.assertEqual(len(self.generator.processor.call_kwargs["images"]), 3)


class FewShotSamplingTests(unittest.TestCase):
    def test_fixed_sampling_is_deterministic_and_excludes_targets(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            dataset_generator = DatasetGenerator(_args(root), model=None)
            source = [
                ImageSample(
                    str(index),
                    image_bytes=_png_bytes("red"),
                    meta={"row": [str(index), "name.jpg", f"caption {index}", "sid"]},
                )
                for index in range(6)
            ]
            target = [source[0], source[1]]
            with patch.object(
                dataset_generator,
                "_few_shot_source_samples",
                return_value=source,
            ):
                first = dataset_generator._prepare_few_shot_examples(
                    "flickr", target
                )
                second = dataset_generator._prepare_few_shot_examples(
                    "flickr", target
                )

            first_ids = [example.sample.image_id for example in first]
            second_ids = [example.sample.image_id for example in second]
            self.assertEqual(first_ids, second_ids)
            self.assertEqual(len(first_ids), 2)
            self.assertTrue(set(first_ids).isdisjoint({"0", "1"}))

    def test_disabled_mode_does_not_add_manifest_fields(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            dataset_generator = DatasetGenerator(
                _args(root, enabled=False),
                model=None,
            )

        self.assertNotIn("use_few_shot_examples", dataset_generator.manifest)
        self.assertNotIn("few_shot_num", dataset_generator.manifest)
        self.assertEqual(
            dataset_generator._prepare_few_shot_examples("flickr", []),
            [],
        )

    def test_webqa_reference_uses_qrels_question_and_corpus_answer(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            dataset_generator = DatasetGenerator(_args(root), model=None)
            frames = {
                "qrels-00000-of-00001.parquet": pd.DataFrame(
                    [{"query-id": "q1", "corpus-id": "c1"}]
                ),
                "query-00000-of-00001.parquet": pd.DataFrame(
                    [{"id": "q1", "text": "What is visible?"}]
                ),
                "corpus-00000-of-00002.parquet": pd.DataFrame(
                    [
                        {
                            "id": "c1",
                            "text": "A red object is visible.",
                            "image": {"bytes": _png_bytes("red")},
                        }
                    ]
                ),
                "corpus-00001-of-00002.parquet": pd.DataFrame(
                    columns=["id", "text", "image"]
                ),
            }

            def read_frame(path: Path) -> pd.DataFrame:
                return frames[path.name]

            with patch.object(
                dataset_generator,
                "_read_parquet",
                side_effect=read_frame,
            ):
                samples = dataset_generator._t2it_samples(
                    "webqa",
                    "val",
                    include_reference=True,
                )

        self.assertEqual(len(samples), 1)
        response = json.loads(
            dataset_generator._few_shot_response("webqa", samples[0])
        )
        self.assertEqual(response["question"], "What is visible?")
        self.assertEqual(response["answer"], "A red object is visible.")


if __name__ == "__main__":
    unittest.main()
