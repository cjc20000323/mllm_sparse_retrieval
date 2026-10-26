from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import reranker


class RerankerProcessModeTests(unittest.TestCase):
    def test_uninitialized_distributed_runtime_is_main_process(self) -> None:
        with (
            patch.object(reranker.dist, "is_available", return_value=True),
            patch.object(reranker.dist, "is_initialized", return_value=False),
            patch.object(
                reranker.dist,
                "get_rank",
                side_effect=AssertionError("get_rank must not be called"),
            ),
        ):
            self.assertTrue(reranker._is_main_process())

    def test_initialized_distributed_runtime_uses_actual_rank(self) -> None:
        with (
            patch.object(reranker.dist, "is_available", return_value=True),
            patch.object(reranker.dist, "is_initialized", return_value=True),
            patch.object(reranker.dist, "get_rank", return_value=1),
        ):
            self.assertFalse(reranker._is_main_process())

        with (
            patch.object(reranker.dist, "is_available", return_value=True),
            patch.object(reranker.dist, "is_initialized", return_value=True),
            patch.object(reranker.dist, "get_rank", return_value=0),
        ):
            self.assertTrue(reranker._is_main_process())

    def test_caption_rerank_does_not_require_a_process_group(self) -> None:
        ranker = reranker.Reranker.__new__(reranker.Reranker)
        with patch.object(
            reranker.dist,
            "get_world_size",
            side_effect=AssertionError("get_world_size must not be called"),
        ):
            result = ranker.caption_generation_rerank(
                {},
                "pointwise",
                20,
                SimpleNamespace(reps_loc="after_pad"),
                SimpleNamespace(fp16=False, task_type="ir"),
                SimpleNamespace(
                    model_name_or_path="llava-hf-llava-v1.6-mistral-7b-hf"
                ),
                SimpleNamespace(modify_type="no"),
            )

        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
