"""Select alpha on val when supplied, otherwise reuse target retrieval scores."""
from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import test_qwen25_retrieval as shared

pipeline = shared.pipeline


class Dataset:
    data_name = "flickr"

    def get_target(self, native_id, query_type=None):
        return ["good-" + native_id]


class QwenSearchAlphaSelectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="qwen alpha selection ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = shared.Data(use_generated_validation_data=True)
        self.args = SimpleNamespace(model_name_or_path="Qwen-fixture")
        self.training = SimpleNamespace(task_type="ir", output_dir=str(self.root))
        self.model = SimpleNamespace(config=SimpleNamespace(model_type="qwen2_5_vl"))
        self.search = SimpleNamespace(query_type="image", use_fixed_alpha=False, alpha=0.9, depth=2,
                                      passage_reps="test-dense", sparse_index="test-sparse",
                                      val_passage_reps=None, val_sparse_index=None,
                                      validation_dataset_name=None, use_fixed_validation_dataset=False)
        self.records = [dict(id=f"q{i}", native_id=f"q{i}") for i in range(2)]
        self.indexes = []
        self.val_error = None
        self.zero_scores = False
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(redirect_stderr(io.StringIO()))
        self.runtime = stack.enter_context(patch.object(pipeline, "load_runtime", side_effect=lambda *a: (
            self.model, None, 0, 1)))
        self.build = stack.enter_context(patch.object(pipeline, "build_dataset", side_effect=self.build_dataset))
        stack.enter_context(patch.object(pipeline, "dataset_records", side_effect=lambda *a: self.records))
        stack.enter_context(patch.object(pipeline, "prompt_metadata", side_effect=self.metadata))
        self.encoder = stack.enter_context(patch.object(pipeline, "NativeQwenEncoder", side_effect=lambda *a: SimpleNamespace(last_prompts=[])))
        stack.enter_context(patch.object(pipeline, "CorpusSearch", side_effect=self.make_index))
        stack.enter_context(patch.object(pipeline.dist, "is_initialized", return_value=False))

    def build_dataset(self, data, task, processor, **kwargs):
        dataset = Dataset()
        dataset.data_name = data.dataset_name
        return dataset

    def metadata(self, args, data, training, processor, kind, *, family):
        return dict(fingerprint="fixture", kind=kind, config=dict(
            version=shared.prompts.PROMPT_VERSIONS[family], split=data.dataset_split))

    def make_index(self, dense_path, sparse_path, expected, search, rank):
        is_val = expected["config"]["split"] == "val"
        if is_val and self.val_error is not None:
            raise self.val_error
        def retrieve(encoder, records, data):
            dense, sparse = {}, {}
            for record in records:
                correct, wrong = "good-" + record["id"], "wrong-" + record["id"]
                # Test favors dense, while val favors sparse. At alpha=0.5 the
                # wrong doc wins the tie, so target's best alpha starts at 0.6.
                high = 0.0 if self.zero_scores else 1.0
                dense[record["id"]] = pipeline._run({wrong: high if is_val else 0.0,
                                                    correct: 0.0 if is_val else high})
                sparse[record["id"]] = pipeline._run({wrong: 0.0 if is_val else high,
                                                     correct: high if is_val else 0.0})
            return dense, sparse
        index = SimpleNamespace(paths=(dense_path, sparse_path), retrieve=Mock(side_effect=retrieve), close=Mock())
        self.indexes.append(index)
        return index

    def run_search(self):
        pipeline.run_search(self.args, self.data, self.search, self.training)
        return json.loads((Path(self.training.output_dir) / "recall_metrics.json").read_text(encoding="utf-8"))

    def test_target_selection_reuses_one_retrieval_for_all_three_families(self):
        from hybrid import fuse
        for family in sorted(shared.prompts.MODEL_TYPES):
            with self.subTest(family=family):
                self.model.config.model_type = family
                self.training.output_dir = str(self.root / family)
                self.indexes.clear()
                self.encoder.reset_mock()
                self.build.reset_mock()
                with patch("hybrid.fuse", wraps=fuse) as fusion, patch.object(pipeline, "recall", wraps=pipeline.recall) as evaluate:
                    report = self.run_search()
                self.assertEqual(report["alpha"], 0.6)
                self.assertEqual(report["alpha_selection"], dict(source="target", dataset="flickr", split="test"))
                self.assertEqual(self.encoder.call_count, 1)
                self.assertEqual(self.build.call_count, 1)
                self.assertNotIn("generated", self.build.call_args.kwargs)
                self.assertEqual(len(self.indexes), 1)
                self.indexes[0].retrieve.assert_called_once()
                self.indexes[0].close.assert_called_once()
                self.assertEqual(fusion.call_count, 9)  # Best run and metrics are reused.
                self.assertEqual(evaluate.call_count, 11)  # Two baselines and nine hybrid candidates.
                self.assertEqual([call.args[1][0] for call in fusion.call_args_list[:9]], [i / 10 for i in range(1, 10)])
                self.assertEqual(report["metrics"]["hybrid"]["Recall@1"], 100)
                first = (Path(self.training.output_dir) / "hybrid_run.trec").read_text().splitlines()[0]
                self.assertTrue(first.split()[2].startswith("good-"))
                output = Path(self.training.output_dir)
                self.assertTrue((output / "alpha_sweep.csv").read_bytes().startswith(b"\xef\xbb\xbf"))
                with (output / "alpha_sweep.csv").open(encoding="utf-8-sig", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                with (output / "best.csv").open(encoding="utf-8-sig", newline="") as stream:
                    best = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 27)
                self.assertEqual(len(best), 3)
                self.assertEqual({float(row["alpha"]) for row in rows}, {i / 10 for i in range(1, 10)})
                self.assertEqual(sum(row["selected"] == "True" for row in rows), 1)
                self.assertEqual(best, [row for row in rows if float(row["alpha"]) == report["alpha"]])
                self.assertFalse((output / "best.xlsx").exists())

    def test_available_validation_wins_even_when_target_prefers_a_different_alpha(self):
        self.search.val_passage_reps, self.search.val_sparse_index = "val-dense", "val-sparse"
        report = self.run_search()
        self.assertEqual(report["alpha"], 0.1)
        self.assertEqual(report["alpha_selection"], dict(source="validation", dataset="flickr", split="val"))
        self.assertEqual(report["metrics"]["hybrid"]["Recall@1"], 0)
        self.assertEqual(self.encoder.call_count, 2)
        self.assertEqual(self.build.call_count, 2)
        self.assertTrue(self.build.call_args.kwargs["generated"])
        self.assertEqual([index.paths for index in self.indexes], [("test-dense", "test-sparse"), ("val-dense", "val-sparse")])
        for index in self.indexes:
            index.retrieve.assert_called_once()
            index.close.assert_called_once()
        with (self.root / "alpha_sweep.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        with (self.root / "best.csv").open(encoding="utf-8-sig", newline="") as stream:
            best = list(csv.DictReader(stream))
        self.assertEqual({row["split"] for row in rows}, {"val"})
        self.assertEqual({row["split"] for row in best}, {"test"})
        self.assertEqual({row["selection_source"] for row in rows + best}, {"validation"})

    def test_fixed_alpha_skips_even_explicit_invalid_validation_paths(self):
        from hybrid import fuse
        self.search.use_fixed_alpha, self.search.alpha = True, 0.3
        self.search.val_passage_reps, self.search.val_sparse_index = "/missing dense", "/missing sparse"
        self.val_error = FileNotFoundError("must not read val")
        with patch("hybrid.fuse", wraps=fuse) as fusion:
            report = self.run_search()
        self.assertEqual(report["alpha"], 0.3)
        self.assertEqual(report["alpha_selection"], dict(source="fixed", dataset=None, split=None))
        self.assertEqual(fusion.call_count, 1)
        self.assertEqual(self.build.call_count, 1)
        self.assertEqual(len(self.indexes), 1)
        self.assertTrue((self.root / "best.csv").is_file())
        self.assertFalse((self.root / "alpha_sweep.csv").exists())

    def test_other_target_split_is_recorded_without_relabeling_it_as_test(self):
        self.data.dataset_split = "dev"
        report = self.run_search()
        self.assertEqual(report["alpha"], 0.6)
        self.assertEqual(report["alpha_selection"]["split"], "dev")
        self.assertEqual(self.build.call_count, 1)

    def test_partial_explicit_validation_arguments_fail_before_model_loading(self):
        for paths in (("val-dense", None), (None, "val-sparse"), ("", "")):
            with self.subTest(paths=paths):
                self.search.val_passage_reps, self.search.val_sparse_index = paths
                with self.assertRaisesRegex(ValueError, "requires both"):
                    self.run_search()
        self.runtime.assert_not_called()

    def test_validation_io_or_manifest_errors_do_not_retry_on_target(self):
        self.search.val_passage_reps, self.search.val_sparse_index = "val-dense", "val-sparse"
        for error in (FileNotFoundError("missing val index"), ValueError("Qwen prompt/configuration mismatch")):
            with self.subTest(error=error):
                self.val_error = error
                with self.assertRaisesRegex(type(error), str(error)):
                    self.run_search()
        self.assertFalse((self.root / "recall_metrics.json").exists())

    def test_all_tied_scores_choose_the_lowest_candidate(self):
        self.zero_scores = True
        self.assertEqual(self.run_search()["alpha"], 0.1)

    def test_fashion_iq_target_uses_category_recall_criterion(self):
        self.data.dataset_name = "fashion-iq"
        self.training.task_type = "cir"
        self.search.query_type = "composed"
        self.records = [dict(id=f"q{i}", native_id=f"q{i}", dress_type=category)
                        for i, category in enumerate(("dress", "shirt", "toptee"))]
        report = self.run_search()
        self.assertEqual(report["alpha"], 0.1)  # All three categories tie at R@10/R@50.
        self.assertEqual(report["alpha_selection"], dict(source="target", dataset="fashion-iq", split="test"))
        self.assertEqual(set(report["metrics"]["hybrid"]["by_category"]), {"dress", "shirt", "toptee"})
        with (self.root / "alpha_sweep.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 108)
        self.assertEqual({row["category"] for row in rows}, {"all", "dress", "shirt", "toptee"})
        self.assertEqual(sum(row["selected"] == "True" for row in rows), 4)


if __name__ == "__main__":
    unittest.main()
