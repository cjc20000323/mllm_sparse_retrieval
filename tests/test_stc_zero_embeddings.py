"""Zero-row exclusion, independent ReLU weights and pinned v1-map compatibility."""
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_freeret_dense import temporary_directory, data_args
from test_stc_sparse import Tokenizer, args_at
import adaptive_views as views
import stc_sparse as stc


def legacy_copy(vocabulary, directory):
    """An authentic v1 schema: no zero-policy field or embedding diagnostics."""
    directory = Path(directory)
    manifest = copy.deepcopy(vocabulary.manifest)
    manifest["version"] = manifest["request"]["version"] = 1
    manifest["request"].pop("zero_norm_policy")
    manifest.pop("embedding_filter")
    manifest["fingerprint"] = stc.digest({k: v for k, v in manifest.items() if k != "fingerprint"})
    directory.mkdir(parents=True, exist_ok=True)
    stc._save_arrays(directory / "token_to_cluster.npz", vocabulary.arrays)
    views.atomic_json(directory / "manifest.json", manifest)
    return stc.Vocabulary(directory, expected=manifest["request"])


class ZeroEmbeddingTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "config.json").write_text("{}", encoding="utf-8")
        self.tokenizer = Tokenizer()
        tokenizer_patch = patch.object(stc, "_tokenizer", return_value=self.tokenizer)
        tokenizer_patch.start()
        self.addCleanup(tokenizer_patch.stop)
        self.args = args_at(self.root)
        self.weights = torch.tensor([[1., 1.], [1., 0.], [.99, .01], [0., 1.], [.01, .99],
                                     [1., 0.], [1., 0.], [1., 1.], [1., 1.], [1., 1.]])

    def build(self, inputs=None, outputs=None, args=None, directory=None):
        args = args or self.args
        config, _ = stc.request(args, tokenizer=self.tokenizer)
        return stc.build_mapping(self.weights if inputs is None else inputs, self.weights if outputs is None else outputs,
                                 self.tokenizer, args, directory or stc.cache_path(args, config), config)

    def test_either_zero_side_and_numeric_tokens_are_excluded_for_both_algorithms(self):
        cases = (([1], []), ([], [1]), ([1], [1]), ([5], [6]), ([1, 5], [4, 5]))
        for algorithm in ("agglomerative", "minibatch_kmeans"):
            for index, (input_ids, output_ids) in enumerate(cases):
                with self.subTest(algorithm=algorithm, input=input_ids, output=output_ids):
                    inputs, outputs = self.weights.clone(), self.weights.clone()
                    inputs[input_ids] = 0
                    outputs[output_ids] = 0
                    inputs[[0, 7, 9]] = 0  # special/blank rows were already excluded.
                    args = args_at(self.root, stc_algorithm=algorithm)
                    with redirect_stdout(io.StringIO()) as log:
                        vocabulary = self.build(inputs, outputs, args, self.root / f"{algorithm}_{index}")
                    excluded = sorted(set(input_ids) | set(output_ids))
                    expected_ids = [i for i in (1, 2, 3, 4, 5, 6) if i not in excluded]
                    self.assertEqual(vocabulary.arrays["token_ids"].tolist(), expected_ids)
                    self.assertEqual(vocabulary.arrays["vocab_ids"].tolist(), sorted(self.tokenizer.get_vocab().values()))
                    report = vocabulary.manifest["embedding_filter"]
                    self.assertEqual(report["excluded_zero_token_ids"], excluded)
                    self.assertEqual(report["input_zero_token_ids"], input_ids)
                    self.assertEqual(report["output_zero_token_ids"], output_ids)
                    self.assertEqual(report["checked_token_count"], 6)
                    self.assertEqual(report["retained_token_count"], len(expected_ids))
                    self.assertEqual(report["excluded_zero_count"], len(excluded))
                    self.assertEqual(vocabulary.manifest["numeric_dimensions"], len(set(expected_ids) & {5, 6}))
                    self.assertEqual(vocabulary.manifest["cluster_count"], 2 + len(set(expected_ids) & {5, 6}))
                    self.assertIn("excluded_zero=" + str(len(excluded)), log.getvalue())
                    with redirect_stdout(io.StringIO()):
                        again = self.build(inputs, outputs, args, vocabulary.directory)
                    self.assertEqual(again.fingerprint, vocabulary.fingerprint)

    def test_excluded_token_weights_never_enter_sparse_clusters_or_affect_other_tokens(self):
        inputs = self.weights.clone()
        inputs[1] = 0
        with redirect_stdout(io.StringIO()):
            self.build(inputs)
        encoder = stc.SparseEncoder(self.tokenizer, data_args(sparse_length=10), self.args)
        self.assertNotIn(1, encoder.ids.tolist())
        self.assertIn(1, encoder.vocab_ids.tolist())
        logits = torch.tensor([.02, .17, .17, .30, .01, .15, .10, .03, 0., .05])
        logits[8] = 1000  # nonexistent padded row must not contribute.
        vector = encoder.vector(logits[None])
        self.assertEqual(sum(vector.values()), 73)  # drop .17 zero + .07 special + .03 blank.
        logits[1] = 1e6  # the excluded token cannot suppress other weights.
        self.assertEqual(encoder.vector(logits[None]), vector)
        self.assertTrue(all(key.startswith("stc_") for key in vector))

    def test_nan_inf_and_nonfinite_norms_still_fail_before_publishing(self):
        cases = ((1, float("nan"), "input_embedding"), (6, float("inf"), "output_embedding"),
                 (2, 1e38, "input_embedding"))
        for index, (token, value, matrix) in enumerate(cases):
            with self.subTest(matrix=matrix, value=value):
                inputs, outputs = self.weights.clone(), self.weights.clone()
                # A zero row on the other side cannot conceal the invalid row.
                (inputs if matrix == "output_embedding" else outputs)[token] = 0
                (inputs if matrix == "input_embedding" else outputs)[token] = value
                directory = self.root / f"invalid_{index}"
                with self.assertRaisesRegex(ValueError, "non-finite") as error:
                    self.build(inputs, outputs, directory=directory)
                self.assertIn(matrix, str(error.exception))
                self.assertIn('"id": ' + str(token), str(error.exception))
                self.assertIn("non_finite_rows_or_norms=1", str(error.exception))
                self.assertFalse((directory / "manifest.json").exists())
                self.assertFalse((directory / "token_to_cluster.npz").exists())

    def test_filtering_is_chunked_and_insufficient_remaining_clusters_are_reported(self):
        inputs = self.weights.clone()
        inputs[[1, 2, 3]] = 0
        with patch.object(stc, "EMBEDDING_CHECK_BATCH_SIZE", 2), redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(ValueError, "2 exceeds 1 eligible nonnumeric tokens after excluding 3"):
            self.build(inputs)
        self.assertFalse((stc.cache_path(self.args) / "manifest.json").exists())

    def test_v2_filter_report_and_mapping_must_agree_even_after_resigning(self):
        inputs = self.weights.clone()
        inputs[1] = 0
        with redirect_stdout(io.StringIO()):
            vocabulary = self.build(inputs)
        original = vocabulary.manifest
        for problem in ("missing", "wrong_count", "missing_union", "mapped_excluded", "wrong_policy"):
            with self.subTest(problem=problem):
                modified = copy.deepcopy(original)
                if problem == "missing":
                    modified.pop("embedding_filter")
                elif problem == "wrong_count":
                    modified["embedding_filter"]["retained_token_count"] += 1
                elif problem == "missing_union":
                    modified["embedding_filter"]["excluded_zero_token_ids"] = []
                elif problem == "wrong_policy":
                    modified["request"]["zero_norm_policy"] = "allow_zero"
                else:
                    report = modified["embedding_filter"]
                    report.update(input_zero_token_ids=[2], excluded_zero_token_ids=[2])
                modified["fingerprint"] = stc.digest({k: v for k, v in modified.items() if k != "fingerprint"})
                views.atomic_json(vocabulary.directory / "manifest.json", modified)
                with self.assertRaisesRegex(ValueError, "filter|policy"):
                    stc.Vocabulary(vocabulary.directory)
        views.atomic_json(vocabulary.directory / "manifest.json", original)
        self.assertEqual(stc.load(self.args).fingerprint, vocabulary.fingerprint)

    def test_v1_is_readable_only_as_a_pinned_map_with_current_relu_weighting(self):
        with redirect_stdout(io.StringIO()):
            fresh = self.build(directory=self.root / "source_v2")
        old_request = dict(fresh.manifest["request"], version=1)
        old_request.pop("zero_norm_policy")
        legacy = legacy_copy(fresh, stc.cache_path(self.args, old_request))
        self.assertNotEqual(stc.cache_path(self.args), legacy.directory)
        self.assertFalse(stc.cache_path(self.args).exists())
        with self.assertRaisesRegex(ValueError, "Missing or corrupt"):
            stc.load(self.args)  # Automatic lookup never falls back to the legacy cache.
        bound = copy.deepcopy(self.args)
        bound.stc_vocab_path = str(legacy.directory)
        expected = dict(version=1, method="token_topk_quantize_merge_then_cluster_sum", weighting_version=3,
                        weight_transform="log1p_relu", mapping_fingerprint=legacy.fingerprint,
                        request=old_request, cluster_count=legacy.manifest["cluster_count"],
                        weight_scale=100, quantization="round_token_x100_then_merge_floor_mean_drop_nonpositive",
                        term_mapping="normalized_term_min_valid_token_id", prefix_matching=False)
        self.assertEqual(stc.configuration(bound), expected)
        artifact = self.root / "old_corpus"
        stc.publish_bundle(artifact, legacy)
        stc.validate_bundle(artifact, expected)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            stc.validate_bundle(artifact, dict(expected, version=2))
        bound.stc_n_clusters = 3
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            stc.load(bound)
        self.args.dry_run, self.args.lora_model_path = False, None
        sentinel = object()
        with patch.object(stc, "_build_new", return_value=sentinel) as builder, redirect_stdout(io.StringIO()):
            self.assertIs(stc.build(self.args), sentinel)
        self.assertEqual(builder.call_args.args[-1]["version"], 2)
        self.assertEqual(builder.call_args.args[-1]["zero_norm_policy"], stc.ZERO_NORM_POLICY)
        self.assertFalse(list(stc.cache_path(self.args).parent.glob("*.lock")))
        self.assertEqual(stc.Vocabulary(legacy.directory).fingerprint, legacy.fingerprint)


if __name__ == "__main__":
    unittest.main()
