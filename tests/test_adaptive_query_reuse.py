"""Probability-mass query encodings survive real Stage-1/2 reuse validation."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import copy
import io
from itertools import product
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from test_freeret_dense import temporary_directory
from test_adaptive_search import corpus
import test_adaptive_view_shuffle as selections
import adaptive_view_encoding as backend
import adaptive_view_probability_mass as mass
import adaptive_views as views
import search_adaptive_views as search


ENCODING = dict(eol_type="disassembleeol_separate", calculate_type="concat", sparse_manual=False)


class QueryReuseTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.api = backend.pipeline_module()

    def fixture(self, name, *, method="dense", threshold=.95, mode="full", dataset="RSTPReid", shuffled=False, pmi=False):
        fixture = corpus(self.root / name, dataset=dataset, hierarchical=True, threshold=threshold,
                         mode=mode, shuffled=shuffled, pmi=pmi, sparse_length=30,
                         tbpr_type="type" if dataset == "RSTPReid" else "origin_type")
        experiment = search.prepare(fixture.args, identity_loader=lambda *a, **k: fixture.identity)
        args = experiment.pipeline_args
        args.view_probability_mass = threshold is not None
        args.view_probability_mass_threshold = threshold
        args.stage1_sparse_index = str(fixture.roots[1] / "collection/index")
        args.stage2_dense_index = str(fixture.roots[0])
        args.stage2_sparse_index = str(fixture.roots[1]) if method == "hybrid" else None
        args.stage2_method, args.stage1_depth, args.stage2_depth = method, 2, 1
        args.hybrid_alpha = .5
        fixture.pipeline_args = args
        return fixture

    @contextmanager
    def encoder(self, fixture, runtime=None):
        runtime = runtime or NS(encoder=NS(config=NS(model_type="qwen2_5_vl")), processor=NS(),
                                query_cache={}, device=torch.device("cpu"))
        records = [dict(id="q0", text="first query"), dict(id="q1", text="second query")]
        if fixture.pipeline_args.task_type == "it2t":
            records = [dict(row, image="query.png") for row in records]
        encoder = NS(last_prompts=["fixture prompt"])
        encoder.encode = Mock(side_effect=lambda rows: (
            [torch.zeros(len(fixture.selection["selected"]), 3) for row in rows],
            torch.tensor([[1., 0.] if row["id"] == "q0" else [0., 1.] for row in rows])))
        with patch.object(backend, "configure_processor"), \
                patch.object(views, "model_identity", return_value=fixture.identity), \
                patch.object(backend, "AdaptiveEncoder", return_value=encoder) as factory, \
                patch.object(backend, "sparse_vectors", side_effect=lambda e, rows, logits: [{"word": 2} for row in rows]), \
                patch.object(self.api, "build_dataset", return_value=NS()), \
                patch.object(self.api, "_query_records", return_value=records), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            yield runtime, factory

    def encode(self, fixture, runtime, ids=("q0", "q1")):
        return self.api.encode_queries_online(fixture.pipeline_args, runtime, **ENCODING,
            progress_prefix="test", requested_query_ids=list(ids))

    def test_full_prompt_mass_queries_pass_stage1_then_dense_or_hybrid_stage2(self):
        for method in ("dense", "hybrid"):
            with self.subTest(method=method):
                fixture = self.fixture(method, method=method)
                saved = {path: path.read_bytes() for root in fixture.roots for path in root.rglob("*") if path.is_file()}
                saved[fixture.selection_path] = fixture.selection_path.read_bytes()
                args = fixture.pipeline_args
                with self.encoder(fixture) as (runtime, factory):
                    encoded = self.encode(fixture, runtime)
                    self.assertIs(self.encode(fixture, runtime), encoded)
                    factory.assert_called_once()
                self.assertEqual(encoded.qwen_metadata["pipeline_settings"][mass.FIELD]["threshold"], .95)
                runtime.query_cache.clear()  # Stage 1 releases its model before handing CPU queries to Stage 2.
                sparse_search = Mock(side_effect=lambda index, topics, qids, **kw:
                                     {qid: {"20": 2., "10": 1.} for qid in qids})
                with patch.object(self.api, "retrieve_sparse_topics_from_lucene", sparse_search), \
                        patch.object(self.api, "encode_queries_online", side_effect=AssertionError("must reuse Stage-1 queries")), \
                        redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    # Keep _validate_qwen_artifacts, validate_encoded and corpus validation real.
                    stage1 = self.api.run_stage1(args, encoded_queries=encoded)
                    dense, sparse, selected = self.api.run_stage2(args, stage1, encoded_queries=encoded)
                self.assertEqual(set(selected), {"q0", "q1"})
                self.assertTrue(all(set(run).issubset(stage1[qid]) for qid, run in selected.items()))
                if method == "dense":
                    self.assertEqual(selected, dense)
                    self.assertFalse(sparse)
                    self.assertEqual(list(dense["q0"]), ["10"])
                    self.assertEqual(list(dense["q1"]), ["20"])
                self.assertEqual(sparse_search.call_count, 1 if method == "dense" else 2)
                for path, data in saved.items():
                    self.assertEqual(path.read_bytes(), data)

    def test_existing_cache_keys_and_metadata_are_compatible_across_tasks_and_prompt_modes(self):
        for dataset, mode, threshold in product(("flickr", "RSTPReid", "webqa", "remuq", "llava"),
                                                ("ordinary", "dense", "constraints", "full"), (None, .95)):
            with self.subTest(dataset=dataset, mode=mode, threshold=threshold):
                fixture = self.fixture(f"{dataset}_{mode}_{threshold}", dataset=dataset, mode=mode, threshold=threshold,
                                       shuffled=threshold is not None, pmi=threshold is not None)
                args = fixture.pipeline_args
                # Reproduce the pre-fix encoder contract independently of the new helper.
                previous = self.api._qwen_encoding_settings(args, **ENCODING)
                if threshold is not None:
                    previous = mass.add_metadata(previous, fixture.selection)
                    previous["views_fingerprint"] = fixture.selection["views_fingerprint"]
                with self.encoder(fixture) as (runtime, factory):
                    encoded = self.encode(fixture, runtime)
                    self.assertIs(self.encode(fixture, runtime), encoded)
                    factory.assert_called_once()
                self.assertEqual(encoded.qwen_metadata["pipeline_settings"], previous)
                self.assertEqual(list(runtime.query_cache), [("adaptive", views.digest(previous), ("q0", "q1"))])
                backend.validate_encoded(args, encoded, [(str(fixture.roots[0]), True), (str(fixture.roots[1]), False)],
                                         self.api, **ENCODING)
                if threshold is None:
                    self.assertNotIn(mass.FIELD, previous)
                    self.assertNotIn("selection_strategy", previous)

    def test_requested_threshold_mode_selection_and_prompt_changes_still_fail(self):
        fixture = self.fixture("conflicts")
        with self.encoder(fixture) as (runtime, _):
            encoded = self.encode(fixture, runtime)
        args = fixture.pipeline_args
        for field, value, message in (("view_probability_mass_threshold", .9, "threshold differs"),
                                      ("view_probability_mass", False, "mode differs"),
                                      ("use_freeret_full_prompt", False, "Adaptive query reuse")):
            with self.subTest(field=field):
                changed = copy.deepcopy(args)
                setattr(changed, field, value)
                with self.assertRaisesRegex(ValueError, message):
                    backend.validate_encoded(changed, encoded, [], self.api, **ENCODING)
        alternative = mass.derive(selections.choose(selections.options(shuffled=True, hierarchical=True),
                                  identity=fixture.identity, dataset="RSTPReid"), .95)
        self.assertNotEqual(alternative["views_fingerprint"], fixture.selection["views_fingerprint"])
        path = self.root / "other_selection.json"
        views.atomic_json(path, alternative)
        changed = copy.deepcopy(args)
        changed.view_selection_file = str(path)
        with self.assertRaisesRegex(ValueError, "Adaptive query reuse.*|views_fingerprint"):
            backend.validate_encoded(changed, encoded, [], self.api, **ENCODING)

    def test_stored_fields_are_compared_not_copied_into_requested_settings(self):
        fixture = self.fixture("stored_conflicts")
        with self.encoder(fixture) as (runtime, _):
            encoded = self.encode(fixture, runtime)
        for field, value in (("views_fingerprint", "0" * 64), ("selection_strategy", "wrong"),
                             (mass.FIELD, mass.configuration(.8))):
            with self.subTest(field=field):
                changed = copy.deepcopy(encoded)
                changed.qwen_metadata["pipeline_settings"][field] = value
                with self.assertRaisesRegex(ValueError, field):
                    backend.validate_encoded(fixture.pipeline_args, changed, [], self.api, **ENCODING)
        # Corpus compatibility remains checked after successful query-settings validation.
        bad = copy.deepcopy(encoded)
        bad.qwen_metadata["configuration"]["views_fingerprint"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Adaptive corpus"):
            backend.validate_encoded(fixture.pipeline_args, bad, [(str(fixture.roots[1]), False)], self.api, **ENCODING)

    def test_threshold_changes_isolate_cached_queries_and_rank_local_shards(self):
        fixture = self.fixture("cache_isolation")
        with self.encoder(fixture) as (runtime, factory):
            first = self.encode(fixture, runtime, ids=("q0",))
            second = self.encode(fixture, runtime, ids=("q1",))
            self.assertEqual(first.qwen_metadata["pipeline_settings"], second.qwen_metadata["pipeline_settings"])
            self.assertEqual(factory.call_count, 2)
            alternative = mass.derive(selections.choose(selections.options(shuffled=False, hierarchical=True),
                                      identity=fixture.identity, dataset="RSTPReid"), .7)
            path = self.root / "threshold_07.json"
            views.atomic_json(path, alternative)
            fixture.pipeline_args.view_selection_file = str(path)
            fixture.pipeline_args.view_probability_mass_threshold = .7
            third = self.encode(fixture, runtime, ids=("q0",))
            self.assertIsNot(third, first)
            self.assertEqual(len(runtime.query_cache), 3)
            self.assertEqual(third.qwen_metadata["pipeline_settings"][mass.FIELD]["threshold"], .7)
        with self.assertRaisesRegex(ValueError, "Adaptive query reuse"):
            backend.validate_encoded(fixture.pipeline_args, first, [], self.api, **ENCODING)
        # Empty distributed query shards still return before opening any selection files.
        empty = self.api.EncodedQueries([], np.empty((0, 0), dtype="float32"), {})
        backend.validate_encoded(NS(view_selection_file="missing"), empty, [], self.api, **ENCODING)


if __name__ == "__main__":
    unittest.main()
