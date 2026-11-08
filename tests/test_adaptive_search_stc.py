"""STC dictionaries across standalone search lookup, query encoding and reports."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import pickle
import shutil
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

import test_freeret_dense as fixtures
import test_adaptive_view_shuffle as selections
import test_adaptive_search as legacy
import test_stc_sparse as stc_fixtures
import adaptive_views as views
import adaptive_view_encoding as backend
import adaptive_search_batch as batch
import dense_representation as dense_mode
import search_adaptive_views as search
import stc_sparse as stc


class SearchSTCTests(unittest.TestCase):
    def setUp(self):
        self.temp = fixtures.temporary_directory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tokenizer = stc_fixtures.Tokenizer()
        tokenizer_patch = patch.object(stc, "_tokenizer", return_value=self.tokenizer)
        tokenizer_patch.start()
        self.addCleanup(tokenizer_patch.stop)
        self.embeddings = torch.tensor([[1., 1.], [1., 0.], [.99, .01], [0., 1.], [.01, .99],
                                        [1., 0.], [1., 0.], [1., 1.], [1., 1.], [1., 1.]])

    def corpus(self, name, *, algorithm="agglomerative", mode="ordinary", dataset="RSTPReid",
               split="test", kind=None, query=None, model_dir=None, alternate_mapping=False):
        root = self.root / name
        model_dir = model_dir or root / "checkpoint"
        model_dir.mkdir(parents=True, exist_ok=True)
        if not (model_dir / "config.json").exists():
            (model_dir / "config.json").write_text('{}', encoding="utf-8")
        identity = dict(family="qwen2_5_vl", checkpoint=dict(path=str(model_dir)), dtype="float16")
        opts = selections.options(shuffled=False)
        selection = selections.choose(opts, identity=identity, dataset=dataset)
        selection_path = root / "view_selection.json"
        views.atomic_json(selection_path, selection)
        model = fixtures.model_args(mode != "ordinary", str(model_dir))
        model.use_freeret_prompt_constraints, model.use_freeret_full_prompt = mode == "constraints", mode == "full"
        cache = root / "dictionary"
        options = stc_fixtures.args_at(model_dir, stc_algorithm=algorithm, stc_cache_root=str(cache), stc_dtype="float16")
        stc.copy_options(options, model, dtype="float16")
        request, _ = stc.request(model, tokenizer=self.tokenizer)
        with redirect_stdout(io.StringIO()):
            vocabulary = stc.build_mapping(self.embeddings, self.embeddings, self.tokenizer, model,
                                          stc.cache_path(model, request), request)
        if alternate_mapping:
            # A self-consistent alternative vocabulary must still be rejected
            # when used to tune alpha for a target with another mapping.
            arrays = copy.deepcopy(vocabulary.arrays)
            labels = arrays["cluster_ids"]
            labels[labels < 2] = 1 - labels[labels < 2]
            manifest = dict(vocabulary.manifest, arrays_sha256=stc.arrays_digest(arrays))
            manifest["fingerprint"] = stc.digest({k: v for k, v in manifest.items() if k != "fingerprint"})
            stc._save_arrays(vocabulary.directory / "token_to_cluster.npz", arrays)
            views.atomic_json(vocabulary.directory / "manifest.json", manifest)
            vocabulary = stc.Vocabulary(vocabulary.directory, expected=request)
        data = fixtures.data_args(dataset_name=dataset, dataset_split=split,
                                  dense_output_dir=str(root / "dense"), sparse_output_dir=str(root / "sparse"))
        task = search.TASKS[dataset]
        kind = kind or ("image" if task in {"ir", "tbpr"} else "corpus")
        training = NS(task_type=task, encode_type=kind)
        roots = backend.artifact_dirs(model, data, training, selection)
        expected = backend.metadata(model, data, task, kind, identity, selection)
        for path in roots:
            stc.publish_bundle(path, vocabulary)
            views.atomic_json(path / views.MANIFEST, dict(expected, num_shards=1, selection_file=str(selection_path),
                                                         selection_fingerprint=selection["fingerprint"]))
            dense_mode.write_manifest(path, mode != "ordinary", mode == "constraints", mode == "full")
        with (roots[0] / "corpus_0.pkl").open("wb") as stream:
            pickle.dump((np.array([[1., 0.], [0., 1.]], dtype=np.float32), ["10", "20"]), stream)
        collection = roots[1] / "collection"
        (collection / "index").mkdir(parents=True)
        (collection / "corpus_0.jsonl").write_text('{"id":"10","vector":{"stc_000000":100}}\n', encoding="utf-8")
        (collection / "index/segments_1").write_text("fixture", encoding="utf-8")
        args = search.parser().parse_args(["--passage_reps", str(roots[0]), "--sparse_index", str(roots[1]),
                                          "--adaptive_data_root", str(root / "data"),
                                          "--search_output_root", str(root / "results")])
        args.query_type = query
        return NS(root=root, model=model, identity=identity, options=opts, data=data, training=training,
                  selection=selection, selection_path=selection_path, expected=expected, roots=roots,
                  vocabulary=vocabulary, args=args, cache=cache)

    def test_legacy_root_shards_index_without_changing_stc_or_freeret_fingerprints(self):
        import adaptive_sparse_index_batch as indexing
        from test_adaptive_sparse_layout import flatten, files, lucene
        for algorithm in ("agglomerative", "minibatch_kmeans"):
            for mode in ("constraints", "full"):
                with self.subTest(algorithm=algorithm, mode=mode):
                    fixture = self.corpus("legacy_" + algorithm + mode, algorithm=algorithm, mode=mode)
                    root = fixture.roots[1]
                    flatten(root)
                    before = files(root)
                    self.assertEqual(self.prepare(fixture).index, root / "index")
                    self.assertEqual(files(root), before)
                    env = dict(MODEL_FAMILY="qwen25", DATASET="RSTPReid", SPARSE_OUTPUT_ROOT=str(root),
                               USE_FREERET_DENSE="true", USE_FREERET_PROMPT_CONSTRAINTS=str(mode == "constraints").lower(),
                               USE_FREERET_FULL_PROMPT=str(mode == "full").lower(), DRY_RUN="false")
                    settings = indexing.Settings.from_environment("tbpr", env)
                    with patch("qwen_sparse_index.subprocess.run", side_effect=lucene), \
                            patch.object(stc, "build", side_effect=AssertionError("index must not cluster")), \
                            redirect_stdout(io.StringIO()):
                        self.assertEqual(indexing.run(settings), 0)
                    for name, data in before.items():
                        self.assertEqual((root / name).read_bytes(), data)
                    experiment = self.prepare(fixture)
                    self.assertEqual(experiment.index, root / "collection/index")
                    self.assertEqual(experiment.expected, fixture.expected)

    def prepare(self, fixture, args=None):
        return search.prepare(args or fixture.args, identity_loader=lambda *a, **k: fixture.identity)

    def test_inherit_both_algorithms_freeret_modes_and_dense_sparse_only(self):
        for algorithm in ("agglomerative", "minibatch_kmeans"):
            for mode in ("ordinary", "dense", "constraints", "full"):
                fixture = self.corpus(algorithm + mode, algorithm=algorithm, mode=mode)
                for route in ("both", "dense", "sparse"):
                    with self.subTest(algorithm=algorithm, mode=mode, route=route):
                        args = copy.deepcopy(fixture.args)
                        if route == "dense":
                            args.sparse_index = None
                        elif route == "sparse":
                            args.passage_reps = None
                        experiment = self.prepare(fixture, args)
                        self.assertTrue(experiment.args.use_stc_sparse)
                        self.assertEqual(experiment.args.stc_algorithm, algorithm)
                        self.assertEqual(experiment.args.stc_n_clusters, 2)
                        self.assertEqual(experiment.args.stc_weight_scale, 10000)
                        self.assertEqual(experiment.expected, fixture.expected)
                        source = fixture.roots[0] if route == "dense" else fixture.roots[1]
                        self.assertEqual(Path(experiment.pipeline_args.stc_vocab_path), source / stc.BUNDLE)
                        output = search.output_location(experiment, search.run_configuration(experiment))
                        self.assertIn("_stc_" + algorithm + "_k2_", output.name)
                        if mode == "full":
                            self.assertTrue(output.name.endswith(dense_mode.SUFFIX + dense_mode.FULL_PROMPT_SUFFIX))

    def test_explicit_conflicts_report_the_field_and_modes_cannot_mix(self):
        fixture = self.corpus("conflicts", algorithm="minibatch_kmeans")
        for name, value in (("use_stc_sparse", False), ("stc_algorithm", "agglomerative"),
                            ("stc_n_clusters", 3), ("stc_seed", 17), ("stc_batch_size", 16),
                            ("stc_max_iter", 23), ("stc_weight_scale", 1000)):
            args = copy.deepcopy(fixture.args)
            setattr(args, name, value)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name if name != "use_stc_sparse" else "STC mode mismatch"):
                self.prepare(fixture, args)
        other = self.corpus("other", model_dir=Path(fixture.model.model_name_or_path))
        args = copy.deepcopy(fixture.args)
        args.sparse_index = str(other.roots[1])
        with self.assertRaisesRegex(ValueError, "Dense/sparse corpus configuration"):
            self.prepare(fixture, args)
        plain = legacy.corpus(self.root / "plain")
        plain.args.use_stc_sparse = True
        with self.assertRaisesRegex(ValueError, "STC mode mismatch"):
            self.prepare(plain)

    def test_explicit_bundle_does_not_need_cache_and_corruption_fails(self):
        fixture = self.corpus("portable")
        self.assertTrue(fixture.cache.resolve().is_relative_to(self.root.resolve()))
        shutil.rmtree(fixture.cache)
        forbidden = Mock(side_effect=AssertionError("Search must never rebuild the vocabulary or select views"))
        with patch.object(stc, "build", forbidden), patch.object(stc, "build_mapping", forbidden), patch.object(views, "choose_views", forbidden):
            fixture.args.stc_cache_root = str(self.root / "missing_cache")
            self.assertTrue(self.prepare(fixture).pipeline_args.use_stc_sparse)
        forbidden.assert_not_called()
        (fixture.roots[1] / stc.BUNDLE / "token_to_cluster.npz").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "corrupt"):
            self.prepare(fixture)

    def test_ordinary_hash_and_path_ignore_inactive_cluster_parameters(self):
        fixture = legacy.corpus(self.root / "ordinary")
        original = self.prepare(fixture)
        args = copy.deepcopy(fixture.args)
        args.use_stc_sparse, args.stc_algorithm, args.stc_n_clusters = False, "unused", 2
        explicit = self.prepare(fixture, args)
        one, two = search.run_configuration(original), search.run_configuration(explicit)
        self.assertEqual(one, two)
        self.assertNotIn(stc.FIELD, one["encoding"])
        path = search.output_location(original, one)
        self.assertEqual(path, search.output_location(explicit, two))
        self.assertEqual(path.name, views.digest(one)[:16])

    def test_validation_rebinds_its_bundle_and_rejects_another_mapping(self):
        fixture = self.corpus("target")
        for alternate in (False, True):
            val = self.corpus("validation" + str(alternate), split="val", alternate_mapping=alternate,
                              model_dir=Path(fixture.model.model_name_or_path))
            args = copy.deepcopy(fixture.args)
            args.use_fixed_alpha = False
            args.val_passage_reps, args.val_sparse_index = map(str, val.roots)
            args.val_view_selection_file = str(val.selection_path)
            target = self.prepare(fixture, args)
            if alternate:
                with self.assertRaisesRegex(ValueError, "Validation/target STC"):
                    search.prepare_validation(args, target, identity_loader=lambda *a, **k: fixture.identity)
            else:
                validation = search.prepare_validation(args, target, identity_loader=lambda *a, **k: fixture.identity)
                self.assertEqual(Path(validation.pipeline_args.stc_vocab_path), val.roots[1] / stc.BUNDLE)
                self.assertNotEqual(validation.pipeline_args.stc_vocab_path, target.pipeline_args.stc_vocab_path)

    def environment(self, fixture, group, query):
        return dict(MODEL_FAMILY="qwen25", DATASET=fixture.data.dataset_name, IR_QUERY_TYPE=query,
                    MODEL_PATH=fixture.model.model_name_or_path, DATA_ROOT=str(fixture.root / "data"),
                    DENSE_OUTPUT_ROOT=fixture.data.dense_output_dir, SPARSE_OUTPUT_ROOT=fixture.data.sparse_output_dir,
                    SEARCH_OUTPUT_ROOT=str(fixture.root / "results"), SPARSE_LENGTH="20",
                    VIEW_EXAMPLE_MODE="fixed", VIEW_SELECTION_ROUNDS="2", VIEW_TEXTS_PER_ROUND="2", VIEW_IMAGES_PER_ROUND="1",
                    USE_STC_SPARSE="true", STC_ALGORITHM=fixture.model.stc_algorithm, STC_N_CLUSTERS="2",
                    STC_SEED="42", STC_BATCH_SIZE="8", STC_MAX_ITER="15", STC_WEIGHT_SCALE="10000",
                    STC_CACHE_ROOT=str(fixture.cache), DRY_RUN="true")

    def test_three_task_batches_locate_the_encoder_path_and_forward_every_option(self):
        cases = (("ir", "flickr", "image", "text"), ("ir", "flickr", "text", "image"),
                 ("tbpr", "RSTPReid", "image", "text"), ("fused", "webqa", "corpus", "text"),
                 ("fused", "remuq", "corpus", "composed"), ("fused", "llava", "corpus", "composed"))
        for algorithm in ("agglomerative", "minibatch_kmeans"):
            for group, dataset, kind, query in cases:
                with self.subTest(algorithm=algorithm, dataset=dataset, query=query):
                    fixture = self.corpus(algorithm + dataset + query, algorithm=algorithm, dataset=dataset, kind=kind, query=query)
                    env = self.environment(fixture, group, query)
                    settings = batch.Settings.from_environment(group, env)
                    model, data, training = batch.namespaces(settings, "qwen25", dataset, fixture.training.task_type, kind)
                    location, _, _ = batch.selection_location(settings, model, data, training, fixture.identity)
                    views.atomic_json(location, fixture.selection)
                    with patch.object(batch, "annotation_files", return_value=[]):
                        jobs, missing = batch.prepare_jobs(settings, identity_loader=lambda *a, **k: fixture.identity)
                        manual_env = dict(env, DENSE_CORPUS_DIR=str(fixture.roots[0]), SPARSE_CORPUS_DIR=str(fixture.roots[1]))
                        manual, _ = batch.prepare_jobs(batch.Settings.from_environment(group, manual_env), identity_loader=lambda *a, **k: fixture.identity)
                        self.assertFalse(missing)
                        self.assertEqual(len(jobs), 1)
                        self.assertEqual(jobs[0].configuration, manual[0].configuration)
                        for name in stc.NAMES:
                            self.assertIn("--" + name.lower(), jobs[0].command)
                        self.assertIn(str(fixture.roots[0]), jobs[0].command)
                        self.assertIn(str(fixture.roots[1]), jobs[0].command)
                        self.assertEqual(jobs[0].configuration["encoding"][stc.FIELD], fixture.expected["configuration"][stc.FIELD])
                        before = sorted(str(p) for p in fixture.root.rglob("*"))
                        with patch.object(stc, "build", side_effect=AssertionError("no clustering")), redirect_stdout(io.StringIO()):
                            self.assertEqual(batch.run(settings, identity_loader=lambda *a, **k: fixture.identity,
                                                       runner=Mock(side_effect=AssertionError("dry-run"))), 0)
                        self.assertEqual(before, sorted(str(p) for p in fixture.root.rglob("*")))

    def test_worker_encodes_queries_with_locked_mapping_and_writes_all_recall_reports(self):
        fixture = self.corpus("worker", algorithm="minibatch_kmeans")
        args = copy.deepcopy(fixture.args)
        args.output_dir = str(fixture.root / "search_output")
        experiment = self.prepare(fixture, args)
        output = search.output_location(experiment, search.run_configuration(experiment))
        api = backend.pipeline_module()
        captured, resources = [], []
        class Dense:
            def __init__(self, reps): self.parts = []
            def add(self, reps): self.parts.append(reps)
            def search(self, queries, depth):
                scores = queries @ np.concatenate(self.parts).T
                indices = np.argsort(-scores, axis=1)[:, :depth]
                return np.take_along_axis(scores, indices, axis=1), indices
        class Lucene:
            def __init__(self, *unused): self.closed = False; resources.append(self)
            def set_analyzer(self, analyzer): pass
            def batch_search(self, topics, ids, depth, **unused):
                captured.extend(topics)
                return {qid: [NS(docid="10", score=2.)] for qid in ids}
            def close(self): self.closed = True
        class Analyzer:
            def close(self): pass
        modules = {}
        for module, name, value in (("tevatron.retriever.searcher", "FaissFlatSearcher", Dense),
                                    ("pyserini.search.lucene", "LuceneImpactSearcher", Lucene),
                                    ("pyserini.analysis", "JWhiteSpaceAnalyzer", Analyzer)):
            modules[module] = ModuleType(module)
            setattr(modules[module], name, value)
        dataset = NS(text_id_list=["q0", "q1"], text_dict={"q0": "person", "q1": "another person"},
                     img_id_list=["10", "20"], get_target_from_text=lambda q: "p",
                     get_target_from_image=lambda doc: "p" if doc == "10" else "other")
        def encoder_factory(model, processor, model_args, data, task, selection):
            self.assertEqual(stc.configuration(model_args), fixture.expected["configuration"][stc.FIELD])
            self.assertEqual(Path(model_args.stc_vocab_path), fixture.roots[1] / stc.BUNDLE)
            encoder = NS(model_args=model_args, stc_encoder=stc.SparseEncoder(self.tokenizer, data, model_args),
                         last_prompts=["STC query prompt"])
            raw = torch.zeros(len(selection["selected"]), 10)
            encoder.encode = lambda records: ([raw.clone() for _ in records], torch.tensor([[1., 0.]] * len(records)))
            return encoder
        release = Mock()
        with patch.dict(sys.modules, modules), patch.object(backend, "load_model_identity", return_value=fixture.identity), \
                patch.object(api, "_load_retrieval_runtime", return_value=NS(encoder=NS(config=NS()), processor=NS())), \
                patch.object(api, "_release_retrieval_runtime", release), patch.object(api, "build_dataset", return_value=dataset), \
                patch.object(backend, "configure_processor"), patch.object(backend, "AdaptiveEncoder", side_effect=encoder_factory), \
                patch.object(stc, "build", side_effect=AssertionError("must not cluster")), redirect_stdout(io.StringIO()):
            search.execute(args)
        self.assertTrue(captured)
        self.assertTrue(all(all(token.startswith("stc_") for token in topic.split()) for topic in captured))
        report = json.loads((output / "recall_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(set(report["metrics"]), {"dense", "sparse", "hybrid"})
        self.assertTrue(all("Recall@20" in metrics for metrics in report["metrics"].values()))
        metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["configuration"]["encoding"][stc.FIELD], fixture.expected["configuration"][stc.FIELD])
        self.assertEqual(metadata["query_examples"], ["STC query prompt"])
        self.assertTrue((output / "recall_metrics.csv").is_file())
        self.assertTrue(all(item.closed for item in resources))
        release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
