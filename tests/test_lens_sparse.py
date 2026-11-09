"""Centroid LM-head math, isolated forwarding, immutable artifacts and task launchers."""
from contextlib import redirect_stdout
import copy
import importlib
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import test_freeret_dense as fixtures
import test_stc_sparse as stc_fixtures
import test_adaptive_view_shuffle as selections
import adaptive_view_encoding as backend
import adaptive_views as views
import dense_representation as dense
import lens_sparse as lens
import search_adaptive_views as search

ROOT = Path(__file__).resolve().parents[1]


def options(root, **changes):
    values = {key.lower(): (value if key in {"LENS_ALGORITHM", "LENS_CACHE_ROOT"} else int(value))
              for key, value in lens.DEFAULTS.items() if key != "USE_LENS_SPARSE"}
    values.update(use_lens_sparse=True, lens_n_clusters=2, lens_dtype="float32", lens_max_iter=20,
                  model_name_or_path=str(root), lens_cache_root=str(root / "heads"), lora=False)
    values.update(changes)
    return NS(**values)


class LensTests(unittest.TestCase):
    def setUp(self):
        self.temp = fixtures.temporary_directory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "config.json").write_text('{}', encoding="utf-8")
        self.tokenizer = stc_fixtures.Tokenizer()
        token_patch = patch.object(lens, "_identity", return_value=(self.tokenizer, "qwen2_5_vl"))
        token_patch.start()
        self.addCleanup(token_patch.stop)
        self.args = options(self.root)
        self.weights = torch.tensor([[99., 99.], [1., 0.], [3., 0.], [0., 10.], [0., 12.],
                                     [5., 0.], [7., 0.], [99., 99.], [999., 999.], [99., 99.]])

    def build(self, args=None, weights=None, bias=None):
        args = args or self.args
        config, _ = lens.request(args)
        with redirect_stdout(io.StringIO()):
            return lens.build_head(self.weights if weights is None else weights, self.tokenizer, args,
                                   lens.cache_path(args, config), config, bias=bias)

    def test_raw_output_centroids_and_bias_equal_mean_member_logits_for_both_algorithms(self):
        hidden = torch.tensor([[2., -1.], [-3., 4.]])
        bias = torch.arange(10).float() / 10
        for algorithm in ("kmeans", "minibatch_kmeans"):
            args = options(self.root, lens_algorithm=algorithm)
            head = self.build(args, bias=bias)
            self.assertEqual(head.manifest["cluster_count"], 2)
            self.assertEqual(set(head.arrays["token_ids"]), {1, 2, 3, 4, 5, 6})
            # Numeric rows are clustered normally; input embeddings are never provided.
            layer = lens.CentroidLayer(head, nn.Linear(2, 10, bias=True))
            original = F.linear(hidden, self.weights, bias)
            for cluster in range(2):
                ids = head.arrays["token_ids"][head.arrays["cluster_ids"] == cluster]
                expected = self.weights[ids.tolist()].mean(0)
                self.assertTrue(torch.equal(layer.weight[cluster], expected))
                self.assertTrue(torch.allclose(layer(hidden)[:, cluster], original[:, ids.tolist()].mean(1)))
            self.assertGreater(float(layer.weight.norm(dim=1).max()), 1.)  # no L2 normalization.

    def test_zero_output_rows_excluded_and_nonfinite_or_impossible_cluster_count_fails(self):
        weights = self.weights.clone()
        weights[1] = 0
        head = self.build(weights=weights)
        self.assertNotIn(1, head.arrays["token_ids"])
        self.assertEqual(head.manifest["excluded_zero_token_ids"], [1])
        bad = weights.clone()
        bad[2, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite LENS"):
            self.build(options(self.root, lens_seed=7), weights=bad)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.build(options(self.root, lens_n_clusters=100))
        identical = torch.ones_like(weights)
        with self.assertRaisesRegex(ValueError, "nonempty clusters"):
            self.build(options(self.root, lens_seed=8), weights=identical)

    def test_cluster_topk_integer_merge_and_floor_mean(self):
        self.build()
        for merge, mean, expected in (("sum", False, 61), ("max", False, 31),
                                       ("replace", False, 31), ("sum", True, 30)):
            data = fixtures.data_args(sparse_length=1, sparse_value_type=merge, sparse_value_mean=mean)
            encoder = lens.SparseEncoder(self.tokenizer, data, self.args, family="qwen2_5_vl")
            self.assertEqual(encoder.vector(torch.tensor([[.304, .1], [.306, .1]])), {"lens_000000": expected})
            self.assertEqual(encoder.vector(torch.zeros(2, 2)), {})
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                encoder.vector(torch.full((1, 2), float("nan")))

    def test_context_restores_head_input_weights_rng_and_exception(self):
        self.build()
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(10, 2)
                self.head = nn.Linear(2, 10)
            def get_output_embeddings(self): return self.head
            def set_output_embeddings(self, value): self.head = value
            def get_input_embeddings(self): return self.embedding
            def set_input_embeddings(self, value): self.embedding = value
        model = Model()
        original, weight = model.head, model.embedding.weight
        encoder = lens.SparseEncoder(self.tokenizer, fixtures.data_args(), self.args, family="qwen2_5_vl")
        rng = torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            with encoder.activate(model):
                self.assertEqual(model.head(torch.ones(1, 2)).shape, (1, 2))
                self.assertIs(model.embedding.weight, weight)
                raise RuntimeError("forward failed")
        self.assertIs(model.head, original)
        self.assertIs(model.embedding.weight, weight)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_cache_reuse_bundles_checksum_filtering_and_no_disabled_fingerprint_changes(self):
        head = self.build()
        config = lens.configuration(self.args)
        destination = self.root / "corpus"
        lens.publish_bundle(destination, head)
        lens.validate_bundle(destination, config)
        for key, value in (("lens_seed", 17), ("lens_n_clusters", 3), ("lens_is_filtered", True)):
            changed = copy.copy(self.args)
            setattr(changed, key, value)
            self.assertNotEqual(lens.cache_path(changed), lens.cache_path(self.args))
        with patch.object(lens, "_build_new", side_effect=AssertionError("must reuse")), redirect_stdout(io.StringIO()):
            self.assertEqual(lens.build(self.args).fingerprint, head.fingerprint)
        dry = options(self.root, lens_seed=19, dry_run=True)
        with redirect_stdout(io.StringIO()):
            self.assertIsNone(lens.build(dry))
        self.assertFalse(lens.cache_path(dry).exists())
        direct = NS(use_lens_sparse=None)
        lens.bind_search(direct, destination, config)
        direct.model_name_or_path, direct.lens_dtype = str(self.root), "float32"
        shutil.rmtree(self.root / "heads")
        self.assertEqual(lens.load(direct).fingerprint, head.fingerprint)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            lens.validate_bundle(destination, dict(config, head_fingerprint="wrong"))
        archive = destination / lens.BUNDLE / "head.npz"
        arrays = copy.deepcopy(head.arrays)
        arrays["centroids"][0, 0] += 1
        lens._save_arrays(archive, arrays)
        with self.assertRaisesRegex(ValueError, "checksum"):
            lens.validate_bundle(destination, config)
        off = NS(use_lens_sparse=False)
        target = NS()
        lens.copy_options(off, target)
        self.assertEqual(vars(target), {})
        self.assertIsNone(lens.configuration(off))
        self.assertEqual(lens.mark_path("original", None), Path("original"))

    def test_conflicts_and_unsupported_entries_rejected_before_work(self):
        import adaptive_encode_batch as encode_batch
        import adaptive_views_topk_experiment as topk
        for changes in (dict(use_stc_sparse=True), dict(view_overlap_selection=True)):
            args = copy.copy(self.args)
            vars(args).update(changes)
            with self.assertRaises(ValueError): lens.validate(args)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            lens.environment_controls(dict(USE_LENS_SPARSE="true", USE_STC_SPARSE="true"))
        with self.assertRaisesRegex(ValueError, "overlap"):
            encode_batch.Settings.from_environment("tbpr", dict(USE_LENS_SPARSE="true"), encoder="overlap")
        with self.assertRaisesRegex(ValueError, "Top-K"):
            topk.build_jobs(dict(USE_LENS_SPARSE="true"))


class ModelTests(unittest.TestCase):
    def compare(self, model, processor, records, family, adapter="adaptive"):
        if adapter != "adaptive": return
        if family.startswith("llava"):
            processor.chat_template = ("{% for m in messages %}[INST]{% for c in m['content'] %}{% if c['type']=='image' %}<image>\n{% else %}{{ c['text'] }}{% endif %}{% endfor %}[/INST]{% endfor %}"
                if "mistral" in family else "{% for m in messages %}<|start_header_id|>user<|end_header_id|>\n\n{% for c in m['content'] %}{% if c['type']=='image' %}<image>\n{% else %}{{ c['text'] }}{% endif %}{% endfor %}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{% endfor %}")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / family
            root.mkdir()
            model.save_pretrained(root)
            processor.tokenizer.save_pretrained(root)
            args = options(root, dry_run=False)
            with redirect_stdout(io.StringIO()): head = lens.build(args)
            input_module, input_weight, old_head = model.get_input_embeddings(), model.get_input_embeddings().weight, model.get_output_embeddings()
            for mode in ("normal", "constraints", "full"):
                ordinary = fixtures.model_args(mode != "normal", str(root))
                ordinary.use_freeret_prompt_constraints, ordinary.use_freeret_full_prompt = mode == "constraints", mode == "full"
                data = fixtures.data_args(print_sparse=False, sparse_length=1)
                selected = dict(selected=views.view_pool()[:2])
                original = backend.AdaptiveEncoder(model, processor, ordinary, data, "ir", selected)
                replacement = copy.copy(ordinary)
                lens.copy_options(args, replacement, dtype="float32", data=data)
                encoder = backend.AdaptiveEncoder(model, processor, replacement, data, "ir", selected)
                calls = []
                def capture(module, positional, keywords):
                    calls.append({key: value.detach().cpu().clone() for key, value in keywords.items()
                                  if torch.is_tensor(value) and key in {"attention_mask", "position_ids", "input_ids"}})
                hook = model.register_forward_pre_hook(capture, with_kwargs=True)
                try:
                    values, dense_before = original.encode(records)
                    previous = calls[:]
                    calls.clear()
                    new_values, dense_after = encoder.encode(records)
                finally:
                    hook.remove()
                self.assertEqual(len(previous), len(calls))
                for before, after in zip(previous, calls):
                    self.assertIn("attention_mask", before)
                    self.assertEqual(set(before), set(after))
                    for key in before:
                        self.assertTrue(torch.equal(before[key], after[key]), (family, mode, key))
                self.assertTrue(torch.equal(dense_before, dense_after), (family, mode))
                self.assertEqual(original.last_prompts, encoder.last_prompts)
                self.assertIs(model.get_input_embeddings(), input_module)
                self.assertIs(model.get_input_embeddings().weight, input_weight)
                self.assertIs(model.get_output_embeddings(), old_head)
                for before, after in zip(values, new_values):
                    self.assertEqual(before.shape[-1], old_head.weight.shape[0])
                    self.assertEqual(after.shape[-1], 2)
                vectors = backend.sparse_vectors(encoder, records, new_values)
                self.assertTrue(all(all(key.startswith("lens_") for key in row) for row in vectors))
                self.assertEqual(encoder.lens_encoder.head.fingerprint, head.fingerprint)

    def test_five_model_families_keep_dense_prompts_masks_input_embeddings_and_original_head(self):
        case = fixtures.RealModelTests()
        with patch.object(fixtures.RealModelTests, "compare", side_effect=self.compare):
            case.test_native_and_adaptive_qwen_three_families_with_text_image_and_fused_inputs()
            case.test_llava_llama_and_mistral_with_native_image_processor()


class IntegrationTests(unittest.TestCase):
    setUp = LensTests.setUp
    build = LensTests.build

    def corpus(self, dataset="RSTPReid", mode="normal"):
        directory = self.root / (dataset + mode)
        identity = dict(family="qwen2_5_vl", checkpoint=dict(path=str(self.root)), dtype="float16")
        selection = selections.choose(selections.options(shuffled=False), identity=identity, dataset=dataset)
        selected_file = directory / "view_selection.json"
        views.atomic_json(selected_file, selection)
        model = fixtures.model_args(mode != "normal", str(self.root))
        model.use_freeret_prompt_constraints, model.use_freeret_full_prompt = mode == "constraints", mode == "full"
        opts = options(self.root, lens_dtype="float16")
        head = self.build(opts)
        data = fixtures.data_args(dataset_name=dataset, dense_output_dir=str(directory / "dense"),
                                  sparse_output_dir=str(directory / "sparse"))
        lens.copy_options(opts, model, dtype="float16", data=data)
        task = search.TASKS[dataset]
        kind = "image" if task in {"ir", "tbpr"} else "corpus"
        training = NS(task_type=task, encode_type=kind)
        paths = backend.artifact_dirs(model, data, training, selection)
        expected = backend.metadata(model, data, task, kind, identity, selection)
        for path in paths:
            lens.publish_bundle(path, head)
            views.atomic_json(path / views.MANIFEST, dict(expected, num_shards=1,
                selection_file=str(selected_file), selection_fingerprint=selection["fingerprint"]))
            dense.write_manifest(path, mode != "normal", mode == "constraints", mode == "full")
        with (paths[0] / "corpus_0.pkl").open("wb") as stream:
            pickle.dump((np.array([[1., 0.], [0., 1.]], dtype=np.float32), ["10", "20"]), stream)
        (paths[1] / "collection/index").mkdir(parents=True)
        (paths[1] / "collection/index/segments_1").write_text("fixture", encoding="utf-8")
        (paths[1] / "collection/corpus_0.jsonl").write_text('{"id":"10","vector":{"lens_000000":100}}\n', encoding="utf-8")
        args = search.parser().parse_args(["--passage_reps", str(paths[0]), "--sparse_index", str(paths[1]),
            "--adaptive_data_root", str(directory / "data"), "--search_output_root", str(directory / "results")])
        return NS(paths=paths, model=model, data=data, identity=identity, selection=selection,
                  selection_file=selected_file, expected=expected, args=args, head=head)

    def test_index_and_pinned_search_and_pipeline_for_all_tasks_and_prompt_modes(self):
        import adaptive_sparse_index_batch as indexing
        api = backend.pipeline_module()
        for dataset in ("flickr", "RSTPReid", "webqa", "remuq", "llava"):
            for mode in ("normal", "constraints", "full"):
                with self.subTest(dataset=dataset, mode=mode):
                    fixture = self.corpus(dataset, mode)
                    root = fixture.paths[1]
                    manifest = json.loads((root / views.MANIFEST).read_text(encoding="utf-8"))
                    indexing.check_artifact(root, manifest, fixture.expected["configuration"])
                    # Search must work with only the immutable corpus copies.
                    shutil.rmtree(self.root / "heads")
                    experiment = search.prepare(fixture.args, identity_loader=lambda *a, **k: fixture.identity)
                    self.assertTrue(experiment.args.use_lens_sparse)
                    self.assertEqual(experiment.args.lens_n_clusters, 2)
                    self.assertEqual(experiment.expected, fixture.expected)
                    args = experiment.pipeline_args
                    args.stage1_method, args.stage2_method = "sparse_concat", "dense"
                    args.stage1_sparse_index, args.stage2_dense_index = str(root / "collection/index"), str(fixture.paths[0])
                    args.stage2_sparse_index = None
                    with patch.object(backend, "load_model_identity", return_value=fixture.identity):
                        selected = backend.preflight(args, api, mark_output=False)
                    self.assertEqual(selected["fingerprint"], fixture.selection["fingerprint"])
                    model, data = api._retrieval_namespaces(args, eol_type="disassembleeol_separate",
                        calculate_type="concat", sparse_manual=False)
                    self.assertEqual(lens.configuration(model), fixture.expected["configuration"][lens.FIELD])
                    self.assertEqual(Path(model.lens_head_path), root / lens.BUNDLE)
                    output = search.output_location(experiment, search.run_configuration(experiment))
                    self.assertIn("_lens_kmeans_k2_", output.name)
                    bad = copy.deepcopy(fixture.args)
                    bad.use_lens_sparse = False
                    with self.assertRaisesRegex(ValueError, "LENS mode mismatch"):
                        search.prepare(bad, identity_loader=lambda *a, **k: fixture.identity)

    def test_exact_batch_location_and_independent_modes_for_three_task_groups(self):
        import adaptive_retrieval_batch as retrieval
        import adaptive_sparse_index_batch as indexing
        import adaptive_search_batch as searching
        for group, dataset in (("ir", "flickr"), ("tbpr", "RSTPReid"), ("fused", "webqa")):
            with self.subTest(group=group):
                fixture = self.corpus(dataset)
                env = dict(MODEL_FAMILY="qwen25", DATASET=dataset, MODEL_PATH=str(self.root),
                    DATA_ROOT=str(self.root / "data"), DENSE_OUTPUT_ROOT=fixture.data.dense_output_dir,
                    SPARSE_OUTPUT_ROOT=fixture.data.sparse_output_dir,
                    USE_LENS_SPARSE="true", LENS_N_CLUSTERS="2", LENS_MAX_ITER="20", LENS_CACHE_ROOT=str(self.root / "heads"),
                    LLAVA_IMAGE_MODE="anyres", TBPR_TYPE="origin_type", SPARSE_LENGTH="20", DTYPE="float16",
                    USE_FREERET_DENSE="false", USE_FREERET_PROMPT_CONSTRAINTS="false", USE_FREERET_FULL_PROMPT="false",
                    IR_QUERY_TYPE="text", ENCODE_TYPE="image" if group != "fused" else "corpus", DRY_RUN="true",
                    VIEW_EXAMPLE_MODE="fixed", VIEW_SELECTION_ROUNDS="2", VIEW_TEXTS_PER_ROUND="2", VIEW_IMAGES_PER_ROUND="1")
                settings = retrieval.Settings.from_environment(group, env)
                training = NS(task_type=search.TASKS[dataset], encode_type="image" if group != "fused" else "corpus")
                cached, _, _ = retrieval.selection_location(settings, fixture.model, fixture.data, training, fixture.identity)
                views.atomic_json(cached, fixture.selection)
                with patch.object(retrieval, "annotation_files", return_value=[]), patch("llava_resolution.validate_checkpoint"), redirect_stdout(io.StringIO()):
                    jobs, missing = retrieval.prepare_jobs(settings, identity_loader=lambda *a, **k: fixture.identity)
                self.assertFalse(missing)
                self.assertEqual(len(jobs), 1)
                self.assertEqual(jobs[0].variables["USE_LENS_SPARSE"], "true")
                self.assertIn("_lens_kmeans_k2_", str(jobs[0].output))
                with redirect_stdout(io.StringIO()):
                    jobs, missing = indexing.discover_exact(indexing.Settings.from_environment(group, env, exact_encoding=True),
                        identity_loader=lambda *a, **k: fixture.identity)
                self.assertFalse(missing)
                self.assertEqual([job.root for job in jobs], [fixture.paths[1].resolve()])
                with patch.object(retrieval, "annotation_files", return_value=[]), patch.object(searching, "annotation_files", return_value=[]), redirect_stdout(io.StringIO()):
                    jobs, missing = searching.prepare_jobs(searching.Settings.from_environment(group, env),
                        identity_loader=lambda *a, **k: fixture.identity)
                self.assertFalse(missing)
                self.assertEqual(len(jobs), 1)

    def test_index_publication_and_query_worker_use_the_locked_centroid_head(self):
        import adaptive_sparse_index_batch as indexing
        from test_adaptive_sparse_layout import lucene
        fixture = self.corpus()
        root = fixture.paths[1]
        shutil.rmtree(root / "collection/index")
        env = dict(MODEL_FAMILY="qwen25", DATASET="RSTPReid", SPARSE_OUTPUT_ROOT=str(root), DRY_RUN="false")
        settings = indexing.Settings.from_environment("tbpr", env)
        with patch("qwen_sparse_index.subprocess.run", side_effect=lucene), \
                patch.object(lens, "build", side_effect=AssertionError("index must not cluster")), redirect_stdout(io.StringIO()):
            self.assertEqual(indexing.run(settings), 0)
        self.assertTrue((root / "collection/index/segments_1").is_file())
        args = copy.deepcopy(fixture.args)
        experiment = search.prepare(args, identity_loader=lambda *a, **k: fixture.identity)
        output = search.output_location(experiment, search.run_configuration(experiment))
        api, captured, resources = backend.pipeline_module(), [], []
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
            self.assertEqual(lens.configuration(model_args), fixture.expected["configuration"][lens.FIELD])
            self.assertEqual(Path(model_args.lens_head_path), root / lens.BUNDLE)
            encoder = NS(model_args=model_args,
                lens_encoder=lens.SparseEncoder(self.tokenizer, data, model_args, family="qwen2_5_vl"),
                last_prompts=["LENS query prompt"])
            raw = torch.ones(len(selection["selected"]), 2)
            encoder.encode = lambda records: ([raw.clone() for _ in records], torch.tensor([[1., 0.]] * len(records)))
            return encoder
        release = Mock()
        with patch.dict(sys.modules, modules), patch.object(backend, "load_model_identity", return_value=fixture.identity), \
                patch.object(api, "_load_retrieval_runtime", return_value=NS(encoder=NS(config=NS()), processor=NS())), \
                patch.object(api, "_release_retrieval_runtime", release), patch.object(api, "build_dataset", return_value=dataset), \
                patch.object(backend, "configure_processor"), patch.object(backend, "AdaptiveEncoder", side_effect=encoder_factory), \
                patch.object(lens, "build", side_effect=AssertionError("search must not cluster")), redirect_stdout(io.StringIO()):
            search.execute(args)
        self.assertTrue(captured)
        self.assertTrue(all(all(token.startswith("lens_") for token in topic.split()) for topic in captured))
        report = json.loads((output / "recall_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(set(report["metrics"]), {"dense", "sparse", "hybrid"})
        self.assertTrue(all("Recall@20" in metrics for metrics in report["metrics"].values()))
        self.assertTrue((output / "recall_metrics.csv").is_file())
        metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["configuration"]["encoding"][lens.FIELD], fixture.expected["configuration"][lens.FIELD])
        self.assertTrue(all(item.closed for item in resources))
        release.assert_called_once()


class ShellTests(unittest.TestCase):
    def test_precompute_precedes_deepspeed_and_mutually_exclusive_modes_fail(self):
        bash = "D:/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
        if not bash or not Path(bash).is_file(): self.skipTest("Bash unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture, log = root / "capture.sh", root / "calls.txt"
            capture.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" >> "$CALL_LOG"\n', encoding="utf-8")
            env = dict(os.environ, USE_LENS_SPARSE="true", USE_STC_SPARSE="false", LENS_N_CLUSTERS="2",
                MODEL_FAMILY="qwen25", MODEL_PATH="fixture", DATASET="RSTPReid", ENCODE_TYPE="image",
                OUTPUT_DIR="fixture", SPARSE_OUTPUT_ROOT=root.as_posix(), LENS_CACHE_ROOT="", CALL_LOG=log.as_posix(),
                PYTHON_BIN=capture.as_posix(), DEEPSPEED_BIN=capture.as_posix(), DRY_RUN="false")
            script = (ROOT / "scripts/encode_adaptive_views_tbpr.sh").as_posix()
            result = subprocess.run([bash, script, "--run-one"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = log.read_text(encoding="utf-8").splitlines()
            self.assertLess(calls.index("src/lens_sparse.py"), calls.index("src/encode_adaptive_views.py"))
            self.assertNotIn("src/stc_sparse.py", calls)
            self.assertEqual(calls.count((root / "lens_heads").as_posix()), 2)
            result = subprocess.run([bash, script], cwd=ROOT, env=dict(env, USE_STC_SPARSE="true"), capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("mutually exclusive", result.stderr)


if __name__ == "__main__":
    unittest.main()
