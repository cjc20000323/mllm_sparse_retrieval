"""Standalone adaptive full-corpus search: locked views, directions, ranking and launchers."""
import copy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS, ModuleType
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from test_freeret_dense import ROOT, temporary_directory, data_args, model_args
import test_adaptive_view_shuffle as fixtures
import adaptive_views as views
import adaptive_view_encoding as backend
import adaptive_view_probability_mass as mass
import dense_representation as dense_mode
import adaptive_search_batch as batch
import search_adaptive_views as search


def corpus(directory, *, dataset="RSTPReid", family="qwen2_5_vl", hierarchical=False,
           shuffled=False, pmi=False, threshold=None, mode="ordinary", split="test",
           sparse_length=20, tbpr_type="origin_type"):
    root = Path(directory)
    model = root / "checkpoint"
    model.mkdir(parents=True, exist_ok=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    identity = dict(family=family, checkpoint=dict(path=str(model)), dtype="float16")
    opts = fixtures.options(shuffled=shuffled, hierarchical=hierarchical, pmi=pmi)
    selection = fixtures.choose(opts, identity=identity, dataset=dataset)
    if threshold is not None:
        selection = mass.derive(selection, threshold)
    path = root / ("selection_" + split + ".json")
    views.atomic_json(path, selection)
    model_config = model_args(mode != "ordinary", str(model))
    model_config.use_freeret_prompt_constraints = mode == "constraints"
    model_config.use_freeret_full_prompt = mode == "full"
    data = data_args(dataset_name=dataset, dataset_split=split,
        dense_output_dir=str(root / "dense"), sparse_output_dir=str(root / "sparse"),
        sparse_length=sparse_length, text_sparse_length=sparse_length, image_sparse_length=sparse_length,
        tbpr_type=tbpr_type)
    task = search.TASKS[dataset]
    kind = "image" if task in {"ir", "tbpr"} else "corpus"
    training = NS(task_type=task, encode_type=kind)
    roots = backend.artifact_dirs(model_config, data, training, selection)
    expected = backend.metadata(model_config, data, task, kind, identity, selection)
    for artifact in roots:
        views.atomic_json(artifact / views.MANIFEST, dict(expected, num_shards=1,
            selection_file=str(path), selection_fingerprint=selection["fingerprint"]))
        dense_mode.write_manifest(artifact, mode != "ordinary", mode == "constraints", mode == "full")
    with (roots[0] / "corpus_0.pkl").open("wb") as stream:
        pickle.dump((np.array([[1, 0], [0, 1], [.1, .1]], dtype=np.float32), ["10", "20", "30"]), stream)
    collection = roots[1] / "collection"
    collection.mkdir(exist_ok=True)
    (collection / "corpus_0.jsonl").write_text('{"id":"10","vector":{"x":1}}\n', encoding="utf-8")
    (collection / "index").mkdir()
    (collection / "index/segments_1").write_text("fixture", encoding="utf-8")
    args = search.parser().parse_args(["--passage_reps", str(roots[0]), "--sparse_index", str(roots[1]),
        "--adaptive_data_root", str(root / "data"), "--search_output_root", str(root / "results")])
    return NS(root=root, identity=identity, selection=selection, selection_path=path, args=args,
        roots=roots, expected=expected, options=opts, data=data, model=model_config, task=task)


class PreflightTests(unittest.TestCase):
    def test_saved_modes_and_views_are_inherited_without_selection(self):
        with temporary_directory() as directory:
            for family in batch.FAMILIES.values():
                for mode in ("ordinary", "dense", "constraints", "full"):
                    with self.subTest(family=family, mode=mode):
                        fixture = corpus(Path(directory) / (family + mode), family=family, hierarchical=True,
                            shuffled=True, pmi=True, threshold=.85, mode=mode)
                        before = fixture.selection_path.read_bytes()
                        with patch.object(views, "choose_views", side_effect=AssertionError("must reuse selection")):
                            ex = search.prepare(fixture.args, identity_loader=lambda *a, **k: fixture.identity)
                        self.assertEqual(ex.selection, fixture.selection)
                        self.assertEqual(fixture.selection_path.read_bytes(), before)
                        self.assertEqual(ex.args.sparse_length, 20)
                        self.assertEqual(ex.args.use_freeret_dense, mode != "ordinary")
                        self.assertEqual(ex.index, fixture.roots[1] / "collection/index")
                        self.assertEqual(ex.args.view_selection_file, str(fixture.selection_path))

    def test_directions_include_composed_queries(self):
        api = backend.pipeline_module()
        dataset = NS(query_id_list=["q"], id2query={"q": dict(image="image.png", text="what is shown?")})
        for name, task in search.TASKS.items():
            query = "composed" if task == "it2t" else "text"
            self.assertTrue(search.direction(task, query))
            if task == "it2t":
                a = NS(task_type=task, data_root=".", use_freeret_prompt_constraints=False, use_freeret_full_prompt=False)
                records = api._qwen_query_records(a, api._query_records(a, dataset))
                self.assertEqual(records[0]["image"], "image.png")
                self.assertEqual(records[0]["text"], "what is shown?")
                self.assertEqual(search.direction(task, "image"), "image_text_to_text")
        self.assertEqual(search.direction("ir", "image"), "image_to_text")
        with self.assertRaises(ValueError):
            search.direction("tbpr", "image")

    def test_bad_artifacts_and_explicit_mode_conflicts_fail(self):
        with temporary_directory() as directory:
            fixture = corpus(directory, hierarchical=True, threshold=.8)
            loader = lambda *a, **k: fixture.identity
            a = copy.deepcopy(fixture.args)
            a.view_probability_mass = False
            with self.assertRaisesRegex(ValueError, "mode differs"):
                search.prepare(a, identity_loader=loader)
            a = copy.deepcopy(fixture.args)
            a.view_probability_mass_threshold = .7
            with self.assertRaisesRegex(ValueError, "threshold differs"):
                search.prepare(a, identity_loader=loader)
            a = copy.deepcopy(fixture.args)
            a.use_freeret_dense = True
            with self.assertRaises(ValueError):
                search.prepare(a, identity_loader=loader)
            (fixture.roots[1] / "collection/index/segments_1").unlink()
            with self.assertRaisesRegex(ValueError, "segments"):
                search.prepare(fixture.args, identity_loader=loader)

    def test_overlap_legacy_and_mismatched_selection_are_rejected(self):
        from test_adaptive_view_probability_mass import legacy_selection
        with temporary_directory() as directory:
            fixture = corpus(directory)
            saved = json.loads((fixture.roots[1] / views.MANIFEST).read_text(encoding="utf-8"))
            corrupt = copy.deepcopy(saved)
            corrupt["configuration"]["overlap_configuration"] = {"threshold": .1}
            corrupt["fingerprint"] = views.digest(corrupt["configuration"])
            views.atomic_json(fixture.roots[1] / views.MANIFEST, corrupt)
            with self.assertRaisesRegex(ValueError, "Overlap"):
                search.prepare(fixture.args, identity_loader=lambda *a, **k: fixture.identity)
            views.atomic_json(fixture.roots[1] / views.MANIFEST, saved)
            old = legacy_selection(fixture.selection)
            views.atomic_json(fixture.selection_path, old)
            with self.assertRaisesRegex(ValueError, "v1.*v2"):
                search.prepare(fixture.args, identity_loader=lambda *a, **k: fixture.identity)

    def test_dry_run_is_read_only_and_does_not_load_weights(self):
        with temporary_directory() as directory:
            fixture = corpus(directory)
            before = sorted(str(p) for p in fixture.root.rglob("*"))
            fixture.args.dry_run = True
            api = backend.pipeline_module()
            with patch.object(backend, "load_model_identity", return_value=fixture.identity), \
                    patch.object(api, "_load_retrieval_runtime", side_effect=AssertionError("no weights")), \
                    redirect_stdout(io.StringIO()) as output:
                search.execute(fixture.args)
            self.assertEqual(before, sorted(str(p) for p in fixture.root.rglob("*")))
            self.assertIn('"selected_views"', output.getvalue())

    def test_validation_uses_its_own_locked_views(self):
        with temporary_directory() as directory:
            target = corpus(Path(directory) / "target")
            validation = corpus(Path(directory) / "validation", hierarchical=True, pmi=True, split="val")
            # Both splits must use exactly the same checkpoint identity.
            for root in validation.roots:
                value = json.loads((root / views.MANIFEST).read_text(encoding="utf-8"))
                value["configuration"]["model"] = target.identity
                selection = copy.deepcopy(validation.selection)
                selection["model"] = target.identity
                selection["fingerprint"] = views.digest({k: v for k, v in selection.items() if k != "fingerprint"})
                value["selection_fingerprint"] = selection["fingerprint"]
                value["fingerprint"] = views.digest(value["configuration"])
                views.atomic_json(root / views.MANIFEST, value)
            views.atomic_json(validation.selection_path, selection)
            target.args.use_fixed_alpha = False
            target.args.val_passage_reps, target.args.val_sparse_index = map(str, validation.roots)
            ex = search.prepare(target.args, identity_loader=lambda *a, **k: target.identity)
            val = search.prepare_validation(target.args, ex, identity_loader=lambda *a, **k: target.identity)
            self.assertNotEqual(val.selection["selected"], ex.selection["selected"])
            self.assertEqual(val.args.dataset_split, "val")


class RankingTests(unittest.TestCase):
    def test_worker_writes_full_search_reports_and_releases_resources(self):
        api = backend.pipeline_module()
        resources = []
        class Dense:
            def __init__(self, reps):
                self.parts = []
            def add(self, reps):
                self.parts.append(reps)
            def search(self, query, depth):
                scores = query @ np.concatenate(self.parts).T
                order = np.argsort(-scores, axis=1)[:, :depth]
                return np.take_along_axis(scores, order, axis=1), order
        class Lucene:
            def __init__(self, path, ignored):
                self.path, self.closed = path, False
                resources.append(self)
            def set_analyzer(self, analyzer):
                pass
            def batch_search(self, topics, ids, depth, **kwargs):
                return {qid: [NS(docid="20", score=3), NS(docid="10", score=1)] for qid in ids}
            def close(self):
                self.closed = True
        class Analyzer:
            def close(self):
                pass
        modules = {}
        for name, symbol, value in (("tevatron.retriever.searcher", "FaissFlatSearcher", Dense),
                ("pyserini.search.lucene", "LuceneImpactSearcher", Lucene),
                ("pyserini.analysis", "JWhiteSpaceAnalyzer", Analyzer)):
            modules[name] = ModuleType(name)
            setattr(modules[name], symbol, value)
        dataset = NS(text_id_list=["q0", "q1"], text_dict={"q0": "person", "q1": "another person"},
            img_id_list=["10", "20", "30"], get_target_from_text=lambda q: "p",
            get_target_from_image=lambda doc: "p" if doc == "10" else "other")
        selections = []
        def encoder_factory(model, processor, model_args, data, task, selection):
            selections.append(selection)
            return NS(last_prompts=["actual prompt fixture"], encode=lambda records:
                (torch.zeros(len(records), len(selection["selected"]), 10), torch.tensor([[1., 0.]] * len(records))))
        with temporary_directory() as directory:
            fixture = corpus(directory)
            for method in ("fixed", "automatic", "dense", "sparse"):
                a = copy.deepcopy(fixture.args)
                a.output_dir = str(fixture.root / "outputs" / method)
                a.use_fixed_alpha = method != "automatic"
                if method == "dense":
                    a.sparse_index = None
                if method == "sparse":
                    a.passage_reps = None
                release = Mock()
                with patch.dict(sys.modules, modules), patch.object(backend, "load_model_identity", return_value=fixture.identity), \
                        patch.object(api, "_load_retrieval_runtime", return_value=NS(encoder=NS(config=NS()), processor=NS())), \
                        patch.object(api, "_release_retrieval_runtime", release), patch.object(api, "build_dataset", return_value=dataset), \
                        patch.object(backend, "configure_processor"), patch.object(backend, "AdaptiveEncoder", side_effect=encoder_factory), \
                        patch.object(backend, "sparse_vectors", side_effect=lambda encoder, records, logits: [{"x": 2} for _ in records]), \
                        redirect_stdout(io.StringIO()):
                    search.execute(a)
                output = Path(a.output_dir)
                report = json.loads((output / "recall_metrics.json").read_text(encoding="utf-8"))
                metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
                expected = {method} if method in {"dense", "sparse"} else {"dense", "sparse", "hybrid"}
                self.assertEqual(set(report["metrics"]), expected)
                self.assertTrue(all("Recall@20" in values for values in report["metrics"].values()))
                self.assertEqual(metadata["status"], "complete")
                self.assertEqual(metadata["query_examples"], ["actual prompt fixture"])
                self.assertEqual(report["alpha_selection_source"], "target" if method == "automatic" else "fixed")
                for name in expected:
                    self.assertTrue((output / f"{name}_run.trec").is_file())
                if method == "automatic":
                    self.assertEqual(len((output / "alpha_sweep.csv").read_text(encoding="utf-8-sig").splitlines()), 10)
                release.assert_called_once()
            self.assertTrue(all(resource.closed for resource in resources))
            self.assertTrue(all(value == fixture.selection for value in selections))

    def test_full_corpus_dense_is_not_restricted_to_sparse_candidates(self):
        from qwen_vl_pipeline import _run
        index = search.CorpusSearch.__new__(search.CorpusSearch)
        index.experiment = NS(args=NS(depth=1, retrieval_batch_size=2, threads=1))
        index.lookup = ["10", "20", "30"]
        class Dense:
            def search(self, query, depth):
                values = query @ np.array([[1, 0], [0, 1], [.1, .1]], dtype=np.float32).T
                order = np.argsort(-values, axis=1)[:, :depth]
                return np.take_along_axis(values, order, axis=1), order
        index.dense = Dense()
        index.sparse = NS(batch_search=lambda topics, ids, depth, **k: {qid: [NS(docid="20", score=3)] for qid in ids})
        encoder = NS(encode=lambda records: (torch.zeros(len(records), 5, 10), torch.tensor([[1., 0.]] * len(records))))
        with patch.object(backend, "sparse_vectors", return_value=[{"x": 2}, {}]):
            dense, sparse = index.retrieve(encoder, [dict(id="q0"), dict(id="q1")])
        self.assertEqual(list(dense["q0"]["docs"]), ["10"])
        self.assertEqual(list(sparse["q0"]["docs"]), ["20"])
        self.assertEqual(sparse["q1"], _run({}))

    def test_alpha_scan_ties_and_selection_metric(self):
        from qwen_vl_pipeline import _run
        d, s = {"q": _run({"a": 2, "b": 1})}, {"q": _run({"b": 2, "a": 1})}
        evaluator = lambda ex, runs: {"hybrid": {"Recall@1": 10, "Recall@5": 20, "Recall@10": 30, "Recall@20": 40}}
        alpha, rows = search.select_alpha(None, d, s, evaluator)
        self.assertEqual(alpha, .1)
        self.assertEqual(len(rows), 9)
        self.assertEqual(sum(row["selected"] for row in rows), 1)
        self.assertEqual(rows[0]["selection_score"], 60)

    def test_tbpr_recall_at_20_and_empty_queries_use_person_identity(self):
        api = backend.pipeline_module()
        evaluation = api.RecallEvaluation([1, 5, 10, 20], {"q1": {"person"}, "q2": {"person"}}, {},
            {str(i): "person" if i == 19 else "other" for i in range(20)}, 2)
        ex = NS(pipeline_args=NS())
        run = {"q1": {str(i): 20-i for i in range(20)}, "q2": {}}
        with patch.object(api, "_build_recall_evaluation", return_value=evaluation):
            metric = search.evaluate(ex, {"sparse": run}, api)["sparse"]
        self.assertEqual(metric, {"Recall@1": 0, "Recall@5": 0, "Recall@10": 0, "Recall@20": 50})

    def test_gather_handles_empty_shards_and_rejects_duplicate_queries(self):
        context = NS(is_distributed=True, world_size=3)
        def gather(parts, local):
            parts[:] = [{"q0": {}}, {}, {"q1": {}}]
        with patch("torch.distributed.all_gather_object", side_effect=gather):
            self.assertEqual(set(search.gather_run({}, context)), {"q0", "q1"})
        with patch("torch.distributed.all_gather_object", side_effect=lambda parts, local: parts.__setitem__(slice(None), [{"q": {}}, {}, {"q": {}}])):
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                search.gather_run({}, context)
        with patch("torch.distributed.all_gather_object", side_effect=lambda parts, local: parts.__setitem__(slice(None), [None, "bad worker", None])):
            with self.assertRaisesRegex(RuntimeError, "rank 1.*bad worker"):
                search.collective(lambda: 3, context)


class BatchTests(unittest.TestCase):
    def environment(self, fixture):
        return dict(MODEL_FAMILY="qwen25", DATASET="RSTPReid", MODEL_PATH=str(fixture.root / "checkpoint"),
            DATA_ROOT=str(fixture.root / "data"), DENSE_OUTPUT_ROOT=str(fixture.root / "dense"),
            SPARSE_OUTPUT_ROOT=str(fixture.root / "sparse"), SEARCH_OUTPUT_ROOT=str(fixture.root / "results"),
            SPARSE_LENGTH="20", VIEW_EXAMPLE_MODE="fixed", VIEW_SELECTION_ROUNDS="2",
            VIEW_TEXTS_PER_ROUND="2", VIEW_IMAGES_PER_ROUND="1", VIEW_SELECTION_MODALITY="both")

    def test_exact_batch_lookup_and_manual_paths_share_same_experiment(self):
        with temporary_directory() as directory:
            fixture = corpus(directory)
            env = self.environment(fixture)
            settings = batch.Settings.from_environment("tbpr", env)
            model, data, training = batch.namespaces(settings, "qwen25", "RSTPReid", "tbpr", "image")
            location, _, _ = batch.selection_location(settings, model, data, training, fixture.identity)
            views.atomic_json(location, fixture.selection)
            with patch.object(batch, "annotation_files", return_value=[]):
                jobs, missing = batch.prepare_jobs(settings, identity_loader=lambda *a, **k: fixture.identity)
                env.update(DENSE_CORPUS_DIR=str(fixture.roots[0]), SPARSE_CORPUS_DIR=str(fixture.roots[1]))
                manual, _ = batch.prepare_jobs(batch.Settings.from_environment("tbpr", env), identity_loader=lambda *a, **k: fixture.identity)
            self.assertFalse(missing)
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0].configuration, manual[0].configuration)
            self.assertIn("src/search_adaptive_views.py", jobs[0].command)
            self.assertNotIn("--stage1-depth", jobs[0].command)

    def test_defaults_scope_and_single_job_constraints(self):
        for group, count in (("ir", 20), ("tbpr", 15), ("fused", 15)):
            s = batch.Settings.from_environment(group, {})
            self.assertEqual(len(s.families) * len(s.datasets) * (len(s.queries) or 1), count)
            self.assertEqual(s.controls["SELECT_HYBRID_ALPHA"], "false")
        for values in (dict(DENSE_CORPUS_DIR="manual"), dict(USE_VIEW_OVERLAP_SELECTION="true"),
                       dict(LLAVA_IMAGE_MODE="base"), dict(SHUFFLE_VIEW_OPTIONS="true")):
            with self.assertRaises(ValueError):
                batch.Settings.from_environment("tbpr", values)

    def test_batch_failure_preserves_completed_metrics_and_dry_run_writes_nothing(self):
        with temporary_directory() as directory:
            root = Path(directory)
            settings = NS(roots=dict(results=root), controls=dict(DRY_RUN="false"), environment={}, project=ROOT)
            jobs = [batch.Job("one", ["command"], root / "one", {}), batch.Job("two", ["command"], root / "two", {})]
            def runner(command, **kwargs):
                if not (root / "one").exists():
                    views.atomic_json(root / "one/recall_metrics.json", dict(alpha=.5, alpha_selection_source="fixed",
                        metrics={"dense": {"Recall@20": 70}}))
                    views.atomic_json(root / "one/run_metadata.json", dict(selected_views=[]))
                    return NS(returncode=0)
                return NS(returncode=7)
            with patch.object(batch, "prepare_jobs", return_value=(jobs, [])), redirect_stdout(io.StringIO()):
                self.assertEqual(batch.run(settings, runner=runner), 7)
            content = next(root.rglob("batch_summary.csv")).read_text(encoding="utf-8-sig")
            self.assertIn("70", content)
            self.assertIn("failed", content)
            before = sorted(str(p) for p in root.rglob("*"))
            settings.controls["DRY_RUN"] = "true"
            with patch.object(batch, "prepare_jobs", return_value=(jobs, [])), redirect_stdout(io.StringIO()):
                self.assertEqual(batch.run(settings, runner=Mock(side_effect=AssertionError())), 0)
            self.assertEqual(before, sorted(str(p) for p in root.rglob("*")))

    def test_shell_syntax_and_environment_forwarding(self):
        bash = "D:/Git/bin/bash.exe" if os.name == "nt" else "bash"
        # Git Bash cannot access the Windows extended-path prefix used by corpus fixtures.
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / "inspect.py"
            helper.write_text("import os,json; print(json.dumps({k:os.environ.get(k) for k in "
                "['MODEL_FAMILIES','DATASETS','USE_FREERET_DENSE','VIEW_PROBABILITY_MASS_THRESHOLD','SELECT_HYBRID_ALPHA',"
                "'USE_STC_SPARSE','STC_ALGORITHM','STC_N_CLUSTERS','STC_SEED','STC_BATCH_SIZE','STC_MAX_ITER','STC_WEIGHT_SCALE','STC_CACHE_ROOT']}))", encoding="utf-8")
            launcher = Path(directory) / "python"
            launcher.write_text('#!/bin/bash\nexec "' + sys.executable.replace("\\", "/") + '" -B "' + helper.as_posix() + '"\n', encoding="utf-8")
            launcher.chmod(0o755)
            for group in ("ir", "tbpr", "fused"):
                script = ROOT / f"scripts/search_adaptive_views_{group}.sh"
                subprocess.run([bash, "-n", str(script)], check=True, capture_output=True, timeout=45)
                env = dict(os.environ, PYTHON_BIN=str(launcher), USE_FREERET_DENSE="true",
                    VIEW_PROBABILITY_MASS_THRESHOLD="0.8", SELECT_HYBRID_ALPHA="true",
                    USE_STC_SPARSE="true", STC_ALGORITHM="minibatch_kmeans", STC_N_CLUSTERS="32",
                    STC_SEED="17", STC_BATCH_SIZE="256", STC_MAX_ITER="12", STC_WEIGHT_SCALE="100",
                    STC_CACHE_ROOT=(Path(directory) / "dictionary").as_posix())
                result = subprocess.run([bash, str(script)], cwd=ROOT, env=env, check=True, capture_output=True, text=True, timeout=45)
                config = json.loads(result.stdout)
                self.assertEqual(config["USE_FREERET_DENSE"], "true")
                self.assertEqual(config["VIEW_PROBABILITY_MASS_THRESHOLD"], "0.8")
                self.assertEqual(config["SELECT_HYBRID_ALPHA"], "true")
                for name in ("USE_STC_SPARSE", "STC_ALGORITHM", "STC_N_CLUSTERS", "STC_SEED", "STC_BATCH_SIZE",
                             "STC_MAX_ITER", "STC_WEIGHT_SCALE", "STC_CACHE_ROOT"):
                    self.assertEqual(config[name], env[name])


if __name__ == "__main__":
    unittest.main()
