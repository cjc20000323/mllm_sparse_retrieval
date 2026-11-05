"""Independent Top-K selection, prefix equivalence, orchestration and real CSV I/O."""
from contextlib import ExitStack, redirect_stdout
import copy
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch
import test_freeret_dense as fixtures
from test_freeret_dense import ROOT, adaptive, views
from test_recall20 import Dataset
import adaptive_views_topk_experiment as experiment
import adaptive_view_shuffle as shuffle
import three_stage_retrieval_pipeline_adaptive_views as api


def selection(*, hierarchical=False, pmi=False, shuffled=False, uniform=False):
    options = views.AdaptiveViewArguments(view_example_mode="fixed", view_selection_rounds=2,
        view_texts_per_round=1, view_images_per_round=1, view_selection_hierarchical=hierarchical,
        view_selection_pmi=pmi, view_selection_shuffle=shuffled,
        view_selection_shuffle_run_id="1234567890abcdef" if shuffled else None)
    pool = views.view_pool(views.options_pool_mode(options))
    canonical = {row["aspect"]: index + 1 for index, row in enumerate(pool)}
    def score(model, processor, family, displayed, items, modality, **kwargs):
        numbers = np.array([canonical[row["aspect"]] for row in displayed], dtype=float)
        values = np.ones(len(numbers)) if uniform else numbers ** (2 if kwargs.get("prior") else 1)
        values /= values.sum()
        if kwargs.get("return_details"):
            prompt = "Candidate perspectives:\n" + "\n".join(f"{row['label']}. {row['aspect']}" for row in displayed)
            return dict(probabilities=values.tolist(), log_probabilities=np.log(values).tolist(), prompt=prompt, tokens=10)
        return values.tolist(), 10
    model = NS(config=NS(model_type="qwen2_vl", max_position_embeddings=4096))
    identity = dict(model="fixture")
    with patch.object(views, "score_group", side_effect=score), redirect_stdout(io.StringIO()):
        return views.choose_views(model, NS(), identity, "flickr", "test",
            [dict(id="t", text="a")], [dict(id="i", image="image")], options)


class ConfigurationTests(unittest.TestCase):
    def test_matrix_covers_all_models_tasks_and_both_ir_directions(self):
        jobs = experiment.build_jobs(dict(MODEL_FAMILIES=" ".join(experiment.CHECKPOINTS),
            DATASETS=" ".join(experiment.TASKS), IR_QUERY_TYPES="text image"))
        self.assertEqual(len(jobs), 50)
        for job in jobs:
            args = experiment.pipeline_args(job, api)
            self.assertEqual(args.hybrid_alpha, .5)
            if job["dataset"] in {"remuq", "llava"}:
                self.assertEqual((job["task"], job["query_type"], args.modify_type), ("it2t", "image", "modify_single"))
            if job["task"] == "ir":
                self.assertEqual(args.modify_type, "modify_single" if job["query_type"] == "image" else "no")

    def test_stop_mode_ignores_all_ranker_environment_overrides(self):
        job = experiment.build_jobs(dict(STOP_AFTER_STAGE2="true", RANKER_DTYPE="invalid", RANKER_QWEN_MIN_PIXELS="invalid",
            RERANK_NUM="invalid", RERANK_BATCH_SIZE="invalid", RANKER_MODEL_PATH="missing"))[0]
        self.assertNotIn("RANKER_MODEL_PATH", job)
        self.assertIsNone(experiment.pipeline_args(job, api).ranker_model_name_or_path)
        self.assertEqual(experiment.build_jobs(dict(VIEW_TEXTS_PER_ROUND="0",VIEW_IMAGES_PER_ROUND="0"))[0]["VIEW_EXAMPLE_MODE"],"auto")
        with self.assertRaises(ValueError):
            experiment.build_jobs(dict(VIEW_EXAMPLE_MODE="fixed",VIEW_TEXTS_PER_ROUND="0"))
        for change in (dict(HYBRID_ALPHA="nan"), dict(MODEL_FAMILIES="qwen25 qwen3", MODEL_PATH="one"),
                       dict(LLAVA_IMAGE_MODE="base"), dict(USE_FREERET_FULL_PROMPT="true")):
            with self.assertRaises(ValueError):
                experiment.build_jobs(change)

    def test_shuffle_id_is_shared_by_the_batch_and_can_be_explicit_for_resume(self):
        jobs = experiment.build_jobs(dict(DATASETS="flickr coco", IR_QUERY_TYPES="text image", SHUFFLE_VIEW_OPTIONS="true"))
        identifiers = {job["VIEW_SELECTION_RUN_ID"] for job in jobs}
        self.assertEqual(len(identifiers), 1)
        shuffle.validate_run_id(next(iter(identifiers)))

    def test_dry_run_reads_annotations_but_writes_nothing_and_skips_ranker_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"; model.mkdir(); (model / "config.json").write_text("{}")
            annotation = root / "data/flickr/flickr_test.csv"; annotation.parent.mkdir(parents=True); annotation.touch()
            env = dict(MODEL_PATH=str(model), DATA_ROOT=str(root), OUTPUT_ROOT=str(root / "output"),
                STOP_AFTER_STAGE2="true", DRY_RUN="true", RANKER_MODEL_PATH="missing", RANKER_DTYPE="invalid")
            with patch("subprocess.run", side_effect=AssertionError("dry-run launched a process")), redirect_stdout(io.StringIO()):
                self.assertEqual(experiment.run_batch(env), 0)
            self.assertFalse((root / "output").exists())

    def test_batch_launches_all_task_types_and_propagates_the_first_failure(self):
        from adaptive_encode_batch import annotation_files
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            model=root/"model"; model.mkdir(); (model/"config.json").write_text("{}")
            for dataset in ("flickr","CUHK-PEDES","webqa","remuq","llava"):
                for path in annotation_files(root,dataset,"test"):
                    path.parent.mkdir(parents=True,exist_ok=True); path.touch()
            seen=[]
            def runner(command,**kwargs):
                job=json.loads(Path(command[-1]).read_text(encoding="utf-8"))
                seen.append((job["task"],job["query_type"]))
                return NS(returncode=0)
            env=dict(MODEL_PATH=str(model),DATA_ROOT=str(root),OUTPUT_ROOT=str(root/"o"),STOP_AFTER_STAGE2="true",
                     DATASETS="flickr CUHK-PEDES webqa remuq llava")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(experiment.run_batch(env,runner=runner),0)
            self.assertEqual(seen,[("ir","text"),("tbpr","text"),("t2it","text"),("it2t","image"),("it2t","image")])
            seen.clear()
            def fail(command,**kwargs):
                seen.append(command)
                return NS(returncode=19)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(experiment.run_batch(env,runner=fail),19)
            self.assertEqual(len(seen),1)


class RankingTests(unittest.TestCase):
    def test_scores_pmi_ties_and_hierarchical_shuffled_labels(self):
        for hierarchical in (False, True):
            for pmi in (False, True):
                for shuffled in (False, True):
                    source = selection(hierarchical=hierarchical, pmi=pmi, shuffled=shuffled)
                    ranking = experiment.ranked_views(source)
                    expected = [row["label"] for row in source["pool"]]
                    if not pmi:
                        expected.reverse()
                    self.assertEqual([row["label"] for row in ranking], expected)
                    for count in experiment.COUNTS:
                        prefix = experiment.prefix_selection(source, ranking, count)
                        self.assertEqual([row["label"] for row in prefix["selected"]], expected[:count])
                        self.assertEqual(len(prefix["selected"]), count)
        tied = experiment.ranked_views(selection(uniform=True))
        self.assertEqual([row["label"] for row in tied[:7]], list("ABCDEFG"))

    def test_saved_source_is_immutable_and_derived_files_never_use_the_old_loader(self):
        source = selection(hierarchical=True, pmi=True, shuffled=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "view_selection.json"
            views.atomic_json(path, source)
            before = path.read_bytes()
            job = experiment.build_jobs(dict(VIEW_SELECTION_FILE=str(path)))[0]
            saved, location = experiment.resolve_selection(job, NS(), dict(model="fixture"), NS(),
                lambda: self.fail("reused source loaded weights"), api)
            self.assertEqual(saved, source)
            self.assertEqual(location, str(path.resolve()))
            experiment.prefix_selection(saved, experiment.ranked_views(saved), 7)
            self.assertEqual(before, path.read_bytes())

    def test_auto_selection_is_called_once_and_reused_across_directions(self):
        source = selection()
        with tempfile.TemporaryDirectory() as directory:
            job = experiment.build_jobs(dict(SPARSE_OUTPUT_ROOT=directory, VIEW_EXAMPLE_MODE="fixed",
                VIEW_SELECTION_ROUNDS="2", VIEW_TEXTS_PER_ROUND="1", VIEW_IMAGES_PER_ROUND="1"))[0]
            runtime = NS(encoder=NS(), processor=NS())
            with patch.object(adaptive, "source_pools", return_value=([dict(id="t", text="a")], [dict(id="i", image="image")])), \
                 patch.object(views, "choose_views", return_value=source) as choose:
                first = experiment.resolve_selection(job, NS(), dict(model="fixture"), NS(), lambda: runtime, api)
                second = experiment.resolve_selection(dict(job, query_type="image"), NS(), dict(model="fixture"), NS(),
                    lambda: self.fail("cache reuse loaded weights"), api)
            self.assertEqual(first, second)
            self.assertEqual(choose.call_count, 1)


class PrefixTests(unittest.TestCase):
    def test_all_prefixes_use_original_sparse_aggregation_and_actual_denominator(self):
        tokenizer = NS(get_vocab=lambda: {chr(97+i): i for i in range(8)})
        logits = [torch.arange(1, 57, dtype=torch.float32).reshape(7, 8) / 11]
        records = [dict(id="one", text="a")]
        for policy in ("sum", "max", "replace"):
            for mean in (False, True):
                data = fixtures.data_args(sparse_length=8, sparse_value_type=policy, sparse_value_mean=mean)
                encoder = NS(family="qwen2_vl", processor=NS(tokenizer=tokenizer), data=data)
                result = experiment.prefix_vectors(encoder, records, logits)
                for count in experiment.COUNTS:
                    fresh = NS(family="qwen2_vl", processor=NS(tokenizer=tokenizer), data=data)
                    expected = adaptive.sparse_vectors(fresh, records, [logits[0][:count]])
                    self.assertEqual(result[count], expected)
                    self.assertTrue(result[count])
                if policy == "sum":
                    self.assertNotEqual(result[1], result[7])


class PrefixRealModelTests(fixtures.RealModelTests):
    def compare(self, model, processor, records, family, adapter="adaptive"):
        if adapter != "adaptive":
            return
        if family.startswith("llava"):
            # The random-model fixture has no saved chat template. Supply the
            # same native delimiters as its existing full-prompt model tests.
            if "mistral" in family:
                processor.chat_template="{% for m in messages %}[INST]{% for c in m['content'] %}{% if c['type']=='image' %}<image>\n{% else %}{{ c['text'] }}{% endif %}{% endfor %}[/INST]{% endfor %}"
            else:
                processor.chat_template="{% for m in messages %}<|start_header_id|>user<|end_header_id|>\n\n{% for c in m['content'] %}{% if c['type']=='image' %}<image>\n{% else %}{{ c['text'] }}{% endif %}{% endfor %}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{% endfor %}"
        for use_freeret, constraints, full in ((False,False,False),(True,False,False),(True,True,False),(True,False,True)):
            data = fixtures.data_args(sparse_length=12, print_sparse=False)
            args = fixtures.model_args(use_freeret, family)
            args.use_freeret_prompt_constraints = constraints
            args.use_freeret_full_prompt = full
            with redirect_stdout(io.StringIO()):
                encoder = adaptive.AdaptiveEncoder(model, processor, args, data, "ir", {"selected": views.view_pool()[:7]})
            logits, dense = encoder.encode(records)
            prefixes = experiment.prefix_vectors(encoder, records, logits)
            for count in experiment.COUNTS:
                separate = adaptive.AdaptiveEncoder(model, processor, args, data, "ir", {"selected": views.view_pool()[:count]})
                own_logits, own_dense = separate.encode(records)
                self.assertTrue(torch.allclose(dense, own_dense, atol=2e-6, rtol=2e-5), family)
                for first, second in zip(logits, own_logits):
                    self.assertTrue(torch.allclose(first[:count], second, atol=2e-6, rtol=2e-5), family)
                self.assertEqual(prefixes[count], adaptive.sparse_vectors(separate, records, own_logits))
            fixtures.dense.configure_model(model, False)


class WorkflowTests(unittest.TestCase):
    def run_fixture(self, directory, *, stop=True, fail_at=None):
        root = Path(directory)
        model = root / "m"; model.mkdir(exist_ok=True); (model / "config.json").write_text("{}")
        source = selection()
        source_path = root / "selection.json"; views.atomic_json(source_path, source)
        job = experiment.build_jobs(dict(MODEL_PATH=str(model), DATA_ROOT=str(root), OUTPUT_ROOT=str(root / "o"),
            VIEW_SELECTION_FILE=str(source_path), STOP_AFTER_STAGE2=str(stop), DTYPE="float32", SPARSE_LENGTH="8",
            STAGE1_DEPTH="6", STAGE2_DEPTH="6", PER_DEVICE_BATCH_SIZE="2", RETRIEVAL_BATCH_SIZE="3"))[0]
        job.update(batch_dir=str(root / "batch"), job_index=1)
        config = NS(model_type="qwen2_vl", hidden_size=3)
        tokenizer = NS(get_vocab=lambda: {chr(97+i): i for i in range(8)})
        calls = []
        class Encoder:
            def __init__(self, encoder_model, processor, model_args, data, task, selected):
                self.model, self.processor, self.data, self.family = encoder_model, processor, data, "qwen2_vl"
                self.last_prompts = ["whole dense plus seven views"]
                self.selected = selected["selected"]
                if len(self.selected) != 7:
                    raise AssertionError("not one seven-view encoding")
            def encode(self, records):
                calls.append([row["id"] for row in records])
                logits = [torch.arange(1,57,dtype=torch.float32).reshape(7,8) / (int(row["id"])%6+10) for row in records]
                dense = torch.tensor([[1., int(row["id"])%6 / 8, .2] for row in records])
                return logits, torch.nn.functional.normalize(dense, dim=-1)
        candidates = [dict(id=str(10000+i), text=f"candidate {i}") for i in range(6)]
        queries = [dict(id=str(i), text=f"query {i}") for i in range(6)]
        def indexer(sparse_root, **kwargs):
            index = Path(sparse_root) / "collection/index"
            index.mkdir(); (index / "segments_1").write_bytes(b"fake lucene")
            return index
        def search(sparse_root, topics, qids, depth, **kwargs):
            from collections import Counter
            documents = []
            for path in sorted((Path(sparse_root) / "collection").glob("corpus_*.jsonl")):
                documents.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
            result = {}
            for query in qids:
                terms = Counter(topics[query].split())
                scores = {row["id"]:float(sum(weight*terms[token] for token,weight in row["vector"].items())) for row in documents}
                result[query] = dict(sorted(scores.items(), key=lambda pair:-pair[1])[:depth])
            return result
        rerank_calls = []
        def rerank(args, hybrid, *unused, **kwargs):
            rerank_calls.append(1)
            if stop:
                raise AssertionError("ranker used in stop mode")
            if fail_at and len(rerank_calls) == fail_at:
                raise RuntimeError("ranker fixture failure")
            return copy.deepcopy(hybrid)
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(patch.object(adaptive, "load_model_identity", return_value=dict(model="fixture")))
            stack.enter_context(patch("transformers.AutoConfig.from_pretrained", return_value=config))
            stack.enter_context(patch.object(views, "model_identity", return_value=dict(model="fixture")))
            loader = stack.enter_context(patch.object(api, "_load_retrieval_runtime", side_effect=lambda *args: NS(
                encoder=NS(config=config), processor=NS(tokenizer=tokenizer), query_cache={})))
            stack.enter_context(patch.object(adaptive, "configure_processor"))
            stack.enter_context(patch.object(adaptive, "AdaptiveEncoder", Encoder))
            stack.enter_context(patch.object(api, "build_dataset", return_value=Dataset("flickr")))
            stack.enter_context(patch("qwen_vl_pipeline.dataset_records", return_value=candidates))
            stack.enter_context(patch.object(api, "_query_records", return_value=queries))
            stack.enter_context(patch.object(api, "_qwen_query_records", side_effect=lambda args, records: records))
            build = stack.enter_context(patch("qwen_sparse_index.build_sparse_index", side_effect=indexer))
            stack.enter_context(patch.object(api, "retrieve_sparse_topics_from_lucene", side_effect=search))
            stack.enter_context(patch.object(api, "_load_ranker_model", side_effect=AssertionError("ranker loading unexpected")))
            ranker = stack.enter_context(patch.object(api, "run_stage3", side_effect=rerank))
            if fail_at:
                with self.assertRaisesRegex(ValueError, "ranker fixture failure"):
                    experiment.run_worker(job)
            else:
                experiment.run_worker(job)
                first_calls = len(calls)
                experiment.run_worker(job)
                self.assertEqual(len(calls), first_calls)
                self.assertEqual(loader.call_count, 1)
                self.assertEqual(build.call_count, 7)
            self.assertEqual(ranker.call_count, 0 if stop else fail_at or 7)
        return job, calls

    def test_two_stage_all_seven_prefixes_stream_once_and_resume_without_weights(self):
        with fixtures.temporary_directory() as directory:
            job, calls = self.run_fixture(directory)
            self.assertEqual(len(calls), 5)  # three corpus batches, two query batches
            root = Path(job["output_root"])
            with (root / "topk_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 7*4)
            self.assertEqual({int(row["view_count"]) for row in rows}, set(range(1,8)))
            self.assertTrue(all("R@20" in row and row["execution_mode"]=="two_stage" for row in rows))
            self.assertFalse(list(root.rglob("stage3*")))
            self.assertEqual(len(list(root.rglob("corpus_0.pkl"))), 1)
            self.assertEqual(len(list(root.rglob("query_0.pkl"))), 1)
            query_files=list(root.rglob("query_0.jsonl"))
            self.assertEqual(len(query_files),7)
            for path in query_files:
                record=json.loads(path.read_text(encoding="utf-8").splitlines()[0])
                self.assertEqual(set(record),{"id","vector"})
            documents = sorted(root.rglob("corpus_0.jsonl"))
            self.assertEqual(len(documents), 7)
            first = json.loads(documents[0].read_text(encoding="utf-8").splitlines()[0])
            last = json.loads(documents[-1].read_text(encoding="utf-8").splitlines()[0])
            self.assertNotEqual(first["vector"], last["vector"])

    def test_three_stage_reranks_seven_groups_and_retains_incremental_csv(self):
        with fixtures.temporary_directory() as directory:
            job, _ = self.run_fixture(directory, stop=False)
            root = Path(job["output_root"])
            with (root / "topk_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 7*5)
            self.assertEqual(sum(row["stage"]=="stage3" for row in rows), 7)
            self.assertEqual(len(list(root.rglob("stage3_ranker_run.json"))), 7)

    def test_failed_group_preserves_previous_results_and_current_stage2_report(self):
        with fixtures.temporary_directory() as directory:
            job, _ = self.run_fixture(directory, stop=False, fail_at=3)
            root = Path(job["output_root"])
            with (root / "topk_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2*5+4)
            self.assertEqual(sum(row["stage"]=="stage3" for row in rows), 2)
            self.assertEqual({int(row["view_count"]) for row in rows}, {1,2,3})

    def test_corrupt_completed_artifact_never_silently_reuses_or_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = root / "file"; item.write_text("complete")
            experiment.stamp(root, dict(mode=1), [item])
            self.assertTrue(experiment.completed(root, dict(mode=1)))
            item.write_text("corrupt")
            with self.assertRaisesRegex(ValueError, "corrupt"):
                experiment.completed(root, dict(mode=1))
            self.assertEqual(item.read_text(), "corrupt")

    def test_real_index_builder_accepts_every_representation_mode_and_reuses_across_threads(self):
        import dense_representation as dense
        for use,constraints,full in ((False,False,False),(True,False,False),(True,True,False),(True,True,True)):
            with fixtures.temporary_directory() as directory:
                job=experiment.build_jobs(dict(USE_FREERET_DENSE=str(use),USE_FREERET_PROMPT_CONSTRAINTS=str(constraints),
                    USE_FREERET_FULL_PROMPT=str(full)))[0]
                _,roots=experiment.artifact_paths(directory,job)
                root=roots[1]; (root/"collection").mkdir(parents=True)
                (root/"collection/corpus_0.jsonl").write_text('{"id":"one","contents":"","vector":{"a":3}}\n')
                dense.write_manifest(root,use,constraints,full)
                def build(command,**kwargs):
                    target=Path(command[command.index("--index")+1]); target.mkdir()
                    (target/"segments_1").write_bytes(b"index")
                with patch("qwen_sparse_index.subprocess.run",side_effect=build) as call:
                    experiment.ensure_index(root,dict(encoding="same",count=1),threads=1)
                    experiment.ensure_index(root,dict(encoding="same",count=1),threads=8)
                    self.assertEqual(call.call_count,1)
                self.assertTrue((root/"collection/index/segments_1").is_file())


class LauncherTests(unittest.TestCase):
    def test_real_bash_forwards_local_and_external_values_and_errors(self):
        bash = shutil.which("bash") or "D:/Git/bin/bash.exe"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir(); (root / "src").mkdir()
            script = root / "scripts/adaptive_views_topk_experiment.sh"
            text = (ROOT / "scripts/adaptive_views_topk_experiment.sh").read_text(encoding="utf-8")
            script.write_text(text.replace('STOP_AFTER_STAGE2="${STOP_AFTER_STAGE2-false}"',
                                           'STOP_AFTER_STAGE2="${STOP_AFTER_STAGE2-true}"'), encoding="utf-8")
            (root / "src/adaptive_views_topk_experiment.py").write_text(
                "import os,sys,json\nprint(json.dumps({k:os.environ.get(k) for k in ['STOP_AFTER_STAGE2','DATASETS','HYBRID_ALPHA']}))\nsys.exit(17)\n")
            env = dict(os.environ, PYTHON_BIN=sys.executable, DATASETS="remuq llava", HYBRID_ALPHA="0.7")
            env.pop("STOP_AFTER_STAGE2", None)
            result = subprocess.run([bash, script.as_posix()], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 17, result.stderr)
            self.assertEqual(json.loads(result.stdout), dict(STOP_AFTER_STAGE2="true", DATASETS="remuq llava", HYBRID_ALPHA="0.7"))
            env["STOP_AFTER_STAGE2"]="false"
            result = subprocess.run([bash, script.as_posix()], env=env, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout)["STOP_AFTER_STAGE2"], "false")


if __name__ == "__main__":
    unittest.main()
