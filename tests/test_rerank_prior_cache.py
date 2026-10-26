from __future__ import annotations

import ast
import copy
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import rerank_prior_cache as cache
import reranker
import test_reranker_batching as batch_fixtures
import test_freeret_distributed as distributed_fixtures
import test_three_stage_retrieval_pipeline as pipeline_fixtures


class QuietProgress(batch_fixtures.QuietProgress):
    def close(self):
        pass


class CaptionPriorCacheTests(unittest.TestCase):
    def setUp(self):
        self.fixture = batch_fixtures.RerankerBatchTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root / "priors"
        manager = patch.object(reranker, "tqdm", QuietProgress)
        manager.__enter__()
        self.addCleanup(manager.__exit__, None, None, None)

    def make(self, task="ir", size=5):
        ranker, fusion = self.fixture.ranker(task, "image", size)
        query_id = next(iter(fusion))
        # The production text_map contains only corpus candidates, not queries.
        ranker.text_map = {key: ranker.text_map[key] for key in fusion[query_id]}
        if task == "ir":
            ranker.img_path_map["q1"] = ranker.img_path_map[query_id]
        fusion["q1"] = dict(fusion[query_id])
        return ranker, fusion

    def args(self, task="ir", *, enabled=True, template="caption_generation", side="after_pad"):
        return (
            SimpleNamespace(reps_loc=side, dataset_split="test"),
            SimpleNamespace(fp16=False, task_type=task),
            SimpleNamespace(model_name_or_path="llava-hf-llava-v1.6-mistral-7b-hf"),
            SimpleNamespace(modify_type="modify_single", use_rerank_prior_cache=enabled,
                            rerank_prior_cache_dir=str(self.root)),
        )

    def invoke(self, ranker, fusion, args, *, batch=4, template="caption_generation", prepared=False, num=100):
        return ranker.caption_generation_rerank(
            fusion, "pointwise", num, *args, rerank_batch_size=batch,
            rerank_prompt_type=template, prior_cache_prepared=prepared,
        )

    @staticmethod
    def prior_calls(ranker):
        return [call for call in ranker.model.calls if not any(call["images"])]

    def test_scores_and_rankings_match_for_both_tasks_templates_and_padding_sides(self):
        for task in ("ir", "it2t"):
            for template in ("caption_generation", "what_caption_generation"):
                for side in ("before_pad", "after_pad"):
                    with self.subTest(task=task, template=template, side=side):
                        ranker, fusion = self.make(task)
                        args = self.args(task, enabled=False, side=side)
                        expected = self.invoke(ranker, fusion, args, batch=1, template=template)
                        args[-1].use_rerank_prior_cache = True
                        ranker.model.calls.clear()
                        actual = self.invoke(ranker, fusion, args, template=template)
                        self.fixture.assert_scores_equal(expected, actual)
                        # All five priors once, in batches of four and one.
                        self.assertEqual([call["size"] for call in self.prior_calls(ranker)], [4, 1])
                        self.assertFalse(ranker.prior_cache_report["cache_hit"])
                        ranker.model.calls.clear()
                        cached = self.invoke(ranker, fusion, args, batch=2, template=template)
                        self.fixture.assert_scores_equal(expected, cached)
                        self.assertTrue(ranker.prior_cache_report["cache_hit"])
                        self.assertEqual(self.prior_calls(ranker), [])

    def test_prepared_cache_excludes_priors_from_online_loop_and_covers_full_pool(self):
        ranker, fusion = self.make()
        args = self.args()
        report = ranker.prepare_caption_prior_cache(*args, rerank_batch_size=4)
        self.assertEqual(report["candidate_count"], 5)
        ranker.model.calls.clear()
        result = self.invoke(ranker, fusion, args, prepared=True, num=2)
        self.assertEqual([len(run) for run in result.values()], [2, 2])
        self.assertEqual([call["size"] for call in ranker.model.calls], [2, 2])
        self.assertEqual(self.prior_calls(ranker), [])

    def test_new_ranker_loads_without_pure_text_inference(self):
        ranker, fusion = self.make()
        args = self.args()
        expected = self.invoke(ranker, fusion, args)
        other, fusion = self.make()
        actual = self.invoke(other, fusion, args, batch=1)
        self.fixture.assert_scores_equal(expected, actual)
        self.assertTrue(other.prior_cache_report["cache_hit"])
        self.assertEqual(self.prior_calls(other), [])

    def test_batch_padding_state_does_not_change_tokenizer_fingerprint(self):
        ranker, _ = self.make()
        args = self.args()
        state = {"padding": None, "truncation": None, "model": {"vocab": {"a": 0}}}
        ranker.processor.tokenizer.backend_tokenizer = SimpleNamespace(to_str=lambda: json.dumps(state))
        first = ranker.prepare_caption_prior_cache(*args)
        state["padding"] = {"strategy": {"Fixed": 1234}}
        state["truncation"] = {"max_length": 4321}
        second = ranker.prepare_caption_prior_cache(*args, rerank_batch_size=4)
        self.assertEqual(first["fingerprint"], second["fingerprint"])
        self.assertTrue(second["cache_hit"])
        state["model"]["vocab"]["b"] = 1
        third = ranker.prepare_caption_prior_cache(*args)
        self.assertNotEqual(first["fingerprint"], third["fingerprint"])

    def test_cache_disabled_keeps_online_prior_forwards(self):
        ranker, fusion = self.make()
        self.invoke(ranker, fusion, self.args(enabled=False))
        self.assertEqual([call["size"] for call in self.prior_calls(ranker)], [4, 1, 4, 1])
        self.assertFalse(self.root.exists())

    def test_empty_or_nontext_candidates_fail_before_cache_write(self):
        for value in ("", "  ", None, {"text": "not a text candidate"}):
            ranker, _ = self.make()
            ranker.text_map["c0"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "nonempty candidate text"):
                ranker.prepare_caption_prior_cache(*self.args())
        self.assertFalse(self.root.exists())

    def test_offline_uses_actual_candidate_map_and_original_text_only_prompt(self):
        ranker, fusion = self.make("it2t")
        ranker.dataset.split = "train"
        ranker.dataset.text_dict = {"wrong": "must not be encoded"}
        args = self.args("it2t")
        template = ranker._caption_template(args[1], args[2], "caption_generation")
        query = next(iter(fusion))
        expected = ranker._requests(query, list(fusion[query]), "it2t", template,
                                    "caption_generation", caption=True)
        with patch.object(ranker, "_caption_scores", wraps=ranker._caption_scores) as score:
            report = ranker.prepare_caption_prior_cache(*args, rerank_batch_size=4)
        seen = [item for call in score.call_args_list for item in call.args[0]]
        actual_inputs = sorted(item.text for item in seen)
        expected_inputs = sorted(item.text_only_prefix + item.text[len(item.prefix):] for item in expected)
        self.assertEqual(actual_inputs, expected_inputs)
        self.assertEqual(report["candidate_count"], 5)
        payload = json.loads(Path(report["cache_path"]).read_text(encoding="utf-8"))
        self.assertEqual(payload["metadata"]["split"], "test")
        self.assertTrue(all(call.kwargs["text_only"] for call in score.call_args_list))

    def test_changed_candidate_after_preparation_is_rejected_not_scored_with_stale_prior(self):
        ranker, fusion = self.make()
        args = self.args()
        ranker.prepare_caption_prior_cache(*args)
        ranker.text_map["c0"] = "changed candidate text"
        with self.assertRaisesRegex(ValueError, "Cached prior input differs"):
            self.invoke(ranker, fusion, args, prepared=True)

    def test_split_template_model_lora_and_text_changes_invalidate_cache(self):
        ranker, _ = self.make()
        args = self.args()
        original = ranker.prepare_caption_prior_cache(*args)["cache_path"]
        mutations = [
            lambda a: setattr(a[0], "dataset_split", "val"),
            lambda a: setattr(a[2], "model_name_or_path", "another-model"),
            lambda a: (setattr(a[2], "lora", True), setattr(a[2], "lora_model_path", "adapter-a")),
            lambda a: setattr(a[1], "bf16", True),
        ]
        for mutate in mutations:
            modified = copy.deepcopy(args)
            mutate(modified)
            report = ranker.prepare_caption_prior_cache(*modified)
            self.assertNotEqual(report["cache_path"], original)
            self.assertFalse(report["cache_hit"])
        report = ranker.prepare_caption_prior_cache(*args, rerank_prompt_type="what_caption_generation")
        self.assertNotEqual(report["cache_path"], original)
        ranker.text_map["c0"] += " changed"
        report = ranker.prepare_caption_prior_cache(*args)
        self.assertNotEqual(report["cache_path"], original)

    def test_corrupt_missing_and_nonfinite_entries_rebuild(self):
        ranker, _ = self.make()
        args = self.args()
        report = ranker.prepare_caption_prior_cache(*args)
        path = Path(report["cache_path"])
        for corruption in ("json", "missing", "nonfinite", "finite_score", "wrong_text", "wrong_metadata"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if corruption == "json":
                path.write_text("not json", encoding="utf-8")
            else:
                if corruption == "missing":
                    payload["entries"].pop()
                elif corruption == "nonfinite":
                    payload["entries"][0]["mean_nll"] = float("nan")
                elif corruption == "finite_score":
                    payload["entries"][0]["mean_nll"] += 1.0
                elif corruption == "wrong_text":
                    payload["entries"][0]["text_sha256"] = "wrong"
                else:
                    payload["metadata"]["prefix"] += "wrong"
                path.write_text(json.dumps(payload), encoding="utf-8")
            with self.subTest(corruption=corruption):
                report = ranker.prepare_caption_prior_cache(*args)
                self.assertFalse(report["cache_hit"])
                self.assertEqual(len(json.loads(path.read_text())["entries"]), 5)
        self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_precompute_runs_even_with_empty_online_query_shard(self):
        ranker, _ = self.make()
        self.assertEqual(self.invoke(ranker, {}, self.args()), {})
        self.assertEqual(sum(call["size"] for call in self.prior_calls(ranker)), 5)


class PriorCacheProtocolTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.candidates = [(cache.candidate_key(index), f"caption {index}") for index in range(7)]
        self.metadata = {"model": {"name": "fake-model"}, "dataset": "flickr", "split": "test",
                         "scoring": "test-nll", "pool": cache.fingerprint(self.candidates)}

    def run_workers(self, size=3, *, fail_rank=None, mismatch=False):
        group = distributed_fixtures.ThreadGroup(size)
        results, errors, seen = {}, {}, {}

        def worker(rank):
            def score(batch):
                if rank == fail_rank:
                    raise RuntimeError("model failure")
                seen[rank].extend(key for key, _ in batch)
                return [float(json.loads(key)[1]) + 0.5 for key, _ in batch]
            seen[rank] = []
            metadata = {**self.metadata, **({"different": True} if mismatch and rank else {})}
            try:
                results[rank] = cache.load_or_build_prior_cache(
                    metadata, self.candidates, score, 2, cache_dir=self.root,
                    collectives=distributed_fixtures.ThreadCollectives(group, rank),
                )
            except BaseException as exc:
                errors[rank] = exc

        threads = [threading.Thread(target=worker, args=(rank,), daemon=True) for rank in range(size)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(15)
        group.sync.abort()
        self.assertFalse(any(thread.is_alive() for thread in threads), "prior cache protocol hung")
        return results, errors, seen

    def test_distributed_build_shards_once_then_all_workers_hit(self):
        results, errors, seen = self.run_workers()
        self.assertEqual(errors, {})
        self.assertEqual(sorted(key for values in seen.values() for key in values), sorted(key for key, _ in self.candidates))
        self.assertEqual(results[0][0], results[1][0])
        self.assertEqual(results[0][0], results[2][0])
        self.assertEqual(len(list(self.root.rglob("*.json"))), 1)
        results, errors, seen = self.run_workers(size=2)
        self.assertEqual(errors, {})
        self.assertTrue(all(result[1]["cache_hit"] for result in results.values()))
        self.assertTrue(all(not values for values in seen.values()))

    def test_empty_prior_shards_participate(self):
        self.candidates = self.candidates[:1]
        results, errors, seen = self.run_workers(size=3)
        self.assertEqual(errors, {})
        self.assertEqual([len(seen[rank]) for rank in range(3)], [1, 0, 0])
        self.assertEqual(len(results), 3)

    def test_worker_failure_reaches_all_ranks_without_completed_cache(self):
        _, errors, _ = self.run_workers(fail_rank=1)
        self.assertEqual(len(errors), 3)
        self.assertTrue(all("model failure" in str(error) for error in errors.values()))
        self.assertEqual(list(self.root.rglob("*.json")), [])

    def test_mismatched_rank_metadata_fails_without_writing(self):
        _, errors, _ = self.run_workers(mismatch=True)
        self.assertEqual(len(errors), 3)
        self.assertTrue(all("Ranks disagree" in str(error) for error in errors.values()))
        self.assertEqual(list(self.root.rglob("*.json")), [])

    def test_atomic_write_failure_is_broadcast(self):
        with patch.object(cache, "_atomic_save", side_effect=OSError("disk full")):
            _, errors, _ = self.run_workers()
        self.assertEqual(len(errors), 3)
        self.assertTrue(all("disk full" in str(error) for error in errors.values()))

    def test_checkpoint_weight_changes_and_typed_ids(self):
        checkpoint = self.root / "model"
        checkpoint.mkdir()
        weight = checkpoint / "model.safetensors"
        weight.write_bytes(b"fake weights")
        first = cache._checkpoint(checkpoint)
        stat = weight.stat()
        os.utime(weight, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        self.assertNotEqual(first, cache._checkpoint(checkpoint))
        self.assertNotEqual(cache.candidate_key(1), cache.candidate_key("1"))

    def test_invalid_option_combinations_are_rejected(self):
        for modify, template, task, query, kind in (
            ("no", "caption_generation", "ir", "image", "pointwise"),
            ("modify_single", "relevant", "ir", "image", "pointwise"),
            ("modify_single", "caption_generation", "ir", "text", "pointwise"),
            ("modify_single", "caption_generation", "tbpr", "text", "pointwise"),
            ("modify_single", "caption_generation", "t2it", "text", "pointwise"),
            ("modify_single", "caption_generation", "ir", "image", "listwise"),
        ):
            with self.subTest(task=task, query=query, template=template, modify=modify, kind=kind):
                with self.assertRaises(ValueError):
                    cache.validate_prior_cache_options(True, modify, template, task, query, kind)
        cache.validate_prior_cache_options(False, "no", "relevant", "tbpr", "text", "listwise")

    def test_invalid_model_scores_and_batch_sizes_do_not_write_successful_cache(self):
        for value in (float("nan"), float("inf"), -1.0, True):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                cache.load_or_build_prior_cache(
                    self.metadata, self.candidates, lambda batch: [value] * len(batch), 2,
                    cache_dir=self.root,
                )
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                cache.load_or_build_prior_cache(self.metadata, self.candidates, lambda batch: [], size,
                                               cache_dir=self.root)
        self.assertEqual(list(self.root.rglob("*.json")), [])


class PriorCacheEntryTests(unittest.TestCase):
    def test_hf_parser_defaults_and_explicit_cache_switch(self):
        from dataclasses import dataclass, field
        from typing import Optional
        from transformers import HfArgumentParser

        tree = ast.parse((SRC / "arguments.py").read_text(encoding="utf-8"))
        schema = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                      and node.name == "PromptRepsLLMSearchArguments")
        namespace = {"dataclass": dataclass, "field": field, "Optional": Optional, "__name__": __name__}
        exec(compile(ast.Module(body=[schema], type_ignores=[]), "arguments.py", "exec", dont_inherit=True), namespace)
        parser = HfArgumentParser(namespace["PromptRepsLLMSearchArguments"])
        defaults, = parser.parse_args_into_dataclasses([])
        self.assertFalse(defaults.use_rerank_prior_cache)
        self.assertIsNone(defaults.rerank_prior_cache_dir)
        selected, = parser.parse_args_into_dataclasses([
            "--use_rerank_prior_cache", "--rerank_prior_cache_dir", "a-cache",
            "--modify_type", "modify_single",
        ])
        self.assertTrue(selected.use_rerank_prior_cache)
        self.assertEqual(selected.rerank_prior_cache_dir, "a-cache")

    def test_all_three_scripts_forward_independent_switch_and_modify_type(self):
        for script in ("rerank.sh", "rerank_generated_validation.sh", "three_stage_retrieval_pipeline.sh"):
            with self.subTest(script=script):
                text = (SRC.parent / "scripts" / script).read_text(encoding="utf-8")
                self.assertIn('USE_RERANK_PRIOR_CACHE="${USE_RERANK_PRIOR_CACHE:-false}"', text)
                self.assertIn('MODIFY_TYPE="${MODIFY_TYPE:-no}"', text)
                self.assertIn('"${PRIOR_CACHE_ARGS[@]}"', text)
                self.assertIn('--modify_type "${MODIFY_TYPE}"', text)

    def test_legacy_preparation_precedes_observer_and_model_load_validation(self):
        text = (SRC / "rerank.py").read_text(encoding="utf-8")
        tree = ast.parse(text)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run_rerank")
        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
        prepare = next(node for node in calls if isinstance(node.func, ast.Attribute)
                       and node.func.attr == "prepare_caption_prior_cache")
        observers = [node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "StageTimeObserver"
                     and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "rerank"]
        self.assertTrue(all(prepare.lineno < node.lineno for node in observers))
        validation = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "validate_prior_cache_options")
        loads = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "from_pretrained"]
        self.assertTrue(all(validation.lineno < node.lineno for node in loads))

    def test_pipeline_parser_aliases_dispatch_and_auxiliary_timing(self):
        pipeline = pipeline_fixtures.pipeline
        for flag in ("--use-rerank-prior-cache", "--use_rerank_prior_cache"):
            args = pipeline._parser().parse_args([
                "--dataset-name", "flickr", "--query-type", "image", "--rerank-template", "caption_generation",
                "--modify_type", "modify_single", flag, "--rerank-prior-cache-dir", "a-cache",
            ])
            self.assertTrue(args.use_rerank_prior_cache)
            ranker = SimpleNamespace(caption_generation_rerank=lambda *a, **kw: (a, kw))
            arguments, kwargs = pipeline._dispatch_stage3_rerank(ranker, {}, args, None, None, None)
            self.assertTrue(arguments[-1].use_rerank_prior_cache)
            self.assertTrue(kwargs["prior_cache_prepared"])
        timer = pipeline.StageTimer()
        timer.components["reranking"] = 3.0
        timer.auxiliary["prior_cache_prepare"] = 11.0
        timer.prior_cache = {"cache_hit": False}
        report = pipeline._merge_stage_timings([timer.snapshot(0)])
        self.assertEqual(report["compute_seconds"], 3.0)
        self.assertEqual(report["auxiliary_seconds"]["prior_cache_prepare"], 11.0)
        self.assertFalse(report["prior_cache"]["cache_hit"])

    def test_pipeline_prepares_before_online_timer_and_passes_ready_cache(self):
        pipeline = pipeline_fixtures.pipeline
        args = pipeline._parser().parse_args([
            "--dataset-name", "flickr", "--query-type", "image", "--rerank-template", "what_caption_generation",
            "--modify_type", "modify_single", "--use-rerank-prior-cache",
            "--ranker-model-name-or-path", "fake-model",
        ])
        clock = batch_fixtures.fixtures.FakeClock()
        events = []
        expected = {"q": {"c0": 0.5}}

        class Ranker:
            def __init__(self, *a):
                pass

            def prepare_caption_prior_cache(self, *a):
                self_test.assertEqual(a[0].dataset_split, "test")
                self_test.assertEqual(a[5], "what_caption_generation")
                clock.advance(11)
                events.append("prepared")
                return {"cache_hit": False, "prepare_seconds": 11.0}

            def caption_generation_rerank(self, *a, **kw):
                self_test.assertTrue(kw["prior_cache_prepared"])
                self_test.assertEqual(events, ["prepared"])
                clock.advance(3)
                events.append("online")
                return expected

        self_test = self
        processor = SimpleNamespace(tokenizer=SimpleNamespace(get_vocab=lambda: {}))
        dataset = SimpleNamespace(text_dict={"c0": "caption"}, img_dict={"q": "image.png"})
        timer = pipeline.StageTimer()
        with patch.object(pipeline, "_load_ranker_model", return_value=(SimpleNamespace(device="cpu"), processor)), \
                patch.object(pipeline, "build_dataset", return_value=dataset), \
                patch.object(reranker, "Reranker", Ranker), \
                patch.object(pipeline.time, "perf_counter", clock):
            result = pipeline.run_stage3(args, expected, timing=timer)
        self.assertEqual(result, expected)
        self.assertEqual(timer.components, {"reranking": 3.0})
        self.assertEqual(timer.auxiliary, {"prior_cache_prepare": 11.0})
        self.assertEqual(events, ["prepared", "online"])

    def test_pipeline_rejects_unsupported_prior_before_other_setup(self):
        pipeline = pipeline_fixtures.pipeline
        args = pipeline._parser().parse_args([
            "--dataset-name", "flickr", "--query-type", "text", "--rerank-template", "caption_generation",
            "--modify_type", "modify_single", "--use-rerank-prior-cache",
        ])
        with self.assertRaisesRegex(ValueError, "only the existing image-to-text"):
            pipeline._validate_args(args)


if __name__ == "__main__":
    unittest.main()
