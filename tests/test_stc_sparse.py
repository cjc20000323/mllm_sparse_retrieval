"""STC probability aggregation, immutable dictionaries and adaptive pipeline wiring."""
from contextlib import redirect_stdout
import copy
import importlib
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import stc_sparse as stc
import adaptive_view_encoding as encoding
import adaptive_views as views
import test_freeret_dense as fixtures


class Tokenizer:
    all_special_ids = [0, 9]
    words = {"[PAD]": 0, "car": 1, "auto": 2, "dog": 3, "hound": 4, "1": 5, "2": 6, " ": 7, "[END]": 9}

    def get_vocab(self):
        return self.words.copy()

    def batch_decode(self, sequences, skip_special_tokens=True):
        reverse = {value: key for key, value in self.words.items()}
        return ["".join(reverse[i] for i in row if not skip_special_tokens or i not in self.all_special_ids) for row in sequences]


def args_at(root, **changes):
    options = {key: (False if key == "use_stc_sparse" else value) for key, value in stc.OPTIONS.items()}
    options.update(use_stc_sparse=True, stc_n_clusters=2, stc_cache_root=str(root / "cache"),
                   stc_dtype="float32", model_name_or_path=str(root), lora=False,
                   stc_batch_size=8, stc_max_iter=15)
    options.update(changes)
    return NS(**options)


class STCTests(unittest.TestCase):
    def setUp(self):
        self.temp = fixtures.temporary_directory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "config.json").write_text('{}', encoding="utf-8")
        self.tokenizer = Tokenizer()
        self.addCleanup(patch.stopall)
        patch.object(stc, "_tokenizer", return_value=self.tokenizer).start()
        self.args = args_at(self.root)
        self.input = torch.tensor([[1., 1.], [1., 0.], [.99, .01], [0., 1.], [.01, .99],
                                   [1., 0.], [1., 0.], [1., 1.], [1., 1.], [1., 1.]])
        self.output = self.input.clone()

    def build(self, args=None):
        args = args or self.args
        config, _ = stc.request(args, tokenizer=self.tokenizer)
        with redirect_stdout(io.StringIO()):
            return stc.build_mapping(self.input, self.output, self.tokenizer, args, stc.cache_path(args, config), config)

    def test_both_algorithms_are_repeatable_with_numeric_singletons_and_gapped_ids(self):
        for algorithm in ("agglomerative", "minibatch_kmeans"):
            args = args_at(self.root, stc_algorithm=algorithm)
            one = self.build(args)
            two = self.build(args)
            self.assertEqual(one.fingerprint, two.fingerprint)
            mapping = dict(zip(one.arrays["token_ids"], one.arrays["cluster_ids"]))
            self.assertNotIn(0, mapping)
            self.assertNotIn(7, mapping)
            self.assertNotIn(8, mapping)
            self.assertNotIn(9, mapping)
            self.assertNotEqual(mapping[5], mapping[6])
            self.assertNotIn(mapping[5], [mapping[i] for i in (1, 2, 3, 4)])
            self.assertEqual(one.manifest["numeric_dimensions"], 2)
            self.assertEqual(one.manifest["cluster_count"], 4)
            if algorithm == "agglomerative":
                self.assertEqual(mapping[1], mapping[2])
                self.assertEqual(mapping[3], mapping[4])

    def logits(self):
        probabilities = torch.tensor([.02, .17, .17, .30, .01, .15, .10, .03, 0., .05])
        logits = probabilities.clamp_min(1e-30).log()
        logits[8] = 1000  # a padded head row must not enter the softmax denominator
        return logits

    def test_sum_before_topk_uses_raw_logits_and_keeps_original_probability_denominator(self):
        self.build()
        data = fixtures.data_args(sparse_length=1)
        encoder = stc.SparseEncoder(self.tokenizer, data, self.args)
        result = encoder.vector(self.logits()[None])
        # The two .17 synonyms outrank the single .30 token after summation.
        self.assertEqual(result, {"stc_000000": 3400})
        self.assertNotEqual(result, {"stc_000000": 10000})

    def test_view_merge_mean_quantization_and_invalid_logits(self):
        self.build()
        for merge, mean, weight in (("sum", False, 6800), ("max", False, 3400),
                                     ("replace", False, 3400), ("sum", True, 3400)):
            data = fixtures.data_args(sparse_length=1, sparse_value_type=merge, sparse_value_mean=mean)
            result = stc.SparseEncoder(self.tokenizer, data, self.args).vector(self.logits().repeat(2, 1))
            self.assertEqual(result, {"stc_000000": weight})
        encoder = stc.SparseEncoder(self.tokenizer, fixtures.data_args(), self.args)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            encoder.vector(torch.full((1, 10), float("nan")))
        with self.assertRaisesRegex(ValueError, "output-head"):
            encoder.vector(torch.zeros(1, 5))

    def test_bundle_checksum_identity_and_scale_are_locked(self):
        vocabulary = self.build()
        config = stc.configuration(self.args)
        target = self.root / "artifact"
        stc.publish_bundle(target, vocabulary)
        stc.validate_bundle(target, config)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            stc.validate_bundle(target, dict(config, mapping_fingerprint="wrong"))
        other = args_at(self.root, stc_n_clusters=3, stc_vocab_path=str(vocabulary.directory))
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            stc.load(other)
        path = stc.mark_path("result_llava_base_dense_freeret_pre_mlp_freeret_full_prompt_v2", config)
        self.assertEqual(path, stc.mark_path(path, config))
        self.assertTrue(str(path).endswith("_llava_base_dense_freeret_pre_mlp_freeret_full_prompt_v2"))
        self.assertNotEqual(path, stc.mark_path("result_llava_base_dense_freeret_pre_mlp_freeret_full_prompt_v2",
                                              dict(config, weight_scale=100)))
        (target / stc.BUNDLE / "token_to_cluster.npz").write_bytes(b"broken")
        with self.assertRaisesRegex(ValueError, "corrupt"):
            stc.validate_bundle(target, config)

    def test_disabled_mode_does_not_change_namespace_paths_or_configuration(self):
        target = NS(existing="unchanged")
        stc.copy_options(NS(use_stc_sparse=False, stc_algorithm="invalid"), target)
        self.assertEqual(vars(target), {"existing": "unchanged"})
        self.assertIsNone(stc.configuration(target))
        self.assertEqual(stc.mark_path("old_path", None), Path("old_path"))
        with self.assertRaisesRegex(ValueError, "overlap"):
            stc.environment_controls(dict(USE_STC_SPARSE="true", USE_VIEW_OVERLAP_SELECTION="true"))

    def test_unsupported_experiment_entries_fail_instead_of_ignoring_switch(self):
        import adaptive_search_batch
        import adaptive_views_topk_experiment
        settings = adaptive_search_batch.Settings.from_environment("ir", dict(USE_STC_SPARSE="true"))
        self.assertEqual(settings.controls["USE_STC_SPARSE"], "true")
        with self.assertRaisesRegex(ValueError, "Top-K"):
            adaptive_views_topk_experiment.build_jobs(dict(USE_STC_SPARSE="true"))

    def test_cache_hit_never_loads_weights_and_dry_run_does_not_write(self):
        self.build()
        self.args.dry_run = False
        self.args.lora_model_path = None
        with patch("transformers.LlavaNextForConditionalGeneration.from_pretrained", side_effect=AssertionError("weights loaded")):
            with redirect_stdout(io.StringIO()):
                self.assertIsInstance(stc.build(self.args), stc.Vocabulary)
        other = args_at(self.root, stc_n_clusters=3)
        other.dry_run = True
        with redirect_stdout(io.StringIO()):
            self.assertIsNone(stc.build(other))
        self.assertFalse(stc.cache_path(other).exists())

    def test_pipeline_namespace_and_corpus_binding_need_no_external_cache(self):
        self.build()
        api = importlib.import_module("three_stage_retrieval_pipeline_adaptive_views")
        config = stc.configuration(self.args)
        root = self.root / "corpus"
        stc.publish_bundle(root, stc.load(self.args))
        views.atomic_json(root / views.MANIFEST, dict(configuration={stc.FIELD: config}))
        args = api._parser().parse_args(["--adaptive-views", "--dataset-name", "flickr", "--task-type", "ir",
                                        "--query-type", "text", "--use-stc-sparse", "true", "--stc-n-clusters", "2"])
        args.stage1_sparse_index = str(root)
        args.retrieval_model_name_or_path = str(self.root)
        args.retrieval_dtype = "float32"
        args.stc_batch_size, args.stc_max_iter = 8, 15
        stc.bind_corpus(args)
        shutil.rmtree(self.root / "cache")
        model, _ = api._retrieval_namespaces(args, eol_type="disassembleeol_separate", calculate_type="concat", sparse_manual=False)
        self.assertEqual(stc.configuration(model), config)

    def test_exact_index_and_retrieval_batches_locate_stc_for_all_task_groups(self):
        from test_adaptive_sparse_exact import artifacts
        import adaptive_retrieval_batch as retrieval
        import adaptive_sparse_index_batch as indexing
        import dense_representation as dense_mode
        for group in ("ir", "tbpr", "fused"):
            env, rows, loader = artifacts(self.root / group, TASK_GROUP=group)
            env.update(USE_STC_SPARSE="true", STC_N_CLUSTERS="2", STC_BATCH_SIZE="8", STC_MAX_ITER="15",
                       ADAPTIVE_OUTPUT_ROOT=str(self.root / group / "results"))
            settings = retrieval.Settings.from_environment(group, env)
            paths = []
            for previous, selection_path, selection in rows:
                original = json.loads((previous / views.MANIFEST).read_text(encoding="utf-8"))["configuration"]
                model, data, training = retrieval.namespaces(settings, settings.families[0], original["dataset"],
                                                              original["task"], original["kind"])
                config, _ = stc.request(model, tokenizer=self.tokenizer)
                with redirect_stdout(io.StringIO()):
                    vocabulary = stc.build_mapping(self.input, self.output, self.tokenizer, model,
                                                  stc.cache_path(model, config), config)
                dense, sparse = encoding.artifact_dirs(model, data, training, selection)
                identity = loader(str(settings.models[settings.families[0]]))
                completion = dict(encoding.metadata(model, data, training.task_type, training.encode_type, identity, selection),
                                  num_shards=1, selection_file=str(selection_path), selection_fingerprint=selection["fingerprint"])
                for target in (dense, sparse):
                    stc.publish_bundle(target, vocabulary)
                    views.atomic_json(target / views.MANIFEST, completion)
                    dense_mode.write_manifest(target, False)
                (dense / "corpus_0.pkl").write_bytes(b"fixture")
                (sparse / "collection/index").mkdir(parents=True)
                (sparse / "collection/corpus_0.jsonl").write_text('{"id":"a","vector":{"stc_000000":100}}\n', encoding="utf-8")
                (sparse / "collection/index/segments_1").write_bytes(b"fixture")
                paths.append(sparse.resolve())
            with redirect_stdout(io.StringIO()):
                jobs, missing = indexing.discover_exact(indexing.Settings.from_environment(group, env, exact_encoding=True), identity_loader=loader)
            self.assertFalse(missing)
            self.assertEqual([job.root for job in jobs], paths)
            with patch.object(retrieval, "annotation_files", return_value=[]), patch("llava_resolution.validate_checkpoint"):
                jobs, missing = retrieval.prepare_jobs(settings, identity_loader=loader)
            self.assertFalse(missing)
            self.assertEqual(len(jobs), len(paths))
            for job in jobs:
                self.assertEqual(job.variables["USE_STC_SPARSE"], "true")
                self.assertIn("_stc_agglomerative_k2_", str(job.output))
                self.assertIn(stc.FIELD, job.configuration)


class ModelTests(unittest.TestCase):
    def check_model(self, model, processor, records, family, adapter="adaptive"):
        if adapter != "adaptive":
            return
        if family.startswith("llava"):
            processor.chat_template = ("{% for m in messages %}[INST]{% for c in m['content'] %}{% if c['type']=='image' %}<image>\n{% else %}{{ c['text'] }}{% endif %}{% endfor %}[/INST]{% endfor %}"
                if "mistral" in family else
                "{% for m in messages %}<|start_header_id|>user<|end_header_id|>\n\n{% for c in m['content'] %}{% if c['type']=='image' %}<image>\n{% else %}{{ c['text'] }}{% endif %}{% endfor %}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{% endfor %}")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / family
            root.mkdir()
            model.save_pretrained(root)
            processor.tokenizer.save_pretrained(root)
            options = args_at(root)
            options.dry_run, options.lora_model_path = False, None
            with redirect_stdout(io.StringIO()):
                vocabulary = stc.build(options)
            selection = dict(selected=views.view_pool()[:2])
            for mode in ("normal", "constraints", "full"):
                old_args = fixtures.model_args(mode != "normal", str(root))
                old_args.use_freeret_prompt_constraints = mode == "constraints"
                old_args.use_freeret_full_prompt = mode == "full"
                data = fixtures.data_args(sparse_length=2)
                old = encoding.AdaptiveEncoder(model, processor, old_args, data, "ir", selection)
                args = copy.copy(old_args)
                stc.copy_options(options, args, dtype="float32")
                replacement = encoding.AdaptiveEncoder(model, processor, args, data, "ir", selection)
                calls = []
                handle = model.register_forward_pre_hook(lambda _, inputs, kwargs: calls.append(kwargs), with_kwargs=True)
                try:
                    values, dense = old.encode(records)
                    raw, new_dense = replacement.encode(records)
                finally:
                    handle.remove()
                self.assertEqual(len(calls), 2)
                self.assertEqual(calls[0]["attention_mask"].ndim, 4)
                self.assertTrue(torch.equal(calls[0]["attention_mask"], calls[1]["attention_mask"]))
                self.assertEqual(old.last_prompts, replacement.last_prompts)
                self.assertTrue(torch.equal(dense, new_dense), (family, mode))
                for legacy, new in zip(values, raw):
                    self.assertTrue(torch.equal(legacy, torch.log1p(torch.relu(new))), (family, mode))
                vectors = encoding.sparse_vectors(replacement, records, raw)
                self.assertTrue(all(vector and all(key.startswith("stc_") for key in vector) for vector in vectors))
                self.assertEqual(replacement.stc_encoder.vocabulary.fingerprint, vocabulary.fingerprint)

    def test_five_real_tiny_transformers_preserve_prompts_masks_dense_and_use_raw_sparse_logits(self):
        case = fixtures.RealModelTests()
        with patch.object(fixtures.RealModelTests, "compare", side_effect=self.check_model):
            case.test_native_and_adaptive_qwen_three_families_with_text_image_and_fused_inputs()
            case.test_llava_llama_and_mistral_with_native_image_processor()


class ShellTests(unittest.TestCase):
    def test_cpu_precompute_precedes_deepspeed_and_forwards_the_resolved_cache_root(self):
        bash = "D:/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
        if not bash or not Path(bash).is_file():
            self.skipTest("Bash unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture, log = root / "capture.sh", root / "calls.txt"
            capture.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" >> "$CALL_LOG"\n', encoding="utf-8")
            result = subprocess.run([bash, (ROOT / "scripts/encode_adaptive_views_tbpr.sh").as_posix(), "--run-one"],
                cwd=ROOT, env=dict(os.environ, USE_STC_SPARSE="true", STC_ALGORITHM="minibatch_kmeans", STC_N_CLUSTERS="32",
                    MODEL_FAMILY="llava_mistral", MODEL_PATH="fixture", DATASET="RSTPReid", ENCODE_TYPE="image",
                    OUTPUT_DIR="fixture-output", SPARSE_OUTPUT_ROOT=root.as_posix(), STC_CACHE_ROOT="", CALL_LOG=log.as_posix(),
                    PYTHON_BIN=capture.as_posix(), DEEPSPEED_BIN=capture.as_posix(), DRY_RUN="false"),
                text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = log.read_text(encoding="utf-8").splitlines()
            first, second = calls.index("src/stc_sparse.py"), calls.index("src/encode_adaptive_views.py")
            self.assertLess(first, second)
            self.assertIn("build", calls[first:second])
            self.assertEqual(calls.count((root / "stc_vocab").as_posix()), 2)

    def test_configuration_reaches_three_task_dispatchers_and_direct_retrieval(self):
        bash = "D:/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
        if not bash or not Path(bash).is_file():
            self.skipTest("Bash unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            for helper in ("freeret_dense_mode.sh", "stage2_retrieval_mode.sh", "stc_sparse_mode.sh"):
                shutil.copyfile(ROOT / "scripts" / helper, root / "scripts" / helper)
            capture = root / "capture.sh"
            capture.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\nprintf "stc=%s algo=%s k=%s\\n" "$USE_STC_SPARSE" "$STC_ALGORITHM" "$STC_N_CLUSTERS"\n', encoding="utf-8")
            names = [f"{prefix}_{group}.sh" for group in ("ir", "tbpr", "fused") for prefix in
                     ("encode_adaptive_views", "sparse_index_adaptive", "three_stage_retrieval_pipeline_adaptive_views")]
            for name in names:
                shutil.copyfile(ROOT / "scripts" / name, root / "scripts" / name)
                result = subprocess.run([bash, str(root / "scripts" / name)], cwd=root,
                    env=dict(os.environ, USE_STC_SPARSE="true", STC_ALGORITHM="minibatch_kmeans", STC_N_CLUSTERS="32",
                             PYTHON_BIN=capture.as_posix()), text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, (name, result.stderr))
                self.assertIn("stc=true algo=minibatch_kmeans k=32", result.stdout)


if __name__ == "__main__":
    unittest.main()
