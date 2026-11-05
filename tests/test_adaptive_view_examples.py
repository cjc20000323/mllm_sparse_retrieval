"""Automatic context packing, inference reuse, cache isolation and launcher wiring."""
from contextlib import redirect_stdout
import copy
import importlib
import io
from itertools import product
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_freeret_dense import ROOT, temporary_directory
import test_adaptive_view_shuffle as fixtures
import adaptive_views as views
import adaptive_view_examples as examples
import adaptive_view_encoding as encoding
import adaptive_view_overlap as overlap
import adaptive_retrieval_batch as retrieval
import adaptive_sparse_index_batch as indexer
import dense_representation as dense


class Processor:
    def __init__(self):
        self.tokenizer = NS(all_special_ids=[0], encode=lambda value, **kwargs: [ord(value) - 64])
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        return "<user>" + "".join("<image>" if row["type"] == "image" else row["text"]
                                    for row in messages[0]["content"]) + "<assistant>"

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        # Include the entire template/options/examples plus expanded visual positions.
        length = len(kwargs["text"][0].encode("utf-8")) // 5 + 1
        length += sum(image["visual_tokens"] for image in kwargs.get("images", []))
        return dict(input_ids=torch.ones((1, length), dtype=torch.long), attention_mask=torch.ones((1, length), dtype=torch.long))


class Model:
    def __init__(self, family="qwen2_5_vl", limit=4096):
        text = NS(max_position_embeddings=limit, model_type=family.removeprefix("llava_"))
        self.config = NS(model_type="llava_next" if family.startswith("llava_") else family, text_config=text)
        self.lengths = []

    def get_input_embeddings(self):
        return NS(weight=torch.zeros(1))

    def forward(self, input_ids, attention_mask, logits_to_keep=None, **kwargs):
        self.lengths.append(int(attention_mask.sum()))
        length = input_ids.shape[1] if logits_to_keep is None else len(logits_to_keep)
        return NS(logits=torch.arange(32, dtype=torch.float32)[None, None, :].expand(1, length, -1))

    __call__ = forward


TEXTS = [dict(id=f"t{i}", text=("A person in a red coat walks beside a bicycle. " * (1 + i % 3))) for i in range(20)]
IMAGES = [dict(id=f"i{i}", image=dict(visual_tokens=80 + 8 * (i % 3))) for i in range(20)]


def options(**changes):
    values = dict(adaptive_views=True, view_example_mode="auto", view_example_max_tokens=1050,
                  view_selection_modality="both", view_selection_rounds=2, view_texts_per_round=0, view_images_per_round=0,
                  view_selection_seed=42, view_selection_pmi=False, view_selection_hierarchical=False,
                  view_selection_shuffle=False, view_overlap_selection=False, view_selected_count=5,
                  view_overlap_samples=2, view_overlap_threshold=0.1, view_overlap_seed=42)
    values.update(changes)
    return overlap.OverlapViewArguments(**values)


def choose(model, processor, opts, *, identity=None, dataset="flickr", strategy=None):
    identity = identity or dict(family=views.family_of(model.config), checkpoint=dict(path="fixture"))
    with patch("qwen_vl_encoding.read_image", side_effect=lambda value: value), redirect_stdout(io.StringIO()):
        if strategy is not None:
            with patch.object(overlap, "SparseViewProbe", side_effect=lambda *args: lambda record, perspective: {perspective["label"].lower(): 10}):
                return strategy.choose(model, processor, identity, dataset, "test", TEXTS, IMAGES, opts,
                                       report_path=Path("unused-auto-diagnostics.json"))
        return views.choose_views(model, processor, identity, dataset, "test", TEXTS, IMAGES, opts)


class PackingTests(unittest.TestCase):
    def test_cli_defaults_and_explicit_fixed_mode_parse_for_both_entrypoints(self):
        from transformers import HfArgumentParser
        for arguments in (views.AdaptiveViewArguments, overlap.OverlapViewArguments):
            with self.subTest(arguments=arguments.__name__):
                parser = HfArgumentParser(arguments)
                automatic = parser.parse_args_into_dataclasses([])[0]
                self.assertEqual(automatic.view_example_mode, "auto")
                self.assertEqual(automatic.view_example_max_tokens, 0)
                fixed = parser.parse_args_into_dataclasses(["--view_example_mode", "fixed", "--view_example_max_tokens", "8192"])[0]
                self.assertEqual(fixed.view_example_mode, "fixed")
                self.assertEqual(fixed.view_example_max_tokens, 8192)

    def test_exact_fit_and_next_overflow_use_real_prepared_lengths_for_all_families(self):
        for family in ("qwen2_vl", "qwen2_5_vl", "qwen3_vl", "llava_llama", "llava_mistral"):
            for modality, items in (("text", TEXTS), ("image", IMAGES)):
                with self.subTest(family=family, modality=modality), patch("qwen_vl_encoding.read_image", side_effect=lambda value: value):
                    model, processor, pool = Model(family), Processor(), views.view_pool()
                    prepare = lambda group: views.prepare_choice_group(processor, family, pool, group, modality)
                    fitting = prepare(items[:3])["tokens"]
                    policy = examples.configuration(options(view_example_max_tokens=fitting + 1), model.config)
                    processor.calls.clear()
                    group, prepared, details = examples.pack(items, prepare, policy, modality)
                    self.assertEqual(len(group), 3)
                    self.assertEqual(prepared["tokens"], fitting)
                    self.assertEqual(details["input_limit"], fitting)
                    self.assertGreater(details["next_input_tokens"], fitting)
                    self.assertEqual(details["next_sample_id"], items[3]["id"])
                    self.assertEqual(details["stop_reason"], "context_limit")
                    self.assertLessEqual(len(processor.calls), 2 * math.ceil(math.log2(len(items))) + 3)
                    calls = len(processor.calls)
                    views.score_group(model, processor, family, pool, group, modality,
                                      prepared=prepared, input_limit=policy["input_limit"])
                    self.assertEqual(len(processor.calls), calls)
                    self.assertEqual(model.lengths, [fitting])
                    self.assertTrue(all(row["truncation"] is False for row in processor.calls))
                    if modality == "image":
                        self.assertGreaterEqual(fitting, sum(row["image"]["visual_tokens"] for row in group))

    def test_model_budget_pool_exhaustion_and_oversized_first_example(self):
        processor, model, pool = Processor(), Model(limit=4096), views.view_pool()
        prepare = lambda group: views.prepare_choice_group(processor, "qwen2_5_vl", pool, group, "text")
        automatic = examples.configuration(options(view_example_max_tokens=0), model.config)
        self.assertEqual(automatic["input_limit"], 4095)
        self.assertEqual(examples.configuration(options(view_example_max_tokens=99999), model.config)["input_limit"], 4095)
        group, _, details = examples.pack(TEXTS[:2], prepare, automatic, "text")
        self.assertEqual(len(group), 2)
        self.assertEqual(details["stop_reason"], "samples_exhausted")
        self.assertIsNone(details["next_input_tokens"])
        base = prepare([])["tokens"]
        with self.assertRaisesRegex(ValueError, "base prompt"):
            examples.pack(TEXTS, prepare, examples.configuration(options(view_example_max_tokens=base), model.config), "text")
        with self.assertRaisesRegex(ValueError, "First text example id='t0'"):
            examples.pack(TEXTS, prepare, examples.configuration(options(view_example_max_tokens=base + 1), model.config), "text")
        with self.assertRaisesRegex(ValueError, "max_position_embeddings"):
            examples.configuration(options(), NS())

    def test_rounds_are_repeatable_and_probabilities_cover_modes_without_extra_forwards(self):
        for pmi, hierarchical, shuffled, use_overlap in product((False, True), repeat=4):
            with self.subTest(pmi=pmi, hierarchical=hierarchical, shuffle=shuffled, overlap=use_overlap), temporary_directory() as directory:
                opts = options(view_selection_pmi=pmi, view_selection_hierarchical=hierarchical,
                               view_selection_shuffle=shuffled, view_selection_shuffle_run_id=fixtures.RUN if shuffled else None,
                               view_overlap_selection=use_overlap)
                model, processor = Model(), Processor()
                strategy = overlap.OverlapSelectionStrategy(opts, fixtures.model_args(), fixtures.data_args(), "ir") if use_overlap else None
                selected = choose(model, processor, opts, strategy=strategy)
                self.assertEqual(len(model.lengths), 4 + int(pmi))
                self.assertTrue(all(length <= selected[examples.FIELD]["input_limit"] for length in model.lengths))
                again = choose(Model(), Processor(), opts, strategy=strategy)
                self.assertEqual(selected, again)
                for modality in ("text", "image"):
                    source = TEXTS if modality == "text" else IMAGES
                    orders = list(examples.ordered_groups(source, 2, 42, modality))
                    for index, row in enumerate(selected["rounds"]):
                        self.assertEqual(row[modality + "_ids"], [item["id"] for item in orders[index][:len(row[modality + "_ids"])]])
                        self.assertGreater(len(row[modality + "_ids"]), 0)
                if pmi:
                    self.assertNotIn("Text examples:", selected["calibration"]["prior_prompt"])
                    self.assertNotIn("Image examples:", selected["calibration"]["prior_prompt"])
                if use_overlap:
                    self.assertEqual(len(selected["overlap_samples"]), 4)
                path = Path(directory) / "selection.json"
                views.atomic_json(path, selected)
                self.assertEqual(views.load_selection(path), selected)
                self.assertNotEqual(selected["views_fingerprint"], views.views_fingerprint(selected["selected"],
                    pmi=pmi, shuffle=selected.get(views.option_shuffle.FIELD), pool_mode=views.pool_mode(selected)))

    def test_text_or_image_mode_does_not_touch_the_disabled_modality(self):
        for modality in ("text", "image"):
            with self.subTest(modality=modality):
                model = Model()
                selected = choose(model, Processor(), options(view_selection_modality=modality))
                self.assertEqual(len(model.lengths), 2)
                other = "image" if modality == "text" else "text"
                self.assertTrue(all(row[other + "_tokens"] is None and not row[other + "_ids"] for row in selected["rounds"]))

    def test_fixed_mode_preserves_original_sampling_and_hash_and_auto_counts_are_ignored(self):
        opts = options(view_example_mode="fixed", view_texts_per_round=2, view_images_per_round=1)
        selected = choose(Model(), Processor(), opts)
        self.assertNotIn(examples.FIELD, selected)
        self.assertEqual(selected["sampling"], dict(rounds=2, seed=42, texts_per_round=2, images_per_round=1))
        self.assertEqual(selected["views_fingerprint"], views.views_fingerprint(selected["selected"]))
        sampling = views.selection_sampling(options(view_texts_per_round=1), model_config=Model().config)
        self.assertEqual(sampling, views.selection_sampling(options(view_texts_per_round=999), model_config=Model().config))
        with temporary_directory() as directory:
            path = Path(directory) / "old.json"
            views.atomic_json(path, selected)
            self.assertEqual(views.load_selection(path, view_example_mode="fixed"), selected)
            with self.assertRaisesRegex(ValueError, "VIEW_EXAMPLE_MODE=fixed"):
                views.load_selection(path, view_example_mode="auto")

    def test_invalid_round_budget_and_stopping_diagnostics_are_rejected(self):
        selection = choose(Model(), Processor(), options())
        for corrupt in ("tokens", "stop"):
            with self.subTest(corrupt=corrupt), temporary_directory() as directory:
                bad = copy.deepcopy(selection)
                if corrupt == "tokens":
                    bad["rounds"][0]["text_tokens"] = bad[examples.FIELD]["input_limit"] + 1
                else:
                    bad["rounds"][0]["text_examples"]["stop_reason"] = "arbitrary"
                bad.pop("fingerprint")
                bad["fingerprint"] = views.digest(bad)
                path = Path(directory) / "invalid.json"
                views.atomic_json(path, bad)
                with self.assertRaisesRegex(ValueError, "automatic view-example"):
                    views.load_selection(path)


class DownstreamTests(unittest.TestCase):
    def test_auto_sampling_keys_preserve_freeret_dense_invariance_and_overlap_prompt_dependency(self):
        with temporary_directory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text(json.dumps(dict(model_type="qwen2_5_vl", max_position_embeddings=4096,
                                                                    text_config=dict(max_position_embeddings=4096))), encoding="utf-8")
            for group in ("ir", "tbpr", "fused"):
                for use_overlap, pmi in product((False, True), repeat=2):
                    with self.subTest(group=group, overlap=use_overlap, pmi=pmi):
                        env = dict(MODEL_FAMILY="qwen25", MODEL_PATH=str(checkpoint), VIEW_EXAMPLE_MODE="auto",
                                   VIEW_EXAMPLE_MAX_TOKENS="1050", USE_VIEW_OVERLAP_SELECTION=str(use_overlap).lower(),
                                   USE_VIEW_PMI=str(pmi).lower())
                        original = retrieval.Settings.from_environment(group, env, project=root)
                        freer = retrieval.Settings.from_environment(group, dict(env, USE_FREERET_DENSE="true"), project=root)
                        constrained = retrieval.Settings.from_environment(group, dict(env, USE_FREERET_DENSE="true", USE_FREERET_PROMPT_CONSTRAINTS="true"), project=root)
                        for dataset in original.datasets:
                            for task, _, kind in retrieval.job_types(original, dataset):
                                locations = []
                                for settings in (original, freer, constrained):
                                    arguments = retrieval.namespaces(settings, "qwen25", dataset, task, kind)
                                    locations.append(retrieval.selection_location(settings, *arguments, fixtures.IDENTITY)[:2])
                                self.assertEqual(locations[0], locations[1])
                                self.assertEqual(locations[1] == locations[2], not use_overlap)

    def test_encoder_writes_and_reuses_auto_cache_and_separates_changed_budget(self):
        api = importlib.import_module("three_stage_retrieval_pipeline_adaptive_views")
        with temporary_directory() as directory:
            root = Path(directory)
            model, processor = Model(), Processor()
            data = fixtures.data_args(dense_output_dir=str(root / "dense"), sparse_output_dir=str(root / "sparse"))
            def write(_encoder, _records, _data, roots, rank):
                (roots[1] / "collection/corpus_0.jsonl").write_text('{"id":"a","vector":{"word":10}}\n', encoding="utf-8")
                with (roots[0] / "corpus_0.pkl").open("wb") as stream:
                    pickle.dump((np.zeros((1, 8)), ["a"]), stream)
            def run(kind, budget=1050):
                opts = options(view_example_max_tokens=budget)
                opts.adaptive_data_root = directory
                training = NS(task_type="ir", encode_type=kind, output_dir=str(root / "logs"), fp16=True, bf16=False)
                return encoding.run_encode(fixtures.model_args(), data, training, opts)
            with patch("qwen_vl_pipeline.init_process", return_value=(0, 1, torch.device("cpu"))), \
                 patch("qwen_vl_encoding.read_image", side_effect=lambda value: value), \
                 patch.object(api, "_load_retrieval_runtime", return_value=NS(encoder=model, processor=processor)), \
                 patch.object(api, "_release_retrieval_runtime"), patch.object(api, "build_dataset", return_value=NS(img_id_list=[], text_id_list=[], text_dict={})), \
                 patch.object(encoding, "configure_processor"), patch.object(views, "model_identity", return_value=fixtures.IDENTITY), \
                 patch.object(encoding, "source_pools", return_value=(TEXTS, IMAGES)), \
                 patch.object(encoding, "AdaptiveEncoder", side_effect=lambda *args: NS(model=args[0], last_prompts=[])), \
                 patch.object(encoding, "write_shard", side_effect=write), redirect_stdout(io.StringIO()):
                first, selection = run("image")
                self.assertEqual(len(model.lengths), 4)
                _, reused = run("text")
                self.assertEqual(selection, reused)
                self.assertEqual(len(model.lengths), 4)
                second, changed = run("image", budget=1049)
                self.assertEqual(len(model.lengths), 8)
                self.assertNotEqual(first, second)
                self.assertNotEqual(selection["views_fingerprint"], changed["views_fingerprint"])
            paths = list((root / "sparse/view_selection").rglob("view_selection.json"))
            self.assertEqual(len(paths), 2)
            for path in paths:
                self.assertEqual(views.load_selection(path)[examples.FIELD]["mode"], "auto")

    def test_auto_encoding_cache_matches_index_and_retrieval_for_each_task(self):
        for group, dataset in (("ir", "flickr"), ("tbpr", "RSTPReid"), ("fused", "remuq")):
            for all_modes in (False, True):
                with self.subTest(group=group, modes=all_modes), temporary_directory() as directory:
                    root = Path(directory)
                    env = dict(MODEL_FAMILY="qwen25", DATASET=dataset, CHECKPOINT_ROOT=str(root / "models"),
                               DATA_ROOT=str(root), SPARSE_OUTPUT_ROOT=str(root / "sparse"), DENSE_OUTPUT_ROOT=str(root / "dense"),
                               DATASET_SPLIT="test", VIEW_EXAMPLE_MODE="auto", VIEW_EXAMPLE_MAX_TOKENS="1050",
                               VIEW_SELECTION_ROUNDS="2", VIEW_TEXTS_PER_ROUND="0", VIEW_IMAGES_PER_ROUND="0", VIEW_SELECTION_SEED="42",
                               VIEW_OVERLAP_SAMPLES="2", VIEW_SELECTED_COUNT="5", VIEW_OVERLAP_THRESHOLD="0.1", VIEW_OVERLAP_SEED="42",
                               VIEW_SELECTION_RUN_ID=fixtures.RUN if all_modes else "", DRY_RUN="true")
                    for name in ("USE_HIERARCHICAL_VIEWS", "USE_VIEW_PMI", "USE_VIEW_OVERLAP_SELECTION",
                                 "SHUFFLE_VIEW_OPTIONS", "USE_FREERET_DENSE", "USE_FREERET_PROMPT_CONSTRAINTS"):
                        env[name] = str(all_modes).lower()
                    settings = retrieval.Settings.from_environment(group, env, project=ROOT)
                    model_path = settings.models["qwen25"]
                    model_path.mkdir(parents=True)
                    (model_path / "config.json").write_text(json.dumps(dict(model_type="qwen2_5_vl", max_position_embeddings=4096,
                                                                           text_config=dict(max_position_embeddings=4096))), encoding="utf-8")
                    identity = dict(family="qwen2_5_vl", checkpoint=dict(path=str(model_path)))
                    for path in fixtures.encode_batch.annotation_files(root, dataset, "test"):
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.touch()
                    sparse_paths = []
                    for task, _, kind in retrieval.job_types(settings, dataset):
                        model_args, data, training = retrieval.namespaces(settings, "qwen25", dataset, task, kind)
                        path, key, strategy = retrieval.selection_location(settings, model_args, data, training, identity)
                        opts = settings.options
                        selection = choose(Model(), Processor(), opts, identity=identity, dataset=dataset, strategy=strategy)
                        self.assertEqual(selection["sampling"], key["sampling"])
                        encode_key = views.selection_cache_key(identity, dataset, "test", "both", selection["sampling"], root,
                            pmi=all_modes, strategy=strategy.configuration if strategy else None,
                            shuffle=selection.get(views.option_shuffle.FIELD), pool_mode=views.pool_mode(selection))
                        self.assertEqual(views.selection_cache_path(root / "sparse", str(model_path), dataset, encode_key), path)
                        views.atomic_json(path, selection)
                        manifest = dict(encoding.metadata(model_args, data, task, kind, identity, selection), num_shards=1,
                                        selection_file=str(path), selection_fingerprint=selection["fingerprint"])
                        dense_root, sparse_root = encoding.artifact_dirs(model_args, data, training, selection)
                        for artifact in (dense_root, sparse_root):
                            views.atomic_json(artifact / views.MANIFEST, manifest)
                            dense.write_manifest(artifact, all_modes, all_modes)
                        with (dense_root / "corpus_0.pkl").open("wb") as stream:
                            pickle.dump((np.zeros((1, 8)), ["a"]), stream)
                        (sparse_root / "collection/index").mkdir(parents=True)
                        (sparse_root / "collection/index/segments_1").touch()
                        (sparse_root / "collection/corpus_0.jsonl").write_text('{"id":"a","vector":{"word":10}}\n', encoding="utf-8")
                        sparse_paths.append(sparse_root.resolve())
                    with redirect_stdout(io.StringIO()):
                        indexed, missing = indexer.discover_exact(indexer.Settings.from_environment(group, env, exact_encoding=True),
                                                                 identity_loader=lambda *args, **kwargs: identity)
                        jobs, absent = retrieval.prepare_jobs(settings, identity_loader=lambda *args, **kwargs: identity)
                    self.assertFalse(missing)
                    self.assertFalse(absent)
                    self.assertEqual({job.root for job in indexed}, set(sparse_paths))
                    self.assertEqual({Path(job.variables["SPARSE_CORPUS_DIR"]) for job in jobs}, set(sparse_paths))
                    for job in jobs:
                        self.assertEqual(job.variables["VIEW_EXAMPLE_MODE"], "auto")
                        self.assertEqual(job.variables["VIEW_EXAMPLE_MAX_TOKENS"], "1050")
                        self.assertEqual(job.configuration[examples.FIELD]["input_limit"], 1049)
                    different = dict(env, VIEW_EXAMPLE_MAX_TOKENS="1049")
                    with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "Missing exact selection cache"):
                        indexer.discover_exact(indexer.Settings.from_environment(group, different, exact_encoding=True),
                                               identity_loader=lambda *args, **kwargs: identity)

    def test_all_encoding_and_automatic_lookup_scripts_forward_local_and_external_controls(self):
        bash = "D:/Git/bin/bash.exe" if Path("D:/Git/bin/bash.exe").is_file() else shutil.which("bash")
        if not bash:
            self.skipTest("Bash unavailable")
        names = [f"encode_adaptive_views{overlap}{suffix}.sh" for overlap in ("", "_overlap") for suffix in ("", "_ir", "_tbpr", "_fused")]
        names += [f"sparse_index_adaptive_{group}.sh" for group in ("ir", "tbpr", "fused")]
        names += [f"three_stage_adaptive_{group}_multi.sh" for group in ("ir", "tbpr", "fused")]
        names += [f"three_stage_retrieval_pipeline_adaptive_views{overlap}_{group}{multi}.sh"
                  for overlap, multi in (("", ""), ("_overlap", "_multi")) for group in ("ir", "tbpr", "fused")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "scripts"
            scripts.mkdir()
            shutil.copyfile(ROOT / "scripts/freeret_dense_mode.sh", scripts / "freeret_dense_mode.sh")
            capture = root / "capture"
            capture.write_text('#!/usr/bin/env bash\nprintf "mode=%s\\nbudget=%s\\n" "$VIEW_EXAMPLE_MODE" "$VIEW_EXAMPLE_MAX_TOKENS"\nprintf "%s\\n" "$@"\n', encoding="utf-8")
            capture.chmod(0o755)
            for name in names:
                for local in (False, True):
                    with self.subTest(script=name, local=local):
                        body = (ROOT / "scripts" / name).read_text(encoding="utf-8")
                        if local:
                            body = body.replace('${VIEW_EXAMPLE_MODE-auto}', '${VIEW_EXAMPLE_MODE-fixed}')
                            body = body.replace('${VIEW_EXAMPLE_MAX_TOKENS-0}', '${VIEW_EXAMPLE_MAX_TOKENS-2048}')
                        (scripts / name).write_text(body, encoding="utf-8")
                        env = {k: v for k, v in os.environ.items() if k not in {"VIEW_EXAMPLE_MODE", "VIEW_EXAMPLE_MAX_TOKENS", "TARGET_SPARSE_DIR"}}
                        env.update(PYTHON_BIN=capture.as_posix(), DEEPSPEED_BIN=capture.as_posix(), MODEL_FAMILY="qwen25", MODEL_PATH="fixture",
                                   DATASET="flickr", ENCODE_TYPE="image", LLAVA_IMAGE_MODE="anyres", DRY_RUN="false", OUTPUT_DIR="logs")
                        if not local:
                            env.update(VIEW_EXAMPLE_MODE="fixed", VIEW_EXAMPLE_MAX_TOKENS="2048")
                        command = [bash, (scripts / name).as_posix()]
                        if name.startswith("encode_") and name not in {"encode_adaptive_views.sh", "encode_adaptive_views_overlap.sh"}:
                            command.append("--run-one")
                        result = subprocess.run(command, env=env, cwd=root, capture_output=True, text=True, timeout=30)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        lines = result.stdout.splitlines()
                        if name.startswith("encode_"):
                            self.assertEqual(lines[lines.index("--view_example_mode") + 1], "fixed")
                            self.assertEqual(lines[lines.index("--view_example_max_tokens") + 1], "2048")
                        else:
                            self.assertIn("mode=fixed", lines)
                            self.assertIn("budget=2048", lines)


if __name__ == "__main__":
    unittest.main()
