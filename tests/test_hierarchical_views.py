"""Hierarchical pool selection, artifact compatibility and script control."""
import ast
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

import test_adaptive_view_shuffle as fixtures
from test_adaptive_view_shuffle import (views, encoding, overlap, shuffle, encode_batch,
    retrieval_batch, index_batch, ROOT, IDENTITY, RUN, TEXTS, IMAGES,
    data_args, model_args, temporary_directory, options, choose, overlap_choose)


class PoolTests(unittest.TestCase):
    def test_catalog_and_final_instructions_match_the_supplied_dimensions(self):
        expected = [
            "scene, such as location, indoor or outdoor setting and terrain",
            "natural environment, such as weather, lighting, season and time",
            "people, such as number of people, appearance and pose",
            "clothing, such as category, color, material and style",
            "objects, such as category, number, appearance and state",
            "actions, such as behavior, participants, tools and targets",
            "relations, such as spatial relations, interactions and ownership",
            "visual attributes, such as color, shape, size and material",
            "spatial layout, such as foreground and background, distance and direction",
            "high-level semantics, such as events, activities and functions",
        ]
        pool = views.view_pool(views.HIERARCHICAL_POOL)
        self.assertEqual([row["label"] for row in pool], list("ABCDEFGHIJ"))
        self.assertEqual([row["aspect"] for row in pool], expected)
        self.assertEqual(len(views.view_pool()), 20)
        for record, modality in ((dict(text="description"), "sentence"), (dict(image="photo"), "image"),
                                 (dict(text="question", image="photo"), "image and sentence")):
            self.assertEqual(encoding.instructions(pool, record),
                [f"Summary the {aspect} in above {modality} in one word: " for aspect in expected])
        selection = choose(options(hierarchical=True, shuffled=False))
        self.assertEqual(selection["task_views"], {})
        self.assertEqual(selection[views.POOL_MODE_FIELD], views.HIERARCHICAL_POOL)

    def test_both_entrypoints_parse_the_switch_and_require_adaptive_mode(self):
        from transformers import HfArgumentParser
        for kind in (views.AdaptiveViewArguments, overlap.OverlapViewArguments):
            self.assertFalse(HfArgumentParser(kind).parse_args_into_dataclasses([])[0].view_selection_hierarchical)
            self.assertTrue(HfArgumentParser(kind).parse_args_into_dataclasses(
                ["--view_selection_hierarchical", "true"])[0].view_selection_hierarchical)
        for filename in ("encode_adaptive_views.py", "encode_adaptive_views_overlap.py"):
            tree = ast.parse((ROOT / "src" / filename).read_text(encoding="utf-8"))
            main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
            body = ast.get_source_segment((ROOT / "src" / filename).read_text(encoding="utf-8"), main)
            self.assertIn("--view_selection_hierarchical requires --adaptive_views", body)

    def test_candidate_count_limit_is_ten_only_in_the_new_mode(self):
        values = dict(vars(options(hierarchical=True)), view_selected_count=11)
        with self.assertRaisesRegex(ValueError, r"\[1,10\]"):
            overlap.OverlapSelectionStrategy(overlap.OverlapViewArguments(**values), model_args(), data_args(), "ir")
        for group in ("ir", "tbpr", "fused"):
            with self.assertRaisesRegex(ValueError, "VIEW_SELECTED_COUNT"):
                encode_batch.Settings.from_environment(group, dict(USE_HIERARCHICAL_VIEWS="true", VIEW_SELECTED_COUNT="11"), encoder="overlap")
        values["view_selection_hierarchical"] = False
        strategy = overlap.OverlapSelectionStrategy(overlap.OverlapViewArguments(**values), model_args(), data_args(), "ir")
        self.assertEqual(strategy.configuration["selected_count"], 11)
        with self.assertRaisesRegex(ValueError, "USE_HIERARCHICAL_VIEWS"):
            encode_batch.encoding_controls(dict(USE_HIERARCHICAL_VIEWS="maybe"))


class SelectionTests(unittest.TestCase):
    def test_probabilities_pmi_and_overlap_roundtrip_with_both_shuffle_modes(self):
        for pmi in (False, True):
            for shuffled in (False, True):
                for use_overlap in (False, True):
                    with self.subTest(pmi=pmi, shuffled=shuffled, overlap=use_overlap), temporary_directory() as directory:
                        opts = options(hierarchical=True, pmi=pmi, shuffled=shuffled)
                        result = (overlap_choose(opts, model_args(), data_args(), "ir", IDENTITY, "flickr")
                                  if use_overlap else choose(opts))
                        self.assertEqual(len(result["pool"]), 10)
                        self.assertEqual(len(result["selected"]), 5)
                        self.assertEqual(len(result["scores"]), 10)
                        self.assertAlmostEqual(sum(result["scores"]), 1)
                        self.assertEqual(views.is_pmi_selection(result), pmi)
                        self.assertEqual(views.is_overlap_selection(result), use_overlap)
                        if shuffled:
                            self.assertEqual([row["display_label"] for row in result[shuffle.FIELD]["mapping"]], list("ABCDEFGHIJ"))
                        if pmi:
                            prior = result["calibration"]
                            self.assertEqual(len(prior["prior_probabilities"]), 10)
                            self.assertNotIn("Text examples:", prior["prior_prompt"])
                            self.assertIn("J. ", prior["prior_prompt"])
                            self.assertNotIn("K. ", prior["prior_prompt"])
                            for row in result["rounds"]:
                                for modality in ("text", "image"):
                                    logp = np.log(row[modality + "_raw_probabilities"])
                                    expected, _ = views.pmi_probabilities(logp, prior["prior_log_probabilities"])
                                    np.testing.assert_allclose(row[modality + "_probabilities"], expected)
                        path = Path(directory) / "view_selection.json"
                        views.atomic_json(path, result)
                        self.assertEqual(views.load_selection(path, view_selection_hierarchical=True), result)
                        self.assertEqual(views.load_selection(path), result)
                        with self.assertRaisesRegex(ValueError, "View pool mode mismatch"):
                            views.load_selection(path, view_selection_hierarchical=False)

    def test_mode_tampering_and_reusing_an_original_file_are_rejected(self):
        with temporary_directory() as directory:
            path = Path(directory) / "selection.json"
            original = choose(options(shuffled=False))
            views.atomic_json(path, original)
            with self.assertRaisesRegex(ValueError, "View pool mode mismatch"):
                views.load_selection(path, view_selection_hierarchical=True)
            value = choose(options(hierarchical=True, shuffled=False))
            for corruption in ("unknown_mode", "missing_mode", "bad_pool"):
                bad = copy.deepcopy(value)
                if corruption == "unknown_mode": bad[views.POOL_MODE_FIELD] = "future_v2"
                elif corruption == "missing_mode": bad.pop(views.POOL_MODE_FIELD)
                else: bad["pool"][0]["aspect"] = "incorrect description"
                bad["fingerprint"] = views.digest({k: v for k, v in bad.items() if k != "fingerprint"})
                views.atomic_json(path, bad)
                with self.assertRaises(ValueError): views.load_selection(path)

    def test_mode_is_in_all_fingerprints_and_old_payloads_remain_unchanged(self):
        sampling = dict(rounds=2, seed=42, texts_per_round=2, images_per_round=1)
        old = views.selection_cache_key(IDENTITY, "flickr", "test", "both", sampling, "data")
        self.assertEqual(old, dict(model=IDENTITY, dataset="flickr", split="test", pool=views.view_pool(),
                                  selection_modality="both", sampling=sampling, data_root=str(Path("data").resolve())))
        new = views.selection_cache_key(IDENTITY, "flickr", "test", "both", sampling, "data", pool_mode=views.HIERARCHICAL_POOL)
        self.assertNotEqual(views.digest(old), views.digest(new))
        selected = [dict(label=row["label"], aspect=row["aspect"]) for row in views.view_pool()[:5]]
        self.assertEqual(views.views_fingerprint(selected), views.digest(dict(version=views.VERSION, selection_modality="both", selected=selected)))
        self.assertNotEqual(views.views_fingerprint(selected), views.views_fingerprint(selected, pool_mode=views.HIERARCHICAL_POOL))
        old_selection = choose(options(shuffled=False))
        new_selection = choose(options(hierarchical=True, shuffled=False))
        old_config = encoding.encoding_configuration(model_args(), data_args(), "ir", "text", IDENTITY, old_selection)
        new_config = encoding.encoding_configuration(model_args(), data_args(), "ir", "text", IDENTITY, new_selection)
        self.assertNotIn(views.POOL_MODE_FIELD, old_config)
        self.assertEqual(new_config[views.POOL_MODE_FIELD], views.HIERARCHICAL_POOL)


class DownstreamTests(unittest.TestCase):
    def test_encode_driver_shares_the_ten_option_cache_only_within_the_same_run(self):
        fixtures.DriverTests().test_encoder_reuses_only_current_batch_mapping_and_new_launch_has_new_paths(hierarchical=True)

    def test_index_retrieval_and_cache_recovery_accept_new_pool_for_every_task(self):
        builder = fixtures.DownstreamTests()
        for group in ("ir", "tbpr", "fused"):
            for pmi in (False, True):
                for use_overlap in (False, True):
                    with self.subTest(group=group, pmi=pmi, overlap=use_overlap), temporary_directory() as directory:
                        env, settings, files = builder.build(directory, group, pmi, use_overlap, hierarchical=True)
                        jobs, missing = retrieval_batch.prepare_jobs(settings, identity_loader=lambda *a, **k: IDENTITY)
                        self.assertFalse(missing); self.assertTrue(jobs)
                        indexed, missing = index_batch.discover(index_batch.Settings.from_environment(group, env))
                        self.assertFalse(missing); self.assertEqual(len(indexed), len(jobs))
                        for job in jobs:
                            self.assertEqual(job.configuration[views.POOL_MODE_FIELD], views.HIERARCHICAL_POOL)
                            self.assertEqual(job.variables["USE_HIERARCHICAL_VIEWS"], "true")
                            self.assertNotIn("USE_HIERARCHICAL_VIEWS", job.configuration["parameters"])
                        if pmi and use_overlap:
                            for i, path in enumerate(files):
                                backup = Path(directory) / "backups" / str(i) / "view_selection.json"
                                backup.parent.mkdir(parents=True); backup.write_bytes(path.read_bytes()); path.unlink()
                            recovered = retrieval_batch.Settings.from_environment(group,
                                dict(env, VIEW_SELECTION_ROOT=str(Path(directory) / "backups")), project=Path(directory),
                                recover_missing_selection=True)
                            with redirect_stdout(io.StringIO()):
                                recovered_jobs, missing = retrieval_batch.prepare_jobs(recovered, identity_loader=lambda *a, **k: IDENTITY)
                            self.assertFalse(missing); self.assertEqual(len(recovered_jobs), len(jobs))

    def test_validation_file_can_use_a_different_pool_from_the_main_experiment(self):
        fixture = fixtures.AlphaValidationTests()
        for main_hierarchical, val_hierarchical in ((True, False), (False, True), (True, True)):
            with self.subTest(main=main_hierarchical, validation=val_hierarchical), temporary_directory() as directory:
                api, args = fixture.setup_experiments(directory,
                    main_hierarchical=main_hierarchical, val_hierarchical=val_hierarchical)
                before = api._qwen_encoding_settings(args, "disassembleeol_separate", "concat", False)
                validation = fixture.validate(api, args)
                self.assertEqual(validation.output_dir, args.output_dir)
                self.assertEqual(api._qwen_encoding_settings(args, "disassembleeol_separate", "concat", False), before)
                self.assertEqual(bool(views.pool_mode(views.load_selection(validation.view_selection_file))), val_hierarchical)


class LauncherTests(unittest.TestCase):
    def test_local_and_environment_switches_reach_all_encoding_and_batch_retrieval_entries(self):
        bash = shutil.which("bash") if os.name != "nt" else "D:/Git/bin/bash.exe"
        if not bash or not Path(bash).is_file(): self.skipTest("Bash unavailable")
        encoders = ["encode_adaptive_views" + overlap + suffix + ".sh"
                    for overlap in ("", "_overlap") for suffix in ("", "_ir", "_tbpr", "_fused")]
        retrieval = [f"three_stage_retrieval_pipeline_adaptive_views{overlap}_{group}{multi}.sh"
                     for overlap, multi in (("", ""), ("_overlap", "_multi")) for group in ("ir", "tbpr", "fused")]
        retrieval += [f"three_stage_adaptive_{group}_multi.sh" for group in ("ir", "tbpr", "fused")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); scripts = root / "scripts"; scripts.mkdir()
            shutil.copyfile(ROOT / "scripts/freeret_dense_mode.sh", scripts / "freeret_dense_mode.sh")
            capture = root / "capture.sh"
            capture.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@"\nprintf \'hierarchical=%s\\n\' "${USE_HIERARCHICAL_VIEWS-unset}"\n')
            capture.chmod(0o755)
            for name in encoders + retrieval:
                for local in (False, True):
                    with self.subTest(script=name, local=local):
                        body = (ROOT / "scripts" / name).read_text(encoding="utf-8")
                        if local: body = body.replace('${USE_HIERARCHICAL_VIEWS-false}', '${USE_HIERARCHICAL_VIEWS-true}')
                        path = scripts / name; path.write_text(body, encoding="utf-8")
                        env = {k: v for k, v in os.environ.items() if k != "USE_HIERARCHICAL_VIEWS"}
                        env.update(PYTHON_BIN=capture.as_posix(), DEEPSPEED_BIN=capture.as_posix(), LLAVA_IMAGE_MODE="anyres",
                                   DRY_RUN="false", MODEL_FAMILY="qwen25", MODEL_PATH="fixture", DATASET="flickr",
                                   ENCODE_TYPE="image", OUTPUT_DIR="logs")
                        if not local: env["USE_HIERARCHICAL_VIEWS"] = "true"
                        result = subprocess.run([bash, path.as_posix()], cwd=root, env=env,
                                                capture_output=True, text=True, timeout=30)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        lines = result.stdout.splitlines()
                        if name in ("encode_adaptive_views.sh", "encode_adaptive_views_overlap.sh"):
                            self.assertEqual(lines[lines.index("--view_selection_hierarchical") + 1], "true")
                        else: self.assertIn("hierarchical=true", lines)


if __name__ == "__main__":
    unittest.main()
