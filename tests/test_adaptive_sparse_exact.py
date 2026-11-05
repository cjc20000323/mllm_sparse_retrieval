"""Exact index lookup shares the encoding cache contract and isolates old experiments."""
from contextlib import redirect_stdout
import io
from itertools import product
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import Mock, patch

import test_adaptive_view_shuffle as fixtures
from test_freeret_dense import ROOT, temporary_directory
import adaptive_retrieval_batch as retrieval
import adaptive_sparse_index_batch as batch
import adaptive_view_encoding as encoding
import adaptive_view_shuffle as shuffle
import adaptive_views as views
import dense_representation as dense


def artifacts(directory, **changes):
    """Write encoder-contract artifacts with no dataset, dense corpus or Lucene index."""
    root = Path(directory)
    env = dict(VIEW_EXAMPLE_MODE="fixed", MODEL_FAMILY="llava_mistral", DATASET="CUHK-PEDES", ENCODE_TYPE="image", DATASET_SPLIT="test",
               CHECKPOINT_ROOT=str(root / "models"), DATA_ROOT=str(root / "absent_data"),
               DENSE_OUTPUT_ROOT=str(root / "absent_dense"), SPARSE_OUTPUT_ROOT=str(root / "sparse"),
               DTYPE="float16", SPARSE_LENGTH="30", TBPR_TYPE="type", LLAVA_IMAGE_MODE="base",
               USE_HIERARCHICAL_VIEWS="true", USE_VIEW_PMI="false", USE_VIEW_OVERLAP_SELECTION="false",
               VIEW_SELECTION_ROUNDS="2", VIEW_TEXTS_PER_ROUND="2", VIEW_IMAGES_PER_ROUND="1",
               VIEW_OVERLAP_SAMPLES="2", VIEW_SELECTED_COUNT="5", VIEW_OVERLAP_THRESHOLD="0.1", VIEW_OVERLAP_SEED="42",
               SHUFFLE_VIEW_OPTIONS="false", VIEW_SELECTION_RUN_ID="", DRY_RUN="true")
    group = changes.get("TASK_GROUP", "tbpr")
    if group != "tbpr":
        env.update(MODEL_FAMILY="qwen25", DATASET="flickr" if group == "ir" else "webqa",
                   ENCODE_TYPE="", ENCODE_TYPES="image text" if group == "ir" else "corpus",
                   SPARSE_LENGTH="40", TBPR_TYPE="origin_type", LLAVA_IMAGE_MODE="anyres", USE_HIERARCHICAL_VIEWS="false")
    env.update(changes)
    settings = retrieval.Settings.from_environment(group, env, project=ROOT)
    kinds = (env.get("ENCODE_TYPE") or env.get("ENCODE_TYPES", "image")).split()
    records, identities = [], {}
    for family in settings.families:
        path = settings.models[family]
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps(dict(model_type=batch.FAMILIES[family])), encoding="utf-8")
        identity = dict(family=batch.FAMILIES[family], checkpoint=dict(path=str(path)))
        identities[str(path)] = identity
        for dataset, kind in product(settings.datasets, kinds):
            task = retrieval.job_types(settings, dataset)[0][0]
            model, data, training = retrieval.namespaces(settings, family, dataset, task, kind)
            selection_path, _, strategy = retrieval.selection_location(settings, model, data, training, identity)
            if selection_path.is_file():
                selection = views.load_selection(selection_path)
            elif strategy:
                selection = fixtures.overlap_choose(settings.options, model, data, task, identity, dataset)
            else:
                selection = fixtures.choose(settings.options, identity=identity, dataset=dataset)
            views.atomic_json(selection_path, selection)
            sparse = encoding.artifact_dirs(model, data, training, selection)[1].resolve()
            manifest = dict(encoding.metadata(model, data, task, kind, identity, selection), num_shards=1,
                            selection_file=str(selection_path), selection_fingerprint=selection["fingerprint"])
            views.atomic_json(sparse / views.MANIFEST, manifest)
            dense.write_manifest(sparse, env.get("USE_FREERET_DENSE") == "true",
                                 env.get("USE_FREERET_PROMPT_CONSTRAINTS") == "true",
                                 env.get("USE_FREERET_FULL_PROMPT") == "true")
            (sparse / "collection").mkdir()
            (sparse / "collection/corpus_0.jsonl").write_text('{"id":"a","vector":{"word":10}}\n', encoding="utf-8")
            records.append((sparse, selection_path, selection))
    loader = Mock(side_effect=lambda path, **kwargs: identities[path])
    return env, records, loader


class ExactIndexTests(unittest.TestCase):
    def discover(self, env, loader):
        settings = batch.Settings.from_environment(env.get("TASK_GROUP", "tbpr"), env, exact_encoding=True)
        with redirect_stdout(io.StringIO()):
            return batch.discover_exact(settings, identity_loader=loader)

    def test_modes_use_real_locked_fingerprints_without_dense_or_index_inputs(self):
        for hierarchical, pmi, overlap, shuffled in product((False, True), repeat=4):
            for use_dense, constraints in ((False, False), (True, False), (True, True)):
                with self.subTest(hierarchy=hierarchical, pmi=pmi, overlap=overlap, shuffle=shuffled,
                                  dense=use_dense, constraints=constraints), temporary_directory() as directory:
                    env, records, loader = artifacts(directory, USE_HIERARCHICAL_VIEWS=str(hierarchical).lower(),
                        USE_VIEW_PMI=str(pmi).lower(), USE_VIEW_OVERLAP_SELECTION=str(overlap).lower(),
                        SHUFFLE_VIEW_OPTIONS=str(shuffled).lower(), VIEW_SELECTION_RUN_ID=fixtures.RUN if shuffled else "",
                        USE_FREERET_DENSE=str(use_dense).lower(), USE_FREERET_PROMPT_CONSTRAINTS=str(constraints).lower())
                    with patch.object(shuffle, "new_run_id", side_effect=AssertionError("index must not create a run ID")):
                        jobs, missing = self.discover(env, loader)
                    self.assertFalse(missing)
                    self.assertEqual([job.root for job in jobs], [row[0] for row in records])
                    self.assertEqual(jobs[0].fingerprint, records[0][2]["views_fingerprint"])
                    self.assertIsNone(jobs[0].existing_index)
                    self.assertIn("_llava_base", jobs[0].root.name)
                    self.assertFalse(Path(env["DENSE_OUTPUT_ROOT"]).exists())
                    self.assertFalse(Path(env["DATA_ROOT"]).exists())
                    loader.assert_called_once_with(loader.call_args.args[0], dtype="float16", reps_loc="after_pad",
                                                   min_pixels=0, max_pixels=0, image_mode="base")

    def test_ir_and_fused_resolve_all_models_datasets_and_candidate_kinds(self):
        for group in ("ir", "fused"):
            for all_modes in (False, True):
                with self.subTest(group=group, modes=all_modes), temporary_directory() as directory:
                    enabled = str(all_modes).lower()
                    env, records, loader = artifacts(directory, TASK_GROUP=group,
                        MODEL_FAMILY="", MODEL_FAMILIES="qwen25 llava_mistral", DATASET="",
                        DATASETS=" ".join(batch.GROUPS[group]["datasets"]),
                        USE_HIERARCHICAL_VIEWS=enabled, USE_VIEW_PMI=enabled, USE_VIEW_OVERLAP_SELECTION=enabled,
                        SHUFFLE_VIEW_OPTIONS=enabled, VIEW_SELECTION_RUN_ID=fixtures.RUN if all_modes else "",
                        USE_FREERET_DENSE=enabled, USE_FREERET_PROMPT_CONSTRAINTS=enabled)
                    jobs, missing = self.discover(env, loader)
                    expected = set(product(("qwen25", "llava_mistral"), batch.GROUPS[group]["datasets"], batch.GROUPS[group]["kinds"]))
                    self.assertEqual({job.key for job in jobs}, expected)
                    self.assertEqual({job.root for job in jobs}, {row[0] for row in records})
                    self.assertFalse(missing)
                    self.assertEqual(loader.call_count, 2)
                    for job in jobs:
                        configuration = json.loads((job.root / views.MANIFEST).read_text(encoding="utf-8"))["configuration"]
                        self.assertEqual(configuration["task"], batch.GROUPS[group]["datasets"][job.dataset])
                        self.assertEqual(configuration["kind"], job.kind)
                    if group == "ir":
                        self.assertEqual(len({row[1] for row in records}), len(records) // 2)

    def test_ir_fused_isolate_old_experiments_and_support_manual_and_dry_run(self):
        for group in ("ir", "fused"):
            with self.subTest(group=group), temporary_directory() as directory:
                _, old, _ = artifacts(directory, TASK_GROUP=group, USE_HIERARCHICAL_VIEWS="true")
                old[0][1].unlink()
                env, records, loader = artifacts(directory, TASK_GROUP=group)
                jobs, _ = self.discover(env, loader)
                for job in jobs:
                    manual = batch.Settings.from_environment(group, dict(env, ENCODE_TYPE=job.kind, SPARSE_OUTPUT_ROOT=str(job.root)))
                    manual_jobs, missing = batch.discover(manual)
                    self.assertEqual(manual_jobs, [job])
                    self.assertFalse(missing)
                before = {str(p): p.read_bytes() for p in Path(directory).rglob("*") if p.is_file()}
                builder = Mock()
                with redirect_stdout(io.StringIO()) as output:
                    result = batch.run(batch.Settings.from_environment(group, env, exact_encoding=True),
                                       identity_loader=loader, builder=builder)
                self.assertEqual(result, 0)
                self.assertIn(f"planned={len(records)}", output.getvalue())
                builder.assert_not_called()
                self.assertEqual(before, {str(p): p.read_bytes() for p in Path(directory).rglob("*") if p.is_file()})
                with self.assertRaisesRegex(ValueError, "original VIEW_SELECTION_RUN_ID"):
                    self.discover(dict(env, SHUFFLE_VIEW_OPTIONS="true"), loader)
                with self.assertRaisesRegex(ValueError, "Missing exact selection cache"):
                    self.discover(dict(env, USE_VIEW_PMI="true"), loader)

    def test_broken_legacy_corpus_cannot_block_exact_lookup_or_manual_target(self):
        with temporary_directory() as directory:
            old_env, old, _ = artifacts(directory, USE_HIERARCHICAL_VIEWS="false")
            old[0][1].unlink()
            env, records, loader = artifacts(directory)
            with self.assertRaisesRegex(ValueError, "Missing selection file"):
                batch.discover(batch.Settings.from_environment("tbpr", old_env))
            jobs, _ = self.discover(env, loader)
            self.assertEqual(jobs[0].root, records[0][0])
            manual = batch.Settings.from_environment("tbpr", dict(env, SPARSE_OUTPUT_ROOT=str(records[0][0])))
            manual_jobs, missing = batch.discover(manual)
            self.assertFalse(missing)
            self.assertEqual(manual_jobs, jobs)

    def test_dry_run_is_read_only_and_existing_indexes_are_skipped(self):
        with temporary_directory() as directory:
            env, records, loader = artifacts(directory)
            root = Path(directory)
            before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            builder = Mock()
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(batch.run(batch.Settings.from_environment("tbpr", env, exact_encoding=True),
                                           identity_loader=loader, builder=builder), 0)
            builder.assert_not_called()
            self.assertIn("planned=1", output.getvalue())
            self.assertEqual(before, {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()})
            env["DRY_RUN"] = "false"
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(batch.run(batch.Settings.from_environment("tbpr", env, exact_encoding=True),
                                           identity_loader=loader, builder=builder), 0)
            builder.assert_called_once_with(records[0][0], threads=16)
            self.assertIn("built=1", output.getvalue())
            builder.reset_mock()
            index = records[0][0] / "collection/index"
            index.mkdir()
            (index / "segments_1").touch()
            env["DRY_RUN"] = "false"
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(batch.run(batch.Settings.from_environment("tbpr", env, exact_encoding=True),
                                           identity_loader=loader, builder=builder), 0)
            builder.assert_not_called()
            self.assertIn("skipped_existing=1", output.getvalue())

    def test_batch_preflight_checks_all_exact_combinations_before_building(self):
        with temporary_directory() as directory:
            env, records, loader = artifacts(directory, MODEL_FAMILY="", MODEL_FAMILIES="qwen25 llava_mistral",
                                             DATASET="", DATASETS="CUHK-PEDES RSTPReid", LLAVA_IMAGE_MODE="anyres",
                                             DRY_RUN="false")
            jobs, _ = self.discover(env, loader)
            self.assertEqual(len(jobs), 4)
            self.assertEqual(loader.call_count, 2)
            builder = Mock()
            records[-1][1].unlink()
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "Missing exact selection cache"):
                batch.run(batch.Settings.from_environment("tbpr", env, exact_encoding=True), identity_loader=loader, builder=builder)
            builder.assert_not_called()

    def test_sparse_encoding_mismatch_does_not_reuse_an_existing_view_directory(self):
        with temporary_directory() as directory:
            env, records, loader = artifacts(directory)
            with self.assertRaisesRegex(ValueError, "Missing adaptive completion manifest"):
                self.discover(dict(env, SPARSE_LENGTH="31"), loader)
            path = records[0][0] / views.MANIFEST
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["configuration"]["encoding"]["sparse_length"] = 31
            manifest["fingerprint"] = views.digest(manifest["configuration"])
            views.atomic_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "Adaptive corpus.*mismatch"):
                self.discover(env, loader)

    def test_missing_cache_reports_expected_path_and_never_chooses_another_mode(self):
        with temporary_directory() as directory:
            env, records, loader = artifacts(directory)
            for changed in (dict(USE_VIEW_PMI="true"), dict(USE_HIERARCHICAL_VIEWS="false"),
                            dict(VIEW_TEXTS_PER_ROUND="3"), dict(DATA_ROOT=str(Path(directory) / "other"))):
                with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "Missing exact selection cache.*encoding settings="):
                    self.discover(dict(env, **changed), loader)
            records[0][1].unlink()
            with self.assertRaises(ValueError) as error:
                self.discover(env, loader)
            self.assertIn(str(records[0][1]), str(error.exception))
            self.assertIn('"hierarchical": true', str(error.exception))

    def test_shuffle_requires_original_id_and_full_manifest_link_is_validated(self):
        with temporary_directory() as directory:
            env, records, loader = artifacts(directory)
            with self.assertRaisesRegex(ValueError, "original VIEW_SELECTION_RUN_ID"):
                self.discover(dict(env, SHUFFLE_VIEW_OPTIONS="true"), loader)
            loader.assert_not_called()
            root, _, _ = records[0]
            path = root / views.MANIFEST
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["selection_fingerprint"] = "0" * 64
            views.atomic_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "manifest/selection fingerprint"):
                self.discover(env, loader)

    def test_exact_manifest_still_recovers_missing_original_reference_by_full_hash(self):
        with temporary_directory() as directory:
            env, records, loader = artifacts(directory)
            root, _, _ = records[0]
            manifest = json.loads((root / views.MANIFEST).read_text(encoding="utf-8"))
            manifest["selection_file"] = str(Path(directory) / "old_missing_reference.json")
            views.atomic_json(root / views.MANIFEST, manifest)
            jobs, _ = self.discover(env, loader)
            self.assertEqual(jobs[0].root, root)


class LauncherTests(unittest.TestCase):
    def test_ir_fused_defaults_match_encoding_and_manual_override_preserves_flags(self):
        bash = "D:/Git/bin/bash.exe" if Path("D:/Git/bin/bash.exe").is_file() else shutil.which("bash")
        if not bash:
            self.skipTest("Bash not available")
        shared = ("MODEL_FAMILIES MODEL_FAMILY DATASETS DATASET ENCODE_TYPES ENCODE_TYPE DATASET_SPLIT CHECKPOINT_ROOT "
                  "MODEL_PATH MODEL_PATH_QWEN2 MODEL_PATH_QWEN25 MODEL_PATH_QWEN3 MODEL_PATH_LLAVA_LLAMA MODEL_PATH_LLAVA_MISTRAL "
                  "DATA_ROOT SPARSE_OUTPUT_ROOT DENSE_OUTPUT_ROOT DTYPE REPS_LOC SPARSE_LENGTH TEXT_SPARSE_LENGTH IMAGE_SPARSE_LENGTH "
                  "SPARSE_VALUE_TYPE SPARSE_LOWER_OR_UPPER SPARSE_VALUE_MEAN IS_FILTERED NUM_EXPENDED_TOKENS TBPR_TYPE "
                  "LLAVA_IMAGE_MODE QWEN_MIN_PIXELS QWEN_MAX_PIXELS VIEW_SELECTION_SPLIT VIEW_SELECTION_MODALITY "
                  "VIEW_SELECTION_ROUNDS VIEW_TEXTS_PER_ROUND VIEW_IMAGES_PER_ROUND VIEW_SELECTION_SEED USE_VIEW_PMI "
                  "USE_HIERARCHICAL_VIEWS SHUFFLE_VIEW_OPTIONS USE_FREERET_DENSE USE_FREERET_PROMPT_CONSTRAINTS")
        names = shared + " VIEW_SELECTION_RUN_ID USE_VIEW_OVERLAP_SELECTION VIEW_OVERLAP_THRESHOLD THREADS DRY_RUN"
        for group in ("ir", "fused"):
            with self.subTest(group=group), temporary_directory() as directory:
                root = Path(directory)
                capture, args, values = (root / name for name in ("capture", "args", "values"))
                capture.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n'
                    'for name in ' + names + '; do printf "%s=%s\\n" "$name" "${!name}"; done > "$CAPTURE_VALUES"\n', encoding="utf-8")
                capture.chmod(0o755)
                env = {k: v for k, v in os.environ.items() if k not in names.split() and k not in
                       {"TARGET_SPARSE_DIR", "PYTHON_BIN", "VIEW_SELECTION_FILE", "_ADAPTIVE_SHUFFLE_RUN_ID"}}
                env.update(PYTHON_BIN=capture.as_posix(), CAPTURE_ARGS=args.as_posix(), CAPTURE_VALUES=values.as_posix())
                command = [bash, (ROOT / f"scripts/sparse_index_adaptive_{group}.sh").as_posix()]
                result = subprocess.run(command, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(args.read_text().splitlines(), ["-B", "src/adaptive_sparse_index_batch.py", "--task-group", group, "--exact-encoding"])
                actual = dict(line.split("=", 1) for line in values.read_text().splitlines())
                encode = [bash, (ROOT / f"scripts/encode_adaptive_views_{group}.sh").as_posix(), "--help"]
                result = subprocess.run(encode, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                encoded = dict(line.split("=", 1) for line in values.read_text().splitlines())
                encoded["USE_VIEW_PMI"] = encoded["USE_VIEW_PMI"] or "false"
                encoded["VIEW_SELECTION_MODALITY"] = encoded["VIEW_SELECTION_MODALITY"] or "both"
                self.assertEqual({key: actual[key] for key in shared.split()}, {key: encoded[key] for key in shared.split()})
                target = root / "manual"
                target.mkdir()
                (target / views.MANIFEST).touch()
                env.update(TARGET_SPARSE_DIR=target.as_posix(), SPARSE_OUTPUT_ROOT="/ignored_parent", MODEL_FAMILY="llava_mistral",
                           DATASET="flickr" if group == "ir" else "remuq", ENCODE_TYPE="text" if group == "ir" else "corpus",
                           DRY_RUN="true", USE_FREERET_DENSE="true", USE_FREERET_PROMPT_CONSTRAINTS="true",
                           USE_VIEW_PMI="true", USE_HIERARCHICAL_VIEWS="true", USE_VIEW_OVERLAP_SELECTION="true",
                           VIEW_OVERLAP_THRESHOLD="0.25", SHUFFLE_VIEW_OPTIONS="true", VIEW_SELECTION_RUN_ID=fixtures.RUN)
                result = subprocess.run(command, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("--exact-encoding", args.read_text())
                manual = dict(line.split("=", 1) for line in values.read_text().splitlines())
                for key in ("MODEL_FAMILY", "DATASET", "ENCODE_TYPE", "DRY_RUN", "USE_FREERET_DENSE", "USE_FREERET_PROMPT_CONSTRAINTS",
                            "USE_VIEW_PMI", "USE_HIERARCHICAL_VIEWS", "USE_VIEW_OVERLAP_SELECTION", "VIEW_OVERLAP_THRESHOLD", "VIEW_SELECTION_RUN_ID"):
                    self.assertEqual(manual[key], env[key])
                self.assertEqual(manual["SPARSE_OUTPUT_ROOT"], target.as_posix())

    def test_tbpr_default_auto_mode_and_manual_override_forward_config(self):
        bash = shutil.which("bash")
        if os.name == "nt" and Path("D:/Git/bin/bash.exe").is_file():
            bash = "D:/Git/bin/bash.exe"
        if not bash:
            self.skipTest("Bash not available")
        with temporary_directory() as directory:
            root = Path(directory)
            capture = root / "capture"
            args, values = root / "args", root / "values"
            names = ("SPARSE_OUTPUT_ROOT MODEL_FAMILIES DATASETS ENCODE_TYPES DATASET_SPLIT SPARSE_LENGTH TBPR_TYPE "
                     "LLAVA_IMAGE_MODE USE_HIERARCHICAL_VIEWS USE_VIEW_PMI SHUFFLE_VIEW_OPTIONS VIEW_SELECTION_RUN_ID "
                     "USE_FREERET_DENSE USE_FREERET_PROMPT_CONSTRAINTS VIEW_TEXTS_PER_ROUND VIEW_IMAGES_PER_ROUND")
            capture.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n'
                               'for name in ' + names + '; do printf "%s=%s\\n" "$name" "${!name}"; done > "$CAPTURE_VALUES"\n', encoding="utf-8")
            capture.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if k not in names.split() and k not in
                   {"MODEL_FAMILY", "DATASET", "ENCODE_TYPE", "TARGET_SPARSE_DIR", "PYTHON_BIN"}}
            env.update(PYTHON_BIN=capture.as_posix(), CAPTURE_ARGS=args.as_posix(), CAPTURE_VALUES=values.as_posix())
            command = [bash, (ROOT / "scripts/sparse_index_adaptive_tbpr.sh").as_posix()]
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--exact-encoding", args.read_text())
            actual = dict(line.split("=", 1) for line in values.read_text().splitlines())
            for name, expected in dict(MODEL_FAMILIES="llava_mistral", DATASETS="CUHK-PEDES", ENCODE_TYPES="image",
                SPARSE_LENGTH="30", TBPR_TYPE="type", LLAVA_IMAGE_MODE="base", USE_HIERARCHICAL_VIEWS="true",
                USE_VIEW_PMI="false", USE_FREERET_DENSE="false", USE_FREERET_PROMPT_CONSTRAINTS="false",
                VIEW_TEXTS_PER_ROUND="8", VIEW_IMAGES_PER_ROUND="2").items():
                self.assertEqual(actual[name], expected)
            target = root / "manual"
            target.mkdir()
            (target / views.MANIFEST).touch()
            env.update(TARGET_SPARSE_DIR=target.as_posix(), SPARSE_OUTPUT_ROOT="/ignored_parent")
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("--exact-encoding", args.read_text())
            self.assertIn("SPARSE_OUTPUT_ROOT=" + target.as_posix(), values.read_text())


if __name__ == "__main__":
    unittest.main()
