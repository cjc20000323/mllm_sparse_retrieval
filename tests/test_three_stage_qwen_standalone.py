"""Exercise the standalone Bash launcher using real paths and a recording runner."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import unittest

import test_qwen_retrieval_launchers as artifacts
import test_three_stage_launcher_paths as datasets
import test_three_stage_retrieval_pipeline as core


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "three_stage_qwen.sh"


@unittest.skipUnless(artifacts.BASH, "Bash is required for launcher integration")
class StandaloneQwenTests(unittest.TestCase):
    def setUp(self):
        self.fixture = artifacts.QwenRetrievalLauncherTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        # No other launcher/helper exists in this test project.
        for path in (self.root / "scripts").iterdir():
            path.unlink()
        self.script = self.root / "scripts" / SCRIPT
        shutil.copyfile(ROOT / "scripts" / SCRIPT, self.script)
        (self.root / "src/three_stage_retrieval_pipeline.py").touch()
        self.record = self.fixture.record
        self.env = self.fixture.env.copy()
        keep = {key: self.env[key] for key in (
            "QWEN_CHECKPOINT_ROOT", "DENSE_OUTPUT_ROOT", "SPARSE_OUTPUT_ROOT", "QWEN_OUTPUT_ROOT", "DEEPSPEED_BIN",
        )}
        names = set(re.findall(r"\$\{([A-Z_][A-Z_0-9]*)(?::-|-)", self.script.read_text(encoding="utf-8")))
        for name in names:
            self.env.pop(name, None)
        self.env.update(keep, DATA_ROOT=self.root.as_posix())
        self.fixture.env = self.env

    def prepare(self, dataset="flickr", *, layout="nested", **settings):
        if "RETRIEVAL_REPS_LOC" in settings:
            settings["REPS_LOC"] = settings["RETRIEVAL_REPS_LOC"]
        dense, sparse = self.fixture.prepare_corpus(dataset, layout=layout, **settings)
        split = settings.get("DATASET_SPLIT", "test")
        directory, files = datasets.PROFILES[dataset][-2:]
        annotation = self.root / "data" / directory
        annotation.mkdir(parents=True, exist_ok=True)
        for filename in files:
            (annotation / filename.format(split=split)).touch()
        return dense, sparse

    def tables(self, **tables):
        source = self.script.read_text(encoding="utf-8")
        for name, values in tables.items():
            declaration = "declare -A " + name + "=(" + " ".join(
                f"[{key}]={shlex.quote(Path(value).as_posix() if isinstance(value, Path) else str(value))}"
                for key, value in values.items()) + ")"
            original = f"declare -A {name}=()"
            self.assertIn(original, source)
            source = source.replace(original, declaration)
        self.script.write_text(source, encoding="utf-8", newline="\n")

    def launch(self, *args, **settings):
        self.record.unlink(missing_ok=True)
        result = subprocess.run(
            [artifacts.BASH, self.script.as_posix(), *args], cwd=self.root.parent,
            env={**self.env, **settings}, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30,
        )
        calls = [row.decode("utf-8").rstrip("\0").split("\0")
                 for row in self.record.read_bytes().splitlines()] if self.record.exists() else []
        parsed = []
        for call in calls:
            self.assertEqual(call[1:5], ["--num_gpus", settings.get("GPUS", "1"), "--num_nodes", settings.get("NUM_NODES", "1")])
            start = call.index("src/three_stage_retrieval_pipeline.py")
            parsed.append(core.pipeline._parser().parse_args(call[start + 1:]))
        return result, parsed

    def success(self, **settings):
        result, parsed = self.launch(**settings)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result, parsed

    def rejected(self, **settings):
        result, parsed = self.launch(**settings)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertFalse(parsed)
        return result

    def test_syntax_help_and_independence(self):
        syntax = subprocess.run([artifacts.BASH, "-n", self.script.as_posix()], capture_output=True, timeout=30)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        for flag in ("-h", "--help"):
            result, parsed = self.launch(flag, QWEN_FAMILY="bad", DEEPSPEED_BIN="missing")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(parsed)
            self.assertIn("HYBRID_ALPHA=0.5", result.stdout)
        self.prepare()
        self.success()
        self.assertEqual([path.name for path in (self.root / "scripts").iterdir()], [SCRIPT])

    def test_defaults_fixed_alpha_and_matching_corpus_paths(self):
        dense, sparse = self.prepare()
        result, (args,) = self.success()
        self.assertEqual((args.dataset_name, args.dataset_split, args.task_type, args.query_type), ("flickr", "test", "ir", "image"))
        self.assertEqual((args.hybrid_alpha, args.select_hybrid_alpha), (0.5, False))
        self.assertEqual((args.retrieval_dtype, args.ranker_dtype), ("float16", "float16"))
        self.assertEqual((args.stage1_method, args.stage1_depth, args.stage2_depth), ("sparse_concat", 1000, 1000))
        self.assertEqual((args.rerank_type, args.rerank_template, args.rerank_num), ("pointwise", "what_caption_generation", 5))
        self.assertEqual(args.stage2_dense_index, dense.as_posix())
        self.assertEqual(args.stage2_sparse_index, (sparse / "collection").as_posix())
        self.assertEqual(args.stage1_sparse_index, args.stage2_sparse_index)
        self.assertTrue(args.output_dir.endswith("/image_to_text/three_stage/alpha_0.5"))
        self.assertIn("Results: " + args.output_dir, result.stdout)
        self.assertIn((sparse / "collection/index").as_posix(), result.stdout)
        self.assertFalse(args.overwrite)

    def test_all_families_and_ir_directions(self):
        for family, (checkpoint, _) in artifacts.CHECKPOINTS.items():
            for direction in ("image", "text"):
                settings = dict(QWEN_FAMILY=family, IR_QUERY_TYPE=direction)
                with self.subTest(**settings):
                    dense, _ = self.prepare(**settings)
                    _, (args,) = self.success(**settings)
                    self.assertEqual(Path(args.retrieval_model_name_or_path).name, checkpoint)
                    self.assertEqual(args.ranker_model_name_or_path, args.retrieval_model_name_or_path)
                    self.assertEqual(args.query_type, direction)
                    self.assertEqual(args.stage2_dense_index, dense.as_posix())

    def test_batch_all_dataset_mappings_have_distinct_outputs(self):
        names = list(datasets.PROFILES)
        for name in names:
            self.prepare(name)
        result, jobs = self.success(DATASETS="\t" + "\n ".join(names))
        self.assertEqual([job.dataset_name for job in jobs], names)
        self.assertEqual(len({job.output_dir for job in jobs}), len(names))
        for job in jobs:
            self.assertEqual((job.task_type, job.query_type), datasets.PROFILES[job.dataset_name][:2])
        self.assertIn("Completed all 8 dataset jobs", result.stdout)

    def test_explicit_dataset_precedes_batch_and_model_paths_with_spaces(self):
        model = self.root / "my models/Qwen-Qwen2.5-custom"
        ranker = self.root / "other models/ranker"
        self.fixture.checkpoint(model)
        self.fixture.checkpoint(ranker, "qwen3_vl")
        settings = dict(QWEN_MODEL_PATH=model.as_posix() + "///", RANKER_MODEL_PATH=ranker.as_posix(), DATASET="coco", DATASETS="flickr coco")
        dense, _ = self.prepare("coco", **{key: val for key, val in settings.items() if key != "DATASET"})
        _, (args,) = self.success(**settings)
        self.assertEqual(args.dataset_name, "coco")
        self.assertEqual(args.retrieval_model_name_or_path, model.as_posix())
        self.assertEqual(args.ranker_model_name_or_path, ranker.as_posix())
        self.assertEqual(args.stage2_dense_index, dense.as_posix())

    def test_encoding_configuration_is_forwarded_and_in_path(self):
        settings = dict(SPARSE_LENGTH="30", TEXT_SPARSE_LENGTH="31", IMAGE_SPARSE_LENGTH="32",
                        IS_FILTERED="yes", SPARSE_VALUE_MEAN="1", SPARSE_VALUE_TYPE="max", SPARSE_LOWER_OR_UPPER="upper",
                        NUM_EXPENDED_TOKENS="2", PROMPT_TYPE="prompt_random_5", RANDOM_VIEW_SEED="123",
                        RETRIEVAL_REPS_LOC="before_pad", DTYPE="bfloat16", RANKER_DTYPE="float32",
                        QWEN_MIN_PIXELS="64", QWEN_MAX_PIXELS="1024", RANKER_QWEN_MAX_PIXELS="512",
                        GPUS="2", NUM_NODES="2", RETRIEVAL_BATCH_SIZE="3", RERANK_NUM="20", RERANK_BATCH_SIZE="4",
                        STAGE1_DEPTH="200", STAGE2_DEPTH="100", THREADS="8")
        dense, _ = self.prepare(**settings)
        _, (args,) = self.success(**settings)
        self.assertEqual(args.stage2_dense_index, dense.as_posix())
        self.assertEqual((args.sparse_length, args.text_sparse_length, args.image_sparse_length), (30, 31, 32))
        self.assertTrue(args.is_filtered and args.sparse_value_mean)
        self.assertEqual((args.random_view_seed, args.retrieval_reps_loc), (123, "before_pad"))
        self.assertEqual((args.retrieval_dtype, args.ranker_dtype), ("bfloat16", "float32"))
        self.assertEqual((args.retrieval_qwen_min_pixels, args.retrieval_qwen_max_pixels, args.ranker_qwen_max_pixels), (64, 1024, 512))
        self.assertEqual((args.rerank_num, args.rerank_batch_size, args.retrieval_batch_size), (20, 4, 3))
        self.assertEqual((args.stage1_depth, args.stage2_depth, args.sparse_threads), (200, 100, 8))

    def test_tbpr_view_type_and_non_ir_query_mapping(self):
        self.prepare("CUHK-PEDES", TBPR_TYPE="type")
        _, (args,) = self.success(DATASET="CUHK-PEDES", TBPR_TYPE="type", IR_QUERY_TYPE="text")
        self.assertEqual((args.task_type, args.query_type, args.tbpr_type), ("tbpr", "text", "type"))
        self.assertIn("/test/type/", args.stage2_dense_index)

    def test_index_aliases_and_legacy_fallback_match_python_contract(self):
        dense, sparse = self.prepare(layout="legacy")
        for nested in (False, True):
            if nested:
                self.prepare(layout="nested")
            expected = sparse / ("collection/index" if nested else "index")
            for suffix in ("", "/", "/collection", "/collection///", "/index", "/collection/index", "/collection/index///"):
                # Explicit index aliases select that exact directory.
                if not nested and "/collection/index" in suffix:
                    continue
                actual = sparse / "index" if suffix.rstrip("/") == "/index" else expected
                result, _ = self.success(STAGE2_DENSE_INDEX=dense.as_posix(), STAGE2_SPARSE_INDEX=sparse.as_posix() + suffix)
                self.assertIn("Stage-2 Lucene index: " + actual.as_posix(), result.stdout)

    def test_incomplete_nested_index_rejects_legacy_fallback(self):
        _, sparse = self.prepare(layout="legacy")
        (sparse / "collection/index").mkdir()
        result = self.rejected()
        self.assertIn((sparse / "collection/index").as_posix(), result.stderr)

    def test_single_input_and_output_overrides_and_overwrite(self):
        dense, sparse = self.prepare()
        output = (self.root / "my results/exact").as_posix()
        output_path = Path(output)
        output_path.mkdir(parents=True)
        (output_path / "keep.txt").write_text("keep")
        _, (args,) = self.success(STAGE2_DENSE_INDEX=(dense / "corpus_0.pkl").as_posix(),
                                  STAGE2_SPARSE_INDEX=(sparse / "collection/index").as_posix(),
                                  STAGE1_SPARSE_INDEX=sparse.as_posix(), OUTPUT_DIR=output, OVERWRITE="true")
        self.assertEqual(args.output_dir, output)
        self.assertTrue(args.overwrite)
        self.assertEqual((output_path / "keep.txt").read_text(), "keep")

    def test_batch_input_tables_and_explicit_output_guard(self):
        d1, s1 = self.prepare("flickr", DENSE_OUTPUT_ROOT=(self.root / "custom dense").as_posix(),
                              SPARSE_OUTPUT_ROOT=(self.root / "custom sparse").as_posix())
        d2, s2 = self.prepare("coco")
        self.tables(DENSE_INDEXES={"flickr": d1}, SPARSE_INDEXES={"flickr": s1 / "collection/index"})
        _, jobs = self.success(DATASETS="flickr coco")
        self.assertEqual([job.stage2_dense_index for job in jobs], [d1.as_posix(), d2.as_posix()])
        self.assertEqual(jobs[0].stage2_sparse_index, (s1 / "collection/index").as_posix())
        self.assertEqual(jobs[1].stage2_sparse_index, (s2 / "collection").as_posix())
        for setting in ({"OUTPUT_DIR": "/shared"}, {"STAGE2_DENSE_INDEX": d1.as_posix()}, {"STAGE1_SPARSE_INDEX": s1.as_posix()}):
            self.rejected(DATASETS="flickr coco", **setting)

    def test_auto_alpha_target_does_not_autodetect_existing_val(self):
        self.prepare()
        self.prepare(DATASET_SPLIT="val")
        result, (args,) = self.success(SELECT_HYBRID_ALPHA="true")
        self.assertTrue(args.select_hybrid_alpha)
        self.assertIsNone(args.val_passage_reps)
        self.assertTrue(args.output_dir.endswith("/alpha_test"))
        self.assertIn("Skipping validation", result.stdout)

    def test_explicit_validation_and_fixed_mode_ignores_val(self):
        self.prepare()
        dense, sparse = self.prepare(DATASET_SPLIT="val")
        settings = dict(VAL_PASSAGE_REPS=dense.as_posix(), VAL_SPARSE_INDEX=(sparse / "collection").as_posix())
        result, (args,) = self.success(SELECT_HYBRID_ALPHA="true", **settings)
        self.assertEqual((args.validation_dataset_name, args.validation_split), ("flickr", "val"))
        self.assertEqual(args.val_passage_reps, dense.as_posix())
        self.assertTrue(args.output_dir.endswith("/alpha_val"))
        self.assertIn("select 0.1..0.9 on flickr/val", result.stdout)
        _, (args,) = self.success(HYBRID_ALPHA="0.7", VAL_PASSAGE_REPS="/missing", VAL_SPARSE_INDEX="/missing",
                                  VALIDATION_SPLIT="ignored invalid split", VALIDATION_DATASET_NAME="missing")
        self.assertIsNone(args.val_passage_reps)
        self.assertFalse(args.select_hybrid_alpha)
        self.assertTrue(args.output_dir.endswith("/alpha_0.7"))

    def test_validation_pair_missing_annotations_and_incomplete_new_index_fail(self):
        self.prepare()
        dense, sparse = self.prepare(DATASET_SPLIT="val", layout="legacy")
        self.rejected(SELECT_HYBRID_ALPHA="true", VAL_PASSAGE_REPS=dense.as_posix())
        settings = dict(SELECT_HYBRID_ALPHA="true", VAL_PASSAGE_REPS=dense.as_posix(), VAL_SPARSE_INDEX=sparse.as_posix())
        (sparse / "collection/index").mkdir()
        self.assertIn("Validation", self.rejected(**settings).stderr)
        (sparse / "collection/index/segments_2").touch()
        (self.root / "data/flickr/flickr_val.csv").unlink()
        self.assertIn("annotation", self.rejected(**settings).stderr)

    def test_batch_validation_tables_and_shared_validation_rules(self):
        for name in ("flickr", "coco"):
            self.prepare(name)
        vd1, vs1 = self.prepare("flickr", DATASET_SPLIT="val")
        vd2, vs2 = self.prepare("coco", DATASET_SPLIT="val")
        settings = dict(DATASETS="flickr coco", SELECT_HYBRID_ALPHA="true")
        shared = dict(VAL_PASSAGE_REPS=vd1.as_posix(), VAL_SPARSE_INDEX=vs1.as_posix())
        self.rejected(**settings, **shared)
        _, jobs = self.success(**settings, **shared, VALIDATION_DATASET_NAME="flickr")
        self.assertEqual([job.validation_dataset_name for job in jobs], ["flickr", "flickr"])
        self.tables(VAL_DENSE_INDEXES={"flickr": vd1, "coco": vd2}, VAL_SPARSE_INDEXES={"flickr": vs1, "coco": vs2})
        _, jobs = self.success(**settings)
        self.assertEqual([job.validation_dataset_name for job in jobs], ["flickr", "coco"])
        self.assertEqual([job.val_passage_reps for job in jobs], [vd1.as_posix(), vd2.as_posix()])

    def test_validation_task_mismatch_is_rejected_before_launch(self):
        self.prepare("webqa")
        dense, sparse = self.prepare("flickr", DATASET_SPLIT="val")
        result = self.rejected(DATASET="webqa", SELECT_HYBRID_ALPHA="true", VAL_PASSAGE_REPS=dense.as_posix(),
                               VAL_SPARSE_INDEX=sparse.as_posix(), VALIDATION_DATASET_NAME="flickr")
        self.assertIn("task mismatch", result.stderr)

    def test_batch_validation_does_not_leak_into_target_only_job(self):
        for name in ("flickr", "coco"):
            self.prepare(name)
        dense, sparse = self.prepare("flickr", DATASET_SPLIT="val")
        self.tables(VAL_DENSE_INDEXES={"flickr": dense}, VAL_SPARSE_INDEXES={"flickr": sparse})
        _, (first, second) = self.success(DATASETS="flickr coco", SELECT_HYBRID_ALPHA="true")
        self.assertEqual(first.validation_dataset_name, "flickr")
        self.assertTrue(first.output_dir.endswith("/alpha_val"))
        self.assertIsNone(second.val_passage_reps)
        self.assertIsNone(second.val_sparse_index)
        self.assertIsNone(second.validation_dataset_name)
        self.assertTrue(second.output_dir.endswith("/alpha_test"))

    def test_all_jobs_checked_before_first_launch_and_execution_stops_on_failure(self):
        self.prepare("flickr")
        self.rejected(DATASETS="flickr coco")
        for name in ("coco", "webqa"):
            self.prepare(name)
        result, jobs = self.launch(DATASETS="flickr coco webqa", LAUNCHER_TEST_FAIL_MATCH="coco", LAUNCHER_TEST_FAIL_CODE="17")
        self.assertEqual(result.returncode, 17, result.stdout + result.stderr)
        self.assertEqual([job.dataset_name for job in jobs], ["flickr", "coco"])
        self.assertIn("Completed: flickr", result.stdout)
        self.assertNotIn("Completed all", result.stdout)

    def test_prior_correction_flags_and_task_constraints(self):
        self.prepare()
        _, (args,) = self.success(MODIFY_TYPE="modify_single", USE_RERANK_PRIOR_CACHE="true")
        self.assertTrue(args.use_rerank_prior_cache)
        self.assertEqual(args.modify_type, "modify_single")
        self.rejected(USE_RERANK_PRIOR_CACHE="true")
        self.prepare("webqa")
        self.rejected(DATASET="webqa", MODIFY_TYPE="modify_single")

    def test_invalid_configuration_and_missing_files_start_no_jobs(self):
        self.prepare()
        cases = ({"QWEN_FAMILY": "bad"}, {"DATASETS": ""}, {"DATASETS": "flickr flickr"},
                 {"DATASETS": "flickr missing"}, {"GPUS": "0"}, {"NUM_NODES": "0"},
                 {"HYBRID_ALPHA": "1.1"}, {"SELECT_HYBRID_ALPHA": "maybe"}, {"OVERWRITE": "maybe"},
                 {"RETRIEVAL_DTYPE": "int8"}, {"RANKER_DTYPE": "int8"}, {"RANDOM_VIEW_SEED": "bad"},
                 {"QWEN_MIN_PIXELS": "1024", "QWEN_MAX_PIXELS": "64"}, {"QWEN_MAX_PIXELS": "-1"},
                 {"EOL_TYPE": "prompteol"}, {"SPARSE_MANUAL": "true"}, {"CALCULATE_TYPE": "separate"},
                 {"STAGE1_DEPTH": "0"}, {"RERANK_NUM": "0"}, {"SPARSE_LENGTH": "0"},
                 {"TBPR_TYPE": "other"}, {"MODIFY_TYPE": "other"}, {"IR_QUERY_TYPE": "composed"},
                 {"STAGE2_DENSE_INDEX": "/missing"}, {"STAGE2_SPARSE_INDEX": "/missing"},
                 {"STAGE1_SPARSE_INDEX": "/missing"}, {"QWEN_MODEL_PATH": "/missing"}, {"DATA_ROOT": "/missing"})
        for settings in cases:
            with self.subTest(**settings):
                self.rejected(**settings)
        for args in (("--dataset-name", "flickr"), ("--help", "extra"), ("--",)):
            result, jobs = self.launch(*args)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(jobs)


if __name__ == "__main__":
    unittest.main()
