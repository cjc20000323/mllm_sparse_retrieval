"""Standalone TBPR launcher tests using fake artifacts and DeepSpeed, not GPUs."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

import test_three_stage_launcher_paths as fixtures


SCRIPT_NAME = "three_stage_retrieval_pipeline_tbpr.sh"
DEFAULT_MODEL = "llava-hf-llava-v1.6-mistral-7b-hf"
DATASET_FILES = {"CUHK-PEDES": "reid_raw.json", "ICFG-PEDES": "ICFG-PEDES.json",
                 "RSTPReid": "data_captions.json"}
BASH = fixtures.BASH


@unittest.skipUnless(BASH, "Bash is required for TBPR launcher tests")
class TbprLauncherTests(unittest.TestCase):
    # Reuse the general launcher's path assertions, not its tests or shell file.
    assert_configuration = fixtures.ThreeStageLauncherPathTests.assert_configuration
    make_dense = staticmethod(fixtures.ThreeStageLauncherPathTests.make_dense)
    make_sparse = staticmethod(fixtures.ThreeStageLauncherPathTests.make_sparse)
    index_relative = staticmethod(fixtures.ThreeStageLauncherPathTests.index_relative)

    @classmethod
    def setUpClass(cls):
        temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        cls.root = Path(temp.name)
        (cls.root / "scripts").mkdir()
        (cls.root / "src").mkdir()
        (cls.root / "src/three_stage_retrieval_pipeline.py").touch()
        cls.script = cls.root / "scripts" / SCRIPT_NAME
        shutil.copyfile(fixtures.PROJECT_ROOT / "scripts" / SCRIPT_NAME, cls.script)
        for name, filename in DATASET_FILES.items():
            directory = cls.root / "data" / name
            directory.mkdir(parents=True)
            (directory / filename).touch()
            for split in ("test", "val"):
                for view in ("origin_type", "type"):
                    relative = cls.index_relative(name, "image", split, view)
                    cls.make_dense(cls.root / "dense_output" / relative)
                    cls.make_sparse(cls.root / "sparse_output" / relative)
        checkpoint = cls.root / "checkpoint"
        checkpoint.mkdir()
        default_relative = cls.index_relative("CUHK-PEDES", "image", tbpr_type="origin_type")
        default_relative = default_relative.replace("model/", DEFAULT_MODEL + "/", 1)
        cls.make_dense(cls.root / "dense_output" / default_relative)
        cls.make_sparse(cls.root / "sparse_output" / default_relative)
        (cls.root / "checkpoints" / DEFAULT_MODEL).mkdir(parents=True)
        cls.record = cls.root / "argv.bin"
        stub = cls.root / "deepspeed"
        stub.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "$TBPR_TEST_ARGV"\n', encoding="utf-8")
        stub.chmod(0o755)
        names = set(re.findall(r"\$\{([A-Z_][A-Z_0-9]*):-", cls.script.read_text(encoding="utf-8")))
        cls.base_env = {key: value for key, value in os.environ.items() if key not in names}
        cls.base_env.update(DATA_ROOT=cls.root.as_posix(), BASE_MODEL="model", ARTIFACT_MODEL_NAME="model",
                            RETRIEVAL_MODEL_PATH=checkpoint.as_posix(), RANKER_MODEL_PATH=checkpoint.as_posix(),
                            DEEPSPEED_BIN=stub.as_posix(), TBPR_TEST_ARGV=cls.record.as_posix(),
                            PATH=str(Path(BASH).parent) + os.pathsep + os.environ.get("PATH", ""))

    def launch(self, *args, **env):
        if self.record.exists():
            self.record.unlink()
        result = subprocess.run([BASH, self.script.as_posix(), *args], env={**self.base_env, **env},
                                capture_output=True, text=True, timeout=30)
        argv = self.record.read_bytes().decode().rstrip("\0").split("\0") if self.record.exists() else []
        return result, argv

    def successful(self, *args, **env):
        result, argv = self.launch(*args, **env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(argv)
        for option in fixtures.PATH_OPTIONS:
            self.assertEqual(argv.count(option), 1, (option, argv))
        self.assertEqual(argv.count("src/three_stage_retrieval_pipeline.py"), 1)
        self.assertNotIn("--use-rerank-prior-cache", argv)
        self.assertNotIn("--use_rerank_prior_cache", argv)
        return result, argv

    def rejected(self, args=(), message="", **env):
        result, argv = self.launch(*args, **env)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn(message, result.stderr)
        self.assertEqual(argv, [])

    def test_syntax_and_independent_launch(self):
        subprocess.run([BASH, "-n", self.script.as_posix()], check=True, timeout=30)
        self.assertFalse((self.root / "scripts/three_stage_retrieval_pipeline.sh").exists())
        self.assert_configuration(*self.successful(), "CUHK-PEDES", "text")

    def test_default_pipeline_parameters_and_workers(self):
        _, argv = self.successful()
        for option, expected in (("--num_gpus", "4"), ("--num_nodes", "1"),
                                 ("--stage1-method", "sparse_concat"), ("--stage1-depth", "1000"),
                                 ("--stage2-depth", "1000"), ("--hybrid-alpha", "0.5"),
                                 ("--rerank-type", "pointwise"), ("--rerank-num", "5"),
                                 ("--rerank-template", "caption_generation"), ("--modify_type", "no"),
                                 ("--retrieval-dtype", "float16"), ("--ranker-dtype", "float16")):
            self.assertEqual(argv[argv.index(option) + 1], expected)

    def test_default_model_paths_and_artifact_model_name(self):
        result, argv = self.successful(BASE_MODEL="", ARTIFACT_MODEL_NAME="",
                                      RETRIEVAL_MODEL_PATH="", RANKER_MODEL_PATH="")
        for option in ("--retrieval-model-name-or-path", "--ranker-model-name-or-path"):
            self.assertEqual(argv[argv.index(option) + 1], "./checkpoints/" + DEFAULT_MODEL)
        for option in ("--stage2-dense-index", "--stage2-sparse-index", "--output-dir"):
            self.assertIn("/" + DEFAULT_MODEL + "/CUHK-PEDES/", argv[argv.index(option) + 1])
        self.assertIn("text_to_image", result.stdout)

    def test_each_dataset_and_view_type_via_environment(self):
        for name in DATASET_FILES:
            for view in ("origin_type", "type"):
                with self.subTest(dataset=name, view=view):
                    self.assert_configuration(*self.successful(DATASET=name, TBPR_TYPE=view),
                                              name, "text", tbpr_type=view)

    def test_each_dataset_via_cli_both_argument_forms(self):
        for name in DATASET_FILES:
            for equals in (False, True):
                with self.subTest(dataset=name, equals=equals):
                    args = ([f"--dataset-name={name}", "--tbpr-type=type"] if equals else
                            ["--dataset-name", name, "--tbpr-type", "type"])
                    self.assert_configuration(*self.successful(*args), name, "text", tbpr_type="type")

    def test_cli_priority_applies_before_task_checks_and_paths(self):
        result, argv = self.successful("--dataset-name=ICFG-PEDES", "--query-type=text", "--task-type", "tbpr",
                                      "--tbpr-type=type", "--dataset-split=val", DATASET="flickr", TASK_TYPE="ir",
                                      QUERY_TYPE="image", TBPR_TYPE="invalid", DATASET_SPLIT="test")
        self.assert_configuration(result, argv, "ICFG-PEDES", "text", "val", "type")

    def test_last_cli_value_wins(self):
        result, argv = self.successful("--dataset-name=flickr", "--dataset-name", "RSTPReid",
                                      "--task-type=ir", "--task-type=tbpr", "--query-type=image", "--query-type", "text",
                                      "--tbpr-type=invalid", "--tbpr-type", "type", "--dataset-split=val",
                                      "--dataset-split", "test")
        self.assert_configuration(result, argv, "RSTPReid", "text", tbpr_type="type")

    def test_custom_output_path_is_preserved(self):
        for args, env, expected in (
            ([], {"OUTPUT_DIR": "./my output/image_to_text"}, "./my output/image_to_text"),
            (["--output-dir", "./my cli output"], {"OUTPUT_DIR": "ignored"}, "./my cli output"),
            (["--output-dir=first", "--output-dir=./last output"], {}, "./last output"),
        ):
            with self.subTest(args=args, env=env):
                result, argv = self.successful(*args, **env)
                self.assertEqual(argv[argv.index("--output-dir") + 1], expected)
                self.assertIn(f"Output directory:   {expected}", result.stdout)

    def test_custom_indexes_and_stage1_default_reuse(self):
        dense = self.make_dense(self.root / "custom dense") / "corpus_0.pkl"
        stage1 = self.make_sparse(self.root / "custom sparse one")
        stage2 = self.make_sparse(self.root / "custom sparse two")
        result, argv = self.successful("--stage2-dense-index", dense.as_posix(),
                                      f"--stage1-sparse-index={stage1.as_posix()}",
                                      "--stage2-sparse-index", stage2.as_posix(),
                                      STAGE2_DENSE_INDEX="missing", STAGE1_SPARSE_INDEX="missing",
                                      STAGE2_SPARSE_INDEX="missing")
        for option, path in (("--stage2-dense-index", dense), ("--stage1-sparse-index", stage1),
                             ("--stage2-sparse-index", stage2)):
            self.assertEqual(argv[argv.index(option) + 1], path.as_posix())
            self.assertIn(path.as_posix(), result.stdout)
        _, argv = self.successful(STAGE2_DENSE_INDEX=dense.as_posix(), STAGE2_SPARSE_INDEX=stage2.as_posix())
        self.assertEqual(argv[argv.index("--stage1-sparse-index") + 1], stage2.as_posix())

    def test_custom_data_root_is_checked_and_forwarded(self):
        directory = self.root / "alternate data" / "data/RSTPReid"
        directory.mkdir(parents=True)
        (directory / "data_captions.json").touch()
        data_root = directory.parent.parent.as_posix()
        _, argv = self.successful("--dataset-name=RSTPReid", "--data-root", data_root, DATA_ROOT="missing")
        self.assertEqual(argv[argv.index("--data-root") + 1], data_root)

    def test_hyperparameters_are_configurable_and_remaining_cli_is_forwarded(self):
        _, argv = self.successful("--stage2-depth=30", "--rerank-num", "10", "--overwrite",
                                 GPUS="2", NUM_NODES="2", STAGE1_DEPTH="200", STAGE2_DEPTH="20", RERANK_NUM="20",
                                 RERANK_BATCH_SIZE="3", RETRIEVAL_BATCH_SIZE="2", HYBRID_ALPHA="0.7")
        for option, expected in (("--num_gpus", "2"), ("--num_nodes", "2"), ("--stage1-depth", "200"),
                                 ("--stage2-depth", "20"), ("--rerank-num", "20"),
                                 ("--rerank-batch-size", "3"), ("--retrieval-batch-size", "2"), ("--hybrid-alpha", "0.7")):
            self.assertEqual(argv[argv.index(option) + 1], expected)
        self.assertEqual(argv[-4:], ["--stage2-depth=30", "--rerank-num", "10", "--overwrite"])
        _, argv = self.successful(OVERWRITE="true")
        self.assertIn("--overwrite", argv)

    def test_non_tbpr_datasets_and_directions_are_rejected(self):
        for name in ("flickr", "coco", "webqa", "remuq", "llava", "all", "unknown"):
            with self.subTest(dataset=name):
                self.rejected([f"--dataset-name={name}"], "TBPR DATASET must be")
        for env, message in (({"TASK_TYPE": "ir"}, "requires TASK_TYPE=tbpr"),
                             ({"QUERY_TYPE": "image"}, "requires QUERY_TYPE=text"),
                             ({"TBPR_TYPE": "invalid"}, "TBPR_TYPE must be")):
            with self.subTest(env=env):
                self.rejected(message=message, **env)
        self.rejected(["--task-type=ir"], "requires TASK_TYPE=tbpr")
        self.rejected(["--query-type", "composed"], "requires QUERY_TYPE=text")
        self.rejected(["--tbpr-type=invalid"], "TBPR_TYPE must be")

    def test_prior_cache_is_rejected_early_in_environment_and_cli(self):
        for value in ("true", "1", "yes", "TRUE"):
            with self.subTest(value=value):
                self.rejected(message="TBPR does not support", USE_RERANK_PRIOR_CACHE=value)
        for flag in ("--use-rerank-prior-cache", "--use_rerank_prior_cache"):
            with self.subTest(flag=flag):
                self.rejected([flag], "TBPR does not support")
        self.rejected(message="expected a boolean", USE_RERANK_PRIOR_CACHE="invalid")
        self.successful(USE_RERANK_PRIOR_CACHE="0")

    def test_missing_annotations_and_model_directories_are_reported(self):
        empty_root = self.root / "empty datasets"
        for name, filename in DATASET_FILES.items():
            (empty_root / "data" / name).mkdir(parents=True)
            with self.subTest(dataset=name):
                self.rejected([f"--dataset-name={name}", f"--data-root={empty_root.as_posix()}"], filename)
        self.rejected(message="retrieval checkpoint directory does not exist", RETRIEVAL_MODEL_PATH="missing model")
        self.rejected(message="ranker checkpoint directory does not exist", RANKER_MODEL_PATH="missing ranker")

    def test_invalid_dense_and_sparse_indexes_are_reported_before_launch(self):
        empty_dense = self.root / "empty dense"
        empty_dense.mkdir()
        bad_dense = self.root / "not_corpus.pkl"
        bad_dense.touch()
        empty_sparse = self.root / "empty sparse"
        empty_sparse.mkdir()
        cases = [(["--stage2-dense-index=missing"], "Stage-2 dense index directory does not exist"),
                 ([f"--stage2-dense-index={empty_dense.as_posix()}"], "no corpus*.pkl found"),
                 ([f"--stage2-dense-index={bad_dense.as_posix()}"], "not a corpus*.pkl file"),
                 ([f"--stage1-sparse-index={empty_sparse.as_posix()}"], "no Lucene segments_* file"),
                 ([f"--stage2-sparse-index={empty_sparse.as_posix()}"], "no Lucene segments_* file")]
        for args, message in cases:
            with self.subTest(args=args):
                self.rejected(args, message)

    def test_missing_managed_values_and_invalid_gpu_counts(self):
        for args in (["--dataset-name"], ["--query-type="], ["--output-dir", "--tbpr-type=type"],
                     ["--stage2-dense-index", ""], ["--data-root="], ["--task-type"]):
            with self.subTest(args=args):
                self.rejected(args, "requires a")
        for env, message in (({"GPUS": "0"}, "GPUS must be a positive integer"),
                             ({"NUM_NODES": "invalid"}, "NUM_NODES must be a positive integer"),
                             ({"DEEPSPEED_BIN": "no-such-deepspeed"}, "DeepSpeed executable was not found")):
            with self.subTest(env=env):
                self.rejected(message=message, **env)


if __name__ == "__main__":
    unittest.main()
