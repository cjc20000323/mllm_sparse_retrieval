"""Test the standalone multimodal Bash launcher without GPUs or real datasets."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

import test_three_stage_launcher_paths as fixtures


SCRIPT_NAME = "three_stage_retrieval_pipeline_multimodal.sh"
DEFAULT_MODEL = "llava-hf-llava-v1.6-mistral-7b-hf"
DATASETS = ("webqa", "remuq", "llava")
BASH = fixtures.BASH
MANAGED_OPTIONS = tuple(option for option in fixtures.PATH_OPTIONS if option != "--tbpr-type") + (
    "--modify_type", "--rerank-template", "--rerank-type", "--rerank-prior-cache-dir",
)


@unittest.skipUnless(BASH, "Bash is required for multimodal launcher tests")
class MultimodalLauncherTests(unittest.TestCase):
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
        for name in DATASETS:
            _, _, _, _, data_dir, files = fixtures.PROFILES[name]
            directory = cls.root / "data" / data_dir
            directory.mkdir(parents=True)
            for filename in files:
                (directory / filename).touch()
            for split in ("test", "val"):
                relative = cls.index_relative(name, "corpus", split)
                cls.make_dense(cls.root / "dense_output" / relative)
                cls.make_sparse(cls.root / "sparse_output" / relative)
        checkpoint = cls.root / "checkpoint"
        checkpoint.mkdir()
        default_relative = cls.index_relative("webqa", "corpus").replace("model/", DEFAULT_MODEL + "/", 1)
        cls.make_dense(cls.root / "dense_output" / default_relative)
        cls.make_sparse(cls.root / "sparse_output" / default_relative)
        (cls.root / "checkpoints" / DEFAULT_MODEL).mkdir(parents=True)
        cls.record = cls.root / "argv.bin"
        stub = cls.root / "deepspeed"
        stub.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "$MULTIMODAL_TEST_ARGV"\n', encoding="utf-8")
        stub.chmod(0o755)
        names = set(re.findall(r"\$\{([A-Z_][A-Z_0-9]*):-", cls.script.read_text(encoding="utf-8")))
        cls.base_env = {key: value for key, value in os.environ.items() if key not in names}
        cls.base_env.update(DATA_ROOT=cls.root.as_posix(), BASE_MODEL="model", ARTIFACT_MODEL_NAME="model",
                            RETRIEVAL_MODEL_PATH=checkpoint.as_posix(), RANKER_MODEL_PATH=checkpoint.as_posix(),
                            DEEPSPEED_BIN=stub.as_posix(), MULTIMODAL_TEST_ARGV=cls.record.as_posix(),
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
        for option in MANAGED_OPTIONS:
            self.assertEqual(argv.count(option), 1, (option, argv))
        self.assertEqual(argv.count("src/three_stage_retrieval_pipeline.py"), 1)
        self.assertNotIn("--tbpr-type", argv)
        return result, argv

    def rejected(self, args=(), message="", **env):
        result, argv = self.launch(*args, **env)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn(message, result.stderr)
        self.assertEqual(argv, [])

    def assert_configuration(self, result, argv, name, query=None, split="test"):
        task, default_query, _, direction, _, _ = fixtures.PROFILES[name]
        query = default_query if query is None else query
        relative = self.index_relative(name, "corpus", split)
        output = f"./three_stage_results/model/{name}/{direction}/{split}/sparse_concat_stage1"
        for option, expected in (("--dataset-name", name), ("--dataset-split", split), ("--task-type", task),
                                 ("--query-type", query), ("--output-dir", output),
                                 ("--stage2-dense-index", "dense_output/" + relative),
                                 ("--stage2-sparse-index", "sparse_output/" + relative),
                                 ("--stage1-sparse-index", "sparse_output/" + relative)):
            self.assertEqual(argv[argv.index(option) + 1], expected)
        self.assertIn(f"Dataset:            {name} ({split}, {direction})", result.stdout)
        self.assertIn(f"Task/query/corpus:  {task} / {query} / corpus", result.stdout)
        self.assertIn(f"Output directory:   {output}", result.stdout)

    def test_syntax_and_independent_launch(self):
        subprocess.run([BASH, "-n", self.script.as_posix()], check=True, timeout=30)
        self.assertFalse((self.root / "scripts/three_stage_retrieval_pipeline.sh").exists())
        self.assertFalse((self.root / "scripts/three_stage_retrieval_pipeline_tbpr.sh").exists())
        self.assert_configuration(*self.successful(), "webqa")

    def test_defaults_preserve_pipeline_parameters(self):
        _, argv = self.successful()
        for option, expected in (("--num_gpus", "4"), ("--num_nodes", "1"),
                                 ("--stage1-method", "sparse_concat"), ("--stage1-depth", "1000"),
                                 ("--stage2-depth", "1000"), ("--hybrid-alpha", "0.5"),
                                 ("--rerank-type", "pointwise"), ("--rerank-num", "5"),
                                 ("--rerank-template", "caption_generation"), ("--modify_type", "no"),
                                 ("--retrieval-dtype", "float16"), ("--ranker-dtype", "float16")):
            self.assertEqual(argv[argv.index(option) + 1], expected)
        self.assertNotIn("--use-rerank-prior-cache", argv)

    def test_default_model_and_cache_paths(self):
        _, argv = self.successful(BASE_MODEL="", ARTIFACT_MODEL_NAME="",
                                  RETRIEVAL_MODEL_PATH="", RANKER_MODEL_PATH="")
        for option in ("--retrieval-model-name-or-path", "--ranker-model-name-or-path"):
            self.assertEqual(argv[argv.index(option) + 1], "./checkpoints/" + DEFAULT_MODEL)
        for option in ("--stage2-dense-index", "--stage2-sparse-index", "--output-dir"):
            self.assertIn("/" + DEFAULT_MODEL + "/webqa/", argv[argv.index(option) + 1])
        self.assertTrue(argv[argv.index("--rerank-prior-cache-dir") + 1].endswith("/cache/rerank_prior"))

    def test_each_dataset_via_environment_and_cli_forms(self):
        for name in DATASETS:
            with self.subTest(dataset=name, source="env"):
                self.assert_configuration(*self.successful(DATASET=name, TBPR_TYPE="type"), name)
            for equals in (False, True):
                with self.subTest(dataset=name, equals=equals):
                    args = [f"--dataset-name={name}"] if equals else ["--dataset-name", name]
                    self.assert_configuration(*self.successful(*args), name)

    def test_it2t_query_modes_always_use_same_direction_and_corpus(self):
        for name in ("remuq", "llava"):
            for query in ("image", "composed"):
                with self.subTest(dataset=name, query=query):
                    self.assert_configuration(*self.successful("--query-type", query, DATASET=name), name, query)

    def test_cli_priority_applies_before_task_checks_and_path_generation(self):
        result, argv = self.successful("--dataset-name=llava", "--query-type=composed", "--task-type", "it2t",
                                      "--dataset-split=val", DATASET="webqa", TASK_TYPE="t2it", QUERY_TYPE="text",
                                      DATASET_SPLIT="test")
        self.assert_configuration(result, argv, "llava", "composed", "val")
        result, argv = self.successful("--dataset-name=webqa", "--query-type=text", "--task-type=t2it",
                                      DATASET="remuq", TASK_TYPE="it2t", QUERY_TYPE="image")
        self.assert_configuration(result, argv, "webqa")

    def test_last_managed_cli_values_win(self):
        result, argv = self.successful("--dataset-name=coco", "--dataset-name", "remuq",
                                      "--task-type=t2it", "--task-type=it2t", "--query-type=text", "--query-type=image",
                                      "--dataset-split=val", "--dataset-split", "test")
        self.assert_configuration(result, argv, "remuq")

    def test_explicit_output_paths_are_preserved(self):
        for args, env, expected in (
            ([], {"OUTPUT_DIR": "./my output/text_to_image"}, "./my output/text_to_image"),
            (["--output-dir", "./my cli output"], {"OUTPUT_DIR": "ignored"}, "./my cli output"),
            (["--output-dir=first", "--output-dir=./last output"], {}, "./last output"),
        ):
            with self.subTest(args=args, env=env):
                result, argv = self.successful(*args, **env)
                self.assertEqual(argv[argv.index("--output-dir") + 1], expected)
                self.assertIn(f"Output directory:   {expected}", result.stdout)

    def test_custom_indexes_and_shared_stage1_index(self):
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
        flat_index = self.root / "flat lucene"
        flat_index.mkdir()
        (flat_index / "segments_1").touch()
        _, argv = self.successful(STAGE2_DENSE_INDEX=dense.as_posix(), STAGE2_SPARSE_INDEX=flat_index.as_posix())
        self.assertEqual(argv[argv.index("--stage1-sparse-index") + 1], flat_index.as_posix())

    def test_cli_data_root_is_used_for_checks_and_command(self):
        directory = self.root / "alternate data" / "data/UMRB-LLaVA"
        directory.mkdir(parents=True)
        for filename in ("query.parquet", "qrels.parquet", "corpus.parquet"):
            (directory / filename).touch()
        data_root = directory.parent.parent.as_posix()
        _, argv = self.successful("--dataset-name=llava", "--data-root", data_root, DATA_ROOT="missing")
        self.assertEqual(argv[argv.index("--data-root") + 1], data_root)

    def test_hyperparameters_and_unmanaged_args_are_forwarded(self):
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

    def test_invalid_dataset_task_and_query_fail_before_launch(self):
        for name in ("flickr", "coco", "CUHK-PEDES", "all", "unknown"):
            with self.subTest(dataset=name):
                self.rejected([f"--dataset-name={name}"], "Multimodal DATASET must be")
        for name, query in (("webqa", "image"), ("webqa", "composed"), ("remuq", "text"), ("llava", "text")):
            with self.subTest(dataset=name, query=query):
                self.rejected([f"--dataset-name={name}", f"--query-type={query}"], "requires QUERY_TYPE")
        self.rejected(["--task-type=it2t"], "webqa requires TASK_TYPE=t2it")
        self.rejected(["--dataset-name=remuq", "--task-type=t2it"], "remuq requires TASK_TYPE=it2t")

    def test_prior_cache_valid_combinations_and_templates(self):
        for name in ("remuq", "llava"):
            for template in ("caption_generation", "what_caption_generation"):
                with self.subTest(dataset=name, template=template):
                    result, argv = self.successful(DATASET=name, USE_RERANK_PRIOR_CACHE="true",
                                                  MODIFY_TYPE="modify_single", RERANK_TEMPLATE=template,
                                                  RERANK_PRIOR_CACHE_DIR="./offline priors")
                    self.assertEqual(argv.count("--use-rerank-prior-cache"), 1)
                    self.assertEqual(argv[argv.index("--rerank-prior-cache-dir") + 1], "./offline priors")
                    self.assertIn("Prior cache:        true (./offline priors)", result.stdout)

    def test_prior_cache_cli_aliases_priority_and_last_values(self):
        for flag in ("--use-rerank-prior-cache", "--use_rerank_prior_cache"):
            with self.subTest(flag=flag):
                _, argv = self.successful("--dataset-name=remuq", flag, "--modify_type=no",
                                          "--modify-type", "modify_single", "--rerank-template=relevant",
                                          "--rerank-template=what_caption_generation", "--rerank-type=pointwise",
                                          "--rerank-prior-cache-dir=first", "--rerank_prior_cache_dir", "./last priors",
                                          USE_RERANK_PRIOR_CACHE="false", MODIFY_TYPE="no",
                                          RERANK_PRIOR_CACHE_DIR="ignored", RERANK_TYPE="listwise")
                self.assertEqual(argv.count("--use-rerank-prior-cache"), 1)
                self.assertNotIn("--use_rerank_prior_cache", argv)
                self.assertNotIn("--modify-type", argv)
                self.assertEqual(argv[argv.index("--modify_type") + 1], "modify_single")
                self.assertEqual(argv[argv.index("--rerank-prior-cache-dir") + 1], "./last priors")
        _, argv = self.successful("--rerank_prior_cache_dir=./disabled priors", USE_RERANK_PRIOR_CACHE="0")
        self.assertNotIn("--use-rerank-prior-cache", argv)
        self.assertEqual(argv[argv.index("--rerank-prior-cache-dir") + 1], "./disabled priors")

    def test_prior_cache_rejects_webqa_and_invalid_scoring_configuration(self):
        for args, env, message in (
            ([], {"USE_RERANK_PRIOR_CACHE": "true", "MODIFY_TYPE": "modify_single"}, "webqa does not support"),
            (["--use_rerank_prior_cache"], {}, "webqa does not support"),
            (["--dataset-name=llava", "--use-rerank-prior-cache"], {}, "requires MODIFY_TYPE=modify_single"),
            ([], {"DATASET": "remuq", "USE_RERANK_PRIOR_CACHE": "true", "MODIFY_TYPE": "modify_single",
                  "RERANK_TEMPLATE": "relevant"}, "requires a caption_generation"),
            (["--rerank-type=listwise"], {}, "requires RERANK_TYPE=pointwise"),
            ([], {"USE_RERANK_PRIOR_CACHE": "invalid"}, "expected a boolean"),
            (["--use-rerank-prior-cache=true"], {}, "is a flag"),
        ):
            with self.subTest(args=args, env=env):
                self.rejected(args, message, **env)

    def test_every_dataset_annotation_is_required(self):
        for name in DATASETS:
            _, _, _, _, data_dir, files = fixtures.PROFILES[name]
            for omitted in files:
                with self.subTest(dataset=name, missing=omitted):
                    alternate = self.root / "missing annotations" / name / omitted
                    directory = alternate / "data" / data_dir
                    directory.mkdir(parents=True)
                    for filename in files:
                        if filename != omitted:
                            (directory / filename).touch()
                    self.rejected([f"--dataset-name={name}", f"--data-root={alternate.as_posix()}"], omitted)

    def test_missing_models_and_invalid_indexes_fail_before_launch(self):
        self.rejected(message="retrieval checkpoint directory does not exist", RETRIEVAL_MODEL_PATH="missing model")
        self.rejected(message="ranker checkpoint directory does not exist", RANKER_MODEL_PATH="missing ranker")
        empty_dense = self.root / "empty dense"
        empty_dense.mkdir()
        bad_dense = self.root / "not_corpus.pkl"
        bad_dense.touch()
        empty_sparse = self.root / "empty sparse"
        empty_sparse.mkdir()
        for args, message in (
            (["--stage2-dense-index=missing"], "Stage-2 dense index directory does not exist"),
            ([f"--stage2-dense-index={empty_dense.as_posix()}"], "--encode_type corpus --task_type t2it"),
            (["--dataset-name=remuq", f"--stage2-dense-index={empty_dense.as_posix()}"], "--encode_type corpus --task_type it2t"),
            ([f"--stage2-dense-index={bad_dense.as_posix()}"], "not a corpus*.pkl file"),
            ([f"--stage1-sparse-index={empty_sparse.as_posix()}"], "no Lucene segments_* file"),
            ([f"--stage2-sparse-index={empty_sparse.as_posix()}"], "no Lucene segments_* file"),
        ):
            with self.subTest(args=args):
                self.rejected(args, message)

    def test_missing_managed_values_and_invalid_launch_settings(self):
        for args in (["--dataset-name"], ["--query-type="], ["--output-dir", "--task-type=it2t"],
                     ["--stage2-dense-index", ""], ["--data-root="], ["--modify-type"],
                     ["--rerank-template="], ["--rerank_prior_cache_dir"], ["--rerank-type"]):
            with self.subTest(args=args):
                self.rejected(args, "requires a")
        for env, message in (({"GPUS": "0"}, "GPUS must be a positive integer"),
                             ({"NUM_NODES": "invalid"}, "NUM_NODES must be a positive integer"),
                             ({"DEEPSPEED_BIN": "no-such-deepspeed"}, "DeepSpeed executable was not found")):
            with self.subTest(env=env):
                self.rejected(message=message, **env)


if __name__ == "__main__":
    unittest.main()
