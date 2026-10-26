"""Paired standalone launcher checks against the real Python index path builder."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tevatron/src"))
BASH = shutil.which("bash") or next((path for path in ("D:/Git/bin/bash.exe", "C:/Program Files/Git/bin/bash.exe")
                                    if Path(path).is_file()), None)


@unittest.skipUnless(BASH, "Bash is required for launcher tests")
class DenseModelLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import HfArgumentParser
        from arguments import (ModelArguments, PromptRepsLLMDataArguments,
                               PromptRepsLLMSearchArguments, TrainingArguments)
        from dense_encoding import DenseEncoderArguments
        from dense_search import DenseSearchArguments
        cls.parser = HfArgumentParser((ModelArguments, PromptRepsLLMDataArguments,
            PromptRepsLLMSearchArguments, TrainingArguments, DenseEncoderArguments, DenseSearchArguments))

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "scripts").mkdir()
        for kind in ("mmret", "unime"):
            for mode in ("encode", "search"):
                filename = f"{mode}_dense_{kind}.sh"
                shutil.copyfile(PROJECT_ROOT / "scripts" / filename, self.root / "scripts" / filename)
        self.record = self.root / "argv.bin"
        self.visible = self.root / "visible.txt"
        stub = self.root / "fake-python"
        stub.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "$PAIR_TEST_ARGV"\n'
                        'printf "%s" "${CUDA_VISIBLE_DEVICES-unset}" > "$PAIR_TEST_VISIBLE"\n', encoding="utf-8")
        stub.chmod(0o755)
        keys = ("PYTHON_BIN", "MODEL_NAME_OR_PATH", "DENSE_MODEL_TYPE", "DATA_ROOT", "DATASET", "DATASET_SPLIT",
                "TASK_TYPE", "QUERY_TYPE", "ENCODE_TYPE", "DENSE_OUTPUT_DIR", "BATCH_SIZE", "DTYPE", "GPUS",
                "NUM_PROCESSES", "MASTER_PORT", "PROMPT_TYPE", "CALCULATE_TYPE", "EOL_TYPE", "TBPR_TYPE",
                "NUM_EXPENDED_TOKENS", "SPARSE_LENGTH", "SPARSE_VALUE_TYPE", "REPS_LOC", "SPARSE_LOWER_OR_UPPER",
                "IS_FILTERED", "SPARSE_MANUAL", "USE_OUTPUT_EMBEDDING_CLUSTER", "CLUSTER_SUM", "SPARSE_VALUE_MEAN",
                "PASSAGE_REPS", "SAVE_DIR", "DEPTH", "USE_GPU", "MMRET_QUERY_INSTRUCTION", "CUDA_VISIBLE_DEVICES")
        self.env = {key: value for key, value in os.environ.items() if key not in keys}
        self.env.update(PYTHON_BIN=stub.as_posix(), PAIR_TEST_ARGV=self.record.as_posix(),
                        PAIR_TEST_VISIBLE=self.visible.as_posix(), DENSE_OUTPUT_DIR=(self.root / "dense output").as_posix(),
                        PATH=str(Path(BASH).parent) + os.pathsep + os.environ.get("PATH", ""))

    def launch(self, kind, mode, *args, **env):
        self.record.unlink(missing_ok=True)
        result = subprocess.run([BASH, (self.root / f"scripts/{mode}_dense_{kind}.sh").as_posix(), *args],
                                env={**self.env, **env}, text=True, capture_output=True, timeout=30)
        argv = self.record.read_bytes().decode().rstrip("\0").split("\0") if self.record.exists() else []
        return result, argv

    def success(self, kind, mode, *args, **env):
        result, argv = self.launch(kind, mode, *args, **env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(argv)
        self.assertEqual(argv.count("--dense_model_type"), 1)
        self.assertEqual(self.value(argv, "--dense_model_type"), kind)
        self.assertTrue(any(item.endswith(f"/src/{mode}_dense.py") for item in argv))
        return result, argv

    @staticmethod
    def value(argv, key):
        return argv[argv.index(key) + 1]

    def index_from_encode(self, argv):
        from dense_encoding import output_directory
        start = next(index for index, value in enumerate(argv) if value.endswith("/src/encode_dense.py"))
        model, data, _, train, _, _ = self.parser.parse_args_into_dataclasses(argv[start + 1:])
        return output_directory(model, data, train)

    def pair(self, kind, *args, **env):
        encoded, encoder_argv = self.success(kind, "encode", *args, **env)
        expected = self.index_from_encode(encoder_argv)
        self.assertIn(expected.as_posix(), encoded.stdout)
        expected.mkdir(parents=True, exist_ok=True)
        (expected / "corpus_0.pkl").touch()
        searched, search_argv = self.success(kind, "search", *args, **env)
        self.assertEqual(self.value(search_argv, "--passage_reps"), expected.as_posix())
        return encoder_argv, search_argv

    def test_four_standalone_scripts_and_defaults(self):
        for kind in ("mmret", "unime"):
            for mode in ("encode", "search"):
                subprocess.run([BASH, "-n", (self.root / f"scripts/{mode}_dense_{kind}.sh").as_posix()],
                               check=True, timeout=30)
            encoder, search = self.pair(kind)
            self.assertEqual(self.value(encoder, "--dataset_name"), "flickr")
            self.assertEqual(self.value(encoder, "--encode_type"), "image")
            self.assertEqual(self.value(search, "--query_type"), "text")
            self.assertIn("--fp16", encoder)
            self.assertIn("--fp16", search)
            self.assertTrue(self.value(search, "--save_dir").endswith("/flickr/text/test"))
            self.assertNotIn("deepspeed.launcher.runner", encoder)
            self.assertNotIn("deepspeed.launcher.runner", search)

    def test_all_eight_dataset_paths_match_python(self):
        for kind in ("mmret", "unime"):
            for name, task, query, corpus in (("coco", "ir", "text", "image"), ("flickr", "ir", "text", "image"),
                    ("CUHK-PEDES", "tbpr", "text", "image"), ("ICFG-PEDES", "tbpr", "text", "image"),
                    ("RSTPReid", "tbpr", "text", "image"), ("webqa", "t2it", "text", "corpus"),
                    ("remuq", "it2t", "image", "corpus"), ("llava", "it2t", "image", "corpus")):
                with self.subTest(kind=kind, dataset=name):
                    encoder, search = self.pair(kind, DATASET=name)
                    self.assertEqual(self.value(encoder, "--task_type"), task)
                    self.assertEqual(self.value(encoder, "--encode_type"), corpus)
                    self.assertEqual(self.value(search, "--query_type"), query)

    def test_direction_inference_from_either_side(self):
        for kind in ("mmret", "unime"):
            for name in ("coco", "flickr"):
                for env in ({"QUERY_TYPE": "image"}, {"ENCODE_TYPE": "text"}):
                    encoder, search = self.pair(kind, DATASET=name, **env)
                    self.assertEqual(self.value(encoder, "--encode_type"), "text")
                    self.assertEqual(self.value(search, "--query_type"), "image")
            encoder, search = self.pair(kind, "--dataset_name=remuq", "--query_type", "composed")
            self.assertEqual(self.value(search, "--query_type"), "composed")
            self.assertEqual(self.value(encoder, "--encode_type"), "corpus")

    def test_cli_overrides_environment_and_last_value_wins(self):
        output = (self.root / "custom index root").as_posix()
        for kind in ("mmret", "unime"):
            encoder, search = self.pair(kind, "--dataset_name=coco", "--dataset_name", "RSTPReid",
                "--dataset_split=val", "--query_type=text", "--data_root", "/data with spaces",
                "--model_name_or_path", "/arbitrary local/checkpoint/", "--dense_output_dir", output,
                "--prompt_type=other", "--prompt_type", "prompt_1", "--tbpr_type", "augmented",
                DATASET="remuq", MODEL_NAME_OR_PATH="/old weights")
            for argv in (encoder, search):
                for key, value in (("--dataset_name", "RSTPReid"), ("--dataset_split", "val"),
                                   ("--task_type", "tbpr"), ("--prompt_type", "prompt_1"),
                                   ("--model_name_or_path", "/arbitrary local/checkpoint/")):
                    self.assertEqual(argv.count(key), 1)
                    self.assertEqual(self.value(argv, key), value)

    def test_all_custom_path_labels_and_boolean_forms(self):
        options = ["--calculate_type", "concat", "--eol_type=disassembleeol_separate",
                   "--num_expended_tokens=0003", "--sparse_length", "040", "--sparse_value_type=sum",
                   "--reps_loc", "after_pad", "--sparse_lower_or_upper", "upper", "--is_filtered",
                   "--sparse_manual=yes", "--use_output_embedding_cluster", "1", "--cluster_sum=0016",
                   "--sparse_value_mean", "true"]
        for kind in ("mmret", "unime"):
            encoder, search = self.pair(kind, *options)
            self.assertIn("/filter/concat/prompt_5/test/3_manual_40_sum_cluster_16_after_pad_", self.value(search, "--passage_reps"))
            self.pair(kind, *options, "--is_filtered=false", "--sparse_manual", "false",
                      "--use_output_embedding_cluster=no", "--sparse_value_mean=0")
            self.pair(kind, MODEL_NAME_OR_PATH="C:\\weights\\custom-checkpoint\\",
                      IS_FILTERED="true", SPARSE_MANUAL="1", USE_OUTPUT_EMBEDDING_CLUSTER="yes", SPARSE_VALUE_MEAN="true")

    def test_explicit_index_directory_or_file_and_result_directory(self):
        explicit = self.root / "explicit index"
        explicit.mkdir()
        shard = explicit / "corpus_0.pkl"
        shard.touch()
        result_dir = (self.root / "my results").as_posix()
        for kind in ("mmret", "unime"):
            for path in (explicit, shard):
                _, argv = self.success(kind, "search", "--passage_reps", path.as_posix(), "--save_dir", result_dir,
                    PASSAGE_REPS="/not selected", SAVE_DIR="/not selected", DENSE_OUTPUT_DIR="/irrelevant")
                self.assertEqual(self.value(argv, "--passage_reps"), path.as_posix())
                self.assertEqual(self.value(argv, "--save_dir"), result_dir)
                self.assertEqual(self.value(argv, "--output_dir"), result_dir + "/.trainer")

    def test_search_specific_options_and_passthrough(self):
        index = self.root / "index"
        index.mkdir()
        (index / "corpus_0.pkl").touch()
        for kind in ("mmret", "unime"):
            options = ["--depth=25", "--use_gpu", "--quiet=false", "--retrieval_batch_size", "16"]
            if kind == "mmret":
                options.extend(["--mmret_query_instruction", "Retrieve matching answers."])
            _, argv = self.success(kind, "search", *options, PASSAGE_REPS=index.as_posix())
            self.assertEqual(self.value(argv, "--depth"), "25")
            self.assertEqual(self.value(argv, "--use_gpu"), "true")
            self.assertIn("--quiet=false", argv)
            self.assertEqual(self.value(argv, "--retrieval_batch_size"), "16")
        _, argv = self.success("mmret", "search", PASSAGE_REPS=index.as_posix(), MMRET_QUERY_INSTRUCTION="Custom task")
        self.assertEqual(self.value(argv, "--mmret_query_instruction"), "Custom task")

    def test_precision_flags_and_physical_gpu_ids(self):
        for kind in ("mmret", "unime"):
            for mode in ("encode", "search"):
                index = self.root / "gpu index"
                index.mkdir(exist_ok=True)
                (index / "corpus_0.pkl").touch()
                common = dict(PASSAGE_REPS=index.as_posix()) if mode == "search" else {}
                for dtype, flag in (("float32", None), ("float16", "--fp16"), ("bfloat16", "--bf16")):
                    _, argv = self.success(kind, mode, DTYPE=dtype, **common)
                    self.assertEqual("--fp16" in argv, flag == "--fp16")
                    self.assertEqual("--bf16" in argv, flag == "--bf16")
                _, argv = self.success(kind, mode, "--bf16", "--fp16=false", DTYPE="float16", **common)
                self.assertIn("--bf16", argv)
                self.assertNotIn("--fp16", argv)
                _, argv = self.success(kind, mode, NUM_PROCESSES="2", CUDA_VISIBLE_DEVICES="2,3,5", **common)
                self.assertEqual(argv[:2], ["-m", "deepspeed.launcher.runner"])
                self.assertEqual(self.value(argv, "--include"), "localhost:2,3")
                self.assertEqual(self.visible.read_text(), "unset")
                _, argv = self.success(kind, mode, GPUS="2", **common)
                self.assertEqual(self.value(argv, "--num_gpus"), "2")

    def test_invalid_configuration_and_missing_index_fail_before_python(self):
        for kind in ("mmret", "unime"):
            for mode in ("encode", "search"):
                for env in ({"DATASET": "unknown"}, {"QUERY_TYPE": "text", "ENCODE_TYPE": "text"},
                    {"DATASET": "webqa", "QUERY_TYPE": "image"}, {"DATASET": "remuq", "ENCODE_TYPE": "image"},
                    {"DATASET": "CUHK-PEDES", "TASK_TYPE": "ir"}, {"NUM_PROCESSES": "0"},
                    {"BATCH_SIZE": "0"}, {"MASTER_PORT": "65536"}, {"IS_FILTERED": "maybe"},
                    {"DENSE_MODEL_TYPE": "other"}, {"NUM_EXPENDED_TOKENS": "invalid"}):
                    result, argv = self.launch(kind, mode, **env)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertEqual(argv, [])
                for options in (("--dataset_name",), ("--data_root=",), ("--dataset_name", "--fp16")):
                    result, argv = self.launch(kind, mode, *options)
                    self.assertEqual(result.returncode, 2)
                    self.assertEqual(argv, [])
            result, argv = self.launch(kind, "search")
            self.assertEqual(result.returncode, 2)
            self.assertIn("run scripts/encode_dense_", result.stderr)
            self.assertEqual(argv, [])
            empty = self.root / f"empty {kind}"
            empty.mkdir()
            for path in (empty, self.root / "missing", PROJECT_ROOT / "src/dense_encoding.py"):
                result, argv = self.launch(kind, "search", PASSAGE_REPS=path.as_posix())
                self.assertEqual(result.returncode, 2)
                self.assertEqual(argv, [])


if __name__ == "__main__":
    unittest.main()
