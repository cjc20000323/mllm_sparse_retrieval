"""Standalone VLM2Vec launcher argv/path tests without large model downloads."""

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
class VLM2VecLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import HfArgumentParser
        from arguments import ModelArguments, PromptRepsLLMDataArguments, TrainingArguments
        from dense_encoding import DenseEncoderArguments
        cls.parser = HfArgumentParser((ModelArguments, PromptRepsLLMDataArguments, TrainingArguments, DenseEncoderArguments))

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "scripts").mkdir()
        self.script = self.root / "scripts/encode_dense_vlm2vec.sh"
        shutil.copyfile(PROJECT_ROOT / "scripts/encode_dense_vlm2vec.sh", self.script)
        self.record = self.root / "argv.bin"
        self.visible = self.root / "visible.txt"
        stub = self.root / "fake-python"
        stub.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "$VLM_TEST_ARGV"\n'
                        'printf "%s" "${CUDA_VISIBLE_DEVICES-unset}" > "$VLM_TEST_VISIBLE"\n', encoding="utf-8")
        stub.chmod(0o755)
        keys = ("PYTHON_BIN", "MODEL_NAME_OR_PATH", "DENSE_MODEL_TYPE", "DATA_ROOT", "DATASET", "DATASET_SPLIT",
                "TASK_TYPE", "QUERY_TYPE", "ENCODE_TYPE", "DENSE_OUTPUT_DIR", "BATCH_SIZE", "DTYPE", "GPUS",
                "NUM_PROCESSES", "MASTER_PORT", "PROMPT_TYPE", "CALCULATE_TYPE", "EOL_TYPE", "TBPR_TYPE",
                "NUM_EXPENDED_TOKENS", "SPARSE_LENGTH", "SPARSE_VALUE_TYPE", "REPS_LOC", "SPARSE_LOWER_OR_UPPER",
                "IS_FILTERED", "SPARSE_MANUAL", "USE_OUTPUT_EMBEDDING_CLUSTER", "CLUSTER_SUM", "SPARSE_VALUE_MEAN",
                "CACHE_DIR", "VLM2VEC_IMAGE_RESOLUTION", "CUDA_VISIBLE_DEVICES")
        self.env = {key: value for key, value in os.environ.items() if key not in keys}
        self.env.update(PYTHON_BIN=stub.as_posix(), VLM_TEST_ARGV=self.record.as_posix(),
                        VLM_TEST_VISIBLE=self.visible.as_posix(), DENSE_OUTPUT_DIR=(self.root / "dense output").as_posix(),
                        PATH=str(Path(BASH).parent) + os.pathsep + os.environ.get("PATH", ""))

    def launch(self, *args, **env):
        self.record.unlink(missing_ok=True)
        result = subprocess.run([BASH, self.script.as_posix(), *args], env={**self.env, **env},
                                text=True, capture_output=True, timeout=30)
        argv = self.record.read_bytes().decode().rstrip("\0").split("\0") if self.record.exists() else []
        return result, argv

    def success(self, *args, **env):
        from vlm2vec_dense import vlm2vec_output_directory
        result, argv = self.launch(*args, **env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(argv)
        start = next(index for index, value in enumerate(argv) if value.endswith("/src/encode_dense.py"))
        # This check compares paths only; allow parsing the GPU-default BF16 flags on CPU CI.
        model, data, train, dense = self.parser.parse_args_into_dataclasses(argv[start + 1:] + ["--use_cpu", "true"])
        self.assertEqual(dense.dense_model_type, "vlm2vec")
        self.assertIn(vlm2vec_output_directory(model, data, train, dense.vlm2vec_image_resolution).as_posix(), result.stdout)
        return argv

    @staticmethod
    def value(argv, key):
        return argv[argv.index(key) + 1]

    def test_defaults_use_hf_repo_bf16_and_high_resolution(self):
        subprocess.run([BASH, "-n", self.script.as_posix()], check=True, timeout=30)
        argv = self.success()
        self.assertEqual(self.value(argv, "--model_name_or_path"), "TIGER-Lab/VLM2Vec-LLaVa-Next")
        self.assertEqual(self.value(argv, "--vlm2vec_image_resolution"), "high")
        self.assertIn("--bf16", argv)
        self.assertNotIn("--fp16", argv)
        self.assertNotIn("deepspeed.launcher.runner", argv)

    def test_eight_datasets_and_both_ir_modalities(self):
        for name, task, mode in (("coco", "ir", "image"), ("flickr", "ir", "image"),
                ("CUHK-PEDES", "tbpr", "image"), ("ICFG-PEDES", "tbpr", "image"),
                ("RSTPReid", "tbpr", "image"), ("webqa", "t2it", "corpus"),
                ("remuq", "it2t", "corpus"), ("llava", "it2t", "corpus")):
            argv = self.success(DATASET=name)
            self.assertEqual(self.value(argv, "--task_type"), task)
            self.assertEqual(self.value(argv, "--encode_type"), mode)
        for name in ("coco", "flickr"):
            argv = self.success(DATASET=name, QUERY_TYPE="image")
            self.assertEqual(self.value(argv, "--encode_type"), "text")

    def test_local_model_cache_resolution_and_cli_override(self):
        for resolution in ("high", "mid", "low", "original"):
            argv = self.success("--dataset_name=webqa", "--vlm2vec_image_resolution=mid",
                "--vlm2vec_image_resolution", resolution, "--model_name_or_path", "/local weights/checkpoint/",
                "--cache_dir", "/custom cache", "--prompt_type", "prompt_1", "--is_filtered=true",
                DATASET="flickr", CACHE_DIR="/old cache", VLM2VEC_IMAGE_RESOLUTION="low", DTYPE="float32")
            self.assertEqual(self.value(argv, "--model_name_or_path"), "/local weights/checkpoint/")
            self.assertEqual(self.value(argv, "--cache_dir"), "/custom cache")
            self.assertEqual(argv.count("--vlm2vec_image_resolution"), 1)
            self.assertNotIn("--bf16", argv)

    def test_physical_gpu_selection_and_precision_override(self):
        argv = self.success("--fp16", NUM_PROCESSES="2", CUDA_VISIBLE_DEVICES="2,3,5", MASTER_PORT="29503")
        self.assertEqual(argv[:2], ["-m", "deepspeed.launcher.runner"])
        self.assertEqual(self.value(argv, "--include"), "localhost:2,3")
        self.assertEqual(self.visible.read_text(), "unset")
        self.assertIn("--fp16", argv)
        self.assertNotIn("--bf16", argv)

    def test_invalid_configuration_fails_before_python(self):
        for env in ({"DATASET": "bad"}, {"VLM2VEC_IMAGE_RESOLUTION": "bad"}, {"DENSE_MODEL_TYPE": "mmret"},
                    {"DATASET": "webqa", "ENCODE_TYPE": "image"}, {"QUERY_TYPE": "text", "ENCODE_TYPE": "text"},
                    {"NUM_PROCESSES": "0"}, {"NUM_PROCESSES": "2", "CUDA_VISIBLE_DEVICES": "1,1"}):
            result, argv = self.launch(**env)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertEqual(argv, [])
        for args in (("--cache_dir",), ("--vlm2vec_image_resolution=",)):
            result, argv = self.launch(*args)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(argv, [])


if __name__ == "__main__":
    unittest.main()
