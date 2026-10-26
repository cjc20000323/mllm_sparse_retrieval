"""Inspect launcher argv using a fake Python executable, without GPUs/DeepSpeed."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash") or next((path for path in ("D:/Git/bin/bash.exe", "C:/Program Files/Git/bin/bash.exe")
                                    if Path(path).is_file()), None)


@unittest.skipUnless(BASH, "Bash is required for launcher tests")
class MMRetLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        cls.root = Path(temp.name)
        (cls.root / "scripts").mkdir()
        (cls.root / "src").mkdir()
        cls.script = cls.root / "scripts/encode_dense_mmret.sh"
        shutil.copyfile(PROJECT_ROOT / "scripts/encode_dense_mmret.sh", cls.script)
        cls.record = cls.root / "argv.bin"
        cls.visible = cls.root / "visible.txt"
        stub = cls.root / "fake-python"
        stub.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "$MMRET_TEST_ARGV"\n'
                        'printf "%s" "${CUDA_VISIBLE_DEVICES-unset}" > "$MMRET_TEST_VISIBLE"\n', encoding="utf-8")
        stub.chmod(0o755)
        keys = ("PYTHON_BIN", "MODEL_NAME_OR_PATH", "DATA_ROOT", "DATASET", "DATASET_SPLIT", "DENSE_OUTPUT_DIR",
                "BATCH_SIZE", "DTYPE", "NUM_PROCESSES", "MASTER_PORT", "TASK_TYPE", "ENCODE_TYPE", "CUDA_VISIBLE_DEVICES")
        cls.env = {key: value for key, value in os.environ.items() if key not in keys}
        cls.env.update(PYTHON_BIN=stub.as_posix(), MMRET_TEST_ARGV=cls.record.as_posix(),
                       MMRET_TEST_VISIBLE=cls.visible.as_posix(),
                       PATH=str(Path(BASH).parent) + os.pathsep + os.environ.get("PATH", ""))

    def launch(self, *args, **env):
        self.record.unlink(missing_ok=True)
        self.visible.unlink(missing_ok=True)
        result = subprocess.run([BASH, self.script.as_posix(), *args], env={**self.env, **env},
                                text=True, capture_output=True, timeout=30)
        argv = self.record.read_bytes().decode().rstrip("\0").split("\0") if self.record.exists() else []
        return result, argv

    def success(self, *args, **env):
        result, argv = self.launch(*args, **env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(argv)
        self.assertEqual(argv.count("--dense_model_type"), 1)
        self.assertEqual(argv[argv.index("--dense_model_type") + 1], "mmret")
        return argv

    def test_syntax_and_single_process_defaults(self):
        subprocess.run([BASH, "-n", self.script.as_posix()], check=True, timeout=30)
        argv = self.success()
        self.assertNotIn("deepspeed.launcher.runner", argv)
        for option, value in (("--dataset_name", "flickr"), ("--dataset_split", "test"),
                              ("--task_type", "ir"), ("--encode_type", "image"), ("--per_device_batch_size", "1")):
            self.assertEqual(argv[argv.index(option) + 1], value)
        self.assertIn("--fp16", argv)
        self.assertTrue(argv[0].endswith("/src/encode_dense.py"))

    def test_all_eight_datasets_and_ir_text_direction(self):
        for name, task, modality in (("coco", "ir", "image"), ("flickr", "ir", "image"),
                                     ("CUHK-PEDES", "tbpr", "image"), ("ICFG-PEDES", "tbpr", "image"),
                                     ("RSTPReid", "tbpr", "image"), ("webqa", "t2it", "corpus"),
                                     ("remuq", "it2t", "corpus"), ("llava", "it2t", "corpus")):
            with self.subTest(name=name):
                argv = self.success(DATASET=name)
                self.assertEqual(argv[argv.index("--task_type") + 1], task)
                self.assertEqual(argv[argv.index("--encode_type") + 1], modality)
        for name in ("coco", "flickr"):
            argv = self.success(DATASET=name, ENCODE_TYPE="text")
            self.assertEqual(argv[argv.index("--encode_type") + 1], "text")

    def test_cli_overrides_are_normalized_before_defaults(self):
        argv = self.success("--dataset_name=webqa", "--dataset_split", "val", "--data_root", "/new root",
                            "--model_name_or_path", "/weights/S2", "--per_device_batch_size=4",
                            "--prompt_type", "prompt_1", DATASET="flickr")
        for option, value in (("--dataset_name", "webqa"), ("--dataset_split", "val"), ("--task_type", "t2it"),
                              ("--encode_type", "corpus"), ("--data_root", "/new root"),
                              ("--model_name_or_path", "/weights/S2"), ("--per_device_batch_size", "4"),
                              ("--prompt_type", "prompt_1")):
            self.assertEqual(argv.count(option), 1)
            self.assertEqual(argv[argv.index(option) + 1], value)

    def test_dtype_flags_and_custom_output(self):
        for dtype, expected in (("float16", "--fp16"), ("bfloat16", "--bf16"), ("float32", None)):
            argv = self.success(DTYPE=dtype, DENSE_OUTPUT_DIR="/my output")
            self.assertEqual("--fp16" in argv, expected == "--fp16")
            self.assertEqual("--bf16" in argv, expected == "--bf16")
            self.assertEqual(argv[argv.index("--dense_output_dir") + 1], "/my output")

    def test_multi_process_uses_same_python_and_preserves_physical_ids(self):
        argv = self.success(NUM_PROCESSES="2", CUDA_VISIBLE_DEVICES="2,3,5", MASTER_PORT="29503")
        self.assertEqual(argv[:2], ["-m", "deepspeed.launcher.runner"])
        self.assertEqual(argv[argv.index("--include") + 1], "localhost:2,3")
        self.assertNotIn("--num_gpus", argv)
        self.assertEqual(argv[argv.index("--hostfile") + 1], "")
        self.assertEqual(argv[argv.index("--master_port") + 1], "29503")
        self.assertEqual(self.visible.read_text(), "unset")

    def test_no_visible_mask_and_single_process_mask(self):
        argv = self.success(NUM_PROCESSES="3")
        self.assertEqual(argv[argv.index("--num_gpus") + 1], "3")
        self.assertNotIn("--include", argv)
        self.success(CUDA_VISIBLE_DEVICES="4")
        self.assertEqual(self.visible.read_text(), "4")

    def test_invalid_configuration_fails_before_python(self):
        for env in ({"DATASET": "other"}, {"DATASET": "webqa", "ENCODE_TYPE": "image"},
                    {"DATASET": "CUHK-PEDES", "TASK_TYPE": "ir"}, {"BATCH_SIZE": "0"},
                    {"NUM_PROCESSES": "0"}, {"NUM_PROCESSES": "auto"}, {"MASTER_PORT": "65536"},
                    {"DTYPE": "int4"}, {"NUM_PROCESSES": "2", "CUDA_VISIBLE_DEVICES": "0"},
                    {"NUM_PROCESSES": "2", "CUDA_VISIBLE_DEVICES": "1,1"},
                    {"NUM_PROCESSES": "2", "CUDA_VISIBLE_DEVICES": "GPU-abc"}):
            with self.subTest(env=env):
                result, argv = self.launch(**env)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertEqual(argv, [])
        for args in (("--dataset_name",), ("--data_root=",), ("--dataset_name", "--fp16")):
            result, argv = self.launch(*args)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(argv, [])


if __name__ == "__main__":
    unittest.main()
