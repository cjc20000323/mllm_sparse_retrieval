"""Search launcher argv checks without weights, GPUs or DeepSpeed."""

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
class DenseSearchLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        cls.root = Path(temp.name)
        (cls.root / "scripts").mkdir()
        cls.script = cls.root / "scripts/search_dense.sh"
        shutil.copyfile(PROJECT_ROOT / "scripts/search_dense.sh", cls.script)
        cls.record = cls.root / "argv.bin"
        cls.visible = cls.root / "visible.txt"
        stub = cls.root / "fake-python"
        stub.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "$DENSE_TEST_ARGV"\n'
                        'printf "%s" "${CUDA_VISIBLE_DEVICES-unset}" > "$DENSE_TEST_VISIBLE"\n', encoding="utf-8")
        stub.chmod(0o755)
        keys = ("PYTHON_BIN", "MODEL_NAME_OR_PATH", "DENSE_MODEL_TYPE", "DATA_ROOT", "DATASET", "DATASET_SPLIT",
                "PASSAGE_REPS", "SAVE_DIR", "BATCH_SIZE", "DEPTH", "DTYPE", "NUM_PROCESSES", "GPUS",
                "MASTER_PORT", "TASK_TYPE", "QUERY_TYPE", "CUDA_VISIBLE_DEVICES")
        cls.env = {key: value for key, value in os.environ.items() if key not in keys}
        cls.env.update(PYTHON_BIN=stub.as_posix(), DENSE_TEST_ARGV=cls.record.as_posix(),
                       DENSE_TEST_VISIBLE=cls.visible.as_posix(), PASSAGE_REPS="/matching corpus",
                       PATH=str(Path(BASH).parent) + os.pathsep + os.environ.get("PATH", ""))

    def launch(self, *args, **env):
        self.record.unlink(missing_ok=True)
        result = subprocess.run([BASH, self.script.as_posix(), *args], env={**self.env, **env},
                                text=True, capture_output=True, timeout=30)
        argv = self.record.read_bytes().decode().rstrip("\0").split("\0") if self.record.exists() else []
        return result, argv

    def success(self, *args, **env):
        result, argv = self.launch(*args, **env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(argv)
        self.assertTrue(any(value.endswith("/src/search_dense.py") for value in argv))
        return argv

    def test_syntax_defaults_and_entrypoint(self):
        subprocess.run([BASH, "-n", self.script.as_posix()], check=True, timeout=30)
        argv = self.success()
        self.assertNotIn("deepspeed.launcher.runner", argv)
        for key, value in (("--dataset_name", "flickr"), ("--query_type", "text"), ("--task_type", "ir"),
                           ("--passage_reps", "/matching corpus")):
            self.assertEqual(argv[argv.index(key) + 1], value)
        self.assertIn("--fp16", argv)
        self.assertNotIn("--remove_query", argv)

    def test_all_datasets_and_cli_overrides(self):
        for name, task, query in (("coco", "ir", "text"), ("flickr", "ir", "text"),
                                  ("CUHK-PEDES", "tbpr", "text"), ("ICFG-PEDES", "tbpr", "text"),
                                  ("RSTPReid", "tbpr", "text"), ("webqa", "t2it", "text"),
                                  ("remuq", "it2t", "image"), ("llava", "it2t", "image")):
            argv = self.success(DATASET=name)
            self.assertEqual(argv[argv.index("--task_type") + 1], task)
            self.assertEqual(argv[argv.index("--query_type") + 1], query)
        argv = self.success("--dataset_name=remuq", "--query_type", "composed", "--dense_model_type", "unime",
                            "--model_name_or_path", "/renamed weights", "--passage_reps=/my corpus",
                            "--save_dir", "/my results", "--dataset_split=val", "--mmret_query_instruction", "a task")
        for key, value in (("--query_type", "composed"), ("--dense_model_type", "unime"),
                           ("--passage_reps", "/my corpus"), ("--save_dir", "/my results"),
                           ("--dataset_split", "val"), ("--mmret_query_instruction", "a task")):
            self.assertEqual(argv.count(key), 1)
            self.assertEqual(argv[argv.index(key) + 1], value)

    def test_precision_and_physical_gpu_selection(self):
        for dtype, flag in (("float32", None), ("bfloat16", "--bf16"), ("float16", "--fp16")):
            argv = self.success(DTYPE=dtype)
            self.assertEqual("--fp16" in argv, flag == "--fp16")
            self.assertEqual("--bf16" in argv, flag == "--bf16")
        argv = self.success(NUM_PROCESSES="2", CUDA_VISIBLE_DEVICES="2,3,5", MASTER_PORT="29503")
        self.assertEqual(argv[:2], ["-m", "deepspeed.launcher.runner"])
        self.assertEqual(argv[argv.index("--include") + 1], "localhost:2,3")
        self.assertEqual(self.visible.read_text(), "unset")
        argv = self.success(NUM_PROCESSES="2")
        self.assertEqual(argv[argv.index("--num_gpus") + 1], "2")

    def test_invalid_options_fail_before_launch(self):
        for env in ({"PASSAGE_REPS": ""}, {"DATASET": "bad"}, {"DATASET": "webqa", "QUERY_TYPE": "image"},
                    {"DATASET": "remuq", "TASK_TYPE": "ir"}, {"BATCH_SIZE": "0"}, {"DEPTH": "0"},
                    {"DTYPE": "bad"}, {"NUM_PROCESSES": "0"}, {"MASTER_PORT": "65536"},
                    {"NUM_PROCESSES": "2", "CUDA_VISIBLE_DEVICES": "1,1"},
                    {"NUM_PROCESSES": "2", "CUDA_VISIBLE_DEVICES": "3"}):
            result, argv = self.launch(**env)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertEqual(argv, [])
        for args in (("--dataset_name",), ("--passage_reps=",), ("--data_root", "--fp16")):
            result, argv = self.launch(*args)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(argv, [])


if __name__ == "__main__":
    unittest.main()
