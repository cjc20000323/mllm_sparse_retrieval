"""Run the standalone FreeRet estimate launcher with recording GPU/DS stubs."""

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

import test_freeret_time_estimation as fixtures


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "estimate_freeret_retrieval_time.sh"
BASH = shutil.which("bash") or next((str(path) for path in (
    Path("D:/Git/bin/bash.exe"), Path("C:/Program Files/Git/bin/bash.exe")
) if path.is_file()), None)
freeret = fixtures.freeret


@unittest.skipUnless(BASH, "Bash is required for launcher tests")
class FreeRetTimeLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="freeret time launcher ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "scripts").mkdir()
        (self.root / "src").mkdir()
        self.script = self.root / "scripts" / SCRIPT
        shutil.copyfile(PROJECT_ROOT / "scripts" / SCRIPT, self.script)
        # No other launcher or Python entrypoint is available in this fake repo.
        (self.root / "src" / "freeret_retrieval.py").touch()
        self.capture = self.root / "argv.bin"
        self.visible = self.root / "visible.txt"
        self.stub = self.root / "python stub"
        self.stub.write_text(r'''#!/usr/bin/env bash
if [[ "${1:-}" == -c ]]; then
  case "$2" in
    *torch.cuda.device_count*) printf '%s\n' "${FREERET_TEST_GPU_COUNT:-4}"; exit "${FREERET_TEST_CUDA_ERROR:-0}" ;;
    *deepspeed.launcher.runner*) exit "${FREERET_TEST_DS_ERROR:-0}" ;;
    *) exit 9 ;;
  esac
else
  printf '%s\0' "$@" > "$FREERET_TEST_ARGV"
  printf '%s' "${CUDA_VISIBLE_DEVICES-unset}" > "$FREERET_TEST_VISIBLE"
  exit "${FREERET_TEST_EXIT:-0}"
fi
''', encoding="utf-8")
        self.stub.chmod(0o755)
        self.env = dict(os.environ)
        variables = set(re.findall(r"\$\{([A-Z_][A-Z_0-9]*):-", self.script.read_text(encoding="utf-8")))
        for key in variables | {"CUDA_VISIBLE_DEVICES", "ESTIMATE_TIME_ONLY"}:
            self.env.pop(key, None)
        for key in list(self.env):
            if key.startswith("FREERET_TEST_"):
                self.env.pop(key)
        self.env.update(
            PYTHON_BIN=self.stub.as_posix(),
            FREERET_TEST_ARGV=self.capture.as_posix(),
            FREERET_TEST_VISIBLE=self.visible.as_posix(),
        )

    def launch(self, *args, **overrides):
        self.capture.unlink(missing_ok=True)
        self.visible.unlink(missing_ok=True)
        result = subprocess.run(
            [BASH, self.script.as_posix(), *args],
            cwd=self.root.parent, env={**self.env, **overrides},
            capture_output=True, text=True, timeout=20,
        )
        argv = self.capture.read_bytes().decode("utf-8").split("\0")[:-1] if self.capture.exists() else []
        return result, argv

    def success(self, **overrides):
        result, argv = self.launch(**overrides)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(argv[:2], ["-m", "deepspeed.launcher.runner"])
        entries = [index for index, value in enumerate(argv) if value.endswith("/src/freeret_retrieval.py")]
        self.assertEqual(len(entries), 1)
        args = freeret.build_parser().parse_args(argv[entries[0] + 1:])
        freeret.validate_args(args)
        self.assertTrue(args.estimate_time_only)
        self.assertEqual(argv.count("--estimate_time_only"), 1)
        self.assertFalse(any("three_stage" in value for value in argv))
        self.assertEqual(self.visible.read_text(), "unset")
        return result, argv, args

    @staticmethod
    def value(argv, option):
        return argv[argv.index(option) + 1]

    def test_syntax_standalone_entrypoint_and_defaults(self):
        subprocess.run([BASH, "-n", self.script.as_posix()], check=True, timeout=20)
        result, argv, args = self.success()
        self.assertEqual(args.datasets, "all")
        self.assertEqual((args.split, args.direction), ("test", "auto"))
        self.assertEqual(args.model_path, "/root/autodl-fs/checkpoints/Qwen2.5-VL-7B-Instruct")
        self.assertEqual(args.data_root, "/root/autodl-tmp")
        entrypoint = next(value for value in argv if value.endswith("/src/freeret_retrieval.py"))
        # Git Bash can expose Windows temporary directories as /tmp rather than
        # their native C:/... paths; compare within the launcher's own namespace.
        shell_root = entrypoint.removesuffix("/src/freeret_retrieval.py")
        self.assertEqual(args.output_dir, shell_root + "/results/freeret_time_estimates")
        self.assertEqual((args.dtype, args.attn_implementation, args.device_map), ("bfloat16", "sdpa", "auto"))
        self.assertEqual((args.target_batch_size, args.query_batch_size, args.rerank_batch_size), (8, 4, 2))
        self.assertEqual((args.rerank_num, args.stage1_topk), (50, 50))
        self.assertEqual((args.max_pixels, args.similarity_chunk_size, args.max_target_gpu_gb), (0, 8192, 4))
        self.assertEqual((args.max_queries, args.seed, args.metrics_ks), (-1, 2026, [1, 5, 10]))
        self.assertFalse(args.cache_targets)
        self.assertEqual(self.value(argv, "--include"), "localhost:0,1,2,3")
        self.assertEqual(self.value(argv, "--hostfile"), "")
        self.assertEqual(self.value(argv, "--master_addr"), "127.0.0.1")
        self.assertEqual(self.value(argv, "--master_port"), "29500")
        self.assertNotIn("--num_gpus", argv)
        self.assertIn("time_estimate_summary.json", result.stdout)

    def test_all_eight_datasets_resolve_to_ten_auto_direction_tasks(self):
        _, _, args = self.success()
        expected = {
            "coco": ["t2i", "i2t"], "flickr": ["t2i", "i2t"],
            "CUHK-PEDES": ["t2i"], "ICFG-PEDES": ["t2i"], "RSTPReid": ["t2i"],
            "webqa": ["t2it"], "remuq": ["it2t"], "llava": ["it2t"],
        }
        names = freeret.parse_dataset_names(args.datasets)
        self.assertEqual(set(names), set(expected))
        tasks = [(name, direction) for name in names for direction in freeret.directions_for(name, args.direction)]
        self.assertEqual(len(tasks), 10)
        for name, directions in expected.items():
            with self.subTest(dataset=name):
                _, _, selected = self.success(DATASETS=name)
                self.assertEqual(freeret.directions_for(selected.datasets, selected.direction), directions)

    def test_environment_overrides_and_spaces_are_preserved(self):
        _, argv, args = self.success(
            DATASETS="coco,flickr", SPLIT="val", DIRECTION="i2t", DTYPE="float16",
            MODEL_PATH="/models/my checkpoint", DATASET_ROOT="/datasets/root with spaces",
            OUTPUT_DIR="/output/time estimates", DEVICE_MAP="cuda:7", ATTN_IMPLEMENTATION="eager",
            MAX_PIXELS="128", TARGET_BATCH_SIZE="2", QUERY_BATCH_SIZE="3", RERANK_BATCH_SIZE="4",
            RERANK_NUM="20", STAGE1_TOPK="30", SIMILARITY_CHUNK_SIZE="1024", MAX_TARGET_GPU_GB="1.5",
            METRICS_KS="1 10 20", MAX_QUERIES="60", SEED="42", CACHE_TARGETS="1", MASTER_PORT="29517",
        )
        expected = {
            "datasets": "coco,flickr", "split": "val", "direction": "i2t", "dtype": "float16",
            "model_path": "/models/my checkpoint", "data_root": "/datasets/root with spaces",
            "output_dir": "/output/time estimates", "device_map": "cuda:7", "attn_implementation": "eager",
            "max_pixels": 128, "target_batch_size": 2, "query_batch_size": 3, "rerank_batch_size": 4,
            "rerank_num": 20, "stage1_topk": 30, "similarity_chunk_size": 1024, "max_target_gpu_gb": 1.5,
            "metrics_ks": [1, 10, 20], "max_queries": 60, "seed": 42, "cache_targets": True,
        }
        for key, value in expected.items():
            self.assertEqual(getattr(args, key), value, key)
        self.assertEqual(self.value(argv, "--master_port"), "29517")

    def test_inherited_full_evaluation_setting_cannot_disable_estimate_only(self):
        for value in ("false", "0", "true", "1"):
            with self.subTest(value=value):
                self.success(ESTIMATE_TIME_ONLY=value)

    def test_auto_manual_single_worker_and_physical_gpu_selection(self):
        for count, requested, ids, selected in (
            ("4", "auto", None, "0,1,2,3"), ("4", "2", None, "0,1"),
            ("4", "1", None, "0"), ("1", "auto", None, "0"),
            ("3", "auto", "2,4,7", "2,4,7"), ("3", "2", "2,4,7", "2,4"),
            ("3", "1", "7,4,2", "7"),
        ):
            with self.subTest(count=count, requested=requested, ids=ids):
                env = {"FREERET_TEST_GPU_COUNT": count, "NUM_PROCESSES": requested}
                if ids is not None:
                    env["CUDA_VISIBLE_DEVICES"] = ids
                result, argv, _ = self.success(**env)
                self.assertEqual(self.value(argv, "--include"), "localhost:" + selected)
                self.assertIn("DEVICE_MAP is ignored", result.stdout)

    def test_invalid_process_ports_nodes_and_gpu_masks_fail_before_launch(self):
        cases = [({"NUM_PROCESSES": value}, "NUM_PROCESSES") for value in
                 ("0", "-1", "text", "5", "999999999999999999999")]
        cases += [({"MASTER_PORT": value}, "MASTER_PORT") for value in ("0", "65536", "text", "-1")]
        cases += [({"NUM_NODES": "2"}, "NUM_NODES")]
        cases += [({"CUDA_VISIBLE_DEVICES": value}, "CUDA_VISIBLE_DEVICES") for value in
                  ("", "1,1,2,3", "0,,1", "0,", "01,2", "GPU-123", "MIG-123", "2,3")]
        for env, message in cases:
            with self.subTest(env=env):
                result, argv = self.launch(**env)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn(message, result.stderr)
                self.assertEqual(argv, [])

    def test_missing_gpu_or_deepspeed_never_falls_back_to_python(self):
        for env, message in (
            ({"FREERET_TEST_GPU_COUNT": "0"}, "at least one visible CUDA GPU"),
            ({"FREERET_TEST_CUDA_ERROR": "1"}, "Cannot query GPUs"),
            ({"FREERET_TEST_GPU_COUNT": "invalid"}, "Invalid visible GPU count"),
            ({"FREERET_TEST_DS_ERROR": "1"}, "DeepSpeed could not be imported"),
        ):
            with self.subTest(env=env):
                result, argv = self.launch(**env)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn(message, result.stderr)
                self.assertEqual(argv, [])

    def test_launcher_exit_status_is_preserved(self):
        result, argv = self.launch(FREERET_TEST_EXIT="7")
        self.assertEqual(result.returncode, 7)
        self.assertTrue(argv)

    def test_unrecognized_cli_arguments_are_not_silently_ignored(self):
        result, argv = self.launch("--datasets", "webqa")
        self.assertEqual(result.returncode, 2)
        self.assertIn("environment variables", result.stderr)
        self.assertEqual(argv, [])


if __name__ == "__main__":
    unittest.main()
