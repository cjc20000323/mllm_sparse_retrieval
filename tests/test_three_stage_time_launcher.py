"""Standalone estimation launcher checks using fake CUDA/DeepSpeed probes."""

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unittest

import test_three_stage_launcher_paths as fixtures

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import estimate_three_stage_retrieval_time as estimate

BASH = fixtures.BASH
SCRIPT = "estimate_three_stage_retrieval_time.sh"


@unittest.skipUnless(BASH, "Bash is required for launcher tests")
class TimeLauncherTests(unittest.TestCase):
    index_relative = staticmethod(fixtures.ThreeStageLauncherPathTests.index_relative)
    make_dense = staticmethod(fixtures.ThreeStageLauncherPathTests.make_dense)
    make_sparse = staticmethod(fixtures.ThreeStageLauncherPathTests.make_sparse)
    launch = fixtures.ThreeStageLauncherPathTests.launch

    def setUp(self):
        fixtures.ThreeStageLauncherPathTests.setUp(self)
        self.script = self.root / "scripts" / SCRIPT
        shutil.copyfile(PROJECT_ROOT / "scripts" / SCRIPT, self.script)
        (self.root / "src/estimate_three_stage_retrieval_time.py").touch()
        self.visible = self.root / "visible.txt"
        stub = self.root / "fake-python"
        stub.write_text('''#!/usr/bin/env bash
if [[ "$1" == -c ]]; then
    case "$2" in
        *torch.cuda.device_count*) printf '%s\n' "${ESTIMATE_TEST_GPU_COUNT:-3}" ;;
        *deepspeed.launcher.runner*) [[ "${ESTIMATE_TEST_NO_DEEPSPEED:-0}" == 0 ]] ;;
        *) exit 9 ;;
    esac
else
    printf '%s\\0' "$@" > "$PIPELINE_TEST_ARGV"
    printf '%s' "${CUDA_VISIBLE_DEVICES-unset}" > "$ESTIMATE_TEST_VISIBLE"
fi
''', encoding="utf-8")
        stub.chmod(0o755)
        # Do not inherit a user's launch configuration into argv-only tests.
        keys = set(re.findall(r"\$\{([A-Z_][A-Z_0-9]*):-", self.script.read_text(encoding="utf-8")))
        for key in keys | {"GPUS", "CUDA_VISIBLE_DEVICES", "ESTIMATE_TEST_GPU_COUNT", "ESTIMATE_TEST_NO_DEEPSPEED"}:
            self.env.pop(key, None)
        self.env.update(DATA_ROOT=self.root.as_posix(), BASE_MODEL="model", ARTIFACT_MODEL_NAME="model",
                        RETRIEVAL_MODEL_PATH=(self.root / "checkpoint").as_posix(),
                        RANKER_MODEL_PATH=(self.root / "checkpoint").as_posix(),
                        PYTHON_BIN=stub.as_posix(), ESTIMATE_TEST_VISIBLE=self.visible.as_posix())

    @staticmethod
    def value(argv, option):
        return argv[argv.index(option) + 1]

    def success(self, *args, **env):
        result, argv = self.launch(*args, **env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(argv[:2], ["-m", "deepspeed.launcher.runner"])
        start = next(index for index, value in enumerate(argv) if value.endswith("/src/estimate_three_stage_retrieval_time.py"))
        parsed = estimate.parser().parse_args(argv[start + 1:])
        return result, argv, parsed

    def test_syntax_defaults_and_standalone_launcher(self):
        subprocess.run([BASH, "-n", self.script.as_posix()], check=True, timeout=30)
        result, argv, args = self.success()
        self.assertEqual(args.dataset_name, "webqa")
        self.assertEqual(args.task_type, "t2it")
        self.assertEqual(args.query_type, "text")
        self.assertEqual(args.retrieval_dtype, "float16")
        self.assertEqual(args.rerank_num, 5)
        self.assertEqual(args.rerank_batch_size, 1)
        self.assertEqual(args.retrieval_batch_size, 1)
        self.assertEqual(self.value(argv, "--include"), "localhost:0,1,2")
        self.assertEqual(self.value(argv, "--hostfile"), "")
        self.assertNotIn("--num_gpus", argv)
        self.assertTrue(args.output_dir.startswith("./three_stage_time_estimates/"))
        self.assertIn("time_estimate.json", result.stdout)

    def test_all_eight_dataset_index_paths(self):
        for name, (task, query, target, direction, _, _) in fixtures.PROFILES.items():
            result, argv, args = self.success(DATASET=name)
            relative = self.index_relative(name, target, tbpr_type="origin_type" if task == "tbpr" else None)
            self.assertEqual(args.task_type, task)
            self.assertEqual(args.query_type, query)
            self.assertEqual(args.stage2_dense_index, "dense_output/" + relative)
            self.assertEqual(args.stage1_sparse_index, "sparse_output/" + relative)
            self.assertEqual(args.stage2_sparse_index, args.stage1_sparse_index)
            self.assertIn(f"/{name}/{direction}/test/sparse_concat_stage1", args.output_dir)
            self.assertEqual(argv.count("--dataset-name"), 1)

    def test_reverse_ir_tbpr_view_and_composed_directions(self):
        for name in ("coco", "flickr"):
            _, _, args = self.success("--query-type", "text", DATASET=name)
            self.assertIn(f"/{name}/image/", args.stage2_dense_index)
            self.assertIn("/text_to_image/", args.output_dir)
        _, _, args = self.success("--dataset-name", "RSTPReid", "--tbpr-type=type")
        self.assertIn("/test/type/", args.stage2_sparse_index)
        self.assertTrue(args.output_dir.endswith("/type"))
        _, _, args = self.success("--dataset-name=remuq", "--query-type=composed")
        self.assertEqual(args.query_type, "composed")
        self.assertEqual(args.task_type, "it2t")

    def test_explicit_paths_and_cli_last_value_override_environment(self):
        dense = self.make_dense(self.root / "custom dense")
        sparse = self.make_sparse(self.root / "custom sparse")
        _, _, args = self.success("--", "--dataset-name=webqa", "--dataset-name", "llava", "--dataset-split=val",
            "--stage2-dense-index", dense.as_posix(), "--stage2-sparse-index", sparse.as_posix(),
            "--stage1-sparse-index", sparse.as_posix(), "--output-dir", "/explicit output",
            "--rerank-num=17", "--rerank-num", "20", "--retrieval-batch-size=2", "--rerank-batch-size=3",
            "--overwrite", DATASET="coco", RERANK_NUM="5", OUTPUT_DIR="/wrong output")
        self.assertEqual(args.dataset_name, "llava")
        self.assertEqual(args.dataset_split, "val")
        self.assertEqual(args.rerank_num, 20)
        self.assertEqual(args.stage2_dense_index, dense.as_posix())
        self.assertEqual(args.stage1_sparse_index, sparse.as_posix())
        self.assertEqual(args.output_dir, "/explicit output")
        self.assertTrue(args.overwrite)
        self.assertEqual((args.retrieval_batch_size, args.rerank_batch_size), (2, 3))

    def test_physical_gpu_ids_auto_explicit_alias_and_single_worker(self):
        for settings, expected in (({}, "0,1,2"), ({"GPUS": "2"}, "0,1"),
                ({"GPUS": "1", "NUM_PROCESSES": "2"}, "0,1"),
                ({"CUDA_VISIBLE_DEVICES": "2,4,7", "NUM_PROCESSES": "2"}, "2,4"),
                ({"CUDA_VISIBLE_DEVICES": "2,4,7", "NUM_PROCESSES": "1"}, "2")):
            _, argv, _ = self.success(MASTER_PORT="29503", **settings)
            self.assertEqual(self.value(argv, "--include"), "localhost:" + expected)
            self.assertEqual(self.value(argv, "--master_port"), "29503")
            self.assertEqual(self.visible.read_text(), "unset")

    def test_filtered_numeric_labels_and_model_overrides(self):
        relative = "alternate/flickr/text/filter/concat/prompt_custom/val/3_no_manual_40_sum_no_cluster_before_pad_disassembleeol_separate_upper_mean"
        self.make_dense(self.root / "dense_output" / relative)
        self.make_sparse(self.root / "sparse_output" / relative)
        _, argv, args = self.success("--dataset-name=flickr", "--dataset-split=val", "--is-filtered",
            "--sparse-value-mean", "--num-expended-tokens=003", "--sparse-length=040", "--prompt-type=prompt_custom",
            "--retrieval-reps-loc=before_pad", "--sparse-lower-or-upper=upper", "--artifact-model-name=alternate",
            "--retrieval-model-name-or-path", (self.root / "checkpoint").as_posix(),
            "--retrieval-lora-model-path", "/adapter snapshot", "--retrieval-dtype=bfloat16")
        self.assertEqual(args.stage2_dense_index, "dense_output/" + relative)
        self.assertTrue(args.is_filtered and args.sparse_value_mean)
        self.assertEqual(args.retrieval_lora_model_path, "/adapter snapshot")
        self.assertEqual(args.retrieval_dtype, "bfloat16")
        self.assertEqual(argv.count("--sparse-length"), 1)

    def test_missing_artifacts_and_invalid_configurations_fail_before_launch(self):
        for env in ({"NUM_PROCESSES": "0"}, {"NUM_PROCESSES": "4"}, {"NUM_NODES": "2"},
                    {"DATASET": "unknown"}, {"RERANK_TYPE": "listwise"}, {"MASTER_PORT": "65536"},
                    {"CUDA_VISIBLE_DEVICES": "2,2,3"}, {"CUDA_VISIBLE_DEVICES": "GPU-id"},
                    {"CUDA_VISIBLE_DEVICES": "2,3"}, {"ESTIMATE_TEST_GPU_COUNT": "0"},
                    {"ESTIMATE_TEST_NO_DEEPSPEED": "1"}, {"STAGE2_DENSE_INDEX": "/missing"}):
            result, argv = self.launch(**env)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(argv, [])
        for argv in (("--dataset-name",), ("--stage2-dense-index=",), ("--query-type", "image"),
                     ("--overwrite=false",), ("--is-filtered=true",)):
            result, recorded = self.launch(*argv)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(recorded, [])


if __name__ == "__main__":
    unittest.main()
