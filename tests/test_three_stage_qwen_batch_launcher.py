"""Run the new batch launcher with recording children; standard library only."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAME = "three_stage_retrieval_pipeline_qwen.sh"
SCRIPT = PROJECT_ROOT / "scripts" / SCRIPT_NAME
EXISTING = PROJECT_ROOT / "scripts/three_stage_retrieval_pipeline.sh"
BASH = shutil.which("bash") or next((str(path) for path in (
    Path("D:/Git/bin/bash.exe"), Path("C:/Program Files/Git/bin/bash.exe")
) if path.is_file()), None)
FIELDS = (
    "DATASET", "TASK_TYPE", "QUERY_TYPE", "BASE_MODEL", "ARTIFACT_MODEL_NAME",
    "RETRIEVAL_MODEL_PATH", "RANKER_MODEL_PATH", "OUTPUT_DIR", "STAGE1_SPARSE_INDEX",
    "STAGE2_DENSE_INDEX", "STAGE2_SPARSE_INDEX", "GPUS", "DATA_ROOT", "DATASET_SPLIT",
    "RETRIEVAL_DTYPE", "RANKER_DTYPE", "RETRIEVAL_QWEN_MIN_PIXELS", "RETRIEVAL_QWEN_MAX_PIXELS",
    "RANKER_QWEN_MIN_PIXELS", "RANKER_QWEN_MAX_PIXELS", "RERANK_TEMPLATE", "SPARSE_LENGTH", "ARGC",
)


@unittest.skipUnless(BASH, "Bash is required for batch launcher integration tests")
class QwenBatchLauncherTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="qwen batch launcher ")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "scripts").mkdir()
        self.script = self.root / "scripts" / SCRIPT_NAME
        shutil.copyfile(SCRIPT, self.script)
        self.child = self.root / "scripts/three_stage_retrieval_pipeline.sh"
        values = " ".join('"${' + field + '-}"' for field in FIELDS[:-1]) + ' "$#"'
        self.child.write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\n'
            "printf '%s\\0' " + values + ' >> "$BATCH_TEST_RECORD"\n'
            'if [[ "${BATCH_TEST_FAIL_DATASET-}" == "$DATASET" ]]; then\n'
            '  exit "${BATCH_TEST_FAIL_CODE:-7}"\nfi\n', encoding="utf-8",
        )
        self.record = self.root / "jobs.bin"
        names = set(re.findall(r"\$\{([A-Z_][A-Z_0-9]*)(?::-|-)", SCRIPT.read_text(encoding="utf-8") + EXISTING.read_text(encoding="utf-8")))
        names.update(FIELDS)
        self.env = {key: value for key, value in os.environ.items() if key not in names}
        self.env.update(
            QWEN_CHECKPOINT_ROOT=(self.root / "models").as_posix(),
            QWEN_OUTPUT_ROOT=(self.root / "results").as_posix(),
            DATA_ROOT=(self.root / "datasets").as_posix(), BATCH_TEST_RECORD=self.record.as_posix(),
            PATH=str(Path(BASH).parent) + os.pathsep + os.environ.get("PATH", ""),
        )

    def launch(self, *args, **overrides):
        self.record.unlink(missing_ok=True)
        result = subprocess.run([BASH, self.script.as_posix(), *args], cwd=self.root.parent,
                                env={**self.env, **overrides}, capture_output=True, text=True, timeout=30)
        values = self.record.read_bytes().decode().split("\0")[:-1] if self.record.exists() else []
        self.assertEqual(len(values) % len(FIELDS), 0)
        jobs = [dict(zip(FIELDS, values[start:start + len(FIELDS)])) for start in range(0, len(values), len(FIELDS))]
        return result, jobs

    def success(self, *args, **overrides):
        result, jobs = self.launch(*args, **overrides)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result, jobs

    def test_syntax_and_help_need_no_model_or_jobs(self):
        result = subprocess.run([BASH, "-n", self.script.as_posix()], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ("-h", "--help"):
            result, jobs = self.success(flag)
            self.assertIn("QWEN_FAMILY", result.stdout)
            self.assertFalse(jobs)

    def test_defaults_run_two_datasets_with_independent_outputs(self):
        result, jobs = self.success()
        self.assertEqual([job["DATASET"] for job in jobs], ["flickr", "coco"])
        checkpoint = (self.root / "models/Qwen2.5-VL-7B-Instruct").as_posix()
        for job in jobs:
            self.assertEqual(job["RETRIEVAL_MODEL_PATH"], checkpoint)
            self.assertEqual(job["RANKER_MODEL_PATH"], checkpoint)
            self.assertEqual(job["BASE_MODEL"], "Qwen2.5-VL-7B-Instruct")
            self.assertEqual(job["ARTIFACT_MODEL_NAME"], job["BASE_MODEL"])
            self.assertEqual(job["QUERY_TYPE"], "image")
            self.assertEqual(job["GPUS"], "1")
            self.assertEqual(job["ARGC"], "0")
            self.assertEqual(job["OUTPUT_DIR"], (self.root / "results/Qwen2.5-VL-7B-Instruct" / job["DATASET"] / "image_to_text/test").as_posix())
            for field in ("RETRIEVAL_QWEN_MIN_PIXELS", "RETRIEVAL_QWEN_MAX_PIXELS", "RANKER_QWEN_MIN_PIXELS", "RANKER_QWEN_MAX_PIXELS"):
                self.assertEqual(job[field], "0")
        self.assertLess(result.stdout.index("Completed: flickr"), result.stdout.index("Completed: coco"))

    def test_each_family_and_custom_model_path_with_spaces(self):
        for family, name in (("qwen2", "Qwen2-VL-7B-Instruct"), ("qwen25", "Qwen2.5-VL-7B-Instruct"),
                             ("qwen3", "Qwen3-VL-8B-Instruct")):
            with self.subTest(family=family):
                _, jobs = self.success(QWEN_FAMILY=family, DATASETS="flickr")
                self.assertEqual(jobs[0]["RETRIEVAL_MODEL_PATH"], (self.root / "models" / name).as_posix())
        custom = (self.root / "custom Qwen weights").as_posix()
        _, jobs = self.success(QWEN_FAMILY="qwen3", QWEN_MODEL_PATH=custom + "/", GPUS="2")
        self.assertEqual(len(jobs), 2)
        for job in jobs:
            self.assertEqual(job["RETRIEVAL_MODEL_PATH"], custom)
            self.assertEqual(job["RANKER_MODEL_PATH"], custom)
            self.assertEqual(job["ARTIFACT_MODEL_NAME"], "custom Qwen weights")
            self.assertEqual(job["GPUS"], "2")

    def test_mixed_datasets_and_ir_direction_are_selected_per_job(self):
        names = ["coco", "flickr", "CUHK-PEDES", "ICFG-PEDES", "RSTPReid", "webqa", "remuq", "llava"]
        _, jobs = self.success(DATASETS="\t" + "\n  ".join(names) + "  ", IR_QUERY_TYPE="text", DATASET_SPLIT="val")
        self.assertEqual([job["DATASET"] for job in jobs], names)
        self.assertEqual([job["QUERY_TYPE"] for job in jobs], ["text"] * 6 + ["image"] * 2)
        self.assertEqual(len(set(job["OUTPUT_DIR"] for job in jobs)), 8)
        self.assertTrue(all(job["TASK_TYPE"] == "" for job in jobs))
        self.assertTrue(all(job["DATASET_SPLIT"] == "val" for job in jobs))
        self.assertIn("text_to_image/val", jobs[0]["OUTPUT_DIR"])
        self.assertTrue(jobs[2]["OUTPUT_DIR"].endswith("text_to_image/val/origin_type"))

    def test_parent_job_settings_cannot_override_dataset_model_or_paths(self):
        inherited = {field: "stale-value" for field in (
            "DATASET", "TASK_TYPE", "QUERY_TYPE", "BASE_MODEL", "ARTIFACT_MODEL_NAME", "RETRIEVAL_MODEL_PATH",
            "RANKER_MODEL_PATH", "OUTPUT_DIR", "STAGE1_SPARSE_INDEX", "STAGE2_DENSE_INDEX", "STAGE2_SPARSE_INDEX",
        )}
        _, jobs = self.success(**inherited, DATASETS="flickr webqa", RERANK_TEMPLATE="what_caption_generation",
                               SPARSE_LENGTH="64", RETRIEVAL_QWEN_MIN_PIXELS="64", RETRIEVAL_QWEN_MAX_PIXELS="256",
                               RANKER_QWEN_MIN_PIXELS="144", RANKER_QWEN_MAX_PIXELS="512")
        self.assertEqual([job["DATASET"] for job in jobs], ["flickr", "webqa"])
        self.assertEqual([job["QUERY_TYPE"] for job in jobs], ["image", "text"])
        for job in jobs:
            self.assertEqual(job["TASK_TYPE"], "")
            self.assertTrue(all(job[field] == "" for field in ("STAGE1_SPARSE_INDEX", "STAGE2_DENSE_INDEX", "STAGE2_SPARSE_INDEX")))
            self.assertNotIn("stale-value", job.values())
            self.assertEqual(job["RERANK_TEMPLATE"], "what_caption_generation")
            self.assertEqual(job["SPARSE_LENGTH"], "64")
            self.assertEqual([job[field] for field in FIELDS[16:20]], ["64", "256", "144", "512"])

    def test_custom_index_tables_do_not_leak_into_other_jobs(self):
        dense = (self.root / "my dense/flickr").as_posix()
        sparse = (self.root / "my sparse/flickr").as_posix()
        text = self.script.read_text(encoding="utf-8")
        text = text.replace("declare -A DENSE_INDEXES=()", "declare -A DENSE_INDEXES=([flickr]=" + shlex.quote(dense) + ")")
        text = text.replace("declare -A SPARSE_INDEXES=()", "declare -A SPARSE_INDEXES=([flickr]=" + shlex.quote(sparse) + ")")
        self.script.write_text(text, encoding="utf-8")
        _, jobs = self.success(STAGE2_DENSE_INDEX="wrong-global", STAGE2_SPARSE_INDEX="wrong-global")
        self.assertEqual(jobs[0]["STAGE2_DENSE_INDEX"], dense)
        self.assertEqual(jobs[0]["STAGE1_SPARSE_INDEX"], sparse)
        self.assertEqual(jobs[0]["STAGE2_SPARSE_INDEX"], sparse)
        self.assertEqual(jobs[1]["STAGE2_DENSE_INDEX"], "")
        self.assertEqual(jobs[1]["STAGE2_SPARSE_INDEX"], "")

    def test_invalid_configuration_and_unchecked_cli_launch_no_jobs(self):
        for settings in ({"DATASETS": ""}, {"DATASETS": "  \t"}, {"DATASETS": "flickr flickr"},
                         {"DATASETS": "flickr unknown"}, {"QWEN_FAMILY": "llava"},
                         {"IR_QUERY_TYPE": "composed"}, {"GPUS": "0"}):
            with self.subTest(settings=settings):
                result, jobs = self.launch(**settings)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertFalse(jobs)
        for args in (("--dataset-name", "webqa"), ("--output-dir=/tmp/shared",), ("--",), ("--help", "--dataset-name=coco")):
            with self.subTest(args=args):
                result, jobs = self.launch(*args)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(jobs)

    def test_failure_stops_batch_and_preserves_exit_status(self):
        result, jobs = self.launch(DATASETS="flickr coco webqa", BATCH_TEST_FAIL_DATASET="coco", BATCH_TEST_FAIL_CODE="17")
        self.assertEqual(result.returncode, 17)
        self.assertEqual([job["DATASET"] for job in jobs], ["flickr", "coco"])
        self.assertIn("Completed: flickr", result.stdout)
        self.assertIn("Failed: coco", result.stderr)
        self.assertNotIn("Completed all", result.stdout)

    def test_real_existing_launcher_resolves_distinct_indexes_and_outputs(self):
        shutil.copyfile(EXISTING, self.child)
        (self.root / "src").mkdir()
        (self.root / "src/three_stage_retrieval_pipeline.py").touch()
        model = self.root / "models/Qwen2.5-VL-7B-Instruct"
        model.mkdir(parents=True)
        leaf = "0_no_manual_40_sum_no_cluster_after_pad_disassembleeol_separate_lower_no_mean"
        for dataset in ("flickr", "coco"):
            directory = self.root / "datasets/data" / dataset
            directory.mkdir(parents=True)
            (directory / f"{dataset}_test.csv").touch()
            for kind in ("text", "image"):
                relative = Path(model.name) / dataset / kind / "no_filter/concat/prompt_5/test" / leaf
                dense = self.root / "dense_output" / relative
                dense.mkdir(parents=True)
                (dense / "corpus_0.pkl").touch()
                sparse = self.root / "sparse_output" / relative / "index"
                sparse.mkdir(parents=True)
                (sparse / "segments_1").touch()
        stub = self.root / "record-deepspeed"
        stub.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@" >> "$BATCH_TEST_RECORD"\nprintf "\\n" >> "$BATCH_TEST_RECORD"\n', encoding="utf-8")
        stub.chmod(0o755)
        for direction, kind in (("image", "text"), ("text", "image")):
            with self.subTest(direction=direction):
                self.record.unlink(missing_ok=True)
                result = subprocess.run([BASH, self.script.as_posix()], cwd=self.root.parent,
                                        env={**self.env, "DEEPSPEED_BIN": stub.as_posix(), "IR_QUERY_TYPE": direction},
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                jobs = [line.decode().rstrip("\0").split("\0") for line in self.record.read_bytes().splitlines()]
                self.assertEqual(len(jobs), 2)
                for dataset, argv in zip(("flickr", "coco"), jobs):
                    def value(option):
                        return argv[argv.index(option) + 1]
                    self.assertEqual(value("--dataset-name"), dataset)
                    self.assertEqual(value("--task-type"), "ir")
                    self.assertEqual(value("--query-type"), direction)
                    self.assertIn(f"/{dataset}/{kind}/", value("--stage2-dense-index"))
                    self.assertEqual(value("--stage1-sparse-index"), value("--stage2-sparse-index"))
                    self.assertIn(f"/{dataset}/", value("--output-dir"))
                    self.assertEqual(value("--retrieval-model-name-or-path"), model.as_posix())


if __name__ == "__main__":
    unittest.main()
