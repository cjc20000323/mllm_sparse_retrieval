"""Run the real Bash launcher against fake artifacts and a recording DeepSpeed.

Only the Python standard library is needed; no model, CUDA or datasets are read.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash") or next((str(path) for path in (
    Path("D:/Git/bin/bash.exe"), Path("C:/Program Files/Git/bin/bash.exe")
) if path.is_file()), None)
LEAF = "0_no_manual_40_sum_no_cluster_after_pad_disassembleeol_separate_lower_no_mean"
PROFILES = {
    "flickr": ("ir", "image", "text", "image_to_text", "flickr", ["flickr_{split}.csv"]),
    "coco": ("ir", "image", "text", "image_to_text", "coco", ["coco_{split}.csv"]),
    "CUHK-PEDES": ("tbpr", "text", "image", "text_to_image", "CUHK-PEDES", ["reid_raw.json"]),
    "ICFG-PEDES": ("tbpr", "text", "image", "text_to_image", "ICFG-PEDES", ["ICFG-PEDES.json"]),
    "RSTPReid": ("tbpr", "text", "image", "text_to_image", "RSTPReid", ["data_captions.json"]),
    "webqa": ("t2it", "text", "corpus", "text_to_imagetext", "WebQA", [
        "query-00000-of-00001.parquet", "qrels-00000-of-00001.parquet",
        "corpus-00000-of-00002.parquet", "corpus-00001-of-00002.parquet",
    ]),
    "remuq": ("it2t", "image", "corpus", "imagetext_to_text", "UMRB-ReMuQ", [
        "query.parquet", "qrels.parquet", "corpus.parquet",
    ]),
    "llava": ("it2t", "image", "corpus", "imagetext_to_text", "UMRB-LLaVA", [
        "query.parquet", "qrels.parquet", "corpus.parquet",
    ]),
}
PATH_OPTIONS = (
    "--dataset-name", "--dataset-split", "--task-type", "--query-type", "--tbpr-type",
    "--data-root", "--output-dir", "--stage1-sparse-index", "--stage2-dense-index",
    "--stage2-sparse-index",
)


@unittest.skipUnless(BASH, "Bash is required for launcher integration tests")
class ThreeStageLauncherPathTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "scripts").mkdir()
        (self.root / "src").mkdir()
        (self.root / "src/three_stage_retrieval_pipeline.py").touch()
        self.script = self.root / "scripts/three_stage_retrieval_pipeline.sh"
        shutil.copyfile(PROJECT_ROOT / "scripts/three_stage_retrieval_pipeline.sh", self.script)
        for name, (task, _, target, _, data_dir, files) in PROFILES.items():
            directory = self.root / "data" / data_dir
            directory.mkdir(parents=True)
            for split in ("test", "val"):
                for filename in files:
                    (directory / filename.format(split=split)).touch()
                for corpus_type in (("text", "image") if task == "ir" else (target,)):
                    for tbpr_type in (("origin_type", "type") if task == "tbpr" else (None,)):
                        relative = self.index_relative(name, corpus_type, split, tbpr_type)
                        self.make_dense(self.root / "dense_output" / relative)
                        self.make_sparse(self.root / "sparse_output" / relative)
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        self.record = self.root / "argv.bin"
        stub = self.root / "deepspeed"
        stub.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "$PIPELINE_TEST_ARGV"\n', encoding="utf-8")
        stub.chmod(0o755)
        names = set(re.findall(r"\$\{([A-Z_][A-Z_0-9]*):-", self.script.read_text(encoding="utf-8")))
        self.env = {key: value for key, value in os.environ.items() if key not in names}
        self.env.update(DATA_ROOT=self.root.as_posix(), BASE_MODEL="model", ARTIFACT_MODEL_NAME="model",
                        RETRIEVAL_MODEL_PATH=checkpoint.as_posix(), RANKER_MODEL_PATH=checkpoint.as_posix(),
                        DEEPSPEED_BIN=stub.as_posix(), PIPELINE_TEST_ARGV=self.record.as_posix(), GPUS="2",
                        PATH=str(Path(BASH).parent) + os.pathsep + os.environ.get("PATH", ""))

    @staticmethod
    def index_relative(name, target, split="test", tbpr_type=None):
        prefix = f"model/{name}/{target}/no_filter/concat/prompt_5/{split}/"
        return prefix + (f"{tbpr_type}/" if tbpr_type else "") + LEAF

    @staticmethod
    def make_dense(path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "corpus_0.pkl").touch()
        return path

    @staticmethod
    def make_sparse(path):
        (path / "index").mkdir(parents=True, exist_ok=True)
        (path / "index/segments_1").touch()
        return path

    def launch(self, *args, **env):
        if self.record.exists():
            self.record.unlink()
        result = subprocess.run([BASH, self.script.as_posix(), *args], env={**self.env, **env},
                                capture_output=True, text=True, timeout=30)
        argv = self.record.read_bytes().decode().rstrip("\0").split("\0") if self.record.exists() else []
        return result, argv

    def successful(self, *args, **env):
        result, argv = self.launch(*args, **env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(argv)
        # All shell-managed path options appear exactly once, after normalization.
        for option in PATH_OPTIONS:
            self.assertEqual(argv.count(option), 1, (option, argv))
        return result, argv

    def assert_configuration(self, result, argv, name, query, split="test", tbpr_type="origin_type"):
        task, _, target, direction, _, _ = PROFILES[name]
        if task == "ir":
            target, direction = ("text", "image_to_text") if query == "image" else ("image", "text_to_image")
        relative = self.index_relative(name, target, split, tbpr_type if task == "tbpr" else None)
        output = f"./three_stage_results/model/{name}/{direction}/{split}/sparse_concat_stage1"
        if task == "tbpr":
            output += f"/{tbpr_type}"
        for option, expected in (("--dataset-name", name), ("--dataset-split", split), ("--task-type", task),
                                 ("--query-type", query), ("--tbpr-type", tbpr_type), ("--output-dir", output),
                                 ("--stage2-dense-index", "dense_output/" + relative),
                                 ("--stage2-sparse-index", "sparse_output/" + relative),
                                 ("--stage1-sparse-index", "sparse_output/" + relative)):
            self.assertEqual(argv[argv.index(option) + 1], expected)
        self.assertIn(f"Dataset:            {name} ({split}, {direction})", result.stdout)
        self.assertIn(f"Task/query/corpus:  {task} / {query} / {target}", result.stdout)
        self.assertIn(f"Output directory:   {output}", result.stdout)

    def test_bash_syntax(self):
        subprocess.run([BASH, "-n", self.script.as_posix()], check=True, timeout=30)

    def test_defaults_remain_flickr_image_to_text(self):
        self.assert_configuration(*self.successful(), "flickr", "image")

    def test_environment_selects_both_ir_directions(self):
        for dataset in ("flickr", "coco"):
            for query in ("image", "text"):
                with self.subTest(dataset=dataset, query=query):
                    self.assert_configuration(*self.successful(DATASET=dataset, QUERY_TYPE=query), dataset, query)

    def test_cli_selects_ir_directions_with_both_argument_forms(self):
        for dataset in ("flickr", "coco"):
            for query in ("image", "text"):
                for equals in (False, True):
                    with self.subTest(dataset=dataset, query=query, equals=equals):
                        args = ([f"--dataset-name={dataset}", f"--query-type={query}"] if equals else
                                ["--dataset-name", dataset, "--query-type", query])
                        self.assert_configuration(*self.successful(*args), dataset, query)

    def test_cli_overrides_environment_before_task_validation_and_path_generation(self):
        result, argv = self.successful("--dataset-name", "coco", "--query-type=text", "--task-type", "ir",
                                      "--dataset-split=val", DATASET="webqa", QUERY_TYPE="image",
                                      TASK_TYPE="t2it", DATASET_SPLIT="test")
        self.assert_configuration(result, argv, "coco", "text", "val")

    def test_last_cli_value_wins_and_no_duplicate_managed_options_remain(self):
        result, argv = self.successful("--dataset-name=webqa", "--dataset-name", "coco",
                                      "--query-type", "image", "--query-type=text",
                                      "--dataset-split=val", "--dataset-split", "test",
                                      "--task-type=t2it", "--task-type=ir", QUERY_TYPE="image")
        self.assert_configuration(result, argv, "coco", "text")

    def test_other_datasets_keep_task_specific_directions(self):
        for name, (_, query, _, _, _, _) in PROFILES.items():
            if name in {"coco", "flickr"}:
                continue
            with self.subTest(dataset=name):
                self.assert_configuration(*self.successful("--dataset-name", name), name, query)
        for name in ("remuq", "llava"):
            with self.subTest(dataset=name, query="composed"):
                self.assert_configuration(*self.successful("--dataset-name", name, "--query-type=composed"),
                                          name, "composed")

    def test_cli_tbpr_view_type_changes_both_output_and_index_paths(self):
        result, argv = self.successful("--dataset-name=CUHK-PEDES", "--tbpr-type", "origin_type",
                                      "--tbpr-type=type", TBPR_TYPE="origin_type")
        self.assert_configuration(result, argv, "CUHK-PEDES", "text", tbpr_type="type")

    def test_explicit_output_directories_are_kept_verbatim(self):
        for args, env, expected in (
            ([], {"OUTPUT_DIR": "./custom/image_to_text/results with spaces"},
             "./custom/image_to_text/results with spaces"),
            (["--output-dir", "./cli output"], {"OUTPUT_DIR": "./environment output"}, "./cli output"),
            (["--output-dir=./first", "--output-dir=./last output"], {}, "./last output"),
        ):
            with self.subTest(args=args, env=env):
                result, argv = self.successful("--query-type=text", *args, **env)
                self.assertEqual(argv[argv.index("--output-dir") + 1], expected)
                self.assertIn(f"Output directory:   {expected}", result.stdout)
                self.assertEqual(argv[argv.index("--query-type") + 1], "text")

    def test_cli_data_root_is_used_by_preflight_and_command(self):
        alternate = self.root / "other data root"
        (alternate / "data/coco").mkdir(parents=True)
        (alternate / "data/coco/coco_val.csv").touch()
        result, argv = self.successful("--dataset-name=coco", "--dataset-split=val", "--data-root",
                                      alternate.as_posix(), DATA_ROOT=(self.root / "missing data").as_posix())
        self.assertEqual(argv[argv.index("--data-root") + 1], alternate.as_posix())
        self.assert_configuration(result, argv, "coco", "image", "val")

    def test_explicit_index_paths_override_environment_and_are_checked(self):
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
        missing = self.root / "missing dense"
        result, argv = self.launch(f"--stage2-dense-index={missing.as_posix()}")
        self.assertEqual(result.returncode, 2)
        self.assertIn(missing.as_posix(), result.stderr)
        self.assertEqual(argv, [])

    def test_default_stage1_index_follows_explicit_stage2_sparse_index(self):
        sparse = self.make_sparse(self.root / "shared sparse")
        _, argv = self.successful("--stage2-sparse-index", sparse.as_posix())
        self.assertEqual(argv[argv.index("--stage1-sparse-index") + 1], sparse.as_posix())
        _, argv = self.successful(STAGE2_SPARSE_INDEX=sparse.as_posix())
        self.assertEqual(argv[argv.index("--stage1-sparse-index") + 1], sparse.as_posix())

    def test_environment_index_paths_are_not_rewritten_on_direction_change(self):
        dense = self.make_dense(self.root / "explicit dense")
        sparse = self.make_sparse(self.root / "explicit sparse")
        result, argv = self.successful("--query-type", "text", STAGE2_DENSE_INDEX=dense.as_posix(),
                                      STAGE2_SPARSE_INDEX=sparse.as_posix())
        self.assertEqual(argv[argv.index("--stage2-dense-index") + 1], dense.as_posix())
        self.assertEqual(argv[argv.index("--stage2-sparse-index") + 1], sparse.as_posix())
        self.assertIn("/flickr/text_to_image/", argv[argv.index("--output-dir") + 1])
        self.assertIn("text_to_image", result.stdout)

    def test_unmanaged_arguments_are_forwarded_in_order(self):
        _, argv = self.successful("--query-type=text", "--rerank-num", "20", "--overwrite",
                                  "--stage2-depth=100", "--query-type", "image")
        self.assertEqual(argv[-4:], ["--rerank-num", "20", "--overwrite", "--stage2-depth=100"])
        self.assertEqual(argv[argv.index("--query-type") + 1], "image")

    def test_missing_or_empty_managed_values_fail_before_launch(self):
        for option in PATH_OPTIONS:
            for args in ([option], [option, "--query-type=text"], [option + "="], [option, ""]):
                with self.subTest(args=args):
                    result, argv = self.launch(*args)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(f"{option} requires", result.stderr)
                    self.assertEqual(argv, [])

    def test_invalid_dataset_task_or_query_fails_before_launch(self):
        cases = [(["--dataset-name=unknown"], "DATASET must be"),
                 (["--query-type=composed"], "requires QUERY_TYPE"),
                 (["--dataset-name=webqa", "--query-type=image"], "requires QUERY_TYPE=text"),
                 (["--dataset-name=remuq", "--query-type=text"], "requires QUERY_TYPE=image or composed"),
                 (["--task-type=tbpr"], "requires TASK_TYPE=ir"),
                 (["--dataset-name=CUHK-PEDES", "--tbpr-type=invalid"], "TBPR_TYPE must be")]
        for args, message in cases:
            with self.subTest(args=args):
                result, argv = self.launch(*args)
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)
                self.assertEqual(argv, [])


if __name__ == "__main__":
    unittest.main()
