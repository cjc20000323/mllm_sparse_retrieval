"""Real Bash launcher tests without importing Torch, loading models or Java."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("encode", "sparse_index", "search", "rerank")
COMMON = "qwen_retrieval_common.sh"
BASH = shutil.which("bash") or next((str(path) for path in (
    Path("D:/Git/bin/bash.exe"), Path("C:/Program Files/Git/bin/bash.exe")
) if path.is_file()), None)
CHECKPOINTS = {
    "qwen2": ("Qwen2-VL-7B-Instruct", "qwen2_vl"),
    "qwen25": ("Qwen2.5-VL-7B-Instruct", "qwen2_5_vl"),
    "qwen3": ("Qwen3-VL-8B-Instruct", "qwen3_vl"),
}


def artifact_function():
    """Use the production path function as the oracle, with no ML imports."""
    tree = ast.parse((ROOT / "src/qwen_vl_pipeline.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "artifact_dirs")
    namespace = {"Path": Path}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "artifact_dirs", "exec"), namespace)
    return namespace["artifact_dirs"]


def value(argv, option):
    for argument in argv:
        if argument.startswith(option + "="):
            return argument.split("=", 1)[1]
    return argv[argv.index(option) + 1]


@unittest.skipUnless(BASH, "Bash is required for launcher integration tests")
class QwenRetrievalLauncherTests(unittest.TestCase):
    def setUp(self):
        # Keep paths below Windows' legacy 260-character limit even after adding
        # collection/index, while still testing spaces in launcher arguments.
        temporary = tempfile.TemporaryDirectory(prefix="qwen launch ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        scripts = self.root / "scripts"
        scripts.mkdir()
        for filename in (COMMON, *(stage + "_qwen.sh" for stage in STAGES)):
            shutil.copyfile(ROOT / "scripts" / filename, scripts / filename)
        (self.root / "src").mkdir()
        for stage in ("encode", "search", "rerank"):
            (self.root / "src" / (stage + ".py")).touch()
        for filename in ("qwen_sparse_index.py", "qwen_vl_paths.py"):
            shutil.copyfile(ROOT / "src" / filename, self.root / "src" / filename)
        for name, model_type in CHECKPOINTS.values():
            self.checkpoint(self.root / "models" / name, model_type)
        self.record = self.root / "calls.bin"
        tool = self.root / "record tool"
        tool.write_bytes(b'''#!/usr/bin/env bash
set -euo pipefail
printf '%s\\0' "$0" "$@" >> "$LAUNCHER_TEST_RECORD"
printf '\\n' >> "$LAUNCHER_TEST_RECORD"
if [[ -n "${LAUNCHER_TEST_FAIL_MATCH-}" && " $* " == *"${LAUNCHER_TEST_FAIL_MATCH}"* ]]; then
    exit "${LAUNCHER_TEST_FAIL_CODE:-17}"
fi
# Simulate the indexer's output, so the next stage can consume it.
index=""
while (($#)); do
    if [[ "$1" == --sparse-root ]]; then index="$2/collection/index"; break; fi
    shift
done
if [[ -n "$index" ]]; then mkdir -- "$index"; touch -- "$index/segments_1"; fi
''')
        tool.chmod(0o755)
        config = "\n".join((scripts / filename).read_text(encoding="utf-8")
                           for filename in (COMMON, "encode_qwen.sh", "search_qwen.sh"))
        names = set(re.findall(r"\$\{([A-Z_][A-Z_0-9]*)(?::-|-)", config))
        names.update(("QWEN_MODEL_PATH", "CALCULATE_TYPE", "LAUNCHER_TEST_FAIL_MATCH", "LAUNCHER_TEST_FAIL_CODE"))
        self.env = {key: val for key, val in os.environ.items() if key not in names}
        self.env.update(
            QWEN_CHECKPOINT_ROOT=(self.root / "models").as_posix(),
            DENSE_OUTPUT_ROOT=(self.root / "dense").as_posix(),
            SPARSE_OUTPUT_ROOT=(self.root / "sparse").as_posix(),
            QWEN_OUTPUT_ROOT=(self.root / "results").as_posix(),
            DEEPSPEED_BIN=tool.as_posix(), PYTHON_BIN=tool.as_posix(),
            LAUNCHER_TEST_RECORD=self.record.as_posix(),
            PATH=str(Path(BASH).parent) + os.pathsep + os.environ.get("PATH", ""),
        )
        self.native_artifacts = artifact_function()

    def checkpoint(self, path, model_type="qwen2_5_vl"):
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")

    def launch(self, stage, *args, **settings):
        self.record.unlink(missing_ok=True)
        result = subprocess.run(
            [BASH, (self.root / "scripts" / (stage + "_qwen.sh")).as_posix(), *args],
            cwd=self.root.parent, env={**self.env, **settings}, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30,
        )
        calls = [line.decode("utf-8").rstrip("\0").split("\0")
                 for line in self.record.read_bytes().splitlines()] if self.record.exists() else []
        return result, calls

    def success(self, stage, *args, **settings):
        result, calls = self.launch(stage, *args, **settings)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return calls

    def corpus_paths(self, dataset, **settings):
        """Supply launcher inputs to the production artifact_dirs contract."""
        env = {**self.env, **settings}
        family = env.get("QWEN_FAMILY", "qwen25")
        model_path = env.get("QWEN_MODEL_PATH", env["QWEN_CHECKPOINT_ROOT"] + "/" + CHECKPOINTS[family][0]).rstrip("/")
        method = env.get("EOL_TYPE", "disassembleeol_separate")
        task = "ir" if dataset in {"flickr", "coco"} else "tbpr" if dataset in {"CUHK-PEDES", "ICFG-PEDES", "RSTPReid"} else "t2it" if dataset == "webqa" else "it2t"
        kind = ("text" if env.get("IR_QUERY_TYPE", "image") == "image" else "image") if task == "ir" else "image" if task == "tbpr" else "corpus"
        is_true = lambda name: env.get(name, "false").lower() in {"true", "1", "yes"}
        length = int(env.get("SPARSE_LENGTH", "40"))
        data = SimpleNamespace(
            dense_output_dir=env["DENSE_OUTPUT_ROOT"], sparse_output_dir=env["SPARSE_OUTPUT_ROOT"],
            dataset_name=dataset, dataset_split=env.get("DATASET_SPLIT", "test"),
            sparse_manual=is_true("SPARSE_MANUAL"), sparse_length=length,
            text_sparse_length=int(env.get("TEXT_SPARSE_LENGTH", str(length))),
            image_sparse_length=int(env.get("IMAGE_SPARSE_LENGTH", str(length))),
            num_expended_tokens=int(env.get("NUM_EXPENDED_TOKENS", "0")),
            sparse_value_type=env.get("SPARSE_VALUE_TYPE", "sum"), reps_loc=env.get("REPS_LOC", "after_pad"),
            sparse_lower_or_upper=env.get("SPARSE_LOWER_OR_UPPER", "lower"),
            sparse_value_mean=is_true("SPARSE_VALUE_MEAN"), is_filtered=is_true("IS_FILTERED"),
            tbpr_type=env.get("TBPR_TYPE", "origin_type"), prompt_type=env.get("PROMPT_TYPE", "prompt_5"),
        )
        model = SimpleNamespace(model_name_or_path=model_path, eol_type=method,
                                calculate_type="separate" if method == "prompteol" else "concat")
        return self.native_artifacts(model, data, SimpleNamespace(encode_type=kind, task_type=task))

    def prepare_corpus(self, dataset, *, indexed=True, layout="legacy", **settings):
        dense, sparse = self.corpus_paths(dataset, **settings)
        dense.mkdir(parents=True, exist_ok=True)
        (dense / "corpus_0.pkl").write_bytes(b"recording fixture")
        (sparse / "collection").mkdir(parents=True, exist_ok=True)
        (sparse / "collection/corpus_0.jsonl").write_text('{"id":"1","vector":{"word":1}}\n', encoding="utf-8")
        (sparse / "qwen_prompt_manifest.json").write_text('{"keep":"manifest outside collection"}', encoding="utf-8")
        if indexed:
            index = sparse / ("collection/index" if layout == "nested" else "index")
            index.mkdir(exist_ok=True)
            (index / "segments_1").touch()
        return dense, sparse

    def test_syntax_and_help_without_dependencies(self):
        for filename in (COMMON, *(stage + "_qwen.sh" for stage in STAGES)):
            result = subprocess.run([BASH, "-n", (self.root / "scripts" / filename).as_posix()],
                                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
        for stage in STAGES:
            for help_flag in ("-h", "--help"):
                self.assertFalse(self.success(stage, help_flag, QWEN_FAMILY="bad", DEEPSPEED_BIN="missing", PYTHON_BIN="missing"))

    def test_default_four_stage_workflow_and_native_path_contract(self):
        calls = [self.success("encode", DATASET=dataset)[0] for dataset in ("flickr", "coco")]
        self.assertEqual([value(call, "--dataset_name") for call in calls], ["flickr", "coco"])
        for call in calls:
            self.assertIn("src/encode.py", call)
            self.assertEqual(value(call, "--encode_type"), "text")
            self.assertEqual(value(call, "--num_gpus"), "1")
            self.assertEqual(value(call, "--num_nodes"), "1")
            self.assertIn("--fp16", call)
            self.assertEqual(value(call, "--calculate_type"), "concat")
            self.assertEqual(value(call, "--text_sparse_length"), "40")
            self.assertEqual(value(call, "--image_sparse_length"), "40")
            self.assertEqual(value(call, "--qwen_max_pixels"), "0")
        # Simulate completed encoding, using the real encoder's output path function.
        corpora = {dataset: self.prepare_corpus(dataset, indexed=False) for dataset in ("flickr", "coco")}
        calls = self.success("sparse_index")
        self.assertEqual(len(calls), 2)
        for dataset, call in zip(corpora, calls):
            _, sparse = corpora[dataset]
            self.assertEqual(call[1:3], ["-B", "src/qwen_sparse_index.py"])
            self.assertEqual(value(call, "--sparse-root"), sparse.as_posix())
            self.assertEqual(value(call, "--threads"), "16")
            self.assertTrue((sparse / "collection/index/segments_1").exists())
            self.assertEqual(json.loads((sparse / "qwen_prompt_manifest.json").read_text()), {"keep": "manifest outside collection"})
        outputs = []
        for stage in ("search", "rerank"):
            calls = ([self.success(stage, DATASET=name, USE_FIXED_ALPHA="true", ALPHA="0.5")[0]
                      for name in corpora] if stage == "search" else self.success(stage))
            for dataset, call in zip(corpora, calls):
                dense, sparse = corpora[dataset]
                self.assertEqual(value(call, "--passage_reps"), dense.as_posix())
                self.assertEqual(value(call, "--sparse_index"), (sparse / "collection" if stage == "search" else sparse).as_posix())
                self.assertEqual(value(call, "--query_type"), "image")
                self.assertIn("--use_fixed_alpha", call)
                self.assertEqual(value(call, "--alpha"), "0.5")
                self.assertNotIn("--val_passage_reps", call)
                outputs.append(value(call, "--output_dir"))
                if stage == "rerank":
                    self.assertEqual(value(call, "--rerank_template"), "what_caption_generation")
                    self.assertEqual(value(call, "--rerank_type"), "pointwise")
                    self.assertEqual(value(call, "--rerank_num"), "20")
                    self.assertEqual(value(call, "--rerank_batch_size"), "1")
                    self.assertEqual(value(call, "--modify_type"), "no")
        self.assertEqual(len(set(outputs)), 4)

    def test_all_families_methods_and_ir_directions(self):
        for family in CHECKPOINTS:
            for method in ("prompteol", "disassembleeol_separate"):
                for direction in ("image", "text"):
                    settings = dict(QWEN_FAMILY=family, EOL_TYPE=method, IR_QUERY_TYPE=direction, DATASETS="flickr")
                    with self.subTest(**settings):
                        call, = self.success("encode", **settings, ENCODE_TYPE="text" if direction == "image" else "image")
                        self.assertEqual(Path(value(call, "--model_name_or_path")).name, CHECKPOINTS[family][0])
                        self.assertEqual(value(call, "--calculate_type"), "separate" if method == "prompteol" else "concat")
                        dense, sparse = self.prepare_corpus("flickr", layout="nested", **settings)
                        for stage in ("search", "rerank"):
                            call, = self.success(stage, **settings, USE_FIXED_ALPHA="true")
                            self.assertEqual(value(call, "--query_type"), direction)
                            self.assertEqual(value(call, "--passage_reps"), dense.as_posix())
                            self.assertEqual(value(call, "--sparse_index"), (sparse / "collection" if stage == "search" else sparse).as_posix())
                            self.assertIn("image_to_text" if direction == "image" else "text_to_image", value(call, "--output_dir"))

    def test_mixed_datasets_map_tasks_corpora_and_queries(self):
        names = ["coco", "flickr", "CUHK-PEDES", "ICFG-PEDES", "RSTPReid", "webqa", "remuq", "llava"]
        settings = dict(DATASETS="\t" + "\n ".join(names), IR_QUERY_TYPE="text", TBPR_TYPE="type", DATASET_SPLIT="val", EOL_TYPE="prompteol")
        tasks = ["ir", "ir", "tbpr", "tbpr", "tbpr", "t2it", "it2t", "it2t"]
        kinds = ["image"] * 5 + ["corpus"] * 3
        calls = [self.success("encode", **settings, DATASET=name, TASK_TYPE=task, ENCODE_TYPE=kind)[0]
                 for name, task, kind in zip(names, tasks, kinds)]
        self.assertEqual([value(call, "--dataset_name") for call in calls], names)
        self.assertEqual([value(call, "--task_type") for call in calls], ["ir", "ir", "tbpr", "tbpr", "tbpr", "t2it", "it2t", "it2t"])
        self.assertEqual([value(call, "--encode_type") for call in calls], ["image"] * 5 + ["corpus"] * 3)
        corpora = {name: self.prepare_corpus(name, **settings) for name in names}
        for stage in ("search", "rerank"):
            calls = ([self.success(stage, **settings, DATASET=name)[0] for name in names]
                     if stage == "search" else self.success(stage, **settings))
            self.assertEqual([value(call, "--query_type") for call in calls], ["text"] * 6 + ["image"] * 2)
            for name, call in zip(names, calls):
                self.assertEqual(value(call, "--passage_reps"), corpora[name][0].as_posix())
                if name in names[2:5]:
                    self.assertIn("/val/type/", value(call, "--sparse_index"))

    def test_shared_settings_custom_model_spaces_and_prior_cache(self):
        model = self.root / "my custom Qwen weights"
        self.checkpoint(model, "qwen3_vl")
        settings = dict(QWEN_FAMILY="qwen3", QWEN_MODEL_PATH=model.as_posix() + "/", DATASETS="flickr", DTYPE="bfloat16",
                        GPUS="2", PER_DEVICE_BATCH_SIZE="2", RETRIEVAL_BATCH_SIZE="4", QWEN_MIN_PIXELS="64", QWEN_MAX_PIXELS="256",
                        SPARSE_LENGTH="64", TEXT_SPARSE_LENGTH="32", IMAGE_SPARSE_LENGTH="96", SPARSE_MANUAL="yes",
                        SPARSE_VALUE_MEAN="1", IS_FILTERED="true", SPARSE_VALUE_TYPE="max", SPARSE_LOWER_OR_UPPER="upper",
                        NUM_EXPENDED_TOKENS="2", PROMPT_TYPE="prompt_3", REPS_LOC="before_pad", USE_GPU="true",
                        RERANK_NUM="10", RERANK_BATCH_SIZE="3", MODIFY_TYPE="modify_single", USE_RERANK_PRIOR_CACHE="true")
        encode, = self.success("encode", **settings)
        dense, sparse = self.prepare_corpus("flickr", **settings)
        search, = self.success("search", **settings, USE_FIXED_ALPHA="true")
        rerank, = self.success("rerank", **settings)
        for call in (encode, search, rerank):
            for option, expected in (("--model_name_or_path", model.as_posix()), ("--qwen_min_pixels", "64"),
                                     ("--qwen_max_pixels", "256"), ("--text_sparse_length", "32"),
                                     ("--image_sparse_length", "96"), ("--per_device_batch_size", "2")):
                self.assertEqual(value(call, option), expected)
            for flag in ("--bf16", "--sparse_manual", "--sparse_value_mean", "--is_filtered"):
                self.assertIn(flag, call)
            self.assertNotIn("--fp16", call)
        self.assertEqual(value(search, "--passage_reps"), dense.as_posix())
        self.assertEqual(value(rerank, "--sparse_index"), sparse.as_posix())
        self.assertIn("--use_gpu", search)
        self.assertIn("--use_rerank_prior_cache", rerank)
        self.assertEqual(value(rerank, "--modify_type"), "modify_single")
        self.assertEqual(value(rerank, "--rerank_batch_size"), "3")
        self.assertIn("/top_10/modify_single/", value(rerank, "--output_dir"))
        call, = self.success("encode", DATASETS="flickr", DTYPE="float32")
        self.assertNotIn("--fp16", call)
        self.assertNotIn("--bf16", call)

    def test_alpha_selection_uses_validation_only_when_explicitly_configured(self):
        for dataset in ("flickr", "coco"):
            self.prepare_corpus(dataset)
        self.success("search", USE_FIXED_ALPHA="true")
        result, calls = self.launch("search")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        call, = calls
        self.assertNotIn("--val_passage_reps", call)
        self.assertNotIn("--use_fixed_alpha", call)
        self.assertIn("Skipping validation", result.stdout)
        self.assertIn("alpha_test", value(call, "--output_dir"))
        for dataset in ("flickr", "coco"):
            self.prepare_corpus(dataset, DATASET_SPLIT="val")
        call, = self.success("search")
        self.assertNotIn("--val_passage_reps", call)
        self.assertIn("alpha_test", value(call, "--output_dir"))
        for stage in ("search", "rerank"):
            calls = ([self.success(stage, DATASET=name,
                                   VAL_PASSAGE_REPS=self.corpus_paths(name, DATASET_SPLIT="val")[0].as_posix(),
                                   VAL_SPARSE_INDEX=self.corpus_paths(name, DATASET_SPLIT="val")[1].as_posix())[0]
                      for name in ("flickr", "coco")]
                     if stage == "search" else self.success(stage, USE_FIXED_ALPHA="false"))
            for dataset, call in zip(("flickr", "coco"), calls):
                dense, sparse = self.corpus_paths(dataset, DATASET_SPLIT="val")
                self.assertEqual(value(call, "--val_passage_reps"), dense.as_posix())
                self.assertEqual(value(call, "--val_sparse_index"), (sparse / "collection" if stage == "search" else sparse).as_posix())
                self.assertNotIn("--use_fixed_alpha", call)
                self.assertIn("alpha_val", value(call, "--output_dir"))
        call, = self.success("search", DATASETS="flickr", USE_FIXED_ALPHA="true", ALPHA="0.7")
        self.assertEqual(value(call, "--alpha"), "0.7")
        self.assertIn("alpha_0.7", value(call, "--output_dir"))

    def test_invalid_configuration_launches_no_jobs(self):
        cases = ({"DATASETS": ""}, {"DATASETS": " \t\n"}, {"DATASETS": "flickr flickr"},
                 {"DATASETS": "flickr unknown"}, {"QWEN_FAMILY": "llava"}, {"EOL_TYPE": "unknown"},
                 {"EOL_TYPE": "prompteol", "CALCULATE_TYPE": "concat"}, {"CALCULATE_TYPE": "separate"},
                 {"IR_QUERY_TYPE": "composed"}, {"GPUS": "0"}, {"NUM_NODES": "0"}, {"DTYPE": "int8"},
                 {"PER_DEVICE_BATCH_SIZE": "0"}, {"QWEN_MIN_PIXELS": "-1"},
                 {"QWEN_MIN_PIXELS": "512", "QWEN_MAX_PIXELS": "256"}, {"USE_FIXED_ALPHA": "maybe"},
                 {"ALPHA": "1.1"}, {"RERANK_BATCH_SIZE": "0"}, {"SPARSE_VALUE_TYPE": "fusion"},
                 {"SPARSE_LENGTH": "0"}, {"MODIFY_TYPE": "other"})
        for number, settings in enumerate(cases):
            with self.subTest(**settings):
                # Fusion/direction/reranking options belong to the other launchers.
                stage = STAGES[number % len(STAGES)]
                if any(key in settings for key in ("IR_QUERY_TYPE", "ALPHA")):
                    stage = "search"
                if "MODIFY_TYPE" in settings:
                    stage = "rerank"
                result, calls = self.launch(stage, **settings)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertFalse(calls)
        for stage in STAGES:
            for args in (("--dataset_name", "webqa"), ("--output_dir=/tmp/shared",), ("--",), ("--help", "extra")):
                result, calls = self.launch(stage, *args)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(calls)

    def test_missing_inputs_and_existing_artifacts_stop_before_any_job(self):
        result, calls = self.launch("encode", QWEN_MODEL_PATH=(self.root / "missing").as_posix())
        self.assertEqual(result.returncode, 2)
        self.assertFalse(calls)
        for stage in ("sparse_index", "search", "rerank"):
            result, calls = self.launch(stage)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(calls)
        self.prepare_corpus("flickr")
        # Explicit invalid validation inputs must still fail before launching.
        result, calls = self.launch("search", VAL_PASSAGE_REPS="/missing validation")
        self.assertEqual(result.returncode, 2)
        self.assertFalse(calls)
        self.assertIn("Validation", result.stderr)
        # A missing second dataset must be detected before running Flickr.
        result, calls = self.launch("rerank")
        self.assertEqual(result.returncode, 2)
        self.assertFalse(calls)
        self.prepare_corpus("flickr", layout="nested")
        for stage in ("sparse_index",):
            result, calls = self.launch(stage, DATASETS="flickr")
            self.assertEqual(result.returncode, 2)
            self.assertFalse(calls)
            self.assertIn("already exist", result.stderr)
        # Dense data without a built sparse index is also rejected.
        self.prepare_corpus("coco", indexed=False)
        result, calls = self.launch("rerank", DATASETS="coco")
        self.assertEqual(result.returncode, 2)
        self.assertFalse(calls)
        self.assertIn("Lucene", result.stderr)

    def test_each_stage_stops_on_failure_and_preserves_status(self):
        for stage in STAGES:
            with self.subTest(stage=stage):
                settings = dict(DATASETS="flickr coco webqa", DENSE_OUTPUT_ROOT=(self.root / stage / "dense").as_posix(),
                                SPARSE_OUTPUT_ROOT=(self.root / stage / "sparse").as_posix(),
                                LAUNCHER_TEST_FAIL_MATCH="coco", LAUNCHER_TEST_FAIL_CODE="17")
                if stage != "encode":
                    for dataset in ("flickr", "coco", "webqa"):
                        self.prepare_corpus(dataset, indexed=stage != "sparse_index", **settings)
                # Encoding/search are single jobs; indexing/reranking retain batching.
                single = stage in {"encode", "search"}
                overrides = {"DATASET": "coco"} if single else {}
                if stage == "search":
                    overrides["USE_FIXED_ALPHA"] = "true"
                result, calls = self.launch(stage, **settings, **overrides)
                self.assertEqual(result.returncode, 17, result.stdout + result.stderr)
                self.assertEqual(len(calls), 1 if single else 2)
                if not single:
                    self.assertIn("Completed: " + stage + " flickr", result.stdout)
                    self.assertIn("Failed: " + stage + " coco", result.stderr)
                self.assertNotIn("Completed all", result.stdout)

    def test_encode_is_standalone_and_defaults_to_one_text_corpus(self):
        # The referenced encode.sh, random_token_views.sh and common helper are absent.
        (self.root / "scripts" / COMMON).unlink()
        self.assertFalse((self.root / "scripts/encode.sh").exists())
        call, = self.success("encode")
        self.assertEqual(value(call, "--dataset_name"), "flickr")
        self.assertEqual(value(call, "--task_type"), "ir")
        self.assertEqual(value(call, "--encode_type"), "text")
        self.assertEqual(value(call, "--output_dir"), "./output")
        self.assertEqual(value(call, "--model_name_or_path"), (self.root / "models/Qwen2.5-VL-7B-Instruct").as_posix())
        self.assertEqual(value(call, "--dense_output_dir"), self.env["DENSE_OUTPUT_ROOT"])
        self.assertEqual(value(call, "--sparse_output_dir"), self.env["SPARSE_OUTPUT_ROOT"])
        self.assertNotIn("--use_generated_validation_data", call)
        self.assertNotIn("--generated_validation_data_root", call)

    def test_encode_launcher_allows_existing_dense_and_sparse_artifacts(self):
        dense, sparse = self.prepare_corpus("flickr")
        call, = self.success("encode", DATASET="flickr")
        self.assertIn("src/encode.py", call)
        # File replacement/index retirement is owned by Python, once on rank 0.
        self.assertTrue((dense / "corpus_0.pkl").exists())
        self.assertTrue((sparse / "collection/corpus_0.jsonl").exists())
        self.assertTrue((sparse / "index/segments_1").exists())

    def test_encode_explicit_dataset_and_modality_take_precedence(self):
        call, = self.success("encode", DATASET="CUHK-PEDES", DATASETS="flickr coco", TASK_TYPE="tbpr",
                             ENCODE_TYPE="image", IR_QUERY_TYPE="invalid-and-unused", OUTPUT_DIR="/my encoding logs")
        self.assertEqual(value(call, "--dataset_name"), "CUHK-PEDES")
        self.assertEqual(value(call, "--task_type"), "tbpr")
        self.assertEqual(value(call, "--encode_type"), "image")
        self.assertEqual(value(call, "--output_dir"), "/my encoding logs")
        # IR_QUERY_TYPE alone no longer changes the explicitly configured encode mode.
        call, = self.success("encode", IR_QUERY_TYPE="text")
        self.assertEqual(value(call, "--encode_type"), "text")

    def test_encode_legacy_dataset_alias_accepts_only_one_name(self):
        call, = self.success("encode", DATASETS="\n coco \t")
        self.assertEqual(value(call, "--dataset_name"), "coco")
        for dataset in ("", " \t", "flickr coco", "flickr\ncoco"):
            result, calls = self.launch("encode", DATASETS=dataset)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertFalse(calls)
            self.assertIn("separately", result.stderr)

    def test_encode_generated_validation_and_random_views_need_no_other_script(self):
        (self.root / "scripts" / COMMON).unlink()
        root = (self.root / "generated validation").as_posix()
        call, = self.success("encode", PROMPT_TYPE="prompt_random_5", RANDOM_VIEW_SEED="123", DATASET_SPLIT="test",
                             USE_GENERATED_VALIDATION_DATA="true", GENERATED_VALIDATION_DATA_ROOT=root)
        self.assertEqual(value(call, "--dataset_split"), "val")
        self.assertEqual(value(call, "--prompt_type"), "prompt_random_5")
        self.assertEqual(value(call, "--random_view_seed"), "123")
        self.assertIn("--use_generated_validation_data", call)
        self.assertEqual(value(call, "--generated_validation_data_root"), root)
        result, calls = self.launch("encode", USE_GENERATED_VALIDATION_DATA="maybe")
        self.assertEqual(result.returncode, 2)
        self.assertFalse(calls)

    def test_inherited_other_launcher_settings_do_not_change_jobs(self):
        inherited = {name: "stale-value" for name in (
            "DATASET", "TASK_TYPE", "QUERY_TYPE", "BASE_MODEL", "ARTIFACT_MODEL_NAME", "OUTPUT_DIR",
            "RETRIEVAL_MODEL_PATH", "RANKER_MODEL_PATH", "STAGE1_SPARSE_INDEX", "STAGE2_DENSE_INDEX", "STAGE2_SPARSE_INDEX",
        )}
        for dataset in ("flickr", "coco"):
            self.prepare_corpus(dataset)
        for stage in ("search", "rerank"):
            if stage == "search":
                # DATASET and OUTPUT_DIR now belong to standalone search itself.
                settings = {key: val for key, val in inherited.items() if key not in {"DATASET", "OUTPUT_DIR"}}
                calls = self.success(stage, **settings, USE_FIXED_ALPHA="true")
            else:
                calls = self.success(stage, **inherited)
            self.assertEqual([value(call, "--dataset_name") for call in calls],
                             ["flickr"] if stage == "search" else ["flickr", "coco"])
            for call in calls:
                self.assertNotIn("stale-value", " ".join(call))
                self.assertEqual(value(call, "--query_type"), "image")

    def test_search_is_standalone_and_ignores_unrequested_existing_validation(self):
        (self.root / "scripts" / COMMON).unlink()
        self.assertFalse((self.root / "scripts/search.sh").exists())
        self.prepare_corpus("flickr")
        val_dense, val_sparse = self.prepare_corpus("flickr", DATASET_SPLIT="val")
        call, = self.success("search")
        self.assertIn("src/search.py", call)
        self.assertEqual(value(call, "--dataset_name"), "flickr")
        self.assertEqual(value(call, "--dataset_split"), "test")
        self.assertEqual(value(call, "--task_type"), "ir")
        self.assertEqual(value(call, "--encode_type"), "text")
        self.assertEqual(value(call, "--num_gpus"), "1")
        self.assertEqual(value(call, "--num_nodes"), "1")
        self.assertEqual(value(call, "--alpha"), "0.9")
        self.assertNotIn("--val_passage_reps", call)
        self.assertNotIn("--val_sparse_index", call)
        self.assertEqual(value(call, "--random_view_seed"), "42")
        self.assertEqual(value(call, "--depth"), "1000")
        self.assertEqual(value(call, "--threads"), "16")
        self.assertIn("--fp16", call)
        self.assertNotIn("--use_fixed_alpha", call)
        self.assertNotIn("--use_generated_validation_data", call)
        self.assertNotIn("--use_gpu", call)
        self.assertTrue(value(call, "--output_dir").endswith("/image_to_text/search/alpha_test"))

    def test_search_explicit_dataset_precedes_alias_and_alias_is_single_value_only(self):
        self.prepare_corpus("coco")
        call, = self.success("search", DATASET="coco", DATASETS="flickr coco", USE_FIXED_ALPHA="true")
        self.assertEqual(value(call, "--dataset_name"), "coco")
        call, = self.success("search", DATASETS="\n coco \t", USE_FIXED_ALPHA="true")
        self.assertEqual(value(call, "--dataset_name"), "coco")
        for names in ("", " \t", "flickr coco", "flickr\ncoco"):
            result, calls = self.launch("search", DATASETS=names, USE_FIXED_ALPHA="true")
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertFalse(calls)
            self.assertIn("separately", result.stderr)

    def test_search_explicit_index_and_result_paths_with_spaces(self):
        settings = dict(DENSE_OUTPUT_ROOT=(self.root / "custom dense").as_posix(),
                        SPARSE_OUTPUT_ROOT=(self.root / "custom sparse").as_posix(), SPARSE_LENGTH="30")
        dense, sparse = self.prepare_corpus("flickr", **settings)
        val_dense, val_sparse = self.prepare_corpus("flickr", **settings, DATASET_SPLIT="val")
        output = (self.root / "custom results").as_posix()
        call, = self.success("search", SPARSE_LENGTH="30", PASSAGE_REPS=dense.as_posix(),
                             SPARSE_INDEX=(sparse / "index").as_posix(), VAL_PASSAGE_REPS=val_dense.as_posix(),
                             VAL_SPARSE_INDEX=val_sparse.as_posix(), OUTPUT_DIR=output)
        self.assertEqual(value(call, "--passage_reps"), dense.as_posix())
        self.assertEqual(value(call, "--sparse_index"), (sparse / "collection").as_posix())
        self.assertEqual(value(call, "--val_passage_reps"), val_dense.as_posix())
        self.assertEqual(value(call, "--val_sparse_index"), (val_sparse / "collection").as_posix())
        self.assertEqual(value(call, "--output_dir"), output)
        self.assertEqual(value(call, "--sparse_length"), "30")
        call, = self.success("search", SPARSE_LENGTH="30", PASSAGE_REPS=(dense / "corpus_0.pkl").as_posix(),
                             SPARSE_INDEX=sparse.as_posix(), USE_FIXED_ALPHA="yes", ALPHA="0.2",
                             VAL_PASSAGE_REPS="/missing val dense", VAL_SPARSE_INDEX="/missing val sparse",
                             USE_GENERATED_VALIDATION_DATA="true")
        self.assertEqual(value(call, "--passage_reps"), (dense / "corpus_0.pkl").as_posix())
        self.assertIn("--use_fixed_alpha", call)
        self.assertNotIn("--val_passage_reps", call)
        self.assertNotIn("--val_sparse_index", call)
        self.assertNotIn("--use_generated_validation_data", call)
        self.assertTrue(value(call, "--output_dir").endswith("/search/alpha_0.2"))

    def test_search_generated_validation_keeps_test_split_and_random_seed(self):
        settings = dict(PROMPT_TYPE="prompt_random_5", SPARSE_LENGTH="30")
        self.prepare_corpus("flickr", **settings)
        val_dense, val_sparse = self.prepare_corpus("flickr", **settings, DATASET_SPLIT="val")
        root = (self.root / "generated validation").as_posix()
        call, = self.success("search", **settings, RANDOM_VIEW_SEED="123",
                             USE_GENERATED_VALIDATION_DATA="yes", GENERATED_VALIDATION_DATA_ROOT=root,
                             VAL_PASSAGE_REPS=val_dense.as_posix(), VAL_SPARSE_INDEX=val_sparse.as_posix())
        self.assertEqual(value(call, "--dataset_split"), "test")
        self.assertEqual(value(call, "--prompt_type"), "prompt_random_5")
        self.assertEqual(value(call, "--random_view_seed"), "123")
        self.assertEqual(value(call, "--generated_validation_data_root"), root)
        self.assertIn("--use_generated_validation_data", call)
        self.assertNotIn("_generated_val", value(call, "--val_passage_reps"))

    def test_search_normalizes_all_sparse_aliases_to_collection(self):
        dense, sparse = self.prepare_corpus("flickr")
        val_dense, val_sparse = self.prepare_corpus("flickr", DATASET_SPLIT="val")
        for suffix in ("", "/", "/index", "/index///", "/collection", "/collection///", "/collection/index", "/collection/index///"):
            with self.subTest(suffix=suffix):
                result, calls = self.launch("search", PASSAGE_REPS=dense.as_posix(),
                                            SPARSE_INDEX=sparse.as_posix() + suffix,
                                            VAL_PASSAGE_REPS=val_dense.as_posix(),
                                            VAL_SPARSE_INDEX=val_sparse.as_posix() + suffix)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                call, = calls
                self.assertEqual(value(call, "--sparse_index"), (sparse / "collection").as_posix())
                self.assertEqual(value(call, "--val_sparse_index"), (val_sparse / "collection").as_posix())
                self.assertNotIn("/collection/collection", " ".join(call))
                self.assertIn("Test Lucene index: " + (sparse / "index").as_posix(), result.stdout)
                self.assertIn("Validation Lucene index: " + (val_sparse / "index").as_posix(), result.stdout)

    def test_search_collection_still_requires_built_lucene_index(self):
        _, sparse = self.prepare_corpus("flickr", indexed=False)
        result, calls = self.launch("search", SPARSE_INDEX=(sparse / "collection").as_posix())
        self.assertEqual(result.returncode, 2)
        self.assertFalse(calls)
        self.assertIn((sparse / "collection/index").as_posix(), result.stderr)
        self.assertIn("sparse_index_qwen.sh", result.stderr)

    def test_nested_index_preferred_for_test_and_val_and_incomplete_blocks_fallback(self):
        _, sparse = self.prepare_corpus("flickr")
        val_dense, val_sparse = self.prepare_corpus("flickr", DATASET_SPLIT="val")
        for root in (sparse, val_sparse):
            self.prepare_corpus("flickr", layout="nested", DATASET_SPLIT="val" if root == val_sparse else "test")
        settings = dict(VAL_PASSAGE_REPS=val_dense.as_posix(), VAL_SPARSE_INDEX=(val_sparse / "collection/index").as_posix())
        result, calls = self.launch("search", **settings)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Test Lucene index: " + (sparse / "collection/index").as_posix(), result.stdout)
        self.assertIn("Validation Lucene index: " + (val_sparse / "collection/index").as_posix(), result.stdout)
        self.assertEqual(len(calls), 1)
        for root in (val_sparse, sparse):
            (root / "collection/index/segments_1").unlink()
            for stage in ("search", "rerank"):
                result, calls = self.launch(stage, **(settings if stage == "search" else dict(DATASETS="flickr", USE_FIXED_ALPHA="false")))
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertFalse(calls)
                self.assertIn((root / "collection/index").as_posix(), result.stderr)
            (root / "collection/index/segments_1").touch()

    def test_index_launcher_allows_legacy_index_and_builds_nested(self):
        _, sparse = self.prepare_corpus("flickr")
        call, = self.success("sparse_index", DATASETS="flickr")
        self.assertEqual(value(call, "--sparse-root"), sparse.as_posix())
        self.assertTrue((sparse / "index/segments_1").exists())
        self.assertTrue((sparse / "collection/index/segments_1").exists())

    def test_search_skips_partial_default_validation_without_generated_flags(self):
        settings = dict(DATASET_SPLIT="dev", SPARSE_LENGTH="30")
        self.prepare_corpus("flickr", **settings)
        val_dense, val_sparse = self.prepare_corpus("flickr", SPARSE_LENGTH="30", DATASET_SPLIT="val", indexed=False)
        result, calls = self.launch("search", **settings, USE_GENERATED_VALIDATION_DATA="true")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        call, = calls
        for flag in ("--val_passage_reps", "--val_sparse_index", "--use_fixed_alpha",
                     "--use_generated_validation_data", "--generated_validation_data_root"):
            self.assertNotIn(flag, call)
        self.assertEqual(value(call, "--dataset_split"), "dev")
        self.assertTrue(value(call, "--output_dir").endswith("/search/alpha_dev"))
        self.assertIn("Skipping validation", result.stdout)
        self.assertIn("no val paths specified", result.stdout)
        # Supplying a val override makes this a configuration error, even though
        # the same incomplete default val was optional in the preceding run.
        result, calls = self.launch("search", **settings, VAL_PASSAGE_REPS=val_dense.as_posix())
        self.assertEqual(result.returncode, 2)
        self.assertFalse(calls)
        self.assertIn("Validation", result.stderr)
        # A complete sparse val alone also falls back if dense val is missing.
        (val_dense / "corpus_0.pkl").unlink()
        (val_sparse / "index").mkdir()
        (val_sparse / "index/segments_1").touch()
        call, = self.success("search", **settings)
        self.assertNotIn("--val_sparse_index", call)

    def test_search_validation_manifest_errors_are_left_to_python(self):
        self.prepare_corpus("flickr")
        val_dense, val_sparse = self.prepare_corpus("flickr", DATASET_SPLIT="val")
        (val_sparse / "qwen_prompt_manifest.json").write_text("corrupt metadata")
        call, = self.success("search", VAL_PASSAGE_REPS=val_dense.as_posix(), VAL_SPARSE_INDEX=val_sparse.as_posix())
        # Presence detection is not a catch-all retry: the Python reader must
        # validate this existing index, rather than silently selecting on test.
        self.assertIn("--val_passage_reps", call)
        self.assertIn("--val_sparse_index", call)

    def test_search_rejects_invalid_own_configuration_before_launch(self):
        self.prepare_corpus("flickr")
        self.prepare_corpus("flickr", DATASET_SPLIT="val")
        cases = ({"DEPTH": "0"}, {"THREADS": "bad"}, {"RETRIEVAL_BATCH_SIZE": "0"},
                 {"RANDOM_VIEW_SEED": "bad"}, {"USE_GPU": "maybe"}, {"SPARSE_MANUAL": "maybe"},
                 {"USE_GENERATED_VALIDATION_DATA": "maybe"}, {"ALPHA": "1.1"},
                 {"EOL_TYPE": "prompteol", "CALCULATE_TYPE": "concat"}, {"DTYPE": "int8"},
                 {"PASSAGE_REPS": "/missing dense"}, {"SPARSE_INDEX": "/missing sparse"},
                 {"VAL_PASSAGE_REPS": "/missing val dense"}, {"VAL_SPARSE_INDEX": "/missing val sparse"})
        for settings in cases:
            with self.subTest(**settings):
                result, calls = self.launch("search", **settings)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertFalse(calls)


if __name__ == "__main__":
    unittest.main()
