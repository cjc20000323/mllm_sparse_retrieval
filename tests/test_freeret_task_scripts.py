"""Contract checks for the nine single-task FreeRet launchers and safe indexing."""
import ast
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_freeret_dense import ROOT, dense, temporary_directory
import freeret_sparse_index as indexer


GROUPS = ("ir", "tbpr", "fused")
STAGES = ("encode", "sparse_index", "three_stage_retrieval")
NAMES = [f"{stage}_freeret_{group}.sh" for group in GROUPS for stage in STAGES]
BASH = "D:/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
CASES = [("ir", dataset, query, "ir", "text" if query == "image" else "image")
         for dataset in ("flickr", "coco") for query in ("image", "text")]
CASES += [("tbpr", dataset, "text", "tbpr", "image") for dataset in ("CUHK-PEDES", "ICFG-PEDES", "RSTPReid")]
CASES += [("fused", "webqa", "text", "t2it", "corpus")]
CASES += [("fused", dataset, query, "it2t", "corpus") for dataset in ("remuq", "llava") for query in ("image", "composed")]


def environment(**values):
    # The invoking shell's experimental settings must not steer the fixture.
    configured = set()
    for name in NAMES:
        configured.update(re.findall(r'([A-Z_0-9]+)="\$\{', (ROOT / "scripts" / name).read_text(encoding="utf-8")))
    configured.update(("MODEL_FAMILIES", "DATASETS", "ENCODE_TYPES"))
    return dict({k: v for k, v in os.environ.items() if k not in configured}, **values)


def report(output, prefix):
    return next(row[len(prefix):] for row in output.splitlines() if row.startswith(prefix))


@unittest.skipUnless(BASH and Path(BASH).is_file(), "Bash unavailable")
class LauncherTests(unittest.TestCase):
    def run_script(self, name, env, *, cwd=ROOT, scripts=None):
        path = (ROOT / "scripts" if scripts is None else scripts) / name
        return subprocess.run([BASH, path.as_posix()], cwd=cwd, env=env, capture_output=True, text=True, timeout=30)

    def test_help_syntax_and_dry_run_work_without_models_or_artifacts(self):
        for name in NAMES:
            with self.subTest(script=name):
                script = ROOT / "scripts" / name
                result = subprocess.run([BASH, "-n", script.as_posix()], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                result = subprocess.run([BASH, script.as_posix(), "--help"], env=environment(), capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("One model and one dataset", result.stdout)
                result = self.run_script(name, environment(DRY_RUN="true", MODEL_PATH="/missing/custom/checkpoint"))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("FreeRet dense=false constraints=false", result.stdout)
                self.assertNotIn(dense.SUFFIX, report(result.stdout, "Dense corpus: "))

    def test_all_families_tasks_and_directions_share_the_same_corpus_paths(self):
        families = ("llava_llama", "llava_mistral", "qwen2", "qwen25", "qwen3")
        modes = ((False, False), (True, False), (True, True))
        for family_index, family in enumerate(families):
            for case_index, (group, dataset, query, task, kind) in enumerate(CASES):
                use, prompts = modes[(family_index + case_index) % len(modes)]
                image_mode = "base" if family.startswith("llava") and case_index % 2 else "anyres"
                env = environment(DRY_RUN="true", MODEL_FAMILY=family, MODEL_PATH="/custom checkpoint/model name/",
                    DATASET=dataset, QUERY_TYPE=query, LLAVA_IMAGE_MODE=image_mode,
                    USE_FREERET_DENSE=str(use).lower(), USE_FREERET_PROMPT_CONSTRAINTS=str(prompts).lower(),
                    TBPR_TYPE="type", SPARSE_VALUE_MEAN="true", IS_FILTERED="true", SPARSE_LENGTH="30")
                paths = []
                for stage in STAGES:
                    with self.subTest(family=family, dataset=dataset, query=query, stage=stage):
                        result = self.run_script(f"{stage}_freeret_{group}.sh", env)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        corpus = report(result.stdout, "Sparse corpus: ")
                        dense_corpus = report(result.stdout, "Dense corpus: ")
                        self.assertIn(f"model name/{dataset}/{kind}/filter/concat/prompt_5/test/", corpus)
                        self.assertIn("/type/", corpus) if task == "tbpr" else None
                        leaf = "0_no_manual_30_sum_no_cluster_after_pad_disassembleeol_separate_lower_mean"
                        if image_mode == "base" and (kind == "image" or task == "t2it"): leaf += "_llava_base"
                        if use: leaf += dense.SUFFIX
                        if prompts: leaf += dense.PROMPT_SUFFIX
                        self.assertTrue(corpus.endswith(leaf), corpus)
                        index = report(result.stdout, "Lucene index: ")
                        self.assertEqual(index, corpus + ("/collection/index" if family.startswith("qwen") else "/index"))
                        paths.append((corpus, dense_corpus, index))
                        command = shlex.split(report(result.stdout, "Command: "))
                        if stage == "encode":
                            self.assertEqual(command[command.index("--encode_type") + 1], kind)
                            self.assertEqual(command[command.index("--task_type") + 1], task)
                        elif stage == "three_stage_retrieval":
                            self.assertEqual(command[command.index("--query-type") + 1], query)
                            self.assertEqual(command[command.index("--task-type") + 1], task)
                            self.assertEqual(command[command.index("--stage1-sparse-index") + 1], index)
                            self.assertEqual(command[command.index("--stage2-dense-index") + 1], dense_corpus)
                            self.assertIn("stage1_1000_stage2_1000", report(result.stdout, "Results: "))
                self.assertTrue(all(path == paths[0] for path in paths))

    def test_explicit_qwen_aliases_are_idempotent_and_bad_configurations_fail(self):
        path = "/existing/experiment" + dense.SUFFIX + dense.PROMPT_SUFFIX
        for alias in (path, path + "/collection", path + "/collection/index"):
            result = self.run_script("three_stage_retrieval_freeret_ir.sh", environment(DRY_RUN="true", MODEL_FAMILY="qwen25",
                SPARSE_CORPUS_DIR=alias, DENSE_CORPUS_DIR=path, USE_FREERET_DENSE="true", USE_FREERET_PROMPT_CONSTRAINTS="true"))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(report(result.stdout, "Stage 1 index: "), path + "/collection/index")
            self.assertEqual(report(result.stdout, "Stage 2 dense: "), path)
        for name in NAMES:
            for invalid in (dict(USE_FREERET_DENSE="false", USE_FREERET_PROMPT_CONSTRAINTS="true"),
                            dict(USE_FREERET_DENSE="invalid"), dict(MODEL_FAMILIES="qwen2 qwen3")):
                result = self.run_script(name, environment(DRY_RUN="true", **invalid))
                self.assertNotEqual(result.returncode, 0)
        for group, values in (("ir", dict(QUERY_TYPE="composed")), ("tbpr", dict(ENCODE_TYPE="text")),
                              ("fused", dict(DATASET="flickr")), ("ir", dict(MODEL_FAMILY="qwen3", LLAVA_IMAGE_MODE="base"))):
            result = self.run_script(f"encode_freeret_{group}.sh", environment(DRY_RUN="true", **values))
            self.assertNotEqual(result.returncode, 0)

    def test_local_and_environment_switches_reach_actual_commands_and_failures_propagate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); scripts = root / "scripts"; scripts.mkdir()
            for name in NAMES + ["freeret_task_common.sh", "freeret_dense_mode.sh"]:
                shutil.copyfile(ROOT / "scripts" / name, scripts / name)
            model = root / "checkpoint with spaces"; model.mkdir(); (model / "config.json").write_text("{}")
            for dataset, folder, filename in (("flickr", "flickr", "flickr_test.csv"),
                    ("CUHK-PEDES", "CUHK-PEDES", "reid_raw.json"), ("webqa", "WebQA", "query-00000-of-00001.parquet")):
                parent = root / "data" / folder; parent.mkdir(parents=True)
                (parent / filename).touch()
                if dataset == "webqa":
                    for extra in ("qrels-00000-of-00001.parquet", "corpus-00000-of-00002.parquet", "corpus-00001-of-00002.parquet"):
                        (parent / extra).touch()
            capture = root / "capture.sh"
            capture.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' BEGIN "$@" >> "$CAPTURE_FILE"\nexit "${FAKE_EXIT-0}"\n')
            capture.chmod(0o755)
            log = root / "capture.txt"
            for name in NAMES:
                for use, prompts in ((False, False), (True, False), (True, True)):
                    for local in (False, True):
                        with self.subTest(script=name, use=use, prompts=prompts, local=local):
                            body = (ROOT / "scripts" / name).read_text(encoding="utf-8")
                            if local:
                                body = body.replace('${USE_FREERET_DENSE-false}', '${USE_FREERET_DENSE-' + str(use).lower() + '}')
                                body = body.replace('${USE_FREERET_PROMPT_CONSTRAINTS-false}', '${USE_FREERET_PROMPT_CONSTRAINTS-' + str(prompts).lower() + '}')
                            (scripts / name).write_text(body, encoding="utf-8")
                            env = environment(MODEL_PATH=model.as_posix(), DATA_ROOT=root.as_posix(),
                                PYTHON_BIN=capture.as_posix(), DEEPSPEED_BIN=capture.as_posix(), CAPTURE_FILE=log.as_posix())
                            if not local:
                                env.update(USE_FREERET_DENSE=str(use).lower(), USE_FREERET_PROMPT_CONSTRAINTS=str(prompts).lower())
                            log.write_text("")
                            result = self.run_script(name, env, scripts=scripts, cwd=root)
                            self.assertEqual(result.returncode, 0, result.stderr)
                            lines = log.read_text().splitlines()
                            for option, expected in (("--use_freeret_dense", use), ("--use_freeret_prompt_constraints", prompts)):
                                positions = [i for i, value in enumerate(lines) if value == option]
                                self.assertTrue(positions)
                                self.assertTrue(all(lines[i + 1] == str(expected).lower() for i in positions))
                            if name.startswith("three_stage"):
                                self.assertIn("check", lines)
                                self.assertIn("src/three_stage_retrieval_pipeline.py", lines)
            (scripts / NAMES[0]).write_text((ROOT / "scripts" / NAMES[0]).read_text(encoding="utf-8"), encoding="utf-8")
            env = environment(MODEL_PATH=model.as_posix(), DEEPSPEED_BIN=capture.as_posix(), CAPTURE_FILE=log.as_posix(), FAKE_EXIT="19")
            self.assertEqual(self.run_script(NAMES[0], env, scripts=scripts, cwd=root).returncode, 19)


class IndexTests(unittest.TestCase):
    def artifact(self, directory, family, use, prompts):
        root = Path(dense.mode_path(Path(directory) / "experiment", use, prompts))
        source = root / "collection" if family.startswith("qwen") else root
        source.mkdir(parents=True)
        for index in range(2):
            (source / f"corpus_{index}.jsonl").write_text(json.dumps(dict(id=str(index), vector=dict(word=10))) + "\n")
        (source / "manifest.json").write_text("{}")
        (root / "corpus_0.pkl").touch()
        dense.write_manifest(root, use, prompts)
        return root, source

    def test_only_corpus_shards_are_indexed_for_each_layout_and_mode(self):
        for family in ("llava_mistral", "qwen25"):
            for use, prompts in ((False, False), (True, False), (True, True)):
                with self.subTest(family=family, use=use, prompts=prompts), temporary_directory() as directory:
                    root, source = self.artifact(directory, family, use, prompts)
                    indexer.check_dense(root, use, prompts)
                    with self.assertRaisesRegex(ValueError, "Missing Lucene"):
                        indexer.check_sparse(root, family, use, prompts)
                    def run(command, **kwargs):
                        self.assertTrue(kwargs["check"])
                        inputs = Path(command[command.index("--input") + 1])
                        self.assertEqual(sorted(file.name for file in inputs.iterdir()), ["corpus_0.jsonl", "corpus_1.jsonl"])
                        self.assertIn("pyserini.index.lucene", command)
                        output = Path(command[command.index("--index") + 1]); output.mkdir(); (output / "segments_1").touch()
                    with patch.object(indexer.subprocess, "run", side_effect=run), redirect_stdout(io.StringIO()):
                        target = indexer.build_index(root, family, use_dense=use, use_constraints=prompts)
                    self.assertEqual(target, source / "index")
                    indexer.check_sparse(target, family, use, prompts)
                    self.assertFalse(list(root.glob(".freeret-index-*")))
                    with self.assertRaisesRegex(ValueError, "already exists"):
                        indexer.build_index(root, family, use_dense=use, use_constraints=prompts)

    def test_metadata_conflicts_and_indexer_failures_never_publish_an_index(self):
        with temporary_directory() as directory:
            root, source = self.artifact(directory, "llava_llama", True, True)
            with self.assertRaisesRegex(ValueError, "mode mismatch"):
                indexer.check_dense(root, False, False)
            with patch.object(indexer.subprocess, "run", side_effect=subprocess.CalledProcessError(17, "indexer")), redirect_stdout(io.StringIO()):
                status = indexer.main(["build", "--family", "llava_llama", "--sparse-root", str(root),
                                       "--use_freeret_dense", "true", "--use_freeret_prompt_constraints", "true"])
            self.assertEqual(status, 17)
            self.assertFalse((source / "index").exists())
            self.assertFalse(list(root.glob(".freeret-index-*")))
            dense.write_manifest(root, True, False)
            with self.assertRaisesRegex(ValueError, "mode mismatch"):
                indexer.sparse_layout(root, "llava_llama", True, True)


class ModelDirectoryTests(unittest.TestCase):
    def test_llava_output_model_name_handles_custom_relative_and_standard_paths(self):
        source = (ROOT / "src/encode.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        assignment = next(node for node in main.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "model_output_name" for target in node.targets))
        expression = compile(ast.Expression(assignment.value), "<output-name>", "eval")
        from types import SimpleNamespace
        for path in ("/root/autodl-fs/checkpoints/model", "/other root/checkpoints/model/", "./checkpoints/model", "model"):
            self.assertEqual(eval(expression, {"os": os, "model_args": SimpleNamespace(model_name_or_path=path)}), "model")
        self.assertNotIn("model_args.model_name_or_path[model_begin_indice:]", source)
        self.assertIn("model_args.lora_model_path[model_begin_indice:]", source)


if __name__ == "__main__":
    unittest.main()
