"""Public task-launcher settings reach both batch dispatch and actual worker argv."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from test_freeret_dense import ROOT
import adaptive_retrieval_batch as batch


BASH = "D:/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
SCRIPTS = {group: f"three_stage_retrieval_pipeline_adaptive_views_{group}.sh" for group in ("ir", "tbpr", "fused")}
DEFAULTS = dict(MODEL_FAMILIES="qwen25", STAGE2_METHOD="dense", USE_STC_SPARSE="false",
    USE_VIEW_PROBABILITY_MASS="true", VIEW_PROBABILITY_MASS_THRESHOLD="0.95", USE_HIERARCHICAL_VIEWS="true",
    VIEW_EXAMPLE_MODE="auto", VIEW_EXAMPLE_MAX_TOKENS="16384", USE_FREERET_DENSE="true",
    USE_FREERET_FULL_PROMPT="true", USE_FREERET_PROMPT_CONSTRAINTS="false", GPUS="2", NUM_NODES="1",
    DTYPE="float16", RETRIEVAL_DTYPE="float16", RANKER_DTYPE="float16", OVERWRITE="true",
    REPS_LOC="after_pad", SPARSE_LENGTH="30", TEXT_SPARSE_LENGTH="30", IMAGE_SPARSE_LENGTH="30",
    SPARSE_VALUE_TYPE="sum", SPARSE_LOWER_OR_UPPER="lower", TBPR_TYPE="type", LLAVA_IMAGE_MODE="anyres",
    QWEN_MIN_PIXELS="0", QWEN_MAX_PIXELS="401408", RANKER_QWEN_MIN_PIXELS="0", RANKER_QWEN_MAX_PIXELS="401408",
    VIEW_SELECTION_ROUNDS="20", VIEW_TEXTS_PER_ROUND="30", VIEW_IMAGES_PER_ROUND="10", VIEW_SELECTION_SEED="42",
    STAGE1_DEPTH="200", STAGE2_DEPTH="200", RETRIEVAL_BATCH_SIZE="4", SPARSE_THREADS="16",
    RERANK_NUM="20", RERANK_BATCH_SIZE="20", HYBRID_ALPHA="0.5", SELECT_HYBRID_ALPHA="false")
DATASETS = dict(ir="flickr", tbpr="CUHK-PEDES ICFG-PEDES RSTPReid", fused="webqa remuq llava")


@unittest.skipUnless(BASH and Path(BASH).is_file(), "Bash unavailable")
class LauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        scripts = self.root / "scripts"
        scripts.mkdir()
        names = [*SCRIPTS.values(), "freeret_dense_mode.sh", "stage2_retrieval_mode.sh", "stc_sparse_mode.sh"]
        self.variables = set(DEFAULTS) | {"CHECKPOINT_ROOT", "DATA_ROOT", "DENSE_OUTPUT_ROOT", "SPARSE_OUTPUT_ROOT",
            "ADAPTIVE_OUTPUT_ROOT", "MODEL_FAMILY", "DATASET", "DATASETS", "IR_QUERY_TYPES", "IR_QUERY_TYPE",
            "OUTPUT_DIR", "MODEL_PATH", "STAGE2_MODE_REPORTED", "VIEW_SELECTION_FILE", "DENSE_CORPUS_DIR",
            "SPARSE_CORPUS_DIR", "STAGE1_SPARSE_INDEX", "JOB_MODIFY_TYPE"}
        for name in names:
            body = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.variables.update(re.findall(r'([A-Z_0-9]+)="\$\{', body))
            (scripts / name).write_text(body, encoding="utf-8")
        self.args_file, self.values_file = self.root / "argv", self.root / "values"
        self.capture = self.root / "capture.sh"
        self.capture.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n'
            'for name in ' + " ".join(sorted(self.variables)) + '; do\n'
            'printf "%s=%s\\n" "$name" "${!name-}"\ndone > "$CAPTURE_VALUES"\n', encoding="utf-8")
        self.capture.chmod(0o755)

    def launch(self, group, *, worker=False, **changes):
        env = {key: value for key, value in os.environ.items() if key not in self.variables}
        env.update(PYTHON_BIN=self.capture.as_posix(), DEEPSPEED_BIN=self.capture.as_posix(),
                   CAPTURE_ARGS=self.args_file.as_posix(), CAPTURE_VALUES=self.values_file.as_posix(),
                   MSYS_NO_PATHCONV="1", MSYS2_ARG_CONV_EXCL="*", MSYS2_ENV_CONV_EXCL="*")
        env.update(changes)
        command = [BASH, (self.root / "scripts" / SCRIPTS[group]).as_posix()]
        if worker:
            command.append("--run-one")
        result = subprocess.run(command, env=env, cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = self.args_file.read_text(encoding="utf-8").splitlines()
        values = dict(line.split("=", 1) for line in self.values_file.read_text(encoding="utf-8").splitlines())
        return argv, values

    def option(self, argv, name):
        return argv[argv.index(name) + 1]

    def test_defaults_reach_dispatcher_and_preserve_all_task_directions(self):
        for group in SCRIPTS:
            with self.subTest(group=group):
                argv, values = self.launch(group, SELECT_HYBRID_ALPHA="true", HYBRID_ALPHA="0.9")
                self.assertEqual(argv[:5], ["-B", "src/adaptive_retrieval_batch.py", "--task-group", group, "--launcher"])
                self.assertTrue(argv[5].endswith(SCRIPTS[group]))
                for name, expected in DEFAULTS.items():
                    self.assertEqual(values[name], expected, name)
                self.assertEqual(values["DATASETS"], DATASETS[group])
                for key, expected in dict(CHECKPOINT_ROOT="/root/gpufree-share/checkpoints", DATA_ROOT="/root/gpufree-share",
                    DENSE_OUTPUT_ROOT="/root/gpufree-share/dense_output", SPARSE_OUTPUT_ROOT="/root/gpufree-share/sparse_output",
                    ADAPTIVE_OUTPUT_ROOT="/root/gpufree-share/three_stage_adaptive_results").items():
                    self.assertEqual(values[key], expected)
                settings = batch.Settings.from_environment(group, values, project=self.root)
                self.assertEqual(settings.families, ("qwen25",))
                jobs = [(dataset, *task) for dataset in settings.datasets for task in batch.job_types(settings, dataset)]
                if group == "ir":
                    self.assertEqual(jobs, [("flickr", "ir", "image", "text"), ("flickr", "ir", "text", "image")])
                elif group == "tbpr":
                    self.assertEqual(jobs, [(name, "tbpr", "text", "image") for name in DATASETS[group].split()])
                else:
                    self.assertEqual(jobs, [("webqa", "t2it", "text", "corpus"), ("remuq", "it2t", "image", "corpus"),
                                            ("llava", "it2t", "image", "corpus")])

    def test_worker_argv_has_reference_defaults_and_direction_derived_modify_type(self):
        cases = [("ir", "flickr", "ir", "image", "modify_single"), ("ir", "flickr", "ir", "text", "no")]
        cases += [("tbpr", name, "tbpr", "text", "no") for name in DATASETS["tbpr"].split()]
        cases += [("fused", "webqa", "t2it", "text", "no"), ("fused", "remuq", "it2t", "image", "modify_single"),
                  ("fused", "llava", "it2t", "image", "modify_single")]
        for group, dataset, task, query, modify in cases:
            with self.subTest(group=group, dataset=dataset, query=query):
                self.assertEqual(batch.caption_modify_type(task, query), modify)
                suffix = "_dense_freeret_pre_mlp_freeret_full_prompt_v2"
                output = self.root / "custom results" / ("stage1_200_stage2_200_stage2_dense" + suffix)
                argv, _ = self.launch(group, worker=True, MODEL_FAMILY="qwen25", DATASET=dataset, IR_QUERY_TYPE=query,
                    MODEL_PATH="/root/gpufree-share/checkpoints/Qwen-Qwen2.5-VL-7B-Instruct", OUTPUT_DIR=output.as_posix(),
                    VIEW_SELECTION_FILE="selection.json", DENSE_CORPUS_DIR="dense" + suffix, SPARSE_CORPUS_DIR="sparse" + suffix,
                    JOB_MODIFY_TYPE=modify, SELECT_HYBRID_ALPHA="true", HYBRID_ALPHA="0.9")
                self.assertEqual(argv[:3], ["--num_gpus=2", "--num_nodes=1", "src/three_stage_retrieval_pipeline_adaptive_views.py"])
                expected = {"--task-type": task, "--query-type": query, "--modify-type": modify,
                    "--stage2-method": "dense", "--view-probability-mass": "true", "--view-probability-mass-threshold": "0.95",
                    "--use_freeret_dense": "true", "--use_freeret_full_prompt": "true", "--use_freeret_prompt_constraints": "false",
                    "--stage1-depth": "200", "--stage2-depth": "200", "--retrieval-batch-size": "4", "--sparse-length": "30",
                    "--text-sparse-length": "30", "--image-sparse-length": "30", "--tbpr-type": "type",
                    "--retrieval-qwen-max-pixels": "401408", "--ranker-qwen-max-pixels": "401408",
                    "--rerank-num": "20", "--rerank-batch-size": "20", "--data-root": "/root/gpufree-share",
                    "--output-dir": output.as_posix(), "--rerank-template": "what_caption_generation"}
                for flag, value in expected.items():
                    self.assertEqual(self.option(argv, flag), value, flag)
                self.assertIn("--overwrite", argv)
                self.assertNotIn("--select-hybrid-alpha", argv)

    def test_environment_overrides_keep_batch_lists_custom_roots_and_worker_flags(self):
        for group in SCRIPTS:
            with self.subTest(group=group):
                changes = dict(MODEL_FAMILIES="qwen25 llava_mistral", DATASETS="flickr coco" if group == "ir" else DATASETS[group],
                    STAGE2_METHOD="hybrid", USE_VIEW_PROBABILITY_MASS="false", VIEW_PROBABILITY_MASS_THRESHOLD="0.7",
                    USE_HIERARCHICAL_VIEWS="false", USE_FREERET_FULL_PROMPT="false", USE_FREERET_PROMPT_CONSTRAINTS="true",
                    SPARSE_LENGTH="11", GPUS="1", RETRIEVAL_BATCH_SIZE="2", RERANK_NUM="3", RERANK_BATCH_SIZE="1",
                    OVERWRITE="false", VIEW_EXAMPLE_MODE="fixed", VIEW_EXAMPLE_MAX_TOKENS="2048",
                    HYBRID_ALPHA="0.7", SELECT_HYBRID_ALPHA="true", ADAPTIVE_OUTPUT_ROOT=(self.root / "other results").as_posix(),
                    USE_STC_SPARSE="true", STC_ALGORITHM="minibatch_kmeans", STC_N_CLUSTERS="8")
                _, values = self.launch(group, **changes)
                for name, value in changes.items():
                    self.assertEqual(values[name], value, name)
                settings = batch.Settings.from_environment(group, values, project=self.root)
                self.assertEqual(settings.families, ("qwen25", "llava_mistral"))
                self.assertEqual(settings.roots["results"], self.root / "other results")
                dataset = "flickr" if group == "ir" else "RSTPReid" if group == "tbpr" else "webqa"
                argv, _ = self.launch(group, worker=True, **dict(changes, MODEL_FAMILY="llava_mistral", DATASET=dataset,
                    MODEL_PATH="fixture/checkpoint", OUTPUT_DIR="fixture/output", VIEW_SELECTION_FILE="selection.json",
                    DENSE_CORPUS_DIR="fixture/dense", SPARSE_CORPUS_DIR="fixture/sparse", JOB_MODIFY_TYPE="no", IR_QUERY_TYPE="text"))
                self.assertIn("--num_gpus=1", argv)
                self.assertIn("--select-hybrid-alpha", argv)
                self.assertNotIn("--overwrite", argv)
                for flag, value in {"--stage2-method": "hybrid", "--hybrid-alpha": "0.7", "--sparse-length": "11",
                    "--retrieval-batch-size": "2", "--use_freeret_full_prompt": "false", "--use_freeret_prompt_constraints": "true",
                    "--use_stc_sparse": "true", "--stc_algorithm": "minibatch_kmeans", "--stc_n_clusters": "8"}.items():
                    self.assertEqual(self.option(argv, flag), value, flag)


if __name__ == "__main__":
    unittest.main()
