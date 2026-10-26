"""Paired VLM2Vec encode/search launcher tests without any model downloads."""

from pathlib import Path
import shutil
import subprocess
import sys
import unittest

import test_vlm2vec_dense_launcher as fixtures

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tevatron/src"))
BASH = fixtures.BASH


@unittest.skipUnless(BASH, "Bash is required for launcher tests")
class VLM2VecSearchLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import HfArgumentParser
        from arguments import (ModelArguments, PromptRepsLLMDataArguments,
                               PromptRepsLLMSearchArguments, TrainingArguments)
        from dense_encoding import DenseEncoderArguments
        from dense_search import DenseSearchArguments
        cls.parser = HfArgumentParser((ModelArguments, PromptRepsLLMDataArguments, PromptRepsLLMSearchArguments,
                                      TrainingArguments, DenseEncoderArguments, DenseSearchArguments))

    def setUp(self):
        fixtures.VLM2VecLauncherTests.setUp(self)
        shutil.copyfile(PROJECT_ROOT / "scripts/search_dense_vlm2vec.sh", self.root / "scripts/search_dense_vlm2vec.sh")
        for key in ("PASSAGE_REPS", "SAVE_DIR", "DEPTH", "USE_GPU", "VLM2VEC_QUERY_INSTRUCTION"):
            self.env.pop(key, None)

    value = staticmethod(fixtures.VLM2VecLauncherTests.value)

    def launch(self, mode, *args, **env):
        self.record.unlink(missing_ok=True)
        script = self.root / f"scripts/{mode}_dense_vlm2vec.sh"
        result = subprocess.run([BASH, script.as_posix(), *args], env={**self.env, **env},
                                text=True, capture_output=True, timeout=30)
        argv = self.record.read_bytes().decode().rstrip("\0").split("\0") if self.record.exists() else []
        return result, argv

    def success(self, mode, *args, **env):
        result, argv = self.launch(mode, *args, **env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(argv)
        self.assertEqual(self.value(argv, "--dense_model_type"), "vlm2vec")
        return result, argv

    def parse(self, argv, mode):
        start = next(index for index, value in enumerate(argv) if value.endswith(f"/src/{mode}_dense.py"))
        # Validate argv and path calculations on CPU without changing the GPU-default launch flags.
        return self.parser.parse_args_into_dataclasses(argv[start + 1:] + ["--use_cpu", "true"])

    def pair(self, *args, **env):
        from vlm2vec_dense import vlm2vec_output_directory
        encoded, encoder = self.success("encode", *args, **env)
        model, data, _, train, dense, _ = self.parse(encoder, "encode")
        directory = vlm2vec_output_directory(model, data, train, dense.vlm2vec_image_resolution)
        self.assertIn(directory.as_posix(), encoded.stdout)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "corpus_0.pkl").touch()
        searched, query = self.success("search", *args, **env)
        self.assertEqual(self.value(query, "--passage_reps"), directory.as_posix())
        self.assertIn(directory.as_posix(), searched.stdout)
        self.parse(query, "search")
        return encoder, query

    def test_syntax_defaults_match_encoder_and_resolution_result_path(self):
        for mode in ("encode", "search"):
            subprocess.run([BASH, "-n", (self.root / f"scripts/{mode}_dense_vlm2vec.sh").as_posix()], check=True, timeout=30)
        encoder, query = self.pair()
        self.assertEqual(self.value(query, "--model_name_or_path"), "TIGER-Lab/VLM2Vec-LLaVa-Next")
        self.assertEqual(self.value(query, "--query_type"), "text")
        self.assertEqual(self.value(query, "--vlm2vec_image_resolution"), "high")
        for argv in (encoder, query):
            self.assertIn("--bf16", argv)
            self.assertNotIn("deepspeed.launcher.runner", argv)
        # Git Bash may expose Windows TEMP as /tmp; derive its own project-root spelling.
        entrypoint = next(value for value in query if value.endswith("/src/search_dense.py"))
        expected = entrypoint.removesuffix("/src/search_dense.py") + "/search_results/VLM2Vec-LLaVa-Next/flickr/text/test/vlm2vec_high"
        self.assertEqual(self.value(query, "--save_dir"), expected)

    def test_all_eight_datasets_match_python_index_paths(self):
        for name, task, direction, corpus in (("coco", "ir", "text", "image"), ("flickr", "ir", "text", "image"),
                ("CUHK-PEDES", "tbpr", "text", "image"), ("ICFG-PEDES", "tbpr", "text", "image"),
                ("RSTPReid", "tbpr", "text", "image"), ("webqa", "t2it", "text", "corpus"),
                ("remuq", "it2t", "image", "corpus"), ("llava", "it2t", "image", "corpus")):
            with self.subTest(dataset=name):
                encoder, query = self.pair(DATASET=name)
                self.assertEqual(self.value(query, "--task_type"), task)
                self.assertEqual(self.value(query, "--query_type"), direction)
                self.assertEqual(self.value(encoder, "--encode_type"), corpus)

    def test_both_ir_directions_and_composed_queries(self):
        for name in ("coco", "flickr"):
            for env in ({"QUERY_TYPE": "image"}, {"ENCODE_TYPE": "text"}):
                encoder, query = self.pair(DATASET=name, **env)
                self.assertEqual(self.value(query, "--query_type"), "image")
                self.assertEqual(self.value(encoder, "--encode_type"), "text")
        _, query = self.pair("--dataset_name=remuq", "--query_type=composed")
        self.assertEqual(self.value(query, "--query_type"), "composed")

    def test_local_cache_and_resolution_overrides_match_encoder(self):
        for resolution in ("high", "mid", "low", "original"):
            _, query = self.pair("--dataset_name=webqa", "--vlm2vec_image_resolution=low",
                "--vlm2vec_image_resolution", resolution, "--cache_dir", "/custom cache",
                "--model_name_or_path", "/local weights/checkpoint/", VLM2VEC_IMAGE_RESOLUTION="mid")
            self.assertEqual(self.value(query, "--vlm2vec_image_resolution"), resolution)
            self.assertEqual(query.count("--vlm2vec_image_resolution"), 1)
            self.assertEqual(self.value(query, "--cache_dir"), "/custom cache")
            self.assertTrue(self.value(query, "--save_dir").endswith(f"/vlm2vec_{resolution}"))

    def test_custom_instruction_search_options_and_passthrough(self):
        self.pair()
        _, query = self.success("search", "--vlm2vec_query_instruction=First task:",
            "--vlm2vec_query_instruction", "Final task:", "--depth=17", "--use_gpu=true", "--remove_query",
            "--quiet", "--retrieval_batch_size=3", "--use_generated_validation_data=true",
            VLM2VEC_QUERY_INSTRUCTION="Environment task:")
        _, data, args, _, _, native = self.parse(query, "search")
        self.assertEqual(native.vlm2vec_query_instruction, "Final task:")
        self.assertEqual(query.count("--vlm2vec_query_instruction"), 1)
        self.assertEqual(args.depth, 17)
        self.assertTrue(args.use_gpu)
        self.assertTrue(args.remove_query)
        self.assertEqual(args.retrieval_batch_size, 3)
        self.assertTrue(data.use_generated_validation_data)

    def test_explicit_index_and_result_directory_are_preserved(self):
        directory = self.root / "external index"
        directory.mkdir()
        file = directory / "corpus_9.pkl"
        file.touch()
        for path in (directory, file):
            _, query = self.success("search", "--passage_reps", path.as_posix(), "--save_dir", "/custom results",
                                    PASSAGE_REPS="/wrong index", SAVE_DIR="/wrong results")
            self.assertEqual(self.value(query, "--passage_reps"), path.as_posix())
            self.assertEqual(self.value(query, "--save_dir"), "/custom results")

    def test_path_compatibility_labels_and_cli_last_value_win(self):
        _, query = self.pair("--dataset_name=coco", "--dataset_name", "RSTPReid", "--dataset_split=val",
            "--is_filtered=yes", "--sparse_manual=true", "--use_output_embedding_cluster=true", "--cluster_sum=008",
            "--sparse_value_mean=true", "--num_expended_tokens=003", "--sparse_length=040", "--prompt_type=prompt_1",
            "--eol_type=eol", "--reps_loc=after_pad", "--sparse_value_type=sum", "--sparse_lower_or_upper=upper",
            "--calculate_type=concat", "--tbpr_type=custom", DATASET="webqa")
        self.assertEqual(self.value(query, "--dataset_name"), "RSTPReid")
        self.assertIn("/custom/3_manual_40_sum_cluster_8_after_pad_eol_upper_mean_vlm2vec_high",
                      self.value(query, "--passage_reps"))

    def test_precision_physical_gpu_selection_and_single_process_mask(self):
        for processes in ("1", "2"):
            encoder, query = self.pair("--fp16", NUM_PROCESSES=processes, CUDA_VISIBLE_DEVICES="2,3,5")
            for argv in (encoder, query):
                self.assertIn("--fp16", argv)
                self.assertNotIn("--bf16", argv)
            if processes == "2":
                self.assertEqual(query[:2], ["-m", "deepspeed.launcher.runner"])
                self.assertEqual(self.value(query, "--include"), "localhost:2,3")
                self.assertEqual(self.visible.read_text(), "unset")
            else:
                self.assertEqual(self.visible.read_text(), "2,3,5")

    def test_invalid_settings_and_missing_indexes_fail_before_python(self):
        result, argv = self.launch("search")
        self.assertEqual(result.returncode, 2)
        self.assertIn("encode_dense_vlm2vec.sh", result.stderr)
        self.assertEqual(argv, [])
        empty = self.root / "empty"
        empty.mkdir()
        for env in ({"DATASET": "bad"}, {"VLM2VEC_IMAGE_RESOLUTION": "bad"}, {"DEPTH": "0"},
                    {"DENSE_MODEL_TYPE": "unime"}, {"DATASET": "webqa", "QUERY_TYPE": "image"},
                    {"VLM2VEC_QUERY_INSTRUCTION": "   "}, {"VLM2VEC_QUERY_INSTRUCTION": "Task <image>"},
                    {"PASSAGE_REPS": empty.as_posix()}):
            result, argv = self.launch("search", **env)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertEqual(argv, [])
        for args in (("--vlm2vec_query_instruction=",), ("--passage_reps",), ("--vlm2vec_image_resolution=",)):
            result, argv = self.launch("search", *args)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(argv, [])


if __name__ == "__main__":
    unittest.main()
