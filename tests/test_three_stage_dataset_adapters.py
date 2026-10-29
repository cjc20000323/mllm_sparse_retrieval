from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from PIL import Image

import test_three_stage_retrieval_pipeline as fixtures


pipeline = fixtures.pipeline
PROJECT_ROOT = fixtures.PROJECT_ROOT
PROFILES = {
    "CUHK-PEDES": ("tbpr", "text", "image", "CUHK-PEDES"),
    "ICFG-PEDES": ("tbpr", "text", "image", "ICFG-PEDES"),
    "RSTPReid": ("tbpr", "text", "image", "RSTPReid"),
    "webqa": ("t2it", "text", "corpus", "WebQA"),
    "remuq": ("it2t", "image", "corpus", "UMRB-ReMuQ"),
    "llava": ("it2t", "image", "corpus", "UMRB-LLaVA"),
}
PERSON_FILES = {
    "CUHK-PEDES": "reid_raw.json", "ICFG-PEDES": "ICFG-PEDES.json",
    "RSTPReid": "data_captions.json",
}
PATH_CONSTANTS = (
    "dataset_path_prefix", "coco_file_path", "flickr_file_path", "fashion_iq_file_path",
    "cuhk_pedes_file_path", "icfg_pedes_flie_path", "rstpreid_file_path",
    "webqa_file_path", "remuq_file_path", "llava_file_path", "edis_file_path",
)


def args_for(name, **overrides):
    task, query_type, _, _ = PROFILES[name]
    values = dict(dataset_name=name, task_type=task, query_type=query_type,
                  ranker_dtype="float32", rerank_batch_size=2)
    values.update(overrides)
    return fixtures.ThreeStageRecallTests.args(**values)


def write_dataset(root, name, image_size=(4, 3)):
    """Tiny real JSON/Parquet files; never download data or start a VLM."""
    task, _, _, directory = PROFILES[name]
    data_dir = root / "data" / directory
    data_dir.mkdir(parents=True)
    image = Image.new("RGB", image_size, "red")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image.save(data_dir / "query.png")
    images = [{"bytes": buffer.getvalue(), "path": None},
              {"bytes": None, "path": "query.png"}]
    if task == "tbpr":
        (data_dir / "imgs").mkdir()
        records = []
        for index, person in enumerate((7, 7, 8)):
            image.save(data_dir / "imgs" / f"{index}.png")
            path_key = "img_path" if name == "RSTPReid" else "file_path"
            records.append({"split": "test", "id": person, path_key: f"{index}.png",
                            "captions": [f"person description {index}"]})
        (data_dir / PERSON_FILES[name]).write_text(json.dumps(records), encoding="utf-8")
        return
    query = pd.DataFrame({"id": [1, 2], "text": ["question one", "question two"]})
    if task == "t2it":
        query.to_parquet(data_dir / "query-00000-of-00001.parquet", index=False)
        pd.DataFrame({"query-id": [1, 1, 2], "corpus-id": [10, 11, 11]}).to_parquet(
            data_dir / "qrels-00000-of-00001.parquet", index=False)
        for index in range(2):
            pd.DataFrame({"id": [10 + index], "text": [f"candidate {index}"],
                          "image": [images[index]]}).to_parquet(
                data_dir / f"corpus-{index:05d}-of-00002.parquet", index=False)
    else:
        query["image"] = images
        query.to_parquet(data_dir / "query.parquet", index=False)
        pd.DataFrame({"query-id": [1, 2], "corpus-id": [10, 11]}).to_parquet(
            data_dir / "qrels.parquet", index=False)
        pd.DataFrame({"id": [10, 11], "text": ["answer one", "answer two"]}).to_parquet(
            data_dir / "corpus.parquet", index=False)


def relevant_run(name):
    if PROFILES[name][0] == "tbpr":
        # A different photo of the same person must count as a correct result.
        return {"0": {"1": 1.0}, "1": {"0": 1.0}, "2": {"2": 1.0}}
    if name == "webqa":
        return {"1": {"11": 1.0}, "2": {"11": 1.0}}
    return {"1": {"10": 1.0}, "2": {"11": 1.0}}


class DatasetAdapterTests(unittest.TestCase):
    def setUp(self):
        src_path = patch.object(sys, "path", [str(PROJECT_ROOT / "src"), *sys.path])
        src_path.start()
        self.addCleanup(src_path.stop)

    @contextmanager
    def dataset(self, name):
        import dataset as dataset_module
        paths = {key: getattr(dataset_module, key) for key in PATH_CONSTANTS}
        with tempfile.TemporaryDirectory() as directory, patch.multiple(dataset_module, **paths):
            root = Path(directory)
            write_dataset(root, name)
            args = args_for(name, data_root=str(root), output_dir=str(root / "output"))
            with redirect_stdout(StringIO()):
                dataset = pipeline.build_dataset(args)
            yield root, args, dataset

    def test_all_six_load_native_formats_query_sides_and_recall(self):
        for name in PROFILES:
            with self.subTest(dataset=name), self.dataset(name) as (_, args, dataset):
                pipeline._validate_args(args)
                records = pipeline._query_records(args, dataset)
                self.assertEqual([record["id"] for record in records], list(relevant_run(name)))
                self.assertTrue(all(record["text"] for record in records))
                with patch.object(pipeline, "build_dataset", return_value=dataset):
                    evaluation = pipeline._build_recall_evaluation(args, relevant_run(name))
                self.assertEqual(evaluation.k_values, [1, 5, 10, 100, 200, 300, 400])
                for stage in ("stage1", "stage2", "stage3"):
                    row = pipeline.evaluate_recall(relevant_run(name), evaluation,
                                                   stage=stage, variant="test")[0]
                    self.assertTrue(all(value == 1 for value in row["recall"].values()))
                if args.task_type == "it2t":
                    self.assertTrue(all(pipeline._image_from_value(r["image"]).size == (4, 3) for r in records))
                elif args.task_type == "t2it":
                    self.assertEqual(len(dataset.get_target(1)), 2)
                    self.assertTrue(all(pipeline._image_from_value(r["image"]).size == (4, 3)
                                        for r in dataset.id2candidate.values()))

    def test_task_specific_encoding_dispatch_keeps_both_query_modalities(self):
        for name in PROFILES:
            with self.subTest(dataset=name), self.dataset(name) as (_, args, dataset):
                records = pipeline._query_records(args, dataset)
                model = Mock()
                captured = []
                def processor(**kwargs):
                    captured.append(kwargs)
                    return {"input_ids": np.ones((len(kwargs["text"]), 1))}
                runtime = SimpleNamespace(model=model, processor=processor, device="cpu",
                                          model_path="llava-hf-llava-v1.6-mistral-7b-hf")
                model_args, data_args = pipeline._retrieval_namespaces(
                    args, eol_type=pipeline.STAGE2_EOL_TYPE, calculate_type="concat", sparse_manual=False)
                pipeline._encode_query_batch(runtime, args, records, model_args, data_args)
                method = getattr(model, "encode_data_concat_for_" + args.task_type)
                method.assert_called_once()
                self.assertEqual(method.call_args.args[1], "text" if args.task_type == "tbpr" else "query")
                if args.task_type == "it2t":
                    self.assertEqual(len(captured[0]["images"]), len(records))
                    for record, text in zip(records, captured[0]["text"]):
                        self.assertIn(record["text"], text)
                        self.assertIn("<image>", text)

    def test_real_ranker_requests_accept_native_ids_and_caption_templates(self):
        import reranker
        for name in PROFILES:
            for template in ("caption_generation", "what_caption_generation", "relevant"):
                with self.subTest(dataset=name, template=template), self.dataset(name) as (root, args, dataset):
                    args.rerank_template = template
                    args.ranker_model_name_or_path = "llava-hf-llava-v1.6-mistral-7b-hf"
                    seen = []
                    def scores(ranker, requests, *unused, **kwargs):
                        seen.extend(requests)
                        self.assertTrue(all(group[0].size == (4, 3) for group in ranker._batch_images(requests)))
                        return [0.5] * len(requests)
                    processor = SimpleNamespace(tokenizer=SimpleNamespace(get_vocab=lambda: {}))
                    with (
                        patch.object(pipeline, "build_dataset", return_value=dataset),
                        patch.object(pipeline, "_load_ranker_model", return_value=(SimpleNamespace(), processor)),
                        patch.object(pipeline.gc, "collect"),
                        patch.object(reranker, "configure_rerank_processor"),
                        patch.object(reranker.Reranker, "_caption_scores", autospec=True, side_effect=scores),
                        patch.object(reranker.Reranker, "_pointwise_scores", autospec=True, side_effect=scores),
                        redirect_stderr(StringIO()), redirect_stdout(StringIO()),
                    ):
                        result = pipeline.run_stage3(args, relevant_run(name))
                    self.assertEqual(set(result), set(relevant_run(name)))
                    for query_id, values in result.items():
                        self.assertEqual(set(values), set(relevant_run(name)[query_id]))
                    self.assertEqual(len(seen), len(result))
                    if args.task_type == "it2t":
                        for request, native_query in zip(seen, dataset.query_id_list):
                            self.assertIn(dataset.get_query(native_query)["text"], request.text)

    def test_multimodal_prompts_keep_text_and_dataset_specific_views(self):
        import template
        for name in ("remuq", "llava"):
            for model_path in ("llava-hf-llava-v1.6-mistral-7b-hf", "llava-hf-llama3-llava-next-8b-hf"):
                with self.subTest(dataset=name, model=model_path):
                    prompt = pipeline._concat_image_prompt(SimpleNamespace(model_path=model_path), args_for(name), "it2t")
                    self.assertIn("<sent>", prompt)
                    self.assertIn("<image>", prompt)
                    expected = (template.retrieval_disassemble_query_prompts_llava_it2t_retrieval_for_concat
                                if name == "llava" else template.retrieval_disassemble_query_prompts_it2t_retrieval_for_concat)
                    self.assertTrue(all(view in prompt for view in expected))

    def test_id_errors_are_explicit_and_happen_before_model_loading(self):
        args = args_for("llava")
        dataset = SimpleNamespace(query_id_list=[1], candidate_id_list=[10])
        for run in ({"missing": {"10": 1}}, {"1": {"missing": 1}}):
            with self.subTest(run=run), patch.object(pipeline, "build_dataset", return_value=dataset), \
                    patch.object(pipeline, "_load_ranker_model") as loader:
                with self.assertRaisesRegex(ValueError, "ID.*split"):
                    pipeline.run_stage3(args, run)
                loader.assert_not_called()
        for query_ids, candidate_ids in (([1, "1"], [10]), ([1], [10, "10.0"])):
            with self.assertRaisesRegex(ValueError, "歧义"):
                pipeline._native_ranker_run(args, SimpleNamespace(
                    query_id_list=query_ids, candidate_id_list=candidate_ids), {"1": {"10": 1}})

    def test_image_byte_and_path_fallbacks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            image = Image.new("L", (3, 2))
            image.save(path)
            payload = path.read_bytes()
            for value in (image, payload, memoryview(payload), {"bytes": payload, "path": "missing"},
                          {"bytes": None, "path": str(path)}, {"bytes": b"", "path": str(path)}):
                decoded = pipeline._image_from_value(value)
                self.assertEqual((decoded.mode, decoded.size), ("RGB", (3, 2)))
            for value in ({"bytes": None}, {"bytes": b"", "path": None}, None):
                with self.assertRaisesRegex(ValueError, "图像"):
                    pipeline._image_from_value(value)

    def test_incompatible_tasks_queries_and_prior_cache_are_rejected(self):
        for name, (task, query, _, _) in PROFILES.items():
            with self.subTest(dataset=name):
                with self.assertRaisesRegex(ValueError, "task-type"):
                    pipeline._validate_args(args_for(name, task_type="ir"))
                with self.assertRaisesRegex(ValueError, "query-type"):
                    pipeline._validate_args(args_for(name, query_type="image" if query == "text" else "text"))
                args = args_for(name, use_rerank_prior_cache=True, modify_type="modify_single",
                                rerank_template="caption_generation")
                if task == "it2t":
                    pipeline._validate_args(args)
                else:
                    with self.assertRaisesRegex(ValueError, "Prior caching"):
                        pipeline._validate_args(args)
        with self.assertRaisesRegex(ValueError, "tbpr-type"):
            pipeline._validate_args(args_for("CUHK-PEDES", tbpr_type="unknown"))

    def test_all_six_preserve_three_stage_report_outputs(self):
        for name in PROFILES:
            with self.subTest(dataset=name), self.dataset(name) as (root, args, dataset):
                run_path = root / "run.json"
                run_path.write_text(json.dumps(relevant_run(name)), encoding="utf-8")
                argv = ["pipeline", "--dataset-name", name, "--task-type", args.task_type,
                        "--query-type", args.query_type, "--data-root", str(root),
                        "--stage1-run", str(run_path), "--stage2-run", str(run_path),
                        "--ranker-run", str(run_path), "--output-dir", args.output_dir]
                with (
                    patch.object(sys, "argv", argv),
                    patch.object(pipeline, "_progress", side_effect=fixtures.ProgressRecorder()),
                    patch.object(pipeline.RecallReporter, "_output_dependencies", side_effect=fixtures.fake_excel_dependencies),
                    redirect_stdout(StringIO()), redirect_stderr(StringIO()),
                ):
                    pipeline.main()
                output = Path(args.output_dir)
                metadata = json.loads((output / "pipeline_metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["task_type"], args.task_type)
                self.assertEqual(metadata["recall"]["query_count"], len(relevant_run(name)))
                timing = json.loads((output / "timing_metrics.json").read_text(encoding="utf-8"))
                self.assertEqual(timing["status"], "completed")
                for filename in metadata["files"]:
                    self.assertTrue((output / filename).is_file(), filename)

    def test_query_partitions_and_stage2_reuse_for_every_dataset(self):
        import torch.distributed as dist
        for name in PROFILES:
            with self.dataset(name) as (_, args, dataset):
                args.stage1_method, args.stage1_run, args.stage2_run = "sparse_concat", None, None
                args.stage2_dense_index, args.stage2_sparse_index = "dense", "sparse"
                full_run = relevant_run(name)
                for world_size in (1, 2, 8):
                    packets, all_ids = [], []
                    current_rank = 0
                    def scatter(received, outgoing, src):
                        if outgoing is not None:
                            packets[:] = outgoing
                        received[0] = packets[current_rank]
                    with patch.object(pipeline, "build_dataset", return_value=dataset), \
                            patch.object(dist, "scatter_object_list", side_effect=scatter):
                        for current_rank in range(world_size):
                            context = pipeline.DistributedContext(current_rank, current_rank, world_size)
                            ids, _, _ = pipeline._stage1_query_shard(args, context)
                            all_ids.extend(ids)
                            encoded = pipeline.EncodedQueries(ids, np.ones((len(ids), 2), dtype="float32"),
                                                              {key: "topic" for key in ids})
                            local_run = {key: full_run[key] for key in ids}
                            timer = pipeline.StageTimer()
                            with (
                                patch.object(pipeline, "encode_queries_online", side_effect=AssertionError("reencoded")),
                                patch.object(pipeline, "retrieve_dense_from_corpus", return_value=local_run),
                                patch.object(pipeline, "retrieve_sparse_topics_from_lucene", return_value=local_run),
                                patch.object(pipeline, "_progress", side_effect=fixtures.ProgressRecorder()),
                            ):
                                _, _, result = pipeline.run_stage2(args, local_run, encoded_queries=encoded, timing=timer)
                            self.assertEqual(set(result), set(ids))
                            self.assertEqual(timer.components["query_encoding"], 0)
                    self.assertCountEqual(all_ids, list(full_run))


@unittest.skipUnless(shutil.which("bash"), "Bash is required to execute the launcher tests")
class DatasetLauncherTests(unittest.TestCase):
    def test_bash_syntax(self):
        subprocess.run([shutil.which("bash"), "-n", (PROJECT_ROOT / "scripts/three_stage_retrieval_pipeline.sh").as_posix()], check=True)

    def test_default_paths_commands_missing_files_and_task_conflicts(self):
        for name, (task, query, target, directory) in PROFILES.items():
            with self.subTest(dataset=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / "scripts").mkdir()
                (root / "src").mkdir()
                (root / "src/three_stage_retrieval_pipeline.py").touch()
                script = root / "scripts/three_stage_retrieval_pipeline.sh"
                shutil.copyfile(PROJECT_ROOT / "scripts/three_stage_retrieval_pipeline.sh", script)
                write_dataset(root, name)
                model = root / "checkpoint"
                model.mkdir()
                leaf = "0_no_manual_40_sum_no_cluster_after_pad_disassembleeol_separate_lower_no_mean"
                relative = f"model/{name}/{target}/no_filter/concat/prompt_5/test/"
                relative += ("origin_type/" if task == "tbpr" else "") + leaf
                dense, sparse = root / "dense_output" / relative, root / "sparse_output" / relative
                dense.mkdir(parents=True)
                (sparse / "index").mkdir(parents=True)
                (dense / "corpus_0.pkl").touch()
                (sparse / "index/segments_1").touch()
                stub = root / "deepspeed"
                stub.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "$PIPELINE_TEST_ARGV"\n', encoding="utf-8")
                stub.chmod(0o755)
                output = root / "argv.bin"
                env = {key: value for key, value in os.environ.items() if key not in {
                    "TASK_TYPE", "QUERY_TYPE", "TBPR_TYPE", "STAGE1_SPARSE_INDEX", "STAGE2_DENSE_INDEX",
                    "STAGE2_SPARSE_INDEX", "OUTPUT_DIR", "USE_RERANK_PRIOR_CACHE", "ARTIFACT_MODEL_NAME",
                    "BASE_MODEL", "SPARSE_LENGTH", "DATASET_SPLIT", "RETRIEVAL_REPS_LOC", "NUM_EXPENDED_TOKENS",
                    "SPARSE_VALUE_TYPE", "SPARSE_LOWER_OR_UPPER", "PROMPT_TYPE"}}
                env.update(DATASET=name, DATA_ROOT=root.as_posix(), BASE_MODEL="model", ARTIFACT_MODEL_NAME="model",
                           RETRIEVAL_MODEL_PATH=model.as_posix(), RANKER_MODEL_PATH=model.as_posix(),
                           DEEPSPEED_BIN=stub.as_posix(), PIPELINE_TEST_ARGV=output.as_posix(), GPUS="2", NUM_NODES="1")
                command = [shutil.which("bash"), script.as_posix()]
                result = subprocess.run(command, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                argv = output.read_bytes().decode().strip("\0").split("\0")
                for option, expected in (("--task-type", task), ("--query-type", query), ("--num_gpus", "2"),
                                         ("--tbpr-type", "origin_type"), ("--stage2-dense-index", f"dense_output/{relative}")):
                    self.assertEqual(argv[argv.index(option) + 1], expected)
                direction = {"tbpr": "text_to_image", "t2it": "text_to_imagetext", "it2t": "imagetext_to_text"}[task]
                expected_output = f"./three_stage_results/model/{name}/{direction}/test/sparse_concat_stage1"
                if task == "tbpr":
                    expected_output += "/origin_type"
                self.assertEqual(argv[argv.index("--output-dir") + 1], expected_output)
                conflict = subprocess.run(command, env={**env, "TASK_TYPE": "ir"}, capture_output=True, text=True)
                self.assertEqual(conflict.returncode, 2)
                self.assertIn("requires TASK_TYPE", conflict.stderr)
                annotation = PERSON_FILES.get(name, "query-00000-of-00001.parquet" if task == "t2it" else "query.parquet")
                (root / "data" / directory / annotation).unlink()
                missing = subprocess.run(command, env=env, capture_output=True, text=True)
                self.assertEqual(missing.returncode, 2)
                self.assertIn(annotation, missing.stderr)


if __name__ == "__main__":
    unittest.main()
