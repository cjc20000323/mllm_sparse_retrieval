from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "src" / "three_stage_retrieval_pipeline.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "three_stage_retrieval_pipeline", MODULE_PATH
)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Cannot load {MODULE_PATH}")
pipeline: Any = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = pipeline
MODULE_SPEC.loader.exec_module(pipeline)


class FakeProgress:
    def __init__(self, iterable: Any = None, **kwargs: Any) -> None:
        self.iterable = () if iterable is None else iterable
        self.kwargs = kwargs
        self.updated = 0
        self.closed = False

    def __iter__(self) -> Any:
        return iter(self.iterable)

    def __enter__(self) -> "FakeProgress":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def update(self, amount: int = 1) -> None:
        self.updated += amount

    def close(self) -> None:
        self.closed = True


class ProgressRecorder:
    def __init__(self) -> None:
        self.instances: list[FakeProgress] = []

    def __call__(self, iterable: Any = None, **kwargs: Any) -> FakeProgress:
        progress = FakeProgress(iterable, **kwargs)
        self.instances.append(progress)
        return progress


class FakeExcelFrame:
    """Capture the worksheet table without requiring optional Excel packages."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def to_excel(self, path: Path, **kwargs: Any) -> None:
        Path(path).write_text(json.dumps(self.rows), encoding="utf-8")


class FakeClock:
    def __init__(self) -> None:
        self.seconds = 0.0

    def __call__(self) -> float:
        return self.seconds

    def advance(self, seconds: float) -> None:
        self.seconds += seconds


def fake_excel_dependencies() -> tuple[Any, str]:
    return SimpleNamespace(DataFrame=FakeExcelFrame), "test"


class FakeRanker:
    def __init__(self) -> None:
        self.caption_call: tuple[tuple[Any, ...], dict[str, Any]] | None = None
        self.pointwise_call: tuple[tuple[Any, ...], dict[str, Any]] | None = None

    def caption_generation_rerank(self, *args: Any, **kwargs: Any) -> dict[str, dict[str, float]]:
        self.caption_call = (args, kwargs)
        return {"query": {"caption": 1.0}}

    def rerank(self, *args: Any, **kwargs: Any) -> dict[str, dict[str, float]]:
        self.pointwise_call = (args, kwargs)
        return {"query": {"pointwise": 1.0}}


class FakeClipProcessor:
    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(model_max_length=77)
        self.call_kwargs: dict[str, Any] | None = None

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        import torch

        self.call_kwargs = kwargs
        if "images" in kwargs:
            return {
                "pixel_values": torch.ones((len(kwargs["images"]), 3, 2, 2))
            }
        batch_size = len(kwargs["text"])
        sequence_length = kwargs["max_length"]
        return {
            "input_ids": torch.ones((batch_size, sequence_length), dtype=torch.long),
            "attention_mask": torch.ones(
                (batch_size, sequence_length), dtype=torch.long
            ),
        }


class FakeClipModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(max_position_embeddings=77)
        )
        self.input_shape: tuple[int, ...] | None = None
        self.attention_shape: tuple[int, ...] | None = None

    def get_text_features(self, input_ids: Any, attention_mask: Any) -> Any:
        import torch

        self.input_shape = tuple(input_ids.shape)
        self.attention_shape = tuple(attention_mask.shape)
        return torch.ones((input_ids.shape[0], 4))

    def get_image_features(self, pixel_values: Any) -> Any:
        import torch

        return torch.ones((pixel_values.shape[0], 4))


class ThreeStageModifyTypeTests(unittest.TestCase):
    def test_parser_supports_both_modify_type_spellings(self) -> None:
        parser = pipeline._parser()

        default_args = parser.parse_args(["--dataset-name", "flickr"])
        underscore_args = parser.parse_args(
            ["--dataset-name", "flickr", "--modify_type", "modify_single"]
        )
        hyphen_args = parser.parse_args(
            ["--dataset-name", "flickr", "--modify-type", "modify_single"]
        )

        self.assertEqual(default_args.modify_type, "no")
        self.assertEqual(underscore_args.modify_type, "modify_single")
        self.assertEqual(hyphen_args.modify_type, "modify_single")

    def test_caption_generation_dispatches_modify_type(self) -> None:
        ranker = FakeRanker()
        args = SimpleNamespace(
            rerank_template="caption_generation",
            modify_type="modify_single",
            rerank_type="pointwise",
            rerank_num=20,
            rerank_batch_size=2,
        )

        result = pipeline._dispatch_stage3_rerank(
            ranker,
            {"query": {"doc": 0.5}},
            args,
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
        )

        self.assertEqual(result, {"query": {"caption": 1.0}})
        self.assertIsNotNone(ranker.caption_call)
        self.assertIsNone(ranker.pointwise_call)
        call_args, call_kwargs = ranker.caption_call
        self.assertEqual(call_args[6].modify_type, "modify_single")
        self.assertEqual(call_kwargs["rerank_batch_size"], 2)
        self.assertEqual(call_kwargs["rerank_prompt_type"], "caption_generation")

    def test_non_caption_template_uses_pointwise_rerank(self) -> None:
        ranker = FakeRanker()
        args = SimpleNamespace(
            rerank_template="relevant",
            modify_type="no",
            rerank_type="pointwise",
            rerank_num=20,
            rerank_batch_size=1,
        )

        result = pipeline._dispatch_stage3_rerank(
            ranker,
            {"query": {"doc": 0.5}},
            args,
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
        )

        self.assertEqual(result, {"query": {"pointwise": 1.0}})
        self.assertIsNone(ranker.caption_call)
        self.assertIsNotNone(ranker.pointwise_call)

    def test_modify_single_rejects_non_caption_template(self) -> None:
        parser = pipeline._parser()
        with tempfile.TemporaryDirectory() as temp_dir:
            args = parser.parse_args(
                [
                    "--dataset-name",
                    "flickr",
                    "--stage1-method",
                    "clip",
                    "--stage2-run",
                    "stage2.json",
                    "--ranker-run",
                    "stage3.json",
                    "--rerank-template",
                    "relevant",
                    "--modify_type",
                    "modify_single",
                    "--output-dir",
                    str(Path(temp_dir) / "new-output"),
                ]
            )

            with self.assertRaisesRegex(ValueError, "caption_generation"):
                pipeline._validate_args(args)


class ThreeStageDistributedHelpersTests(unittest.TestCase):
    def test_distributed_initialization_uses_local_cuda_rank(self) -> None:
        import torch
        import torch.distributed as dist

        environment = {
            "WORLD_SIZE": "4",
            "RANK": "2",
            "LOCAL_RANK": "1",
        }
        with (
            patch.dict(pipeline.os.environ, environment, clear=False),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "set_device") as set_device,
            patch.object(dist, "is_initialized", return_value=False),
            patch.object(dist, "init_process_group") as init_process_group,
            patch.object(dist, "get_rank", return_value=2),
            patch.object(dist, "get_world_size", return_value=4),
        ):
            context = pipeline._initialise_distributed()

        self.assertEqual(
            context,
            pipeline.DistributedContext(rank=2, local_rank=1, world_size=4),
        )
        set_device.assert_called_once_with(1)
        init_process_group.assert_called_once()
        self.assertEqual(init_process_group.call_args.args, ("nccl",))

    def test_partition_and_merge_cover_each_query_once(self) -> None:
        run = {
            "q0": {"d0": 1.0},
            "q1": {"d1": 0.9},
            "q2": {"d2": 0.8},
            "q3": {"d3": 0.7},
            "q4": {"d4": 0.6},
        }

        shards = pipeline._partition_run(run, 3)

        self.assertEqual(list(shards[0]), ["q0", "q3"])
        self.assertEqual(list(shards[1]), ["q1", "q4"])
        self.assertEqual(list(shards[2]), ["q2"])
        self.assertEqual(pipeline._merge_run_shards(shards), run)

    def test_merge_rejects_duplicate_query_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "重复 query ID: q0"):
            pipeline._merge_run_shards(
                [
                    {"q0": {"d0": 1.0}},
                    {"q0": {"d1": 0.5}},
                ]
            )

    def test_query_record_selection_preserves_shard_order(self) -> None:
        records = [
            {"id": "q0", "text": "zero"},
            {"id": "q1", "text": "one"},
            {"id": "q2", "text": "two"},
        ]

        selected = pipeline._select_query_records(records, ["q2", "q0"])

        self.assertEqual([record["id"] for record in selected], ["q2", "q0"])
        with self.assertRaisesRegex(KeyError, "missing"):
            pipeline._select_query_records(records, ["missing"])

    def test_auto_device_map_targets_local_rank(self) -> None:
        context = pipeline.DistributedContext(rank=2, local_rank=1, world_size=4)

        self.assertEqual(
            pipeline._device_map_for_process("auto", context),
            {"": 1},
        )
        self.assertEqual(
            pipeline._device_map_for_process("none", context),
            "none",
        )
        self.assertEqual(
            pipeline._device_map_for_process(
                "auto",
                pipeline.DistributedContext(),
            ),
            "auto",
        )

    def test_gather_merges_rank_shards_on_main_process(self) -> None:
        import torch.distributed as dist

        context = pipeline.DistributedContext(rank=0, local_rank=0, world_size=2)

        def fake_gather(local_run: Any, gathered: Any, dst: int) -> None:
            self.assertEqual(local_run, {"q0": {"d0": 1.0}})
            self.assertEqual(dst, 0)
            gathered[:] = [local_run, {"q1": {"d1": 0.5}}]

        with patch.object(dist, "gather_object", side_effect=fake_gather):
            result = pipeline._gather_run({"q0": {"d0": 1.0}}, context)

        self.assertEqual(
            result,
            {"q0": {"d0": 1.0}, "q1": {"d1": 0.5}},
        )

    def test_parser_accepts_both_local_rank_spellings(self) -> None:
        parser = pipeline._parser()

        underscore = parser.parse_args(
            ["--dataset-name", "flickr", "--local_rank", "2"]
        )
        hyphen = parser.parse_args(
            ["--dataset-name", "flickr", "--local-rank", "3"]
        )

        self.assertEqual(underscore.local_rank, 2)
        self.assertEqual(hyphen.local_rank, 3)

    def test_shell_uses_deepspeed_gpu_process_count(self) -> None:
        script = (
            PROJECT_ROOT / "scripts" / "three_stage_retrieval_pipeline.sh"
        ).read_text(encoding="utf-8")

        self.assertIn('GPUS="${GPUS:-4}"', script)
        self.assertIn('DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"', script)
        self.assertIn('--num_gpus "${GPUS}"', script)


class ThreeStageSparseLuceneTests(unittest.TestCase):
    def test_topic_token_frequencies_match_local_pyserini_encoder(self) -> None:
        self.assertEqual(
            pipeline._topic_token_frequencies(" red  blue red "),
            {"red": 2, "blue": 1},
        )

    def test_sparse_search_uses_local_java_bindings_without_openai(self) -> None:
        progress_recorder = ProgressRecorder()

        class FakeIndexReader:
            instances: list[Any] = []

            def __init__(self, index_path: str) -> None:
                self.index_path = index_path
                self.closed = False
                self.instances.append(self)

            def stats(self) -> dict[str, int]:
                return {"documents": 3}

            def terms(self) -> list[Any]:
                return [
                    SimpleNamespace(term="red", df=1),
                    SimpleNamespace(term="blue", df=2),
                    SimpleNamespace(term="common", df=3),
                    SimpleNamespace(term="zero", df=0),
                ]

            def close(self) -> None:
                self.closed = True

        class FakeArrayList(list[Any]):
            def add(self, value: Any) -> None:
                self.append(value)

        class FakeHashMap(dict[str, int]):
            def put(self, key: str, value: int) -> None:
                self[key] = value

        class FakeEntry:
            def __init__(self, key: str, value: list[Any]) -> None:
                self.key = key
                self.value = value

            def getKey(self) -> str:
                return self.key

            def getValue(self) -> list[Any]:
                return self.value

        class FakeJavaResults:
            def entrySet(self) -> Any:
                entries = [
                    FakeEntry("q1", [SimpleNamespace(docid=101, score=2.5)])
                ]
                return SimpleNamespace(toArray=lambda: entries)

        class FakeSimpleImpactSearcher:
            instances: list[Any] = []

            def __init__(self, index_path: str) -> None:
                self.index_path = index_path
                self.closed = False
                self.call: tuple[Any, ...] | None = None
                self.instances.append(self)

            def batch_search(
                self,
                queries: list[Any],
                qids: list[str],
                depth: int,
                threads: int,
            ) -> FakeJavaResults:
                self.call = (
                    [dict(query) for query in queries],
                    list(qids),
                    depth,
                    threads,
                )
                return FakeJavaResults()

            def close(self) -> None:
                self.closed = True

        def fake_autoclass(class_name: str) -> type[FakeSimpleImpactSearcher]:
            self.assertEqual(
                class_name, "io.anserini.search.SimpleImpactSearcher"
            )
            return FakeSimpleImpactSearcher

        pyserini_module = ModuleType("pyserini")
        pyserini_module.__path__ = []  # type: ignore[attr-defined]
        index_module = ModuleType("pyserini.index")
        index_module.__path__ = []  # type: ignore[attr-defined]
        lucene_module = ModuleType("pyserini.index.lucene")
        lucene_module.LuceneIndexReader = FakeIndexReader  # type: ignore[attr-defined]
        pyclass_module = ModuleType("pyserini.pyclass")
        pyclass_module.JArrayList = FakeArrayList  # type: ignore[attr-defined]
        pyclass_module.JHashMap = FakeHashMap  # type: ignore[attr-defined]
        pyclass_module.JInt = int  # type: ignore[attr-defined]
        pyclass_module.autoclass = fake_autoclass  # type: ignore[attr-defined]

        fake_modules = {
            "pyserini": pyserini_module,
            "pyserini.index": index_module,
            "pyserini.index.lucene": lucene_module,
            "pyserini.pyclass": pyclass_module,
            "pyserini.encode": None,
            "openai": None,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            index_dir = Path(temp_dir) / "index"
            index_dir.mkdir()
            with (
                patch.dict(sys.modules, fake_modules),
                patch.object(
                    pipeline,
                    "_progress",
                    side_effect=progress_recorder,
                ),
            ):
                result = pipeline.retrieve_sparse_topics_from_lucene(
                    Path(temp_dir),
                    {
                        "q1": "red red blue common zero",
                        "empty": "",
                    },
                    ["q1", "empty", "missing"],
                    depth=20,
                    threads=0,
                    progress_prefix="[stage2]",
                )

        self.assertEqual(result, {"q1": {"101": 2.5}, "empty": {}})
        self.assertEqual(len(FakeIndexReader.instances), 1)
        self.assertTrue(FakeIndexReader.instances[0].closed)
        self.assertEqual(len(FakeSimpleImpactSearcher.instances), 1)
        searcher = FakeSimpleImpactSearcher.instances[0]
        self.assertEqual(searcher.index_path, str(index_dir))
        self.assertEqual(
            searcher.call,
            ([{"red": 2, "blue": 1}], ["q1"], 20, 1),
        )
        self.assertTrue(searcher.closed)
        progress_descriptions = [
            instance.kwargs["desc"] for instance in progress_recorder.instances
        ]
        self.assertEqual(
            progress_descriptions,
            [
                "[stage2] scan vocabulary",
                "[stage2] prepare sparse queries",
                "[stage2] Lucene search",
                "[stage2] collect sparse hits",
            ],
        )
        self.assertEqual(progress_recorder.instances[2].updated, 1)

    def test_hybrid_fusion_reports_query_progress_without_changing_scores(self) -> None:
        progress_recorder = ProgressRecorder()
        with patch.object(
            pipeline,
            "_progress",
            side_effect=progress_recorder,
        ):
            result = pipeline.fuse_restricted_runs(
                {"q1": {"d1": 2.0, "d2": 1.0}},
                {"q1": {"d1": 10.0, "d2": 0.0}},
                {"q1": {"d1": 0.0, "d2": 4.0}},
                alpha=0.5,
                candidate_depth=2,
            )

        self.assertEqual(result, {"q1": {"d1": 0.5, "d2": 0.5}})
        self.assertEqual(len(progress_recorder.instances), 1)
        self.assertEqual(
            progress_recorder.instances[0].kwargs,
            {
                "desc": "[stage2] fuse runs",
                "unit": "query",
                "total": 1,
            },
        )


class ThreeStageErrorReportingTests(unittest.TestCase):
    def test_main_updates_pipeline_progress_after_each_stage(self) -> None:
        progress_recorder = ProgressRecorder()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            args = pipeline._parser().parse_args(
                [
                    "--dataset-name",
                    "flickr",
                    "--stage1-method",
                    "clip",
                    "--stage1-run",
                    "stage1.json",
                    "--stage2-run",
                    "stage2.json",
                    "--ranker-run",
                    "stage3.json",
                    "--output-dir",
                    str(output_dir),
                ]
            )
            parser = SimpleNamespace(
                parse_args=lambda: args,
                error=lambda message: self.fail(message),
            )
            stage1_run = {"q1": {"d1": 1.0}}
            stage2_run = {"q1": {"d1": 0.5}}
            stage3_run = {"q1": {"d1": 0.25}}
            with (
                patch.object(sys, "path", [str(PROJECT_ROOT / "src"), *sys.path]),
                patch.object(pipeline, "_parser", return_value=parser),
                patch.object(pipeline, "run_stage1", return_value=stage1_run),
                patch.object(
                    pipeline,
                    "run_stage2",
                    return_value=({}, {}, stage2_run),
                ),
                patch.object(pipeline, "run_stage3", return_value=stage3_run),
                patch.object(
                    pipeline, "build_dataset", return_value=SimpleNamespace(
                        text_id_list=["q1"],
                        get_target=lambda query_id, query_type: "d1",
                    ),
                ),
                patch.object(
                    pipeline.RecallReporter, "_output_dependencies",
                    side_effect=fake_excel_dependencies,
                ),
                patch.object(
                    pipeline,
                    "_progress",
                    side_effect=progress_recorder,
                ),
            ):
                pipeline.main()

        self.assertEqual(len(progress_recorder.instances), 1)
        pipeline_progress = progress_recorder.instances[0]
        self.assertEqual(pipeline_progress.kwargs["desc"], "[pipeline] stages")
        self.assertEqual(pipeline_progress.kwargs["total"], 3)
        self.assertEqual(pipeline_progress.updated, 3)
        self.assertTrue(pipeline_progress.closed)

    def test_main_keeps_validation_errors_as_cli_errors(self) -> None:
        errors: list[str] = []

        def parser_error(message: str) -> None:
            errors.append(message)
            raise SystemExit(2)

        parser = SimpleNamespace(
            parse_args=lambda: SimpleNamespace(),
            error=parser_error,
        )
        with (
            patch.object(pipeline, "_parser", return_value=parser),
            patch.object(
                pipeline,
                "_validate_args",
                side_effect=ValueError("invalid local configuration"),
            ),
        ):
            with self.assertRaisesRegex(SystemExit, "2"):
                pipeline.main()

        self.assertEqual(errors, ["invalid local configuration"])

    def test_main_preserves_unexpected_runtime_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(
                output_dir=str(Path(temp_dir) / "output"),
                stage1_method="clip",
            )
            parser = SimpleNamespace(
                parse_args=lambda: args,
                error=lambda message: self.fail(
                    f"runtime error was incorrectly converted to argparse: {message}"
                ),
            )
            with (
                patch.object(pipeline, "_parser", return_value=parser),
                patch.object(pipeline, "_validate_args"),
                patch.object(
                    pipeline,
                    "run_stage1",
                    side_effect=RuntimeError("local sparse failure"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "local sparse failure"):
                    pipeline.main()


class ThreeStageRecallTests(unittest.TestCase):
    def setUp(self) -> None:
        path_patch = patch.object(sys, "path", [str(PROJECT_ROOT / "src"), *sys.path])
        path_patch.start()
        self.addCleanup(path_patch.stop)

    @staticmethod
    def args(**overrides: Any) -> Any:
        args = pipeline._parser().parse_args([
            "--dataset-name", "flickr", "--query-type", "image",
            "--stage1-method", "clip", "--stage1-run", "stage1.json",
            "--stage2-run", "stage2.json",
            "--ranker-model-name-or-path", "fake-ranker",
        ])
        for name, value in overrides.items():
            setattr(args, name, value)
        return args

    @staticmethod
    def dataset() -> Any:
        image_to_text = {"1": ["11", "12"], "2": ["21", "22"]}
        text_to_image = {"11": "1", "12": "1", "21": "2", "22": "2"}
        return SimpleNamespace(
            data_name="flickr",
            img_id_list=list(image_to_text),
            text_id_list=list(text_to_image),
            get_target=lambda qid, query_type: (
                text_to_image[qid] if query_type == "text" else image_to_text[qid]
            ),
        )

    def evaluation(self, args: Any, dataset: Any, query_ids: Any) -> Any:
        with patch.object(pipeline, "build_dataset", return_value=dataset) as builder:
            result = pipeline._build_recall_evaluation(args, query_ids)
        builder.assert_called_once_with(args)
        return result

    def test_shared_cutoffs_and_hits_match_existing_recall_metrics(self) -> None:
        from metrices import RecallMetrics, get_recall_k_settings
        import torch
        import torch.distributed as dist

        args = self.args()
        dataset = self.dataset()
        run = {"1": {"21": 2.0, "12": 1.0}, "2": {}}
        evaluation = self.evaluation(args, dataset, run)
        with (
            patch.object(dist, "get_world_size", return_value=1),
            patch.object(torch.Tensor, "cuda", lambda tensor: tensor),
        ):
            legacy = RecallMetrics(dataset, {}, {}, run, [], list(run), args)
            legacy.sort_and_count()
        result = pipeline.evaluate_recall(run, evaluation, stage="stage1", variant="clip")[0]

        self.assertEqual(evaluation.k_values, legacy.recall_k_setting_list)
        self.assertEqual(evaluation.k_values, [1, 5, 10, 100, 200, 300, 400])
        self.assertEqual(result["hits"], {str(k): count for k, count in legacy.fusion_counts.items()})
        self.assertEqual(get_recall_k_settings("fashion-iq"), [1, 5, 10, 20, 30, 50, 100, 200, 300, 400])

    def test_image_to_text_accepts_any_correct_caption_and_sorts_scores(self) -> None:
        run = {"1": {"11": -2.0, "21": -3.0, "12": -1.0}, "2": {"21": -1.0}}
        evaluation = self.evaluation(self.args(), self.dataset(), run)
        result = pipeline.evaluate_recall(run, evaluation, stage="stage3", variant="ranker")[0]
        self.assertEqual(result["recall"]["1"], 1.0)
        self.assertEqual(result["hits"]["1"], 2)
        self.assertEqual((result["min_candidates"], result["max_candidates"]), (1, 3))

    def test_text_to_image_counts_empty_and_missing_results_as_misses(self) -> None:
        evaluation = self.evaluation(
            self.args(query_type="text"), self.dataset(), ["11", "12", "21"],
        )
        result = pipeline.evaluate_recall(
            {"11": {"1": 1.0}, "12": {}}, evaluation, stage="stage2", variant="sparse",
        )[0]
        self.assertEqual(result["query_count"], 3)
        self.assertEqual(result["recall"]["1"], 1 / 3)
        self.assertEqual(result["recall"]["400"], 1 / 3)
        self.assertEqual(result["min_candidates"], 0)

    def test_person_retrieval_matches_person_identity_not_image_id(self) -> None:
        image_to_person = {"image-a": 7, "image-b": 7, "image-c": 8}
        dataset = SimpleNamespace(
            text_id_list=["query"], img_id_list=list(image_to_person),
            get_target_from_text=lambda qid: 7,
            get_target_from_image=image_to_person.__getitem__,
        )
        evaluation = self.evaluation(
            self.args(dataset_name="CUHK-PEDES", task_type="tbpr", query_type="text"),
            dataset, ["query"],
        )
        result = pipeline.evaluate_recall(
            {"query": {"image-b": 1.0}}, evaluation, stage="stage2", variant="hybrid",
        )[0]
        self.assertEqual(result["recall"]["1"], 1.0)
        with self.assertRaisesRegex(ValueError, "unknown.*身份标签"):
            pipeline.evaluate_recall(
                {"query": {"unknown": 1.0}}, evaluation, stage="stage3", variant="ranker",
            )

    def test_composed_and_multimodal_tasks_use_native_query_keys(self) -> None:
        for task_type, targets in (("t2it", [9, 10]), ("it2t", 10)):
            with self.subTest(task_type=task_type):
                calls = []

                def get_target(query_id: Any) -> Any:
                    calls.append(query_id)
                    return targets

                evaluation = self.evaluation(
                    self.args(dataset_name="webqa" if task_type == "t2it" else "llava", task_type=task_type),
                    SimpleNamespace(query_id_list=[1], get_target=get_target), ["1"],
                )
                result = pipeline.evaluate_recall(
                    {"1": {"10": 2.0}}, evaluation, stage="stage1", variant="clip",
                )[0]
                self.assertEqual(calls, [1])
                self.assertEqual(result["recall"]["1"], 1.0)

    def test_fashion_iq_reports_categories_with_actual_query_denominators(self) -> None:
        targets = {"q-dress": "dress-target", "q-shirt": "shirt-target", "q-toptee": "toptee-target"}
        dataset = SimpleNamespace(
            composed_id_list=list(targets), get_target=targets.__getitem__,
            get_dress_type=lambda target: target.split("-")[0],
        )
        evaluation = self.evaluation(
            self.args(dataset_name="fashion-iq", task_type="cir", query_type="composed"),
            dataset, targets,
        )
        rows = pipeline.evaluate_recall(
            {"q-dress": {"dress-target": 1.0}, "q-shirt": {}, "q-toptee": {}},
            evaluation, stage="stage3", variant="ranker",
        )
        grouped = {row["category"]: row for row in rows}
        self.assertEqual(set(grouped), {"all", "dress", "shirt", "toptee"})
        self.assertEqual(grouped["dress"]["query_count"], 1)
        self.assertEqual(grouped["dress"]["recall"]["20"], 1.0)
        self.assertEqual(grouped["shirt"]["recall"]["50"], 0.0)
        self.assertEqual(grouped["all"]["recall"]["1"], 1 / 3)

    def test_empty_query_set_and_missing_ground_truth_are_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "没有查询"):
            self.evaluation(self.args(), self.dataset(), [])
        with self.assertRaisesRegex(ValueError, "unknown.*真实标签"):
            self.evaluation(self.args(), self.dataset(), ["unknown"])
        for target in (None, [], ""):
            with self.subTest(target=target):
                dataset = SimpleNamespace(img_id_list=["1"], get_target=lambda *args: target)
                with self.assertRaisesRegex(ValueError, "真实标签缺失"):
                    self.evaluation(self.args(), dataset, ["1"])

    def test_query_subset_is_deduplicated_and_unexpected_queries_are_rejected(self) -> None:
        evaluation = self.evaluation(self.args(), self.dataset(), ["1", "1"])
        self.assertEqual(len(evaluation.targets), 1)
        self.assertEqual(evaluation.dataset_query_count, 2)
        with self.assertRaisesRegex(ValueError, "未纳入评估.*2"):
            pipeline.evaluate_recall({"2": {}}, evaluation, stage="stage1", variant="clip")

    def test_reporter_incrementally_saves_matching_json_and_excel_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with (
                patch.object(pipeline, "build_dataset", return_value=self.dataset()) as builder,
                patch.object(pipeline.RecallReporter, "_output_dependencies", side_effect=fake_excel_dependencies),
                patch.object(pipeline.tqdm, "write") as write,
            ):
                reporter = pipeline.RecallReporter(self.args(), ["1"], output_dir)
                for stage, variant in (("stage1", "clip"), ("stage2", "hybrid"), ("stage3", "ranker")):
                    reporter.record(stage, {variant: {"1": {"11": 1.0}}})
                    payload = json.loads((output_dir / "recall_metrics.json").read_text(encoding="utf-8"))
                    self.assertEqual(payload["results"][-1]["stage"], stage)
                builder.assert_called_once()
            table = json.loads((output_dir / "recall_metrics.xlsx").read_text(encoding="utf-8"))
            self.assertEqual(payload["query_count"], 1)
            self.assertEqual(payload["dataset_query_count"], 2)
            self.assertEqual(payload["recall_scale"], "0-1")
            self.assertEqual(len(table), 3)
            for row, json_row in zip(table, payload["results"]):
                for k in payload["recall_k_values"]:
                    self.assertEqual(row[f"R@{k}"], json_row["recall"][str(k)])
            self.assertIn("R@1=100.00%", "\n".join(call.args[0] for call in write.call_args_list))
            self.assertFalse((output_dir / "recall_metrics.tmp.json").exists())
            self.assertFalse((output_dir / "recall_metrics.tmp.xlsx").exists())

    def test_missing_excel_dependency_has_an_actionable_error(self) -> None:
        with patch.dict(sys.modules, {"openpyxl": None}):
            with self.assertRaisesRegex(RuntimeError, "openpyxl"):
                pipeline.RecallReporter._output_dependencies()

    @unittest.skipUnless(importlib.util.find_spec("openpyxl"), "optional Excel dependency openpyxl is unavailable")
    def test_real_excel_workbook_matches_json_metrics(self) -> None:
        import openpyxl

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with patch.object(pipeline, "build_dataset", return_value=self.dataset()):
                reporter = pipeline.RecallReporter(self.args(), ["1", "2"], output_dir)
            reporter.record("stage1", {"clip": {"1": {"11": 1.0}, "2": {}}})
            reporter.record("stage2", {"hybrid": {"1": {"12": 1.0}, "2": {"21": 1.0}}})
            workbook = openpyxl.load_workbook(output_dir / "recall_metrics.xlsx", read_only=True)
            try:
                values = list(workbook.active.values)
                rows = [dict(zip(values[0], row)) for row in values[1:]]
            finally:
                workbook.close()
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["R@1"], 0.5)
            self.assertEqual(rows[1]["R@1"], 1.0)
            payload = json.loads((output_dir / "recall_metrics.json").read_text(encoding="utf-8"))
            for row, json_row in zip(rows, payload["results"]):
                for k in payload["recall_k_values"]:
                    self.assertEqual(row[f"R@{k}"], json_row["recall"][str(k)])

    def test_main_reports_each_stage_before_the_next_and_preserves_failed_runs(self) -> None:
        for fail_stage in (None, "stage2", "stage3"):
            with self.subTest(fail_stage=fail_stage), tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir) / "output"
                args = self.args(output_dir=str(output_dir))
                run = {"1": {"11": 1.0}, "2": {"21": 1.0}}

                def recorded_stages() -> list[str]:
                    payload = json.loads((output_dir / "recall_metrics.json").read_text(encoding="utf-8"))
                    return [row["stage"] for row in payload["results"]]

                def stage2(*stage_args: Any, **kwargs: Any) -> Any:
                    self.assertEqual(recorded_stages(), ["stage1"])
                    if fail_stage == "stage2":
                        raise RuntimeError("stage2 failure")
                    return run, run, run

                def stage3(*stage_args: Any, **kwargs: Any) -> Any:
                    self.assertEqual(recorded_stages(), ["stage1", "stage2", "stage2", "stage2"])
                    if fail_stage == "stage3":
                        raise RuntimeError("stage3 failure")
                    return run

                with (
                    patch.object(pipeline, "_parser", return_value=SimpleNamespace(parse_args=lambda: args)),
                    patch.object(pipeline, "build_dataset", return_value=self.dataset()),
                    patch.object(pipeline, "run_stage1", return_value=run),
                    patch.object(pipeline, "run_stage2", side_effect=stage2),
                    patch.object(pipeline, "run_stage3", side_effect=stage3),
                    patch.object(pipeline, "_progress", side_effect=ProgressRecorder()),
                    patch.object(pipeline.RecallReporter, "_output_dependencies", side_effect=fake_excel_dependencies),
                ):
                    if fail_stage:
                        with self.assertRaisesRegex(RuntimeError, f"{fail_stage} failure"):
                            pipeline.main()
                    else:
                        pipeline.main()
                expected = ["stage1"]
                if fail_stage != "stage2":
                    expected.extend(["stage2"] * 3)
                if fail_stage is None:
                    expected.append("stage3")
                    metadata = json.loads((output_dir / "pipeline_metadata.json").read_text(encoding="utf-8"))
                    self.assertIn("recall_metrics.xlsx", metadata["files"])
                    self.assertIn("recall_metrics.json", metadata["files"])
                    self.assertEqual(metadata["recall"]["query_count"], 2)
                self.assertEqual(recorded_stages(), expected)
                self.assertTrue((output_dir / "recall_metrics.xlsx").exists())
                timings = json.loads((output_dir / "timing_metrics.json").read_text(encoding="utf-8"))
                self.assertEqual(timings["status"], "failed" if fail_stage else "completed")
                if fail_stage:
                    self.assertIsNone(timings["totals"]["compute_seconds"])
                    self.assertEqual(
                        next(row for row in timings["stages"] if row["stage"] == fail_stage)["status"], "failed",
                    )
                self.assertEqual(timings["stages"][0]["status"], "completed")

    def test_precomputed_runs_with_real_flickr_and_coco_csv_labels(self) -> None:
        import dataset as dataset_module

        path_names = (
            "dataset_path_prefix", "coco_file_path", "flickr_file_path",
            "fashion_iq_file_path", "cuhk_pedes_file_path", "icfg_pedes_flie_path",
            "rstpreid_file_path", "webqa_file_path", "remuq_file_path",
            "llava_file_path", "edis_file_path",
        )
        saved_paths = {name: getattr(dataset_module, name) for name in path_names}
        for dataset_name in ("flickr", "coco"):
            for query_type in ("image", "text"):
                with (
                    self.subTest(dataset_name=dataset_name, query_type=query_type),
                    tempfile.TemporaryDirectory() as temp_dir,
                    patch.multiple(dataset_module, **saved_paths),
                ):
                    root = Path(temp_dir)
                    dataset_dir = root / "data" / dataset_name
                    dataset_dir.mkdir(parents=True)
                    if dataset_name == "flickr":
                        csv = "imgid,filename,caption,sentid\n1,1.jpg,first,11\n1,1.jpg,second,12\n2,2.jpg,third,21\n2,2.jpg,fourth,22\n"
                    else:
                        csv = "imgid,filepath,filename,caption,sentid\n1,test,1.jpg,first,11\n1,test,1.jpg,second,12\n2,test,2.jpg,third,21\n2,test,2.jpg,fourth,22\n"
                    (dataset_dir / f"{dataset_name}_test.csv").write_text(csv, encoding="utf-8")
                    q1, q2 = ("1", "2") if query_type == "image" else ("11", "21")
                    d1, d2 = ("11", "21") if query_type == "image" else ("1", "2")
                    runs = (
                        {q1: {d2: 2.0, d1: 1.0}, q2: {d2: 2.0, d1: 1.0}},
                        {q1: {d1: 2.0, d2: 1.0}, q2: {d2: 2.0}},
                        {q1: {d1: -1.0}, q2: {}},
                    )
                    for index, run in enumerate(runs, start=1):
                        pipeline.write_json_run(run, root / f"stage{index}.json")
                    args = self.args(
                        dataset_name=dataset_name, query_type=query_type,
                        data_root=str(root), output_dir=str(root / "output"),
                        stage1_run=str(root / "stage1.json"),
                        stage2_run=str(root / "stage2.json"),
                        ranker_run=str(root / "stage3.json"),
                    )
                    with (
                        patch.object(pipeline, "_parser", return_value=SimpleNamespace(parse_args=lambda: args)),
                        patch.object(pipeline, "_progress", side_effect=ProgressRecorder()),
                        patch.object(pipeline.tqdm, "write"),
                        patch.object(pipeline.RecallReporter, "_output_dependencies", side_effect=fake_excel_dependencies),
                    ):
                        pipeline.main()
                    payload = json.loads((root / "output" / "recall_metrics.json").read_text(encoding="utf-8"))
                    self.assertEqual([row["recall"]["1"] for row in payload["results"]], [0.5, 1.0, 0.5])
                    self.assertEqual(payload["results"][0]["recall"]["5"], 1.0)
                    self.assertEqual(payload["query_count"], 2)
                    self.assertEqual(payload["dataset_query_count"], 2 if query_type == "image" else 4)
                    timings = json.loads((root / "output" / "timing_metrics.json").read_text(encoding="utf-8"))
                    self.assertEqual(timings["status"], "completed")
                    self.assertEqual(timings["totals"]["compute_seconds"], 0.0)
                    self.assertTrue(all(row["source"] == "precomputed" for row in timings["stages"]))

    def test_distributed_main_evaluates_merged_queries_once(self) -> None:
        full_run = {"1": {"11": 1.0}, "2": {"21": 1.0}}
        local_run = {"1": {"11": 1.0}}
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            args = self.args(output_dir=str(output_dir))
            with (
                patch.object(pipeline, "_parser", return_value=SimpleNamespace(parse_args=lambda: args)),
                patch.object(pipeline, "_initialise_distributed", return_value=pipeline.DistributedContext(world_size=2)),
                patch.object(pipeline, "_shutdown_distributed"),
                patch.object(pipeline, "_scatter_run", return_value=local_run),
                patch.object(pipeline, "_gather_run", return_value=full_run),
                patch.object(
                    pipeline, "_collect_stage_timing",
                    side_effect=lambda timer, context: pipeline._merge_stage_timings([timer.snapshot(context.rank)]),
                ),
                patch.object(pipeline, "build_dataset", return_value=self.dataset()),
                patch.object(pipeline, "run_stage1", return_value=full_run),
                patch.object(pipeline, "run_stage2", return_value=(local_run, local_run, local_run)),
                patch.object(pipeline, "run_stage3", return_value=local_run),
                patch.object(pipeline, "_progress", side_effect=ProgressRecorder()),
                patch.object(pipeline.RecallReporter, "_output_dependencies", side_effect=fake_excel_dependencies),
                patch.object(pipeline, "evaluate_recall", wraps=pipeline.evaluate_recall) as evaluate,
            ):
                pipeline.main()
            self.assertEqual(evaluate.call_count, 5)
            for call in evaluate.call_args_list:
                self.assertEqual(call.args[0], full_run)
            payload = json.loads((output_dir / "recall_metrics.json").read_text(encoding="utf-8"))
            self.assertTrue(all(row["query_count"] == 2 for row in payload["results"]))
            self.assertTrue(all(row["hits"]["1"] == 2 for row in payload["results"]))

    def test_distributed_worker_never_creates_reporter_or_writes_results(self) -> None:
        local_run = {"1": {"11": 1.0}}
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            args = self.args(output_dir=str(output_dir))
            with (
                patch.object(pipeline, "_parser", return_value=SimpleNamespace(parse_args=lambda: args)),
                patch.object(pipeline, "_initialise_distributed", return_value=pipeline.DistributedContext(rank=1, world_size=2)),
                patch.object(pipeline, "_shutdown_distributed"),
                patch.object(pipeline, "_scatter_run", return_value=local_run),
                patch.object(pipeline, "_gather_run", return_value={}),
                patch.object(pipeline, "_collect_stage_timing", return_value=None),
                patch.object(pipeline, "run_stage2", return_value=(local_run, local_run, local_run)),
                patch.object(pipeline, "run_stage3", return_value=local_run),
                patch.object(pipeline, "RecallReporter") as reporter,
                patch.object(pipeline, "TimingReporter") as timing_reporter,
                patch.object(pipeline, "write_json_run") as write_json,
            ):
                pipeline.main()
            reporter.assert_not_called()
            timing_reporter.assert_not_called()
            write_json.assert_not_called()
            self.assertFalse(output_dir.exists())


class ThreeStageTimingTests(unittest.TestCase):
    def setUp(self) -> None:
        path_patch = patch.object(sys, "path", [str(PROJECT_ROOT / "src"), *sys.path])
        path_patch.start()
        self.addCleanup(path_patch.stop)

    def test_cpu_spans_exclude_untimed_work_and_accumulate(self) -> None:
        import torch
        clock = FakeClock()
        timer = pipeline.StageTimer()
        with (
            patch.object(pipeline.time, "perf_counter", clock),
            patch.object(torch.cuda, "synchronize") as synchronize,
            patch.object(torch.cuda, "current_device") as current_device,
        ):
            clock.advance(100)  # model/index loading
            with timer.measure("query_encoding", device="cpu"):
                clock.advance(2)
            clock.advance(50)  # evaluation/output
            with timer.measure("query_encoding", device="cpu"):
                clock.advance(3)
        self.assertEqual(timer.snapshot(0)["compute_seconds"], 5)
        self.assertEqual(timer.components, {"query_encoding": 5})
        synchronize.assert_not_called()
        current_device.assert_not_called()
        self.assertEqual(pipeline._format_duration(3661.125), "01:01:01.125")
        self.assertEqual(pipeline._format_duration(3599.9999), "01:00:00.000")
        self.assertEqual(pipeline._format_duration(None), "N/A")

    def test_gpu_spans_synchronize_only_model_devices_before_and_after(self) -> None:
        import torch
        clock = FakeClock()
        model = SimpleNamespace(
            hf_device_map={"a": 1, "b": "cuda:3", "c": "cpu", "d": "disk"},
            parameters=lambda: iter([SimpleNamespace(device=torch.device("cuda:3"))]),
        )
        timer = pipeline.StageTimer()
        events = []
        with (
            patch.object(pipeline.time, "perf_counter", clock),
            patch.object(torch.cuda, "synchronize", side_effect=lambda device: events.append(device)),
        ):
            with timer.measure("reranking", model=model):
                self.assertEqual(events, ["cuda:1", "cuda:3"])
                clock.advance(4)
        self.assertEqual(events, ["cuda:1", "cuda:3", "cuda:1", "cuda:3"])
        self.assertEqual(timer.components["reranking"], 4)

    def test_failed_span_preserves_exception_and_does_not_claim_completed_time(self) -> None:
        timer = pipeline.StageTimer()
        with self.assertRaisesRegex(RuntimeError, "failed compute"):
            with timer.measure("reranking"):
                raise RuntimeError("failed compute")
        self.assertEqual(timer.components, {})

    def test_dense_timer_excludes_index_loading_and_returns_identical_results(self) -> None:
        import numpy as np
        clock = FakeClock()
        original_argsort = np.argsort

        def load_corpus(path: Path) -> Any:
            clock.advance(30)
            return np.array([[1.0, 0.0], [0.0, 1.0]]), ["a", "b"]

        def argsort(*args: Any, **kwargs: Any) -> Any:
            clock.advance(4)
            return original_argsort(*args, **kwargs)

        timer = pipeline.StageTimer()
        with (
            patch.object(pipeline.time, "perf_counter", clock),
            patch.object(pipeline, "load_dense_corpus", side_effect=load_corpus),
            patch.object(np, "argsort", side_effect=argsort),
            patch.object(pipeline, "_progress", side_effect=ProgressRecorder()),
        ):
            result = pipeline.retrieve_dense_from_corpus(
                np.array([[1.0, 0.0]]), ["q"], Path("unused"), ["q"], depth=2, timing=timer,
            )
        self.assertEqual(result, {"q": {"a": 1.0, "b": 0.0}})
        self.assertEqual(timer.components, {"dense_search": 4})
        self.assertEqual(clock.seconds, 34)

    def test_sparse_timer_excludes_index_initialization_vocabulary_and_cleanup(self) -> None:
        clock = FakeClock()

        class IndexReader:
            def __init__(self, path: str) -> None:
                clock.advance(10)

            def close(self) -> None:
                clock.advance(1)

        class Searcher:
            def __init__(self, path: str) -> None:
                clock.advance(20)

            def batch_search(self, *args: Any) -> Any:
                clock.advance(5)
                return SimpleNamespace(entrySet=lambda: SimpleNamespace(toArray=lambda: []))

            def close(self) -> None:
                clock.advance(3)

        class ArrayList(list):
            def add(self, value: Any) -> None:
                self.append(value)

        class HashMap(dict):
            def put(self, key: Any, value: Any) -> None:
                self[key] = value

        pyserini = ModuleType("pyserini")
        pyserini.__path__ = []
        index = ModuleType("pyserini.index")
        index.__path__ = []
        lucene = ModuleType("pyserini.index.lucene")
        lucene.LuceneIndexReader = IndexReader
        java = ModuleType("pyserini.pyclass")
        java.JArrayList, java.JHashMap, java.JInt = ArrayList, HashMap, int
        java.autoclass = lambda name: Searcher

        def vocabulary(*args: Any, **kwargs: Any) -> Any:
            clock.advance(30)
            return {"red"}

        timer = pipeline.StageTimer()
        with (
            patch.dict(sys.modules, {"pyserini": pyserini, "pyserini.index": index,
                                    "pyserini.index.lucene": lucene, "pyserini.pyclass": java}),
            patch.object(pipeline.time, "perf_counter", clock),
            patch.object(pipeline, "_positive_idf_terms", side_effect=vocabulary),
            patch.object(pipeline, "_progress", side_effect=ProgressRecorder()),
        ):
            result = pipeline.retrieve_sparse_topics_from_lucene(
                Path("unused"), {"q": "red"}, ["q"], 5, 1, timing=timer,
            )
        self.assertEqual(result, {"q": {}})
        self.assertEqual(timer.components, {"sparse_search": 5})
        self.assertEqual(clock.seconds, 69)

    def test_feature_retrieval_excludes_model_dataset_loading_and_cleanup(self) -> None:
        import numpy as np
        import torch
        args = ThreeStageRecallTests.args()
        clock = FakeClock()
        timer = pipeline.StageTimer()
        original_argsort = np.argsort

        def dataset(*args: Any) -> Any:
            clock.advance(10)
            return SimpleNamespace()

        def load_model(*args: Any) -> Any:
            clock.advance(50)
            return SimpleNamespace(), None, torch.device("cpu")

        def encode(*args: Any) -> Any:
            clock.advance(3)
            return torch.tensor([[1.0, 0.0]])

        def argsort(*args: Any, **kwargs: Any) -> Any:
            clock.advance(2)
            return original_argsort(*args, **kwargs)

        with (
            patch.object(pipeline.time, "perf_counter", clock),
            patch.object(pipeline, "build_dataset", side_effect=dataset),
            patch.object(pipeline, "_direct_stage1_items", return_value=([("1", object())], [("11", "caption")])),
            patch.object(pipeline, "_load_feature_model", side_effect=load_model),
            patch.object(pipeline, "_encode_feature_batch", side_effect=encode),
            patch.object(pipeline.gc, "collect", side_effect=lambda: clock.advance(20)),
            patch.object(np, "argsort", side_effect=argsort),
            patch.object(pipeline, "_progress", side_effect=ProgressRecorder()),
        ):
            result = pipeline.retrieve_stage1_feature_model(args, "clip", timing=timer)
        self.assertEqual(result, {"1": {"11": 1.0}})
        self.assertEqual(timer.components, {"query_encoding": 3, "corpus_encoding": 3, "feature_search": 2})
        self.assertEqual(clock.seconds, 88)

    def test_online_encoding_excludes_dataset_preparation_and_cache_hits(self) -> None:
        import numpy as np
        import torch
        args = ThreeStageRecallTests.args()
        clock = FakeClock()
        timer = pipeline.StageTimer()
        runtime = SimpleNamespace(query_cache={}, processor=None, encoder=None,
                                  device=torch.device("cpu"), dtype=torch.float32)

        def dataset(*args: Any) -> Any:
            clock.advance(10)
            return SimpleNamespace()

        def encode(*args: Any) -> Any:
            clock.advance(3)
            return None, np.array([[1.0, 0.0]])

        kwargs = dict(eol_type="disassembleeol_separate", calculate_type="concat",
                      sparse_manual=False, progress_prefix="test", timing=timer)
        with (
            patch.object(pipeline.time, "perf_counter", clock),
            patch.object(pipeline, "build_dataset", side_effect=dataset) as builder,
            patch.object(pipeline, "_query_records", return_value=[{"id": "1", "text": "q"}]),
            patch.object(pipeline, "_retrieval_namespaces", return_value=(SimpleNamespace(), SimpleNamespace())),
            patch.object(pipeline, "_encode_query_batch", side_effect=encode),
            patch.object(pipeline, "_batch_sparse_topics", return_value={"1": "red"}),
            patch.object(pipeline, "_progress", side_effect=ProgressRecorder()),
        ):
            first = pipeline.encode_queries_online(args, runtime, **kwargs)
            second = pipeline.encode_queries_online(args, runtime, **kwargs)
        self.assertIs(first, second)
        builder.assert_called_once()
        self.assertEqual(timer.components, {"query_encoding": 3})
        self.assertEqual(clock.seconds, 13)

    def test_ranker_timer_excludes_loading_construction_and_cleanup(self) -> None:
        import torch
        clock = FakeClock()
        args = ThreeStageRecallTests.args()
        fake_ranker = ModuleType("reranker")
        fake_ranker.path_prefix = "original"

        def construct(*args: Any) -> Any:
            clock.advance(7)
            return SimpleNamespace()

        fake_ranker.Reranker = construct

        def load(*args: Any) -> Any:
            clock.advance(50)
            return SimpleNamespace(), SimpleNamespace(tokenizer=SimpleNamespace(get_vocab=lambda: {}))

        def dataset(*args: Any) -> Any:
            clock.advance(10)
            return SimpleNamespace(text_dict={}, img_dict={})

        def rerank(*args: Any) -> Any:
            clock.advance(4)
            return {"1": {"11": -1.0}}

        timer = pipeline.StageTimer()
        with (
            patch.dict(sys.modules, {"reranker": fake_ranker}),
            patch.object(pipeline.time, "perf_counter", clock),
            patch.object(pipeline, "_load_ranker_model", side_effect=load),
            patch.object(pipeline, "build_dataset", side_effect=dataset),
            patch.object(pipeline, "_dispatch_stage3_rerank", side_effect=rerank),
            patch.object(pipeline.gc, "collect", side_effect=lambda: clock.advance(8)),
            patch.object(torch.cuda, "is_available", return_value=False),
        ):
            result = pipeline.run_stage3(args, {"1": {"11": 1.0}}, timing=timer)
            empty_timer = pipeline.StageTimer()
            pipeline.run_stage3(args, {}, timing=empty_timer)
        self.assertEqual(result, {"1": {"11": -1.0}})
        self.assertEqual(timer.components, {"reranking": 4})
        self.assertEqual(empty_timer.snapshot(0)["compute_seconds"], 0.0)
        self.assertEqual(fake_ranker.path_prefix, "original")

    def test_stage2_passes_one_timer_to_encoding_search_and_fusion(self) -> None:
        import numpy as np
        clock = FakeClock()
        timer = pipeline.StageTimer()
        args = ThreeStageRecallTests.args(stage2_run=None, stage2_dense_index="dense", stage2_sparse_index="sparse")
        run = {"1": {"11": 1.0}}

        def encode(*call_args: Any, **kwargs: Any) -> Any:
            self.assertIs(kwargs["timing"], timer)
            with timer.measure("query_encoding"):
                clock.advance(2)
            return pipeline.EncodedQueries(["1"], np.array([[1.0]]), {"1": "red"})

        def search(component: str, seconds: float, **kwargs: Any) -> Any:
            self.assertIs(kwargs["timing"], timer)
            with timer.measure(component):
                clock.advance(seconds)
            return run

        with (
            patch.object(pipeline.time, "perf_counter", clock),
            patch.object(pipeline, "encode_queries_online", side_effect=encode),
            patch.object(pipeline, "retrieve_dense_from_corpus", side_effect=lambda *a, **kw: search("dense_search", 3, **kw)),
            patch.object(pipeline, "retrieve_sparse_topics_from_lucene", side_effect=lambda *a, **kw: search("sparse_search", 4, **kw)),
            patch.object(pipeline, "_progress", side_effect=ProgressRecorder()),
        ):
            _, _, result = pipeline.run_stage2(args, run, SimpleNamespace(), timing=timer)
        self.assertEqual(set(result["1"]), {"11"})
        self.assertEqual(timer.snapshot(0)["compute_seconds"], 9)
        self.assertEqual(set(timer.components), {"query_encoding", "dense_search", "sparse_search", "fusion"})

    def test_precomputed_and_external_sources_do_not_claim_compute_time(self) -> None:
        args = ThreeStageRecallTests.args(ranker_run="stage3.json")
        clock = FakeClock()
        run = {"1": {"11": 1.0}}

        def read(*args: Any) -> Any:
            clock.advance(100)
            return run

        with (
            patch.object(pipeline.time, "perf_counter", clock),
            patch.object(pipeline, "read_run", side_effect=read),
            patch.object(pipeline, "_progress", side_effect=ProgressRecorder()),
            patch.object(pipeline, "_run_external_command", side_effect=lambda *args: clock.advance(30)),
        ):
            for stage in (1, 2, 3):
                timer = pipeline.StageTimer()
                if stage == 1:
                    pipeline.run_stage1(args, timing=timer)
                elif stage == 2:
                    pipeline.run_stage2(args, run, timing=timer)
                else:
                    pipeline.run_stage3(args, run, timing=timer)
                self.assertEqual(timer.source, "precomputed")
                self.assertEqual(timer.snapshot(0)["compute_seconds"], 0.0)
            args.stage1_command = "external"
            timer = pipeline.StageTimer()
            pipeline.run_stage1(args, timing=timer)
        self.assertIsNone(timer.snapshot(0)["compute_seconds"])
        self.assertEqual(timer.auxiliary, {"external_command_wall": 30})

    def test_distributed_reduction_uses_one_slowest_rank_path_not_component_maxima(self) -> None:
        first, second, empty = pipeline.StageTimer(), pipeline.StageTimer(), pipeline.StageTimer()
        first.components = {"query_encoding": 4, "dense_search": 1}
        second.components = {"query_encoding": 1, "dense_search": 6}
        result = pipeline._merge_stage_timings([first.snapshot(0), second.snapshot(1), empty.snapshot(2)])
        self.assertEqual(result["compute_seconds"], 7)
        self.assertEqual(result["critical_rank"], 1)
        self.assertEqual(result["components_seconds"], second.components)
        self.assertEqual(result["per_rank_compute_seconds"], {"0": 5, "1": 7, "2": 0})

    def test_distributed_timing_gather_and_worker_no_output(self) -> None:
        import torch.distributed as dist
        timer = pipeline.StageTimer()
        timer.components = {"reranking": 2}

        def gather(local: Any, output: Any, dst: int) -> None:
            self.assertEqual(dst, 0)
            if output is not None:
                other = pipeline.StageTimer()
                other.components = {"reranking": 5}
                output[:] = [local, other.snapshot(1)]

        with patch.object(dist, "gather_object", side_effect=gather) as gather_mock:
            result = pipeline._collect_stage_timing(timer, pipeline.DistributedContext(world_size=2))
            worker = pipeline.DistributedContext(rank=1, world_size=2)
            self.assertIsNone(pipeline._collect_stage_timing(timer, worker))
            pipeline._record_stage_timing("stage3", timer, worker, None)
        self.assertEqual(result["compute_seconds"], 5)
        self.assertEqual(gather_mock.call_count, 3)

    def test_reporter_totals_outputs_and_failed_finalize_need_no_collectives(self) -> None:
        import torch.distributed as dist
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with (
                patch.object(pipeline.time, "perf_counter", clock),
                patch.object(pipeline.RecallReporter, "_output_dependencies", side_effect=fake_excel_dependencies),
                patch.object(pipeline.tqdm, "write") as write,
                patch.object(dist, "gather_object") as gather,
            ):
                reporter = pipeline.TimingReporter(ThreeStageRecallTests.args(), output_dir, clock())
                for index, seconds in enumerate((2, 3, 4), start=1):
                    clock.advance(100 + seconds)  # simulated loading/evaluation/I/O overhead
                    timer = pipeline.StageTimer()
                    timer.components = {"compute": seconds}
                    reporter.record(f"stage{index}", pipeline._merge_stage_timings([timer.snapshot(0)]))
                    payload = json.loads((output_dir / "timing_metrics.json").read_text(encoding="utf-8"))
                    self.assertEqual(payload["stages"][index - 1]["status"], "completed")
                clock.advance(10)
                reporter.finish(completed=True)
                payload = json.loads((output_dir / "timing_metrics.json").read_text(encoding="utf-8"))
                table = json.loads((output_dir / "timing_metrics.xlsx").read_text(encoding="utf-8"))
                self.assertEqual(payload["totals"]["compute_seconds"], 9)
                self.assertEqual(payload["totals"]["wall_seconds"], 319)
                totals = {row["component"]: row["seconds"] for row in table if row["record_type"] == "total"}
                self.assertEqual(totals, {"core_compute_total": 9, "end_to_end_total": 319})
                self.assertIn("core_compute=9.000s", "\n".join(call.args[0] for call in write.call_args_list))
                reporter.finish(completed=False, failed_stage="stage3")
                failed = json.loads((output_dir / "timing_metrics.json").read_text(encoding="utf-8"))
                self.assertEqual(failed["status"], "failed")
                self.assertIsNone(failed["totals"]["compute_seconds"])
                self.assertEqual(failed["totals"]["measured_compute_seconds"], 5)
                self.assertEqual(failed["stages"][0]["status"], "completed")
                self.assertEqual(failed["stages"][2]["status"], "failed")
                gather.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("openpyxl"), "optional Excel dependency openpyxl is unavailable")
    def test_timing_excel_workbook_matches_json(self) -> None:
        import openpyxl
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(pipeline.tqdm, "write"):
            output_dir = Path(temp_dir)
            reporter = pipeline.TimingReporter(ThreeStageRecallTests.args(), output_dir, pipeline.time.perf_counter())
            for stage, seconds in (("stage1", 1.25), ("stage2", 2.5), ("stage3", 3.75)):
                timer = pipeline.StageTimer()
                timer.components = {"compute": seconds}
                reporter.record(stage, pipeline._merge_stage_timings([timer.snapshot(0)]))
            reporter.finish(completed=True)
            workbook = openpyxl.load_workbook(output_dir / "timing_metrics.xlsx", read_only=True)
            try:
                values = list(workbook.active.values)
                rows = [dict(zip(values[0], row)) for row in values[1:]]
            finally:
                workbook.close()
            payload = json.loads((output_dir / "timing_metrics.json").read_text(encoding="utf-8"))
            stage_rows = [row for row in rows if row["record_type"] == "stage"]
            self.assertEqual([row["seconds"] for row in stage_rows], [1.25, 2.5, 3.75])
            total = next(row for row in rows if row.get("component") == "core_compute_total")
            self.assertEqual(total["seconds"], payload["totals"]["compute_seconds"])
            self.assertEqual(total["seconds"], 7.5)


class ThreeStageClipEncodingTests(unittest.TestCase):
    def test_long_text_is_truncated_by_processor_to_model_limit(self) -> None:
        import torch

        model = FakeClipModel()
        processor = FakeClipProcessor()

        features = pipeline._encode_feature_batch(
            model,
            processor,
            ["long caption " * 100],
            "text",
            torch.device("cpu"),
        )

        self.assertIsNotNone(processor.call_kwargs)
        self.assertTrue(processor.call_kwargs["truncation"])
        self.assertEqual(processor.call_kwargs["max_length"], 77)
        self.assertEqual(model.input_shape, (1, 77))
        self.assertEqual(model.attention_shape, (1, 77))
        self.assertEqual(tuple(features.shape), (1, 4))

    def test_clip_text_limit_falls_back_to_tokenizer(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace())
        processor = SimpleNamespace(
            tokenizer=SimpleNamespace(model_max_length=64)
        )

        self.assertEqual(pipeline._clip_text_max_length(model, processor), 64)

    def test_t2it_text_is_truncated_by_processor_to_model_limit(self) -> None:
        import torch

        model = FakeClipModel()
        processor = FakeClipProcessor()

        features = pipeline._encode_clip_t2it_batch(
            model,
            processor,
            [object()],
            ["long caption " * 100],
            torch.device("cpu"),
        )

        self.assertTrue(processor.call_kwargs["truncation"])
        self.assertEqual(processor.call_kwargs["max_length"], 77)
        self.assertEqual(model.input_shape, (1, 77))
        self.assertEqual(tuple(features.shape), (1, 4))


if __name__ == "__main__":
    unittest.main()
