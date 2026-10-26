from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import test_three_stage_retrieval_pipeline as fixtures


pipeline = fixtures.pipeline


def encodings(ids=("1", "2")):
    return pipeline.EncodedQueries(
        list(ids), np.array([[float(index), 1.0] for index in range(len(ids))], dtype="float32").reshape(-1, 2),
        {query_id: f"topic-{query_id}" for query_id in ids},
    )


def run_for(ids):
    return {query_id: {"11": 2.0, "21": 1.0} for query_id in ids}


class QueryRepresentationTests(unittest.TestCase):
    def test_selection_preserves_requested_order_and_empty_shape(self):
        encoded = encodings(("1", "2", "3"))
        shard = pipeline._select_encoded_queries(encoded, ["3", "1"])
        self.assertEqual(shard.query_ids, ["3", "1"])
        np.testing.assert_array_equal(shard.dense_representations, encoded.dense_representations[[2, 0]])
        self.assertEqual(shard.sparse_topics, {"3": "topic-3", "1": "topic-1"})
        self.assertFalse(np.shares_memory(shard.dense_representations, encoded.dense_representations))
        empty = pipeline._select_encoded_queries(encoded, [])
        self.assertEqual(empty.dense_representations.shape, (0, 2))
        pipeline._validate_encoded_queries(empty, [])

    def test_selection_rejects_missing_and_duplicate_ids(self):
        for ids in (["missing"], ["1", "1"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                pipeline._select_encoded_queries(encodings(), ids)

    def test_validation_rejects_corrupt_or_misaligned_representations(self):
        invalid = [
            pipeline.EncodedQueries(["1", "1"], np.ones((2, 2)), {"1": "topic"}),
            pipeline.EncodedQueries(["1"], np.ones((2, 2)), {"1": "topic"}),
            pipeline.EncodedQueries(["1"], np.ones((1, 2)), {}),
            pipeline.EncodedQueries(["1"], np.array([[np.nan]]), {"1": "topic"}),
            pipeline.EncodedQueries(["1"], np.ones((1, 0)), {"1": "topic"}),
            pipeline.EncodedQueries([1], np.ones((1, 2)), {1: "topic"}),
        ]
        for encoded in invalid:
            with self.subTest(encoded=encoded), self.assertRaises(ValueError):
                pipeline._validate_encoded_queries(encoded)
        with self.assertRaisesRegex(ValueError, "order"):
            pipeline._validate_encoded_queries(encodings(), ["2", "1"])

    def test_distributed_scatter_matches_run_shards_without_padding(self):
        import torch.distributed as dist

        for count, world_size in ((5, 3), (2, 4), (0, 3)):
            with self.subTest(count=count, world_size=world_size):
                ids = [str(index) for index in range(count)]
                encoded = encodings(ids)
                # The run need not use the dataset's encoding order.
                run = run_for(list(reversed(ids)))
                shards = pipeline._partition_run(run, world_size)
                packets = []
                current_rank = 0

                def scatter(received, outgoing, src):
                    self.assertEqual(src, 0)
                    if outgoing is not None:
                        # Round-trip as real object collectives do: workers must
                        # not depend on the root's Python object references.
                        packets[:] = pickle.loads(pickle.dumps(outgoing))
                    received[0] = packets[current_rank]

                all_ids = []
                with patch.object(dist, "scatter_object_list", side_effect=scatter):
                    for current_rank in range(world_size):
                        context = pipeline.DistributedContext(current_rank, current_rank, world_size)
                        shard = pipeline._scatter_encoded_queries(
                            encoded if context.is_main else None,
                            run if context.is_main else {}, shards[current_rank], context,
                        )
                        self.assertEqual(shard.query_ids, list(shards[current_rank]))
                        for row, query_id in zip(shard.dense_representations, shard.query_ids):
                            np.testing.assert_array_equal(row, encoded.dense_representations[ids.index(query_id)])
                            self.assertEqual(shard.sparse_topics[query_id], f"topic-{query_id}")
                        all_ids.extend(shard.query_ids)
                self.assertCountEqual(all_ids, ids)

    def test_scatter_propagates_root_validation_error_to_every_rank(self):
        import torch.distributed as dist

        packets = []
        current_rank = 0

        def scatter(received, outgoing, src):
            if outgoing is not None:
                packets[:] = outgoing
            received[0] = packets[current_rank]

        with patch.object(dist, "scatter_object_list", side_effect=scatter):
            for current_rank in range(3):
                context = pipeline.DistributedContext(current_rank, current_rank, 3)
                with self.assertRaisesRegex(ValueError, "coverage differ"):
                    pipeline._scatter_encoded_queries(
                        encodings() if context.is_main else None,
                        run_for(["1"]) if context.is_main else {}, {}, context,
                    )


class StageQueryReuseTests(unittest.TestCase):
    def args(self, **kwargs):
        defaults = dict(
            stage1_method="sparse_concat", stage1_run=None, stage1_sparse_index="sparse",
            stage2_run=None, stage2_dense_index="dense", stage2_sparse_index="sparse",
            retrieval_model_name_or_path="fake-query-model",
        )
        defaults.update(kwargs)
        return fixtures.ThreeStageRecallTests.args(**defaults)

    def test_stage1_accepts_encodings_without_loading_or_running_encoder(self):
        encoded = encodings()
        with (
            patch.object(pipeline, "encode_queries_online", side_effect=AssertionError("reencoded")),
            patch.object(pipeline, "retrieve_sparse_topics_from_lucene", return_value=run_for(encoded.query_ids)) as search,
        ):
            pipeline.run_stage1(self.args(), encoded_queries=encoded)
        self.assertIs(search.call_args.args[1], encoded.sparse_topics)
        self.assertEqual(search.call_args.kwargs["qids"], encoded.query_ids)

    def test_stage2_matches_recomputed_rankings_and_recall_without_encoding(self):
        args = self.args()
        encoded = encodings()
        run = run_for(encoded.query_ids)
        dense = {"1": {"11": 1.0, "21": 3.0}, "2": {"11": 3.0, "21": 1.0}}
        timer = pipeline.StageTimer()

        def retrieve(vectors, query_ids, *call_args, **kwargs):
            np.testing.assert_array_equal(vectors, encoded.dense_representations)
            self.assertEqual(query_ids, encoded.query_ids)
            return dense

        with (
            patch.object(pipeline, "encode_queries_online", return_value=encoded) as encode,
            patch.object(pipeline, "retrieve_dense_from_corpus", side_effect=retrieve),
            patch.object(pipeline, "retrieve_sparse_topics_from_lucene", return_value=run),
            patch.object(pipeline, "_progress", side_effect=fixtures.ProgressRecorder()),
        ):
            before = pipeline.run_stage2(args, run, SimpleNamespace())
            encode.assert_called_once()
            encode.reset_mock()
            after = pipeline.run_stage2(args, run, encoded_queries=encoded, timing=timer)
            encode.assert_not_called()
        self.assertEqual(before, after)
        with patch.object(pipeline, "build_dataset", return_value=fixtures.ThreeStageRecallTests.dataset()):
            evaluation = pipeline._build_recall_evaluation(args, run)
        for old, new in zip(before, after):
            self.assertEqual(
                pipeline.evaluate_recall(old, evaluation, stage="stage2", variant="test"),
                pipeline.evaluate_recall(new, evaluation, stage="stage2", variant="test"),
            )
        self.assertEqual(timer.components["query_encoding"], 0.0)

    def test_stage2_rejects_incorrect_shard_without_fallback(self):
        with patch.object(pipeline, "encode_queries_online") as encode:
            for encoded in (encodings(["1"]), encodings(["2", "1"])):
                with self.subTest(encoded=encoded), self.assertRaisesRegex(ValueError, "IDs/order"):
                    pipeline.run_stage2(self.args(), run_for(["1", "2"]), encoded_queries=encoded)
            args = self.args()
            args.stage1_method = "sparse_manual"
            with self.assertRaisesRegex(ValueError, "sparse_concat"):
                pipeline.run_stage2(args, run_for(["1", "2"]), encoded_queries=encodings())
            encode.assert_not_called()

    def test_empty_stage2_shard_does_not_encode_or_search(self):
        timer = pipeline.StageTimer()
        with (
            patch.object(pipeline, "encode_queries_online") as encode,
            patch.object(pipeline, "retrieve_dense_from_corpus") as dense,
            patch.object(pipeline, "retrieve_sparse_topics_from_lucene") as sparse,
        ):
            result = pipeline.run_stage2(self.args(), {}, encoded_queries=encodings([]), timing=timer)
        self.assertEqual(result, ({}, {}, {}))
        encode.assert_not_called()
        dense.assert_not_called()
        sparse.assert_not_called()
        self.assertEqual(timer.components, {"query_encoding": 0.0})

    def test_main_encodes_once_reports_reuse_and_counts_encoding_only_in_stage1(self):
        clock = fixtures.FakeClock()
        encoded = encodings()
        run = run_for(encoded.query_ids)
        runtime = SimpleNamespace(encoder=None)

        def encode(*args, **kwargs):
            with kwargs["timing"].measure("query_encoding"):
                clock.advance(7)
            return encoded

        def search(*args, **kwargs):
            with kwargs["timing"].measure("sparse_search"):
                clock.advance(1)
            return run

        def dense(*args, **kwargs):
            with kwargs["timing"].measure("dense_search"):
                clock.advance(3)
            return run

        def load(*args):
            clock.advance(20)
            return runtime

        with tempfile.TemporaryDirectory() as temporary:
            args = self.args()
            args.output_dir = str(Path(temporary) / "results")
            with (
                patch.object(pipeline, "_parser", return_value=SimpleNamespace(parse_args=lambda: args)),
                patch.object(pipeline.time, "perf_counter", clock),
                patch.object(pipeline, "_load_retrieval_runtime", side_effect=load) as loader,
                patch.object(pipeline, "_release_retrieval_runtime") as release,
                patch.object(pipeline, "encode_queries_online", side_effect=encode) as encoder,
                patch.object(pipeline, "_scatter_encoded_queries", side_effect=AssertionError("transferred vectors")) as scatter,
                patch.object(pipeline, "retrieve_sparse_topics_from_lucene", side_effect=search),
                patch.object(pipeline, "retrieve_dense_from_corpus", side_effect=dense),
                patch.object(pipeline, "run_stage3", return_value=run),
                patch.object(pipeline, "build_dataset", return_value=fixtures.ThreeStageRecallTests.dataset()),
                patch.object(pipeline.RecallReporter, "_output_dependencies", side_effect=fixtures.fake_excel_dependencies),
                patch.object(pipeline, "_progress", side_effect=fixtures.ProgressRecorder()),
            ):
                pipeline.main()
            loader.assert_called_once()
            encoder.assert_called_once()
            scatter.assert_not_called()
            self.assertEqual(encoder.call_args.kwargs["progress_prefix"], "[stage1]")
            self.assertEqual(encoder.call_args.kwargs["requested_query_ids"], ["1", "2"])
            release.assert_any_call(runtime)
            metadata = json.loads((Path(args.output_dir) / "pipeline_metadata.json").read_text(encoding="utf-8"))
            timings = json.loads((Path(args.output_dir) / "timing_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["stage2"]["query_encoding"], "reused_stage1")
        self.assertEqual(metadata["stage2"]["reused_query_count"], 2)
        self.assertEqual(metadata["stage1"]["per_rank_query_counts"], {"0": 2})
        self.assertFalse(metadata["distributed"]["stage1_query_sharded"])
        stage1, stage2, _ = timings["stages"]
        self.assertEqual(stage1["components_seconds"]["query_encoding"], 7)
        self.assertEqual(stage2["components_seconds"]["query_encoding"], 0)
        self.assertNotIn("query_encoding_transfer", stage2["auxiliary_seconds"])
        self.assertEqual(timings["totals"]["compute_seconds"], 12)
        self.assertEqual(timings["totals"]["wall_seconds"], 32)

    def test_worker_encodes_its_own_queries_once_and_reuses_locally(self):
        args = self.args()
        local_run = run_for(["2"])
        shard = encodings(["2"])
        context = pipeline.DistributedContext(rank=1, local_rank=1, world_size=3)
        with tempfile.TemporaryDirectory() as temporary:
            args.output_dir = str(Path(temporary) / "results")
            with (
                patch.object(pipeline, "_parser", return_value=SimpleNamespace(parse_args=lambda: args)),
                patch.object(pipeline, "_initialise_distributed", return_value=context),
                patch.object(pipeline, "_shutdown_distributed"),
                patch.object(pipeline, "_collect_stage_timing", return_value=None),
                patch.object(pipeline, "_stage1_query_shard", return_value=(["2"], [], [1, 1, 0])),
                patch.object(pipeline, "_scatter_run", side_effect=AssertionError("repartitioned run")),
                patch.object(pipeline, "_scatter_encoded_queries", side_effect=AssertionError("transferred vectors")) as scatter,
                patch.object(pipeline, "_gather_run", return_value={}),
                patch.object(pipeline, "_load_retrieval_runtime", return_value=SimpleNamespace()) as loader,
                patch.object(pipeline, "_release_retrieval_runtime"),
                patch.object(pipeline, "encode_queries_online", return_value=shard) as encode,
                patch.object(pipeline, "retrieve_sparse_topics_from_lucene", return_value=local_run),
                patch.object(pipeline, "retrieve_dense_from_corpus", return_value=local_run),
                patch.object(pipeline, "run_stage3", return_value=local_run),
                patch.object(pipeline, "_progress", side_effect=fixtures.ProgressRecorder()),
            ):
                pipeline.main()
            loader.assert_called_once_with(args, context)
            encode.assert_called_once()
            self.assertEqual(encode.call_args.kwargs["requested_query_ids"], ["2"])
            self.assertEqual(encode.call_args.kwargs["progress_prefix"], "[stage1][rank 1/3]")
            scatter.assert_not_called()

    def test_precomputed_stages_keep_existing_encoding_paths(self):
        encoded = encodings()
        run = run_for(encoded.query_ids)
        for stage1_run, stage2_run, expected_prefix, expected_source in (
            ("stage1.json", None, "[stage2]", "online"),
            (None, "stage2.json", "[stage1]", "precomputed_run"),
        ):
            with self.subTest(stage1_run=stage1_run, stage2_run=stage2_run), tempfile.TemporaryDirectory() as temporary:
                args = self.args(stage1_run=stage1_run, stage2_run=stage2_run)
                args.output_dir = str(Path(temporary) / "results")
                with (
                    patch.object(pipeline, "_parser", return_value=SimpleNamespace(parse_args=lambda: args)),
                    patch.object(pipeline, "_load_retrieval_runtime", return_value=SimpleNamespace()) as loader,
                    patch.object(pipeline, "_release_retrieval_runtime"),
                    patch.object(pipeline, "encode_queries_online", return_value=encoded) as encoder,
                    patch.object(pipeline, "_scatter_encoded_queries", side_effect=AssertionError("unexpected reuse")),
                    patch.object(pipeline, "retrieve_sparse_topics_from_lucene", return_value=run),
                    patch.object(pipeline, "retrieve_dense_from_corpus", return_value=run),
                    patch.object(pipeline, "read_run", return_value=run),
                    patch.object(pipeline, "run_stage3", return_value=run),
                    patch.object(pipeline, "build_dataset", return_value=fixtures.ThreeStageRecallTests.dataset()),
                    patch.object(pipeline.RecallReporter, "_output_dependencies", side_effect=fixtures.fake_excel_dependencies),
                    patch.object(pipeline, "_progress", side_effect=fixtures.ProgressRecorder()),
                ):
                    pipeline.main()
                loader.assert_called_once()
                encoder.assert_called_once()
                self.assertEqual(encoder.call_args.kwargs["progress_prefix"], expected_prefix)
                metadata = json.loads((Path(args.output_dir) / "pipeline_metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["stage2"]["query_encoding"], expected_source)
                self.assertEqual(metadata["stage2"]["reused_query_count"], 0)

    def test_sparse_manual_keeps_distinct_stage2_encoding(self):
        args = self.args(stage1_method="sparse_manual")
        encoded = encodings()
        run = run_for(encoded.query_ids)
        with (
            patch.object(pipeline, "encode_queries_online", return_value=encoded) as encoder,
            patch.object(pipeline, "retrieve_sparse_topics_from_lucene", return_value=run),
            patch.object(pipeline, "retrieve_dense_from_corpus", return_value=run),
            patch.object(pipeline, "_progress", side_effect=fixtures.ProgressRecorder()),
        ):
            runtime = SimpleNamespace()
            pipeline.run_stage1(args, runtime)
            pipeline.run_stage2(args, run, runtime)
        self.assertEqual(encoder.call_count, 2)
        first, second = encoder.call_args_list
        self.assertEqual((first.kwargs["eol_type"], first.kwargs["calculate_type"]), ("prompteol", "separate"))
        self.assertEqual((second.kwargs["eol_type"], second.kwargs["calculate_type"]), ("disassembleeol_separate", "concat"))

    def test_script_defaults_to_sparse_concat_without_clip_dependency(self):
        script = (fixtures.PROJECT_ROOT / "scripts" / "three_stage_retrieval_pipeline.sh").read_text(encoding="utf-8")
        self.assertIn("--stage1-method sparse_concat", script)
        self.assertIn('STAGE1_SPARSE_INDEX="${STAGE1_SPARSE_INDEX:-${STAGE2_SPARSE_INDEX}}"', script)
        self.assertIn('--stage1-sparse-index "${STAGE1_SPARSE_INDEX}"', script)
        self.assertIn('require_lucene_index "${STAGE1_SPARSE_INDEX}"', script)
        self.assertIn('require_lucene_index "${STAGE2_SPARSE_INDEX}"', script)
        self.assertIn('OUTPUT_VARIANT="sparse_concat_stage1"', script)
        self.assertIn('/${DIRECTION_LABEL}/${DATASET_SPLIT}/${OUTPUT_VARIANT}}', script)
        self.assertNotIn("CLIP_MODEL_PATH", script)


if __name__ == "__main__":
    unittest.main()
