from __future__ import annotations

import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import test_three_stage_retrieval_pipeline as fixtures


pipeline = fixtures.pipeline


def sparse_args():
    return fixtures.ThreeStageRecallTests.args(
        stage1_method="sparse_concat", stage1_run=None, stage1_sparse_index="sparse",
        stage2_run=None, stage2_sparse_index="sparse", stage2_dense_index="dense",
        retrieval_model_name_or_path="fake-query-model",
    )


def encode_ids(ids):
    return pipeline.EncodedQueries(
        list(ids), np.array([[float(query_id), 1.0] for query_id in ids], dtype="float32").reshape(-1, 2),
        {query_id: f"topic-{query_id}" for query_id in ids},
    )


def sparse_run(ids):
    return {query_id: {"11": 2.0, "21": 1.0} for query_id in ids}


class Stage1QueryPartitionTests(unittest.TestCase):
    def test_partition_preserves_query_order_without_padding_or_duplication(self):
        import torch.distributed as dist

        for count, world_size in ((5, 3), (2, 4), (1, 1), (7, 2)):
            with self.subTest(count=count, world_size=world_size):
                ids = [str(index) for index in reversed(range(1, count + 1))]
                packets = []
                current_rank = 0

                def scatter(received, outgoing, src):
                    self.assertEqual(src, 0)
                    if outgoing is not None:
                        packets[:] = pickle.loads(pickle.dumps(outgoing))
                    received[0] = packets[current_rank]

                all_ids = []
                with (
                    patch.object(pipeline, "build_dataset", return_value=SimpleNamespace(img_id_list=ids)) as builder,
                    patch.object(dist, "scatter_object_list", side_effect=scatter),
                ):
                    for current_rank in range(world_size):
                        context = pipeline.DistributedContext(current_rank, current_rank, world_size)
                        shard, global_ids, counts = pipeline._stage1_query_shard(sparse_args(), context)
                        self.assertEqual(shard, ids[current_rank::world_size])
                        self.assertEqual(global_ids, ids if context.is_main else [])
                        self.assertEqual(counts, [len(ids[rank::world_size]) for rank in range(world_size)])
                        all_ids.extend(shard)
                    builder.assert_called_once()
                self.assertCountEqual(all_ids, ids)

    def test_empty_duplicate_and_missing_metadata_errors_reach_all_ranks(self):
        import torch.distributed as dist

        for dataset in (SimpleNamespace(img_id_list=[]), SimpleNamespace(img_id_list=[1, "1.0"]), SimpleNamespace()):
            with self.subTest(dataset=dataset):
                packets = []
                current_rank = 0

                def scatter(received, outgoing, src):
                    if outgoing is not None:
                        packets[:] = outgoing
                    received[0] = packets[current_rank]

                with (
                    patch.object(pipeline, "build_dataset", return_value=dataset),
                    patch.object(dist, "scatter_object_list", side_effect=scatter),
                    patch.object(pipeline, "_load_retrieval_runtime") as loader,
                ):
                    for current_rank in range(3):
                        context = pipeline.DistributedContext(current_rank, current_rank, 3)
                        with self.assertRaisesRegex(ValueError, "Cannot partition Stage-1 queries"):
                            pipeline._run_sparse_concat_stage1(sparse_args(), context, pipeline.StageTimer())
                    loader.assert_not_called()

    def test_query_id_source_matches_retrieval_direction(self):
        dataset = SimpleNamespace(img_id_list=["image"], text_id_list=["text"],
                                  query_id_list=["question"], composed_id_list=["composed"])
        for task, query_type, expected in (
            ("ir", "image", "image"), ("ir", "text", "text"), ("tbpr", "text", "text"),
            ("it2t", "image", "question"), ("t2it", "text", "question"), ("cir", "composed", "composed"),
        ):
            with self.subTest(task=task, query_type=query_type):
                self.assertEqual(
                    pipeline._dataset_query_ids(SimpleNamespace(task_type=task, query_type=query_type), dataset),
                    [expected],
                )


class Stage1ParallelExecutionTests(unittest.TestCase):
    def test_empty_worker_skips_weights_encoding_and_search_but_joins_gather(self):
        context = pipeline.DistributedContext(rank=2, local_rank=2, world_size=4)
        timer = pipeline.StageTimer()
        with (
            patch.object(pipeline, "_stage1_query_shard", return_value=([], [], [1, 1, 0, 0])),
            patch.object(pipeline, "_load_retrieval_runtime", side_effect=AssertionError("loaded empty worker")),
            patch.object(pipeline, "encode_queries_online", side_effect=AssertionError("encoded empty worker")),
            patch.object(pipeline, "retrieve_sparse_topics_from_lucene", side_effect=AssertionError("searched empty worker")),
            patch.object(pipeline, "_gather_run", return_value={}) as gather,
        ):
            local, encoded, merged, counts = pipeline._run_sparse_concat_stage1(sparse_args(), context, timer)
        self.assertEqual((local, merged, counts), ({}, {}, [1, 1, 0, 0]))
        gather.assert_called_once_with({}, context)
        self.assertEqual(encoded.dense_representations.shape, (0, 0))
        self.assertEqual(timer.snapshot(2)["compute_seconds"], 0.0)
        self.assertEqual(pipeline.run_stage2(sparse_args(), local, encoded_queries=encoded), ({}, {}, {}))

    def test_worker_releases_model_but_keeps_local_cpu_encodings(self):
        import torch

        context = pipeline.DistributedContext(rank=1, local_rank=1, world_size=3)
        ids = ["2", "5"]
        encoded = encode_ids(ids)
        runtime = SimpleNamespace(model=object(), encoder=object(), query_cache={"cached": encoded})
        with (
            patch.object(pipeline, "_stage1_query_shard", return_value=(ids, [], [2, 2, 1])),
            patch.object(pipeline, "_load_retrieval_runtime", return_value=runtime) as loader,
            patch.object(pipeline, "encode_queries_online", return_value=encoded) as encoder,
            patch.object(pipeline, "retrieve_sparse_topics_from_lucene", return_value=sparse_run(list(reversed(ids)))),
            patch.object(pipeline, "_gather_run", return_value={}),
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(pipeline.gc, "collect"),
        ):
            local, retained, _, _ = pipeline._run_sparse_concat_stage1(sparse_args(), context, pipeline.StageTimer())
        loader.assert_called_once()
        encoder.assert_called_once()
        self.assertEqual(encoder.call_args.kwargs["requested_query_ids"], ids)
        self.assertEqual(list(local), ids)
        self.assertIs(retained, encoded)
        np.testing.assert_array_equal(retained.dense_representations, [[2.0, 1.0], [5.0, 1.0]])
        self.assertIsNone(runtime.model)
        self.assertIsNone(runtime.encoder)
        self.assertEqual(runtime.query_cache, {})

    def test_root_checks_global_coverage_and_restores_dataset_order(self):
        context = pipeline.DistributedContext(rank=0, local_rank=0, world_size=2)
        ids, global_ids = ["3", "2"], ["3", "1", "2"]
        for gathered, succeeds in ((sparse_run(["2", "1", "3"]), True), (sparse_run(["2", "3"]), False)):
            with (
                self.subTest(succeeds=succeeds),
                patch.object(pipeline, "_stage1_query_shard", return_value=(ids, global_ids, [2, 1])),
                patch.object(pipeline, "_load_retrieval_runtime", return_value=SimpleNamespace()),
                patch.object(pipeline, "_release_retrieval_runtime"),
                patch.object(pipeline, "encode_queries_online", return_value=encode_ids(ids)),
                patch.object(pipeline, "retrieve_sparse_topics_from_lucene", return_value=sparse_run(ids)),
                patch.object(pipeline, "_gather_run", return_value=gathered),
            ):
                if succeeds:
                    _, _, merged, _ = pipeline._run_sparse_concat_stage1(sparse_args(), context, pipeline.StageTimer())
                    self.assertEqual(list(merged), global_ids)
                else:
                    with self.assertRaisesRegex(ValueError, "Stage 1 query.*missing=1"):
                        pipeline._run_sparse_concat_stage1(sparse_args(), context, pipeline.StageTimer())

    def test_local_missing_query_is_not_silently_reencoded(self):
        runtime = SimpleNamespace()
        with (
            patch.object(pipeline, "_stage1_query_shard", return_value=(["1", "2"], ["1", "2"], [2])),
            patch.object(pipeline, "_load_retrieval_runtime", return_value=runtime),
            patch.object(pipeline, "_release_retrieval_runtime") as release,
            patch.object(pipeline, "encode_queries_online", return_value=encode_ids(["1", "2"])) as encode,
            patch.object(pipeline, "retrieve_sparse_topics_from_lucene", return_value=sparse_run(["1"])),
            patch.object(pipeline, "_gather_run") as gather,
        ):
            with self.assertRaisesRegex(ValueError, "Stage 1 local query.*missing=2"):
                pipeline._run_sparse_concat_stage1(sparse_args(), pipeline.DistributedContext(), pipeline.StageTimer())
        encode.assert_called_once()
        release.assert_called_once_with(runtime)
        gather.assert_not_called()

    def test_serial_and_sharded_runs_and_recall_match_with_one_encoding_per_query(self):
        ids = ["1", "2", "3", "4", "5"]
        args = sparse_args()

        def execute(world_size):
            seen = []
            counts = [len(ids[rank::world_size]) for rank in range(world_size)]

            def partition(args, context):
                return ids[context.rank::world_size], ids if context.is_main else [], counts

            def encode(*call_args, **kwargs):
                seen.extend(kwargs["requested_query_ids"])
                return encode_ids(kwargs["requested_query_ids"])

            def sparse(index, topics, qids, **kwargs):
                self.assertEqual(set(topics), set(qids))
                return sparse_run(qids)

            def dense(vectors, qids, *call_args, **kwargs):
                np.testing.assert_array_equal(vectors[:, 0], [float(query_id) for query_id in qids])
                return {query_id: {"11": float(query_id), "21": 6.0 - float(query_id)} for query_id in qids}

            local_stage1, local_hybrid = [], []
            with (
                patch.object(pipeline, "_stage1_query_shard", side_effect=partition),
                patch.object(pipeline, "_load_retrieval_runtime", return_value=SimpleNamespace()) as loader,
                patch.object(pipeline, "_release_retrieval_runtime"),
                patch.object(pipeline, "encode_queries_online", side_effect=encode) as encoder,
                patch.object(pipeline, "retrieve_sparse_topics_from_lucene", side_effect=sparse),
                patch.object(pipeline, "retrieve_dense_from_corpus", side_effect=dense),
                patch.object(pipeline, "_gather_run", return_value=sparse_run(list(reversed(ids)))),
                patch.object(pipeline, "_progress", side_effect=fixtures.ProgressRecorder()),
            ):
                for rank in range(world_size):
                    context = pipeline.DistributedContext(rank, rank, world_size)
                    local, encoded, _, _ = pipeline._run_sparse_concat_stage1(args, context, pipeline.StageTimer())
                    local_stage1.append(local)
                    timer = pipeline.StageTimer()
                    _, _, hybrid = pipeline.run_stage2(args, local, encoded_queries=encoded, timing=timer)
                    self.assertEqual(timer.components["query_encoding"], 0.0)
                    local_hybrid.append(hybrid)
                self.assertEqual(encoder.call_count, min(world_size, len(ids)))
                self.assertEqual(loader.call_count, min(world_size, len(ids)))
            self.assertCountEqual(seen, ids)
            return pipeline._merge_run_shards(local_stage1), pipeline._merge_run_shards(local_hybrid)

        baseline = execute(1)
        evaluation = pipeline.RecallEvaluation(
            [1, 5], {query_id: {"11"} for query_id in ids}, {}, None, len(ids),
        )
        for world_size in (2, 4, 8):
            with self.subTest(world_size=world_size):
                actual = execute(world_size)
                self.assertEqual(actual, baseline)
                self.assertEqual(
                    pipeline.evaluate_recall(actual[1], evaluation, stage="stage2", variant="hybrid"),
                    pipeline.evaluate_recall(baseline[1], evaluation, stage="stage2", variant="hybrid"),
                )


if __name__ == "__main__":
    unittest.main()
