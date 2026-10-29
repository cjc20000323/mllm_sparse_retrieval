"""Encoding restarts replace artifacts without keeping stale shards or indexes."""
from __future__ import annotations

import io
import json
import pickle
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import test_qwen25_retrieval as shared

pipeline = shared.pipeline


class QwenEncodeOverwriteTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="qwen encode restart ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.data = shared.Data(dense_output_dir=str(self.root / "dense"),
                                sparse_output_dir=str(self.root / "sparse"))
        self.args = SimpleNamespace(model_name_or_path="Qwen-test", eol_type="disassembleeol_separate",
                                    calculate_type="concat")
        self.training = SimpleNamespace(task_type="ir", encode_type="text")
        self.model = SimpleNamespace(config=SimpleNamespace(model_type="qwen2_5_vl", hidden_size=2))
        self.metadata = dict(fingerprint="fixture", config=dict(version=pipeline.PROMPT_VERSION), kind="text")
        self.records = [dict(id="a"), dict(id="b")]
        self.generation = 1
        self.rank, self.world, self.distributed = 0, 1, False
        self.last_packet = None
        self.events = []
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(redirect_stderr(io.StringIO()))
        stack.enter_context(patch.object(pipeline, "load_runtime", side_effect=lambda *a: (
            self.model, None, self.rank, self.world)))
        stack.enter_context(patch.object(pipeline, "build_dataset", return_value=object()))
        stack.enter_context(patch.object(pipeline, "dataset_records", side_effect=lambda *a: self.records))
        stack.enter_context(patch.object(pipeline, "prompt_metadata", side_effect=lambda *a, **k: self.metadata))
        self.encoder = stack.enter_context(patch.object(pipeline, "NativeQwenEncoder", return_value=SimpleNamespace(last_prompts=[])))
        self.batches = stack.enter_context(patch.object(pipeline, "encode_batches", side_effect=self.encode_batches))
        stack.enter_context(patch.object(pipeline.dist, "is_initialized", side_effect=lambda: self.distributed))
        stack.enter_context(patch.object(pipeline.dist, "broadcast_object_list", side_effect=self.broadcast))
        stack.enter_context(patch.object(pipeline.dist, "barrier", side_effect=self.barrier))
        self.dense, self.sparse = pipeline.artifact_dirs(self.args, self.data, self.training)

    def encode_batches(self, encoder, records, data):
        yield records, np.full((len(records), 2), self.generation, dtype="float32"), [
            {"word": self.generation} for _ in records]

    def broadcast(self, packet, src):
        self.assertEqual(src, 0)
        self.events.append(("broadcast", self.rank))
        if self.rank == 0:
            self.last_packet = list(packet)
        else:
            packet[:] = self.last_packet

    def barrier(self):
        self.events.append(("barrier", self.rank))
        if self.rank == 0:
            for root in (self.dense, self.sparse):
                self.assertFalse((root / pipeline.manifest_name(self.metadata)).exists())

    def run_encode(self):
        pipeline.run_encode(self.args, self.data, self.training)

    def read_dense(self, rank=0):
        path = self.dense / ("query.pkl" if self.data.encode_is_query else f"corpus_{rank}.pkl")
        with path.open("rb") as stream:
            return pickle.load(stream)

    def check_complete(self):
        for root in (self.dense, self.sparse):
            manifest = pipeline.validate_manifest(root, self.metadata)
            self.assertEqual(manifest["num_shards"], self.world)

    def test_repeated_encoding_replaces_both_outputs_and_backs_up_old_index(self):
        self.run_encode()
        index = self.sparse / "index"
        index.mkdir()
        (index / "segments_1").write_bytes(b"old Lucene index")
        self.generation = 2
        self.records = [dict(id="new")]
        self.run_encode()
        dense, ids = self.read_dense()
        self.assertEqual(ids, ["new"])
        np.testing.assert_array_equal(dense, [[2, 2]])
        docs = [json.loads(line) for line in (self.sparse / "collection/corpus_0.jsonl").read_text().splitlines()]
        self.assertEqual(docs, [dict(id="new", contents="", vector={"word": 2})])
        self.assertFalse(index.exists())
        backups = list(self.sparse.glob("index.stale_*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / "segments_1").read_bytes(), b"old Lucene index")
        index.mkdir()  # A subsequent sparse_index job can use the normal output path.
        self.check_complete()

    def test_fewer_workers_remove_obsolete_shards_only_from_current_leaf(self):
        self.run_encode()
        for rank in (1, 2, 7):
            (self.dense / f"corpus_{rank}.pkl").write_bytes(b"stale shard")
            (self.sparse / f"collection/corpus_{rank}.jsonl").write_text("stale shard")
        (self.dense / "notes.txt").write_text("keep")
        (self.sparse / "query.tsv").write_text("keep other mode")
        other = self.dense.parent / "another_configuration"
        other.mkdir()
        (other / "corpus_0.pkl").write_bytes(b"keep other configuration")
        self.run_encode()
        self.assertEqual([path.name for path in self.dense.glob("corpus*.pkl")], ["corpus_0.pkl"])
        self.assertEqual([path.name for path in (self.sparse / "collection").glob("corpus*.jsonl")], ["corpus_0.jsonl"])
        self.assertEqual((self.dense / "notes.txt").read_text(), "keep")
        self.assertEqual((self.sparse / "query.tsv").read_text(), "keep other mode")
        self.assertEqual((other / "corpus_0.pkl").read_bytes(), b"keep other configuration")
        self.check_complete()

    def test_both_index_layouts_are_backed_up_outside_collection(self):
        self.run_encode()
        for relative in ("index", "collection/index"):
            index = self.sparse / relative
            index.mkdir()
            (index / "segments_1").write_text(relative)
        self.run_encode()
        self.assertFalse((self.sparse / "index").exists())
        self.assertFalse((self.sparse / "collection/index").exists())
        backups = list(self.sparse.glob("index.stale_*"))
        self.assertEqual({(path / "segments_1").read_text() for path in backups}, {"index", "collection/index"})
        self.assertFalse(list((self.sparse / "collection").glob("index*")))
        with self.assertRaisesRegex(ValueError, "Missing Qwen Lucene index"):
            pipeline.resolve_sparse_index(self.sparse)
        self.check_complete()

    def test_failed_restart_has_no_completion_manifest_and_can_be_retried(self):
        self.run_encode()
        def failing_batches(encoder, records, data):
            yield from self.encode_batches(encoder, records[:1], data)
            raise RuntimeError("interrupted encoder")
        self.batches.side_effect = failing_batches
        with self.assertRaisesRegex(RuntimeError, "interrupted encoder"):
            self.run_encode()
        self.assertTrue((self.sparse / "collection/corpus_0.jsonl").exists())
        for root in (self.dense, self.sparse):
            with self.assertRaisesRegex(ValueError, "Missing"):
                pipeline.validate_manifest(root, self.metadata)
        self.batches.side_effect = self.encode_batches
        self.run_encode()
        self.check_complete()

    def test_query_reencoding_replaces_query_files_without_retiring_corpus_index(self):
        self.data.encode_is_query = True
        self.run_encode()
        index = self.sparse / "index"
        index.mkdir()
        (index / "segments_1").write_bytes(b"keep corpus index")
        nested = self.sparse / "collection/index"
        nested.mkdir(parents=True)
        (nested / "segments_2").write_bytes(b"keep nested index")
        self.generation = 3
        self.records = [dict(id="q")]
        self.run_encode()
        dense, ids = self.read_dense()
        self.assertEqual(ids, ["q"])
        np.testing.assert_array_equal(dense, [[3, 3]])
        self.assertEqual((self.sparse / "query.tsv").read_text(), "q\tword word word\n")
        self.assertTrue((index / "segments_1").exists())
        self.assertTrue((nested / "segments_2").exists())
        self.assertFalse(list(self.sparse.glob("index.stale_*")))
        self.check_complete()

    def test_only_rank_zero_resets_outputs_and_completion_follows_barrier(self):
        self.distributed, self.world = True, 2
        with patch.object(pipeline, "prepare_encode_outputs", wraps=pipeline.prepare_encode_outputs) as reset:
            self.run_encode()
            shard0 = (self.dense / "corpus_0.pkl").read_bytes()
            manifest = (self.dense / pipeline.manifest_name(self.metadata)).read_bytes()
            self.rank = 1
            self.run_encode()
            self.assertEqual(reset.call_count, 1)
        self.assertEqual((self.dense / "corpus_0.pkl").read_bytes(), shard0)
        self.assertEqual((self.dense / pipeline.manifest_name(self.metadata)).read_bytes(), manifest)
        self.assertEqual(self.read_dense(0)[1], ["a"])
        self.assertEqual(self.read_dense(1)[1], ["b"])
        self.assertEqual(self.events, [("broadcast", 0), ("barrier", 0), ("broadcast", 1), ("barrier", 1)])
        self.check_complete()

    def test_cleanup_io_error_is_broadcast_and_no_worker_starts_encoding(self):
        self.distributed, self.world = True, 2
        with patch.object(pipeline, "prepare_encode_outputs", side_effect=PermissionError("permission denied")) as reset:
            for rank in (0, 1):
                self.rank = rank
                with self.assertRaisesRegex(ValueError, "permission denied"):
                    self.run_encode()
            self.assertEqual(reset.call_count, 1)
        self.encoder.assert_not_called()
        self.assertEqual(self.events, [("broadcast", 0), ("broadcast", 1)])

    def test_linked_collection_is_rejected_before_touching_existing_artifacts(self):
        self.run_encode()
        before = (self.dense / "corpus_0.pkl").read_bytes()
        real_resolve = Path.resolve
        def resolve(path, *args, **kwargs):
            if path == self.sparse / "collection":
                return self.root / "another_dataset"
            return real_resolve(path, *args, **kwargs)
        with patch.object(Path, "resolve", resolve):
            with self.assertRaisesRegex(ValueError, "linked Qwen artifact"):
                self.run_encode()
        self.assertEqual((self.dense / "corpus_0.pkl").read_bytes(), before)
        self.check_complete()


if __name__ == "__main__":
    unittest.main()
