"""Legacy Adaptive root shards normalize without changing encoding fingerprints."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from test_freeret_dense import temporary_directory
from test_adaptive_sparse_exact import artifacts
from test_adaptive_search import corpus
import adaptive_sparse_index as layout
import adaptive_sparse_index_batch as batch
import adaptive_views as views
import search_adaptive_views as search


def flatten(root):
    for shard in (root / "collection").glob("corpus_*.jsonl"):
        shard.rename(root / shard.name)
    index = root / "collection/index"
    if index.exists():
        index.rename(root / "index")
    (root / "collection").rmdir()


def files(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def lucene(command, *, check):
    """Exercise the real staging/publish code without requiring Java on the test host."""
    assert check
    source = Path(command[command.index("--input") + 1])
    output = Path(command[command.index("--index") + 1])
    assert sorted(path.name for path in source.iterdir()) == ["corpus_0.jsonl"]
    assert json.loads((source / "corpus_0.jsonl").read_text(encoding="utf-8"))["vector"]
    output.mkdir()
    (output / "segments_1").write_text("fixture", encoding="utf-8")


class LayoutTests(unittest.TestCase):
    def fixture(self, directory, **changes):
        env, records, loader = artifacts(directory, DRY_RUN="false", **changes)
        return env, records[0][0], loader

    def test_root_shards_all_task_groups_and_modes_publish_canonical_indexes(self):
        for group in ("ir", "tbpr", "fused"):
            for mode in ("ordinary", "constraints", "full", "overlap"):
                with self.subTest(group=group, mode=mode), temporary_directory() as directory:
                    env, records, loader = artifacts(directory, TASK_GROUP=group, DRY_RUN="false",
                        USE_FREERET_DENSE=str(mode in {"constraints", "full"}).lower(),
                        USE_FREERET_PROMPT_CONSTRAINTS=str(mode == "constraints").lower(),
                        USE_FREERET_FULL_PROMPT=str(mode == "full").lower(),
                        USE_VIEW_OVERLAP_SELECTION=str(mode == "overlap").lower())
                    snapshots = {}
                    for root, _, _ in records:
                        flatten(root)
                        (root / "index").mkdir()
                        (root / "index/segments_legacy").write_text("legacy", encoding="utf-8")
                        (root / "unrelated.json").write_text("{}", encoding="utf-8")
                        snapshots[root] = files(root)
                    settings = batch.Settings.from_environment(group, env, exact_encoding=True, overlap_only=mode == "overlap")
                    with patch("qwen_sparse_index.subprocess.run", side_effect=lucene) as process, redirect_stdout(io.StringIO()):
                        self.assertEqual(batch.run(settings, identity_loader=loader), 0)
                    self.assertEqual(process.call_count, len(records))
                    for root, _, _ in records:
                        self.assertTrue((root / "collection/index/segments_1").is_file())
                        self.assertEqual((root / "collection/corpus_0.jsonl").read_bytes(), (root / "corpus_0.jsonl").read_bytes())
                        for name, data in snapshots[root].items():
                            self.assertEqual((root / name).read_bytes(), data)
                        self.assertFalse(list(root.glob(".qwen-index-*")))
                    with redirect_stdout(io.StringIO()), patch("qwen_sparse_index.subprocess.run") as process:
                        self.assertEqual(batch.run(settings, identity_loader=loader), 0)
                    process.assert_not_called()

    def test_dry_run_is_read_only_and_reports_copy_even_with_legacy_index(self):
        with temporary_directory() as directory:
            env, root, loader = self.fixture(directory)
            flatten(root)
            (root / "index").mkdir()
            (root / "index/segments_legacy").touch()
            before = files(Path(directory))
            settings = batch.Settings.from_environment("tbpr", dict(env, DRY_RUN="true"), exact_encoding=True)
            with redirect_stdout(io.StringIO()) as output, patch("qwen_sparse_index.subprocess.run") as process:
                self.assertEqual(batch.run(settings, identity_loader=loader), 0)
            process.assert_not_called()
            self.assertIn("copy 1 root shard", output.getvalue())
            self.assertIn("planned=1", output.getvalue())
            self.assertEqual(files(Path(directory)), before)
            self.assertFalse((root / "collection").exists())

    def test_partial_matching_copies_resume_and_complete_collection_wins(self):
        with temporary_directory() as directory:
            _, root, _ = self.fixture(directory)
            flatten(root)
            manifest = json.loads((root / views.MANIFEST).read_text(encoding="utf-8"))
            manifest["num_shards"] = 2
            views.atomic_json(root / views.MANIFEST, manifest)
            (root / "corpus_1.jsonl").write_bytes(b'{"id":"b","vector":{"second":20}}\n')
            collection = root / "collection"
            collection.mkdir()
            original = collection / "corpus_0.jsonl"
            original.write_bytes((root / original.name).read_bytes())
            mtime = original.stat().st_mtime_ns
            with redirect_stdout(io.StringIO()):
                layout.normalize_collection(root)
            self.assertEqual(original.stat().st_mtime_ns, mtime)
            self.assertEqual((collection / "corpus_1.jsonl").read_bytes(), (root / "corpus_1.jsonl").read_bytes())
            # Old root files can remain after a later canonical re-encoding.
            (root / "corpus_0.jsonl").write_bytes(b"old root content")
            plan = layout.collection_plan(root)
            self.assertFalse(plan.copies)
            self.assertTrue(all(path.parent == collection for path in plan.documents))

    def test_partial_conflict_missing_and_extra_shards_fail_without_mutation(self):
        for problem in ("conflict", "missing", "extra", "filename"):
            with self.subTest(problem=problem), temporary_directory() as directory:
                _, root, _ = self.fixture(directory)
                flatten(root)
                manifest = json.loads((root / views.MANIFEST).read_text(encoding="utf-8"))
                manifest["num_shards"] = 2
                views.atomic_json(root / views.MANIFEST, manifest)
                if problem != "missing":
                    (root / "corpus_1.jsonl").write_bytes(b"second")
                collection = root / "collection"
                collection.mkdir()
                if problem == "conflict":
                    (collection / "corpus_0.jsonl").write_bytes(b"wrong")
                elif problem == "filename":
                    (collection / "corpus_other.jsonl").write_bytes(b"wrong name")
                elif problem == "extra":
                    for i in range(3):
                        (collection / f"corpus_{i}.jsonl").write_bytes(b"extra")
                before = files(root)
                with self.assertRaisesRegex(ValueError, "Conflicting|shard count mismatch"):
                    layout.normalize_collection(root)
                self.assertEqual(files(root), before)

    def test_failed_copy_does_not_publish_partial_shard(self):
        with temporary_directory() as directory:
            _, root, _ = self.fixture(directory)
            flatten(root)
            def fail_copy(source, target):
                target.write(b"partial")
                raise OSError("interrupted copy")
            before = files(root)
            with patch.object(layout.shutil, "copyfileobj", side_effect=fail_copy), self.assertRaisesRegex(OSError, "interrupted"):
                layout.normalize_collection(root)
            self.assertEqual(files(root), before)
            self.assertFalse(list((root / "collection").iterdir()))

    def test_failed_index_keeps_originals_and_never_publishes_invalid_index(self):
        for failure in ("exception", "no_segments"):
            with self.subTest(failure=failure), temporary_directory() as directory:
                _, root, _ = self.fixture(directory)
                flatten(root)
                original = (root / "corpus_0.jsonl").read_bytes()
                effect = subprocess.CalledProcessError(9, "lucene") if failure == "exception" else None
                with patch("qwen_sparse_index.subprocess.run", side_effect=effect), redirect_stdout(io.StringIO()):
                    with self.assertRaises((subprocess.CalledProcessError, ValueError)):
                        layout.build_sparse_index(root)
                self.assertFalse((root / "collection/index").exists())
                self.assertEqual((root / "corpus_0.jsonl").read_bytes(), original)
                self.assertEqual((root / "collection/corpus_0.jsonl").read_bytes(), original)
                self.assertFalse(list(root.glob(".qwen-index-*")))

    def test_incomplete_canonical_index_blocks_copy_and_legacy_fallback(self):
        with temporary_directory() as directory:
            _, root, _ = self.fixture(directory)
            flatten(root)
            (root / "index").mkdir()
            (root / "index/segments_legacy").touch()
            (root / "collection/index").mkdir(parents=True)
            before = files(root)
            with self.assertRaisesRegex(ValueError, "Incomplete existing index"):
                layout.build_sparse_index(root)
            with self.assertRaisesRegex(ValueError, "Incomplete existing index"):
                layout.resolve_sparse_index(root)
            self.assertEqual(files(root), before)

    def test_all_jobs_preflight_before_any_layout_migration(self):
        with temporary_directory() as directory:
            env, records, loader = artifacts(directory, TASK_GROUP="ir", DRY_RUN="false")
            for root, _, _ in records:
                flatten(root)
            (records[-1][0] / "corpus_0.jsonl").unlink()
            before = files(Path(directory))
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "shard count mismatch"), \
                    patch("qwen_sparse_index.subprocess.run") as process:
                batch.run(batch.Settings.from_environment("ir", env, exact_encoding=True), identity_loader=loader)
            process.assert_not_called()
            self.assertEqual(files(Path(directory)), before)
            self.assertFalse((records[0][0] / "collection").exists())

    def test_direct_cli_and_search_readonly_legacy_then_prefer_new_index(self):
        with temporary_directory() as directory:
            fixture = corpus(directory, mode="full")
            root = fixture.roots[1]
            flatten(root)
            before = files(root)
            loader = lambda *a, **k: fixture.identity
            ex = search.prepare(copy.deepcopy(fixture.args), identity_loader=loader)
            self.assertEqual(ex.index, root / "index")
            self.assertEqual(files(root), before)
            args = ["adaptive_views.py", "index", "--sparse-root", str(root)]
            with patch.object(sys, "argv", args + ["--dry-run"]), redirect_stdout(io.StringIO()):
                views.main()
            self.assertEqual(files(root), before)
            with patch.object(sys, "argv", args), patch("qwen_sparse_index.subprocess.run", side_effect=lucene), \
                    redirect_stdout(io.StringIO()):
                views.main()
            ex = search.prepare(copy.deepcopy(fixture.args), identity_loader=loader)
            self.assertEqual(ex.index, root / "collection/index")
            self.assertTrue((root / "index/segments_1").exists())
            before = files(root)
            with patch.object(sys, "argv", args), patch("qwen_sparse_index.subprocess.run") as process, \
                    redirect_stdout(io.StringIO()):
                views.main()
            process.assert_not_called()
            self.assertEqual(files(root), before)

    def test_search_missing_index_has_adaptive_command_without_writing(self):
        with temporary_directory() as directory:
            fixture = corpus(directory)
            root = fixture.roots[1]
            (root / "collection/index/segments_1").unlink()
            (root / "collection/index").rmdir()
            flatten(root)
            before = files(root)
            with self.assertRaisesRegex(ValueError, "src/adaptive_views.py index --sparse-root"):
                search.prepare(fixture.args, identity_loader=lambda *a, **k: fixture.identity)
            self.assertEqual(files(root), before)
            self.assertFalse((root / "collection").exists())

    def test_non_adaptive_flat_artifacts_are_not_migrated(self):
        with temporary_directory() as directory:
            root = Path(directory)
            (root / "corpus_0.jsonl").write_bytes(b"ordinary FreeRet output")
            before = files(root)
            with self.assertRaises(ValueError):
                layout.normalize_collection(root)
            self.assertEqual(files(root), before)
            self.assertFalse((root / "collection").exists())

    def test_concurrent_destination_is_not_overwritten(self):
        with temporary_directory() as directory:
            _, root, _ = self.fixture(directory)
            flatten(root)
            def publish_race(source, destination):
                destination.write_bytes(b"concurrent content")
                raise FileExistsError(str(destination))
            with patch.object(layout.os, "link", side_effect=publish_race), \
                    self.assertRaisesRegex(ValueError, "Conflicting adaptive corpus shard"):
                layout.normalize_collection(root)
            self.assertEqual((root / "collection/corpus_0.jsonl").read_bytes(), b"concurrent content")
            self.assertTrue((root / "corpus_0.jsonl").exists())
            self.assertFalse(list((root / "collection").glob(".adaptive-shard-*")))

    def test_linked_shards_are_rejected_without_creating_collection(self):
        with temporary_directory() as directory:
            _, root, _ = self.fixture(directory)
            flatten(root)
            original = root / "corpus_0.jsonl"
            saved = root / "original.jsonl"
            original.rename(saved)
            try:
                original.symlink_to(saved)
            except OSError as exc:
                self.skipTest(f"Host does not allow symlinks: {exc}")
            with self.assertRaisesRegex(ValueError, "unlinked"):
                layout.normalize_collection(root)
            self.assertFalse((root / "collection").exists())


if __name__ == "__main__":
    unittest.main()
