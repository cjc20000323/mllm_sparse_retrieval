"""Real filesystem tests; the Java subprocess is replaced by a recording builder."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import qwen_sparse_index as builder
from qwen_vl_paths import artifact_root, resolve_sparse_index


class SparseIndexTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="qwen index contract ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.collection = self.root / "collection"
        self.collection.mkdir()
        for rank in range(2):
            (self.collection / f"corpus_{rank}.jsonl").write_text(
                json.dumps(dict(id=str(rank), contents="", vector={"word": rank + 1})) + "\n", encoding="utf-8")
        (self.collection / "notes.json").write_text("not a document")
        (self.root / "qwen25_prompt_manifest.json").write_text("{}")
        self.calls = []

    def fake_build(self, command, *, check):
        self.assertTrue(check)
        self.assertEqual(command[:4], [sys.executable, "-B", "-m", "pyserini.index.lucene"])
        self.assertIn("--impact", command)
        self.assertIn("--pretokenized", command)
        source = Path(command[command.index("--input") + 1])
        output = Path(command[command.index("--index") + 1])
        self.assertFalse(source.is_relative_to(self.collection))
        self.assertFalse(output.is_relative_to(self.collection))
        self.assertEqual(sorted(item.name for item in source.rglob("*")), ["corpus_0.jsonl", "corpus_1.jsonl"])
        for document in source.iterdir():
            self.assertEqual(document.read_bytes(), (self.collection / document.name).read_bytes())
        self.calls.append((source, output))
        output.mkdir()
        (output / "segments_1").write_bytes(b"new index")

    def assert_clean(self):
        self.assertFalse(list(self.root.glob(".qwen-index-*")))

    def test_build_publishes_nested_index_only_after_success_and_keeps_legacy(self):
        legacy = self.root / "index"
        legacy.mkdir()
        (legacy / "segments_1").write_bytes(b"old index")
        with patch.object(builder.subprocess, "run", side_effect=self.fake_build):
            target = builder.build_sparse_index(self.root, threads=3)
        self.assertEqual(target, self.collection / "index")
        self.assertEqual((target / "segments_1").read_bytes(), b"new index")
        self.assertEqual((legacy / "segments_1").read_bytes(), b"old index")
        self.assertEqual(resolve_sparse_index(self.root), target)
        self.assert_clean()

    def test_hardlink_failure_falls_back_to_copy(self):
        with patch.object(builder.os, "link", side_effect=OSError("no hardlinks")), \
                patch.object(builder.shutil, "copy2", wraps=builder.shutil.copy2) as copy, \
                patch.object(builder.subprocess, "run", side_effect=self.fake_build):
            builder.build_sparse_index(self.collection)
        self.assertEqual(copy.call_count, 2)
        self.assert_clean()

    def test_subprocess_failure_never_publishes_partial_index(self):
        def fail(command, **kwargs):
            self.fake_build(command, **kwargs)
            raise subprocess.CalledProcessError(17, command)
        with patch.object(builder.subprocess, "run", side_effect=fail):
            with self.assertRaises(subprocess.CalledProcessError):
                builder.build_sparse_index(self.root)
        self.assertFalse((self.collection / "index").exists())
        self.assert_clean()

    def test_cli_preserves_indexer_exit_status(self):
        with patch.object(sys, "argv", ["qwen_sparse_index.py", "--sparse-root", str(self.root)]), \
                patch.object(builder, "build_sparse_index", side_effect=subprocess.CalledProcessError(17, ["indexer"])):
            with self.assertRaises(SystemExit) as failure:
                builder.main()
        self.assertEqual(failure.exception.code, 17)

    def test_success_without_segments_is_rejected_and_cleaned(self):
        with patch.object(builder.subprocess, "run"):
            with self.assertRaisesRegex(ValueError, "produced no segments"):
                builder.build_sparse_index(self.root)
        self.assertFalse((self.collection / "index").exists())
        self.assert_clean()

    def test_existing_nested_index_is_never_overwritten(self):
        target = self.collection / "index"
        target.mkdir()
        (target / "keep").write_text("keep")
        with patch.object(builder.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "already exists"):
                builder.build_sparse_index(self.root)
        run.assert_not_called()
        self.assertEqual((target / "keep").read_text(), "keep")
        self.assert_clean()

    def test_missing_shards_and_linked_collection_fail_before_build(self):
        with patch.object(builder.subprocess, "run") as run:
            for document in self.collection.glob("corpus_*.jsonl"):
                document.unlink()
            with self.assertRaisesRegex(ValueError, "No corpus_"):
                builder.build_sparse_index(self.root)
            real_resolve = Path.resolve
            def resolve(path, *args, **kwargs):
                return self.root / "elsewhere" if path == self.collection else real_resolve(path, *args, **kwargs)
            with patch.object(Path, "resolve", resolve), self.assertRaisesRegex(ValueError, "linked Qwen"):
                builder.build_sparse_index(self.root)
        run.assert_not_called()
        self.assert_clean()

    def test_old_new_aliases_preference_and_incomplete_new_index(self):
        old, new = self.root / "index", self.collection / "index"
        old.mkdir()
        (old / "segments_1").touch()
        for path in (self.root, self.collection, old, str(self.collection) + "/"):
            self.assertEqual(artifact_root(path), self.root)
            self.assertEqual(resolve_sparse_index(path), old)
        new.mkdir()
        for path in (self.root, self.collection, new):
            self.assertEqual(artifact_root(path), self.root)
            with self.assertRaisesRegex(ValueError, "no segments"):
                resolve_sparse_index(path)
        # An explicit legacy path remains usable even when a new build is broken.
        self.assertEqual(resolve_sparse_index(old), old)
        (new / "segments_2").touch()
        for path in (self.root, self.collection, new):
            self.assertEqual(resolve_sparse_index(path), new)

    def test_three_stage_resolver_and_validation_share_rules_without_changing_llava(self):
        tree = ast.parse((SRC / "three_stage_retrieval_pipeline.py").read_text(encoding="utf-8"))
        resolver = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "_resolve_lucene_index")
        namespace = {"Path": Path}
        exec(compile(ast.Module(body=[resolver], type_ignores=[]), "three-stage resolver", "exec"), namespace)
        resolve = namespace["_resolve_lucene_index"]
        old, new = self.root / "index", self.collection / "index"
        old.mkdir()
        (old / "segments_1").touch()
        self.assertEqual(resolve(self.root), old)
        new.mkdir()
        with self.assertRaisesRegex(ValueError, "no segments"):
            resolve(self.root)
        (new / "segments_2").touch()
        self.assertEqual(resolve(self.root), new)
        self.assertEqual(resolve(new), new)
        # The unrelated LLaVA layout still resolves its usual sibling index.
        (self.root / "qwen25_prompt_manifest.json").unlink()
        self.assertEqual(resolve(self.root), old)
        self.assertEqual(resolve(old), old)


if __name__ == "__main__":
    unittest.main()
