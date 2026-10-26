"""CPU-only regression tests; no model, dataset, DeepSpeed or Java required."""

from __future__ import annotations

import ast
from contextlib import contextmanager, redirect_stdout
import importlib.util
from io import StringIO
import math
import os
from pathlib import Path
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "compute_retrieval_similarity", PROJECT_ROOT / "src/compute_retrieval_similarity.py"
)
similarity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(similarity)


def run_record(**scores):
    return {"docs": scores}


def save_run(directory, condition, dense=None, sparse=None):
    path = Path(directory) / f"test_runs_{condition}.pkl"
    payload = {
        "dense_run": dense if dense is not None else {"q": run_record(a=3, b=2, c=1)},
        "sparse_run": sparse if sparse is not None else {"q": run_record(a=3, b=2, c=1)},
        "metadata": {"eol_type": "disassembleeol_separate", "calculate_type": "concat"},
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


@contextmanager
def capture_workbook():
    """Exercise the real dataframe export without an optional Excel engine."""
    sheets = {}
    def to_excel(frame, writer, *, sheet_name, index):
        assert index is False
        sheets[sheet_name] = frame.copy()
    with patch.object(similarity.pd, "ExcelWriter") as writer, patch.object(
            similarity.pd.DataFrame, "to_excel", to_excel):
        yield sheets, writer


def load_search_helpers():
    """Load just the production helpers, avoiding GPU/Java imports in search."""
    tree = ast.parse((PROJECT_ROOT / "src/search_similarity.py").read_text(encoding="utf-8"))
    names = {
        "RETRIEVAL_SIMILARITY_STATS_DIR_NAME", "RETRIEVAL_SIMILARITY_CONFIGURATIONS",
        "_similarity_stats_configuration", "_gather_test_run_for_similarity",
        "_save_test_runs_for_similarity",
    }
    nodes = [node for node in tree.body if (
        isinstance(node, ast.FunctionDef) and node.name in names
    ) or (
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)
    )]
    namespace = {"os": os, "pickle": pickle, "dist": Mock(),
                 "ENABLE_RETRIEVAL_SIMILARITY_STATS": True}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PROJECT_ROOT / "src/search_similarity.py"), "exec"), namespace)
    return namespace


class SimilarityReportTests(unittest.TestCase):
    def test_single_conditions_have_no_fabricated_comparison_rows(self):
        for condition in ("no_manual", "manual"):
            with self.subTest(condition=condition), tempfile.TemporaryDirectory() as directory:
                save_run(directory, condition)
                with capture_workbook() as (sheets, writer):
                    output = similarity.compute_from_directory(directory, allow_partial=True)
                self.assertEqual(output, str(Path(directory) / "dense_sparse_similarity_test.xlsx"))
                writer.assert_called_once_with(output)
                self.assertEqual(set(sheets), {"summary", "per_query", "metadata"})
                self.assertEqual(sheets["summary"]["condition"].tolist(), [condition])
                self.assertEqual(sheets["per_query"]["condition"].tolist(), [condition])
                metadata = dict(zip(sheets["metadata"]["item"], sheets["metadata"]["value"]))
                self.assertEqual(metadata["top_k"], "10,20,50,100")
                self.assertEqual(metadata["rbo_p"], 0.9)
                self.assertEqual(metadata[f"{condition}_calculate_type"], "concat")
                other = "manual" if condition == "no_manual" else "no_manual"
                self.assertNotIn(f"{other}_run_file", metadata)
                for depth in similarity.TOP_KS:
                    for metric in ("Overlap", "Spearman", "RBO"):
                        self.assertAlmostEqual(sheets["summary"].iloc[0][f"{metric}@{depth}"], 1.0)

    def test_second_condition_updates_same_workbook_with_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            save_run(directory, "no_manual", sparse={"q": run_record(a=1, b=2, c=3)})
            with capture_workbook() as (sheets, _):
                first = similarity.compute_from_directory(directory, allow_partial=True)
                self.assertEqual(len(sheets["summary"]), 1)
                save_run(directory, "manual")
                second = similarity.compute_from_directory(directory, allow_partial=True)
            self.assertEqual(first, second)
            self.assertEqual(sheets["summary"]["condition"].tolist(),
                             ["manual", "no_manual", "manual - no_manual"])
            self.assertEqual(sheets["per_query"]["condition"].tolist(), ["manual", "no_manual"])
            for depth in similarity.TOP_KS:
                self.assertAlmostEqual(sheets["summary"].iloc[2][f"Spearman@{depth}"], 2.0)

    def test_legacy_waits_for_both_and_preserves_paired_api(self):
        with tempfile.TemporaryDirectory() as directory, capture_workbook() as (sheets, writer):
            manual = save_run(directory, "manual")
            self.assertIsNone(similarity.compute_from_directory(directory))
            writer.assert_not_called()
            no_manual = save_run(directory, "no_manual")
            expected = str(Path(directory) / "dense_sparse_similarity_test.xlsx")
            self.assertEqual(similarity.compute_from_directory(directory), expected)
            custom = Path(directory) / "nested/custom.xlsx"
            self.assertEqual(similarity.compute_similarity(manual, no_manual, custom), str(custom))
            self.assertTrue(custom.parent.is_dir())
            self.assertEqual(sheets["summary"]["condition"].tolist(),
                             ["manual", "no_manual", "manual - no_manual"])

    def test_missing_runs_do_not_open_workbook(self):
        with tempfile.TemporaryDirectory() as directory, capture_workbook() as (_, writer):
            for allow_partial in (False, True):
                self.assertIsNone(similarity.compute_from_directory(directory, allow_partial=allow_partial))
            writer.assert_not_called()

    def test_nested_multiview_runs_are_averaged_and_normalized(self):
        run = {1: [{"1": run_record(a=4, b=2)}, [run_record(a=2, c=1)]],
               "q2": {"view": run_record(x=5)}, "bad": None}
        self.assertEqual(similarity.normalize_run(run),
                         {"1": {"a": 3.0, "b": 2.0, "c": 1.0}, "q2": {"x": 5.0}})
        with tempfile.TemporaryDirectory() as directory, capture_workbook() as (sheets, _):
            save_run(directory, "no_manual", dense=run,
                     sparse={"1": run_record(a=3, b=2, c=1), "q2": run_record(x=5),
                             "sparse_only": run_record(x=1)})
            similarity.compute_from_directory(directory, allow_partial=True)
            self.assertEqual(sheets["per_query"]["query_id"].tolist(), ["1", "q2"])
            self.assertEqual(sheets["summary"].iloc[0]["queries_with_dense_and_sparse"], 2)

    def test_degenerate_spearman_stays_nan(self):
        examples = [({"a": 2}, {"a": 1}), ({"a": 2}, {"b": 1}),
                    ({"a": 2, "b": 2}, {"a": 1, "b": 0})]
        for dense, sparse in examples:
            with self.subTest(dense=dense, sparse=sparse):
                metrics = similarity._query_metrics(dense, sparse, 10, 0.9)
                self.assertTrue(math.isnan(metrics["spearman"]))
        summary, _ = similarity._calculate_condition("no_manual", {"q": {"a": 1}},
                                                      {"q": {"a": 1}}, 0.9)
        self.assertTrue(math.isnan(summary["Spearman@10"]))
        self.assertEqual(summary["ValidSpearmanQueries@10"], 0)

    def test_overlap_and_rbo_formulas_are_unchanged(self):
        metrics = similarity._query_metrics({"a": 3, "b": 2}, {"a": 3, "c": 2}, 10, 0.9)
        self.assertEqual(metrics["overlap"], 0.5)
        self.assertAlmostEqual(metrics["rbo"], 0.55)
        for p in (0, 1, -1):
            with self.assertRaises(ValueError):
                similarity.rank_biased_overlap(["a"], ["a"], 10, p)

    def test_bad_pickle_payload_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pkl"
            for payload in ([], {"dense_run": {}, "sparse_run": []}):
                with path.open("wb") as handle:
                    pickle.dump(payload, handle)
                with self.assertRaises(ValueError):
                    similarity.load_saved_runs(path)

    def test_cli_partial_flag_and_custom_settings(self):
        with tempfile.TemporaryDirectory() as directory, capture_workbook() as (sheets, _):
            save_run(directory, "no_manual")
            output = str(Path(directory) / "custom.xlsx")
            argv = ["compute_retrieval_similarity.py", "--stats_directory", directory,
                    "--allow_partial", "--output", output, "--rbo_p", "0.8"]
            with patch.object(sys, "argv", argv), redirect_stdout(StringIO()) as log:
                similarity.main()
            self.assertIn(output, log.getvalue())
            self.assertEqual(sheets["metadata"].set_index("item").loc["rbo_p", "value"], 0.8)

    def test_cli_missing_files_error_explains_requirements(self):
        with tempfile.TemporaryDirectory() as directory:
            for flags, expected in (([], "Both"), (["--allow_partial"], "At least one")):
                with patch.object(sys, "argv", ["stats", "--stats_directory", directory, *flags]):
                    with self.assertRaisesRegex(FileNotFoundError, expected):
                        similarity.main()
            save_run(directory, "no_manual")
            with patch.object(sys, "argv", ["stats", "--stats_directory", directory]):
                with self.assertRaisesRegex(FileNotFoundError, "Both"):
                    similarity.main()

    @unittest.skipUnless(importlib.util.find_spec("openpyxl"), "openpyxl is required for real Excel readback")
    def test_real_excel_partial_then_paired(self):
        with tempfile.TemporaryDirectory() as directory:
            save_run(directory, "no_manual")
            output = similarity.compute_from_directory(directory, allow_partial=True)
            self.assertEqual(pd.read_excel(output, sheet_name="summary")["condition"].tolist(), ["no_manual"])
            save_run(directory, "manual")
            self.assertEqual(similarity.compute_from_directory(directory, allow_partial=True), output)
            self.assertEqual(pd.read_excel(output, sheet_name="summary")["condition"].tolist(),
                             ["manual", "no_manual", "manual - no_manual"])


class SearchIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.helpers = load_search_helpers()
        self.dist = self.helpers["dist"]
        self.dist.get_world_size.return_value = 2
        self.dist.get_rank.return_value = 0
        self.search_args = SimpleNamespace(query_type="image", depth=1000)
        self.data_args = SimpleNamespace(dataset_name="flickr", sparse_manual=False)
        self.model_args = SimpleNamespace(eol_type="disassembleeol_separate", calculate_type="concat")
        self.save = self.helpers["_save_test_runs_for_similarity"]
        module_patch = patch.dict(sys.modules, {"compute_retrieval_similarity": similarity})
        module_patch.start()
        self.addCleanup(module_patch.stop)

    def output(self, root, manual=False):
        label = "manual" if manual else "no_manual"
        return str(Path(root) / f"0_{label}_128_sum_no_cluster_after_pad_{self.model_args.eol_type}_lower_no_mean_single" / "best.xlsx")

    def gather(self, extra_query="rank1"):
        def all_gather(destination, local):
            destination[:] = [local, {extra_query: run_record(a=3, b=2)}]
        self.dist.all_gather_object.side_effect = all_gather

    def test_concat_global_gather_partial_and_then_paired(self):
        self.gather()
        with tempfile.TemporaryDirectory() as directory, capture_workbook() as (sheets, _), redirect_stdout(StringIO()):
            self.save({"rank0": run_record(a=3, b=2)}, {"rank0": run_record(a=3, b=2)},
                      self.output(directory), self.model_args, self.data_args, self.search_args)
            files = list(Path(directory).rglob("test_runs_no_manual.pkl"))
            self.assertEqual(len(files), 1)
            self.assertIn("retrieval_similarity_disassembleeol_separate_concat", files[0].parts)
            with files[0].open("rb") as handle:
                payload = pickle.load(handle)
            self.assertEqual(set(payload["dense_run"]), {"rank0", "rank1"})
            self.assertEqual(set(payload["sparse_run"]), {"rank0", "rank1"})
            self.assertEqual(sheets["summary"].iloc[0]["queries_with_dense_and_sparse"], 2)
            self.assertEqual(self.dist.all_gather_object.call_count, 2)
            self.data_args.sparse_manual = True
            self.save({"rank0": run_record(a=3, b=2)}, {"rank0": run_record(a=3, b=2)},
                      self.output(directory, manual=True), self.model_args, self.data_args, self.search_args)
            self.assertTrue((files[0].parent / "test_runs_manual.pkl").is_file())
            self.assertEqual(sheets["summary"]["condition"].tolist(),
                             ["manual", "no_manual", "manual - no_manual"])

    def test_origin_text_concat_global_gather_partial_and_then_paired(self):
        self.model_args.eol_type = "disassembleeol_separate_origin_text"
        self.gather()
        for query_type in ("text", "image"):
            with self.subTest(query_type=query_type), tempfile.TemporaryDirectory() as directory, \
                    capture_workbook() as (sheets, writer), redirect_stdout(StringIO()):
                self.search_args.query_type = query_type
                self.data_args.sparse_manual = False
                self.dist.all_gather_object.reset_mock()
                local_run = {"rank0": run_record(a=3, b=2)}
                self.save(local_run, local_run, self.output(directory),
                          self.model_args, self.data_args, self.search_args)
                run_file, = Path(directory).rglob("test_runs_no_manual.pkl")
                self.assertIn("retrieval_similarity_disassembleeol_separate_origin_text_concat",
                              run_file.parts)
                with run_file.open("rb") as handle:
                    payload = pickle.load(handle)
                for key in ("dense_run", "sparse_run"):
                    self.assertEqual(set(payload[key]), {"rank0", "rank1"})
                self.assertEqual(payload["metadata"]["eol_type"], self.model_args.eol_type)
                self.assertEqual(payload["metadata"]["query_type"], query_type)
                self.assertEqual(self.dist.all_gather_object.call_count, 2)
                writer.assert_called_once_with(str(run_file.parent / "dense_sparse_similarity_test.xlsx"))
                self.assertEqual(sheets["summary"]["condition"].tolist(), ["no_manual"])
                self.assertEqual(sheets["summary"].iloc[0]["queries_with_dense_and_sparse"], 2)
                for depth in similarity.TOP_KS:
                    for metric in ("Overlap", "Spearman", "RBO"):
                        self.assertAlmostEqual(sheets["summary"].iloc[0][f"{metric}@{depth}"], 1.0)
                self.data_args.sparse_manual = True
                self.save(local_run, local_run, self.output(directory, manual=True),
                          self.model_args, self.data_args, self.search_args)
                self.assertTrue((run_file.parent / "test_runs_manual.pkl").is_file())
                self.assertEqual(sheets["summary"]["condition"].tolist(),
                                 ["manual", "no_manual", "manual - no_manual"])
                self.assertEqual(writer.call_count, 2)

    def test_legacy_waits_and_is_isolated_from_concat(self):
        self.gather()
        with tempfile.TemporaryDirectory() as directory, capture_workbook() as (_, writer), redirect_stdout(StringIO()):
            self.save({}, {}, self.output(directory), self.model_args, self.data_args, self.search_args)
            self.model_args.eol_type = "disassembleeol_separate_origin_text"
            self.save({}, {}, self.output(directory), self.model_args, self.data_args, self.search_args)
            self.model_args.eol_type, self.model_args.calculate_type = "prompteol", "separate"
            writer.reset_mock()
            self.save({}, {}, self.output(directory), self.model_args, self.data_args, self.search_args)
            writer.assert_not_called()
            files = list(Path(directory).rglob("test_runs_no_manual.pkl"))
            self.assertEqual(len(files), 3)
            self.assertEqual({path.parent.parent.name for path in files}, {
                "retrieval_similarity_prompteol_separate", "retrieval_similarity_disassembleeol_separate_concat",
                "retrieval_similarity_disassembleeol_separate_origin_text_concat"})
            self.data_args.sparse_manual = True
            self.save({}, {}, self.output(directory, True), self.model_args, self.data_args, self.search_args)
            writer.assert_called_once()

    def test_non_main_rank_gathers_but_never_writes(self):
        self.gather()
        self.dist.get_rank.return_value = 1
        for eol_type in ("disassembleeol_separate", "disassembleeol_separate_origin_text"):
            with self.subTest(eol_type=eol_type), tempfile.TemporaryDirectory() as directory, \
                    patch.object(similarity, "compute_from_directory") as compute:
                self.model_args.eol_type = eol_type
                self.dist.all_gather_object.reset_mock()
                self.save({}, {}, self.output(directory), self.model_args, self.data_args, self.search_args)
                self.assertEqual(list(Path(directory).iterdir()), [])
                compute.assert_not_called()
                self.assertEqual(self.dist.all_gather_object.call_count, 2)

    def test_disabled_switch_does_not_validate_gather_or_write(self):
        self.helpers["ENABLE_RETRIEVAL_SIMILARITY_STATS"] = False
        self.model_args.eol_type = "unsupported"
        with patch("builtins.open") as opened:
            self.save({}, {}, "unused", self.model_args, self.data_args, self.search_args)
        opened.assert_not_called()
        self.dist.all_gather_object.assert_not_called()

    def test_invalid_pairs_fail_before_gather_or_file_writes(self):
        for eol_type, calculate_type in (("prompteol", "concat"), ("disassembleeol_separate", "separate"),
                                         ("disassembleeol_separate_origin_text", "separate"),
                                         ("other", "concat")):
            self.model_args.eol_type, self.model_args.calculate_type = eol_type, calculate_type
            with self.subTest(eol=eol_type, calculate=calculate_type):
                with self.assertRaisesRegex(ValueError, "only supports"):
                    self.save({}, {}, "unused", self.model_args, self.data_args, self.search_args)
        self.dist.all_gather_object.assert_not_called()

    def test_main_validates_mode_before_gpu_initialization(self):
        tree = ast.parse((PROJECT_ROOT / "src/search_similarity.py").read_text(encoding="utf-8"))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
        validation_line = next(node.lineno for node in calls if isinstance(node.func, ast.Name)
                               and node.func.id == "_similarity_stats_configuration")
        device_lines = [node.lineno for node in calls if isinstance(node.func, ast.Attribute)
                        and node.func.attr in ("device", "init_process_group", "from_pretrained")]
        self.assertLess(validation_line, min(device_lines))


BASH = shutil.which("bash") or next((str(path) for path in (
    Path("D:/Git/bin/bash.exe"), Path("C:/Program Files/Git/bin/bash.exe")
) if path.is_file()), None)


@unittest.skipUnless(BASH, "Bash is required for launcher tests")
class SearchLauncherTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "scripts").mkdir()
        self.script = self.root / "scripts/search_similarity.sh"
        shutil.copyfile(PROJECT_ROOT / "scripts/search_similarity.sh", self.script)
        stub = self.root / "fake-deepspeed.sh"
        stub.write_text('#!/usr/bin/env bash\n'
                        'printf "%s\\0" "$ENABLE_RETRIEVAL_SIMILARITY_STATS" "$@" __END__ >> "$SIMILARITY_TEST_ARGV"\n',
                        encoding="utf-8")
        stub.chmod(0o755)
        self.record = self.root / "argv.bin"
        env_names = set(re.findall(r"\$\{([A-Z_][A-Z_0-9]*):-", self.script.read_text(encoding="utf-8")))
        self.env = {key: value for key, value in os.environ.items() if key not in env_names}
        self.env.update(DEEPSPEED_BIN=stub.as_posix(), SIMILARITY_TEST_ARGV=self.record.as_posix(),
                        PATH=str(Path(BASH).parent) + os.pathsep + os.environ.get("PATH", ""))

    def launch(self, mode=None, **env):
        command = [BASH, self.script.as_posix()]
        if mode is not None:
            command.append(mode)
        result = subprocess.run(command, env={**self.env, **env}, capture_output=True, text=True, timeout=30)
        calls = []
        if self.record.exists():
            calls = [part.split("\0") for part in self.record.read_bytes().decode().split("\0__END__\0") if part]
        return result, calls

    def assert_mode(self, calls, eol, calculate, conditions):
        self.assertEqual(len(calls), len(conditions))
        for call, condition in zip(calls, conditions):
            self.assertEqual(call[0], "true")
            self.assertIn("--num_gpus=2", call)
            self.assertIn("src/search_similarity.py", call)
            self.assertEqual(call[call.index("--eol_type") + 1], eol)
            self.assertEqual(call[call.index("--calculate_type") + 1], calculate)
            self.assertEqual("--sparse_manual" in call, condition == "manual")
            for arg, prefix in (("--passage_reps", "dense_output"), ("--sparse_index", "sparse_output"),
                                ("--output_dir", "output")):
                path = call[call.index(arg) + 1]
                self.assertTrue(path.startswith(prefix + "/"))
                self.assertIn(f"/no_filter/{calculate}/prompt_5/test/", path)
                self.assertIn(f"_after_pad_{eol}_lower_", path)
                length = "128_128" if condition == "manual" else "128"
                self.assertIn(f"/0_{condition}_{length}_sum_", path)

    def test_legacy_default_is_paired(self):
        result, calls = self.launch()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_mode(calls, "prompteol", "separate", ["no_manual", "manual"])

    def test_concat_default_is_single(self):
        result, calls = self.launch(EOL_TYPE="disassembleeol_separate", CALCULATE_TYPE="concat")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_mode(calls, "disassembleeol_separate", "concat", ["no_manual"])

    def test_origin_text_concat_default_is_single(self):
        eol = "disassembleeol_separate_origin_text"
        result, calls = self.launch(EOL_TYPE=eol, CALCULATE_TYPE="concat")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_mode(calls, eol, "concat", ["no_manual"])
        call, = calls
        self.assertEqual(call[call.index("--sparse_type") + 1], "single")
        self.assertEqual(call[call.index("--num_expended_tokens") + 1], "0")

    def test_concat_explicit_both(self):
        for eol in ("disassembleeol_separate", "disassembleeol_separate_origin_text"):
            with self.subTest(eol=eol):
                if self.record.exists():
                    self.record.unlink()
                result, calls = self.launch("both", EOL_TYPE=eol, CALCULATE_TYPE="concat")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assert_mode(calls, eol, "concat", ["no_manual", "manual"])

    def test_explicit_single_conditions_in_all_configurations(self):
        for eol, calculate in (("prompteol", "separate"), ("disassembleeol_separate", "concat"),
                               ("disassembleeol_separate_origin_text", "concat")):
            for mode in ("manual", "no_manual"):
                with self.subTest(eol=eol, mode=mode):
                    if self.record.exists():
                        self.record.unlink()
                    result, calls = self.launch(mode, EOL_TYPE=eol, CALCULATE_TYPE=calculate)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assert_mode(calls, eol, calculate, [mode])

    def test_invalid_pair_is_rejected_without_launch(self):
        for eol, calculate in (("prompteol", "concat"), ("disassembleeol_separate_origin_text", "separate")):
            with self.subTest(eol=eol, calculate=calculate):
                result, calls = self.launch(EOL_TYPE=eol, CALCULATE_TYPE=calculate)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Unsupported EOL_TYPE/CALCULATE_TYPE", result.stderr)
                self.assertEqual(calls, [])

    def test_invalid_run_mode_is_rejected_without_launch(self):
        result, calls = self.launch("bad")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Usage:", result.stderr)
        self.assertEqual(calls, [])

    def test_stats_can_be_disabled_and_custom_options_forwarded(self):
        result, calls = self.launch("no_manual", ENABLE_RETRIEVAL_SIMILARITY_STATS="false",
                                    GPUS="4", BASE_MODEL="model", DATASET="coco", QUERY_TYPE="text",
                                    TARGET_TYPE="image", USE_FAISS_GPU="0", PRECISION="bf16")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        call, = calls
        self.assertEqual(call[0], "false")
        self.assertIn("--num_gpus=4", call)
        self.assertIn("--bf16", call)
        self.assertNotIn("--fp16", call)
        self.assertNotIn("--use_gpu", call)
        self.assertEqual(call[call.index("--query_type") + 1], "text")
        self.assertIn("dense_output/model/coco/image/", call[call.index("--passage_reps") + 1])

    def test_origin_text_text_query_with_stats_disabled(self):
        eol = "disassembleeol_separate_origin_text"
        result, calls = self.launch(EOL_TYPE=eol, CALCULATE_TYPE="concat", QUERY_TYPE="text",
                                    TARGET_TYPE="image", ENABLE_RETRIEVAL_SIMILARITY_STATS="false")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        call, = calls
        self.assertEqual(call[0], "false")
        self.assertEqual(call[call.index("--eol_type") + 1], eol)
        self.assertEqual(call[call.index("--calculate_type") + 1], "concat")
        self.assertEqual(call[call.index("--query_type") + 1], "text")
        self.assertNotIn("--sparse_manual", call)
        self.assertIn("/flickr/image/", call[call.index("--passage_reps") + 1])
        self.assertIn(f"_after_pad_{eol}_lower_", call[call.index("--sparse_index") + 1])


if __name__ == "__main__":
    unittest.main()
