"""Recall@20 boundaries, output formats, distributed counts and unchanged objectives."""
import ast
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import csv
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import textwrap
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import metrices
import dense_search
import qwen_vl_pipeline as qwen
import freeret_retrieval as freeret

RANKS = (10, 11, 20, 21, None, 2)
NORMAL_KS = [1, 5, 10, 20, 100, 200, 300, 400]


def rankings():
    result = {}
    for qid, rank in enumerate(RANKS):
        if rank is None:
            result[str(qid)] = {}
            continue
        ids = [str(30000 + qid * 100 + i) for i in range(25 if rank > 3 else 3)]
        ids[rank-1] = str(10000 + qid)
        result[str(qid)] = {identifier: float(len(ids)-i) for i, identifier in enumerate(ids)}
    return result


class Dataset:
    def __init__(self, name):
        self.data_name = name
        self.text_id_list = list(range(len(RANKS)))
        self.query_id_list = self.text_id_list
        self.img_id_list = sorted({identifier for run in rankings().values() for identifier in run})

    def get_target(self, qid, query_type=None):
        target = str(10000 + int(qid))
        if self.data_name in {"coco", "flickr", "webqa"}:
            return [target, "999999"]
        return target

    def get_target_from_text(self, qid):
        return 20000 + int(qid)

    def get_target_from_image(self, identifier):
        value = int(identifier)
        return value+10000 if 10000 <= value < 10000+len(RANKS) else -value


def gathered(object_list, obj):
    object_list[:] = [obj] * len(object_list)


class SharedMetricsTests(unittest.TestCase):
    def metric(self, dataset, run, *, world=1):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(metrices.dist, "get_rank", return_value=0))
        stack.enter_context(patch.object(metrices.dist, "get_world_size", return_value=world))
        stack.enter_context(patch.object(metrices.dist, "all_gather_object", side_effect=gathered))
        stack.enter_context(patch.object(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor))
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(redirect_stderr(io.StringIO()))
        wrapped = {qid: {"docs": docs} for qid, docs in run.items()}
        return metrices.RecallMetrics(dataset, wrapped, wrapped, run, list(range(30)), list(run), NS(query_type="text"))

    def test_all_task_groups_count_dense_sparse_hybrid_and_preserve_old_cutoffs(self):
        for name in ("flickr", "coco", "CUHK-PEDES", "ICFG-PEDES", "RSTPReid", "webqa", "remuq", "llava"):
            with self.subTest(dataset=name):
                metric = self.metric(Dataset(name), rankings())
                self.assertEqual(metric.recall_k_setting_list, NORMAL_KS)
                metric.sort_and_count()
                for kind in ("dense", "sparse", "fusion"):
                    counts = getattr(metric, kind+"_counts")
                    self.assertEqual(counts[10], 2)
                    self.assertEqual(counts[20], 4)
                    self.assertEqual(counts[100], 5)
                metric.all_gather_object()
                for kind in ("dense", "sparse", "fusion"):
                    self.assertAlmostEqual(sum(getattr(metric, kind+"_recall_lists")[20]), 4/6)

    def test_csv_and_console_use_identical_metric_columns_for_all_methods_and_fashion_iq(self):
        for name in ("flickr", "fashion-iq"):
            with self.subTest(dataset=name), tempfile.TemporaryDirectory() as directory:
                metric = self.metric(Dataset(name), rankings())
                if name == "fashion-iq":
                    for kind in ("dense", "sparse", "fusion"):
                        setattr(metric, kind+"_recall_lists", {category: {k: [k/1000] for k in metric.recall_k_setting_list}
                            for category in metric.fashion_iq_list})
                else:
                    metric.sort_and_count(); metric.all_gather_object()
                output = io.StringIO()
                path = Path(directory)/"metrics.csv"
                with redirect_stdout(output), patch.dict(sys.modules,{"openpyxl":None}):
                    metric.print_recall(path)
                table = pd.read_csv(path)
                self.assertEqual(list(table.columns), list(map(str,metric.recall_k_setting_list)))
                self.assertEqual(len(table), 9 if name == "fashion-iq" else 3)
                self.assertTrue(np.allclose(table["20"], .02 if name == "fashion-iq" else 4/6))
                self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf"))
                self.assertFalse(list(Path(directory).glob("*.xlsx")))
                for label in ("Dense", "Sparse", "Fusion/Hybrid"):
                    self.assertIn(label+" recall @ 20:", output.getvalue())
                self.assertEqual(metric.recall_k_setting_list.count(20), 1)

    def test_distributed_aggregation_keeps_query_denominator_and_new_metric(self):
        run = {key: value for key,value in rankings().items() if key in {"1", "3"}}
        metric = self.metric(Dataset("flickr"), run, world=2)
        metric.sort_and_count(); metric.all_gather_object()
        for kind in ("dense", "sparse", "fusion"):
            recalled = getattr(metric, kind+"_recall_lists")
            self.assertEqual(sum(recalled[10]), 0)
            self.assertEqual(sum(recalled[20]), .5)


class PipelineTests(unittest.TestCase):
    def test_qwen_and_native_dense_count_the_same_boundaries_in_both_ir_directions_and_other_tasks(self):
        run = rankings()
        records = [dict(id=str(qid), native_id=qid) for qid in range(len(RANKS))]
        for name, task in (("flickr","ir"),("CUHK-PEDES","tbpr"),("webqa","t2it"),("remuq","it2t"),("llava","it2t")):
            for direction in ("text", "image") if task == "ir" else ("text",):
                dataset = Dataset(name)
                result = qwen.recall(dataset, records, run, task, direction)
                counts = dense_search.recall_counts(dataset, name, direction, run)
                self.assertEqual(tuple(NORMAL_KS), dense_search.RECALL_K)
                self.assertAlmostEqual(result["Recall@20"], 100*4/6)
                self.assertEqual(counts[NORMAL_KS.index(20)], 4)
                self.assertEqual(counts[-1], 6)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            counts = dense_search.recall_counts(Dataset("flickr"),"flickr","text",run)
            with patch.dict(sys.modules,{"openpyxl":None}):
                target = dense_search.write_recall(counts, Path(directory))
            self.assertEqual(target,Path(directory)/"dense.csv")
            table = pd.read_csv(target)
            self.assertAlmostEqual(table["20"][0], 4/6)

    def test_both_three_stage_reports_publish_recall20_for_dense_sparse_hybrid_and_ranker(self):
        for name in ("three_stage_retrieval_pipeline", "three_stage_retrieval_pipeline_adaptive_views"):
            api = importlib.import_module(name)
            args = NS(dataset_name="flickr", dataset_split="test", task_type="ir", query_type="text")
            with self.subTest(module=name), tempfile.TemporaryDirectory() as directory, \
                 patch.object(api,"build_dataset",return_value=Dataset("flickr")), redirect_stdout(io.StringIO()):
                reporter = api.RecallReporter(args, list(rankings()), Path(directory))
                reporter.record("stage2", {method:rankings() for method in ("dense","sparse","hybrid")})
                reporter.record("stage3", {"ranker":rankings()})
                payload = json.loads((Path(directory)/"recall_metrics.json").read_text(encoding="utf-8"))
                self.assertIn(20, payload["recall_k_values"])
                self.assertEqual(len(payload["results"]), 4)
                with (Path(directory)/"recall_metrics.csv").open(encoding="utf-8-sig",newline="") as stream:
                    reader = csv.DictReader(stream); rows = list(reader)
                    self.assertEqual(reader.fieldnames.index("R@20"), reader.fieldnames.index("R@10")+1)
                for row, stored in zip(rows, payload["results"]):
                    self.assertAlmostEqual(float(row["R@20"]), 4/6)
                    self.assertEqual(float(row["R@20"]), stored["recall"]["20"])

    def test_qwen_csv_outputs_include_metric_and_alpha_objectives_ignore_it(self):
        old = {"Recall@1":10., "Recall@5":20., "Recall@10":30.}
        expanded = dict(old, **{"Recall@20":100.})
        self.assertEqual(qwen._alpha_quality(old,"ir"), qwen._alpha_quality(expanded,"ir"))
        for name in ("three_stage_retrieval_pipeline", "three_stage_retrieval_pipeline_adaptive_views"):
            api = importlib.import_module(name)
            before = [{"category":"all", "recall":{"1":.1,"5":.2,"10":.3}}]
            after = [{"category":"all", "recall":dict(before[0]["recall"], **{"20":1.})}]
            self.assertEqual(api._alpha_objective(before,"flickr"), api._alpha_objective(after,"flickr"))
        with tempfile.TemporaryDirectory() as directory:
            metrics = {kind:expanded for kind in ("dense","sparse","hybrid")}
            rows = qwen._metric_rows(metrics, NS(dataset_name="flickr",dataset_split="test"),"fixed",.5)
            path = Path(directory)/"best.csv"; qwen._write_metrics_csv(path, rows)
            with path.open(encoding="utf-8-sig",newline="") as stream:
                self.assertTrue(all(float(row["Recall@20"])==100 for row in csv.DictReader(stream)))


class FreeRetAndDiagnosticsTests(unittest.TestCase):
    def test_freeret_custom_cutoffs_always_include20_without_changing_old_metrics_or_mrr(self):
        records = [dict(relevant_ids=[str(10000+int(qid))], stage1=[dict(candidate_id=key) for key in docs],
                        final=[dict(candidate_id=key) for key in docs]) for qid,docs in rankings().items()]
        for cutoffs in ([1,5,10], [20,1,20,50], []):
            for key in ("stage1","final"):
                result = freeret.evaluate_records(records, cutoffs, key)
                self.assertAlmostEqual(result["recall@20"], 4/6)
                self.assertEqual(list(k for k in result if k=="recall@20"), ["recall@20"])
                self.assertEqual(freeret.evaluate_records([],cutoffs,key)["recall@20"], 0)
                expected_mrr = sum(1/k for k in RANKS if k is not None)/len(RANKS)
                self.assertAlmostEqual(result["mrr"], expected_mrr)
        args = NS(target_batch_size=1,query_batch_size=1,rerank_batch_size=1,rerank_num=5,stage1_topk=10,
                  similarity_chunk_size=1,max_pixels=0,max_target_gpu_gb=0,metrics_ks=[10,1,10])
        freeret.validate_args(args)
        self.assertEqual(args.metrics_ks, [1,10,20])

    def test_dspy_diagnostic_metric_is_new_but_optimizer_feedback_is_identical(self):
        source = ROOT/"src/dspy_prompt_schema_generation_encode_and_search.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        names = {"_canonical_retrieval_id","_as_retrieval_id_list","_get_retrieval_target",
                 "collect_retrieval_diagnostics","_retrieval_failure_sort_key","build_gepa_retrieval_feedback"}
        namespace = dict(torch=torch,np=np,json=json,is_distributed=lambda:False,is_main_process=lambda:True,
            broadcast_object_from_main=lambda value:value,_describe_retrieval_item=lambda *args:"fixture",
            _DSPY_REFLECTION_CONFIG=NS(dspy_feedback_failures_per_direction=0,dspy_reflection_mode="text"),
            normalize_dspy_aspects=lambda value:value,_pool_constraint=lambda:"")
        module = ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names],type_ignores=[])
        exec(compile(module,str(source),"exec"),namespace)
        wrapped = {key:{"docs":docs} for key,docs in rankings().items()}
        diagnostics = namespace["collect_retrieval_diagnostics"](Dataset("flickr"),wrapped,"text")
        self.assertAlmostEqual(diagnostics["recall"][20],100*4/6)
        direction = dict(name="text_to_image",score=1.,diagnostics=diagnostics)
        with_metric = namespace["build_gepa_retrieval_feedback"]([],1.,[direction])
        without_metric = dict(diagnostics,recall={k:v for k,v in diagnostics["recall"].items() if k!=20})
        self.assertEqual(with_metric,namespace["build_gepa_retrieval_feedback"]([],1.,[dict(direction,diagnostics=without_metric)]))

    def test_per_query_outputs_all_report20_with_correct_boundary(self):
        tree = ast.parse((ROOT/"src/search_val_then_test.py").read_text(encoding="utf-8"))
        # This program keeps its optional case-study exporter as quoted Python.
        # Validate those templates without enabling the exporter as a side effect.
        trees = [tree] + [ast.parse(textwrap.dedent(node.value)) for node in ast.walk(tree)
            if isinstance(node,ast.Constant) and isinstance(node.value,str) and "'r@20':" in node.value]
        cases = [node for subtree in trees for node in ast.walk(subtree) if isinstance(node,ast.Dict)
                 and any(isinstance(key,ast.Constant) and key.value=="r@20" for key in node.keys)]
        self.assertEqual(len(cases), 6)
        for case in cases:
            value = next(value for key,value in zip(case.keys,case.values) if isinstance(key,ast.Constant) and key.value=="r@20")
            expression = compile(ast.Expression(value),"case","eval")
            for rank, expected in ((20,True),(21,False)):
                docs = torch.arange(1,26); target = torch.tensor([rank])
                self.assertEqual(eval(expression,dict(torch=torch,search_results={20:docs[:20]},target=target)),expected)


if __name__=="__main__": unittest.main()
