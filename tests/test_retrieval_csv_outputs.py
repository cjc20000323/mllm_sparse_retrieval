"""Every retrieval report is CSV, including the combined similarity tables."""
import ast
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import test_recall20 as recall_tests
from test_recall20 import Dataset, rankings
from test_freeret_dense import ROOT
import compute_retrieval_similarity as similarity


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as stream:
        reader=csv.DictReader(stream)
        return reader.fieldnames,list(reader)


class CsvOutputTests(unittest.TestCase):
    def test_shared_writer_normalizes_legacy_excel_paths_without_touching_history(self):
        owner=recall_tests.SharedMetricsTests()
        try:
            metric=owner.metric(Dataset("flickr"),rankings())
            metric.sort_and_count(); metric.all_gather_object()
            with tempfile.TemporaryDirectory() as directory:
                for suffix in (".xlsx",".xls",""):
                    legacy=Path(directory)/("metrics"+suffix)
                    legacy.write_bytes(b"historical workbook")
                    with patch.dict(sys.modules,{"openpyxl":None}):
                        metric.print_recall(legacy)
                    self.assertEqual(legacy.read_bytes(),b"historical workbook")
                    csv_path=legacy.with_suffix(".csv")
                    fields,rows=read_rows(csv_path)
                    self.assertEqual(fields,["1","5","10","20","100","200","300","400"])
                    self.assertEqual(len(rows),3)
                    self.assertAlmostEqual(float(rows[0]["20"]),4/6)
        finally:
            owner.doCleanups()

    def test_similarity_tables_share_one_csv_with_stable_fields_and_round_trip_text(self):
        note='中文, "quoted"\nsecond line'
        summaries=[dict(condition="manual",queries_with_dense_and_sparse=2,**{"Overlap@20":.5,"Spearman@20":float("nan")})]
        per_query=[dict(condition="manual",query_id="q1",**{"Overlap@20":1.,"Spearman@20":.75})]
        metadata=[dict(item="说明",value=note)]
        with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules,{"openpyxl":None}):
            old=Path(directory)/"report.xlsx"; old.write_bytes(b"historical workbook")
            actual=Path(similarity._write_csv_report(old,summaries,per_query,metadata))
            self.assertEqual(actual,old.with_suffix(".csv"))
            self.assertEqual(old.read_bytes(),b"historical workbook")
            self.assertTrue(actual.read_bytes().startswith(b"\xef\xbb\xbf"))
            fields,rows=read_rows(actual)
            self.assertEqual(fields,["table_type","condition","queries_with_dense_and_sparse","Overlap@20","Spearman@20","query_id","item","value"])
            self.assertEqual([row["table_type"] for row in rows],["summary","per_query","metadata"])
            self.assertEqual(rows[0]["queries_with_dense_and_sparse"],"2")
            self.assertEqual(rows[0]["Spearman@20"],"")
            self.assertEqual(rows[2]["value"],note)
            table=pd.read_csv(actual)
            self.assertEqual(table.loc[table["table_type"]=="metadata","value"].iloc[0],note)
            self.assertTrue(pd.isna(table.loc[0,"Spearman@20"]))
            self.assertEqual(len(list(Path(directory).glob("*.csv"))),1)

    def test_similarity_partial_then_paired_report_reuses_one_csv_and_retains_all_rows(self):
        dense={"q1":{"docs":{"a":2.,"b":1.}},"q2":{"docs":{"c":2.,"d":1.}}}
        sparse={"q1":{"docs":{"b":2.,"a":1.}},"q2":{"docs":{"c":2.,"e":1.}}}
        with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules,{"openpyxl":None}):
            root=Path(directory)
            self.assertIsNone(similarity.compute_from_directory(root))
            def save(name,values):
                with (root/name).open("wb") as stream:
                    pickle.dump(dict(dense_run=dense,sparse_run=values,metadata={"note":"中文, note\nline"}),stream)
            save("test_runs_manual.pkl",sparse)
            self.assertIsNone(similarity.compute_from_directory(root))
            partial=similarity.compute_from_directory(root,allow_partial=True)
            self.assertEqual(Path(partial),root/"dense_sparse_similarity_test.csv")
            first=pd.read_csv(partial)
            self.assertEqual(len(first[first["table_type"]=="summary"]),1)
            self.assertEqual(len(first[first["table_type"]=="per_query"]),2)
            save("test_runs_no_manual.pkl",dense)
            paired=similarity.compute_from_directory(root)
            self.assertEqual(paired,partial)
            result=pd.read_csv(paired)
            self.assertEqual(result[result["table_type"]=="summary"]["condition"].tolist(),["manual","no_manual","manual - no_manual"])
            self.assertEqual(len(result[result["table_type"]=="per_query"]),4)
            self.assertFalse(list(root.glob("*.xlsx")))
            self.assertEqual(len(list(root.glob("*.csv"))),1)

    def test_empty_similarity_csv_still_has_type_header(self):
        with tempfile.TemporaryDirectory() as directory:
            result=similarity._write_csv_report(Path(directory)/"empty",[],[],[])
            fields,rows=read_rows(result)
            self.assertEqual(fields,["table_type"]); self.assertEqual(rows,[])

    def test_all_repository_retrieval_paths_and_writers_use_csv(self):
        for path in (ROOT/"src").rglob("*.py"):
            if "__pycache__" in path.parts: continue
            text=path.read_text(encoding="utf-8")
            tree=ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                    self.assertNotIn(node.func.attr,{"to_excel","ExcelWriter"},str(path))
                if isinstance(node,ast.Constant) and isinstance(node.value,str):
                    self.assertNotIn(".xlsx",node.value,str(path))
        for name in ("search_dense_mmret.sh","search_dense_unime.sh","search_dense_vlm2vec.sh"):
            text=(ROOT/"scripts"/name).read_text(encoding="utf-8")
            self.assertIn("${SAVE_DIR}/dense.csv",text)
            self.assertNotIn("dense.xlsx",text)


if __name__=="__main__": unittest.main()
