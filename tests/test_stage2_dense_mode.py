"""Restricted dense Stage 2, unchanged hybrid defaults, reports and launcher plumbing."""
from contextlib import redirect_stdout, redirect_stderr
import copy
import importlib
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import stage2_retrieval_mode as mode

MODULES = ("three_stage_retrieval_pipeline", "three_stage_retrieval_pipeline_adaptive_views")
BASH = "D:/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")


class Stage2Tests(unittest.TestCase):
    def test_dense_sorts_only_stage1_candidates_without_lucene_or_fusion(self):
        for module in MODULES:
            api = importlib.import_module(module)
            with self.subTest(module=module), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with (root / "corpus_0.pkl").open("wb") as stream:
                    pickle.dump((np.array([[.6,.8],[.8,.6],[1.,0.]], dtype=np.float32), ["a","b","outside"]), stream)
                a = NS(stage2_method="dense", stage2_run=None, stage1_method="sparse_concat", stage1_depth=2,
                    stage2_depth=1, stage2_dense_index=str(root), stage2_sparse_index=None, no_dense_normalize=False)
                encoded = api.EncodedQueries(["q"], np.array([[1.,0.]], dtype=np.float32), {"q":"unused"})
                forbidden = Mock(side_effect=AssertionError("dense Stage 2 must not call sparse/fusion/encoding"))
                with patch.object(api, "retrieve_sparse_topics_from_lucene", forbidden), \
                        patch.object(api, "fuse_restricted_runs", forbidden), patch.object(api, "encode_queries_online", forbidden), \
                        patch.object(api, "_validate_qwen_artifacts") as validate:
                    dense, sparse, selected = api.run_stage2(a, {"q":{"a":3.,"b":2.,"outside":1.}}, encoded_queries=encoded)
                self.assertEqual(list(dense["q"]), ["b"])
                self.assertEqual(selected, dense)
                self.assertFalse(sparse)
                self.assertEqual(validate.call_args.args[2], [(str(root), True)])
                forbidden.assert_not_called()

    def test_precomputed_dense_filters_before_topk_and_empty_shards_do_no_work(self):
        for module in MODULES:
            api = importlib.import_module(module)
            a = NS(stage2_method="dense", stage2_run="fixture", stage1_depth=2, stage2_depth=1)
            with patch.object(api,"read_run",return_value={"q":{"outside":100.,"a":1.,"b":2.}}):
                dense, sparse, selected = api.run_stage2(a,{"q":{"a":3.,"b":2.}})
            self.assertEqual(selected,{"q":{"b":2.}})
            self.assertEqual(dense,selected)
            self.assertEqual(sparse,{})
            with patch.object(api,"read_run",side_effect=AssertionError()):
                self.assertEqual(api.run_stage2(a,{}),({},{},{}))

    def test_configure_discards_inactive_inputs_before_validation(self):
        for module in MODULES:
            api=importlib.import_module(module)
            a=api._parser().parse_args(["--dataset-name","flickr","--task-type","ir","--query-type","text",
                "--retrieval-model-name-or-path","fixture", "--stage2-method","dense", "--select-hybrid-alpha",
                "--stage2-sparse-index","unused_dense_freeret_pre_mlp", "--val-passage-reps","missing_validation"])
            with redirect_stdout(io.StringIO()):
                api._validate_args(a)
            self.assertIsNone(a.stage2_sparse_index)
            self.assertIsNone(a.hybrid_alpha)
            self.assertFalse(a.select_hybrid_alpha)
            self.assertIsNone(a.val_passage_reps)
            default=api._parser().parse_args(["--dataset-name","flickr"])
            self.assertEqual(default.stage2_method,"hybrid")
            self.assertEqual(mode.method(NS()),"hybrid")

    def test_precomputed_pipeline_reports_dense_and_passes_it_to_stage3(self):
        for module in MODULES:
            api=importlib.import_module(module)
            for method in ("dense","hybrid"):
                with self.subTest(module=module,method=method), tempfile.TemporaryDirectory() as directory:
                    root=Path(directory)
                    (root/"first.json").write_text(json.dumps({"q":{"a":3.,"b":2.}}),encoding="utf-8")
                    (root/"second.json").write_text(json.dumps({"q":{"b":5.,"a":1.}}),encoding="utf-8")
                    args=["pipeline","--dataset-name","flickr","--task-type","ir","--query-type","text",
                        "--stage1-method","clip","--stage1-run",str(root/"first.json"),
                        "--stage2-run",str(root/"second.json"),"--ranker-run",str(root/"unused.json"),
                        "--stage2-method",method,"--stage2-depth","1","--output-dir",str(root/"out")]
                    evaluation=api.RecallEvaluation([1,5,10,20],{"q":{"b"}},{},None,1)
                    stage3=Mock(side_effect=lambda args,run,*unused,**kwargs:run)
                    with patch.object(sys,"argv",args), patch.object(api,"_build_recall_evaluation",return_value=evaluation), \
                            patch.object(api,"run_stage3",stage3), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        api.main()
                    self.assertEqual(stage3.call_args.args[1],{"q":{"b":5.}})
                    metadata=json.loads((root/"out/pipeline_metadata.json").read_text(encoding="utf-8"))
                    self.assertEqual(metadata["stage2"]["method"],method)
                    self.assertTrue((root/f"out/stage2_{method}_run.json").is_file())
                    report=json.loads((root/"out/recall_metrics.json").read_text(encoding="utf-8"))
                    rows=[row for row in report["results"] if row["stage"]=="stage2"]
                    self.assertEqual({row["variant"] for row in rows},{method})
                    self.assertEqual(rows[0]["recall"]["20"],1.)
                    if method=="dense":
                        self.assertIsNone(metadata["stage2"]["alpha"])
                        self.assertFalse((root/"out/stage2_hybrid_run.json").exists())
                        self.assertFalse((root/"out/alpha_selection.json").exists())

    def test_hybrid_still_requires_sparse_and_calls_fusion(self):
        for module in MODULES:
            api=importlib.import_module(module)
            a=NS(stage2_method="hybrid",stage2_run=None,stage1_method="sparse_concat",stage1_depth=2,
                 stage2_depth=1,stage2_dense_index="dense",stage2_sparse_index=None)
            with self.assertRaisesRegex(ValueError,"stage2_sparse_index"):
                api.run_stage2(a,{"q":{"a":1}},retrieval_runtime=NS())
            a.stage2_sparse_index="sparse"; a.no_dense_normalize=False; a.sparse_threads=1; a.hybrid_alpha=.5
            encoded=api.EncodedQueries(["q"],np.array([[1.,0.]]),{"q":"x"})
            with patch.object(api,"_validate_qwen_artifacts"),patch.object(api,"retrieve_dense_from_corpus",return_value={"q":{"a":2}}), \
                    patch.object(api,"retrieve_sparse_topics_from_lucene",return_value={"q":{"a":1}}) as sparse, \
                    patch.object(api,"fuse_restricted_runs",return_value={"q":{"a":.5}}) as fusion:
                _,_,selected=api.run_stage2(a,{"q":{"a":1}},encoded_queries=encoded)
            self.assertEqual(selected,{"q":{"a":.5}})
            sparse.assert_called_once(); fusion.assert_called_once()


class NamespaceTests(unittest.TestCase):
    def test_namespace_is_idempotent_and_keeps_freeret_suffix_last(self):
        for suffix in ("","_dense_freeret_pre_mlp","_dense_freeret_pre_mlp_freeret_constraints_v3",
                       "_dense_freeret_pre_mlp_freeret_full_prompt_v2"):
            path=Path("results")/("leaf"+suffix)
            self.assertEqual(mode.output_path(path,"hybrid"),path)
            dense=mode.output_path(path,"dense")
            self.assertEqual(dense.name,"leaf_stage2_dense"+suffix)
            self.assertEqual(mode.output_path(dense,"dense"),dense)

    def test_adaptive_markers_are_idempotent_with_shuffle_mass_and_full_prompt(self):
        import adaptive_view_shuffle as shuffle
        import adaptive_view_probability_mass as mass
        selection={"version":mass.VERSION,"views_fingerprint":"a"*64,
                   shuffle.FIELD:{"run_id":"b"*16}}
        path=Path("results/leaf_stage2_dense_dense_freeret_pre_mlp_freeret_full_prompt_v2")
        marked=mode.adaptive_output_path(path,selection)
        self.assertEqual(mode.adaptive_output_path(marked,selection),marked)
        self.assertEqual(marked.name.count("_shuffle_"),1)
        self.assertEqual(marked.name.count("_pm_"),1)
        self.assertTrue(marked.name.endswith("_freeret_full_prompt_v2"))

    def test_dense_overwrite_removes_only_inactive_stage2_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ("stage2_hybrid_run.json","stage2_sparse_run.trec","alpha_selection.json","stage1_run.json"):
                (root/name).touch()
            mode.clear_inactive_reports(root,NS(stage2_method="dense",overwrite=True))
            self.assertEqual([p.name for p in root.iterdir()],["stage1_run.json"])
            source=root/"stage2_sparse_run.trec"
            source.touch()
            mode.clear_inactive_reports(root,NS(stage2_method="dense",overwrite=True,stage1_run=str(source)))
            self.assertTrue(source.is_file())

    def test_adaptive_batches_keep_old_hybrid_hash_and_ignore_dense_alpha(self):
        from test_adaptive_sparse_exact import artifacts
        from test_freeret_dense import temporary_directory
        import adaptive_retrieval_batch as batch
        import adaptive_views as views
        import adaptive_view_encoding as backend
        with temporary_directory() as directory:
            env,records,loader=artifacts(directory)
            env["ADAPTIVE_OUTPUT_ROOT"]=str(Path(directory)/"results")
            settings=batch.Settings.from_environment("tbpr",env)
            family=settings.families[0]; dataset=settings.datasets[0]
            model,data,training=batch.namespaces(settings,family,dataset,"tbpr","image")
            identity=loader(str(settings.models[family]),dtype="float16")
            selection=records[0][2]
            dense,sparse=backend.artifact_dirs(model,data,training,selection)
            manifest=json.loads((sparse/views.MANIFEST).read_text(encoding="utf-8"))
            views.atomic_json(dense/views.MANIFEST,manifest)
            (dense/"corpus_0.pkl").write_bytes(b"fixture")
            (sparse/"collection/index").mkdir()
            (sparse/"collection/index/segments_1").write_text("fixture",encoding="utf-8")
            def prepare(values):
                with patch.object(batch,"annotation_files",return_value=[]), patch("llava_resolution.validate_checkpoint"):
                    jobs,missing=batch.prepare_jobs(batch.Settings.from_environment("tbpr",values),identity_loader=loader)
                self.assertFalse(missing)
                return jobs[0]
            baseline=prepare(env)
            explicit=prepare(dict(env,STAGE2_METHOD="hybrid"))
            self.assertEqual(baseline.fingerprint,explicit.fingerprint)
            one=prepare(dict(env,STAGE2_METHOD="dense",HYBRID_ALPHA=".1",SELECT_HYBRID_ALPHA="true",VAL_PASSAGE_REPS="ignored"))
            two=prepare(dict(env,STAGE2_METHOD="dense",HYBRID_ALPHA=".9"))
            self.assertEqual(one.fingerprint,two.fingerprint)
            self.assertNotEqual(one.fingerprint,baseline.fingerprint)
            self.assertIn("_stage2_dense",str(one.output))
            self.assertNotIn("HYBRID_ALPHA",one.configuration["parameters"])

    @unittest.skipUnless(BASH,"Bash unavailable")
    def test_shell_helper_preserves_hybrid_and_exports_dense_without_fusion(self):
        for selected in ("hybrid","dense"):
            command='source scripts/stage2_retrieval_mode.sh; stage2_mode_configure; printf "mode=%s alpha=%s select=%s path=%s\\n" "$STAGE2_METHOD" "$HYBRID_ALPHA" "$SELECT_HYBRID_ALPHA" "$(stage2_mode_path result_dense_freeret_pre_mlp_freeret_full_prompt_v2)"'
            result=subprocess.run([BASH,"-c",command],cwd=ROOT,env=dict(os.environ,STAGE2_METHOD=selected,
                HYBRID_ALPHA=".9",SELECT_HYBRID_ALPHA="true"),capture_output=True,text=True,check=True,timeout=30)
            self.assertIn("mode="+selected,result.stdout)
            self.assertIn("select="+("false" if selected=="dense" else "true"),result.stdout)
            self.assertIn("result"+("_stage2_dense" if selected=="dense" else "")+"_dense_freeret_pre_mlp_freeret_full_prompt_v2",result.stdout)


if __name__=="__main__":
    unittest.main()
