"""Selection permutations, probability alignment, cache isolation and launcher wiring."""
import ast
import copy
from contextlib import redirect_stdout
import importlib
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_freeret_dense as fixtures
from test_freeret_dense import ROOT, data_args, model_args, temporary_directory
import adaptive_views as views
import adaptive_view_shuffle as shuffle
import adaptive_view_encoding as encoding
import adaptive_view_overlap as overlap
import adaptive_encode_batch as encode_batch
import adaptive_retrieval_batch as retrieval_batch
import adaptive_sparse_index_batch as index_batch
import dense_representation as dense

RUN = "0123456789abcdef"
IDENTITY = dict(family="qwen2_5_vl",checkpoint={"path":"fixture"})
TEXTS = [dict(id=f"t{i}",text=f"example {i}") for i in range(4)]
IMAGES = [dict(id=f"i{i}",image=f"image-{i}") for i in range(3)]


def options(*, shuffled=True, pmi=False, run_id=RUN, mode="both", hierarchical=False):
    return views.AdaptiveViewArguments(adaptive_views=True, view_example_mode="fixed", view_selection_shuffle=shuffled,
        view_selection_shuffle_run_id=run_id if shuffled else None, view_selection_pmi=pmi,
        view_selection_hierarchical=hierarchical,
        view_selection_modality=mode,view_selection_rounds=2,view_texts_per_round=2,view_images_per_round=1)


class Processor:
    tokenizer = NS(all_special_ids=[0], encode=lambda label,**kwargs:[ord(label)-64])
    def apply_chat_template(self,messages,**kwargs):
        return "".join("<image>" if part["type"]=="image" else part["text"] for part in messages[0]["content"])
    def __call__(self,**kwargs):
        self.prompt=kwargs["text"][0]
        return dict(input_ids=torch.tensor([[1,2,3]]),attention_mask=torch.ones(1,3,dtype=torch.long))


def synthetic_score(model,processor,family,pool,items,modality,*,prior=False,return_details=False):
    by_aspect={row["aspect"]:i for pool in (views.view_pool(),views.view_pool(views.HIERARCHICAL_POOL))
               for i,row in enumerate(pool)}
    # Different nonuniform priors make a wrong PMI permutation detectable.
    logits=np.array([by_aspect[row["aspect"]]*(0.3 if prior else 0.2) for row in pool])
    logp=logits-np.log(np.exp(logits).sum())
    prompt=views.render_choice(processor,family,pool,texts=None if prior or modality=="image" else [row["text"] for row in items],
        image_count=len(items) if modality=="image" else 0,prior=prior)
    if return_details:
        return dict(probabilities=np.exp(logp).tolist(),log_probabilities=logp.tolist(),tokens=40,prompt=prompt)
    return np.exp(logp).tolist(),40


def choose(opts=None,*,identity=IDENTITY,dataset="flickr"):
    opts=options() if opts is None else opts
    with patch.object(views,"score_group",side_effect=synthetic_score),redirect_stdout(io.StringIO()):
        return views.choose_views(NS(config=NS(model_type="qwen2_5_vl")),Processor(),identity,dataset,"test",TEXTS,IMAGES,opts)


def overlap_choose(opts,model,data,task,identity,dataset):
    values=dict(vars(opts),view_overlap_selection=True,
        view_selected_count=5,view_overlap_samples=2,view_overlap_threshold=0.1,view_overlap_seed=42)
    strategy_options=overlap.OverlapViewArguments(**values)
    strategy=overlap.OverlapSelectionStrategy(strategy_options,model,data,task)
    factory=lambda *args:lambda record,perspective:{perspective["label"].lower():10}
    with patch.object(views,"score_group",side_effect=synthetic_score),patch.object(overlap,"SparseViewProbe",side_effect=factory),redirect_stdout(io.StringIO()):
        return strategy.choose(NS(config=NS(model_type="qwen2_5_vl")),Processor(),identity,dataset,"test",TEXTS,IMAGES,
                               strategy_options,report_path=Path("unused-diagnostics.json"))


class MappingTests(unittest.TestCase):
    def test_mapping_is_reproducible_bijective_and_scoped_without_mutating_pool(self):
        pool=views.view_pool(); before=copy.deepcopy(pool)
        a=shuffle.configuration(RUN,IDENTITY,"flickr",pool)
        self.assertEqual(a,shuffle.configuration(RUN,IDENTITY,"flickr",pool))
        self.assertEqual([x["display_label"] for x in a["mapping"]],list("ABCDEFGHIJKLMNOPQRST"))
        self.assertEqual({x["canonical_label"] for x in a["mapping"]},set("ABCDEFGHIJKLMNOPQRST"))
        self.assertNotEqual([x["canonical_label"] for x in a["mapping"]],list("ABCDEFGHIJKLMNOPQRST"))
        for identity,dataset,run in ((dict(IDENTITY,checkpoint={"path":"other"}),"flickr",RUN),(IDENTITY,"coco",RUN),
                                     (IDENTITY,"flickr","fedcba9876543210")):
            self.assertNotEqual(a["mapping"],shuffle.configuration(run,identity,dataset,pool)["mapping"])
        self.assertEqual(pool,before)
        shuffle.validate(a,pool,identity=IDENTITY,dataset="flickr")
        self.assertNotEqual(shuffle.new_run_id(),shuffle.new_run_id())

    def test_vectors_roundtrip_and_corrupt_mapping_is_rejected(self):
        pool=views.view_pool(); config=shuffle.configuration(RUN,IDENTITY,"flickr",pool)
        values=list(range(20))
        self.assertEqual(shuffle.canonical_values(shuffle.display_values(values,pool,config),pool,config),values)
        display=shuffle.display_pool(pool,config)
        self.assertEqual([r["aspect"] for r in display],[r["aspect"] for r in config["mapping"]])
        bad=copy.deepcopy(config); bad["mapping"][0]["canonical_label"]=bad["mapping"][1]["canonical_label"]
        bad["fingerprint"]=shuffle.digest({k:v for k,v in bad.items() if k!="fingerprint"})
        with self.assertRaisesRegex(ValueError,"bijection"):shuffle.validate(bad,pool)
        with self.assertRaisesRegex(ValueError,"model/dataset"):shuffle.validate(config,pool,identity=IDENTITY,dataset="coco")
        with self.assertRaisesRegex(ValueError,"16 lowercase"):shuffle.validate_run_id("latest")

    def test_false_cache_payload_and_paths_match_legacy_contract(self):
        sampling=dict(rounds=2,seed=42,texts_per_round=2,images_per_round=1)
        key=views.selection_cache_key(IDENTITY,"flickr","test","both",sampling,"data")
        self.assertEqual(key,dict(model=IDENTITY,dataset="flickr",split="test",pool=views.view_pool(),
            selection_modality="both",sampling=sampling,data_root=str(Path("data").resolve())))
        self.assertEqual(views.selection_cache_path("sparse","model","flickr",key),
            Path("sparse/view_selection/model/flickr")/views.digest(key)[:16]/"view_selection.json")
        self.assertIs(shuffle.mark_path("unchanged",None),"unchanged")
        selected=[dict(label=row["label"],aspect=row["aspect"]) for row in views.view_pool()[:5]]
        self.assertEqual(views.views_fingerprint(selected),views.digest(dict(version=views.VERSION,selection_modality="both",selected=selected)))

    def test_paths_preserve_image_and_dense_markers_and_reject_different_runs(self):
        config=shuffle.configuration(RUN,IDENTITY,"flickr")
        for tail in ("","_llava_base",dense.SUFFIX,"_llava_base"+dense.SUFFIX+dense.PROMPT_SUFFIX):
            path=Path("out")/("leaf"+tail)
            actual=shuffle.mark_path(path,config)
            self.assertEqual(actual.name,"leaf_shuffle_"+RUN+tail)
            self.assertEqual(shuffle.mark_path(actual,config),actual)
            self.assertEqual(shuffle.mark_path(path/"collection/index",config),actual/"collection/index")
            with self.assertRaisesRegex(ValueError,"different shuffle run"):
                shuffle.mark_path(actual,dict(run_id="fedcba9876543210"))


class ScoringTests(unittest.TestCase):
    def test_real_score_group_scores_new_letters_before_canonical_mapping(self):
        pool=views.view_pool(); config=shuffle.configuration(RUN,IDENTITY,"flickr",pool)
        model=fixtures.ToyModel().eval(); processor=Processor()
        model.get_input_embeddings=lambda:model.embedding
        displayed=shuffle.display_pool(pool,config)
        details=views.score_group(model,processor,"qwen2_5_vl",displayed,TEXTS[:1],"text",return_details=True)
        with torch.no_grad():
            logits=model(torch.tensor([[1,2,3]]),output_hidden_states=False).logits[0,-1,1:21]
        np.testing.assert_allclose(details["probabilities"],logits.softmax(-1).tolist(),rtol=1e-6)
        for row in config["mapping"]:
            self.assertIn(row["display_label"]+". "+row["aspect"],details["prompt"])

    def test_all_rounds_modalities_and_pmi_prior_share_mapping_and_restore_scores(self):
        for pmi in (False,True):
            with self.subTest(pmi=pmi),patch.object(views,"score_group",side_effect=synthetic_score) as scorer,redirect_stdout(io.StringIO()):
                selection=views.choose_views(NS(config=NS(model_type="qwen2_5_vl")),Processor(),IDENTITY,"flickr","test",TEXTS,IMAGES,options(pmi=pmi))
                expected=shuffle.display_pool(views.view_pool(),selection[shuffle.FIELD])
                self.assertEqual(scorer.call_count,5 if pmi else 4)
                self.assertTrue(all(call.args[3]==expected for call in scorer.call_args_list))
                logits=np.arange(20)*(0.2-0.3 if pmi else 0.2)
                expected_scores=np.exp(logits)/np.exp(logits).sum()
                np.testing.assert_allclose(selection["scores"],expected_scores,rtol=1e-6)
                self.assertEqual(selection["selected"][0]["label"],"A" if pmi else "T")
                shuffle.validate_diagnostics(selection)
                for row in selection["rounds"]:
                    for modality in ("text","image"):
                        self.assertEqual(shuffle.canonical_values(row[modality+"_display_probabilities"],selection["pool"],selection[shuffle.FIELD]),row[modality+"_probabilities"])
                if pmi:views.validate_pmi_selection(selection)
                with temporary_directory() as directory:
                    path=Path(directory)/"view_selection.json"; views.atomic_json(path,selection)
                    self.assertEqual(views.load_selection(path),selection)

    def test_shuffling_preserves_example_sampling_and_false_diagnostics(self):
        before=choose(options(shuffled=False)); after=choose()
        self.assertNotIn(shuffle.FIELD,before)
        self.assertNotIn("text_display_probabilities",before["rounds"][0])
        for a,b in zip(before["rounds"],after["rounds"]):
            self.assertEqual(a["text_ids"],b["text_ids"]);self.assertEqual(a["image_ids"],b["image_ids"])
        self.assertEqual(before["selected"],after["selected"])
        self.assertNotEqual(before["views_fingerprint"],after["views_fingerprint"])
        bad=copy.deepcopy(after);bad["rounds"][0]["text_display_probabilities"][0]+=0.1
        with self.assertRaisesRegex(ValueError,"round probabilities"):shuffle.validate_diagnostics(bad)

    def test_overlap_votes_remain_canonical_for_both_pmi_modes(self):
        for pmi in (False,True):
            selection=overlap_choose(options(pmi=pmi),model_args(),data_args(),"ir",IDENTITY,"flickr")
            self.assertEqual(set(selection["acceptance_counts"]),set("ABCDEFGHIJKLMNOPQRST"))
            self.assertEqual(overlap.validate_selection(selection),5)
            for sample in selection["overlap_samples"]:
                self.assertTrue(all(row["label"] in "ABCDEFGHIJKLMNOPQRST" for row in sample["tested"]))
            with temporary_directory() as directory:
                path=Path(directory)/"view_selection.json";views.atomic_json(path,selection)
                self.assertEqual(views.load_selection(path),selection)


class DriverTests(unittest.TestCase):
    def test_rank_zero_generates_one_run_id_and_other_ranks_receive_it(self):
        packet={}
        def broadcast(value,src):
            self.assertEqual(src,0)
            if value[0] is None:value[0]=copy.deepcopy(packet["data"])
            else:packet["data"]=copy.deepcopy(value[0])
        with patch.object(encoding.dist,"is_initialized",return_value=True),\
             patch.object(encoding.dist,"broadcast_object_list",side_effect=broadcast),\
             patch.object(shuffle,"new_run_id",return_value=RUN) as create:
            first=encoding.root_action(0,shuffle.new_run_id)
            second=encoding.root_action(1,lambda:self.fail("worker must not generate a permutation identifier"))
            self.assertEqual(first,second);self.assertEqual(first,RUN);create.assert_called_once()

    def test_encoder_reuses_only_current_batch_mapping_and_new_launch_has_new_paths(self, hierarchical=False):
        api=importlib.import_module("three_stage_retrieval_pipeline_adaptive_views")
        dataset=NS(img_id_list=[],text_id_list=[],text_dict={})
        runtime=NS(encoder=fixtures.ToyModel(),processor=Processor())
        with temporary_directory() as directory:
            data=data_args(dense_output_dir=str(Path(directory)/"dense"),sparse_output_dir=str(Path(directory)/"sparse"))
            def write(_encoder,_records,_data,roots,rank):
                (roots[1]/"collection/corpus_0.jsonl").write_text('{"id":"a","vector":{"word":10}}\n')
                with (roots[0]/"corpus_0.pkl").open("wb") as stream:pickle.dump((np.zeros((1,8)),["a"]),stream)
            def run(opts,kind):
                training=NS(task_type="ir",encode_type=kind,output_dir=str(Path(directory)/"logs"),fp16=False,bf16=False)
                opts.adaptive_data_root=directory
                return encoding.run_encode(model_args(),data,training,opts)
            with patch("qwen_vl_pipeline.init_process",return_value=(0,1,torch.device("cpu"))),\
                 patch.object(api,"_load_retrieval_runtime",return_value=runtime),patch.object(api,"_release_retrieval_runtime"),\
                 patch.object(api,"build_dataset",return_value=dataset),patch.object(encoding,"configure_processor"),\
                 patch.object(views,"model_identity",return_value=IDENTITY),patch.object(encoding,"source_pools",return_value=(TEXTS,IMAGES)),\
                 patch.object(encoding,"AdaptiveEncoder",side_effect=lambda *args:NS(model=args[0],last_prompts=[])),\
                 patch.object(encoding,"write_shard",side_effect=write),patch.object(views,"score_group",side_effect=synthetic_score) as scorer,redirect_stdout(io.StringIO()):
                first,selected=run(options(hierarchical=hierarchical),"image")
                self.assertEqual(scorer.call_count,4)
                self.assertEqual(len(selected["pool"]),10 if hierarchical else 20)
                second,reused=run(options(hierarchical=hierarchical),"text")
                self.assertEqual(scorer.call_count,4);self.assertEqual(selected,reused)
                fresh_opts=options(run_id=None,hierarchical=hierarchical)
                third,fresh=run(fresh_opts,"image")
                self.assertEqual(scorer.call_count,8)
                self.assertNotEqual(first,third);self.assertNotEqual(selected[shuffle.FIELD]["run_id"],fresh[shuffle.FIELD]["run_id"])
                mappings=list(Path(data.sparse_output_dir).rglob("option_mapping.json"))
                self.assertEqual(len(mappings),2)
                self.assertEqual(json.loads(mappings[0].read_text())["mapping"].__len__(),10 if hierarchical else 20)
                old_file=Path(directory)/"already.json";old_file.write_text("{}")
                bad=options();bad.view_selection_file=str(old_file)
                with self.assertRaisesRegex(ValueError,"starts a new experiment"):run(bad,"image")

    def test_batch_scope_fresh_ids_and_cli_parsers(self):
        with temporary_directory() as directory:
            env=dict(SHUFFLE_VIEW_OPTIONS="true",MODEL_FAMILIES="qwen25 llava_llama",DATASETS="flickr coco",DATA_ROOT=directory,
                CHECKPOINT_ROOT=str(Path(directory)/"checkpoints"),DENSE_OUTPUT_ROOT=str(Path(directory)/"dense"),SPARSE_OUTPUT_ROOT=str(Path(directory)/"sparse"),ENCODE_LOG_ROOT=str(Path(directory)/"logs"))
            settings=encode_batch.Settings.from_environment("ir",env,project=Path(directory))
            self.assertNotEqual(settings.shuffle_run_id,encode_batch.Settings.from_environment("ir",env,project=Path(directory)).shuffle_run_id)
            for path in settings.models.values():(path).mkdir(parents=True);(path/"config.json").touch()
            for dataset in settings.datasets:
                for path in encode_batch.annotation_files(settings.data_root,dataset,"test"):
                    path.parent.mkdir(parents=True,exist_ok=True);path.touch()
            jobs,missing=encode_batch.prepare_jobs(settings)
            self.assertFalse(missing);self.assertEqual(len(jobs),8)
            self.assertEqual({job.variables["_ADAPTIVE_SHUFFLE_RUN_ID"] for job in jobs},{settings.shuffle_run_id})
            self.assertTrue(all("_shuffle_"+settings.shuffle_run_id in job.variables["OUTPUT_DIR"] for job in jobs))
        from transformers import HfArgumentParser
        for klass in (views.AdaptiveViewArguments,overlap.OverlapViewArguments):
            parsed=HfArgumentParser(klass).parse_args_into_dataclasses(["--view_selection_shuffle","true"])[0]
            self.assertTrue(parsed.view_selection_shuffle)
            self.assertFalse(HfArgumentParser(klass).parse_args_into_dataclasses([])[0].view_selection_shuffle)


class DownstreamTests(unittest.TestCase):
    def build(self,directory,group,pmi,use_overlap,*,run_id=RUN,shuffled=True,split="test",hierarchical=False):
        root=Path(directory)
        env=dict(VIEW_EXAMPLE_MODE="fixed",MODEL_FAMILY="qwen25",CHECKPOINT_ROOT=str(root/"checkpoints"),DATA_ROOT=str(root),DENSE_OUTPUT_ROOT=str(root/"dense"),
            SPARSE_OUTPUT_ROOT=str(root/"sparse"),ADAPTIVE_OUTPUT_ROOT=str(root/"results"),VIEW_SELECTION_RUN_ID=run_id if shuffled else "",
            DATASET_SPLIT=split,VIEW_SELECTION_SPLIT="test",USE_HIERARCHICAL_VIEWS=str(hierarchical).lower(),
            USE_VIEW_PMI=str(pmi).lower(),USE_VIEW_OVERLAP_SELECTION=str(use_overlap).lower(),VIEW_SELECTION_ROUNDS="2",VIEW_TEXTS_PER_ROUND="2",VIEW_IMAGES_PER_ROUND="1",
            VIEW_OVERLAP_SAMPLES="2",VIEW_SELECTED_COUNT="5",VIEW_OVERLAP_THRESHOLD="0.1",VIEW_OVERLAP_SEED="42",
            USE_FREERET_DENSE="true",USE_FREERET_PROMPT_CONSTRAINTS="true")
        settings=retrieval_batch.Settings.from_environment(group,env,project=root)
        model_path=settings.models["qwen25"];model_path.mkdir(parents=True,exist_ok=True);(model_path/"config.json").write_text('{"model_type":"qwen2_5_vl"}')
        files=[]
        for dataset in settings.datasets:
            for source_split in {"test",split}:
                for path in encode_batch.annotation_files(root,dataset,source_split):
                    path.parent.mkdir(parents=True,exist_ok=True);path.touch()
            for task,query,kind in retrieval_batch.job_types(settings,dataset):
                model,data,training=retrieval_batch.namespaces(settings,"qwen25",dataset,task,kind)
                path,key,strategy=retrieval_batch.selection_location(settings,model,data,training,IDENTITY)
                if path.exists():selection=views.load_selection(path)
                else:
                    opts=settings.options
                    if use_overlap:selection=overlap_choose(opts,model,data,task,IDENTITY,dataset)
                    else:selection=choose(opts,identity=IDENTITY,dataset=dataset)
                    views.atomic_json(path,selection);files.append(path)
                expected=encoding.metadata(model,data,task,kind,IDENTITY,selection)
                dense_root,sparse_root=encoding.artifact_dirs(model,data,training,selection)
                completion=dict(expected,num_shards=1,selection_file=str(path),selection_fingerprint=selection["fingerprint"])
                for artifact in (dense_root,sparse_root):
                    views.atomic_json(artifact/views.MANIFEST,completion);dense.write_manifest(artifact,True,True)
                with (dense_root/"corpus_0.pkl").open("wb") as stream:pickle.dump((np.zeros((1,8)),["a"]),stream)
                (sparse_root/"collection/index").mkdir(parents=True)
                (sparse_root/"collection/corpus_0.jsonl").write_text('{"id":"a","vector":{"word":10}}\n')
                (sparse_root/"collection/index/segments_1").touch()
        return env,settings,files

    def test_index_and_retrieval_locate_exact_runs_for_every_task_and_selection_mode(self):
        for group in ("ir","tbpr","fused"):
            for pmi in (False,True):
                for use_overlap in (False,True):
                    with self.subTest(group=group,pmi=pmi,overlap=use_overlap),temporary_directory() as directory:
                        env,settings,files=self.build(directory,group,pmi,use_overlap)
                        jobs,missing=retrieval_batch.prepare_jobs(settings,identity_loader=lambda *args,**kwargs:IDENTITY)
                        self.assertFalse(missing);self.assertTrue(jobs)
                        for job in jobs:
                            self.assertIn("_shuffle_"+RUN,str(job.output))
                            self.assertEqual(job.configuration[shuffle.FIELD]["run_id"],RUN)
                        indexed,index_missing=index_batch.discover(index_batch.Settings.from_environment(group,env))
                        self.assertEqual(len(indexed),len(jobs))
                        self.assertFalse(index_missing)
                        wrong=retrieval_batch.Settings.from_environment(group,dict(env,VIEW_SELECTION_RUN_ID="fedcba9876543210"),project=Path(directory))
                        wrong_jobs,wrong_missing=retrieval_batch.prepare_jobs(wrong,identity_loader=lambda *args,**kwargs:IDENTITY)
                        self.assertFalse(wrong_jobs);self.assertTrue(wrong_missing)

    def test_exact_cache_recovery_and_single_pipeline_output_marking(self):
        with temporary_directory() as directory:
            env,settings,files=self.build(directory,"ir",True,False)
            jobs,_=retrieval_batch.prepare_jobs(settings,identity_loader=lambda *args,**kwargs:IDENTITY)
            api=importlib.import_module("three_stage_retrieval_pipeline_adaptive_views")
            job=jobs[0]
            args=api._parser().parse_args(["--dataset-name",job.variables["DATASET"]])
            model,data,training=retrieval_batch.namespaces(settings,"qwen25",args.dataset_name,"ir","text")
            for name,value in vars(data).items():setattr(args,name,value)
            args.adaptive_views=True;args.task_type="ir";args.query_type="image";args.stage1_method="sparse_concat"
            args.view_selection_file=job.variables["VIEW_SELECTION_FILE"];args.retrieval_model_name_or_path=model.model_name_or_path
            args.use_freeret_dense=True;args.use_freeret_prompt_constraints=True
            args.output_dir=str(Path(directory)/"custom")
            args.stage1_sparse_index=job.variables["SPARSE_CORPUS_DIR"];args.stage2_sparse_index=args.stage1_sparse_index
            args.stage2_dense_index=job.variables["DENSE_CORPUS_DIR"]
            args.view_selection_run_id=RUN
            with patch.object(encoding,"load_model_identity",return_value=IDENTITY):
                encoding.preflight(args,api)
                self.assertIn("_shuffle_"+RUN,args.output_dir)
                args.view_selection_run_id="fedcba9876543210"
                with self.assertRaisesRegex(ValueError,"requested VIEW_SELECTION_RUN_ID"):encoding.preflight(args,api)
            backup_root=Path(directory)/"backups"
            for i,path in enumerate(files):
                backup=backup_root/str(i)/"view_selection.json";backup.parent.mkdir(parents=True);backup.write_bytes(path.read_bytes());path.unlink()
            recovered=retrieval_batch.Settings.from_environment("ir",dict(env,VIEW_SELECTION_ROOT=str(backup_root)),project=Path(directory),recover_missing_selection=True)
            with redirect_stdout(io.StringIO()):
                recovered_jobs,missing=retrieval_batch.prepare_jobs(recovered,identity_loader=lambda *args,**kwargs:IDENTITY)
            self.assertFalse(missing);self.assertEqual(len(recovered_jobs),len(jobs))


class AlphaValidationTests(unittest.TestCase):
    def setup_experiments(self,directory,*,main_shuffled=True,val_shuffled=True,shared=False,val_dataset="flickr",
                          main_hierarchical=False,val_hierarchical=False):
        api=importlib.import_module("three_stage_retrieval_pipeline_adaptive_views")
        builder=DownstreamTests()
        root=Path(directory)
        _,main_settings,_=builder.build(root/"main","ir",True,False,shuffled=main_shuffled,hierarchical=main_hierarchical)
        val_id=RUN if shared else "fedcba9876543210"
        _,val_settings,_=builder.build(root/"validation","ir",True,False,run_id=val_id,shuffled=val_shuffled,split="val",hierarchical=val_hierarchical)
        main_jobs,_=retrieval_batch.prepare_jobs(main_settings,identity_loader=lambda *args,**kwargs:IDENTITY)
        val_jobs,_=retrieval_batch.prepare_jobs(val_settings,identity_loader=lambda *args,**kwargs:IDENTITY)
        main_job=next(job for job in main_jobs if job.variables["DATASET"]=="flickr" and job.variables["IR_QUERY_TYPE"]=="image")
        val_job=next(job for job in val_jobs if job.variables["DATASET"]==val_dataset and job.variables["IR_QUERY_TYPE"]=="image")
        args=api._parser().parse_args(["--dataset-name","flickr","--query-type","image"])
        model,data,_=retrieval_batch.namespaces(main_settings,"qwen25","flickr","ir","text")
        for name,value in vars(data).items():setattr(args,name,value)
        args.adaptive_views=True;args.use_freeret_dense=True;args.use_freeret_prompt_constraints=True
        args.retrieval_model_name_or_path=model.model_name_or_path
        args.view_selection_file=main_job.variables["VIEW_SELECTION_FILE"]
        args.view_selection_run_id=RUN if main_shuffled else None
        args.stage1_sparse_index=main_job.variables["SPARSE_CORPUS_DIR"]
        args.stage2_sparse_index=args.stage1_sparse_index
        args.stage2_dense_index=main_job.variables["DENSE_CORPUS_DIR"]
        args.output_dir=str(root/"main_results")
        args.select_hybrid_alpha=True;args.validation_split="val"
        args.validation_dataset_name=val_dataset
        args.val_passage_reps=val_job.variables["DENSE_CORPUS_DIR"]
        args.val_sparse_index=val_job.variables["SPARSE_CORPUS_DIR"]
        args.val_view_selection_file=None if shared else val_job.variables["VIEW_SELECTION_FILE"]
        if shared:
            # Publish the validation shards with the identical main locked file.
            for artifact in (args.val_passage_reps,args.val_sparse_index):
                path=Path(artifact)/views.MANIFEST
                manifest=json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(manifest["selection_fingerprint"],views.load_selection(args.view_selection_file)["fingerprint"])
                manifest["selection_file"]=args.view_selection_file
                views.atomic_json(path,manifest)
        return api,args

    def validate(self,api,args):
        with patch.object(dense,"validate_checkpoint"),patch.object(encoding,"load_model_identity",return_value=IDENTITY),\
             patch.object(api,"_load_retrieval_runtime") as load_runtime,\
             patch.object(encoding,"preflight",wraps=encoding.preflight) as preflight,redirect_stdout(io.StringIO()):
            api._validate_args(args)
            load_runtime.assert_not_called()
            self.assertEqual(preflight.call_count,2)
            self.assertEqual(preflight.call_args_list[1].kwargs,{"mark_output":False})
        return preflight.call_args_list[1].args[0]

    def test_independent_validation_modes_keep_main_results_and_cache_identity(self):
        for main_shuffle,val_shuffle in ((True,True),(True,False),(False,True),(False,False)):
            with self.subTest(main_shuffle=main_shuffle,val_shuffle=val_shuffle),temporary_directory() as directory:
                api,args=self.setup_experiments(directory,main_shuffled=main_shuffle,val_shuffled=val_shuffle)
                original_id=args.view_selection_run_id
                original_file=args.view_selection_file
                before=api._qwen_encoding_settings(args,"disassembleeol_separate","concat",False)
                validation=self.validate(api,args)
                self.assertIsNone(validation.view_selection_run_id)
                self.assertEqual(validation.view_selection_file,args.val_view_selection_file)
                self.assertEqual(args.view_selection_run_id,original_id)
                self.assertEqual(args.view_selection_file,original_file)
                self.assertEqual(validation.output_dir,args.output_dir)
                self.assertEqual("_shuffle_"+RUN in args.output_dir,main_shuffle)
                self.assertNotIn("fedcba9876543210",args.output_dir)
                after=api._qwen_encoding_settings(args,"disassembleeol_separate","concat",False)
                self.assertEqual(before,after)
                val_settings=api._qwen_encoding_settings(validation,"disassembleeol_separate","concat",False)
                self.assertNotEqual(views.digest(after),views.digest(val_settings))
                self.assertEqual(val_settings["data"]["dataset_split"],"val")
                self.assertEqual(after["data"]["dataset_split"],"test")

    def test_shared_file_keeps_main_run_id_when_implicit_or_explicit(self):
        for explicit in (False,True):
            with self.subTest(explicit=explicit),temporary_directory() as directory:
                api,args=self.setup_experiments(directory,shared=True)
                if explicit:args.val_view_selection_file=args.view_selection_file
                validation=self.validate(api,args)
                self.assertEqual(validation.view_selection_run_id,RUN)
                self.assertEqual(validation.view_selection_file,args.view_selection_file)
                self.assertEqual(validation.output_dir,args.output_dir)

    def test_cross_dataset_validation_uses_its_own_locked_file(self):
        with temporary_directory() as directory:
            api,args=self.setup_experiments(directory,val_dataset="coco")
            validation=self.validate(api,args)
            self.assertEqual(validation.dataset_name,"coco")
            self.assertEqual(args.dataset_name,"flickr")
            self.assertIsNone(validation.view_selection_run_id)
            self.assertEqual(validation.output_dir,args.output_dir)

    def test_wrong_main_id_and_corrupt_validation_file_still_fail(self):
        with temporary_directory() as directory:
            api,args=self.setup_experiments(directory)
            args.view_selection_run_id="aaaaaaaaaaaaaaaa"
            with patch.object(dense,"validate_checkpoint"),patch.object(encoding,"load_model_identity",return_value=IDENTITY),\
                 redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,"requested VIEW_SELECTION_RUN_ID"):
                api._validate_args(args)
            self.assertEqual(args.view_selection_run_id,"aaaaaaaaaaaaaaaa")
            args.view_selection_run_id=RUN
            path=Path(args.val_view_selection_file)
            selection=json.loads(path.read_text(encoding="utf-8"));selection["dataset"]="unexpected"
            views.atomic_json(path,selection)
            with patch.object(dense,"validate_checkpoint"),patch.object(encoding,"load_model_identity",return_value=IDENTITY),\
                 redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,"Corrupt adaptive view selection fingerprint"):
                api._validate_args(args)


class LauncherTests(unittest.TestCase):
    def test_retrieval_scripts_forward_local_and_external_run_identifiers(self):
        bash=shutil.which("bash") if os.name!="nt" else "D:/Git/bin/bash.exe"
        if not bash or not Path(bash).is_file():self.skipTest("Bash unavailable")
        names=["three_stage_retrieval_pipeline_adaptive_views.sh","run_adaptive_views_three_stage.sh"]
        names += [f"three_stage_retrieval_pipeline_adaptive_views{overlap}_{group}{multi}.sh"
                  for overlap,multi in (("",""),("_overlap","_multi")) for group in ("ir","tbpr","fused")]
        names += [f"three_stage_adaptive_{group}_multi.sh" for group in ("ir","tbpr","fused")]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);scripts=root/"scripts";scripts.mkdir()
            shutil.copyfile(ROOT/"scripts/freeret_dense_mode.sh",scripts/"freeret_dense_mode.sh")
            for name in names:shutil.copyfile(ROOT/"scripts"/name,scripts/name)
            capture=root/"capture.sh"
            capture.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@"\nprintf \'run=%s\\n\' "$VIEW_SELECTION_RUN_ID"\n');capture.chmod(0o755)
            for local in (False,True):
                for name in names:
                    with self.subTest(script=name,local=local):
                        body=(ROOT/"scripts"/name).read_text(encoding="utf-8")
                        if local:body=body.replace('${VIEW_SELECTION_RUN_ID-}','${VIEW_SELECTION_RUN_ID-'+RUN+'}')
                        path=scripts/name;path.write_text(body,encoding="utf-8")
                        env={k:v for k,v in os.environ.items() if k!="VIEW_SELECTION_RUN_ID"}
                        env.update(PYTHON_BIN=capture.as_posix(),DEEPSPEED_BIN=capture.as_posix(),MODEL_FAMILY="qwen25",
                            MODEL_PATH="fixture",DATASET="flickr",VIEW_SELECTION_FILE="selection.json",DENSE_CORPUS_DIR="dense",
                            SPARSE_CORPUS_DIR="sparse",LLAVA_IMAGE_MODE="anyres",DRY_RUN="false")
                        if not local:env["VIEW_SELECTION_RUN_ID"]=RUN
                        result=subprocess.run([bash,path.as_posix()],cwd=root,env=env,capture_output=True,text=True,timeout=30)
                        self.assertEqual(result.returncode,0,result.stderr)
                        lines=result.stdout.splitlines()
                        if name=="three_stage_retrieval_pipeline_adaptive_views.sh":
                            self.assertEqual(lines[lines.index("--view-selection-run-id")+1],RUN)
                        else:self.assertIn("run="+RUN,lines)

    def test_eight_encoding_scripts_forward_local_and_external_switch_and_internal_run_id(self):
        bash=shutil.which("bash") if os.name!="nt" else "D:/Git/bin/bash.exe"
        if not bash or not Path(bash).is_file():self.skipTest("Bash unavailable")
        names=["encode_adaptive_views"+overlap+suffix+".sh" for overlap in ("","_overlap") for suffix in ("","_ir","_tbpr","_fused")]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);scripts=root/"scripts";scripts.mkdir()
            shutil.copyfile(ROOT/"scripts/freeret_dense_mode.sh",scripts/"freeret_dense_mode.sh")
            capture=root/"capture.sh";capture.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@"\nprintf \'flag=%s\\n\' "$SHUFFLE_VIEW_OPTIONS"\n');capture.chmod(0o755)
            for name in names:
                for local in (False,True):
                    with self.subTest(script=name,local=local):
                        body=(ROOT/"scripts"/name).read_text(encoding="utf-8")
                        if local:body=body.replace('${SHUFFLE_VIEW_OPTIONS-false}','${SHUFFLE_VIEW_OPTIONS-true}')
                        path=scripts/name;path.write_text(body,encoding="utf-8")
                        env={k:v for k,v in os.environ.items() if k!="SHUFFLE_VIEW_OPTIONS"}
                        env.update(PYTHON_BIN=capture.as_posix(),DEEPSPEED_BIN=capture.as_posix(),LLAVA_IMAGE_MODE="anyres",
                                   DRY_RUN="false",MODEL_FAMILY="qwen25",MODEL_PATH="fixture",DATASET="flickr",ENCODE_TYPE="image",OUTPUT_DIR="logs",_ADAPTIVE_SHUFFLE_RUN_ID=RUN)
                        if not local:env["SHUFFLE_VIEW_OPTIONS"]="true"
                        result=subprocess.run([bash,path.as_posix()],cwd=root,env=env,capture_output=True,text=True,timeout=30)
                        self.assertEqual(result.returncode,0,result.stderr)
                        lines=result.stdout.splitlines()
                        if name in {"encode_adaptive_views.sh","encode_adaptive_views_overlap.sh"}:
                            self.assertEqual(lines[lines.index("--view_selection_shuffle")+1],"true")
                            self.assertEqual(lines[lines.index("--view_selection_shuffle_run_id")+1],RUN)
                        else:self.assertIn("flag=true",lines)


if __name__=="__main__":unittest.main()
