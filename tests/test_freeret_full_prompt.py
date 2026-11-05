"""Complete-prompt parity, model forwards, artifact isolation and launcher wiring."""
from contextlib import redirect_stdout
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
import test_freeret_dense as fixtures
from test_freeret_dense import ROOT, data_args, model_args, dense, adaptive, views, NativeQwenEncoder
import freeret_full_prompt as full
import freeret_prompt_constraints as constraints


def options(use=True, constraints_mode=False):
    args=model_args(True)
    args.use_freeret_full_prompt=use
    args.use_freeret_prompt_constraints=constraints_mode
    return args


class PromptTests(unittest.TestCase):
    def test_all_dataset_roles_exactly_match_native_freeret_and_replace_one_rule(self):
        import freeret_retrieval as original
        for dataset,key in full.DATASETS.items():
            for role in ("query","target"):
                image=(role=="target") if dataset not in {"remuq","llava"} else role=="query"
                if dataset=="webqa": image=role=="target"
                record=dict(text="literal Summary in one word user text",_freeret_role=role)
                if image: record["image"]="sample.png"
                item=NS(text=record["text"],image_path="sample.png" if image else None)
                with self.subTest(dataset=dataset,role=role):
                    baseline=original.one_word_prompt(key,role,item)
                    self.assertEqual(full.build_prompt(dataset,record),baseline)
                    aspect="clothing, such as category, color, material and style"
                    actual=full.build_prompt(dataset,record,aspect)
                    before=baseline.splitlines(keepends=True)
                    after=actual.splitlines(keepends=True)
                    self.assertEqual(len(before),len(after))
                    self.assertEqual(sum(a!=b for a,b in zip(before,after)),1)
                    self.assertIn(aspect,actual)
                    self.assertEqual(actual.count(record["text"]),baseline.count(record["text"]))
                    self.assertNotIn("Summary above",actual)
                    self.assertEqual(actual.count(full.IMAGE_TOKEN),int(image))
        for value in (None,float("nan"),["a","b"],"text with trailing spaces  "):
            record=dict(text=value)
            item=NS(text=value,image_path=None)
            self.assertEqual(full.build_prompt("flickr",record),original.one_word_prompt("flickr_t2i","query",item))

    def test_projection_keeps_modalities_separate_without_missing_fields(self):
        for dataset in ("webqa","remuq","llava"):
            for role in ("query","target"):
                for image in (False,True):
                    record=dict(_freeret_role=role,_freeret_projection=True)
                    record.update(image="sample.png") if image else record.update(text="actual words")
                    with self.subTest(dataset=dataset,role=role,image=image):
                        prompt=full.build_prompt(dataset,record,"color")
                        self.assertEqual(prompt.count(full.IMAGE_TOKEN),int(image))
                        self.assertNotIn("Summary",prompt)
                        self.assertIn("color",prompt)
                        self.assertNotIn("<question>:",prompt) if image else self.assertNotIn("<image>:",prompt)
        with self.assertRaisesRegex(ValueError,"requires 1 image"):
            full.build_prompt("remuq",dict(text="question",_freeret_role="query"))

    def test_mode_dependency_precedence_and_disabled_configuration(self):
        args=options(); args.use_freeret_dense=False
        with self.assertRaisesRegex(ValueError,"requires USE_FREERET_DENSE"):
            full.enabled(args)
        args.use_freeret_dense=True
        for old in (False,True):
            args.use_freeret_prompt_constraints=old
            self.assertFalse(constraints.enabled(args))
            self.assertTrue(constraints.active(args))
            self.assertEqual(constraints.configuration({},args),{"freeret_full_prompt":full.CONFIGURATION})
            with self.assertRaisesRegex(ValueError,"No complete FreeRet"):
                constraints.validate(args,"fashion-iq")
        self.assertEqual(dense.configuration({},True,True,True),dense.configuration({},True,False,True))

    def test_all_registered_fixed_and_random_view_aspects_have_no_summary_in_result(self):
        for dataset,task in (("flickr","ir"),("CUHK-PEDES","tbpr"),("webqa","t2it"),("remuq","it2t"),("llava","it2t")):
            for image in (False,True):
                record=dict(text="words")
                if image: record["image"]="sample.png"
                data=data_args(dataset_name=dataset,tbpr_type="type")
                aspects=full.fixed_aspects(record,task,data,fixtures.qwen_processor())
                self.assertEqual(len(aspects),5)
                self.assertTrue(all("Summary" not in aspect for aspect in aspects))
        with self.assertRaisesRegex(ValueError,"Cannot determine"):
            full.aspect_from_instruction("an unstructured custom prompt")


class ArtifactTests(unittest.TestCase):
    def test_new_sidecars_paths_and_explicit_mode_mismatches(self):
        with fixtures.temporary_directory() as directory:
            base=Path(directory)/"corpus"
            root=dense.mode_path(base,True,False,True)
            self.assertTrue(root.name.endswith(dense.SUFFIX+full.SUFFIX))
            self.assertEqual(root,dense.mode_path(base,True,True,True))
            root.mkdir(); dense.write_manifest(root,True,True,True)
            dense.validate_artifact(root,True,False,True)
            dense.validate_artifact(root,True,True,True)
            self.assertFalse(dense.reuse_sparse_index(root))
            for alias in (root,root/"collection/index",root/"corpus_0.pkl"):
                self.assertEqual(dense.mode_path(alias,True,False,True),alias)
                with self.assertRaisesRegex(ValueError,"mode mismatch"):
                    dense.mode_path(alias,True)
            constrained=dense.mode_path(base,True,True)
            with self.assertRaisesRegex(ValueError,"new artifact"):
                dense.mode_path(constrained,True,True,True)

    def test_all_tasks_exact_index_and_retrieval_find_full_mode_with_old_constraint_flag_ignored(self):
        from test_adaptive_sparse_exact import artifacts
        import adaptive_sparse_index_batch as index
        for group in ("ir","tbpr","fused"):
            with fixtures.temporary_directory() as directory:
                env,records,loader=artifacts(directory,TASK_GROUP=group,USE_FREERET_DENSE="true",USE_FREERET_FULL_PROMPT="true")
                for old in ("false","true"):
                    settings=index.Settings.from_environment(group,dict(env,USE_FREERET_PROMPT_CONSTRAINTS=old),exact_encoding=True)
                    with redirect_stdout(io.StringIO()): jobs,missing=index.discover_exact(settings,identity_loader=loader)
                    self.assertFalse(missing)
                    self.assertEqual([job.root for job in jobs],[record[0] for record in records])
                    self.assertTrue(all(job.root.name.endswith(full.SUFFIX) for job in jobs))

    def test_selection_cache_changes_only_for_overlap_and_full_flag_reaches_namespaces(self):
        import adaptive_retrieval_batch as batch
        for group in ("ir","tbpr","fused"):
            for overlap in (False,True):
                env=dict(VIEW_EXAMPLE_MODE="fixed",MODEL_FAMILY="qwen25",USE_FREERET_DENSE="true",USE_VIEW_OVERLAP_SELECTION=str(overlap).lower())
                before=batch.Settings.from_environment(group,env)
                after=batch.Settings.from_environment(group,dict(env,USE_FREERET_FULL_PROMPT="true"))
                for dataset in before.datasets:
                    for task,query,kind in batch.job_types(before,dataset):
                        a=batch.namespaces(before,"qwen25",dataset,task,kind)
                        b=batch.namespaces(after,"qwen25",dataset,task,kind)
                        self.assertTrue(full.enabled(b[0]))
                        self.assertEqual(batch.selection_location(before,*a,{"model":"fixture"})[:2]==
                                         batch.selection_location(after,*b,{"model":"fixture"})[:2],not overlap)


class FullRealModelTests(fixtures.RealModelTests):
    def compare(self,model,processor,records,family,adapter="adaptive"):
        if family.startswith("llava"):
            # Tiny local fixtures have no checkpoint chat template; use their
            # corresponding native user/assistant conversation delimiters.
            if "mistral" in family:
                processor.chat_template="{% for m in messages %}[INST]{% for c in m['content'] %}{% if c['type']=='image' %}<image>\n{% else %}{{ c['text'] }}{% endif %}{% endfor %}[/INST]{% endfor %}"
            else:
                processor.chat_template="{% for m in messages %}<|start_header_id|>user<|end_header_id|>\n\n{% for c in m['content'] %}{% if c['type']=='image' %}<image>\n{% else %}{{ c['text'] }}{% endif %}{% endfor %}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{% endfor %}"
        for task,dataset in (("ir","flickr"),("tbpr","CUHK-PEDES"),("t2it","webqa"),("it2t","remuq"),("it2t","llava")):
            for location in ("after_pad","before_pad"):
                with self.subTest(family=family,adapter=adapter,dataset=dataset,padding=location):
                    data=data_args(dataset_name=dataset,reps_loc=location,tbpr_type="type")
                    args=options()
                    args.model_name_or_path=family
                    with redirect_stdout(io.StringIO()):
                        encoder=(adaptive.AdaptiveEncoder(model,processor,args,data,task,{"selected":views.view_pool()[:2]})
                            if adapter=="adaptive" else NativeQwenEncoder(model,processor,args,data,task))
                    captured=[]
                    _,norm=dense._last_norm(model)
                    handle=norm.register_forward_pre_hook(lambda _m,inputs:captured.append(inputs[0].detach().clone()))
                    try: sparse,reps=encoder.encode(records)
                    finally: handle.remove()
                    count=2 if adapter=="adaptive" else 5
                    self.assertEqual(len(captured),count+1)
                    batch=full.prepare(processor,views.family_of(model.config),records,dataset,
                                       padding_side="left" if location=="after_pad" else "right")
                    expected=torch.nn.functional.normalize(torch.stack([captured[0][i,row[0][1]-1]
                        for i,row in enumerate(batch.spans)]),dim=-1)
                    self.assertTrue(torch.equal(reps,expected))
                    self.assertTrue(all(row.shape[0]==count for row in sparse))
                    self.assertTrue(all("Summary" not in prompt for prompt in encoder.last_prompts))
                    self.assertTrue(torch.isfinite(reps).all())
                    dense.configure_model(model,False)
                    if dataset=="flickr" and location=="after_pad" and adapter=="adaptive":
                        self.check_encoding_and_online_query(model,processor,records,family)

    def check_encoding_and_online_query(self,model,processor,records,family):
        """Exercise the real shared writer and both online query pipelines."""
        import pickle
        import qwen_vl_pipeline as pipeline
        import freeret_sparse_index as indexer
        if not hasattr(processor,"image_processor"):
            processor.image_processor=NS()
        with fixtures.temporary_directory() as directory:
            base=Path(directory); checkpoint=base/family
            checkpoint.mkdir(); (checkpoint/"config.json").write_text(model.config.to_json_string(),encoding="utf-8")
            args=options(); args.model_name_or_path=str(checkpoint)
            data=data_args(dense_output_dir=str(base/"dense"),sparse_output_dir=str(base/"sparse"),
                encode_is_query=False,use_generated_validation_data=False,qwen_min_pixels=0,qwen_max_pixels=0,print_sparse=False,
                llava_image_mode="base" if family.startswith("llava") else "anyres")
            training=NS(task_type="ir",encode_type="image")
            if family.startswith("llava"):
                import freeret_retrieval as original
                picture=base/"sample.png"; records[1]["image"].save(picture)
                item=dict(image=str(picture),text="")
                runner=original.FreeRetQwen.__new__(original.FreeRetQwen)
                runner.processor=processor; runner.device=torch.device("cpu"); runner.transformers_version="fixture"
                native=runner._prepare_llava_next_inputs([full.build_prompt("flickr",item)],[[str(picture)]])
                actual=full.prepare(processor,views.family_of(model.config),[item],"flickr").inputs
                for field in ("input_ids","attention_mask","pixel_values","image_sizes"):
                    self.assertTrue(torch.equal(native[field],actual[field]),field)
            with patch.object(pipeline,"load_runtime",return_value=(model,processor,0,1)), \
                 patch.object(pipeline,"build_dataset",return_value=object()), \
                 patch.object(pipeline,"dataset_records",return_value=[records[1]]),redirect_stdout(io.StringIO()):
                pipeline.run_encode(args,data,training)
            roots=pipeline.artifact_dirs(args,data,training)
            for root in roots:
                self.assertTrue((root/full.MANIFEST).is_file())
            kind="llava_mistral" if family.startswith("llava") else "qwen25"
            collection=roots[1] if family.startswith("llava") else roots[1]/"collection"
            self.assertTrue((collection/"corpus_0.jsonl").is_file())
            (collection/"index").mkdir(); (collection/"index/segments_1").touch()
            indexer.check_dense(roots[0],True,False,True)
            indexer.check_sparse(roots[1],kind,True,False,True)
            with (roots[0]/"corpus_0.pkl").open("rb") as stream:
                vectors,ids=pickle.load(stream)
            self.assertEqual(len(vectors),1); self.assertEqual(ids,["i"])
            for name in ("three_stage_retrieval_pipeline","three_stage_retrieval_pipeline_adaptive_views"):
                module=importlib.import_module(name)
                opts=module._parser().parse_args(["--dataset-name","flickr","--query-type","text",
                    "--use-freeret-dense","true","--use-freeret-full-prompt","true"])
                for field,value in vars(data).items():
                    if hasattr(opts,field): setattr(opts,field,value)
                opts.retrieval_model_name_or_path=str(checkpoint)
                opts.retrieval_dtype="float32"; opts.retrieval_batch_size=1
                opts.retrieval_llava_image_mode=data.llava_image_mode
                opts.retrieval_reps_loc=data.reps_loc
                runtime=NS(encoder=model,processor=processor,device=torch.device("cpu"),dtype=torch.float32,
                    model_path=str(checkpoint),query_cache={},tokenizer=processor.tokenizer,image_tokenizer=processor,filtered_ids=None,
                    vocab_dict={value:key for key,value in processor.tokenizer.get_vocab().items()},
                    qwen_family=None if family.startswith("llava") else model.config.model_type)
                with patch.object(module,"build_dataset",return_value=object()), \
                     patch.object(module,"_query_records",return_value=[records[0]]),redirect_stdout(io.StringIO()):
                    result=module.encode_queries_online(opts,runtime,eol_type=args.eol_type,calculate_type=args.calculate_type,
                                                  sparse_manual=False,progress_prefix="test")
                self.assertEqual(result.query_ids,["t"])
                module._validate_qwen_artifacts(opts,result,[(roots[0],True),(roots[1],False)],
                    eol_type=args.eol_type,calculate_type=args.calculate_type,sparse_manual=False)
            from adaptive_view_overlap import SparseViewProbe
            projection_data=data_args(dataset_name="remuq",print_sparse=False)
            probe=SparseViewProbe(model,processor,args,projection_data,"it2t",views.view_pool()[0])
            record=dict(text="a b",_freeret_role="query")
            vector=probe(record,views.view_pool()[0])
            self.assertIsInstance(vector,dict)
            self.assertNotIn("Summary",probe.encoder.last_prompts[0])
            self.assertNotIn(full.IMAGE_TOKEN,probe.encoder.last_prompts[0])
            dense.configure_model(model,False)


class LauncherTests(unittest.TestCase):
    def test_all_nine_paired_launchers_use_complete_mode_paths_and_flags(self):
        from test_freeret_task_scripts import environment, report, BASH
        import shlex
        if not BASH: self.skipTest("Bash unavailable")
        for group in ("ir","tbpr","fused"):
            for stage in ("encode","sparse_index","three_stage_retrieval"):
                env=environment(DRY_RUN="true",USE_FREERET_DENSE="true",USE_FREERET_FULL_PROMPT="true",USE_FREERET_PROMPT_CONSTRAINTS="true")
                name=f"{stage}_freeret_{group}.sh"
                result=subprocess.run([BASH,(ROOT/"scripts"/name).as_posix()],env=env,text=True,capture_output=True,timeout=30)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertTrue(report(result.stdout,"Sparse corpus: ").endswith(full.SUFFIX))
                command=shlex.split(report(result.stdout,"Command: "))
                self.assertEqual(command[command.index("--use_freeret_full_prompt")+1],"true")

    def test_local_configuration_reaches_batch_environment_and_commands(self):
        bash="D:/Git/bin/bash.exe" if os.name=="nt" else shutil.which("bash")
        if not bash: self.skipTest("Bash unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); scripts=root/"scripts"; scripts.mkdir()
            shutil.copyfile(ROOT/"scripts/freeret_dense_mode.sh",scripts/"freeret_dense_mode.sh")
            capture=root/"capture.sh"
            capture.write_text('#!/usr/bin/env bash\nprintf "%s" "$USE_FREERET_FULL_PROMPT"\n',encoding="utf-8")
            capture.chmod(0o755)
            names=[f"encode_adaptive_views{suffix}_{group}.sh" for suffix in ("","_overlap") for group in ("ir","tbpr","fused")]
            names += [f"sparse_index_adaptive_{group}.sh" for group in ("ir","tbpr","fused")]
            names += [f"three_stage_retrieval_pipeline_adaptive_views_{group}.sh" for group in ("ir","tbpr","fused")]
            for name in names:
                text=(ROOT/"scripts"/name).read_text(encoding="utf-8").replace('${USE_FREERET_DENSE-false}','${USE_FREERET_DENSE-true}').replace('${USE_FREERET_FULL_PROMPT-false}','${USE_FREERET_FULL_PROMPT-true}')
                path=scripts/name; path.write_text(text,encoding="utf-8")
                env={key:value for key,value in os.environ.items() if key not in {"USE_FREERET_DENSE","USE_FREERET_FULL_PROMPT"}}
                env.update(PYTHON_BIN=capture.as_posix(),DRY_RUN="true",LLAVA_IMAGE_MODE="anyres")
                result=subprocess.run([bash,path.as_posix()],cwd=root,env=env,text=True,capture_output=True,timeout=30)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertEqual(result.stdout,"true",name)


if __name__=="__main__": unittest.main()
