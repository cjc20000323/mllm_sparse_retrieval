"""Prompt, model-forward and launcher checks for the optional FreeRet constraints."""
import ast
import copy
from contextlib import redirect_stdout
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

import test_freeret_dense as fixtures
from test_freeret_dense import (ROOT, data_args, model_args, temporary_directory,
                                qwen_processor, adaptive, dense, views, NativeQwenEncoder)
import freeret_prompt_constraints as constraints
import freeret_prompt_catalog as catalog


def options(use=True, dense_mode=True, name="fixture"):
    value = model_args(dense_mode, name)
    value.use_freeret_prompt_constraints = use
    return value


class PromptTests(unittest.TestCase):
    def test_task_description_is_shared_before_input_with_one_trailing_newline(self):
        for dataset,key in constraints.DATASET_KEYS.items():
            description=catalog.PROMPT_SPECS[key].task_description.rstrip("\n")+"\n"
            with self.subTest(dataset=dataset):
                self.assertEqual(constraints.task_description(dataset),description)
                for context in ("<image>\n", "user text\n", "<image>\nuser text\n"):
                    self.assertEqual(constraints.augment_context(context,options(),dataset),description+context)
                    self.assertIs(constraints.augment_context(context,options(False),dataset),context)
                    self.assertIs(constraints.augment_context(context,options(False,False),dataset),context)
                self.assertFalse(description.endswith("\n\n"))

    def test_template_descriptions_preserve_chat_prefixes_inputs_and_view_elements(self):
        import template
        for dataset in constraints.DATASET_KEYS:
            templates=constraints.Templates(options(),data_args(dataset_name=dataset),input_type="text")
            description=constraints.task_description(dataset)
            for family,header in (("llava_mistral","[INST]"),
                                  ("llama3","<|start_header_id|>user<|end_header_id|>\n\n"),
                                  ("qwen2_5","<|im_start|>user\n"),
                                  ("qwen3","<|im_start|>user\n")):
                for suffix in ("_text_prefix","_image_prefix","_content_element",""):
                    name=family+"_template"+suffix
                    original=getattr(template,name)
                    actual=getattr(templates,name)
                    with self.subTest(dataset=dataset,template=name):
                        if suffix=="_content_element":
                            self.assertEqual(actual,original)
                        else:
                            self.assertEqual(actual,original.replace(header,header+description,1))
                            self.assertEqual(actual.count(description),1)
            messages=[dict(role="system",content="existing system"),
                      dict(role="user",content=[dict(type="image"),dict(type="text",text="Summary above image in one word: ")])]
            original=copy.deepcopy(messages)
            actual=constraints.augment_template(messages,options(),data_args(dataset_name=dataset),input_type="image")
            self.assertEqual(actual[0],original[0])
            self.assertEqual(actual[1]["content"][0],dict(type="text",text=description))
            self.assertEqual(actual[1]["content"][1],dict(type="image"))
            self.assertEqual(messages,original)
            self.assertIs(constraints.augment_template(messages,options(False),data_args()),messages)

    def test_dependency_dataset_support_and_exact_false_behavior(self):
        original = "Summary the written words and symbols in above image in one word: "
        self.assertIs(constraints.augment(original, options(False), "flickr"), original)
        for use_dense in (False, True):
            self.assertIs(constraints.enabled(options(False, use_dense)), False)
        with self.assertRaisesRegex(ValueError, "requires USE_FREERET_DENSE"):
            constraints.enabled(options(True, False))
        with self.assertRaisesRegex(ValueError, "No FreeRet prompt"):
            constraints.validate(options(), "fashion-iq")

    def test_all_dataset_roles_reuse_catalog_instructions_without_visual_tokens(self):
        for dataset, key in constraints.DATASET_KEYS.items():
            spec = catalog.PROMPT_SPECS[key]
            for role in ("query", "target"):
                image = ((role == "target") if dataset not in {"remuq", "llava"} else role == "query")
                if dataset == "webqa" and role == "target": image = True
                record = dict(text="user text", _freeret_role=role)
                if image: record["image"] = "actual-image"
                with self.subTest(dataset=dataset, role=role):
                    expected = constraints.plain_instruction(getattr(spec, role+"_instruction"))
                    self.assertEqual(constraints.instruction(dataset, record), expected)
                    original = "Summary the main people or objects in above sentence in one word: "
                    actual = constraints.augment(original, options(), dataset, record)
                    self.assertEqual(actual, expected+"\n"+original)
                    self.assertTrue(actual.endswith(original))
                    self.assertEqual(constraints.augment(actual, options(), dataset, record), actual)
                    self.assertNotIn("<image>", actual)
                    self.assertNotIn("<|image_pad|>", actual)
                    self.assertNotIn(spec.task_description, actual)
        # A projected text-only corpus item must retain its source role.
        self.assertIn("accompanying information", constraints.instruction("webqa", {"text":"x", "_freeret_role":"target"}))
        self.assertIn("image and question", constraints.instruction("remuq", {"text":"x", "_freeret_role":"query"}))

    def test_legacy_templates_are_local_and_user_text_is_never_rewritten(self):
        import template
        original = template.retrieval_disassemble_text_prompts_for_concat[:]
        old = constraints.Templates(options(False), data_args(), input_type="text")
        new = constraints.Templates(options(), data_args(), input_type="text")
        self.assertIs(old.retrieval_disassemble_text_prompts_for_concat, template.retrieval_disassemble_text_prompts_for_concat)
        prefix = constraints.instruction("flickr", input_type="text")+"\n"
        self.assertEqual(new.retrieval_disassemble_text_prompts_for_concat, [prefix+x for x in original])
        self.assertEqual(template.retrieval_disassemble_text_prompts_for_concat, original)
        body = new.llava_mistral_template_text_prefix + new.llava_mistral_template_content_element.format(new.text_prompt_for_concat)
        user = "Summary above sentence in one word: quoted user text"
        rendered = body.replace("<sent>", user)
        self.assertEqual(rendered.count(prefix), 1)
        self.assertIn(user, rendered)

    def test_original_freeret_entry_reexports_identical_prompt_catalog(self):
        import freeret_retrieval as freeret
        self.assertIs(freeret.PROMPT_SPECS, catalog.PROMPT_SPECS)
        self.assertIs(freeret.FreeRetPromptSpec, catalog.FreeRetPromptSpec)
        for key, spec in catalog.PROMPT_SPECS.items():
            manifest = freeret.prompt_manifest(key)
            self.assertEqual(manifest["stage1_query"], spec.query_input("{query_text}")+spec.query_instruction)
            self.assertEqual(manifest["stage1_target"], spec.target_input("{target_text}")+spec.target_instruction)
            self.assertEqual(manifest["stage2_rerank"], spec.rerank_input("{target_text}", "{query_text}"))

    def test_legacy_encoders_add_constraints_before_substituting_user_text(self):
        from model import MLLMRetrievalModel
        class Captured(Exception): pass
        class Processor:
            def __call__(self, **kwargs):
                self.prompts=kwargs["text"]
                raise Captured()
        user="Summary above sentence in one word: quoted user text"
        for family in ("llava-hf-llava-v1.6-mistral-7b-hf", "llava-hf-llama3-llava-next-8b-hf"):
            for method,dataset,kind,separate in (
                ("encode_data","flickr","text",True),
                ("encode_data_concat","flickr","text",False),
                ("encode_data_for_tbpr","CUHK-PEDES","text",True),
                ("encode_data_concat_for_tbpr","CUHK-PEDES","text",False),
                ("encode_data_concat_for_t2it","webqa","query",False),
                ("encode_data_concat_for_it2t","remuq","corpus",False),
                ("encode_data_concat_for_it2t","llava","corpus",False)):
                with self.subTest(model=family,method=method,dataset=dataset):
                    encoder=fixtures.ToyModel("llava_next","language_model.layers")
                    wrapper=MLLMRetrievalModel(encoder)
                    args=options(False,True,family)
                    if separate: args.eol_type="prompteol"; args.calculate_type="separate"
                    data=data_args(dataset_name=dataset,tbpr_type="type",prompt_generation=False,prompt_generation_model="llama")
                    processor=Processor()
                    with redirect_stdout(io.StringIO()), self.assertRaises(Captured):
                        getattr(wrapper,method)([user],kind,processor,torch.device("cpu"),args,data)
                    original=list(processor.prompts)
                    args.use_freeret_prompt_constraints=True
                    with redirect_stdout(io.StringIO()), self.assertRaises(Captured):
                        getattr(wrapper,method)([user],kind,processor,torch.device("cpu"),args,data)
                    prefix=constraints.instruction(dataset,input_type=kind)+"\n"
                    description=constraints.task_description(dataset)
                    self.assertEqual(processor.prompts[0].replace(prefix,"").replace(description,""),original[0])
                    self.assertEqual(processor.prompts[0].count(description),1)
                    self.assertLess(processor.prompts[0].index(description),processor.prompts[0].index(user))
                    self.assertEqual(processor.prompts[0].count(prefix),original[0].count("Summary")-1)
                    self.assertIn(user,processor.prompts[0])
                    dense.configure_model(encoder,False)

    def test_both_pipeline_image_prompt_builders_preserve_original_summary_blocks(self):
        for module_name in ("three_stage_retrieval_pipeline","three_stage_retrieval_pipeline_adaptive_views"):
            module=importlib.import_module(module_name)
            for family in ("llava-hf-llava-v1.6-mistral-7b-hf","llava-hf-llama3-llava-next-8b-hf"):
                for task,dataset in (("ir","flickr"),("it2t","remuq"),("it2t","llava")):
                    args=options(False); args.dataset_name=dataset; args.prompt_type="prompt_5"
                    runtime=NS(model_path=family)
                    for method in (module._base_image_prompt,module._concat_image_prompt):
                        with self.subTest(module=module_name,family=family,task=task,method=method.__name__):
                            args.use_freeret_prompt_constraints=False
                            original=method(runtime,args,task)
                            args.use_freeret_prompt_constraints=True
                            actual=method(runtime,args,task)
                            prefix=constraints.instruction(dataset,input_type="query" if task=="it2t" else "image")+"\n"
                            description=constraints.task_description(dataset)
                            self.assertEqual(actual.replace(prefix,"").replace(description,""),original)
                            self.assertEqual(actual.count(description),1)
                            self.assertLess(actual.index(description),actual.index("<image>"))
                            self.assertEqual(actual.count(prefix),original.count("Summary"))
                            self.assertEqual(actual.count("<image>"),original.count("<image>"))

    def test_seven_entrypoints_parse_and_forward_new_flag(self):
        from transformers import HfArgumentParser
        import arguments
        from adaptive_view_overlap import OverlapViewArguments
        from adaptive_views import AdaptiveViewArguments
        from time_estimation import RerankTimeEstimationArguments
        namespace = dict(vars(arguments), HfArgumentParser=HfArgumentParser,
            AdaptiveViewArguments=AdaptiveViewArguments, OverlapViewArguments=OverlapViewArguments,
            RerankTimeEstimationArguments=RerankTimeEstimationArguments)
        for name in ("encode", "search", "rerank", "encode_adaptive_views", "encode_adaptive_views_overlap"):
            tree = ast.parse((ROOT/"src"/(name+".py")).read_text(encoding="utf-8"))
            declaration = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id=="parser" for target in node.targets)
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id=="HfArgumentParser")
            parser = eval(compile(ast.Expression(declaration), name, "eval"), namespace)
            for value in (None,"false","true"):
                with self.subTest(entry=name, flag=value):
                    args = ["--model_name_or_path","fixture","--output_dir","unused","--report_to","none","--use_cpu","true",
                            "--use_freeret_dense","true"]
                    if value is not None: args += ["--use_freeret_prompt_constraints",value]
                    self.assertIs(constraints.enabled(parser.parse_args_into_dataclasses(args)[0]),value=="true")
        for name in ("three_stage_retrieval_pipeline","three_stage_retrieval_pipeline_adaptive_views"):
            module = importlib.import_module(name)
            for flag in ("--use_freeret_prompt_constraints","--use-freeret-prompt-constraints"):
                for value in ("false","true"):
                    args = module._parser().parse_args(["--dataset-name","flickr","--use-freeret-dense","true",flag,value])
                    model,_ = module._retrieval_namespaces(args,eol_type="disassembleeol_separate",calculate_type="concat",sparse_manual=False)
                    self.assertIs(constraints.enabled(model),value=="true")
                    if value=="false": self.assertNotIn("use_freeret_prompt_constraints",vars(model))


class ArtifactTests(unittest.TestCase):
    def test_v2_prompt_identity_contains_descriptions_and_rejects_explicit_v1_artifacts(self):
        self.assertEqual(constraints.CONFIGURATION["version"],2)
        self.assertIn("shared_user_context",constraints.CONFIGURATION["description_placement"])
        self.assertEqual(dense.PROMPT_SUFFIX,constraints.SUFFIX)
        with temporary_directory() as directory:
            old=Path(directory)/("corpus"+dense.SUFFIX+"_freeret_constraints_v1")
            old.mkdir()
            dense.sidecar(old).write_text(json.dumps(dict(dense.CONFIGURATION,
                freeret_prompt_constraints=dict(version=1))))
            for alias in (old,old/"collection/index",old/"corpus_0.pkl"):
                self.assertTrue(dense.marked(alias)); self.assertTrue(dense.prompts_marked(alias))
                with self.assertRaisesRegex(ValueError,"v1 lacks task_description"):
                    dense.mode_path(alias,True,True)
                with self.assertRaisesRegex(ValueError,"v1 lacks task_description"):
                    dense.validate_artifact(alias,True,True)
            saved=dense.sidecar(old).read_bytes()
            with self.assertRaisesRegex(ValueError,"v1 lacks task_description"):
                dense.write_manifest(old,True,True)
            self.assertEqual(dense.sidecar(old).read_bytes(),saved)
            self.assertEqual(dense.original_path(old),Path(directory)/"corpus")
            new=dense.mode_path(Path(directory)/"corpus",True,True)
            self.assertNotEqual(new,old)
            self.assertTrue(new.name.endswith("_freeret_constraints_v2"))

    def test_legacy_manifests_do_not_block_new_indexing_but_explicit_v1_is_rejected(self):
        import adaptive_sparse_index_batch as index
        from test_adaptive_sparse_exact import artifacts
        with temporary_directory() as directory:
            env,records,loader=artifacts(directory,USE_FREERET_DENSE="true",USE_FREERET_PROMPT_CONSTRAINTS="true")
            current,_,_=records[0]
            old=current.with_name(current.name.replace("_freeret_constraints_v2","_freeret_constraints_v1"))
            old.mkdir()
            manifest=json.loads((current/views.MANIFEST).read_text(encoding="utf-8"))
            manifest["configuration"]["freeret_prompt_constraints"]={"version":1}
            manifest["fingerprint"]=views.digest(manifest["configuration"])
            views.atomic_json(old/views.MANIFEST,manifest)
            settings=index.Settings.from_environment("tbpr",env)
            skipped=[]
            jobs,missing=index.discover(settings,incompatible=skipped)
            self.assertFalse(missing)
            self.assertEqual([job.root for job in jobs],[current])
            self.assertEqual(len(skipped),1)
            self.assertIn("lacks task_description",skipped[0][1])
            settings=index.Settings.from_environment("tbpr",dict(env,SPARSE_OUTPUT_ROOT=str(old)))
            with self.assertRaisesRegex(ValueError,"v1 lacks task_description"):
                index.discover(settings)
            exact=index.Settings.from_environment("tbpr",env,exact_encoding=True)
            with redirect_stdout(io.StringIO()):
                jobs,missing=index.discover_exact(exact,identity_loader=loader)
            self.assertEqual([job.root for job in jobs],[current])

    def test_index_configuration_and_artifact_validation_select_the_prompt_mode(self):
        import adaptive_sparse_index_batch as index
        for group in ("ir","tbpr","fused"):
            env=dict(USE_FREERET_DENSE="true",USE_FREERET_PROMPT_CONSTRAINTS="true")
            self.assertTrue(index.Settings.from_environment(group,env).use_freeret_prompt_constraints)
            with self.assertRaisesRegex(ValueError,"requires USE_FREERET_DENSE"):
                index.Settings.from_environment(group,dict(env,USE_FREERET_DENSE="false"))
        with temporary_directory() as directory:
            root=dense.mode_path(Path(directory)/("a"*16),True,True)
            (root/"collection").mkdir(parents=True)
            (root/"collection/corpus_0.jsonl").write_text('{"id":"a","vector":{"word":10}}\n')
            configuration=dense.configuration(dict(views_fingerprint="a"*64),True,True)
            manifest=dict(configuration=configuration,fingerprint=views.digest(configuration),num_shards=1)
            dense.write_manifest(root,True,True)
            self.assertIsNone(index.check_artifact(root,manifest,configuration,selection={}))
            (root/"collection/index").mkdir()
            (root/"collection/index/segments_1").touch()
            self.assertEqual(index.check_artifact(root,manifest,configuration,selection={}),root/"collection/index")

    def test_paths_and_sidecars_distinguish_both_modes_and_skip_sparse_reuse(self):
        with temporary_directory() as directory:
            old = Path(directory)/"corpus"
            layer = dense.mode_path(old,True)
            new = dense.mode_path(old,True,True)
            self.assertEqual(new.name,layer.name+constraints.SUFFIX)
            for alias in ("", "collection", "collection/index", "corpus_0.pkl"):
                path = new/alias
                self.assertEqual(dense.mode_path(path,True,True),path)
                self.assertTrue(dense.marked(path)); self.assertTrue(dense.prompts_marked(path))
            self.assertEqual(dense.mode_path(layer,True,True),new)
            self.assertEqual(dense.original_path(new),old)
            dense.write_manifest(new,True,True)
            dense.write_manifest(layer,True)
            dense.validate_artifact(new,True,True)
            for path,use in ((new,False),(layer,True)):
                with self.assertRaisesRegex(ValueError,"mode mismatch"):
                    dense.validate_artifact(path,True,use)
            self.assertFalse(dense.reuse_sparse_index(new))
            old_config = dict(version=1)
            self.assertNotIn("freeret_prompt_constraints",dense.configuration(old_config,True))
            self.assertEqual(dense.configuration(old_config,True,True)["freeret_prompt_constraints"],constraints.CONFIGURATION)

    def test_adaptive_metadata_and_overlap_selection_modes_are_consistent(self):
        import adaptive_view_overlap as overlap
        data = data_args()
        selection = dict(selected=views.view_pool()[:5],selection_modality="both",views_fingerprint="a"*64)
        before = adaptive.metadata(options(False),data,"ir","image",{"model":"fixture"},selection)
        after = adaptive.metadata(options(),data,"ir","image",{"model":"fixture"},selection)
        self.assertNotEqual(before["fingerprint"],after["fingerprint"])
        self.assertEqual(after["configuration"]["views_fingerprint"],before["configuration"]["views_fingerprint"])
        training = NS(task_type="ir",encode_type="image")
        for path in adaptive.artifact_dirs(options(),data,training,selection):
            self.assertTrue(path.name.endswith(dense.SUFFIX+constraints.SUFFIX))
            self.assertEqual(str(path).count(constraints.SUFFIX),1)
        old_strategy = overlap.OverlapSelectionStrategy(overlap.OverlapViewArguments(),options(False),data,"ir")
        strategy = overlap.OverlapSelectionStrategy(overlap.OverlapViewArguments(),options(),data,"ir")
        self.assertNotEqual(old_strategy.configuration,strategy.configuration)
        selection["overlap_configuration"] = old_strategy.configuration
        with self.assertRaisesRegex(ValueError,"Overlap selection prompt"):
            constraints.validate_selection(selection,options())
        selection["overlap_configuration"] = strategy.configuration
        constraints.validate_selection(selection,options())
        with self.assertRaisesRegex(ValueError,"Overlap selection prompt"):
            constraints.validate_selection(selection,options(False))

    def test_batch_selection_keys_only_change_for_overlap_and_forward_flags(self):
        import adaptive_retrieval_batch as batch
        for group in ("ir","tbpr","fused"):
            for overlap in (False,True):
                for pmi in (False,True):
                    env=dict(VIEW_EXAMPLE_MODE="fixed",MODEL_FAMILY="qwen25",USE_FREERET_DENSE="true",USE_VIEW_OVERLAP_SELECTION=str(overlap).lower(),USE_VIEW_PMI=str(pmi).lower())
                    old=batch.Settings.from_environment(group,env)
                    new=batch.Settings.from_environment(group,dict(env,USE_FREERET_PROMPT_CONSTRAINTS="true"))
                    for dataset in old.datasets:
                        for task,query,kind in batch.job_types(old,dataset):
                            before=batch.namespaces(old,"qwen25",dataset,task,kind)
                            after=batch.namespaces(new,"qwen25",dataset,task,kind)
                            self.assertTrue(constraints.enabled(after[0]))
                            a=batch.selection_location(old,*before,{"model":"fixture"})[:2]
                            b=batch.selection_location(new,*after,{"model":"fixture"})[:2]
                            self.assertEqual(a==b,not overlap)

    def test_projected_selection_samples_keep_roles_without_changing_old_samples(self):
        class Dataset:
            query_id_list=["q"]; candidate_id_list=["c"]
            id2query={"q":{"text":"query words"}}
            id2candidate={"c":{"text":"corpus words","image":"candidate-image"}}
        api=NS(build_dataset=lambda args:Dataset(),_plain_text=str)
        args=NS(task_type="t2it",use_freeret_dense=True)
        old=adaptive.source_pools(args,api)
        args.use_freeret_prompt_constraints=True
        new=adaptive.source_pools(args,api)
        self.assertTrue(all("_freeret_role" not in record for group in old for record in group))
        self.assertEqual([record["_freeret_role"] for record in new[0]],["target","query"])
        self.assertEqual(new[1][0]["_freeret_role"],"target")
        self.assertEqual([[record["id"] for record in group] for group in old],[[record["id"] for record in group] for group in new])


class LauncherTests(unittest.TestCase):
    def test_local_configuration_and_external_environment_reach_batch_dispatchers(self):
        bash=shutil.which("bash") if os.name!="nt" else "D:/Git/bin/bash.exe"
        if not bash or not Path(bash).is_file(): self.skipTest("Bash unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); scripts=root/"scripts"; scripts.mkdir()
            shutil.copyfile(ROOT/"scripts/freeret_dense_mode.sh",scripts/"freeret_dense_mode.sh")
            capture=root/"capture.sh"
            capture.write_text('#!/usr/bin/env bash\nprintf \'%s|%s\' "$USE_FREERET_DENSE" "$USE_FREERET_PROMPT_CONSTRAINTS"\n',encoding="utf-8")
            capture.chmod(0o755)
            names=[f"encode_adaptive_views{overlap}_{group}.sh" for overlap in ("","_overlap") for group in ("ir","tbpr","fused")]
            names += [f"three_stage_retrieval_pipeline_adaptive_views{overlap}_{group}{multi}.sh"
                      for overlap,multi in (("",""),("_overlap","_multi")) for group in ("ir","tbpr","fused")]
            names += [f"sparse_index_adaptive_views_overlap_{group}.sh" for group in ("ir","tbpr","fused")]
            names += [f"three_stage_adaptive_{group}_multi.sh" for group in ("ir","tbpr","fused")]
            for local in (False,True):
                for name in names:
                    with self.subTest(script=name,local_configuration=local):
                        body=(ROOT/"scripts"/name).read_text(encoding="utf-8")
                        if local:
                            body=body.replace('${USE_FREERET_DENSE-false}','${USE_FREERET_DENSE-true}')
                            body=body.replace('${USE_FREERET_PROMPT_CONSTRAINTS-false}','${USE_FREERET_PROMPT_CONSTRAINTS-true}')
                        path=scripts/name; path.write_text(body,encoding="utf-8")
                        env={k:v for k,v in os.environ.items() if k not in {"USE_FREERET_DENSE","USE_FREERET_PROMPT_CONSTRAINTS"}}
                        env.update(PYTHON_BIN=capture.as_posix(),LLAVA_IMAGE_MODE="anyres",DRY_RUN="true")
                        if not local: env.update(USE_FREERET_DENSE="true",USE_FREERET_PROMPT_CONSTRAINTS="true")
                        result=subprocess.run([bash,path.as_posix()],cwd=root,env=env,capture_output=True,text=True,timeout=30)
                        self.assertEqual(result.returncode,0,result.stderr)
                        self.assertEqual(result.stdout,"true|true")


class ConstraintRealModelTests(fixtures.RealModelTests):
    def compare(self, model, processor, records, family, adapter="adaptive"):
        for task,dataset in (("ir","flickr"),("tbpr","CUHK-PEDES"),("t2it","webqa"),("it2t","remuq"),("it2t","llava")):
            for location,count in (("after_pad",1),("before_pad",5)):
                with self.subTest(task=task,dataset=dataset,family=family,adapter=adapter,location=location,views=count):
                    data=data_args(dataset_name=dataset,reps_loc=location,tbpr_type="type")
                    args=options(False,True,family)
                    selected={"selected":views.view_pool()[:count]}
                    factory=(lambda:adaptive.AdaptiveEncoder(model,processor,args,data,task,selected)) if adapter=="adaptive" else (
                        lambda:NativeQwenEncoder(model,processor,args,data,task))
                    with redirect_stdout(io.StringIO()): baseline=factory()
                    original=baseline.prepare(records)
                    args.use_freeret_prompt_constraints=True
                    encoder=factory(); batch=encoder.prepare(records)
                    for row,record in enumerate(records):
                        prefix=constraints.instruction(dataset,record)+"\n"
                        description=constraints.task_description(dataset)
                        self.assertEqual(batch.prompts[row].count(prefix),len(batch.spans[row]))
                        self.assertEqual(batch.prompts[row].count(description),1)
                        self.assertLess(batch.prompts[row].index(description),batch.prompts[row].index(prefix))
                        if record.get("image") is not None:
                            marker="<|vision_start|>" if family.startswith("qwen") else "<image>"
                            self.assertLess(batch.prompts[row].index(description),batch.prompts[row].index(marker))
                        if record.get("text"):
                            context_start=batch.prompts[row].index(description)+len(description)
                            self.assertIn(record["text"],batch.prompts[row][context_start:batch.prompts[row].index(prefix)])
                        self.assertEqual(batch.prompts[row].replace(prefix,"").replace(description,""),original.prompts[row])
                        self.assertEqual(batch.prompts[row].count("<image>"),original.prompts[row].count("<image>"))
                        self.assertEqual(batch.prompts[row].count("<|image_pad|>"),original.prompts[row].count("<|image_pad|>"))
                    captured=[]
                    _,norm=dense._last_norm(model)
                    handle=norm.register_forward_pre_hook(lambda _m,inputs:captured.append(inputs[0].detach().clone()))
                    try: sparse,new_dense=encoder.encode(records)
                    finally: handle.remove()
                    offset=2 if family.startswith("llava") else 1
                    expected=torch.nn.functional.normalize(torch.stack([
                        captured[0][row,spans[0][1]-offset] for row,spans in enumerate(batch.spans)]),dim=-1)
                    self.assertTrue(torch.equal(new_dense,expected))
                    self.assertTrue(torch.isfinite(new_dense).all())
                    self.assertTrue(all(torch.isfinite(value).all() for value in sparse))
                    self.assertEqual(len(sparse),len(records))
                    dense.configure_model(model,False)


if __name__=="__main__":
    unittest.main()
