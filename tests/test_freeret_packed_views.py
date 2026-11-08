"""Native-model parity, causal/view masks, BPE boundaries and v1/v2 isolation."""
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
import test_freeret_dense as fixtures
from test_freeret_dense import ROOT, adaptive, dense, views, NativeQwenEncoder
import freeret_full_prompt as full
from qwen_vl_encoding import _forward_inputs


class CountingProcessor:
    def __init__(self, processor):
        self.processor, self.calls = processor, 0
    def __getattr__(self, key):
        return getattr(self.processor, key)
    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.processor(*args, **kwargs)


class PackedRealModelTests(fixtures.RealModelTests):
    def setUp(self):
        torch.manual_seed(17)

    def compare(self, model, processor, records, family, adapter="adaptive"):
        if adapter != "adaptive":
            return
        if family.startswith("llava"):
            if "mistral" in family:
                processor.chat_template="{% for m in messages %}[INST]{% for c in m['content'] %}{% if c['type']=='image' %}<image>\n{% else %}{{ c['text'] }}{% endif %}{% endfor %}[/INST]{% endfor %}"
            else:
                processor.chat_template="{% for m in messages %}<|start_header_id|>user<|end_header_id|>\n\n{% for c in m['content'] %}{% if c['type']=='image' %}<image>\n{% else %}{{ c['text'] }}{% endif %}{% endfor %}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{% endfor %}"
        scenarios = [("ir","flickr",count) for count in (0,1,3,7)] + [
            ("tbpr","CUHK-PEDES",2),("t2it","webqa",2),("it2t","remuq",2),("it2t","llava",2)]
        for location in ("after_pad","before_pad"):
            for task,dataset,count in scenarios:
                with self.subTest(family=family,location=location,dataset=dataset,count=count):
                    data=fixtures.data_args(dataset_name=dataset,reps_loc=location,tbpr_type="type",print_sparse=False)
                    args=fixtures.model_args(True,family); args.use_freeret_full_prompt=True
                    selected=copy.deepcopy(views.view_pool()[:count])
                    if count:
                        encoder=adaptive.AdaptiveEncoder(model,processor,args,data,task,{"selected":selected})
                        aspects=[row["aspect"] for row in selected]
                    else:
                        args.eol_type="prompteol"; args.calculate_type="separate"
                        encoder=NativeQwenEncoder(model,processor,args,data,task)
                        aspects=[]
                    encoder.family=views.family_of(model.config)
                    proxy=CountingProcessor(processor)
                    encoder.processor=proxy
                    calls=[]
                    handle=model.register_forward_pre_hook(lambda module,values,kwargs: calls.append(kwargs),with_kwargs=True)
                    try:
                        sparse,actual_dense=encoder.encode(records)
                    finally:
                        handle.remove()
                    self.assertEqual(len(calls),1)
                    self.assertEqual(proxy.calls,1)
                    self.assertFalse(calls[0]["use_cache"])
                    self.assertEqual(calls[0]["attention_mask"].ndim,4)
                    packed=encoder.prepare(records)
                    self.check_mask(packed,calls[0])
                    reference_sparse=[[] for _ in records]
                    for index,aspect in enumerate([None,*aspects]):
                        standalone=full.prepare(processor,views.family_of(model.config),records,dataset,aspect,
                            padding_side="left" if location=="after_pad" else "right")
                        original=standalone.inputs
                        if family.startswith("qwen"):
                            original=_forward_inputs(model,original)
                        with torch.inference_mode():
                            output=dense.forward(model,**original,use_cache=False,output_hidden_states=True,return_dict=True)
                        ends=torch.tensor([row[0][1]-1 for row in standalone.spans])
                        rows=torch.arange(len(records))
                        # The image features and every visible token are unchanged.
                        for row,spans in enumerate(packed.spans):
                            visible=original["input_ids"][row][original["attention_mask"][row].bool()]
                            shared=packed.inputs["input_ids"][row,:spans[0][0]][packed.inputs["attention_mask"][row,:spans[0][0]].bool()]
                            start,end=spans[index]
                            effective=torch.cat([shared,packed.inputs["input_ids"][row,start:end]])
                            self.assertTrue(torch.equal(visible,effective))
                        if index==0:
                            expected=F.normalize(output.hidden_states[-1][rows,ends],dim=-1)
                            self.assertTrue(torch.allclose(actual_dense,expected,atol=3e-6,rtol=3e-5),family)
                        if index or not aspects:
                            values=torch.log1p(torch.relu(output.logits[rows,ends]))
                            for own,value in zip(reference_sparse,values): own.append(value)
                    for own,reference in zip(sparse,reference_sparse):
                        self.assertTrue(torch.allclose(own,torch.stack(reference),atol=3e-6,rtol=3e-5),family)
                    if count or family.startswith("qwen"):
                        self.assertEqual(adaptive.sparse_vectors(encoder,records,sparse),
                            adaptive.sparse_vectors(encoder,records,[torch.stack(row) for row in reference_sparse]))
                    if task=="ir" and count==3:
                        self.check_isolation(encoder,records,sparse,actual_dense)
                    dense.configure_model(model,False)
        if family=="llava_mistral":
            self.check_sliding_window(model,processor,records,family)

    def check_sliding_window(self,model,processor,records,family):
        config=model.config.text_config
        previous=config.sliding_window
        config.sliding_window=32
        try:
            data=fixtures.data_args(print_sparse=False)
            args=fixtures.model_args(True,family); args.use_freeret_full_prompt=True
            selected=views.view_pool()[:2]
            encoder=adaptive.AdaptiveEncoder(model,processor,args,data,"ir",{"selected":selected})
            sparse,actual_dense=encoder.encode(records)
            batch=encoder.prepare(records)
            call=full.masked_inputs(model,batch,family)
            minimum=torch.finfo(call["attention_mask"].dtype).min
            for row,spans in enumerate(batch.spans):
                self.assertTrue(any(end-start>32 for start,end in spans))
                for start,end in spans:
                    self.assertEqual(float(call["attention_mask"][row,0,end-1,end-33]),minimum)
            for index,aspect in enumerate([None,*[row["aspect"] for row in selected]]):
                single=full.prepare(processor,views.family_of(model.config),records,"flickr",aspect)
                with torch.inference_mode():
                    output=dense.forward(model,**single.inputs,use_cache=False,output_hidden_states=True,return_dict=True)
                ends=torch.tensor([spans[0][1]-1 for spans in single.spans]); rows=torch.arange(len(records))
                if index==0:
                    expected=F.normalize(output.hidden_states[-1][rows,ends],dim=-1)
                    self.assertTrue(torch.allclose(actual_dense,expected,atol=3e-6,rtol=3e-5))
                else:
                    expected=torch.log1p(torch.relu(output.logits[rows,ends]))
                    self.assertTrue(torch.allclose(torch.stack([row[index-1] for row in sparse]),expected,atol=3e-6,rtol=3e-5))
        finally:
            config.sliding_window=previous
            dense.configure_model(model,False)

    def check_mask(self,batch,call):
        mask=call["attention_mask"]
        minimum=torch.finfo(mask.dtype).min
        positions=call["position_ids"]
        for row,spans in enumerate(batch.spans):
            common=spans[0][0]
            pad=~batch.inputs["attention_mask"][row].bool()
            self.assertTrue((mask[row,0,:,pad]==minimum).all())
            for start,end in spans:
                self.assertTrue((mask[row,0,start:end,common:start]==minimum).all())
                self.assertTrue((mask[row,0,start:end,end:]==minimum).all())
                self.assertEqual(float(mask[row,0,start,start]),0.)
                if positions.ndim==3:
                    actual=positions[:,row,start:end]
                    expected=positions[:,row,spans[0][0]][:,None]+torch.arange(end-start)[None,:]
                else:
                    actual=positions[row,start:end]
                    expected=positions[row,spans[0][0]]+torch.arange(end-start)
                self.assertTrue(torch.equal(actual,expected))

    def check_isolation(self,encoder,records,baseline,dense_before):
        original=copy.deepcopy(encoder.selected)
        encoder.selected=copy.deepcopy(original)
        encoder.selected[1]["aspect"]="a b a b a b"
        modified,dense_after=encoder.encode(records)
        self.assertTrue(torch.allclose(dense_before,dense_after,atol=3e-6,rtol=3e-5))
        self.assertTrue(any(not torch.allclose(first[1],second[1],atol=3e-6,rtol=3e-5)
                            for first,second in zip(baseline,modified)))
        for first,second in zip(baseline,modified):
            self.assertTrue(torch.allclose(first[[0,2]],second[[0,2]],atol=3e-6,rtol=3e-5))
        encoder.selected=list(reversed(original))
        reordered,dense_after=encoder.encode(records)
        self.assertTrue(torch.allclose(dense_before,dense_after,atol=3e-6,rtol=3e-5))
        for first,second in zip(baseline,reordered):
            self.assertTrue(torch.allclose(first.flip(0),second,atol=3e-6,rtol=3e-5))
        encoder.selected=original


class TokenAndVersionTests(unittest.TestCase):
    def bpe_processor(self):
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from transformers import PreTrainedTokenizerFast
        vocabulary=["[UNK]","[PAD]",*dict.fromkeys(full.CHAT_SUFFIX+"abcd \n"),"bc"]
        backend=Tokenizer(BPE({word:index for index,word in enumerate(vocabulary)},[("b","c")],unk_token="[UNK]"))
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token="[UNK]",pad_token="[PAD]")
        class Processor:
            def __init__(self): self.tokenizer=tokenizer
            def __call__(self,**kwargs): return dict(tokenizer(**kwargs))
        return Processor()

    def test_bpe_merge_across_prefix_boundary_keeps_standalone_token_sequences(self):
        processor=self.bpe_processor()
        record={"text":"data"}
        with patch.object(full,"_prompt_parts",side_effect=lambda dataset,record,aspect=None:("ab","c"+(aspect or "d"))):
            packed=full.prepare_packed(processor,"qwen2_vl",[record],"flickr",[["a","b"]])
            spans=packed.spans[0]
            self.assertEqual(spans[0][0],1) # 'b' merges with instruction 'c', so shared tokens end at 'a'.
            for index,aspect in enumerate((None,"a","b")):
                reference=full.prepare(processor,"qwen2_vl",[record],"flickr",aspect)
                start,end=spans[index]
                visible=torch.cat([packed.inputs["input_ids"][0,:spans[0][0]],packed.inputs["input_ids"][0,start:end]])
                self.assertTrue(torch.equal(reference.inputs["input_ids"][0],visible))

    def test_inconsistent_views_and_changed_processor_boundaries_fail_without_forward(self):
        processor=self.bpe_processor()
        with self.assertRaisesRegex(ValueError,"same view count"):
            full.prepare_packed(processor,"qwen2_vl",[{},{}],"flickr",[["a"],[]])
        original=full.prepare
        def changed(*args,**kwargs):
            result=original(*args,**kwargs)
            result.inputs["input_ids"][0,-1]=processor.tokenizer.unk_token_id
            return result
        with patch.object(full,"prepare",side_effect=changed),patch.object(full,"_prompt_parts",return_value=("ab","c")):
            with self.assertRaisesRegex(ValueError,"Processor changed"):
                full.prepare_packed(processor,"qwen2_vl",[{}],"flickr",[["a"]])

    def test_v2_namespaces_and_old_v1_paths_cannot_be_silently_reinterpreted(self):
        base=Path("example/corpus")
        self.assertEqual(str(dense.mode_path(base,True,False,True)),str(base)+dense.SUFFIX+"_freeret_full_prompt_v2")
        old=Path(str(base)+dense.SUFFIX+"_freeret_full_prompt_v1")
        for path in (old,old/"collection/index",old/"corpus_0.pkl"):
            with self.assertRaisesRegex(ValueError,"v1 uses independent forwards"):
                dense.mode_path(path,True,False,True)
            with self.assertRaises(ValueError):
                dense.validate_artifact(path,True,False,True)
        self.assertEqual(dense.configuration({"legacy":1},False,False,False),{"legacy":1})
        self.assertEqual(dense.mode_path(base,True,False,False),Path(str(base)+dense.SUFFIX))
        self.assertEqual(full.CONFIGURATION["version"],2)
        self.assertEqual(full.CONFIGURATION["forwarding"],"packed_shared_context_v1")

    def test_shell_generates_v2_and_preserves_explicit_v1_for_a_clear_python_error(self):
        bash=shutil.which("bash") or "D:/Git/bin/bash.exe"
        helper=(ROOT/"scripts/freeret_dense_mode.sh").as_posix()
        command=f'source "{helper}"; dense_mode_path corpus; echo; dense_mode_path corpus_dense_freeret_pre_mlp_freeret_full_prompt_v1'
        import os
        env=dict(os.environ,USE_FREERET_DENSE="true",USE_FREERET_FULL_PROMPT="true",USE_FREERET_PROMPT_CONSTRAINTS="false")
        result=subprocess.run([bash,"-c",command],env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout.splitlines(),["corpus_dense_freeret_pre_mlp_freeret_full_prompt_v2",
                                                    "corpus_dense_freeret_pre_mlp_freeret_full_prompt_v1"])


if __name__=="__main__": unittest.main()
