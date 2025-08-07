import gc
import json
import logging
import os
import pickle
import string
import sys
import itertools
from contextlib import nullcontext

from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from PIL import Image
import faiss

from tqdm import tqdm
from transformers import (
    HfArgumentParser,
)
from transformers import LlavaProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, \
    LlavaNextForConditionalGeneration, Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration, AutoModel, \
    AutoProcessor, LlamaForCausalLM
from arguments import PromptRepsLLMDataArguments, ModelArguments
import torch.distributed as dist
import torch.nn as nn
from arguments import TrainingArguments, LogitInformationAnalysisArguments
from dataset import CrossModalRetrievalDataset
import torch
import torch.utils.data as Data
import torch.nn.functional as F

from template import text_prompt, img_prompt, text_prompt_no_one_word, img_prompt_no_one_word, \
    img_prompt_no_special_llava_v1_5, text_prompt_no_special_llava_v1_5, text_prompt_qwen_v2_5, img_prompt_qwen_v2_5, \
    img_prompt_intern_vl_v2_5, text_prompt_intern_vl_v2_5, task_image_prompts, llama3_template, task_text_prompts, \
    task_text_prompts_copy, task_image_prompts_copy, \
    llama3_retrieval_disassemble_image_prompts, llama3_retrieval_disassemble_text_prompts
from model import MLLMRetrievalModel
from utils import split_model, load_image
from peft import PeftModel, PeftConfig
from encode import get_filtered_ids, get_img_valid_tokens_values, get_img_valid_disassemble_tokens_values, \
    get_img_valid_tokens_values_with_cluster, get_text_valid_tokens_values, get_text_valid_disassemble_tokens_values, \
    get_text_valid_tokens_values_with_cluster


def main():
    parser = HfArgumentParser(
        (ModelArguments, PromptRepsLLMDataArguments, TrainingArguments, LogitInformationAnalysisArguments))

    model_args, data_args, training_args, logit_information_analysis_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: PromptRepsLLMDataArguments
    training_args: TrainingArguments
    logit_information_analysis_args: LogitInformationAnalysisArguments

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device_map = "cuda"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(os.environ.get("WORLD_SIZE"))
    ddp = world_size != 1
    print(ddp)
    # if ddp and False:
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        # gradient_accumulation_steps = gradient_accumulation_steps // world_size

        if not dist.is_initialized():
            torch.distributed.init_process_group("nccl")
        rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        device_id = rank % torch.cuda.device_count()
        device = torch.device(device_id)
        torch.cuda.set_device(device)

        print(device)

    # 下面这部分指定采用的模型精度
    if training_args.bf16:
        torch_type = torch.bfloat16
    elif training_args.fp16:
        torch_type = torch.float16
    else:
        torch_type = torch.float32

    # 指定模型
    if 'llava-hf-llava-1.5-7b-hf' in model_args.model_name_or_path:
        encoder = LlavaForConditionalGeneration.from_pretrained(model_args.model_name_or_path, device_map=device_map,
                                                                torch_dtype=torch_type)
        processor = LlavaProcessor.from_pretrained(model_args.model_name_or_path)

    elif 'Qwen2.5-VL-7B-Instruct' in model_args.model_name_or_path or 'Qwen2.5-VL-3B-Instruct' in model_args.model_name_or_path:
        encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_args.model_name_or_path,
                                                                     device_map=device_map,
                                                                     torch_dtype=torch_type)
        processor = Qwen2_5_VLProcessor.from_pretrained(model_args.model_name_or_path)
    elif 'InternVL2_5-8B' in model_args.model_name_or_path or 'InternVL2_5-4B' in model_args.model_name_or_path:
        # device_map = split_model('InternVL2_5-8B')
        encoder = AutoModel.from_pretrained(model_args.model_name_or_path,
                                            device_map=device_map,
                                            torch_dtype=torch_type,
                                            trust_remote_code=True,
                                            low_cpu_mem_usage=True,
                                            )
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path,
                                                  trust_remote_code=True, )
    else:
        encoder = LlavaNextForConditionalGeneration.from_pretrained(model_args.model_name_or_path,
                                                                    device_map=device_map,
                                                                    torch_dtype=torch_type)
        processor = LlavaNextProcessor.from_pretrained(model_args.model_name_or_path)
        if 'royokong-e5-v' in model_args.model_name_or_path:
            setattr(processor, "patch_size", 14)  # hack for pass

    if data_args.reps_loc == 'after_pad':
        processor.tokenizer.padding_side = "left"
        processor.tokenizer.padding = True
        if dist.get_rank() == 0:
            print(processor.tokenizer.unk_token_id)
            print(processor.tokenizer.eos_token_id)
            print(processor.tokenizer.pad_token_id)

    # 加载词表并获取过滤后的单词id，但目前尚不清楚filtered_ids是做什么的
    if 'InternVL2_5-8B' in model_args.model_name_or_path or 'InternVL2_5-4B' in model_args.model_name_or_path:
        vocab_dict = processor.get_vocab()
        filtered_ids = get_filtered_ids(processor)
    else:
        vocab_dict = processor.tokenizer.get_vocab()
        filtered_ids = get_filtered_ids(processor.tokenizer)
    vocab_dict = {v: k for k, v in vocab_dict.items()}
    print(len(vocab_dict))

    model = MLLMRetrievalModel(encoder)
    model = model.eval()
    print(model.is_ddp)

    with torch.no_grad():
        if 'llava-hf-llava-1.5-7b-hf' in model_args.model_name_or_path or 'llava-hf-llava-v1.6-vicuna-7b-hf' in model_args.model_name_or_path:
            prompt = img_prompt_no_special_llava_v1_5
        elif 'Qwen2.5-VL-7B-Instruct' in model_args.model_name_or_path or 'Qwen2.5-VL-3B-Instruct' in model_args.model_name_or_path:
            prompt = img_prompt_qwen_v2_5
        elif 'InternVL2_5-8B' in model_args.model_name_or_path or 'InternVL2_5-4B' in model_args.model_name_or_path:
            prompt = img_prompt_intern_vl_v2_5
            if dist.get_rank() == 0:
                print(prompt)
        else:
            prompt = img_prompt

        if 'disassembleeol' in model_args.eol_type:
            prompts = llama3_retrieval_disassemble_image_prompts
        else:
            prompts = llama3_retrieval_disassemble_image_prompts

        if logit_information_analysis_args.logit_information_analysis_type == 'text':
            text = logit_information_analysis_args.logit_information_analysis_text
            logits, raw_logits = model.encode_data_for_logit_information_analysis([text], 'text', processor, device,
                                                                                  model_args, data_args)

            if 'disassembleeol_concrete' in model_args.eol_type:
                disassemble_logits = logits[1:]
                logits = logits[:1]
                raw_disassemble_logits = raw_logits[1:]
                raw_logits = raw_logits[:1]
            elif 'disassembleeol' in model_args.eol_type:
                disassemble_logits = logits
                raw_disassemble_logits = raw_logits

            disassemble_probs = F.softmax(disassemble_logits, dim=-1)
            disassemble_raw_probs = F.softmax(raw_disassemble_logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
            raw_probs = F.softmax(raw_logits, dim=-1)

            # Step 2: 计算每个类别的信息量 (单位：nats)
            disassemble_information_content = -torch.log(disassemble_probs)
            disassemble_raw_information_content = -torch.log(disassemble_raw_probs)
            information_content = -torch.log(probs)
            raw_information_content = -torch.log(raw_probs)

            disassemble_entropy = -torch.sum(disassemble_probs * torch.log(disassemble_probs), dim=-1)
            disassemble_raw_entropy = -torch.sum(disassemble_raw_probs * torch.log(disassemble_raw_probs), dim=-1)
            entropy = -torch.sum(probs * torch.log(probs), dim=-1)
            raw_entropy = -torch.sum(raw_probs * torch.log(raw_probs), dim=-1)

            print('Entropy and raw entropy: ')
            print(disassemble_entropy)
            print(disassemble_raw_entropy)
            print(entropy)
            print(raw_entropy)

            print('Information content and raw information content: ')
            print(disassemble_information_content)
            print(disassemble_raw_information_content)
            print(information_content)
            print(raw_information_content)

        else:
            img_path = logit_information_analysis_args.logit_information_analysis_image
            raw_images = [Image.open(img_path).convert('RGB')]
            if 'disassembleeol' in model_args.eol_type:
                # 这是参考metaeol的思路，试图将图文中的不同元素拆解出来，目前先把这个处理放在稀疏检索上，然后再看看密集检索是否使用
                # all_disassembleeol表示稀疏特征和密集特征都用各个子方面（角度）的结果
                if model_args.eol_type != 'all_disassembleeol' and model_args.eol_type != 'all_disassembleeol_origin_text':
                    img_inputs = processor(images=raw_images, text=[prompt], return_tensors="pt", padding=True)
                    imgs = img_inputs.to(device)
                    if model_args.eol_type == 'disassembleeol_concrete' or model_args.eol_type == 'disassembleeol_concrete_origin_text':
                        logits, raw_logits = model.encode_data_for_logit_information_analysis(imgs, 'image', processor,
                                                                                              device, model_args,
                                                                                              data_args)
                    elif model_args.eol_type == 'all_disassembleeol_concrete' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text':
                        logits, raw_logits = model.encode_data_for_logit_information_analysis(imgs, 'image', processor,
                                                                                              device, model_args,
                                                                                              data_args)
                    else:
                        logits, raw_logits = model.encode_data_for_logit_information_analysis(imgs, 'image', processor,
                                                                                              device, model_args,
                                                                                              data_args)

                disassemble_raw_images = [raw_image for raw_image in raw_images for _ in range(len(prompts))]
                disassemble_img_inputs = processor(images=disassemble_raw_images,
                                                   text=prompts,
                                                   return_tensors="pt",
                                                   padding=True)
                disassemble_imgs = disassemble_img_inputs.to(device)
                if model_args.eol_type == 'all_disassembleeol' or model_args.eol_type == 'all_disassembleeol_origin_text' or model_args.eol_type == 'all_disassembleeol_concrete' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text':
                    disassemble_logits, raw_disassemble_logits = model.encode_data(disassemble_imgs, 'image',
                                                                                   processor, device,
                                                                                   model_args, data_args)
                else:
                    disassemble_logits, raw_disassemble_logits = model.encode_data(disassemble_imgs, 'image', processor,
                                                                                   device,
                                                                                   model_args, data_args)
            else:
                pass

            disassemble_probs = F.softmax(disassemble_logits, dim=-1)
            disassemble_raw_probs = F.softmax(raw_disassemble_logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
            raw_probs = F.softmax(raw_logits, dim=-1)

            # Step 2: 计算每个类别的信息量 (单位：nats)
            disassemble_information_content = -torch.log(disassemble_probs)
            disassemble_raw_information_content = -torch.log(disassemble_raw_probs)
            information_content = -torch.log(probs)
            raw_information_content = -torch.log(raw_probs)

            disassemble_entropy = -torch.sum(disassemble_probs * torch.log(disassemble_probs), dim=-1)
            disassemble_raw_entropy = -torch.sum(disassemble_raw_probs * torch.log(disassemble_raw_probs), dim=-1)
            entropy = -torch.sum(probs * torch.log(probs), dim=-1)
            raw_entropy = -torch.sum(raw_probs * torch.log(raw_probs), dim=-1)

            print('Entropy and raw entropy: ')
            print(disassemble_entropy)
            print(disassemble_raw_entropy)
            print(entropy)
            print(raw_entropy)

            print('Information content and raw information content: ')
            print(disassemble_information_content)
            print(disassemble_raw_information_content)
            print(information_content)
            print(raw_information_content)
        if 'disassembleeol' in model_args.eol_type:
            if training_args.encode_type == 'text':
                vector = dict()
                logit = logits
                disassemble_logit = disassemble_logits
                text = logit_information_analysis_args.logit_information_analysis_text
                if model_args.eol_type == 'disassembleeol_concrete' or model_args.eol_type == 'disassembleeol_concrete_origin_text' or model_args.eol_type == 'all_disassembleeol_concrete' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text':
                    tokens, values = get_text_valid_disassemble_tokens_values(text, processor.tokenizer,
                                                                              disassemble_logit,
                                                                              vocab_dict,
                                                                              data_args,
                                                                              filtered_ids, logit,
                                                                              model_args)
                else:
                    tokens, values = get_text_valid_disassemble_tokens_values(text, processor.tokenizer,
                                                                              disassemble_logit,
                                                                              vocab_dict,
                                                                              data_args,
                                                                              filtered_ids, None,
                                                                              model_args)

                for token, v in zip(tokens, values):
                    if token in vector.keys():
                        if data_args.sparse_value_type == 'replace':
                            vector[token] = int(v)
                        elif data_args.sparse_value_type == 'sum':
                            vector[token] += int(v)
                        else:
                            if int(v) > vector[token]:
                                vector[token] = int(v)
                    else:
                        vector[token] = int(v)
            else:
                vector = dict()
                logit = logits
                disassemble_logit = disassemble_logits
                if model_args.eol_type == 'disassembleeol_concrete' or model_args.eol_type == 'disassembleeol_concrete_origin_text' or model_args.eol_type == 'all_disassembleeol_concrete' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text':
                    tokens, values = get_img_valid_disassemble_tokens_values(processor,
                                                                             disassemble_logit,
                                                                             vocab_dict,
                                                                             data_args,
                                                                             filtered_ids, logit,
                                                                             model_args)
                else:
                    tokens, values = get_img_valid_disassemble_tokens_values(processor,
                                                                             disassemble_logit,
                                                                             vocab_dict,
                                                                             data_args,
                                                                             filtered_ids, None,
                                                                             model_args)
                for token, v in zip(tokens, values):
                    if token in vector.keys():
                        if data_args.sparse_value_type == 'replace':
                            vector[token] = int(v)
                        elif data_args.sparse_value_type == 'sum':
                            vector[token] += int(v)
                        else:
                            if int(v) > vector[token]:
                                vector[token] = int(v)
                    else:
                        vector[token] = int(v)
        else:
            if training_args.encode_type == 'text':
                vector = dict()
                logit = logits
                tokens, values = get_text_valid_tokens_values(text, processor.tokenizer,
                                                              logit,
                                                              vocab_dict,
                                                              data_args,
                                                              filtered_ids)
                for token, v in zip(tokens, values):
                    if token in vector.keys():
                        if data_args.sparse_value_type == 'replace':
                            vector[token] = int(v)
                        elif data_args.sparse_value_type == 'sum':
                            vector[token] += int(v)
                        else:
                            if int(v) > vector[token]:
                                vector[token] = int(v)
                    else:
                        vector[token] = int(v)
            else:
                vector = dict()
                logit = logits
                if model_args.eol_type == 'prompteol_same_length':
                    tokens, values = get_img_valid_tokens_values(processor.tokenizer, logit,
                                                                 vocab_dict,
                                                                 data_args, filtered_ids, text=text)
                else:
                    tokens, values = get_img_valid_tokens_values(processor.tokenizer, logit,
                                                                 vocab_dict,
                                                                 data_args, filtered_ids)
                for token, v in zip(tokens, values):
                    if token in vector.keys():
                        if data_args.sparse_value_type == 'replace':
                            vector[token] = int(v)
                        elif data_args.sparse_value_type == 'sum':
                            vector[token] += int(v)
                        else:
                            if int(v) > vector[token]:
                                vector[token] = int(v)
                    else:
                        vector[token] = int(v)

    # 训练结束后添加同步屏障
    dist.barrier()

    # 确保所有进程同步退出
    if dist.get_rank() == 0:
        # 主进程最后退出
        torch.distributed.destroy_process_group()
    else:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
