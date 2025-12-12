import json
import os
import pickle
import string
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import sys

torch.set_printoptions(threshold=sys.maxsize)
from PIL import Image
from nltk import word_tokenize
from nltk.corpus import stopwords
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
)
from transformers import LlavaProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, \
    LlavaNextForConditionalGeneration, Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration, AutoModel, \
    AutoProcessor, LlamaForCausalLM, GPTJForCausalLM, CLIPProcessor, CLIPModel, BlipProcessor, BlipForImageTextRetrieval

from arguments import PromptRepsLLMDataArguments, ModelArguments
from arguments import TrainingArguments
from dataset import CrossModalRetrievalDataset, ComposedTextImageRetrievalDataset, TextPersonRetrievalDataset
from model import MLLMRetrievalModel
from template import img_prompt, img_prompt_no_special_llava_v1_5, img_prompt_qwen_v2_5, \
    img_prompt_intern_vl_v2_5, llama3_template, task_text_prompts_copy, task_image_prompts_copy, \
    llama3_retrieval_disassemble_image_prompts, llama3_retrieval_disassemble_text_prompts, \
    llama3_template_image_prefix, llama3_template_content_element, retrieval_disassemble_image_prompts_3_for_concat, \
    retrieval_disassemble_image_prompts_for_concat, img_prompt_for_concat, \
    retrieval_disassemble_image_prompts_7_for_concat, mistral_img_prompt, llava_mistral_template_image_prefix, \
    llava_mistral_template_text_prefix, llava_mistral_template_content_element, \
    llava_mistral_template_fashion_iq_image_prefix, llama3_template_fashion_iq_image_prefix, \
    retrieval_disassemble_image_prompts_fashion_iq_for_concat, fashion_iq_composed_image_for_concat, \
    fashion_iq_img_prompt_for_concat, llama3_fashion_iq_image_prompt, mistral_fashion_iq_image_prompt, \
    person_retrieval_img_prompt, mistral_person_retrieval_img_prompt, \
    person_retrieval_img_prompt_1, mistral_person_retrieval_img_prompt_1, \
    person_retrieval_img_prompt_for_concat, person_retrieval_img_prompt_for_concat_1, \
    retrieval_disassemble_image_prompts_person_retrieval_for_concat, \
    retrieval_disassemble_image_prompts_person_retrieval_for_concat_1, \
    retrieval_disassemble_image_origin_prompts_person_retrieval_for_concat, mistral_person_retrieval_img_prompt_2, \
    person_retrieval_img_prompt_2, person_retrieval_img_prompt_for_concat_2
from utils import load_image


def main():
    parser = HfArgumentParser((ModelArguments, PromptRepsLLMDataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: PromptRepsLLMDataArguments
    training_args: TrainingArguments

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
    if 'clip' in model_args.model_name_or_path:
        encoder = LlavaForConditionalGeneration.from_pretrained(model_args.model_name_or_path, device_map=device_map,
                                                                torch_dtype=torch_type)
        processor = LlavaProcessor.from_pretrained(model_args.model_name_or_path)
    else:
        encoder = LlavaNextForConditionalGeneration.from_pretrained(model_args.model_name_or_path,
                                                                    device_map=device_map,
                                                                    torch_dtype=torch_type)
        processor = LlavaNextProcessor.from_pretrained(model_args.model_name_or_path)


    if training_args.task_type == 'cir':
        dataset = ComposedTextImageRetrievalDataset(data_args.dataset_name, processor, data_args.dataset_split,
                                                    training_args.encode_type)
    elif training_args.task_type == 'tbpr':
        dataset = TextPersonRetrievalDataset(data_args.dataset_name, processor, data_args.dataset_split, 'single')
    else:
        if training_args.encode_type == 'text':
            dataset = CrossModalRetrievalDataset(data_args.dataset_name, processor, data_args.dataset_split, 'full')
        else:
            dataset = CrossModalRetrievalDataset(data_args.dataset_name, processor, data_args.dataset_split, 'single')
    sampler = Data.DistributedSampler(dataset, num_replicas=dist.get_world_size(), shuffle=True, rank=dist.get_rank())
    test_dataloader = Data.DataLoader(dataset=dataset, sampler=sampler, pin_memory=True,
                                      batch_size=data_args.per_device_batch_size, shuffle=False)

    model = MLLMRetrievalModel(encoder)
    model = model.eval()
    print(model.is_ddp)

    encoded = []
    jsonl_data = []
    lookup_indices = []

    with torch.no_grad():
        sampler.set_epoch(0)

        if training_args.task_type == 'cir':
            for batch_idx, (texts, imgs_path, target_path, text_ids, img_ids, composed_ids, dress_type) in tqdm(enumerate(test_dataloader),
                                                                                                    total=len(test_dataloader)):
                with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
                    prompt_list = [prompt.replace("{}", dress_type_item) for dress_type_item in dress_type]
                    raw_images = [Image.open(path).convert('RGB') for path in imgs_path]
                    img_inputs = processor(images=raw_images, text=prompt_list,
                                           return_tensors="pt",
                                           padding=True)
                    imgs = img_inputs.to(device)
                    logits, reps = model.encode_data_for_cir(texts, imgs, dress_type, 'image', processor, device,
                                                             model_args,
                                                             data_args)
                    # print(logits.shape)
                    reps = F.normalize(reps, dim=-1)
                    if model_args.eol_type == 'all_disassembleeol' or model_args.eol_type == 'all_disassembleeol_origin_text' or model_args.eol_type == 'all_disassembleeol_concrete' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text':
                        prompt_length = 5
                        reps = reps.reshape(-1, prompt_length, reps.shape[1]).mean(1)
                    lookup_indices.extend(img_ids)
                    encoded.append(reps.cpu().detach().float().numpy())

        elif training_args.task_type == 'tbpr':
            for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(test_dataloader),
                                                                         total=len(test_dataloader)):
                if 'Qwen2.5-VL-7B-Instruct' in model_args.model_name_or_path or 'Qwen2.5-VL-3B-Instruct' in model_args.model_name_or_path:
                    prompt = processor.apply_chat_template(
                        img_prompt_qwen_v2_5, tokenize=False, add_generation_prompt=True
                    )
                raw_images = [Image.open(path).convert('RGB') for path in imgs_path]
                img_inputs = processor(images=raw_images, text=[prompt] * len(imgs_path),
                                       return_tensors="pt",
                                       padding=True)
                imgs = img_inputs.to(device)
                logits, reps = model.encode_data_for_tbpr(imgs, 'image', processor, device, model_args,
                                                          data_args)

                # print(logits.shape)
                reps = F.normalize(reps, dim=-1)

                lookup_indices.extend(img_ids)

                encoded.append(reps.cpu().detach().float().numpy())
        else:
            for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(test_dataloader),
                                                                         total=len(test_dataloader)):
                with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
                    if len(texts) != data_args.per_device_batch_size:
                        print(len(texts))
                        print(dist.get_rank())
                    if model_args.calculate_type == 'separate':
                        if training_args.encode_type == 'text':
                            logits, reps = model.encode_data(texts, 'text', processor, device, model_args, data_args)

                        else:
                            if 'Qwen2.5-VL-7B-Instruct' in model_args.model_name_or_path or 'Qwen2.5-VL-3B-Instruct' in model_args.model_name_or_path:
                                prompt = processor.apply_chat_template(
                                    img_prompt_qwen_v2_5, tokenize=False, add_generation_prompt=True
                                )
                            raw_images = [Image.open(path).convert('RGB') for path in imgs_path]
                            img_inputs = processor(images=raw_images, text=[prompt] * len(imgs_path),
                                                   return_tensors="pt",
                                                   padding=True)
                            imgs = img_inputs.to(device)
                            logits, reps = model.encode_data(imgs, 'image', processor, device, model_args,
                                                             data_args)

                    # print(logits.shape)
                    reps = F.normalize(reps, dim=-1)

                    encoded.append(reps.cpu().detach().float().numpy())