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
from arguments import TrainingArguments, LogitInformationAnalysisArguments, PromptGenerationArguments
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
from template import prompt_generation_from_image_prompt, prompt_generation_from_text_prompt, \
    prompt_generation_from_text_prompt_2, prompt_generation_from_image_prompt_2, prompt_generation_text_prompt, \
    prompt_generation_image_prompt


def main():
    parser = HfArgumentParser(
        (ModelArguments, PromptRepsLLMDataArguments, TrainingArguments, PromptGenerationArguments))

    model_args, data_args, training_args, prompt_generation_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: PromptRepsLLMDataArguments
    training_args: TrainingArguments
    prompt_generation_args: PromptGenerationArguments

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
        if prompt_generation_args.prompt_generation_type == 't2t':
            demonstration_sent_1 = 'The white and brown dog is running over the surface of the snow.'
            demonstration_sent_2 = 'Girl in black jacket sifting powdered sugar over a chocolate cake.'
            sent = prompt_generation_args.prompt_generation_text
            if prompt_generation_args.demonstration_num == 0:
                prompt = prompt_generation_text_prompt
                text_input = prompt.replace('<sent>', sent, 1)
            elif prompt_generation_args.demonstration_num == 1:
                prompt = prompt_generation_from_text_prompt
                text_input = prompt.replace('<sent>', demonstration_sent_1, 1)
                text_input = text_input.replace('<sent>', sent, 1)
            elif prompt_generation_args.demonstration_num == 2:
                prompt = prompt_generation_from_text_prompt_2
                text_input = prompt.replace('<sent>', demonstration_sent_1, 1)
                text_input = text_input.replace('<sent>', demonstration_sent_2, 1)
                text_input = text_input.replace('<sent>', sent, 1)
            inputs = processor(text=text_input, return_tensors="pt").to(model.device)
            output = model.encoder.generate(**inputs, max_new_tokens=100)
            if dist.get_rank() == 0:
                print(processor.decode(output[0], skip_special_tokens=True))

        elif prompt_generation_args.prompt_generation_type == 'i2i':
            demonstration_image_path_1 = './data/flickr/flickr30k-images/101654506.jpg'
            demonstration_image_path_2 = './data/flickr/flickr30k-images/100207720.jpg'
            image_path = prompt_generation_args.prompt_generation_image
            if prompt_generation_args.demonstration_num == 0:
                prompt = prompt_generation_from_image_prompt
                image = Image.open(image_path).convert('RGB')
                image_list = [image]
            elif prompt_generation_args.demonstration_num == 1:
                demonstration_image = Image.open(demonstration_image_path_1).convert('RGB')
                prompt = prompt_generation_from_image_prompt
                image = Image.open(image_path).convert('RGB')
                image_list = [demonstration_image, image]
            elif prompt_generation_args.demonstration_num == 2:
                demonstration_image_1 = Image.open(demonstration_image_path_1).convert('RGB')
                demonstration_image_2 = Image.open(demonstration_image_path_2).convert('RGB')
                prompt = prompt_generation_from_image_prompt
                image = Image.open(image_path).convert('RGB')
                image_list = [demonstration_image_1, demonstration_image_2, image]

            img_inputs = processor(images=image_list, text=prompt,
                                   return_tensors="pt",
                                   padding=True)
            output = model.encoder.generate(**img_inputs, max_new_tokens=100)
            if dist.get_rank() == 0:
                print(processor.decode(output[0], skip_special_tokens=True))
        else:
            pass


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