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
    AutoProcessor, LlamaForCausalLM

from arguments import PromptRepsLLMDataArguments, ModelArguments
from arguments import TrainingArguments
from dataset import CrossModalRetrievalDataset
from model import MLLMRetrievalModel
from template import img_prompt, img_prompt_no_special_llava_v1_5, img_prompt_qwen_v2_5, \
    img_prompt_intern_vl_v2_5, llama3_template, task_text_prompts_copy, task_image_prompts_copy, \
    llama3_retrieval_disassemble_image_prompts, llama3_retrieval_disassemble_text_prompts, \
    llama3_template_image_prefix, llama3_template_content_element, retrieval_disassemble_image_prompts_3_for_concat, \
    retrieval_disassemble_image_prompts_for_concat, img_prompt_for_concat, \
    retrieval_disassemble_image_prompts_7_for_concat
from utils import load_image


# from fast_pytorch_kmeans import KMeans
# from faiss import Kmeans


def get_filtered_ids(tokenizer):
    filtered_ids = set()
    for token, id in tokenizer.get_vocab().items():
        if token[0] == '▁' or token[0] == ' ':
            token = token[1:]
        if not token.isalpha() and not token.isdigit():
            continue
        if ord('a') <= ord(token[0]) <= ord('z'):
            filtered_ids.add(id)
    return filtered_ids


def filter_token(token):
    if ord(token[0]) < ord('a') or ord(token[0]) > ord('z'):
        token = token[1:]
    return token


def get_img_valid_disassemble_tokens_values_word_analysis(tokenizer, disassemble_logits, vocab_dict, data_args, filtered_ids,
                                            logits=None, model_args=None):
    word_set = set()
    word_values = dict()
    if data_args.sparse_manual:
        top_k = data_args.sparse_length
    else:
        top_k = data_args.sparse_length

    for disassemble_logit in disassemble_logits:
        top_k_values, top_k_indices = disassemble_logit.topk(top_k, dim=-1)
        word_set.update(top_k_indices.tolist())
        values = np.rint(top_k_values.cpu().detach().float().numpy() * 100).astype(int)
        vocabs = []
        for indice in top_k_indices.cpu().detach().float().numpy():
            vocabs.append(vocab_dict[int(indice.item())])
        if dist.get_rank() == 0:
            print(vocabs)


def get_text_valid_disassemble_tokens_values_word_analysis(text, tokenizer, disassemble_logits, vocab_dict, data_args,
                                             filtered_ids, logits=None, model_args=None):
    word_set = set()
    word_values = dict()
    if data_args.sparse_manual:
        top_k = data_args.sparse_length
    else:
        top_k = data_args.sparse_length
    for disassemble_logit in disassemble_logits:
        top_k_values, top_k_indices = disassemble_logit.topk(top_k, dim=-1)
        word_set.update(top_k_indices.tolist())
        values = np.rint(top_k_values.cpu().detach().float().numpy() * 100).astype(int)
        vocabs = []
        for indice in top_k_indices.cpu().detach().float().numpy():
            vocabs.append(vocab_dict[int(indice.item())])
        if dist.get_rank() == 0:
            print(vocabs)


'''
在官方PromptReps中，有一个指定参数是multi_reps，目测是改取最后一个特征和logit为取多个特征和logit，但我们先只考虑取最后一个看看什么情况
有需要的时候再增加multi_reps
'''


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

    input_token_embeddings = encoder.get_input_embeddings().weight
    output_token_embeddings = encoder.get_output_embeddings().weight[:len(vocab_dict), :]
    output_token_dim = output_token_embeddings.size(1)

    if dist.get_rank() == 0:
        print(input_token_embeddings.shape)
        print(output_token_embeddings.shape)

    centroids_dict = {}  # 这是用来保存各个centroids都有哪些单词
    origin_to_centroids_dict = {}  # 这是用来保存各个原始单词对应哪个聚类中心，键值为token id，value为聚类中心索引
    origin_word_to_centroids_dict = {}  # 这是用来保存各个原始单词对应哪个聚类中心，键值为单词字符串，value为聚类中心索引

    if training_args.encode_type == 'text':
        dataset = CrossModalRetrievalDataset(data_args.dataset_name, processor, 'test', 'full')
    else:
        dataset = CrossModalRetrievalDataset(data_args.dataset_name, processor, 'test', 'single')
    sampler = Data.DistributedSampler(dataset, num_replicas=dist.get_world_size(), shuffle=True, rank=dist.get_rank())
    test_dataloader = Data.DataLoader(dataset=dataset, sampler=sampler, pin_memory=True,
                                      batch_size=data_args.per_device_batch_size, shuffle=False)

    model = MLLMRetrievalModel(encoder)
    model = model.eval()
    print(model.is_ddp)

    with torch.no_grad():
        sampler.set_epoch(0)
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
        for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(test_dataloader),
                                                                     total=len(test_dataloader)):
            with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
                if len(texts) != data_args.per_device_batch_size:
                    print(len(texts))
                    print(dist.get_rank())
                if model_args.calculate_type == 'separate':
                    if training_args.encode_type == 'text':
                        logits, reps = model.encode_data(texts, 'text', processor, device, model_args, data_args)
                        if model_args.eol_type == 'metaeol':
                            logits = logits.reshape(-1, len(task_text_prompts_copy), logits.shape[1]).mean(1)
                            reps = reps.reshape(-1, len(task_text_prompts_copy), reps.shape[1]).mean(1)
                        elif 'disassembleeol_concrete' in model_args.eol_type:
                            disassemble_logits = logits[data_args.per_device_batch_size:]
                            logits = logits[:data_args.per_device_batch_size]
                        elif 'disassembleeol' in model_args.eol_type:
                            disassemble_logits = logits

                    else:
                        # Preparation for inference
                        if 'InternVL2_5-8B' in model_args.model_name_or_path or 'InternVL2_5-4B' in model_args.model_name_or_path:
                            prompt = processor.apply_chat_template(
                                img_prompt_intern_vl_v2_5, tokenize=False, add_generation_prompt=True
                            )
                            imgs = [load_image(path, max_num=12).to(torch_type).cuda() for path in imgs_path]
                            logits, reps = model.encode_data(imgs, 'image', processor, device, model_args, data_args)
                        else:
                            if model_args.eol_type == 'prompteol' or model_args.eol_type == 'prompteol_same_length':
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
                            elif 'disassembleeol' in model_args.eol_type:
                                # 这是参考metaeol的思路，试图将图文中的不同元素拆解出来，目前先把这个处理放在稀疏检索上，然后再看看密集检索是否使用
                                # all_disassembleeol表示稀疏特征和密集特征都用各个子方面（角度）的结果
                                raw_images = [Image.open(path).convert('RGB') for path in imgs_path]
                                if model_args.eol_type != 'all_disassembleeol' and model_args.eol_type != 'all_disassembleeol_origin_text':
                                    img_inputs = processor(images=raw_images, text=[prompt] * len(imgs_path),
                                                           return_tensors="pt",
                                                           padding=True)
                                    imgs = img_inputs.to(device)
                                    if model_args.eol_type == 'disassembleeol_concrete' or model_args.eol_type == 'disassembleeol_concrete_origin_text':
                                        logits, reps = model.encode_data(imgs, 'image', processor, device, model_args,
                                                                         data_args)
                                    elif model_args.eol_type == 'all_disassembleeol_concrete' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text':
                                        logits, _ = model.encode_data(imgs, 'image', processor, device, model_args,
                                                                      data_args)
                                    else:
                                        _, reps = model.encode_data(imgs, 'image', processor, device, model_args,
                                                                    data_args)

                                disassemble_raw_images = [raw_image for raw_image in raw_images for _ in
                                                          range(len(prompts))]
                                disassemble_img_inputs = processor(images=disassemble_raw_images,
                                                                   text=prompts * len(imgs_path),
                                                                   return_tensors="pt",
                                                                   padding=True)
                                disassemble_imgs = disassemble_img_inputs.to(device)
                                if model_args.eol_type == 'all_disassembleeol' or model_args.eol_type == 'all_disassembleeol_origin_text' or model_args.eol_type == 'all_disassembleeol_concrete' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text':
                                    disassemble_logits, disassemble_embs = model.encode_data(disassemble_imgs, 'image',
                                                                                             processor, device,
                                                                                             model_args, data_args)
                                    reps = disassemble_embs
                                else:
                                    disassemble_logits, _ = model.encode_data(disassemble_imgs, 'image', processor,
                                                                              device,
                                                                              model_args, data_args)
                            else:
                                # 希望获得这样的列表[a,a,a,b,b,b,c,c,c......]
                                # 也就是说，对于批次中的每个图像，按照下面每次循环使用的prompt个数，加入到raw_images中
                                raw_images = [Image.open(path).convert('RGB') for
                                              path in imgs_path for _ in range(len(task_image_prompts_copy) // 4)]
                                # 将task_prompt添加到llama3_template中
                                prompts = [llama3_template.format(task_image_prompt) for task_image_prompt in
                                           task_image_prompts_copy]

                                logits = [[] for _ in range(len(imgs_path))]
                                reps = [[] for _ in range(len(imgs_path))]

                                for i in range(4):
                                    # 这个i是为了控制当前轮次使用哪些prompt编码
                                    start = i * len(prompts) // 4
                                    end = (i + 1) * len(prompts) // 4

                                    img_inputs = processor(images=raw_images, text=prompts[start:end] * len(imgs_path),
                                                           return_tensors="pt",
                                                           padding=True)

                                    imgs = img_inputs.to(device)

                                    # 在metaeol模式下，reps应该是[batch_size * len(task_prompts) // 4, reps_dim]
                                    logits_sub, reps_sub = model.encode_data(imgs, 'image', processor, device,
                                                                             model_args,
                                                                             data_args)

                                    for j in range(len(imgs_path)):
                                        # 这个j是为了控制要把第j个样本对应的数据存到对应索引下的列表中
                                        logits[j].append(logits_sub[j * len(prompts) // 4:(j + 1) * len(prompts) // 4])
                                        reps[j].append(reps_sub[j * len(prompts) // 4:(j + 1) * len(prompts) // 4])

                                logits = [item for logit in logits for item in logit]
                                reps = [item for rep in reps for item in rep]

                                logits = torch.cat(logits, dim=0)
                                reps = torch.cat(reps, dim=0)

                                logits = logits.reshape(-1, len(task_image_prompts_copy), logits.shape[1]).mean(1)
                                reps = reps.reshape(-1, len(task_image_prompts_copy), reps.shape[1]).mean(1)

                else:
                    if data_args.prompt_type == 'prompt_5':
                        prompt_template = llama3_template_image_prefix
                        if 'concrete' in model_args.eol_type or 'all' not in model_args.eol_type:
                            prompt_template += llama3_template_content_element.format(img_prompt_for_concat)
                        for llama3_retrieval_disassemble_image_prompt in retrieval_disassemble_image_prompts_for_concat:
                            content_element = llama3_template_content_element.format(
                                llama3_retrieval_disassemble_image_prompt)
                            prompt_template += content_element
                        print(prompt_template)
                    elif data_args.prompt_type == 'prompt_3':
                        prompt_template = llama3_template_image_prefix
                        if 'concrete' in model_args.eol_type or 'all' not in model_args.eol_type:
                            prompt_template += llama3_template_content_element.format(img_prompt_for_concat)
                        for llama3_retrieval_disassemble_image_prompt in retrieval_disassemble_image_prompts_3_for_concat:
                            content_element = llama3_template_content_element.format(
                                llama3_retrieval_disassemble_image_prompt)
                            prompt_template += content_element
                    elif data_args.prompt_type == 'prompt_7':
                        prompt_template = llama3_template_image_prefix
                        if 'concrete' in model_args.eol_type or 'all' not in model_args.eol_type:
                            prompt_template += llama3_template_content_element.format(img_prompt_for_concat)
                        for llama3_retrieval_disassemble_image_prompt in retrieval_disassemble_image_prompts_7_for_concat:
                            content_element = llama3_template_content_element.format(
                                llama3_retrieval_disassemble_image_prompt)
                            prompt_template += content_element
                    else:
                        pass
                    if training_args.encode_type == 'text':
                        logits, reps = model.encode_data_concat(texts, 'text', processor, device, model_args, data_args)
                        if 'disassembleeol_concrete' in model_args.eol_type:
                            disassemble_logits = logits[data_args.per_device_batch_size:]
                            logits = logits[:data_args.per_device_batch_size]
                        elif 'disassembleeol' in model_args.eol_type:
                            disassemble_logits = logits
                    else:
                        raw_images = [Image.open(path).convert('RGB') for path in imgs_path]
                        img_inputs = processor(images=raw_images, text=[prompt_template] * len(imgs_path),
                                               return_tensors="pt",
                                               padding=True)
                        imgs = img_inputs.to(device)
                        logits, reps = model.encode_data_concat(imgs, 'image', processor, device, model_args,
                                                                data_args)
                        if 'disassembleeol_concrete' in model_args.eol_type:
                            disassemble_logits = logits[data_args.per_device_batch_size:]
                            logits = logits[:data_args.per_device_batch_size]
                        elif 'disassembleeol' in model_args.eol_type:
                            disassemble_logits = logits

                ids = text_ids if training_args.encode_type == 'text' else img_ids
                if 'disassembleeol' in model_args.eol_type:
                    if training_args.encode_type == 'text':
                        for text_indice in range(len(ids)):
                            id = ids[text_indice]
                            if model_args.eol_type == 'disassembleeol_concrete' or model_args.eol_type == 'disassembleeol_concrete_origin_text' or model_args.eol_type == 'all_disassembleeol_concrete' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text':
                                logit = logits[text_indice]
                            text = texts[text_indice]
                            if data_args.prompt_type == 'prompt_5':
                                length = 5
                            elif data_args.prompt_type == 'prompt_3':
                                length = 3
                            elif data_args.prompt_type == 'prompt_7':
                                length = 7
                            else:
                                length = 5
                            disassemble_logit = disassemble_logits[
                                                text_indice * length:(text_indice + 1) * length]
                            if dist.get_rank() == 0:
                                print(text)
                            get_text_valid_disassemble_tokens_values_word_analysis(text, processor.tokenizer,
                                                                                          disassemble_logit,
                                                                                          vocab_dict,
                                                                                          data_args,
                                                                                          filtered_ids, None,
                                                                                          model_args)
                    else:
                        for img_indice in range(len(ids)):
                            id = ids[img_indice]
                            if model_args.eol_type == 'disassembleeol_concrete' or model_args.eol_type == 'disassembleeol_concrete_origin_text' or model_args.eol_type == 'all_disassembleeol_concrete' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text':
                                logit = logits[img_indice]
                            text = texts[img_indice]
                            if data_args.prompt_type == 'prompt_5':
                                length = 5
                            elif data_args.prompt_type == 'prompt_3':
                                length = 3
                            elif data_args.prompt_type == 'prompt_7':
                                length = 7
                            else:
                                length = 5
                            disassemble_logit = disassemble_logits[
                                                img_indice * length:(img_indice + 1) * length]
                            if dist.get_rank() == 0:
                                print(text)
                            get_img_valid_disassemble_tokens_values_word_analysis(processor,
                                                                                         disassemble_logit,
                                                                                         vocab_dict,
                                                                                         data_args,
                                                                                         filtered_ids, None,
                                                                                         model_args)

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
