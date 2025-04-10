import json
import logging
import os
import pickle
import string
import sys
import itertools

from matplotlib import pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from PIL import Image

from tqdm import tqdm
from transformers import (
    HfArgumentParser,
)
from transformers import LlavaProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, \
    LlavaNextForConditionalGeneration, Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration, AutoModel, \
    AutoProcessor, \
    AutoTokenizer, PhiForCausalLM, Phi3ForCausalLM, AutoModelForCausalLM
from tevatron.retriever.arguments import ModelArguments
from arguments import PromptRepsLLMDataArguments
import torch.distributed as dist
from arguments import TrainingArguments
from dataset import CrossModalRetrievalDataset
import torch
import torch.utils.data as Data
import torch.nn.functional as F

from template import text_prompt, img_prompt, text_prompt_no_one_word, img_prompt_no_one_word, \
    img_prompt_no_special_llava_v1_5, text_prompt_no_special_llava_v1_5, text_prompt_qwen_v2_5, img_prompt_qwen_v2_5, \
    img_prompt_intern_vl_v2_5, text_prompt_intern_vl_v2_5
from model import MLLMRetrievalModel
from utils import split_model, load_image
from encode import filter_token, get_filtered_ids


def text_statistic(text, tokenizer, logits, vocab_dict, data_args, filtered_ids):
    words = [i for i in word_tokenize(text.lower()) if
             i not in set(stopwords.words('english') + list(string.punctuation))]
    token_ids = set()
    for word in words:
        token_ids.update(tokenizer.encode(word, add_special_tokens=False))

    # top tokens in the text
    token_ids_in_text = torch.tensor(list(token_ids))
    k_in_text = len(token_ids_in_text)
    k_in_text_values, k_in_text_indices = logits[token_ids_in_text].topk(k_in_text, dim=-1)
    in_text_values = np.rint(k_in_text_values.cpu().detach().float().numpy() * 100).astype(int)

    top_k_values, top_k_indices = logits.topk(data_args.sparse_length, dim=-1)
    values = np.rint(logits.cpu().detach().float().numpy() * 100).astype(int)
    out_text_values = [values[int(i.item())] for i in top_k_indices.cpu().detach().float().numpy() if i not in token_ids and i < len(vocab_dict)]

    return in_text_values, out_text_values


def image_statistic(texts, tokenizer, logits, vocab_dict, data_args, filtered_ids):
    all_words = []
    for text in texts:
        words = [i for i in word_tokenize(text.lower()) if
                 i not in set(stopwords.words('english') + list(string.punctuation))]
        all_words.extend(words)
    token_ids = set()
    for word in all_words:
        token_ids.update(tokenizer.encode(word, add_special_tokens=False))
    # 这里是想获得图像的离散值，但是图像是没有文本的，所以不能像原来那样获得对应的token_id，那么是取top 10还是最多128呢，我们先最多128吧

    # if len(token_ids_in_text) == 0:  # if no tokens in the text (rare case), we use top 10 tokens
    #     top_k_values, top_k_indices = logits.topk(10, dim=-1)
    #     values = np.rint(top_k_values.cpu().detach().float().numpy() * 100).astype(int)
    #     tokens = [vocab_dict[i.item()] for i in top_k_indices.cpu().detach().float().numpy()]
    # else:
    # 根据原文，他们遵循了SPLADE设置，为了加强logit离散性，只保留最多128个值
    # 这里应该是先获得了logit，然后再来筛选哪些值，正常来说词表所有位置上的logit都很难是0，
    # 但是这里就默认那些没有的单词还有排名128名之后的单词logit为0了
    # TODO： 这里默认选128个了，但是文本大多数是没有128那么长的，不知道会不会对最终结果有影响

    token_ids_in_text = torch.tensor(list(token_ids))
    k_in_text = len(token_ids_in_text)
    k_in_text_values, k_in_text_indices = logits[token_ids_in_text].topk(k_in_text, dim=-1)
    in_text_values = np.rint(k_in_text_values.cpu().detach().float().numpy() * 100).astype(int)

    top_k_values, top_k_indices = logits.topk(data_args.sparse_length, dim=-1)
    values = np.rint(logits.cpu().detach().float().numpy() * 100).astype(int)
    out_text_values = [values[int(i.item())] for i in top_k_indices.cpu().detach().float().numpy() if i not in token_ids and i < len(vocab_dict)]
    return in_text_values, out_text_values


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


    dataset = CrossModalRetrievalDataset(data_args.dataset_name, processor, 'test', 'single')
    sampler = Data.DistributedSampler(dataset, num_replicas=dist.get_world_size(), shuffle=True, rank=dist.get_rank())
    test_dataloader = Data.DataLoader(dataset=dataset, sampler=sampler, pin_memory=True,
                                      batch_size=data_args.per_device_batch_size, shuffle=False)

    model = MLLMRetrievalModel(encoder)
    model = model.eval()
    print(model.is_ddp)

    image_values_token_id_in_text = []
    text_values_token_id_in_text = []
    image_values_token_id_not_in_text = []
    text_values_token_id_not_in_text = []

    lookup_indices = []

    # 加载词表并获取过滤后的单词id，但目前尚不清楚filtered_ids是做什么的
    if 'InternVL2_5-8B' in model_args.model_name_or_path or 'InternVL2_5-4B' in model_args.model_name_or_path:
        vocab_dict = processor.get_vocab()
        filtered_ids = get_filtered_ids(processor)
    else:
        vocab_dict = processor.tokenizer.get_vocab()
        filtered_ids = get_filtered_ids(processor.tokenizer)
    vocab_dict = {v: k for k, v in vocab_dict.items()}

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
        for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(test_dataloader),
                                                                     total=len(test_dataloader)):
            if len(texts) != data_args.per_device_batch_size:
                print(len(texts))
                print(dist.get_rank())

            raw_images = [Image.open(path).convert('RGB') for path in imgs_path]
            img_inputs = processor(images=raw_images, text=[prompt] * len(imgs_path), return_tensors="pt",
                                   padding=True)
            imgs = img_inputs.to(device)
            image_logits, _ = model.encode_data(imgs, 'image', processor, device, model_args, data_args)

            target_texts = []
            for id in img_ids:
                target_ids = dataset.get_target(id, 'image')
                for target_id in target_ids:
                    target_texts.append(dataset.get_text(target_id))

            if dist.is_initialized():

                image_logits_list = [torch.zeros_like(image_logits) for _ in range(dist.get_world_size())]
                texts_list = [[None] for _ in range(dist.get_world_size())]
                text_ids_list = [[None] for _ in range(dist.get_world_size())]
                image_ids_list = [[None] for _ in range(dist.get_world_size())]

                dist.all_gather(tensor_list=image_logits_list, tensor=image_logits.contiguous())
                dist.all_gather_object(object_list=text_ids_list, obj=text_ids)
                dist.all_gather_object(object_list=image_ids_list, obj=img_ids)
                dist.all_gather_object(object_list=texts_list, obj=target_texts)

                batch_image_logits = torch.cat(image_logits_list)
                batch_text_ids = list(itertools.chain(*text_ids_list))
                batch_image_ids = list(itertools.chain(*image_ids_list))
                batch_texts = list(itertools.chain(*texts_list))

                if dist.get_rank() == 0:
                    if training_args.encode_type == 'text':
                        lookup_indices.extend(batch_text_ids)
                    else:
                        lookup_indices.extend(batch_image_ids)

                    for id, logits, text in zip(batch_image_ids, batch_image_logits, batch_texts):
                        target_ids = dataset.get_target(id, 'image')
                        target_texts = []
                        for target_id in target_ids:
                            target_texts.append(dataset.get_text(target_id))
                        in_text_values, out_text_values = image_statistic(target_texts, processor.tokenizer, logits,
                                                                          vocab_dict,
                                                                          data_args,
                                                                          filtered_ids)
                        image_values_token_id_in_text.extend(in_text_values)
                        image_values_token_id_not_in_text.extend(out_text_values)

                        target_text_logits, _ = model.encode_data(target_texts, 'text', processor, device, model_args, data_args)

                        for target_id, target_text_logit, target_text in zip(target_ids, target_text_logits, target_texts):
                            in_text_values, out_text_values = text_statistic(target_text, processor.tokenizer, target_text_logit,
                                                                             vocab_dict,
                                                                             data_args, filtered_ids)
                            text_values_token_id_in_text.extend(in_text_values)
                            text_values_token_id_not_in_text.extend(out_text_values)



        if dist.get_rank() == 0:
            print(len(text_values_token_id_in_text))
            print(len(text_values_token_id_not_in_text))
            print(len(image_values_token_id_in_text))
            print(len(image_values_token_id_not_in_text))
            print(len(lookup_indices))
            plt.figure(figsize=(5, 5))
            plt.hist(text_values_token_id_in_text, bins=70, alpha=0.5, label='text_values_token_id_in_text', color='red')
            plt.hist(text_values_token_id_not_in_text, bins=70, alpha=0.5, label='text_values_token_id_not_in_text', color='blue')
            plt.savefig(f'text_values_{data_args.dataset_name}.png', dpi=300, bbox_inches='tight')  # 保存为文件

            plt.figure(figsize=(5, 5))
            plt.hist(image_values_token_id_in_text, bins=70, alpha=0.5, label='image_values_token_id_in_text',
                     color='red')
            plt.hist(image_values_token_id_not_in_text, bins=70, alpha=0.5, label='image_values_token_id_not_in_text',
                     color='blue')
            plt.savefig(f'image_values_{data_args.dataset_name}.png', dpi=300, bbox_inches='tight')  # 保存为文件

            plt.figure(figsize=(5, 5))
            plt.hist(text_values_token_id_in_text, bins=70, alpha=0.5, label='text_values_token_id_in_text',
                     color='red')
            plt.hist(image_values_token_id_in_text, bins=70, alpha=0.5, label='image_values_token_id_in_text',
                     color='blue')
            plt.savefig(f'values_in_text_{data_args.dataset_name}.png', dpi=300, bbox_inches='tight')  # 保存为文件




if __name__ == "__main__":
    main()
