import glob
import itertools
import os
import pickle
import faiss
from itertools import chain

from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
)
from tevatron.retriever.searcher import FaissFlatSearcher
from pyserini.search.lucene import LuceneImpactSearcher, LuceneSearcher
from pyserini.analysis import JWhiteSpaceAnalyzer
from contextlib import nullcontext
from PIL import Image

from model import MLLMRetrievalModel
from tevatron.retriever.arguments import ModelArguments
from arguments import PromptRepsLLMDataArguments, PromptRepsLLMSearchArguments
import torch.distributed as dist
from arguments import TrainingArguments
from transformers import LlavaProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, \
    LlavaNextForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, AutoProcessor, \
    AutoModelForCausalLM, AutoModel
from encode import get_filtered_ids
from dataset import CrossModalRetrievalDataset
from metrices import RecallMetrics

import numpy as np
import torch
import torch.utils.data as Data
import torch.nn.functional as F
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from template import text_prompt, img_prompt, text_prompt_no_one_word, img_prompt_no_one_word, \
    img_prompt_no_special_llava_v1_5, text_prompt_qwen_v2_5, img_prompt_qwen_v2_5, img_prompt_intern_vl_v2_5, \
    text_prompt_intern_vl_v2_5
from encode import get_img_valid_tokens_values, get_text_valid_tokens_values
from hybrid import fuse, write_trec_run, read_trec_run, fuse_statistic
from utils import load_image
from search import pickle_load, search_queries, get_run_dict, sparse_search

from template import relevant_prompt, in_one_word_relevant_prompt, text_query_relevant_prompt, \
    image_query_relevant_prompt, precise_caption_prompt, please_relevant_prompt, old_text_query_relevant_prompt, \
    old_image_query_relevant_prompt, origin_old_text_query_relevant_prompt, origin_old_image_query_relevant_prompt, \
    role_relevant_prompt, role_precise_caption_prompt, role_old_image_query_relevant_prompt, \
    role_old_text_query_relevant_prompt, first_precise_caption_prompt, mistral_relevant_prompt, \
    mistral_in_one_word_relevant_prompt, mistral_text_query_relevant_prompt, mistral_image_query_relevant_prompt, \
    mistral_precise_caption_prompt, mistral_please_relevant_prompt, mistral_old_text_query_relevant_prompt, \
    mistral_old_image_query_relevant_prompt, mistral_origin_old_text_query_relevant_prompt, \
    mistral_origin_old_image_query_relevant_prompt, mistral_role_relevant_prompt, mistral_role_precise_caption_prompt, \
    mistral_role_old_text_query_relevant_prompt, mistral_role_old_image_query_relevant_prompt, \
    mistral_first_precise_caption_prompt, mistral_query_generation_paradigm_prompt, query_generation_paradigm_prompt, \
    mistral_query_generation_paradigm_prompt_1, query_generation_paradigm_prompt_1, \
    detailed_mistral_query_generation_paradigm_prompt, detailed_query_generation_paradigm_prompt, \
    detailed_query_generation_paradigm_prompt_1, detailed_mistral_query_generation_paradigm_prompt_1, \
    mistral_query_generation_paradigm_prompt_5, mistral_query_generation_paradigm_prompt_4, \
    query_generation_paradigm_prompt_4, query_generation_paradigm_prompt_5

import random


stopwords = set(stopwords.words('english') + list(string.punctuation))

import logging

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (ModelArguments, PromptRepsLLMDataArguments, PromptRepsLLMSearchArguments, TrainingArguments))

    model_args, data_args, search_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: PromptRepsLLMDataArguments
    search_args: PromptRepsLLMSearchArguments
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

    if training_args.bf16:
        torch_type = torch.bfloat16
    elif training_args.fp16:
        torch_type = torch.float16
    else:
        torch_type = torch.float32

    # 指定模型
    if 'llava-hf-llava-1.5-7b-hf' in model_args.model_name_or_path:
        encoder = LlavaForConditionalGeneration.from_pretrained(model_args.model_name_or_path,
                                                                device_map=device_map,
                                                                torch_dtype=torch_type)
        processor = LlavaProcessor.from_pretrained(model_args.model_name_or_path)
    elif 'Qwen2.5-VL-7B-Instruct' in model_args.model_name_or_path or 'Qwen2.5-VL-3B-Instruct' in model_args.model_name_or_path:
        encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_args.model_name_or_path,
                                                                     device_map=device_map,
                                                                     torch_dtype=torch_type)
        processor = Qwen2_5_VLProcessor.from_pretrained(model_args.model_name_or_path)
    elif 'InternVL2_5-8B' in model_args.model_name_or_path:
        # device_map = split_model('InternVL2_5-8B')
        encoder = AutoModel.from_pretrained(model_args.model_name_or_path,
                                            device_map=device_map,
                                            torch_dtype=torch_type,
                                            trust_remote_code=True,
                                            use_flash_attn=True,
                                            low_cpu_mem_usage=True, )
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path,
                                                  trust_remote_code=True, )
    else:
        encoder = LlavaNextForConditionalGeneration.from_pretrained(model_args.model_name_or_path,
                                                                    device_map=device_map,
                                                                    torch_dtype=torch_type)
        processor = LlavaNextProcessor.from_pretrained(model_args.model_name_or_path)
        if 'royokong-e5-v' in model_args.model_name_or_path:
            setattr(processor, "patch_size", 14)  # hack for pass

    if search_args.query_type == 'text':
        dataset = CrossModalRetrievalDataset(data_args.dataset_name, processor, 'test', 'full')
    else:
        dataset = CrossModalRetrievalDataset(data_args.dataset_name, processor, 'test', 'single')
    sampler = Data.DistributedSampler(dataset, num_replicas=world_size, shuffle=True, rank=0)
    test_dataloader = Data.DataLoader(dataset=dataset, sampler=sampler, batch_size=data_args.per_device_batch_size,
                                      shuffle=False)
    # 在执行的时候，我们只申请一个GPU运行，因此也就相当于是在一个进程中处理这个问题，避免进程间需要通信字典的内容

    model = MLLMRetrievalModel(encoder)
    model = model.eval()
    print(model.is_ddp)

    lookup_indices = []

    length_count_dict = {} # 统计每个长度有多少句
    length_content_dict = {} # 统计每个长度有哪些图文
    sharded_nll_dict = {} # 统计每个长度的平均对数似然


    # 加载词表并获取过滤后的单词id，但目前尚不清楚filtered_ids是做什么的
    if 'InternVL2_5-8B' in model_args.model_name_or_path:
        vocab_dict = processor.get_vocab()
        filtered_ids = get_filtered_ids(processor)
    else:
        vocab_dict = processor.tokenizer.get_vocab()
        filtered_ids = get_filtered_ids(processor.tokenizer)
    vocab_dict = {v: k for k, v in vocab_dict.items()}

    model.eval()

    flickr_length_dict = {3: 3, 4: 5, 5: 26, 6: 83, 7: 196, 8: 316, 9: 376, 10: 447, 11: 446, 12: 455, 13: 399, 14: 403,
                          15: 343, 16: 287, 17: 213, 18: 179, 19: 134, 20: 127, 21: 82, 22: 78, 23: 83, 24: 45, 25: 40,
                          26: 40, 27: 27, 28: 27, 29: 30, 30: 20, 31: 16, 32: 8, 33: 14, 34: 3, 35: 7, 36: 9, 37: 2,
                          38: 4, 39: 3, 40: 3, 41: 3, 42: 1, 43: 2, 44: 1, 45: 2, 46: 2, 47: 1, 48: 1, 52: 1, 54: 1,
                          56: 1, 57: 1, 58: 2, 64: 1, 85: 1}

    coco_length_dict = {7: 2, 8: 691, 9: 2878, 10: 4461, 11: 4937, 12: 4122, 13: 2872, 14: 1815, 15: 1183, 16: 690,
                        17: 445, 18: 298, 19: 183, 20: 118, 21: 85, 22: 48, 23: 35, 24: 35, 25: 26, 26: 21, 27: 15,
                        28: 3, 29: 10, 30: 4, 31: 6, 32: 6, 33: 1, 34: 4, 36: 2, 37: 3, 39: 1, 42: 3, 45: 1, 47: 1,
                        49: 1, 50: 3, 54: 1}

    flickr_length_list_20 = [(3, 4, 5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (15), (16), (17), (18), (19),
                          (20), (21), (22), (23), (24), (25), (26), (27), (28), (29),
                          (30), (31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 52, 54, 56, 57, 58, 64, 85)]

    flickr_length_list_30 = [(3, 4, 5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (15), (16), (17), (18), (19),
                             (20), (21), (22), (23), (24), (25), (26), (27, 28, 29),
                             (
                             30, 31), (32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 52, 54, 56, 57,
                             58, 64, 85)]

    rerank_prompt_type = search_args.rerank_template
    if 'llava-hf-llava-v1.6-mistral-7b-hf' in model_args.model_name_or_path:
        if rerank_prompt_type == 'caption_generation':
            rerank_prompt_template = mistral_query_generation_paradigm_prompt
        elif rerank_prompt_type == 'what_caption_generation':
            rerank_prompt_template = mistral_query_generation_paradigm_prompt_1
        elif rerank_prompt_type == 'detailed_caption_generation':
            rerank_prompt_template = detailed_mistral_query_generation_paradigm_prompt
        elif rerank_prompt_type == 'detailed_caption_generation_1':
            rerank_prompt_template = detailed_mistral_query_generation_paradigm_prompt_1
        elif rerank_prompt_type == 'caption_generation_4':
            rerank_prompt_template = mistral_query_generation_paradigm_prompt_4
        elif rerank_prompt_type == 'caption_generation_5':
            rerank_prompt_template = mistral_query_generation_paradigm_prompt_5
        else:
            rerank_prompt_template = mistral_query_generation_paradigm_prompt
    else:
        if rerank_prompt_type == 'caption_generation':
            rerank_prompt_template = query_generation_paradigm_prompt
        elif rerank_prompt_type == 'what_caption_generation':
            rerank_prompt_template = query_generation_paradigm_prompt_1
        elif rerank_prompt_type == 'detailed_caption_generation':
            rerank_prompt_template = detailed_query_generation_paradigm_prompt
        elif rerank_prompt_type == 'detailed_caption_generation_1':
            rerank_prompt_template = detailed_query_generation_paradigm_prompt_1
        elif rerank_prompt_type == 'caption_generation_4':
            rerank_prompt_template = query_generation_paradigm_prompt_4
        elif rerank_prompt_type == 'caption_generation_5':
            rerank_prompt_template = query_generation_paradigm_prompt_5
        else:
            rerank_prompt_template = query_generation_paradigm_prompt

    nll_sum_dict = {}

    with torch.no_grad():
        for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(test_dataloader),
                                                                     total=len(test_dataloader)):
            with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
                for text, img_path, text_id, img_id in zip(texts, imgs_path, text_ids, img_ids):
                    input_id = processor(text=text, return_tensors="pt")['input_ids'].squeeze().tolist()[1:]
                    if len(input_id) not in length_count_dict.keys():
                        length_count_dict[len(input_id)] = 1
                    else:
                        length_count_dict[len(input_id)] += 1

                    if len(input_id) not in length_content_dict.keys():
                        length_content_dict[len(input_id)] = [(text, img_path, text_id, img_id)]
                    else:
                        length_content_dict[len(input_id)].append((text, img_path, text_id, img_id))
        print(length_content_dict)
        length_count_dict = dict(sorted(length_count_dict.items(), key=lambda item: item[0]))
        print(length_count_dict)

        if search_args.tuple_sum == 20:
            for length_tuple in flickr_length_list_20:
                content_sub_set = set()
                for length in length_tuple:
                    content_sub_set.update(length_content_dict[length])
                selected_items = random.sample(content_sub_set, 20)
                with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
                    nll_sum = 0
                    for item in selected_items:
                        text = item[0]
                        image_path = item[1]
                        raw_image = Image.open(image_path).convert('RGB')
                        text_input = rerank_prompt_template + text
                        inputs = processor(images=raw_image, text=text_input, return_tensors="pt").to(model.device)
                        labels = processor(text=text, return_tensors="pt")['input_ids'].squeeze().tolist()
                        max_inputs_sum = inputs['input_ids'].shape[1]
                        # 去掉label的第一个起始符
                        labels = [-100] * (max_inputs_sum - len(labels[1:])) + labels[1:]
                        labels_view = torch.tensor(labels).to(model.device)
                        output = model(**inputs, output_hidden_states=True, return_dict=True)
                        logits = output.logits
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels_view[..., 1:].contiguous()
                        loss_func = torch.nn.CrossEntropyLoss(reduction='none')
                        nll = loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        nll = nll.view(shift_labels.size())
                        # 这个为啥是sum呢？根据原论文，是要把各个token上预测结果概率的对数似然加和取平均，但这里似乎只是求了和
                        # upr的代码中，指定了每个batch_size是1，也就是每次只针对1个查询计算
                        avg_nll = torch.sum(nll, dim=1)
                        valid_tokens = (labels_view != -100).sum(dim=1).float()
                        avg_nll /= valid_tokens
                        # 目前暂时认为avg_nll的大小是[batch_size]，直接tolist后就是对应img_id的相似度
                        nll_sum += avg_nll
                    nll_sum /= 20
                    nll_sum_dict[length_tuple] = nll_sum
        elif search_args.tuple_sum == 30:
            for length_tuple in flickr_length_list_30:
                content_sub_set = set()
                for length in length_tuple:
                    content_sub_set.update(length_content_dict[length])
                selected_items = random.sample(content_sub_set, 30)
                with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
                    nll_sum = 0
                    for item in selected_items:
                        text = item[0]
                        image_path = item[1]
                        raw_image = Image.open(image_path).convert('RGB')
                        text_input = rerank_prompt_template + text
                        inputs = processor(images=raw_image, text=text_input, return_tensors="pt").to(model.device)
                        labels = processor(text=text, return_tensors="pt")['input_ids'].squeeze().tolist()
                        max_inputs_sum = inputs['input_ids'].shape[1]
                        # 去掉label的第一个起始符
                        labels = [-100] * (max_inputs_sum - len(labels[1:])) + labels[1:]
                        labels_view = torch.tensor(labels).to(model.device)
                        output = model(**inputs, output_hidden_states=True, return_dict=True)
                        logits = output.logits
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels_view[..., 1:].contiguous()
                        loss_func = torch.nn.CrossEntropyLoss(reduction='none')
                        nll = loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        nll = nll.view(shift_labels.size())
                        # 这个为啥是sum呢？根据原论文，是要把各个token上预测结果概率的对数似然加和取平均，但这里似乎只是求了和
                        # upr的代码中，指定了每个batch_size是1，也就是每次只针对1个查询计算
                        avg_nll = torch.sum(nll, dim=1)
                        valid_tokens = (labels_view != -100).sum(dim=1).float()
                        avg_nll /= valid_tokens
                        # 目前暂时认为avg_nll的大小是[batch_size]，直接tolist后就是对应img_id的相似度
                        nll_sum += avg_nll
                    nll_sum /= 30
                    nll_sum_dict[length_tuple] = nll_sum
        else:
            for length_tuple in flickr_length_list_20:
                content_sub_set = set()
                for length in length_tuple:
                    content_sub_set.update(length_content_dict[length])
                selected_items = random.sample(content_sub_set, 20)
                with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
                    nll_sum = 0
                    for item in selected_items:
                        text = item[0]
                        image_path = item[1]
                        raw_image = Image.open(image_path).convert('RGB')
                        text_input = rerank_prompt_template + text
                        inputs = processor(images=raw_image, text=text_input, return_tensors="pt").to(model.device)
                        labels = processor(text=text, return_tensors="pt")['input_ids'].squeeze().tolist()
                        max_inputs_sum = inputs['input_ids'].shape[1]
                        # 去掉label的第一个起始符
                        labels = [-100] * (max_inputs_sum - len(labels[1:])) + labels[1:]
                        labels_view = torch.tensor(labels).to(model.device)
                        output = model(**inputs, output_hidden_states=True, return_dict=True)
                        logits = output.logits
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels_view[..., 1:].contiguous()
                        loss_func = torch.nn.CrossEntropyLoss(reduction='none')
                        nll = loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        nll = nll.view(shift_labels.size())
                        # 这个为啥是sum呢？根据原论文，是要把各个token上预测结果概率的对数似然加和取平均，但这里似乎只是求了和
                        # upr的代码中，指定了每个batch_size是1，也就是每次只针对1个查询计算
                        avg_nll = torch.sum(nll, dim=1)
                        valid_tokens = (labels_view != -100).sum(dim=1).float()
                        avg_nll /= valid_tokens
                        # 目前暂时认为avg_nll的大小是[batch_size]，直接tolist后就是对应img_id的相似度
                        nll_sum += avg_nll
                    nll_sum /= 20
                    nll_sum_dict[length_tuple] = nll_sum

        # 提取键和值
        length_of_input_id = list(length_count_dict.keys())
        count = list(length_count_dict.values())
        print(nll_sum_dict)

        # 创建条形图
        plt.bar(length_of_input_id, count, color='skyblue', edgecolor='navy', alpha=0.7)

        # 添加标签和标题
        plt.xlabel('length_of_input_id')
        plt.ylabel('count')
        plt.title('dataset text input id length')
        plt.grid(axis='y', alpha=0.4)  # 添加网格线便于读数

        plt.savefig(f'my_plot_{data_args.dataset_name}.png')


if __name__ == '__main__':
    main()
