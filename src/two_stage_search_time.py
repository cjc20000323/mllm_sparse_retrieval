import glob
import json
import os
import pickle
import faiss
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
)
from contextlib import nullcontext
from PIL import Image
from itertools import chain

from model import MLLMRetrievalModel
from arguments import PromptRepsLLMDataArguments, PromptRepsLLMSearchArguments, ModelArguments
import torch.distributed as dist
from arguments import TrainingArguments
from transformers import LlavaProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, \
    LlavaNextForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, AutoProcessor, \
    AutoModelForCausalLM, AutoModel, LlamaForCausalLM
from encode import get_filtered_ids
from dataset import CrossModalRetrievalDataset
from metrices import RecallMetrics

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from nltk.corpus import stopwords
import string
from template import img_prompt, \
    img_prompt_no_special_llava_v1_5, img_prompt_qwen_v2_5, img_prompt_intern_vl_v2_5, task_image_prompts, \
    llama3_template, task_text_prompts, llama3_retrieval_disassemble_image_prompts, \
    llama3_retrieval_disassemble_text_prompts
from encode import get_img_valid_tokens_values, get_text_valid_tokens_values, get_img_valid_tokens_values_with_cluster, \
    get_text_valid_tokens_values_with_cluster, get_text_valid_disassemble_tokens_values, \
    get_img_valid_disassemble_tokens_values, llama3_template_image_prefix, llama3_template_content_element, \
    retrieval_disassemble_image_prompts_3_for_concat, \
    retrieval_disassemble_image_prompts_for_concat, img_prompt_for_concat, retrieval_disassemble_image_prompts_7_for_concat
from hybrid import fuse, normalize
from utils import load_image
from peft import PeftModel
from search import pickle_load, search_queries, sparse_search, get_run_dict, search_queries_two_stage
import time
import gc

# from cuml.cluster import KMeans

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

    if data_args.reps_loc == 'after_pad':
        processor.tokenizer.padding_side = "left"
        processor.tokenizer.padding = True

    # 加载词表并获取过滤后的单词id，但目前尚不清楚filtered_ids是做什么的
    if 'InternVL2_5-8B' in model_args.model_name_or_path:
        vocab_dict = processor.get_vocab()
        filtered_ids = get_filtered_ids(processor)
    else:
        vocab_dict = processor.tokenizer.get_vocab()
        filtered_ids = get_filtered_ids(processor.tokenizer)
    vocab_dict = {v: k for k, v in vocab_dict.items()}

    if model_args.use_output_embedding_cluster:
        output_token_embeddings = encoder.get_output_embeddings().weight[:len(vocab_dict), :]

        centroids_dict = {}  # 这是用来保存各个centroids都有哪些单词
        origin_to_centroids_dict = {}  # 这是用来保存各个原始单词对应哪个聚类中心，键值为token id，value为聚类中心索引
        origin_word_to_centroids_dict = {}  # 这是用来保存各个原始单词对应哪个聚类中心，键值为单词字符串，value为聚类中心索引
        output_token_embeddings_for_kmeans = output_token_embeddings.detach().cpu().numpy()

        if dist.get_rank() == 0:
            print('Now load kmeans model.')
            print(f"kmeans_model_{model_args.model_name_or_path[14:]}_{model_args.cluster_sum}.pkl")
        with open(f"kmeans_model_{model_args.model_name_or_path[14:]}_{model_args.cluster_sum}.pkl", "rb") as f:
            kmeans = pickle.load(f)

        # 训练并预测
        # kmeans.fit(output_token_embeddings_for_kmeans)
        labels = kmeans.predict(output_token_embeddings_for_kmeans)
        labels = torch.from_numpy(labels.squeeze()).cuda()
        print(labels)

        # 获取聚类中心
        centroids = kmeans.cluster_centers_

        centroids = torch.from_numpy(centroids).to(dtype=torch_type).cuda()  # 聚类中心

        centroids_dict = {index: [] for index in range(len(centroids))}
        print(centroids)
        print(centroids.shape)

        for i, v in enumerate(labels):
            if i < len(vocab_dict):
                origin_to_centroids_dict[i] = int(v)
                origin_word_to_centroids_dict[vocab_dict[i]] = int(v)

        for k in origin_to_centroids_dict:
            centroids_dict[origin_to_centroids_dict[k]].append(vocab_dict[k])

        new_lm_head = nn.Linear(encoder.language_model.config.hidden_size, model_args.cluster_sum, bias=False,
                                dtype=torch_type).to(device)
        print(new_lm_head.weight.shape)
        new_lm_head.weight.data = centroids.clone()
        del centroids
        del labels
        del kmeans
        if dist.get_rank() == 0:
            print(new_lm_head.weight)

        encoder.language_model.lm_head = new_lm_head

    if model_args.lora:
        if dist.get_rank() == 0:
            print('We use lora model trained few shot here.')
        encoder = PeftModel.from_pretrained(
            encoder,  # 原始模型
            model_args.lora_model_path,  # LoRA 适配器目录
        )
        encoder = encoder.merge_and_unload()

    if search_args.query_type == 'text':
        dataset = CrossModalRetrievalDataset(data_args.dataset_name, processor, 'test', 'full')
    else:
        dataset = CrossModalRetrievalDataset(data_args.dataset_name, processor, 'test', 'single')
    sampler = Data.DistributedSampler(dataset, num_replicas=world_size, shuffle=True, rank=rank)
    test_dataloader = Data.DataLoader(dataset=dataset, sampler=sampler, batch_size=data_args.per_device_batch_size,
                                      shuffle=False)

    model = MLLMRetrievalModel(encoder)
    model = model.eval()
    print(model.is_ddp)

    from tevatron.retriever.searcher import FaissFlatSearcher
    from pyserini.search.lucene import LuceneImpactSearcher
    from pyserini.analysis import JWhiteSpaceAnalyzer

    lookup_indices = []

    model.eval()

    dense_run = {}
    sparse_run = {}
    fusion_run_1 = {}
    fusion_run_2 = {}
    fusion_run_3 = {}
    fusion_run_4 = {}
    fusion_run_5 = {}

    dense_retriever_indices = []
    sparse_retriever_indices = []

    dense_search_time = []
    sparse_search_time = []
    embedding_time = []
    sparse_obtain_time = []
    dense_obtain_time = []
    dense_retriever_time = []

    if search_args.passage_reps is not None:
        # 目前尚不清楚这里是怎么工作的
        # 另外，这里源代码里有multi_reps，暂时先不管，后面再加
        dense_retriever_indices = [search_args.passage_reps]

    if search_args.sparse_index is not None:
        # 目前尚不清楚这里是怎么工作的
        # 另外，这里源代码里有multi_reps，暂时先不管，后面再加
        sparse_retriever_indices = [search_args.sparse_index]

    for i in range(max(len(dense_retriever_indices), len(sparse_retriever_indices))):

        dense_retriever = None
        sparse_retriever = None

        sparse_retriever = LuceneImpactSearcher(os.path.join(sparse_retriever_indices[i], 'index'), None)
        analyzer = JWhiteSpaceAnalyzer()
        sparse_retriever.set_analyzer(analyzer)

        lookup_to_reps = {}

        index_files = glob.glob(os.path.join(dense_retriever_indices[i], 'corpus*.pkl'))
        if dist.get_rank() == 0:
            print(f'Pattern match found {len(index_files)} files; loading them into dense index.')

        p_reps_0, p_lookup_0 = pickle_load(index_files[0])
        if 'large' in dense_retriever_indices[i]:
            p_reps_0 = p_reps_0.squeeze()
        shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
        if len(index_files) > 1:
            shards = tqdm(shards, desc='Loading shards into index', total=len(index_files))
        # 将候选集的数据读出后，构建候选id到候选密集特征的字典，供后面
        candidate_reps = []
        candidate_lookup = []
        look_up = []
        for p_reps, p_lookup in shards:
            candidate_reps.extend(p_reps)
            candidate_lookup.extend(p_lookup)
            look_up += p_lookup
        # 这个候选id到候选密集特征字典的键是字符串
        print(f'candidate length: {len(candidate_reps)}')
        print(type(candidate_reps))
        print(type(candidate_lookup))

        # candidate_reps = torch.tensor(candidate_reps)
        # candidate_lookup = torch.tensor(candidate_lookup)
        # print(f'candidate_reps shape: {candidate_reps.shape}')
        # print(f'candidate_lookup shape: {candidate_lookup.shape}')
        for p_reps, p_lookup in zip(candidate_reps, candidate_lookup):
            if 'large' in dense_retriever_indices[i]:
                lookup_to_reps[str(p_lookup)] = p_reps
            else:
                lookup_to_reps[p_lookup] = p_reps
        if dist.get_rank() == 0:
            print(list(lookup_to_reps.keys())[:12800])

        with torch.no_grad(), torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
            for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(test_dataloader),
                                                                         total=len(test_dataloader)):
                # CPU时间开始
                cpu_start = time.time()
                '''
                # GPU事件开始
                gpu_start = torch.cuda.Event(enable_timing=True)
                gpu_end = torch.cuda.Event(enable_timing=True)
                gpu_start.record()
                '''
                if search_args.query_type == 'text':
                    lookup_indices.extend(text_ids)
                else:
                    lookup_indices.extend(img_ids)
                if model_args.model_name_or_path == './checkpoints/llava-hf-llava-1.5-7b-hf' or model_args.model_name_or_path == './checkpoints/llava-hf-llava-v1.6-vicuna-7b-hf':
                    prompt = img_prompt_no_special_llava_v1_5
                elif 'Qwen2.5-VL-7B-Instruct' in model_args.model_name_or_path or 'Qwen2.5-VL-3B-Instruct' in model_args.model_name_or_path:
                    prompt = img_prompt_qwen_v2_5
                elif 'InternVL2_5-8B' in model_args.model_name_or_path:
                    prompt = img_prompt_intern_vl_v2_5
                else:
                    prompt = img_prompt
                # batch = batch.to(training_args.device)
                # batch['qids'] = batch_ids
                # model_output: EncoderOutput = model(query=batch)
                if 'disassembleeol' in model_args.eol_type:
                    prompts = llama3_retrieval_disassemble_image_prompts
                else:
                    prompts = llama3_retrieval_disassemble_image_prompts

                if model_args.calculate_type == 'separate':
                    if search_args.query_type == 'text':
                        query_logits, query_dense_reps = model.encode_data(texts, 'text', processor, device, model_args,
                                                                           data_args)
                        if 'disassembleeol_concrete' in model_args.eol_type:
                            disassemble_logits = query_logits[data_args.per_device_batch_size:]
                            query_logits = query_logits[:data_args.per_device_batch_size]
                        elif 'disassembleeol' in model_args.eol_type:
                            disassemble_logits = query_logits
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
                            query_logits, query_dense_reps = model.encode_data(imgs, 'image', processor, device,
                                                                               model_args,
                                                                               data_args)
                        elif model_args.eol_type == 'disassembleeol_concrete' or model_args.eol_type == 'disassembleeol_separate' or model_args.eol_type == 'disassembleeol_separate_origin_text' or model_args.eol_type == 'disassembleeol_concrete_origin_text':
                            # 这是参考metaeol的思路，试图将图文中的不同元素拆解出来，目前先把这个处理放在稀疏检索上，然后再看看密集检索是否使用
                            raw_images = [Image.open(path).convert('RGB') for path in imgs_path]
                            img_inputs = processor(images=raw_images, text=[prompt] * len(imgs_path),
                                                   return_tensors="pt",
                                                   padding=True)
                            imgs = img_inputs.to(device)
                            if model_args.eol_type == 'disassembleeol_concrete' or model_args.eol_type == 'disassembleeol_concrete_origin_text':
                                query_logits, query_dense_reps = model.encode_data(imgs, 'image', processor, device,
                                                                                   model_args, data_args)
                            else:
                                _, query_dense_reps = model.encode_data(imgs, 'image', processor, device,
                                                                        model_args, data_args)
                            del imgs

                            # 强制触发垃圾回收
                            gc.collect()
                            # 对于PyTorch，还可以尝试调用torch.cuda.empty_cache()
                            torch.cuda.empty_cache()

                            disassemble_raw_images = [raw_image for raw_image in raw_images for _ in
                                                      range(len(prompts) // 5)]
                            disassemble_logits = [[] for _ in range(len(imgs_path))]
                            for i in range(5):
                                # 这个i是为了控制当前轮次使用哪些prompt编码
                                start = i * len(prompts) // 5
                                end = (i + 1) * len(prompts) // 5

                                disassemble_img_inputs = processor(images=disassemble_raw_images,
                                                                   text=prompts[start:end] * len(imgs_path),
                                                                   return_tensors="pt",
                                                                   padding=True)

                                disassemble_imgs = disassemble_img_inputs.to(device)

                                # 在metaeol模式下，reps应该是[batch_size * len(task_prompts) // 4, reps_dim]
                                disassemble_logits_sub, _ = model.encode_data(disassemble_imgs, 'image', processor,
                                                                              device, model_args,
                                                                              data_args)

                                for j in range(len(imgs_path)):
                                    # 这个j是为了控制要把第j个样本对应的数据存到对应索引下的列表中
                                    disassemble_logits[j].append(
                                        disassemble_logits_sub[j * len(prompts) // 5:(j + 1) * len(prompts) // 5])
                            disassemble_logits = [item for disassemble_logit in disassemble_logits for item in
                                                  disassemble_logit]
                            disassemble_logits = torch.cat(disassemble_logits, dim=0)

                        elif model_args.eol_type == 'all_disassembleeol_concrete' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text':
                            # 这是参考metaeol的思路，试图将图文中的不同元素拆解出来，目前先把这个处理放在稀疏检索上，然后再看看密集检索是否使用
                            raw_images = [Image.open(path).convert('RGB') for path in imgs_path]
                            img_inputs = processor(images=raw_images, text=[prompt] * len(imgs_path),
                                                   return_tensors="pt",
                                                   padding=True)
                            imgs = img_inputs.to(device)
                            query_logits, _ = model.encode_data(imgs, 'image', processor, device,
                                                                model_args, data_args)
                            del imgs

                            # 强制触发垃圾回收
                            gc.collect()
                            # 对于PyTorch，还可以尝试调用torch.cuda.empty_cache()
                            torch.cuda.empty_cache()

                            disassemble_raw_images = [raw_image for raw_image in raw_images for _ in
                                                      range(len(prompts) // 5)]

                            disassemble_logits = [[] for _ in range(len(imgs_path))]
                            disassemble_reps = [[] for _ in range(len(imgs_path))]
                            for i in range(5):
                                # 这个i是为了控制当前轮次使用哪些prompt编码
                                start = i * len(prompts) // 5
                                end = (i + 1) * len(prompts) // 5

                                disassemble_img_inputs = processor(images=disassemble_raw_images,
                                                                   text=prompts[start:end] * len(imgs_path),
                                                                   return_tensors="pt",
                                                                   padding=True)

                                disassemble_imgs = disassemble_img_inputs.to(device)

                                # 在metaeol模式下，reps应该是[batch_size * len(task_prompts) // 4, reps_dim]
                                disassemble_logits_sub, disassemble_reps_sub = model.encode_data(disassemble_imgs,
                                                                                                 'image', processor,
                                                                                                 device, model_args,
                                                                                                 data_args)

                                for j in range(len(imgs_path)):
                                    # 这个j是为了控制要把第j个样本对应的数据存到对应索引下的列表中
                                    disassemble_logits[j].append(
                                        disassemble_logits_sub[j * len(prompts) // 5:(j + 1) * len(prompts) // 5])
                                    disassemble_reps[j].append(
                                        disassemble_reps_sub[j * len(prompts) // 5:(j + 1) * len(prompts) // 5])
                            disassemble_logits = [item for disassemble_logit in disassemble_logits for item in
                                                  disassemble_logit]
                            disassemble_reps = [item for disassemble_rep in disassemble_reps for item in
                                                disassemble_rep]
                            disassemble_logits = torch.cat(disassemble_logits, dim=0)
                            disassemble_reps = torch.cat(disassemble_reps, dim=0)
                            query_dense_reps = disassemble_reps

                        elif model_args.eol_type == 'all_disassembleeol' or model_args.eol_type == 'all_disassembleeol_origin_text':
                            # 这是参考metaeol的思路，试图将图文中的不同元素拆解出来，目前先把这个处理放在稀疏检索上，然后再看看密集检索是否使用
                            raw_images = [Image.open(path).convert('RGB') for path in imgs_path]
                            disassemble_raw_images = [raw_image for raw_image in raw_images for _ in
                                                      range(len(prompts) // 5)]
                            disassemble_logits = [[] for _ in range(len(imgs_path))]
                            disassemble_reps = [[] for _ in range(len(imgs_path))]
                            for i in range(5):
                                # 这个i是为了控制当前轮次使用哪些prompt编码
                                start = i * len(prompts) // 5
                                end = (i + 1) * len(prompts) // 5

                                disassemble_img_inputs = processor(images=disassemble_raw_images,
                                                                   text=prompts[start:end] * len(imgs_path),
                                                                   return_tensors="pt",
                                                                   padding=True)

                                disassemble_imgs = disassemble_img_inputs.to(device)

                                # 在metaeol模式下，reps应该是[batch_size * len(task_prompts) // 4, reps_dim]
                                disassemble_logits_sub, disassemble_reps_sub = model.encode_data(disassemble_imgs,
                                                                                                 'image', processor,
                                                                                                 device, model_args,
                                                                                                 data_args)

                                for j in range(len(imgs_path)):
                                    # 这个j是为了控制要把第j个样本对应的数据存到对应索引下的列表中
                                    disassemble_logits[j].append(
                                        disassemble_logits_sub[j * len(prompts) // 5:(j + 1) * len(prompts) // 5])
                                    disassemble_reps[j].append(
                                        disassemble_reps_sub[j * len(prompts) // 5:(j + 1) * len(prompts) // 5])
                            disassemble_logits = [item for disassemble_logit in disassemble_logits for item in
                                                  disassemble_logit]
                            disassemble_reps = [item for disassemble_rep in disassemble_reps for item in
                                                disassemble_rep]
                            disassemble_logits = torch.cat(disassemble_logits, dim=0)
                            disassemble_reps = torch.cat(disassemble_reps, dim=0)
                            query_dense_reps = disassemble_reps
                else:
                    if data_args.prompt_type == 'prompt_5':
                        prompt_template = llama3_template_image_prefix
                        if 'concrete' in model_args.eol_type or 'all' not in model_args.eol_type:
                            prompt_template += llama3_template_content_element.format(img_prompt_for_concat)
                        for llama3_retrieval_disassemble_image_prompt in retrieval_disassemble_image_prompts_for_concat:
                            content_element = llama3_template_content_element.format(
                                llama3_retrieval_disassemble_image_prompt)
                            prompt_template += content_element
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
                    if search_args.query_type == 'text':
                        query_logits, query_dense_reps = model.encode_data_concat(texts, 'text', processor, device,
                                                                                  model_args, data_args)
                        if 'disassembleeol_concrete' in model_args.eol_type:
                            disassemble_logits = query_logits[data_args.per_device_batch_size:]
                            query_logits = query_logits[:data_args.per_device_batch_size]
                        elif 'disassembleeol' in model_args.eol_type:
                            disassemble_logits = query_logits

                    else:
                        raw_images = [Image.open(path).convert('RGB') for path in imgs_path]
                        img_inputs = processor(images=raw_images, text=[prompt_template] * len(imgs_path),
                                               return_tensors="pt",
                                               padding=True)
                        imgs = img_inputs.to(device)
                        query_logits, query_dense_reps = model.encode_data_concat(imgs, 'image', processor, device,
                                                                                  model_args,
                                                                                  data_args)
                        if 'disassembleeol_concrete' in model_args.eol_type:
                            disassemble_logits = query_logits[data_args.per_device_batch_size:]
                            query_logits = query_logits[:data_args.per_device_batch_size]
                        elif 'disassembleeol' in model_args.eol_type:
                            disassemble_logits = query_logits

                '''
                gpu_end.record()
                torch.cuda.synchronize()  # 等待GPU完成
                '''

                # CPU时间结束
                cpu_end = time.time()

                # model_cpu_time.append(cpu_end - cpu_start)
                # model_gpu_time.append(gpu_start.elapsed_time(gpu_end))
                embedding_time.append(cpu_end - cpu_start)

                query_dense_reps = F.normalize(query_dense_reps, dim=-1)
                if model_args.eol_type == 'all_disassembleeol' or model_args.eol_type == 'all_disassembleeol_origin_text' or model_args.eol_type == 'all_disassembleeol_concrete' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text':
                    query_dense_reps = query_dense_reps.reshape(-1, len(prompts),
                                                                query_dense_reps.shape[1]).mean(1)
                if search_args.query_type == 'text':
                    batch_ids = text_ids
                else:
                    batch_ids = img_ids
                # print(batch_ids)
                # 这里把对应的id和rep关联起来，这里的id应该是int型数据
                id_to_dense = {}
                for id, dense_rep in zip(batch_ids, query_dense_reps):
                    id_to_dense[id] = dense_rep

                batch_topics = []
                if 'disassembleeol' in model_args.eol_type:
                    if search_args.query_type == 'text':
                        # CPU时间开始
                        cpu_start = time.time()
                        '''
                        # GPU事件开始
                        gpu_start = torch.cuda.Event(enable_timing=True)
                        gpu_end = torch.cuda.Event(enable_timing=True)
                        gpu_start.record()
                        '''
                        for text_indice in range(len(batch_ids)):
                            id = batch_ids[text_indice]
                            if model_args.eol_type == 'disassembleeol_concrete' or model_args.eol_type == 'disassembleeol_concrete_origin_text' or model_args.eol_type == 'all_disassembleeol_concrete' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text':
                                logit = query_logits[text_indice]
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
                            vector = dict()
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
                            if data_args.sparse_value_mean:
                                for token in vector.keys():
                                    if data_args.prompt_type == 'prompt_5':
                                        vector[token] //= 5
                                    elif data_args.prompt_type == 'prompt_7':
                                        vector[token] //= 7
                                    else:
                                        vector[token] //= 3
                            query = ""
                            for token, v in vector.items():
                                query += (' ' + token) * v
                            batch_topics.append(query.strip())
                        '''
                        gpu_end.record()
                        torch.cuda.synchronize()  # 等待GPU完成
                        '''

                        # CPU时间结束
                        cpu_end = time.time()
                        sparse_obtain_time.append(cpu_end - cpu_start)
                        # CPU时间开始
                        cpu_start = time.time()
                        '''
                        # GPU事件开始
                        gpu_start = torch.cuda.Event(enable_timing=True)
                        gpu_end = torch.cuda.Event(enable_timing=True)
                        gpu_start.record()
                        '''
                        sparse_scores, sparse_rankings = sparse_search(sparse_retriever, batch_topics,
                                                                       batch_ids,
                                                                       search_args)
                        '''
                        gpu_end.record()
                        torch.cuda.synchronize()  # 等待GPU完成
                        '''

                        # CPU时间结束
                        cpu_end = time.time()
                        sparse_search_time.append(cpu_end - cpu_start)
                    else:
                        # CPU时间开始
                        cpu_start = time.time()
                        '''
                        # GPU事件开始
                        gpu_start = torch.cuda.Event(enable_timing=True)
                        gpu_end = torch.cuda.Event(enable_timing=True)
                        gpu_start.record()
                        '''
                        for img_indice in range(len(batch_ids)):
                            id = batch_ids[img_indice]
                            if model_args.eol_type == 'disassembleeol_concrete' or model_args.eol_type == 'disassembleeol_concrete_origin_text' or model_args.eol_type == 'all_disassembleeol_concrete' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text':
                                logit = query_logits[img_indice]
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
                            vector = dict()
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
                            if data_args.sparse_value_mean:
                                for token in vector.keys():
                                    if data_args.prompt_type == 'prompt_5':
                                        vector[token] //= 5
                                    elif data_args.prompt_type == 'prompt_7':
                                        vector[token] //= 7
                                    else:
                                        vector[token] //= 3
                            query = ""
                            for token, v in vector.items():
                                query += (' ' + token) * v
                            batch_topics.append(query.strip())
                        '''
                        gpu_end.record()
                        torch.cuda.synchronize()  # 等待GPU完成
                        '''

                        # CPU时间结束
                        cpu_end = time.time()
                        sparse_obtain_time.append(cpu_end - cpu_start)
                        # CPU时间开始
                        cpu_start = time.time()
                        '''
                        # GPU事件开始
                        gpu_start = torch.cuda.Event(enable_timing=True)
                        gpu_end = torch.cuda.Event(enable_timing=True)
                        gpu_start.record()
                        '''
                        sparse_scores, sparse_rankings = sparse_search(sparse_retriever, batch_topics,
                                                                       batch_ids,
                                                                       search_args)
                        '''
                        gpu_end.record()
                        torch.cuda.synchronize()  # 等待GPU完成
                        '''

                        # CPU时间结束
                        cpu_end = time.time()
                        sparse_search_time.append(cpu_end - cpu_start)
                else:
                    if search_args.query_type == 'text':
                        for _, logits, text in zip(batch_ids, query_logits, texts):
                            vector = dict()
                            tokens, values = get_text_valid_tokens_values(text, processor.tokenizer,
                                                                          logits,
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

                            query = ""
                            for token, v in vector.items():
                                query += (' ' + token) * v
                            batch_topics.append(query.strip())
                        sparse_scores, sparse_rankings = sparse_search(sparse_retriever, batch_topics,
                                                                       batch_ids,
                                                                       search_args)
                    else:
                        for _, logits, text in zip(batch_ids, query_logits, texts):
                            vector = dict()
                            if model_args.eol_type == 'prompteol_same_length':
                                tokens, values = get_img_valid_tokens_values(processor.tokenizer,
                                                                             logits,
                                                                             vocab_dict,
                                                                             data_args,
                                                                             filtered_ids, text=text)
                            else:
                                tokens, values = get_img_valid_tokens_values(processor.tokenizer,
                                                                             logits,
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

                            query = ""
                            for token, v in vector.items():
                                query += (' ' + token) * v
                            batch_topics.append(query.strip())
                        sparse_scores, sparse_rankings = sparse_search(sparse_retriever, batch_topics,
                                                                       batch_ids,
                                                                       search_args)

                # 到这里上面的部分，一阶段稀疏检索已经完成，下面是为每个数据选择前若干个候选结果，构造新的子集并找出对应的密集特征
                batch_sparse_run = get_run_dict(batch_ids, sparse_scores, sparse_rankings, search_args.remove_query)
                dense_scores_list = []
                dense_rankings_list = []
                # 在batch_sparse_run中，k是int型，v['docs']的键是字符串
                for k, v in batch_sparse_run.items():
                    # sorted_by_value = sorted(v['docs'].items(), key=lambda x: x[1], reverse=True)
                    # sorted_by_value_dict = dict(sorted_by_value[:search_args.first_stage_search_sum])
                    sorted_by_value_dict = dict(list(v['docs'].items())[:search_args.first_stage_search_sum])
                    '''
                    if dist.get_rank() == 0:
                        print(len(sorted_by_value))
                        print(len(sorted_by_value_dict))
                        print(sorted_by_value_dict)
                    '''
                    if len(sorted_by_value_dict) != 0:
                        # CPU时间开始
                        cpu_start = time.time()
                        '''
                        # GPU事件开始
                        gpu_start = torch.cuda.Event(enable_timing=True)
                        gpu_end = torch.cuda.Event(enable_timing=True)
                        gpu_start.record()
                        '''
                        min_value = min(sorted_by_value_dict.values())
                        max_value = max(sorted_by_value_dict.values())
                        batch_sparse_run[k] = {'docs': sorted_by_value_dict, 'min_score': min_value,
                                               'max_score': max_value}
                        query_dense_rep = id_to_dense[k]

                        # 由于经过一阶段粗排后，每个数据的结果都不同，所以要单独处理每个数据的密集检索器
                        single_look_up = []
                        dense_retriever = FaissFlatSearcher(p_reps_0)
                        chosen_lookup_to_reps = []
                        for p_lookup in sorted_by_value_dict.keys():
                            # 这里目前不太确定add输入的np.array应该具体是什么样的格式，通过输出search.sh观察，发现dense_retriever.add
                            # 接受的是[[], [], ..., []]这样的结构，只不过我们现在是一个一个数据增加而不是一批数据增加，
                            # 所以暂时先写成一个np.array里面套了一个array
                            chosen_lookup_to_reps.append(lookup_to_reps[p_lookup])
                            single_look_up += [p_lookup]
                        dense_retriever.add(np.array(chosen_lookup_to_reps))
                        '''
                        gpu_end.record()
                        torch.cuda.synchronize()  # 等待GPU完成
                        '''

                        # CPU时间结束
                        cpu_end = time.time()
                        dense_obtain_time.append(cpu_end - cpu_start)
                        del chosen_lookup_to_reps
                        # 强制触发垃圾回收
                        gc.collect()
                        # 对于PyTorch，还可以尝试调用torch.cuda.empty_cache()
                        torch.cuda.empty_cache()
                        # CPU时间开始
                        cpu_start = time.time()
                        '''
                        # GPU事件开始
                        gpu_start = torch.cuda.Event(enable_timing=True)
                        gpu_end = torch.cuda.Event(enable_timing=True)
                        gpu_start.record()
                        '''
                        if search_args.use_gpu:
                            num_gpus = faiss.get_num_gpus()
                            if num_gpus == 0:
                                logger.error("No GPU found. Back to CPU.")
                            else:
                                logger.info(f"Using {num_gpus} GPU")
                                if num_gpus == 1:
                                    co = faiss.GpuClonerOptions()
                                    co.useFloat16 = True
                                    res = faiss.StandardGpuResources()
                                    dense_retriever.index = faiss.index_cpu_to_gpu(res, 0, dense_retriever.index, co)
                                else:
                                    co = faiss.GpuMultipleClonerOptions()
                                    co.shard = True
                                    co.useFloat16 = True
                                    dense_retriever.index = faiss.index_cpu_to_all_gpus(dense_retriever.index, co,
                                                                                        ngpu=num_gpus)
                        '''
                        gpu_end.record()
                        torch.cuda.synchronize()  # 等待GPU完成
                        '''

                        # CPU时间结束
                        cpu_end = time.time()
                        dense_retriever_time.append(cpu_end - cpu_start)
                        # CPU时间开始
                        cpu_start = time.time()
                        '''
                        # GPU事件开始
                        gpu_start = torch.cuda.Event(enable_timing=True)
                        gpu_end = torch.cuda.Event(enable_timing=True)
                        gpu_start.record()
                        '''
                        if search_args.use_candidate_sum:
                            dense_scores, dense_rankings = search_queries_two_stage(dense_retriever,
                                                                                    query_dense_rep.cpu().unsqueeze(
                                                                                        0).detach().float().numpy(),
                                                                                    single_look_up, search_args,
                                                                                    candidate_sum=len(
                                                                                        sorted_by_value_dict))
                        else:
                            dense_scores, dense_rankings = search_queries_two_stage(dense_retriever,
                                                                                    query_dense_rep.cpu().unsqueeze(
                                                                                        0).detach().float().numpy(),
                                                                                    single_look_up, search_args)
                        '''
                        gpu_end.record()
                        torch.cuda.synchronize()  # 等待GPU完成
                        '''

                        # CPU时间结束
                        cpu_end = time.time()
                        dense_search_time.append(cpu_end - cpu_start)
                        dense_scores_list.append(dense_scores[0])
                        dense_rankings_list.append(dense_rankings[0])
                        '''
                        if dist.get_rank() == 0:
                            # print(len(sorted_by_value))
                            # print(len(sorted_by_value_dict))
                            print(len(dense_scores[0]))
                            print(dense_scores[0])
                            print(dense_rankings[0])
                        '''
                    else:
                        min_value = 0
                        max_value = 1
                        batch_sparse_run[k] = {'docs': {}, 'min_score': min_value,
                                               'max_score': max_value}
                        dense_scores_list.append([])
                        dense_rankings_list.append([])

                sparse_run.update(batch_sparse_run)
                dense_run.update(
                    get_run_dict(batch_ids, dense_scores_list, dense_rankings_list, search_args.remove_query))


    if len(embedding_time) == 0:
        mean_embedding_time = 0
    else:
        mean_embedding_time = sum(embedding_time) / len(embedding_time)
    if len(dense_search_time) == 0:
        mean_dense_search_time = 0
    else:
        mean_dense_search_time = sum(dense_search_time) / len(dense_search_time)
    if len(sparse_search_time) == 0:
        mean_sparse_search_time = 0
    else:
        mean_sparse_search_time = sum(sparse_search_time) / len(sparse_search_time)
    if len(sparse_obtain_time) == 0:
        mean_sparse_obtain_time = 0
    else:
        mean_sparse_obtain_time = sum(sparse_obtain_time) / len(sparse_obtain_time)
    if len(dense_obtain_time) == 0:
        mean_dense_obtain_time = 0
    else:
        mean_dense_obtain_time = sum(dense_obtain_time) / len(dense_obtain_time)
    if len(dense_retriever_time) == 0:
        mean_dense_retriever_time = 0
    else:
        mean_dense_retriever_time = sum(dense_retriever_time) / len(dense_retriever_time)

    print(f'rank {rank}, sum of embedding_time: {sum(embedding_time)}, sum of dense search time: '
          f'{sum(dense_search_time)}, sum of sparse search time: {sum(sparse_search_time)}, '
          f'sum of sparse obtain time: {sum(sparse_obtain_time)}, '
          f'sum of dense obtain time: {sum(dense_obtain_time)}, '
          f'sum of dense retriever time: {sum(dense_retriever_time)}'
          f', mean of embedding_time: {mean_embedding_time}, '
          f'mean of dense search time: {mean_dense_search_time}, '
          f'mean of sparse search time: {mean_sparse_search_time}, '
          f'mean of sparse obtain time: {mean_sparse_obtain_time}, '
          f'mean of dense obtain time: {mean_dense_obtain_time}, '
          f'mean of dense retriever time: {mean_dense_retriever_time}')

    # 训练结束后添加同步屏障
    dist.barrier()

    # 确保所有进程同步退出
    if dist.get_rank() == 0:
        # 主进程最后退出
        torch.distributed.destroy_process_group()
    else:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
