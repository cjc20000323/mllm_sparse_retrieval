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
    retrieval_disassemble_image_prompts_for_concat, img_prompt_for_concat
from hybrid import fuse, normalize
from utils import load_image
from peft import PeftModel
from search import pickle_load, search_queries, sparse_search, get_run_dict
import time

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

    if search_args.embedding_type == 'dense':
        def equal(name):
            return name

        encoder.language_model.lm_head = equal

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
    fusion_run = {}

    dense_retriever_indices = []
    sparse_retriever_indices = []
    model_cpu_time = []
    model_gpu_time = []
    similarity_cpu_time = []
    similarity_gpu_time = []

    if search_args.passage_reps is not None:
        # 目前尚不清楚这里是怎么工作的
        # 另外，这里源代码里有multi_reps，暂时先不管，后面再加
        dense_retriever_indices = [search_args.passage_reps]

    if search_args.sparse_index is not None:
        # 目前尚不清楚这里是怎么工作的
        # 另外，这里源代码里有multi_reps，暂时先不管，后面再加
        sparse_retriever_indices = [search_args.sparse_index]

    if dist.get_rank() == 0:
        print(max(len(dense_retriever_indices), len(sparse_retriever_indices)))
        print(dense_retriever_indices)
        print(sparse_retriever_indices)
    for i in range(max(len(dense_retriever_indices), len(sparse_retriever_indices))):

        dense_retriever = None
        sparse_retriever = None

        if dense_retriever_indices and search_args.embedding_type != 'sparse':
            index_files = glob.glob(os.path.join(dense_retriever_indices[i], 'corpus*.pkl'))
            if dist.get_rank() == 0:
                print(f'Pattern match found {len(index_files)} files; loading them into dense index.')

            p_reps_0, p_lookup_0 = pickle_load(index_files[0])
            print(p_reps_0.shape)
            dense_retriever = FaissFlatSearcher(p_reps_0)
            # 经DeepSeek老师讲解，他说FaissFlatSearcher初始化时仅分配了内存结构，未添加任何数据。所以这里再重新加一下，
            # 这也和源代码中重复add了p_reps_0一致，希望D老师没骗我吧
            # dense_retriever.add(p_reps_0)

            # 在源代码里，并没有将所有数据都转移到某个GPU上面保存，而是各自保存，这样的话corpus会有多个编号，因此会有下面这一段处理多个corpus的代码，
            # 但是我们这里是先集中后保存，这样就只有一个文件，所以就先注释掉了
            # 经过修改，现在是每个gpu在encode的时候处理各自数据并各自保存一个文件，所以现在应当按照原来的方式处理
            shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
            if len(index_files) > 1:
                shards = tqdm(shards, desc='Loading shards into index', total=len(index_files))
            look_up = []
            for p_reps, p_lookup in shards:
                dense_retriever.add(p_reps)
                look_up += p_lookup
            if dist.get_rank() == 0:
                print(len(look_up))
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

        if sparse_retriever_indices and search_args.embedding_type != 'dense':
            sparse_retriever = LuceneImpactSearcher(os.path.join(sparse_retriever_indices[i], 'index'), None)
            analyzer = JWhiteSpaceAnalyzer()
            sparse_retriever.set_analyzer(analyzer)

        with torch.no_grad(), torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
            for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(test_dataloader),
                                                                         total=len(test_dataloader)):
                # CPU时间开始
                cpu_start = time.time()

                # GPU事件开始
                gpu_start = torch.cuda.Event(enable_timing=True)
                gpu_end = torch.cuda.Event(enable_timing=True)
                gpu_start.record()
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

                if 'disassembleeol' in model_args.eol_type:
                    prompts = llama3_retrieval_disassemble_image_prompts
                else:
                    prompts = llama3_retrieval_disassemble_image_prompts

                if model_args.calculate_type == 'separate':
                    if search_args.query_type == 'text':
                        if search_args.embedding_type == 'dense':
                            query_dense_reps = model.encode_data_for_interface(texts, 'text',
                                                                               search_args.embedding_type,
                                                                               processor, device, model_args, data_args)
                            query_dense_reps = F.normalize(query_dense_reps, dim=-1)
                        elif search_args.embedding_type == 'sparse':
                            query_logits = model.encode_data_for_interface(texts, 'text', search_args.embedding_type,
                                                                           processor,
                                                                           device, model_args, data_args)
                            if 'disassembleeol_concrete' in model_args.eol_type:
                                disassemble_logits = query_logits[data_args.per_device_batch_size:]
                                query_logits = query_logits[:data_args.per_device_batch_size]
                            elif 'disassembleeol' in model_args.eol_type:
                                disassemble_logits = query_logits
                        else:
                            query_logits, query_dense_reps = model.encode_data_for_interface(texts, 'text',
                                                                                             search_args.embedding_type,
                                                                                             processor, device,
                                                                                             model_args,
                                                                                             data_args)
                            query_dense_reps = F.normalize(query_dense_reps, dim=-1)
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
                            if search_args.embedding_type == 'dense':
                                query_dense_reps = model.encode_data_for_interface(imgs, 'image',
                                                                                   search_args.embedding_type,
                                                                                   processor,
                                                                                   device, model_args, data_args)
                                query_dense_reps = F.normalize(query_dense_reps, dim=-1)
                            elif search_args.embedding_type == 'sparse':
                                query_logits = model.encode_data_for_interface(imgs, 'image',
                                                                               search_args.embedding_type,
                                                                               processor, device, model_args, data_args)
                            else:
                                query_logits, query_dense_reps = model.encode_data_for_interface(texts, 'text',
                                                                                                 search_args.embedding_type,
                                                                                                 processor, device,
                                                                                                 model_args,
                                                                                                 data_args)
                                query_dense_reps = F.normalize(query_dense_reps, dim=-1)
                        elif model_args.eol_type == 'disassembleeol_concrete' or model_args.eol_type == 'disassembleeol_separate' or model_args.eol_type == 'disassembeleeol_separate_origin_text':
                            # 这是参考metaeol的思路，试图将图文中的不同元素拆解出来，目前先把这个处理放在稀疏检索上，然后再看看密集检索是否使用
                            raw_images = [Image.open(path).convert('RGB') for path in imgs_path]
                            img_inputs = processor(images=raw_images, text=[prompt] * len(imgs_path),
                                                   return_tensors="pt",
                                                   padding=True)
                            imgs = img_inputs.to(device)
                            if search_args.embedding_type == 'dense':
                                query_dense_reps = model.encode_data_for_interface(imgs, 'image',
                                                                                   search_args.embedding_type,
                                                                                   processor,
                                                                                   device, model_args, data_args)
                                query_dense_reps = F.normalize(query_dense_reps, dim=-1)
                            elif search_args.embedding_type == 'sparse':
                                query_logits = model.encode_data_for_interface(imgs, 'image',
                                                                               search_args.embedding_type,
                                                                               processor, device, model_args, data_args)
                            else:
                                if model_args.eol_type == 'disassembleeol_concrete':
                                    query_logits, query_dense_reps = model.encode_data_for_interface(imgs, 'image',
                                                                                                     search_args.embedding_type,
                                                                                                     processor, device,
                                                                                                     model_args,
                                                                                                     data_args)
                                else:
                                    _, query_dense_reps = model.encode_data_for_interface(imgs, 'image',
                                                                                          search_args.embedding_type,
                                                                                          processor, device,
                                                                                          model_args,
                                                                                          data_args)
                                query_dense_reps = F.normalize(query_dense_reps, dim=-1)
                            del imgs

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
                                if search_args.embedding_type != 'dense':
                                    disassemble_logits_sub, _ = model.encode_data_for_interface(disassemble_imgs,
                                                                                                'image',
                                                                                                search_args.embedding_type,
                                                                                                processor,
                                                                                                device, model_args,
                                                                                                data_args)
                                    for j in range(len(imgs_path)):
                                        # 这个j是为了控制要把第j个样本对应的数据存到对应索引下的列表中
                                        disassemble_logits[j].append(
                                            disassemble_logits_sub[j * len(prompts) // 5:(j + 1) * len(prompts) // 5])

                            if search_args.embedding_type != 'dense':
                                disassemble_logits = [item for disassemble_logit in disassemble_logits for item in
                                                      disassemble_logit]
                                disassemble_logits = torch.cat(disassemble_logits, dim=0)

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
                                if search_args.embedding_type == 'dense':
                                    disassemble_reps_sub = model.encode_data_for_interface(disassemble_imgs, 'image',
                                                                                           search_args.embedding_type,
                                                                                           processor,
                                                                                           device, model_args,
                                                                                           data_args)
                                    for j in range(len(imgs_path)):
                                        # 这个j是为了控制要把第j个样本对应的数据存到对应索引下的列表中
                                        disassemble_reps[j].append(
                                            disassemble_reps_sub[j * len(prompts) // 5:(j + 1) * len(prompts) // 5])
                                elif search_args.embedding_type == 'sparse':
                                    disassemble_logits_sub = model.encode_data_for_interface(disassemble_imgs, 'image',
                                                                                             search_args.embedding_type,
                                                                                             processor,
                                                                                             device, model_args,
                                                                                             data_args)

                                    for j in range(len(imgs_path)):
                                        # 这个j是为了控制要把第j个样本对应的数据存到对应索引下的列表中
                                        disassemble_logits[j].append(
                                            disassemble_logits_sub[j * len(prompts) // 5:(j + 1) * len(prompts) // 5])
                                else:
                                    disassemble_logits_sub, disassemble_reps_sub = model.encode_data_for_interface(
                                        disassemble_imgs, 'image',
                                        search_args.embedding_type,
                                        processor,
                                        device, model_args,
                                        data_args)

                                    for j in range(len(imgs_path)):
                                        # 这个j是为了控制要把第j个样本对应的数据存到对应索引下的列表中
                                        disassemble_logits[j].append(
                                            disassemble_logits_sub[j * len(prompts) // 5:(j + 1) * len(prompts) // 5])
                                        disassemble_reps[j].append(
                                            disassemble_reps_sub[j * len(prompts) // 5:(j + 1) * len(prompts) // 5])

                            if search_args.embedding_type == 'dense':
                                disassemble_reps = [item for disassemble_rep in disassemble_reps for item in
                                                    disassemble_rep]
                                disassemble_reps = torch.cat(disassemble_reps, dim=0)
                                query_dense_reps = disassemble_reps
                                query_dense_reps = F.normalize(query_dense_reps, dim=-1)
                            elif search_args.embedding_type == 'sparse':
                                disassemble_logits = [item for disassemble_logit in disassemble_logits for item in
                                                      disassemble_logit]
                                disassemble_logits = torch.cat(disassemble_logits, dim=0)
                            else:
                                disassemble_logits = [item for disassemble_logit in disassemble_logits for item in
                                                      disassemble_logit]
                                disassemble_reps = [item for disassemble_rep in disassemble_reps for item in
                                                    disassemble_rep]
                                disassemble_logits = torch.cat(disassemble_logits, dim=0)
                                disassemble_reps = torch.cat(disassemble_reps, dim=0)
                                query_dense_reps = disassemble_reps
                                query_dense_reps = F.normalize(query_dense_reps, dim=-1)

                        else:
                            # 希望获得这样的列表[a,a,a,b,b,b,c,c,c......]
                            # 也就是说，对于批次中的每个图像，按照下面每次循环使用的prompt个数，加入到raw_images中
                            raw_images = [Image.open(path).convert('RGB') for
                                          path in imgs_path for _ in range(len(task_image_prompts) // 4)]
                            # 将task_prompt添加到llama3_template中
                            prompts = [llama3_template.format(task_image_prompt) for task_image_prompt in
                                       task_image_prompts]

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
                                logits_sub, reps_sub = model.encode_data(imgs, 'image', processor, device, model_args,
                                                                         data_args)

                                for j in range(len(imgs_path)):
                                    # 这个j是为了控制要把第j个样本对应的数据存到对应索引下的列表中
                                    logits[j].append(logits_sub[j * len(prompts) // 4:(j + 1) * len(prompts) // 4])
                                    reps[j].append(reps_sub[j * len(prompts) // 4:(j + 1) * len(prompts) // 4])

                            logits = [item for logit in logits for item in logit]
                            reps = [item for rep in reps for item in rep]

                            logits = torch.cat(logits, dim=0)
                            reps = torch.cat(reps, dim=0)

                            query_logits = logits.reshape(-1, len(task_image_prompts), logits.shape[1]).mean(1)
                            query_dense_reps = reps.reshape(-1, len(task_image_prompts), reps.shape[1]).mean(1)

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

                gpu_end.record()
                torch.cuda.synchronize()  # 等待GPU完成

                # CPU时间结束
                cpu_end = time.time()

                model_cpu_time.append(cpu_end - cpu_start)
                model_gpu_time.append(gpu_start.elapsed_time(gpu_end))

                if search_args.query_type == 'text':
                    batch_ids = text_ids
                else:
                    batch_ids = img_ids
                # CPU时间开始
                cpu_start = time.time()
                # GPU事件开始
                gpu_start = torch.cuda.Event(enable_timing=True)
                gpu_end = torch.cuda.Event(enable_timing=True)
                gpu_start.record()
                if dense_retriever is not None:
                    if isinstance(query_dense_reps, list):
                        for qid, reps in zip(batch_ids, query_dense_reps):
                            reps = torch.stack(reps, dim=0)
                            dense_scores, dense_rankings = search_queries(dense_retriever,
                                                                          reps.cpu().detach().float().numpy(),
                                                                          look_up, search_args)
                            if qid not in dense_run:
                                dense_run[qid] = []
                                for scores, ranking in zip(dense_scores, dense_rankings):
                                    dense_run[qid].append(
                                        [get_run_dict([qid], [scores], [ranking], search_args.remove_query)])
                            else:
                                for i, (scores, ranking) in enumerate(zip(dense_scores, dense_rankings)):
                                    dense_run[qid][i].append(
                                        get_run_dict([qid], [scores], [ranking], search_args.remove_query))

                    else:
                        if model_args.eol_type == 'all_disassembleeol' or model_args.eol_type == 'all_disassembleeol_origin_text':
                            if model_args.calculate_type == 'concat':
                                if data_args.prompt_type == 'prompt_5':
                                    prompt_length = 5
                                elif data_args.prompt_type == 'prompt_3':
                                    prompt_length = 3
                                else:
                                    prompt_length = 5
                            else:
                                prompt_length = 5
                            query_dense_reps = query_dense_reps.reshape(-1, prompt_length,
                                                                        query_dense_reps.shape[1]).mean(1)
                        query_dense_reps = query_dense_reps.cpu().detach().float().numpy()
                        dense_scores, dense_rankings = search_queries(dense_retriever, query_dense_reps, look_up,
                                                                      search_args)
                        dense_run.update(
                            get_run_dict(batch_ids, dense_scores, dense_rankings, search_args.remove_query))
                if sparse_retriever is not None:
                    if isinstance(query_logits, list):
                        if search_args.query_type == 'text':
                            for qid, reps, text in zip(batch_ids, query_logits, texts):
                                batch_topics = []
                                for logits in reps:
                                    if model_args.use_output_embedding_cluster:
                                        if 'InternVL2_5-8B' in model_args.model_name_or_path:
                                            tokens, values = get_text_valid_tokens_values_with_cluster(text, processor,
                                                                                                       logits,
                                                                                                       centroids_dict,
                                                                                                       origin_to_centroids_dict,
                                                                                                       data_args,
                                                                                                       filtered_ids)
                                        else:
                                            tokens, values = get_text_valid_tokens_values_with_cluster(text,
                                                                                                       processor.tokenizer,
                                                                                                       logits,
                                                                                                       centroids_dict,
                                                                                                       origin_to_centroids_dict,
                                                                                                       data_args,
                                                                                                       filtered_ids)
                                    else:
                                        if 'InternVL2_5-8B' in model_args.model_name_or_path:
                                            tokens, values = get_text_valid_tokens_values(text, processor, logits,
                                                                                          vocab_dict,
                                                                                          data_args,
                                                                                          filtered_ids)
                                        else:
                                            tokens, values = get_text_valid_tokens_values(text, processor.tokenizer,
                                                                                          logits,
                                                                                          vocab_dict,
                                                                                          data_args, filtered_ids)
                                    query = ""
                                    for token, v in zip(tokens, values):
                                        query += (' ' + token) * v
                                    batch_topics.append(query.strip())
                                sparse_scores, sparse_rankings = sparse_search(sparse_retriever, batch_topics,
                                                                               [qid] * len(batch_topics),
                                                                               search_args)
                                if qid not in sparse_run:
                                    sparse_run[qid] = []
                                    for scores, ranking in zip(sparse_scores, sparse_rankings):
                                        sparse_run[qid].append(
                                            [get_run_dict([qid], [scores], [ranking], search_args.remove_query)])
                                else:
                                    for i, (scores, ranking) in enumerate(zip(sparse_scores, sparse_rankings)):
                                        sparse_run[qid][i].append(
                                            get_run_dict([qid], [scores], [ranking], search_args.remove_query))
                        if search_args.query_type == 'image':
                            for qid, reps in zip(batch_ids, query_logits):
                                batch_topics = []
                                for logits in reps:
                                    if model_args.use_output_embedding_cluster:
                                        if 'InternVL2_5-8B' in model_args.model_name_or_path:
                                            tokens, values = get_img_valid_tokens_values_with_cluster(processor, logits,
                                                                                                      centroids_dict,
                                                                                                      origin_to_centroids_dict,
                                                                                                      data_args,
                                                                                                      filtered_ids)
                                        else:
                                            tokens, values = get_img_valid_tokens_values_with_cluster(
                                                processor.tokenizer, logits,
                                                centroids_dict, origin_to_centroids_dict,
                                                data_args, filtered_ids)
                                    else:
                                        if 'InternVL2_5-8B' in model_args.model_name_or_path:
                                            tokens, values = get_img_valid_tokens_values(processor, logits, vocab_dict,
                                                                                         data_args, filtered_ids)
                                        else:
                                            tokens, values = get_img_valid_tokens_values(processor.tokenizer, logits,
                                                                                         vocab_dict,
                                                                                         data_args, filtered_ids)
                                    query = ""
                                    for token, v in zip(tokens, values):
                                        query += (' ' + token) * v
                                    batch_topics.append(query.strip())
                                sparse_scores, sparse_rankings = sparse_search(sparse_retriever, batch_topics,
                                                                               [qid] * len(batch_topics),
                                                                               search_args)
                                if qid not in sparse_run:
                                    sparse_run[qid] = []
                                    for scores, ranking in zip(sparse_scores, sparse_rankings):
                                        sparse_run[qid].append(
                                            [get_run_dict([qid], [scores], [ranking], search_args.remove_query)])
                                else:
                                    for i, (scores, ranking) in enumerate(zip(sparse_scores, sparse_rankings)):
                                        sparse_run[qid][i].append(
                                            get_run_dict([qid], [scores], [ranking], search_args.remove_query))

                    else:
                        batch_topics = []
                        if 'disassembleeol' in model_args.eol_type:
                            if search_args.query_type == 'text':
                                for text_indice in range(len(batch_ids)):
                                    id = batch_ids[text_indice]
                                    if model_args.eol_type == 'disassembleeol_concrete':
                                        logit = query_logits[text_indice]
                                    text = texts[text_indice]
                                    if data_args.prompt_type == 'prompt_5':
                                        length = 5
                                    elif data_args.prompt_type == 'prompt_3':
                                        length = 3
                                    else:
                                        length = 5
                                    disassemble_logit = disassemble_logits[
                                                        text_indice * length:(text_indice + 1) * length]
                                    vector = dict()
                                    if model_args.eol_type == 'disassembleeol_concrete':
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
                                                vector[token] /= 5
                                            else:
                                                vector[token] /= 3
                                    query = ""
                                    for token, v in vector.items():
                                        query += (' ' + token) * v
                                    batch_topics.append(query.strip())
                                sparse_scores, sparse_rankings = sparse_search(sparse_retriever, batch_topics,
                                                                               batch_ids,
                                                                               search_args)
                                sparse_run.update(
                                    get_run_dict(batch_ids, sparse_scores, sparse_rankings, search_args.remove_query))
                            else:
                                for img_indice in range(len(batch_ids)):
                                    id = batch_ids[img_indice]
                                    if model_args.eol_type == 'disassembleeol_concrete':
                                        logit = query_logits[img_indice]
                                    text = texts[img_indice]
                                    if data_args.prompt_type == 'prompt_5':
                                        length = 5
                                    elif data_args.prompt_type == 'prompt_3':
                                        length = 3
                                    else:
                                        length = 5
                                    disassemble_logit = disassemble_logits[
                                                        img_indice * length:(img_indice + 1) * length]
                                    vector = dict()
                                    if model_args.eol_type == 'disassembleeol_concrete':
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
                                                vector[token] /= 5
                                            else:
                                                vector[token] /= 3
                                    query = ""
                                    for token, v in vector.items():
                                        query += (' ' + token) * v
                                    batch_topics.append(query.strip())
                                sparse_scores, sparse_rankings = sparse_search(sparse_retriever, batch_topics,
                                                                               batch_ids,
                                                                               search_args)
                                sparse_run.update(
                                    get_run_dict(batch_ids, sparse_scores, sparse_rankings, search_args.remove_query))

                        else:
                            if search_args.query_type == 'text':
                                for _, logits, text in zip(batch_ids, query_logits, texts):
                                    vector = dict()
                                    if model_args.use_output_embedding_cluster:
                                        if 'InternVL2_5-8B' in model_args.model_name_or_path:
                                            tokens, values = get_text_valid_tokens_values_with_cluster(text, processor,
                                                                                                       logits,
                                                                                                       centroids_dict,
                                                                                                       origin_to_centroids_dict,
                                                                                                       data_args,
                                                                                                       filtered_ids)
                                        else:
                                            tokens, values = get_text_valid_tokens_values_with_cluster(text,
                                                                                                       processor.tokenizer,
                                                                                                       logits,
                                                                                                       centroids_dict,
                                                                                                       origin_to_centroids_dict,
                                                                                                       data_args,
                                                                                                       filtered_ids)
                                    else:
                                        if 'InternVL2_5-8B' in model_args.model_name_or_path:
                                            tokens, values = get_text_valid_tokens_values(text, processor, logits,
                                                                                          vocab_dict,
                                                                                          data_args, filtered_ids)
                                        else:
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
                                sparse_run.update(
                                    get_run_dict(batch_ids, sparse_scores, sparse_rankings, search_args.remove_query))

                            else:
                                for _, logits, text in zip(batch_ids, query_logits, texts):
                                    vector = dict()
                                    if model_args.use_output_embedding_cluster:
                                        if 'InternVL2_5-8B' in model_args.model_name_or_path:
                                            tokens, values = get_img_valid_tokens_values_with_cluster(processor, logits,
                                                                                                      centroids_dict,
                                                                                                      origin_to_centroids_dict,
                                                                                                      data_args,
                                                                                                      filtered_ids)
                                        else:
                                            tokens, values = get_img_valid_tokens_values_with_cluster(
                                                processor.tokenizer,
                                                logits,
                                                centroids_dict,
                                                origin_to_centroids_dict,
                                                data_args,
                                                filtered_ids)

                                    else:
                                        if 'InternVL2_5-8B' in model_args.model_name_or_path:
                                            tokens, values = get_img_valid_tokens_values(processor, logits, vocab_dict,
                                                                                         data_args, filtered_ids)
                                        else:
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
                                sparse_run.update(
                                    get_run_dict(batch_ids, sparse_scores, sparse_rankings, search_args.remove_query))

                gpu_end.record()
                torch.cuda.synchronize()  # 等待GPU完成

                # CPU时间结束
                cpu_end = time.time()

                similarity_cpu_time.append(cpu_end - cpu_start)
                similarity_gpu_time.append(gpu_start.elapsed_time(gpu_end))

                if model_args.eol_type == 'metaeol':
                    del query_dense_reps
                    del query_logits

        if dense_retriever:
            del dense_retriever
            torch.cuda.empty_cache()

    del model

    if search_args.passage_reps is not None and search_args.sparse_index is not None:
        fusion_run.update(
            fuse(
                runs=[dense_run, sparse_run],
                weights=[search_args.alpha, search_args.beta]
            )
        )

    metric = RecallMetrics(dataset, dense_run, sparse_run, fusion_run, look_up, lookup_indices, search_args)

    metric.sort_and_count()

    metric.all_gather_object()
    metric.print_recall()

    print(f'rank {rank}, sum of model cpu time: {sum(model_cpu_time)}, sum of model gpu time: {sum(model_gpu_time)}'
          f', mean of model cpu time: {sum(model_cpu_time) / len(model_cpu_time)}, '
          f'mean of model gpu time: {sum(model_gpu_time) / len(model_gpu_time)}')
    print(
        f'rank {rank}, sum of similarity cpu time: {sum(similarity_cpu_time)}, sum of similarity gpu time: {sum(similarity_gpu_time)}'
        f', mean of similarity cpu time: {sum(similarity_cpu_time) / len(similarity_cpu_time)}, '
        f'mean of similarity gpu time: {sum(similarity_gpu_time) / len(similarity_gpu_time)}')

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
