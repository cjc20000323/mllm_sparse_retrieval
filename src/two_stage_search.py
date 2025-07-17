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
    get_img_valid_disassemble_tokens_values
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
        shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
        if len(index_files) > 1:
            shards = tqdm(shards, desc='Loading shards into index', total=len(index_files))
        # 将候选集的数据读出后，构建候选id到候选密集特征的字典，供后面
        candidate_reps = []
        candidate_lookup = []
        for p_reps, p_lookup in shards:
            candidate_reps.extend(p_reps)
            candidate_lookup.extend(p_lookup)
        # 这个候选id到候选密集特征字典的键是字符串
        for p_reps, p_lookup in zip(candidate_reps, candidate_lookup):
            lookup_to_reps[p_lookup] = p_reps

        with torch.no_grad(), torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
            for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(test_dataloader),
                                                                         total=len(test_dataloader)):
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
                if search_args.query_type == 'text':
                    query_logits, query_dense_reps = model.encode_data(texts, 'text', processor, device, model_args,
                                                                       data_args)
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
                    query_logits, query_dense_reps = model.encode_data(imgs, 'image', processor, device,
                                                                       model_args,
                                                                       data_args)

                query_dense_reps = F.normalize(query_dense_reps, dim=-1)
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
                    sorted_by_value = sorted(v['docs'].items(), key=lambda x: x[1], reverse=True)
                    sorted_by_value_dict = dict(sorted_by_value[:search_args.first_stage_search_sum])
                    min_value = min(sorted_by_value_dict.values())
                    max_value = max(sorted_by_value_dict.values())
                    batch_sparse_run[k] = {'docs': sorted_by_value_dict, 'min_score': min_value, 'max_score': max_value}
                    query_dense_rep = id_to_dense[k]

                    # 由于经过一阶段粗排后，每个数据的结果都不同，所以要单独处理每个数据的密集检索器
                    look_up = []
                    dense_retriever = FaissFlatSearcher(p_reps_0)
                    for p_lookup in sorted_by_value_dict.keys():
                        # 这里目前不太确定add输入的np.array应该具体是什么样的格式，通过输出search.sh观察，发现dense_retriever.add
                        # 接受的是[[], [], ..., []]这样的结构，只不过我们现在是一个一个数据增加而不是一批数据增加，
                        # 所以暂时先写成一个np.array里面套了一个array
                        dense_retriever.add(np.array([lookup_to_reps[p_lookup]]))
                        look_up += [p_lookup]
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

                    dense_scores, dense_rankings = search_queries(dense_retriever,
                                                                  query_dense_rep.cpu().detach().float().numpy(),
                                                                  look_up, search_args)
                    dense_scores_list.append(dense_scores[0])
                    dense_rankings_list.append(dense_rankings[0])

                sparse_run.update(batch_sparse_run)
                dense_run.update(
                    get_run_dict(batch_ids, dense_scores_list, dense_rankings_list, search_args.remove_query))

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
