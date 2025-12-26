import gc
import glob
import os
import pickle
from contextlib import nullcontext
from itertools import chain

import faiss
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
)
from transformers import LlavaProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, \
    LlavaNextForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, AutoProcessor, \
    AutoModel, Qwen3VLProcessor, Qwen3VLForConditionalGeneration

from arguments import PromptRepsLLMDataArguments, PromptRepsLLMSearchArguments, ModelArguments
from arguments import TrainingArguments
from dataset import CrossModalRetrievalDataset, ComposedTextImageRetrievalDataset, TextPersonRetrievalDataset
from encode import get_filtered_ids
from metrices import RecallMetrics
from model import MLLMRetrievalModel
from reranker import Reranker

torch.set_printoptions(threshold=10000)  # 数字根据你的张量尺寸调整
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from nltk.corpus import stopwords
import string
from template import img_prompt, \
    img_prompt_no_special_llava_v1_5, img_prompt_qwen_v2_5, img_prompt_intern_vl_v2_5, task_image_prompts, \
    llama3_template, task_text_prompts, llama3_retrieval_disassemble_image_prompts
from encode import get_img_valid_tokens_values, get_text_valid_tokens_values, get_img_valid_tokens_values_with_cluster, \
    get_text_valid_tokens_values_with_cluster, get_text_valid_disassemble_tokens_values, get_text_valid_tokens_values_fusion, get_text_valid_disassemble_tokens_values_fusion, \
    get_img_valid_disassemble_tokens_values, llama3_template_image_prefix, llama3_template_content_element, \
    retrieval_disassemble_image_prompts_3_for_concat, \
    retrieval_disassemble_image_prompts_for_concat, img_prompt_for_concat, \
    retrieval_disassemble_image_prompts_7_for_concat, mistral_img_prompt, llava_mistral_template_image_prefix, \
    llava_mistral_template_content_element, person_retrieval_img_prompt_for_concat, person_retrieval_img_prompt_for_concat_1, \
    retrieval_disassemble_image_prompts_person_retrieval_for_concat, \
    retrieval_disassemble_image_prompts_person_retrieval_for_concat_1, \
    retrieval_disassemble_image_origin_prompts_person_retrieval_for_concat, person_retrieval_img_prompt_for_concat_2, \
    img_prompt_qwen_v3, qwen2_5_img_prompt, qwen3_img_prompt, qwen3_template_image_prefix, \
    qwen2_5_template_image_prefix, qwen2_5_template_content_element, qwen3_template_content_element
from hybrid import fuse
from utils import load_image
from peft import PeftModel
from datetime import timedelta

# from cuml.cluster import KMeans

stopwords = set(stopwords.words('english') + list(string.punctuation))

import logging

logger = logging.getLogger(__name__)


def pickle_load(path):
    with open(path, 'rb') as f:
        reps, lookup = pickle.load(f)
    return np.array(reps), lookup


def search_queries(retriever, q_reps, p_lookup, args):
    if args.retrieval_batch_size > 0:
        all_scores, all_indices = retriever.batch_search(q_reps, args.depth, args.retrieval_batch_size, args.quiet)
    else:
        all_scores, all_indices = retriever.search(q_reps, args.depth)

    psg_indices = [[str(p_lookup[x]) for x in q_dd] for q_dd in all_indices]
    psg_indices = np.array(psg_indices)
    return all_scores, psg_indices


def search_queries_two_stage(retriever, q_reps, p_lookup, args, candidate_sum=None):
    if candidate_sum is not None:
        if args.retrieval_batch_size > 0:
            all_scores, all_indices = retriever.batch_search(q_reps, candidate_sum,
                                                             args.retrieval_batch_size,
                                                             args.quiet)
        else:
            all_scores, all_indices = retriever.search(q_reps, candidate_sum)
    else:
        if args.retrieval_batch_size > 0:
            all_scores, all_indices = retriever.batch_search(q_reps, args.first_stage_search_sum,
                                                             args.retrieval_batch_size,
                                                             args.quiet)
        else:
            all_scores, all_indices = retriever.search(q_reps, args.first_stage_search_sum)

    psg_indices = [[str(p_lookup[x]) for x in q_dd] for q_dd in all_indices]
    psg_indices = np.array(psg_indices)
    return all_scores, psg_indices


def get_run_dict(batch_ids, batch_scores, batch_rankings, remove_query):
    run_dict = {}
    for qid, scores, rankings in zip(batch_ids, batch_scores, batch_rankings):
        run_dict[qid] = {}
        run_dict[qid]['docs'] = {}
        for score, doc in zip(scores, rankings):
            '''
            if remove_query:
                if doc == qid:
                    continue
            '''
            run_dict[qid]['docs'][doc] = score
        if len(scores) == 0:
            run_dict[qid]['min_score'] = 0
            run_dict[qid]['max_score'] = 0
        else:
            run_dict[qid]['min_score'] = min(scores)
            run_dict[qid]['max_score'] = max(scores)
    return run_dict


def sparse_search(sparse_retriever, batch_topics, batch_ids, search_args):
    results = sparse_retriever.batch_search(batch_topics, batch_ids, search_args.depth,
                                            threads=search_args.threads)
    results = [(id_, results[id_]) for id_ in batch_ids]
    sparse_scores = []
    sparse_rankings = []
    for topic, hits in results:
        scores = []
        ranking = []
        for hit in hits:
            scores.append(hit.score)
            ranking.append(hit.docid)
        sparse_scores.append([hit.score for hit in hits])
        sparse_rankings.append(ranking)
    return sparse_scores, sparse_rankings


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
            torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=3600))
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
                                                                dtype=torch_type)
        processor = LlavaProcessor.from_pretrained(model_args.model_name_or_path)
    elif 'Qwen2.5-VL-7B-Instruct' in model_args.model_name_or_path or 'Qwen2.5-VL-3B-Instruct' in model_args.model_name_or_path:
        encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_args.model_name_or_path,
                                                                     device_map=device_map,
                                                                     dtype=torch_type)
        processor = Qwen2_5_VLProcessor.from_pretrained(model_args.model_name_or_path)
    elif 'Qwen3-VL-8B-Instruct' in model_args.model_name_or_path:
        encoder = Qwen3VLForConditionalGeneration.from_pretrained(model_args.model_name_or_path,
                                                                     device_map=device_map,
                                                                     dtype=torch_type)
        processor = Qwen3VLProcessor.from_pretrained(model_args.model_name_or_path)
    elif 'InternVL2_5-8B' in model_args.model_name_or_path:
        # device_map = split_model('InternVL2_5-8B')
        encoder = AutoModel.from_pretrained(model_args.model_name_or_path,
                                            device_map=device_map,
                                            dtype=torch_type,
                                            trust_remote_code=True,
                                            use_flash_attn=True,
                                            low_cpu_mem_usage=True)
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path,
                                                  trust_remote_code=True)
    else:
        encoder = LlavaNextForConditionalGeneration.from_pretrained(model_args.model_name_or_path,
                                                                    device_map=device_map,
                                                                    dtype=torch_type)
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

    if training_args.task_type == 'cir':
        dataset = ComposedTextImageRetrievalDataset(data_args.dataset_name, processor, data_args.dataset_split,
                                                    search_args.query_type)
        val_dataset = ComposedTextImageRetrievalDataset(data_args.dataset_name, processor, 'val',
                                                    search_args.query_type)
    elif training_args.task_type == 'tbpr':
        dataset = TextPersonRetrievalDataset(data_args.dataset_name, processor, data_args.dataset_split, 'single')
        if data_args.dataset_name == 'RSTPReid':
            val_dataset = TextPersonRetrievalDataset(data_args.dataset_name, processor, 'val', 'full')
        else:
            val_dataset = TextPersonRetrievalDataset(data_args.dataset_name, processor, 'train', 'full')
    else:
        if search_args.query_type == 'text':
            dataset = CrossModalRetrievalDataset(data_args.dataset_name, processor, 'test', 'full')
            val_dataset = CrossModalRetrievalDataset(data_args.dataset_name, processor, 'val', 'full')
        else:
            dataset = CrossModalRetrievalDataset(data_args.dataset_name, processor, 'test', 'single')
            val_dataset = CrossModalRetrievalDataset(data_args.dataset_name, processor, 'val', 'single')

    val_sampler = Data.DistributedSampler(val_dataset, num_replicas=world_size, shuffle=True, rank=rank)
    val_dataloader = Data.DataLoader(dataset=val_dataset, sampler=val_sampler,
                                     batch_size=data_args.per_device_batch_size,
                                     shuffle=False)
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
    val_lookup_indices = []

    model.eval()

    dense_run = {}
    sparse_run = {}
    val_dense_run = {}
    val_sparse_run = {}
    fusion_run = [{}] * 9
    val_fusion_run_1 = {}
    val_fusion_run_2 = {}
    val_fusion_run_3 = {}
    val_fusion_run_4 = {}
    val_fusion_run_5 = {}

    dense_retriever_indices = []
    sparse_retriever_indices = []
    val_dense_retriever_indices = []
    val_sparse_retriever_indices = []

    '''

    if search_args.val_passage_reps is not None:
        val_dense_retriever_indices = [search_args.val_passage_reps]

    if search_args.val_sparse_index is not None:
        val_sparse_retriever_indices = [search_args.val_sparse_index]

    if dist.get_rank() == 0:
        print(max(len(val_dense_retriever_indices), len(val_sparse_retriever_indices)))
        print(val_dense_retriever_indices)
        print(val_sparse_retriever_indices)
    for i in range(max(len(val_dense_retriever_indices), len(val_sparse_retriever_indices))):

        val_dense_retriever = None
        val_sparse_retriever = None

        if val_dense_retriever_indices:
            index_files = glob.glob(os.path.join(val_dense_retriever_indices[i], 'corpus*.pkl'))
            if dist.get_rank() == 0:
                print(f'Pattern match found {len(index_files)} files; loading them into dense index.')

            p_reps_0, p_lookup_0 = pickle_load(index_files[0])
            print(p_reps_0.shape)
            val_dense_retriever = FaissFlatSearcher(p_reps_0)
            # 经DeepSeek老师讲解，他说FaissFlatSearcher初始化时仅分配了内存结构，未添加任何数据。所以这里再重新加一下，
            # 这也和源代码中重复add了p_reps_0一致，希望D老师没骗我吧
            # dense_retriever.add(p_reps_0)

            # 在源代码里，并没有将所有数据都转移到某个GPU上面保存，而是各自保存，这样的话corpus会有多个编号，因此会有下面这一段处理多个corpus的代码，
            # 但是我们这里是先集中后保存，这样就只有一个文件，所以就先注释掉了
            # 经过修改，现在是每个gpu在encode的时候处理各自数据并各自保存一个文件，所以现在应当按照原来的方式处理
            shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
            if len(index_files) > 1:
                shards = tqdm(shards, desc='Loading shards into index', total=len(index_files))
            val_look_up = []
            for p_reps, p_lookup in shards:
                val_dense_retriever.add(p_reps)
                val_look_up += p_lookup
            if dist.get_rank() == 0:
                print(len(val_look_up))
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
                        val_dense_retriever.index = faiss.index_cpu_to_gpu(res, 0, val_dense_retriever.index, co)
                    else:
                        co = faiss.GpuMultipleClonerOptions()
                        co.shard = True
                        co.useFloat16 = True
                        val_dense_retriever.index = faiss.index_cpu_to_all_gpus(val_dense_retriever.index, co,
                                                                                ngpu=num_gpus)

        if sparse_retriever_indices:
            val_sparse_retriever = LuceneImpactSearcher(os.path.join(val_sparse_retriever_indices[i], 'index'), None)
            analyzer = JWhiteSpaceAnalyzer()
            val_sparse_retriever.set_analyzer(analyzer)

        with torch.no_grad(), torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
            for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(val_dataloader),
                                                                         total=len(val_dataloader)):
                if search_args.query_type == 'text':
                    val_lookup_indices.extend(text_ids)
                else:
                    val_lookup_indices.extend(img_ids)
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
                        if model_args.eol_type == 'metaeol':
                            query_logits = query_logits.reshape(-1, len(task_text_prompts), query_logits.shape[1]).mean(
                                1)
                            query_dense_reps = query_dense_reps.reshape(-1, len(task_text_prompts),
                                                                        query_dense_reps.shape[1]).mean(1)
                        elif 'disassembleeol_concrete' in model_args.eol_type:
                            disassemble_logits = query_logits[data_args.per_device_batch_size:]
                            query_logits = query_logits[:data_args.per_device_batch_size]
                        elif 'disassembleeol' in model_args.eol_type:
                            disassemble_logits = query_logits
                    else:
                        if 'InternVL2_5-8B' in model_args.model_name_or_path:
                            prompt = processor.apply_chat_template(
                                img_prompt_intern_vl_v2_5, tokenize=False, add_generation_prompt=True
                            )
                            imgs = [load_image(path, max_num=12).to(torch.bfloat16).cuda() for path in imgs_path]
                            query_logits, query_dense_reps = model.encode_data(imgs, 'image', processor, device,
                                                                               model_args,
                                                                               data_args)
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

                if search_args.query_type == 'text':
                    batch_ids = text_ids
                else:
                    batch_ids = img_ids
                # print(batch_ids)
                if val_dense_retriever is not None:
                    if isinstance(query_dense_reps, list):
                        for qid, reps in zip(batch_ids, query_dense_reps):
                            reps = torch.stack(reps, dim=0)
                            dense_scores, dense_rankings = search_queries(val_dense_retriever,
                                                                          reps.cpu().detach().float().numpy(),
                                                                          val_look_up, search_args)
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
                        query_dense_reps = F.normalize(query_dense_reps, dim=-1)
                        if model_args.eol_type == 'all_disassembleeol' or model_args.eol_type == 'all_disassembleeol_origin_text' or model_args.eol_type == 'all_disassembleeol_concrete' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text':
                            if model_args.calculate_type == 'concat':
                                if data_args.prompt_type == 'prompt_5':
                                    prompt_length = 5
                                elif data_args.prompt_type == 'prompt_3':
                                    prompt_length = 3
                                elif data_args.prompt_type == 'prompt_7':
                                    prompt_length = 7
                                else:
                                    prompt_length = 5
                            else:
                                prompt_length = 5
                            query_dense_reps = query_dense_reps.reshape(-1, prompt_length,
                                                                        query_dense_reps.shape[1]).mean(1)
                        query_dense_reps = query_dense_reps.cpu().detach().float().numpy()
                        dense_scores, dense_rankings = search_queries(val_dense_retriever, query_dense_reps,
                                                                      val_look_up,
                                                                      search_args)
                        val_dense_run.update(
                            get_run_dict(batch_ids, dense_scores, dense_rankings, search_args.remove_query))
                if val_sparse_retriever is not None:
                    batch_topics = []
                    if 'disassembleeol' in model_args.eol_type:
                        if search_args.query_type == 'text':
                            if data_args.sparse_type == 'fusion':
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
                                        tokens, values = get_text_valid_disassemble_tokens_values_fusion(text,
                                                                                                         processor.tokenizer,
                                                                                                         disassemble_logit,
                                                                                                         vocab_dict,
                                                                                                         data_args,
                                                                                                         filtered_ids,
                                                                                                         'guess', logit,
                                                                                                         model_args)
                                    else:
                                        tokens, values = get_text_valid_disassemble_tokens_values_fusion(text,
                                                                                                         processor.tokenizer,
                                                                                                         disassemble_logit,
                                                                                                         vocab_dict,
                                                                                                         data_args,
                                                                                                         filtered_ids,
                                                                                                         'guess', None,
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
                                    if model_args.eol_type == 'disassembleeol_concrete' or model_args.eol_type == 'disassembleeol_concrete_origin_text' or model_args.eol_type == 'all_disassembleeol_concrete' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text':
                                        tokens, values = get_text_valid_disassemble_tokens_values_fusion(text,
                                                                                                         processor.tokenizer,
                                                                                                         disassemble_logit,
                                                                                                         vocab_dict,
                                                                                                         data_args,
                                                                                                         filtered_ids,
                                                                                                         'origin_text',
                                                                                                         logit,
                                                                                                         model_args)
                                    else:
                                        tokens, values = get_text_valid_disassemble_tokens_values_fusion(text,
                                                                                                         processor.tokenizer,
                                                                                                         disassemble_logit,
                                                                                                         vocab_dict,
                                                                                                         data_args,
                                                                                                         filtered_ids,
                                                                                                         'origin_text',
                                                                                                         None,
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
                                    query = ""
                                    for token, v in vector.items():
                                        query += (' ' + token) * v
                                    batch_topics.append(query.strip())
                                sparse_scores, sparse_rankings = sparse_search(val_sparse_retriever, batch_topics,
                                                                               batch_ids,
                                                                               search_args)
                                val_sparse_run.update(
                                    get_run_dict(batch_ids, sparse_scores, sparse_rankings, search_args.remove_query))
                            else:
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
                                        tokens, values = get_text_valid_disassemble_tokens_values(text,
                                                                                                  processor.tokenizer,
                                                                                                  disassemble_logit,
                                                                                                  vocab_dict,
                                                                                                  data_args,
                                                                                                  filtered_ids, logit,
                                                                                                  model_args)
                                    else:
                                        tokens, values = get_text_valid_disassemble_tokens_values(text,
                                                                                                  processor.tokenizer,
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
                                sparse_scores, sparse_rankings = sparse_search(val_sparse_retriever, batch_topics,
                                                                               batch_ids,
                                                                               search_args)
                                val_sparse_run.update(
                                    get_run_dict(batch_ids, sparse_scores, sparse_rankings, search_args.remove_query))
                        else:
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
                            sparse_scores, sparse_rankings = sparse_search(val_sparse_retriever, batch_topics,
                                                                           batch_ids,
                                                                           search_args)
                            val_sparse_run.update(
                                get_run_dict(batch_ids, sparse_scores, sparse_rankings, search_args.remove_query))

                    else:
                        if search_args.query_type == 'text':
                            if data_args.sparse_type == 'fusion':
                                for _, logits, text in zip(batch_ids, query_logits, texts):
                                    vector = dict()
                                    tokens, values = get_text_valid_tokens_values_fusion(text, processor.tokenizer,
                                                                                         logits,
                                                                                         vocab_dict,
                                                                                         data_args,
                                                                                         filtered_ids, 'guess')
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

                                    tokens, values = get_text_valid_tokens_values_fusion(text, processor.tokenizer,
                                                                                         logits,
                                                                                         vocab_dict,
                                                                                         data_args,
                                                                                         filtered_ids, 'origin_text')
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
                                sparse_scores, sparse_rankings = sparse_search(val_sparse_retriever, batch_topics,
                                                                               batch_ids,
                                                                               search_args)
                                val_sparse_run.update(
                                    get_run_dict(batch_ids, sparse_scores, sparse_rankings, search_args.remove_query))
                            else:
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
                                sparse_scores, sparse_rankings = sparse_search(val_sparse_retriever, batch_topics,
                                                                               batch_ids,
                                                                               search_args)
                                val_sparse_run.update(
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
                            sparse_scores, sparse_rankings = sparse_search(val_sparse_retriever, batch_topics,
                                                                           batch_ids,
                                                                           search_args)
                            val_sparse_run.update(
                                get_run_dict(batch_ids, sparse_scores, sparse_rankings, search_args.remove_query))

                if model_args.eol_type == 'metaeol':
                    del query_dense_reps
                    del query_logits

        if val_dense_retriever:
            del val_dense_retriever
            gc.collect()
            torch.cuda.empty_cache()
        if val_sparse_retriever:
            del val_sparse_retriever
            gc.collect()
            torch.cuda.empty_cache()

    val_fusion_run_1.update(
        fuse(
            runs=[dense_run, sparse_run],
            weights=[0.5, 0.5]
        )
    )
    val_fusion_run_2.update(
        fuse(
            runs=[dense_run, sparse_run],
            weights=[0.6, 0.4]
        )
    )
    val_fusion_run_3.update(
        fuse(
            runs=[dense_run, sparse_run],
            weights=[0.7, 0.3]
        )
    )
    val_fusion_run_4.update(
        fuse(
            runs=[dense_run, sparse_run],
            weights=[0.8, 0.2]
        )
    )
    val_fusion_run_5.update(
        fuse(
            runs=[dense_run, sparse_run],
            weights=[0.9, 0.1]
        )
    )
    max_val_fusion_metric = 0
    best_weight = 0.5
    val_metric = RecallMetrics(val_dataset, val_dense_run, val_sparse_run, val_fusion_run_1, val_look_up,
                               val_lookup_indices, search_args)
    val_metric.sort_and_count()
    val_metric.all_gather_object()

    fusion_recalls = {k: sum(val_metric.fusion_recall_lists[k]) for k in val_metric.recall_k_setting_list}
    if (fusion_recalls[1] + fusion_recalls[5] + fusion_recalls[10]) / 3 > max_val_fusion_metric:
        max_val_fusion_metric = (fusion_recalls[1] + fusion_recalls[5] + fusion_recalls[10]) / 3
        best_weight = 0.5

    val_metric = RecallMetrics(val_dataset, val_dense_run, val_sparse_run, val_fusion_run_2, val_look_up,
                               val_lookup_indices, search_args)
    val_metric.sort_and_count()
    val_metric.all_gather_object()

    fusion_recalls = {k: sum(val_metric.fusion_recall_lists[k]) for k in val_metric.recall_k_setting_list}
    if (fusion_recalls[1] + fusion_recalls[5] + fusion_recalls[10]) / 3 > max_val_fusion_metric:
        max_val_fusion_metric = (fusion_recalls[1] + fusion_recalls[5] + fusion_recalls[10]) / 3
        best_weight = 0.6

    val_metric = RecallMetrics(val_dataset, val_dense_run, val_sparse_run, val_fusion_run_3, val_look_up,
                               val_lookup_indices, search_args)
    val_metric.sort_and_count()
    val_metric.all_gather_object()

    fusion_recalls = {k: sum(val_metric.fusion_recall_lists[k]) for k in val_metric.recall_k_setting_list}
    if (fusion_recalls[1] + fusion_recalls[5] + fusion_recalls[10]) / 3 > max_val_fusion_metric:
        max_val_fusion_metric = (fusion_recalls[1] + fusion_recalls[5] + fusion_recalls[10]) / 3
        best_weight = 0.7

    val_metric = RecallMetrics(val_dataset, val_dense_run, val_sparse_run, val_fusion_run_4, val_look_up,
                               val_lookup_indices, search_args)
    val_metric.sort_and_count()
    val_metric.all_gather_object()

    fusion_recalls = {k: sum(val_metric.fusion_recall_lists[k]) for k in val_metric.recall_k_setting_list}
    if (fusion_recalls[1] + fusion_recalls[5] + fusion_recalls[10]) / 3 > max_val_fusion_metric:
        max_val_fusion_metric = (fusion_recalls[1] + fusion_recalls[5] + fusion_recalls[10]) / 3
        best_weight = 0.8

    val_metric = RecallMetrics(val_dataset, val_dense_run, val_sparse_run, val_fusion_run_5, val_look_up,
                               val_lookup_indices, search_args)
    val_metric.sort_and_count()
    val_metric.all_gather_object()

    fusion_recalls = {k: sum(val_metric.fusion_recall_lists[k]) for k in val_metric.recall_k_setting_list}
    if (fusion_recalls[1] + fusion_recalls[5] + fusion_recalls[10]) / 3 > max_val_fusion_metric:
        best_weight = 0.9
        
    '''

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

        if dense_retriever_indices:
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

        if sparse_retriever_indices:
            sparse_retriever = LuceneImpactSearcher(os.path.join(sparse_retriever_indices[i], 'index'), None)
            analyzer = JWhiteSpaceAnalyzer()
            sparse_retriever.set_analyzer(analyzer)

        if training_args.task_type == 'cir':
            with torch.no_grad(), torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
                for batch_idx, (texts, imgs_path, target_path, text_ids, img_ids, composed_ids, dress_type) in tqdm(
                        enumerate(test_dataloader),
                        total=len(test_dataloader)):
                    '''
                    if dist.get_rank() == 0:
                        print(composed_ids)
                    '''
                    lookup_indices.extend(composed_ids)
                    if model_args.calculate_type == 'separate':
                        raw_images = [Image.open(path).convert('RGB') for path in imgs_path]
                        query_logits, query_dense_reps = model.encode_data_for_cir(texts, raw_images, dress_type,
                                                                                   'composed', processor, device,
                                                                                   model_args,
                                                                                   data_args)
                    else:
                        raw_images = [Image.open(path).convert('RGB') for path in imgs_path]
                        query_logits, query_dense_reps = model.encode_data_concat_for_cir(texts, raw_images, dress_type,
                                                                                          'composed', processor, device,
                                                                                          model_args, data_args)
                        if 'disassembleeol_concrete' in model_args.eol_type:
                            disassemble_logits = query_logits[data_args.per_device_batch_size:]
                            query_logits = query_logits[:data_args.per_device_batch_size]
                        elif 'disassembleeol' in model_args.eol_type:
                            disassemble_logits = query_logits
                    batch_ids = composed_ids
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
                            query_dense_reps = F.normalize(query_dense_reps, dim=-1)
                            if model_args.eol_type == 'all_disassembleeol' or model_args.eol_type == 'all_disassembleeol_origin_text' or model_args.eol_type == 'all_disassembleeol_concrete' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text':
                                prompt_length = 5
                                query_dense_reps = query_dense_reps.reshape(-1, prompt_length,
                                                                            query_dense_reps.shape[1]).mean(1)
                            query_dense_reps = query_dense_reps.cpu().detach().float().numpy()
                            dense_scores, dense_rankings = search_queries(dense_retriever, query_dense_reps, look_up,
                                                                          search_args)
                            dense_run.update(
                                get_run_dict(batch_ids, dense_scores, dense_rankings, search_args.remove_query))
                    if sparse_retriever is not None:
                        batch_topics = []
                        if 'disassembleeol' in model_args.eol_type:
                            if data_args.sparse_type == 'fusion':
                                for composed_indice in range(len(batch_ids)):
                                    id = batch_ids[composed_indice]
                                    if model_args.eol_type == 'disassembleeol_concrete' or model_args.eol_type == 'disassembleeol_concrete_origin_text' or model_args.eol_type == 'all_disassembleeol_concrete' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text':
                                        logit = query_logits[composed_indice]
                                    text = texts[composed_indice]
                                    length = 5
                                    disassemble_logit = disassemble_logits[
                                                        composed_indice * length:(composed_indice + 1) * length]
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
                                            vector[token] //= 5
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
                                    query = ""
                                    for token, v in vector.items():
                                        query += (' ' + token) * v
                                    batch_topics.append(query.strip())
                                sparse_scores, sparse_rankings = sparse_search(sparse_retriever, batch_topics,
                                                                               batch_ids,
                                                                               search_args)
                                sparse_run.update(
                                    get_run_dict(batch_ids, sparse_scores, sparse_rankings,
                                                 search_args.remove_query))
                            else:
                                for composed_indice in range(len(batch_ids)):
                                    id = batch_ids[composed_indice]
                                    if model_args.eol_type == 'disassembleeol_concrete' or model_args.eol_type == 'disassembleeol_concrete_origin_text' or model_args.eol_type == 'all_disassembleeol_concrete' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text':
                                        logit = query_logits[composed_indice]
                                    text = texts[composed_indice]
                                    length = 5
                                    disassemble_logit = disassemble_logits[
                                                        composed_indice * length:(composed_indice + 1) * length]
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
                                            vector[token] //= 5
                                    query = ""
                                    for token, v in vector.items():
                                        query += (' ' + token) * v
                                    batch_topics.append(query.strip())
                                sparse_scores, sparse_rankings = sparse_search(sparse_retriever, batch_topics,
                                                                               batch_ids,
                                                                               search_args)
                                sparse_run.update(
                                    get_run_dict(batch_ids, sparse_scores, sparse_rankings,
                                                 search_args.remove_query))

                        else:
                            if data_args.sparse_type == 'fusion':
                                for _, logits, text in zip(batch_ids, query_logits, texts):
                                    vector = dict()
                                    tokens, values = get_text_valid_tokens_values_fusion(text, processor.tokenizer,
                                                                                         logits,
                                                                                         vocab_dict,
                                                                                         data_args,
                                                                                         filtered_ids, 'guess')
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

                                    tokens, values = get_text_valid_tokens_values_fusion(text, processor.tokenizer,
                                                                                         logits,
                                                                                         vocab_dict,
                                                                                         data_args,
                                                                                         filtered_ids,
                                                                                         'origin_text')
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
                                    get_run_dict(batch_ids, sparse_scores, sparse_rankings,
                                                 search_args.remove_query))
                            else:
                                for _, logits, text in zip(batch_ids, query_logits, texts):
                                    vector = dict()
                                    if model_args.use_output_embedding_cluster:
                                        if 'InternVL2_5-8B' in model_args.model_name_or_path:
                                            tokens, values = get_text_valid_tokens_values_with_cluster(text,
                                                                                                       processor,
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
                                    get_run_dict(batch_ids, sparse_scores, sparse_rankings,
                                                 search_args.remove_query))

        elif training_args.task_type == 'tbpr':
            with torch.no_grad(), torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
                for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(test_dataloader),
                                                                             total=len(test_dataloader)):
                    lookup_indices.extend(text_ids)
                    if model_args.calculate_type == 'separate':
                        query_logits, query_dense_reps = model.encode_data_for_tbpr(texts, 'text', processor, device,
                                                                                    model_args,
                                                                                    data_args)
                    else:
                        if 'llava-hf-llava-v1.6-mistral-7b-hf' in model_args.model_name_or_path:
                            if data_args.tbpr_type == 'origin_type':
                                prompt_template = llava_mistral_template_image_prefix
                                if 'concrete' in model_args.eol_type or 'all' not in model_args.eol_type:
                                    prompt_template += llava_mistral_template_content_element.format(
                                        img_prompt_for_concat)
                                for llava_mistral_retrieval_disassemble_image_prompt in retrieval_disassemble_image_origin_prompts_person_retrieval_for_concat:
                                    content_element = llava_mistral_template_content_element.format(
                                        llava_mistral_retrieval_disassemble_image_prompt)
                                    prompt_template += content_element
                            elif data_args.tbpr_type == 'type':
                                prompt_template = llava_mistral_template_image_prefix
                                if 'concrete' in model_args.eol_type or 'all' not in model_args.eol_type:
                                    prompt_template += llava_mistral_template_content_element.format(
                                        person_retrieval_img_prompt_for_concat)
                                for llava_mistral_retrieval_disassemble_image_prompt in retrieval_disassemble_image_prompts_person_retrieval_for_concat:
                                    content_element = llava_mistral_template_content_element.format(
                                        llava_mistral_retrieval_disassemble_image_prompt)
                                    prompt_template += content_element
                            elif data_args.tbpr_type == 'type_1':
                                prompt_template = llava_mistral_template_image_prefix
                                if 'concrete' in model_args.eol_type or 'all' not in model_args.eol_type:
                                    prompt_template += llava_mistral_template_content_element.format(
                                        person_retrieval_img_prompt_for_concat_1)
                                for llava_mistral_retrieval_disassemble_image_prompt in retrieval_disassemble_image_prompts_person_retrieval_for_concat_1:
                                    content_element = llava_mistral_template_content_element.format(
                                        llava_mistral_retrieval_disassemble_image_prompt)
                                    prompt_template += content_element
                            elif data_args.tbpr_type == 'type_2':
                                prompt_template = llava_mistral_template_image_prefix
                                if 'concrete' in model_args.eol_type or 'all' not in model_args.eol_type:
                                    prompt_template += llava_mistral_template_content_element.format(
                                        person_retrieval_img_prompt_for_concat_2)
                                for llava_mistral_retrieval_disassemble_image_prompt in retrieval_disassemble_image_prompts_person_retrieval_for_concat_1:
                                    content_element = llava_mistral_template_content_element.format(
                                        llava_mistral_retrieval_disassemble_image_prompt)
                                    prompt_template += content_element
                            else:
                                prompt_template = llava_mistral_template_image_prefix
                                if 'concrete' in model_args.eol_type or 'all' not in model_args.eol_type:
                                    prompt_template += llava_mistral_template_content_element.format(
                                        img_prompt_for_concat)
                                for llava_mistral_retrieval_disassemble_image_prompt in retrieval_disassemble_image_origin_prompts_person_retrieval_for_concat:
                                    content_element = llava_mistral_template_content_element.format(
                                        llava_mistral_retrieval_disassemble_image_prompt)
                                    prompt_template += content_element
                        else:
                            if data_args.tbpr_type == 'origin_type':
                                prompt_template = llama3_template_image_prefix
                                if 'concrete' in model_args.eol_type or 'all' not in model_args.eol_type:
                                    prompt_template += llama3_template_content_element.format(img_prompt_for_concat)
                                for llama3_retrieval_disassemble_image_prompt in retrieval_disassemble_image_origin_prompts_person_retrieval_for_concat:
                                    content_element = llama3_template_content_element.format(
                                        llama3_retrieval_disassemble_image_prompt)
                                    prompt_template += content_element
                            elif data_args.tbpr_type == 'type':
                                prompt_template = llama3_template_image_prefix
                                if 'concrete' in model_args.eol_type or 'all' not in model_args.eol_type:
                                    prompt_template += llama3_template_content_element.format(
                                        person_retrieval_img_prompt_for_concat)
                                for llama3_retrieval_disassemble_image_prompt in retrieval_disassemble_image_prompts_person_retrieval_for_concat:
                                    content_element = llama3_template_content_element.format(
                                        llama3_retrieval_disassemble_image_prompt)
                                    prompt_template += content_element
                            elif data_args.tbpr_type == 'type_1':
                                prompt_template = llama3_template_image_prefix
                                if 'concrete' in model_args.eol_type or 'all' not in model_args.eol_type:
                                    prompt_template += llama3_template_content_element.format(
                                        person_retrieval_img_prompt_for_concat_1)
                                for llama3_retrieval_disassemble_image_prompt in retrieval_disassemble_image_prompts_person_retrieval_for_concat_1:
                                    content_element = llama3_template_content_element.format(
                                        llama3_retrieval_disassemble_image_prompt)
                                    prompt_template += content_element
                            elif data_args.tbpr_type == 'type_2':
                                prompt_template = llama3_template_image_prefix
                                if 'concrete' in model_args.eol_type or 'all' not in model_args.eol_type:
                                    prompt_template += llama3_template_content_element.format(
                                        person_retrieval_img_prompt_for_concat_2)
                                for llama3_retrieval_disassemble_image_prompt in retrieval_disassemble_image_prompts_person_retrieval_for_concat_1:
                                    content_element = llama3_template_content_element.format(
                                        llama3_retrieval_disassemble_image_prompt)
                                    prompt_template += content_element
                            else:
                                prompt_template = llama3_template_image_prefix
                                if 'concrete' in model_args.eol_type or 'all' not in model_args.eol_type:
                                    prompt_template += llama3_template_content_element.format(img_prompt_for_concat)
                                for llama3_retrieval_disassemble_image_prompt in retrieval_disassemble_image_origin_prompts_person_retrieval_for_concat:
                                    content_element = llama3_template_content_element.format(
                                        llama3_retrieval_disassemble_image_prompt)
                                    prompt_template += content_element

                        query_logits, query_dense_reps = model.encode_data_concat_for_tbpr(texts, 'text', processor,
                                                                                           device,
                                                                                           model_args, data_args)
                        if 'disassembleeol_concrete' in model_args.eol_type:
                            disassemble_logits = query_logits[data_args.per_device_batch_size:]
                            query_logits = query_logits[:data_args.per_device_batch_size]
                        elif 'disassembleeol' in model_args.eol_type:
                            disassemble_logits = query_logits

                    batch_ids = text_ids

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
                            query_dense_reps = F.normalize(query_dense_reps, dim=-1)
                            if model_args.eol_type == 'all_disassembleeol' or model_args.eol_type == 'all_disassembleeol_origin_text' or model_args.eol_type == 'all_disassembleeol_concrete' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text':
                                prompt_length = 5
                                query_dense_reps = query_dense_reps.reshape(-1, prompt_length,
                                                                            query_dense_reps.shape[1]).mean(1)
                            query_dense_reps = query_dense_reps.cpu().detach().float().numpy()
                            dense_scores, dense_rankings = search_queries(dense_retriever, query_dense_reps, look_up,
                                                                          search_args)
                            dense_run.update(
                                get_run_dict(batch_ids, dense_scores, dense_rankings, search_args.remove_query))

                    if sparse_retriever is not None:
                        batch_topics = []
                        if 'disassembleeol' in model_args.eol_type:
                            if data_args.sparse_type == 'fusion':
                                for text_indice in range(len(batch_ids)):
                                    id = batch_ids[text_indice]
                                    if model_args.eol_type == 'disassembleeol_concrete' or model_args.eol_type == 'disassembleeol_concrete_origin_text' or model_args.eol_type == 'all_disassembleeol_concrete' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text':
                                        logit = query_logits[text_indice]
                                    text = texts[text_indice]
                                    length = 5
                                    disassemble_logit = disassemble_logits[
                                                        text_indice * length:(text_indice + 1) * length]
                                    vector = dict()
                                    if model_args.eol_type == 'disassembleeol_concrete' or model_args.eol_type == 'disassembleeol_concrete_origin_text' or model_args.eol_type == 'all_disassembleeol_concrete' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text':
                                        tokens, values = get_text_valid_disassemble_tokens_values_fusion(text,
                                                                                                         processor.tokenizer,
                                                                                                         disassemble_logit,
                                                                                                         vocab_dict,
                                                                                                         data_args,
                                                                                                         filtered_ids,
                                                                                                         'guess',
                                                                                                         logit,
                                                                                                         model_args)
                                    else:
                                        tokens, values = get_text_valid_disassemble_tokens_values_fusion(text,
                                                                                                         processor.tokenizer,
                                                                                                         disassemble_logit,
                                                                                                         vocab_dict,
                                                                                                         data_args,
                                                                                                         filtered_ids,
                                                                                                         'guess',
                                                                                                         None,
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
                                            vector[token] //= 5
                                    if model_args.eol_type == 'disassembleeol_concrete' or model_args.eol_type == 'disassembleeol_concrete_origin_text' or model_args.eol_type == 'all_disassembleeol_concrete' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text':
                                        tokens, values = get_text_valid_disassemble_tokens_values_fusion(text,
                                                                                                         processor.tokenizer,
                                                                                                         disassemble_logit,
                                                                                                         vocab_dict,
                                                                                                         data_args,
                                                                                                         filtered_ids,
                                                                                                         'origin_text',
                                                                                                         logit,
                                                                                                         model_args)
                                    else:
                                        tokens, values = get_text_valid_disassemble_tokens_values_fusion(text,
                                                                                                         processor.tokenizer,
                                                                                                         disassemble_logit,
                                                                                                         vocab_dict,
                                                                                                         data_args,
                                                                                                         filtered_ids,
                                                                                                         'origin_text',
                                                                                                         None,
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
                                    query = ""
                                    for token, v in vector.items():
                                        query += (' ' + token) * v
                                    batch_topics.append(query.strip())
                                sparse_scores, sparse_rankings = sparse_search(sparse_retriever, batch_topics,
                                                                               batch_ids,
                                                                               search_args)
                                sparse_run.update(
                                    get_run_dict(batch_ids, sparse_scores, sparse_rankings,
                                                 search_args.remove_query))
                            else:
                                for text_indice in range(len(batch_ids)):
                                    id = batch_ids[text_indice]
                                    if model_args.eol_type == 'disassembleeol_concrete' or model_args.eol_type == 'disassembleeol_concrete_origin_text' or model_args.eol_type == 'all_disassembleeol_concrete' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text':
                                        logit = query_logits[text_indice]
                                    text = texts[text_indice]
                                    length = 5
                                    disassemble_logit = disassemble_logits[
                                                        text_indice * length:(text_indice + 1) * length]
                                    vector = dict()
                                    if model_args.eol_type == 'disassembleeol_concrete' or model_args.eol_type == 'disassembleeol_concrete_origin_text' or model_args.eol_type == 'all_disassembleeol_concrete' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text':
                                        tokens, values = get_text_valid_disassemble_tokens_values(text,
                                                                                                  processor.tokenizer,
                                                                                                  disassemble_logit,
                                                                                                  vocab_dict,
                                                                                                  data_args,
                                                                                                  filtered_ids,
                                                                                                  logit,
                                                                                                  model_args)
                                    else:
                                        tokens, values = get_text_valid_disassemble_tokens_values(text,
                                                                                                  processor.tokenizer,
                                                                                                  disassemble_logit,
                                                                                                  vocab_dict,
                                                                                                  data_args,
                                                                                                  filtered_ids,
                                                                                                  None,
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
                                            vector[token] //= 5
                                    query = ""
                                    for token, v in vector.items():
                                        query += (' ' + token) * v
                                    batch_topics.append(query.strip())
                                sparse_scores, sparse_rankings = sparse_search(sparse_retriever, batch_topics,
                                                                               batch_ids,
                                                                               search_args)
                                sparse_run.update(
                                    get_run_dict(batch_ids, sparse_scores, sparse_rankings,
                                                 search_args.remove_query))

                        else:
                            if data_args.sparse_type == 'fusion':
                                for _, logits, text in zip(batch_ids, query_logits, texts):
                                    vector = dict()
                                    tokens, values = get_text_valid_tokens_values_fusion(text, processor.tokenizer,
                                                                                         logits,
                                                                                         vocab_dict,
                                                                                         data_args,
                                                                                         filtered_ids, 'guess')
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

                                    tokens, values = get_text_valid_tokens_values_fusion(text, processor.tokenizer,
                                                                                         logits,
                                                                                         vocab_dict,
                                                                                         data_args,
                                                                                         filtered_ids,
                                                                                         'origin_text')
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
                                    get_run_dict(batch_ids, sparse_scores, sparse_rankings,
                                                 search_args.remove_query))
                            else:
                                for _, logits, text in zip(batch_ids, query_logits, texts):
                                    vector = dict()
                                    if model_args.use_output_embedding_cluster:
                                        if 'InternVL2_5-8B' in model_args.model_name_or_path:
                                            tokens, values = get_text_valid_tokens_values_with_cluster(text,
                                                                                                       processor,
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
                                    get_run_dict(batch_ids, sparse_scores, sparse_rankings,
                                                 search_args.remove_query))
        else:
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
                        prompt = qwen2_5_img_prompt
                    elif 'Qwen3-VL-8B-Instruct' in model_args.model_name_or_path:
                        prompt = qwen3_img_prompt
                    elif 'InternVL2_5-8B' in model_args.model_name_or_path:
                        prompt = img_prompt_intern_vl_v2_5
                    elif 'llava-hf-llava-v1.6-mistral-7b-hf' in model_args.model_name_or_path:
                        prompt = mistral_img_prompt
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
                            query_logits, query_dense_reps = model.encode_data(texts, 'text', processor, device,
                                                                               model_args,
                                                                               data_args)
                            if model_args.eol_type == 'metaeol':
                                query_logits = query_logits.reshape(-1, len(task_text_prompts),
                                                                    query_logits.shape[1]).mean(
                                    1)
                                query_dense_reps = query_dense_reps.reshape(-1, len(task_text_prompts),
                                                                            query_dense_reps.shape[1]).mean(1)
                            elif 'disassembleeol_concrete' in model_args.eol_type:
                                disassemble_logits = query_logits[data_args.per_device_batch_size:]
                                query_logits = query_logits[:data_args.per_device_batch_size]
                            elif 'disassembleeol' in model_args.eol_type:
                                disassemble_logits = query_logits
                        else:
                            if 'InternVL2_5-8B' in model_args.model_name_or_path:
                                prompt = processor.apply_chat_template(
                                    img_prompt_intern_vl_v2_5, tokenize=False, add_generation_prompt=True
                                )
                                imgs = [load_image(path, max_num=12).to(torch.bfloat16).cuda() for path in imgs_path]
                                query_logits, query_dense_reps = model.encode_data(imgs, 'image', processor, device,
                                                                                   model_args,
                                                                                   data_args)
                            else:
                                if model_args.eol_type == 'prompteol' or model_args.eol_type == 'prompteol_same_length':
                                    if 'Qwen2.5-VL-7B-Instruct' in model_args.model_name_or_path or 'Qwen2.5-VL-3B-Instruct' in model_args.model_name_or_path:
                                        prompt = processor.apply_chat_template(
                                            img_prompt_qwen_v2_5, tokenize=False, add_generation_prompt=True
                                        )
                                    elif 'Qwen3-VL-8B-Instruct' in model_args.model_name_or_path:
                                        prompt = processor.apply_chat_template(
                                            img_prompt_qwen_v3, tokenize=False, add_generation_prompt=True
                                        )
                                    raw_images = [Image.open(path).convert('RGB') for path in imgs_path]
                                    img_inputs = processor(images=raw_images, text=[prompt] * len(imgs_path),
                                                           return_tensors="pt",
                                                           padding=True)
                                    '''
                                    if dist.get_rank() == 0:
                                        print([prompt] * len(imgs_path))
                                        print(img_inputs['input_ids'])
                                        print(img_inputs['input_ids'].shape)
                                        print(img_inputs['attention_mask'])
                                        print(img_inputs['attention_mask'].shape)
                                    '''
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
                                        query_logits, query_dense_reps = model.encode_data(imgs, 'image', processor,
                                                                                           device,
                                                                                           model_args, data_args)
                                    else:
                                        _, query_dense_reps = model.encode_data(imgs, 'image', processor, device,
                                                                                model_args, data_args)
                                    del imgs

                                    disassemble_raw_images = [raw_image for raw_image in raw_images for _ in
                                                              range(len(prompts) // 5)]
                                    '''
                                    disassemble_img_inputs = processor(images=disassemble_raw_images,
                                                                       text=prompts * len(imgs_path),
                                                                       return_tensors="pt",
                                                                       padding=True)
                                    '''
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
                                        disassemble_logits_sub, _ = model.encode_data(disassemble_imgs, 'image',
                                                                                      processor,
                                                                                      device, model_args,
                                                                                      data_args)

                                        for j in range(len(imgs_path)):
                                            # 这个j是为了控制要把第j个样本对应的数据存到对应索引下的列表中
                                            disassemble_logits[j].append(
                                                disassemble_logits_sub[
                                                j * len(prompts) // 5:(j + 1) * len(prompts) // 5])
                                    disassemble_logits = [item for disassemble_logit in disassemble_logits for item in
                                                          disassemble_logit]
                                    disassemble_logits = torch.cat(disassemble_logits, dim=0)
                                    '''
                                    disassemble_imgs = disassemble_img_inputs.to(device)
                                    disassemble_logits, _ = model.encode_data(disassemble_imgs, 'image', processor, device,
                                                                              model_args, data_args)
                                    '''

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
                                        disassemble_logits_sub, disassemble_reps_sub = model.encode_data(
                                            disassemble_imgs,
                                            'image', processor,
                                            device, model_args,
                                            data_args)

                                        for j in range(len(imgs_path)):
                                            # 这个j是为了控制要把第j个样本对应的数据存到对应索引下的列表中
                                            disassemble_logits[j].append(
                                                disassemble_logits_sub[
                                                j * len(prompts) // 5:(j + 1) * len(prompts) // 5])
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
                                    '''
                                    disassemble_img_inputs = processor(images=disassemble_raw_images,
                                                                       text=prompts * len(imgs_path),
                                                                       return_tensors="pt",
                                                                       padding=True)
                                    '''
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
                                        disassemble_logits_sub, disassemble_reps_sub = model.encode_data(
                                            disassemble_imgs,
                                            'image', processor,
                                            device, model_args,
                                            data_args)

                                        for j in range(len(imgs_path)):
                                            # 这个j是为了控制要把第j个样本对应的数据存到对应索引下的列表中
                                            disassemble_logits[j].append(
                                                disassemble_logits_sub[
                                                j * len(prompts) // 5:(j + 1) * len(prompts) // 5])
                                            disassemble_reps[j].append(
                                                disassemble_reps_sub[j * len(prompts) // 5:(j + 1) * len(prompts) // 5])
                                    disassemble_logits = [item for disassemble_logit in disassemble_logits for item in
                                                          disassemble_logit]
                                    disassemble_reps = [item for disassemble_rep in disassemble_reps for item in
                                                        disassemble_rep]
                                    disassemble_logits = torch.cat(disassemble_logits, dim=0)
                                    disassemble_reps = torch.cat(disassemble_reps, dim=0)
                                    query_dense_reps = disassemble_reps
                                    '''
                                    disassemble_imgs = disassemble_img_inputs.to(device)
                                    disassemble_logits, _ = model.encode_data(disassemble_imgs, 'image', processor, device,
                                                                              model_args, data_args)
                                    '''
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

                                        img_inputs = processor(images=raw_images,
                                                               text=prompts[start:end] * len(imgs_path),
                                                               return_tensors="pt",
                                                               padding=True)

                                        imgs = img_inputs.to(device)

                                        # 在metaeol模式下，reps应该是[batch_size * len(task_prompts) // 4, reps_dim]
                                        logits_sub, reps_sub = model.encode_data(imgs, 'image', processor, device,
                                                                                 model_args,
                                                                                 data_args)

                                        for j in range(len(imgs_path)):
                                            # 这个j是为了控制要把第j个样本对应的数据存到对应索引下的列表中
                                            logits[j].append(
                                                logits_sub[j * len(prompts) // 4:(j + 1) * len(prompts) // 4])
                                            reps[j].append(reps_sub[j * len(prompts) // 4:(j + 1) * len(prompts) // 4])

                                    logits = [item for logit in logits for item in logit]
                                    reps = [item for rep in reps for item in rep]

                                    logits = torch.cat(logits, dim=0)
                                    reps = torch.cat(reps, dim=0)

                                    query_logits = logits.reshape(-1, len(task_image_prompts), logits.shape[1]).mean(1)
                                    query_dense_reps = reps.reshape(-1, len(task_image_prompts), reps.shape[1]).mean(1)
                    else:
                        if 'llava-hf-llava-v1.6-mistral-7b-hf' in model_args.model_name_or_path:
                            if data_args.prompt_type == 'prompt_5':
                                prompt_template = llava_mistral_template_image_prefix
                                if 'concrete' in model_args.eol_type or 'all' not in model_args.eol_type:
                                    prompt_template += llava_mistral_template_content_element.format(
                                        img_prompt_for_concat)
                                for llava_mistral_retrieval_disassemble_image_prompt in retrieval_disassemble_image_prompts_for_concat:
                                    content_element = llava_mistral_template_content_element.format(
                                        llava_mistral_retrieval_disassemble_image_prompt)
                                    prompt_template += content_element
                            elif data_args.prompt_type == 'prompt_3':
                                prompt_template = llava_mistral_template_image_prefix
                                if 'concrete' in model_args.eol_type or 'all' not in model_args.eol_type:
                                    prompt_template += llava_mistral_template_content_element.format(
                                        img_prompt_for_concat)
                                for llava_mistral_retrieval_disassemble_image_prompt in retrieval_disassemble_image_prompts_3_for_concat:
                                    content_element = llava_mistral_template_content_element.format(
                                        llava_mistral_retrieval_disassemble_image_prompt)
                                    prompt_template += content_element
                            elif data_args.prompt_type == 'prompt_7':
                                prompt_template = llava_mistral_template_image_prefix
                                if 'concrete' in model_args.eol_type or 'all' not in model_args.eol_type:
                                    prompt_template += llava_mistral_template_content_element.format(
                                        img_prompt_for_concat)
                                for llava_mistral_retrieval_disassemble_image_prompt in retrieval_disassemble_image_prompts_7_for_concat:
                                    content_element = llava_mistral_template_content_element.format(
                                        llava_mistral_retrieval_disassemble_image_prompt)
                                    prompt_template += content_element
                            else:
                                pass

                        elif 'Qwen2.5-VL-7B-Instruct' in model_args.model_name_or_path:
                            if data_args.prompt_type == 'prompt_5':
                                prompt_template = qwen2_5_template_image_prefix
                                if 'concrete' in model_args.eol_type or 'all' not in model_args.eol_type:
                                    prompt_template += qwen2_5_template_content_element.format(
                                        img_prompt_for_concat)
                                for qwen2_5_retrieval_disassemble_image_prompt in retrieval_disassemble_image_prompts_for_concat:
                                    content_element = qwen2_5_template_content_element.format(
                                        qwen2_5_retrieval_disassemble_image_prompt)
                                    prompt_template += content_element
                            elif data_args.prompt_type == 'prompt_3':
                                prompt_template = qwen2_5_template_image_prefix
                                if 'concrete' in model_args.eol_type or 'all' not in model_args.eol_type:
                                    prompt_template += qwen2_5_template_content_element.format(
                                        img_prompt_for_concat)
                                for qwen2_5_retrieval_disassemble_image_prompt in retrieval_disassemble_image_prompts_3_for_concat:
                                    content_element = qwen2_5_template_content_element.format(
                                        qwen2_5_retrieval_disassemble_image_prompt)
                                    prompt_template += content_element
                            elif data_args.prompt_type == 'prompt_7':
                                prompt_template = qwen2_5_template_image_prefix
                                if 'concrete' in model_args.eol_type or 'all' not in model_args.eol_type:
                                    prompt_template += qwen2_5_template_content_element.format(
                                        img_prompt_for_concat)
                                for qwen2_5_retrieval_disassemble_image_prompt in retrieval_disassemble_image_prompts_7_for_concat:
                                    content_element = qwen2_5_template_content_element.format(
                                        qwen2_5_retrieval_disassemble_image_prompt)
                                    prompt_template += content_element
                        elif 'Qwen3-VL-8B-Instruct' in model_args.model_name_or_path:
                            if data_args.prompt_type == 'prompt_5':
                                prompt_template = qwen3_template_image_prefix
                                if 'concrete' in model_args.eol_type or 'all' not in model_args.eol_type:
                                    prompt_template += qwen3_template_content_element.format(
                                        img_prompt_for_concat)
                                for qwen3_retrieval_disassemble_image_prompt in retrieval_disassemble_image_prompts_for_concat:
                                    content_element = qwen3_template_content_element.format(
                                        qwen3_retrieval_disassemble_image_prompt)
                                    prompt_template += content_element
                            elif data_args.prompt_type == 'prompt_3':
                                prompt_template = qwen3_template_image_prefix
                                if 'concrete' in model_args.eol_type or 'all' not in model_args.eol_type:
                                    prompt_template += qwen3_template_content_element.format(
                                        img_prompt_for_concat)
                                for qwen3_retrieval_disassemble_image_prompt in retrieval_disassemble_image_prompts_3_for_concat:
                                    content_element = qwen3_template_content_element.format(
                                        qwen3_retrieval_disassemble_image_prompt)
                                    prompt_template += content_element
                            elif data_args.prompt_type == 'prompt_7':
                                prompt_template = qwen3_template_image_prefix
                                if 'concrete' in model_args.eol_type or 'all' not in model_args.eol_type:
                                    prompt_template += qwen3_template_content_element.format(
                                        img_prompt_for_concat)
                                for qwen3_retrieval_disassemble_image_prompt in retrieval_disassemble_image_prompts_7_for_concat:
                                    content_element = qwen3_template_content_element.format(
                                        qwen3_retrieval_disassemble_image_prompt)
                                    prompt_template += content_element
                            else:
                                pass
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

                    if search_args.query_type == 'text':
                        batch_ids = text_ids
                    else:
                        batch_ids = img_ids
                    # print(batch_ids)
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
                            query_dense_reps = F.normalize(query_dense_reps, dim=-1)
                            if model_args.eol_type == 'all_disassembleeol' or model_args.eol_type == 'all_disassembleeol_origin_text' or model_args.eol_type == 'all_disassembleeol_concrete' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text':
                                if model_args.calculate_type == 'concat':
                                    if data_args.prompt_type == 'prompt_5':
                                        prompt_length = 5
                                    elif data_args.prompt_type == 'prompt_3':
                                        prompt_length = 3
                                    elif data_args.prompt_type == 'prompt_7':
                                        prompt_length = 7
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
                        batch_topics = []
                        if 'disassembleeol' in model_args.eol_type:
                            if search_args.query_type == 'text':
                                if data_args.sparse_type == 'fusion':
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
                                            tokens, values = get_text_valid_disassemble_tokens_values_fusion(text,
                                                                                                             processor.tokenizer,
                                                                                                             disassemble_logit,
                                                                                                             vocab_dict,
                                                                                                             data_args,
                                                                                                             filtered_ids,
                                                                                                             'guess',
                                                                                                             logit,
                                                                                                             model_args)
                                        else:
                                            tokens, values = get_text_valid_disassemble_tokens_values_fusion(text,
                                                                                                             processor.tokenizer,
                                                                                                             disassemble_logit,
                                                                                                             vocab_dict,
                                                                                                             data_args,
                                                                                                             filtered_ids,
                                                                                                             'guess',
                                                                                                             None,
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
                                        if model_args.eol_type == 'disassembleeol_concrete' or model_args.eol_type == 'disassembleeol_concrete_origin_text' or model_args.eol_type == 'all_disassembleeol_concrete' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text':
                                            tokens, values = get_text_valid_disassemble_tokens_values_fusion(text,
                                                                                                             processor.tokenizer,
                                                                                                             disassemble_logit,
                                                                                                             vocab_dict,
                                                                                                             data_args,
                                                                                                             filtered_ids,
                                                                                                             'origin_text',
                                                                                                             logit,
                                                                                                             model_args)
                                        else:
                                            tokens, values = get_text_valid_disassemble_tokens_values_fusion(text,
                                                                                                             processor.tokenizer,
                                                                                                             disassemble_logit,
                                                                                                             vocab_dict,
                                                                                                             data_args,
                                                                                                             filtered_ids,
                                                                                                             'origin_text',
                                                                                                             None,
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
                                        query = ""
                                        for token, v in vector.items():
                                            query += (' ' + token) * v
                                        batch_topics.append(query.strip())
                                    sparse_scores, sparse_rankings = sparse_search(sparse_retriever, batch_topics,
                                                                                   batch_ids,
                                                                                   search_args)
                                    sparse_run.update(
                                        get_run_dict(batch_ids, sparse_scores, sparse_rankings,
                                                     search_args.remove_query))
                                else:
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
                                            tokens, values = get_text_valid_disassemble_tokens_values(text,
                                                                                                      processor.tokenizer,
                                                                                                      disassemble_logit,
                                                                                                      vocab_dict,
                                                                                                      data_args,
                                                                                                      filtered_ids,
                                                                                                      logit,
                                                                                                      model_args)
                                        else:
                                            tokens, values = get_text_valid_disassemble_tokens_values(text,
                                                                                                      processor.tokenizer,
                                                                                                      disassemble_logit,
                                                                                                      vocab_dict,
                                                                                                      data_args,
                                                                                                      filtered_ids,
                                                                                                      None,
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
                                    sparse_scores, sparse_rankings = sparse_search(sparse_retriever, batch_topics,
                                                                                   batch_ids,
                                                                                   search_args)
                                    sparse_run.update(
                                        get_run_dict(batch_ids, sparse_scores, sparse_rankings,
                                                     search_args.remove_query))
                            else:
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
                                sparse_scores, sparse_rankings = sparse_search(sparse_retriever, batch_topics,
                                                                               batch_ids,
                                                                               search_args)
                                sparse_run.update(
                                    get_run_dict(batch_ids, sparse_scores, sparse_rankings, search_args.remove_query))

                        else:
                            if search_args.query_type == 'text':
                                if data_args.sparse_type == 'fusion':
                                    for _, logits, text in zip(batch_ids, query_logits, texts):
                                        vector = dict()
                                        tokens, values = get_text_valid_tokens_values_fusion(text, processor.tokenizer,
                                                                                             logits,
                                                                                             vocab_dict,
                                                                                             data_args,
                                                                                             filtered_ids, 'guess')
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

                                        tokens, values = get_text_valid_tokens_values_fusion(text, processor.tokenizer,
                                                                                             logits,
                                                                                             vocab_dict,
                                                                                             data_args,
                                                                                             filtered_ids,
                                                                                             'origin_text')
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
                                        get_run_dict(batch_ids, sparse_scores, sparse_rankings,
                                                     search_args.remove_query))
                                else:
                                    for _, logits, text in zip(batch_ids, query_logits, texts):
                                        vector = dict()
                                        if model_args.use_output_embedding_cluster:
                                            if 'InternVL2_5-8B' in model_args.model_name_or_path:
                                                tokens, values = get_text_valid_tokens_values_with_cluster(text,
                                                                                                           processor,
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
                                        get_run_dict(batch_ids, sparse_scores, sparse_rankings,
                                                     search_args.remove_query))

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

                    if model_args.eol_type == 'metaeol':
                        del query_dense_reps
                        del query_logits

        if dense_retriever:
            del dense_retriever
            torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        # 获取当前设备
        device = torch.cuda.current_device()

        # 获取内存分配情况（字节）
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)

        # 获取设备总内存
        total = torch.cuda.get_device_properties(device).total_memory

        # 转换为 MB
        allocated_mb = allocated / (1024 ** 2)
        reserved_mb = reserved / (1024 ** 2)
        total_mb = total / (1024 ** 2)

        print(f"当前 GPU 内存使用情况:")
        print(f"已分配: {allocated_mb:.2f} MB")
        print(f"已保留: {reserved_mb:.2f} MB")
        print(f"总内存: {total_mb:.2f} MB")
        print(f"使用率: {allocated_mb / total_mb * 100:.1f}%")
    else:
        print("没有可用的 GPU 设备")

    max_val_fusion_metric = 0
    best_weight = 0.5

    if training_args.task_type == 'cir':
        choice_dataset = ComposedTextImageRetrievalDataset(data_args.dataset_name, processor, 'train', search_args.query_type)
    elif training_args.task_type == 'tbpr':
        choice_dataset = TextPersonRetrievalDataset(data_args.dataset_name, processor, 'train',
                                                           'full')
    else:
        choice_dataset = CrossModalRetrievalDataset(data_args.dataset_name, processor, 'train', 'full')

    if training_args.task_type == 'cir':
        ranker = Reranker(encoder, processor, data_args.dataset_name, search_args.query_type, dataset.text_dict, None,
                          dataset.img_dict, processor.tokenizer.get_vocab(), dataset)
    else:
        if data_args.dataset_name == 'coco':
            ranker = Reranker(encoder, processor, data_args.dataset_name, search_args.query_type, dataset.text_dict,
                              dataset.img2filepath, dataset.img_dict, processor.tokenizer.get_vocab(), choice_dataset)
        else:
            ranker = Reranker(encoder, processor, data_args.dataset_name, search_args.query_type, dataset.text_dict,
                              None, dataset.img_dict, processor.tokenizer.get_vocab(), choice_dataset)

    if data_args.is_filtered:
        filtered = "filter"
    else:
        filtered = "no_filter"

    if data_args.sparse_manual:
        manual = 'manual'
    else:
        manual = "no_manual"

    if model_args.use_output_embedding_cluster:
        cluster = f'cluster_{model_args.cluster_sum}'
    else:
        cluster = 'no_cluster'

    if data_args.sparse_value_mean:
        use_sparse_value_mean = 'mean'
    else:
        use_sparse_value_mean = 'no_mean'

    for i in range(9):
        fusion_run[i].update(
            fuse(
                runs=[dense_run, sparse_run],
                weights=[float((i+1)/10), 1-float((i+1)/10)]
            )
        )
        if training_args.task_type == 'tbpr':
            os.makedirs(
                f'search_results/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{search_args.query_type}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.tbpr_type}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{data_args.sparse_type}_rerank_{search_args.rerank_type}_{search_args.rerank_num}_{search_args.rerank_template}',
                exist_ok=True)

            output_path = os.path.join(
                f'search_results/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{search_args.query_type}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.tbpr_type}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{data_args.sparse_type}_rerank_{search_args.rerank_type}_{search_args.rerank_num}_{search_args.rerank_template}',
                f'0_{i + 1}_0_{10 - i - 1}.xlsx')
        else:
            os.makedirs(
                f'search_results/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{search_args.query_type}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{data_args.sparse_type}_rerank_{search_args.rerank_type}_{search_args.rerank_num}_{search_args.rerank_template}',
                exist_ok=True)

            output_path = os.path.join(
                f'search_results/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{search_args.query_type}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{data_args.sparse_type}_rerank_{search_args.rerank_type}_{search_args.rerank_num}_{search_args.rerank_template}',
                f'0_{i + 1}_0_{10 - i - 1}.xlsx')

        metric = RecallMetrics(dataset, dense_run, sparse_run, fusion_run[i], look_up, lookup_indices, search_args)
        metric.sort_and_count()

        metric.all_gather_object()
        if data_args.dataset_name != 'fashion-iq':
            fusion_recalls = {k: sum(metric.fusion_recall_lists[k]) for k in metric.recall_k_setting_list}
            if (fusion_recalls[1] + fusion_recalls[5] + fusion_recalls[10]) / 3 > max_val_fusion_metric:
                max_val_fusion_metric = (fusion_recalls[1] + fusion_recalls[5] + fusion_recalls[10]) / 3
                best_weight = float((i+1) / 10)
        else:
            fusion_recalls = {
                dress: {k: sum(metric.fusion_recall_lists[dress][k]) for k in metric.recall_k_setting_list} for dress in
                metric.fashion_iq_list}
            if (fusion_recalls['dress'][10] + fusion_recalls['dress'][50] + fusion_recalls['shirt'][10] +
                fusion_recalls['shirt'][50] + fusion_recalls['toptee'][10] + fusion_recalls['toptee'][50]) / 6 > \
                    max_val_fusion_metric:
                max_val_fusion_metric = (fusion_recalls['dress'][10] + fusion_recalls['dress'][50] +
                                         fusion_recalls['shirt'][10] + fusion_recalls['shirt'][50] +
                                         fusion_recalls['toptee'][10] + fusion_recalls['toptee'][50]) / 6
                best_weight = float((i+1) / 10)
        metric.print_recall(output_path)

    best_test_fusion_run = {}
    best_test_fusion_run.update(
        fuse(
            runs=[dense_run, sparse_run],
            weights=[best_weight, 1 - best_weight]
        )
    )

    '''
    if dist.get_rank() == 0:
        print(best_test_fusion_run)
    '''

    if 'caption_generation' in search_args.rerank_template:
        rerank_best_test_fusion_run = ranker.caption_generation_rerank(best_test_fusion_run, search_args.rerank_type,
                                                    search_args.rerank_num, data_args,
                                                    training_args, model_args, search_args,
                                                    rerank_prompt_type=search_args.rerank_template)
    else:
        rerank_best_test_fusion_run = ranker.rerank(best_test_fusion_run, search_args.rerank_type, search_args.rerank_num, data_args,
                                        training_args, model_args, rerank_prompt_type=search_args.rerank_template)


    '''
    if dist.get_rank() == 0:
        print(rerank_best_test_fusion_run)
    '''
    if training_args.task_type == 'tbpr':
        output_path = os.path.join(
            f'search_results/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{search_args.query_type}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.tbpr_type}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{data_args.sparse_type}_rerank_{search_args.rerank_type}_{search_args.rerank_num}_{search_args.rerank_template}',
            f'best.xlsx')
    else:
        output_path = os.path.join(
            f'search_results/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{search_args.query_type}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{data_args.sparse_type}_rerank_{search_args.rerank_type}_{search_args.rerank_num}_{search_args.rerank_template}',
            f'best.xlsx')

    metric = RecallMetrics(dataset, dense_run, sparse_run, rerank_best_test_fusion_run, look_up, lookup_indices, search_args)

    metric.sort_and_count()

    metric.all_gather_object()
    metric.print_recall(output_path)

    '''
    if 'caption_generation' in search_args.rerank_template and search_args.query_type == 'image':
        metric.statistical_error_data(processor, best_test_fusion_run)
    '''

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
