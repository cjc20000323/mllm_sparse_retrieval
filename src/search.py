import glob
import os
import pickle
import faiss
from itertools import chain
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
    LlavaNextForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, AutoProcessor, AutoModelForCausalLM, AutoModel
from encode import get_filtered_ids
from dataset import CrossModalRetrievalDataset

import numpy as np
import torch
import torch.utils.data as Data
import torch.nn.functional as F
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from template import text_prompt, img_prompt, text_prompt_no_one_word, img_prompt_no_one_word, \
    img_prompt_no_special_llava_v1_5, text_prompt_qwen_v2_5, img_prompt_qwen_v2_5
from encode import get_img_valid_tokens_values, get_text_valid_tokens_values
from hybrid import fuse, write_trec_run, read_trec_run

stopwords = set(stopwords.words('english') + list(string.punctuation))

import logging

logger = logging.getLogger(__name__)


def pickle_load(path):
    with open(path, 'rb') as f:
        reps, lookup = pickle.load(f)
    return np.array(reps), lookup


def search_queries(retriever, q_reps, p_lookup, args):
    if args.batch_size > 0:
        all_scores, all_indices = retriever.batch_search(q_reps, args.depth, args.batch_size, args.quiet)
    else:
        all_scores, all_indices = retriever.search(q_reps, args.depth)

    psg_indices = [[str(p_lookup[x]) for x in q_dd] for q_dd in all_indices]
    psg_indices = np.array(psg_indices)
    return all_scores, psg_indices


def get_run_dict(batch_ids, batch_scores, batch_rankings, remove_query):
    run_dict = {}
    for qid, scores, rankings in zip(batch_ids, batch_scores, batch_rankings):
        run_dict[qid] = {}
        run_dict[qid]['docs'] = {}
        for score, doc in zip(scores, rankings):
            if remove_query:
                if doc == qid:
                    continue
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
    sampler = Data.DistributedSampler(dataset, num_replicas=world_size, shuffle=True, rank=rank)
    test_dataloader = Data.DataLoader(dataset=dataset, sampler=sampler, batch_size=data_args.per_device_batch_size,
                                      shuffle=False)

    model = MLLMRetrievalModel(encoder)
    model = model.eval()
    print(model.is_ddp)

    lookup_indices = []

    # 加载词表并获取过滤后的单词id，但目前尚不清楚filtered_ids是做什么的
    if 'InternVL2_5-8B' in model_args.model_name_or_path:
        vocab_dict = processor.get_vocab()
        filtered_ids = get_filtered_ids(processor)
    else:
        vocab_dict = processor.tokenizer.get_vocab()
        filtered_ids = get_filtered_ids(processor.tokenizer)
    vocab_dict = {v: k for k, v in vocab_dict.items()}

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

    if dist.get_rank() == 0:
        print(max(len(dense_retriever_indices), len(sparse_retriever_indices)))
        print(dense_retriever_indices)
        print(sparse_retriever_indices)
    for i in range(max(len(dense_retriever_indices), len(sparse_retriever_indices))):

        dense_retriever = None
        sparse_retriever = None

        if dense_retriever_indices:
            index_files = glob.glob(os.path.join(dense_retriever_indices[i], 'corpus*.pkl'))

            p_reps_0, p_lookup_0 = pickle_load(index_files[0])
            print(p_reps_0.shape)
            dense_retriever = FaissFlatSearcher(p_reps_0)

            look_up = []
            # 经DeepSeek老师讲解，他说FaissFlatSearcher初始化时仅分配了内存结构，未添加任何数据。所以这里再重新加一下，
            # 这也和源代码中重复add了p_reps_0一致，希望D老师没骗我吧
            dense_retriever.add(p_reps_0)
            look_up += p_lookup_0

            if dist.get_rank() == 0:
                print(len(look_up))

            # 在源代码里，并没有将所有数据都转移到某个GPU上面保存，而是各自保存，这样的话corpus会有多个编号，因此会有下面这一段处理多个corpus的代码，
            # 但是我们这里是先集中后保存，这样就只有一个文件，所以就先注释掉了
            '''
            shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
            if len(index_files) > 1:
                shards = tqdm(shards, desc='Loading shards into index', total=len(index_files))
            look_up = []
            for p_reps, p_lookup in shards:
                dense_retriever.add(p_reps)
                look_up += p_lookup
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

        if sparse_retriever_indices:
            sparse_retriever = LuceneImpactSearcher(os.path.join(sparse_retriever_indices[i], 'index'), None)
            analyzer = JWhiteSpaceAnalyzer()
            sparse_retriever.set_analyzer(analyzer)

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
                else:
                    prompt = img_prompt
                # batch = batch.to(training_args.device)
                # batch['qids'] = batch_ids
                # model_output: EncoderOutput = model(query=batch)
                if search_args.query_type == 'text':
                    query_logits, query_dense_reps = model.encode_data(texts, 'text', processor, device, model_args, data_args)
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
                    query_logits, query_dense_reps = model.encode_data(imgs, 'image', processor, device, model_args, data_args)

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
                                    tokens, values = get_text_valid_tokens_values(text, processor.tokenizer, logits,
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
                        if search_args.query_type == 'text':
                            for _, logits, text in zip(batch_ids, query_logits, texts):
                                tokens, values = get_text_valid_tokens_values(text, processor.tokenizer, logits,
                                                                              vocab_dict,
                                                                              data_args,
                                                                              filtered_ids)
                                query = ""
                                for token, v in zip(tokens, values):
                                    query += (' ' + token) * v
                                batch_topics.append(query.strip())
                            sparse_scores, sparse_rankings = sparse_search(sparse_retriever, batch_topics,
                                                                           batch_ids,
                                                                           search_args)
                            sparse_run.update(
                                get_run_dict(batch_ids, sparse_scores, sparse_rankings, search_args.remove_query))

                        else:
                            for _, logits in zip(batch_ids, query_logits):
                                tokens, values = get_img_valid_tokens_values(processor.tokenizer, logits,
                                                                             vocab_dict,
                                                                             data_args,
                                                                             filtered_ids)
                                query = ""
                                for token, v in zip(tokens, values):
                                    query += (' ' + token) * v
                                batch_topics.append(query.strip())
                            sparse_scores, sparse_rankings = sparse_search(sparse_retriever, batch_topics,
                                                                           batch_ids,
                                                                           search_args)
                            sparse_run.update(
                                get_run_dict(batch_ids, sparse_scores, sparse_rankings, search_args.remove_query))

        if dense_retriever:
            del dense_retriever
            torch.cuda.empty_cache()

    '''
    if data_args.multi_reps:
        dense_run = max_pooling_run(dense_run, search_args)
        sparse_run = max_pooling_run(sparse_run, search_args)
    '''

    '''
    if search_args.passage_reps is not None and search_args.sparse_index is not None:
        fusion_run.update(
            fuse(
                runs=[dense_run, sparse_run],
                weights=[search_args.alpha, (1 - search_args.alpha)]
            )
        )
    '''
    del model

    if search_args.passage_reps is not None and search_args.sparse_index is not None:
        fusion_run.update(
            fuse(
                runs=[dense_run, sparse_run],
                weights=[search_args.alpha, (1 - search_args.alpha)]
            )
        )
    dense_count = 0
    sparse_count = 0
    fusion_count = 0
    dense_recall_list = [[None] for _ in range(dist.get_world_size())]
    sparse_recall_list = [[None] for _ in range(dist.get_world_size())]
    fusion_recall_list = [[None] for _ in range(dist.get_world_size())]
    if len(dense_run) > 0:
        for k, v in tqdm(dense_run.items()):
            target = dataset.get_target(k, search_args.query_type)
            if isinstance(target, list):
                target = torch.tensor([int(i) for i in target]).cuda()
            else:
                target = int(target)
            if len(v['docs']) == 0:
                continue
            search_result = list(v['docs'].keys())
            search_result = torch.tensor([int(i) for i in search_result]).cuda()
            if True in torch.isin(search_result, target):
                dense_count += 1

    if len(sparse_run) > 0:
        for k, v in tqdm(sparse_run.items()):
            target = dataset.get_target(k, search_args.query_type)
            if isinstance(target, list):
                target = torch.tensor([int(i) for i in target]).cuda()
            else:
                target = int(target)
            if len(v['docs']) == 0:
                continue
            search_result = list(v['docs'].keys())
            search_result = torch.tensor([int(i) for i in search_result]).cuda()
            if True in torch.isin(search_result, target):
                sparse_count += 1
    if len(fusion_run) > 0:
        for k, v in tqdm(fusion_run.items()):
            target = dataset.get_target(k, search_args.query_type)
            if isinstance(target, list):
                target = torch.tensor([int(i) for i in target]).cuda()
            else:
                target = int(target)
            if len(v) == 0:
                continue

            '''
            if search_args.depth == 1:
                search_result = list(v.keys())
                search_result = torch.tensor([int(i) for i in search_result]).cuda()
                if True in torch.isin(search_result, target):
                    fusion_count += 1
            else:
            '''
            # 这一段代码是要把密集检索和稀疏检索联合计算的分数从大到小排序，然后再取出前K个
            sorted_by_value = sorted(v.items(), key=lambda x: x[1], reverse=True)[:search_args.depth]
            largest_dict = dict(sorted_by_value)
            '''
            if dist.get_rank() == 0:
                print(sorted(v.items(), key=lambda x: x[1], reverse=True))
                print(largest_dict)
            '''
            search_result = list(largest_dict.keys())
            search_result = torch.tensor([int(i) for i in search_result]).cuda()
            if True in torch.isin(search_result, target):
                fusion_count += 1

    if len(dense_run) > 0:
        dense_count /= (len(lookup_indices) * dist.get_world_size())
        dist.all_gather_object(object_list=dense_recall_list, obj=dense_count)

    if len(sparse_run) > 0:
        sparse_count /= (len(lookup_indices) * dist.get_world_size())
        dist.all_gather_object(object_list=sparse_recall_list, obj=sparse_count)

    if len(fusion_run) > 0:
        fusion_count /= (len(lookup_indices) * dist.get_world_size())
        dist.all_gather_object(object_list=fusion_recall_list, obj=fusion_count)

    if dist.get_rank() == 0:
        print(len(lookup_indices) * dist.get_world_size())
        if len(dense_run) > 0:
            print(len(look_up))
            dense_recall = sum(dense_recall_list)
            print(dense_recall_list)
            print('Dense reps recall: ', dense_recall)

        if len(sparse_run) > 0:
            sparse_recall = sum(sparse_recall_list)
            print(sparse_recall_list)
            print('Sparse reps recall: ', sparse_recall)

        if len(fusion_run) > 0:
            fusion_recall = sum(fusion_recall_list)
            print(fusion_recall_list)
            print('Sparse reps recall: ', fusion_recall)


if __name__ == '__main__':
    main()
