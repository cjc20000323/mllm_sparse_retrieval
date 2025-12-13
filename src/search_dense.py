import glob
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
from transformers import CLIPModel, CLIPProcessor, BlipModel, BlipProcessor
from transformers import (
    HfArgumentParser,
)

from arguments import PromptRepsLLMDataArguments, PromptRepsLLMSearchArguments, ModelArguments
from arguments import TrainingArguments
from dataset import CrossModalRetrievalDataset, TextPersonRetrievalDataset
from metrices import RecallMetrics

torch.set_printoptions(threshold=10000)  # 数字根据你的张量尺寸调整
import torch.utils.data as Data
import torch.nn.functional as F
from nltk.corpus import stopwords
import string

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
    if 'clip' in model_args.model_name_or_path:
        encoder = CLIPModel.from_pretrained(model_args.model_name_or_path, device_map=device_map,
                                                torch_dtype=torch_type)
        processor = CLIPProcessor.from_pretrained(model_args.model_name_or_path)
    elif 'blip' in model_args.model_name_or_path:
        encoder = BlipModel.from_pretrained(model_args.model_name_or_path, device_map=device_map,
                                                torch_dtype=torch_type)
        processor = BlipProcessor.from_pretrained(model_args.model_name_or_path)
    else:
        encoder = CLIPModel.from_pretrained(model_args.model_name_or_path, device_map=device_map,
                                                torch_dtype=torch_type)
        processor = CLIPProcessor.from_pretrained(model_args.model_name_or_path)


    if training_args.task_type == 'tbpr':
        dataset = TextPersonRetrievalDataset(data_args.dataset_name, processor, 'test', 'full')
    else:
        if search_args.query_type == 'text':
            dataset = CrossModalRetrievalDataset(data_args.dataset_name, processor, 'test', 'full')
        else:
            dataset = CrossModalRetrievalDataset(data_args.dataset_name, processor, 'test', 'single')

    sampler = Data.DistributedSampler(dataset, num_replicas=world_size, shuffle=True, rank=rank)
    test_dataloader = Data.DataLoader(dataset=dataset, sampler=sampler, batch_size=data_args.per_device_batch_size,
                                      shuffle=False)

    encoder = encoder.eval()

    from tevatron.retriever.searcher import FaissFlatSearcher

    lookup_indices = []

    dense_run = {}

    dense_retriever_indices = []

    if search_args.passage_reps is not None:
        # 目前尚不清楚这里是怎么工作的
        # 另外，这里源代码里有multi_reps，暂时先不管，后面再加
        dense_retriever_indices = [search_args.passage_reps]

    if search_args.sparse_index is not None:
        # 目前尚不清楚这里是怎么工作的
        # 另外，这里源代码里有multi_reps，暂时先不管，后面再加
        sparse_retriever_indices = [search_args.sparse_index]

    for i in range(len(dense_retriever_indices)):

        dense_retriever = None

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


        if training_args.task_type == 'tbpr':
            with torch.no_grad(), torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
                for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(test_dataloader),
                                                                             total=len(test_dataloader)):
                    lookup_indices.extend(text_ids)
                    text_inputs = processor(text=texts, return_tensors="pt", padding=True)
                    query_dense_reps = encoder.get_text_features(text_inputs)

                    batch_ids = text_ids

                    if dense_retriever is not None:
                        query_dense_reps = F.normalize(query_dense_reps, dim=-1)
                        query_dense_reps = query_dense_reps.cpu().detach().float().numpy()
                        dense_scores, dense_rankings = search_queries(dense_retriever, query_dense_reps, look_up,
                                                                      search_args)
                        dense_run.update(
                            get_run_dict(batch_ids, dense_scores, dense_rankings, search_args.remove_query))

        else:
            with torch.no_grad(), torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
                for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(test_dataloader),
                                                                             total=len(test_dataloader)):
                    if search_args.query_type == 'text':
                        lookup_indices.extend(text_ids)
                    else:
                        lookup_indices.extend(img_ids)
                    if search_args.query_type == 'text':
                        text_inputs = processor(text=texts, return_tensors="pt", padding=True)
                        query_dense_reps = encoder.get_text_features(text_inputs)
                    else:
                        raw_images = [Image.open(path).convert('RGB') for path in imgs_path]
                        img_inputs = processor(images=raw_images, return_tensors="pt", padding=True)
                        imgs = img_inputs.to(device)
                        query_dense_reps = encoder.get_image_features(imgs)


                    if search_args.query_type == 'text':
                        batch_ids = text_ids
                    else:
                        batch_ids = img_ids
                    if dense_retriever is not None:
                        query_dense_reps = F.normalize(query_dense_reps, dim=-1)
                        query_dense_reps = query_dense_reps.cpu().detach().float().numpy()
                        dense_scores, dense_rankings = search_queries(dense_retriever, query_dense_reps, look_up,
                                                                      search_args)
                        dense_run.update(
                            get_run_dict(batch_ids, dense_scores, dense_rankings, search_args.remove_query))

        if dense_retriever:
            del dense_retriever
            torch.cuda.empty_cache()

    max_val_fusion_metric = 0

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

    if training_args.task_type == 'tbpr':
        os.makedirs(
            f'search_results/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{search_args.query_type}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.tbpr_type}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{data_args.sparse_type}',
            exist_ok=True)

        output_path = os.path.join(
            f'search_results/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{search_args.query_type}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.tbpr_type}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{data_args.sparse_type}',
            f'dense.xlsx')
    else:
        os.makedirs(
            f'search_results/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{search_args.query_type}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{data_args.sparse_type}',
            exist_ok=True)

        output_path = os.path.join(
            f'search_results/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{search_args.query_type}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{data_args.sparse_type}',
            f'dense.xlsx')

    metric = RecallMetrics(dataset, dense_run, dense_run, dense_run, look_up, lookup_indices, search_args)
    metric.sort_and_count()

    metric.all_gather_object()
    metric.print_recall(output_path)

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
