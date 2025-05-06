import json
import logging
import os
import pickle
import string
import sys
import itertools

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
from peft import PeftModel, PeftConfig
# from fast_pytorch_kmeans import KMeans
# from faiss import Kmeans
from cuml.cluster import KMeans

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

    # 加载词表并获取过滤后的单词id，但目前尚不清楚filtered_ids是做什么的
    if 'InternVL2_5-8B' in model_args.model_name_or_path or 'InternVL2_5-4B' in model_args.model_name_or_path:
        vocab_dict = processor.get_vocab()
    else:
        vocab_dict = processor.tokenizer.get_vocab()
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
    if model_args.use_output_embedding_cluster:
        if dist.get_rank() == 0:
            print('kmeans will be initialized.')
        output_token_embeddings_for_kmeans = output_token_embeddings.detach().cpu().numpy()
        '''
        kmeans = faiss.Kmeans(
            d=output_token_dim,  # 特征维度
            k=model_args.cluster_sum,  # 聚类数
            gpu=True,  # 启用GPU加速
            niter=100,  # 迭代次数
            verbose=True
        )

        # 执行聚类
        kmeans.train(output_token_embeddings_for_kmeans)

        # 获取结果
        centroids = torch.from_numpy(kmeans.centroids).to(dtype=torch_type).cuda()  # 聚类中心

        centroids_dict = {index: [] for index in range(len(centroids))}

        print(centroids)
        print(centroids.shape)

        _, labels = kmeans.index.search(output_token_embeddings_for_kmeans, 1)  # 标签
        labels = torch.from_numpy(labels.squeeze()).cuda()
        print(labels)
        '''
        # output_token_embeddings_for_kmeans = output_token_embeddings.clone()

        '''
        # 执行聚类
        labels, centroids = kmeans(
            X=output_token_embeddings_for_kmeans,
            num_clusters=10,
            distance='euclidean',  # 可选 'cosine'
            device=torch.device('cuda')
        )
        '''
        # 初始化模型
        kmeans = KMeans(
            n_clusters=model_args.cluster_sum,
            n_init=10,  # 减少初始化随机性
            random_state=42,  # 固定随机种子
            algorithm="elkan",  # 对密集数据更快
            verbose=1,
        )

        # 训练并预测
        kmeans.fit(output_token_embeddings_for_kmeans)
        labels = kmeans.predict(output_token_embeddings_for_kmeans)
        labels = torch.from_numpy(labels.squeeze()).cuda()
        print(labels)

        # 获取聚类中心
        centroids = kmeans.cluster_centers_

        centroids = torch.from_numpy(centroids).to(dtype=torch_type).cuda()  # 聚类中心

        print(centroids)
        print(centroids.shape)

        if dist.get_rank() == 0:
            with open(f"kmeans_model_{model_args.model_name_or_path[14:]}_{model_args.cluster_sum}.pkl", "wb") as f:
                pickle.dump(kmeans, f)



if __name__ == "__main__":
    main()
