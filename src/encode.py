import json
import logging
import os
import pickle
import sys
import itertools

from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from PIL import Image

from tqdm import tqdm
from transformers import (
    HfArgumentParser,
)
from transformers import LlavaProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, \
    LlavaNextForConditionalGeneration
from tevatron.retriever.arguments import ModelArguments
from arguments import PromptRepsLLMDataArguments
import torch.distributed as dist
from arguments import TrainingArguments
from dataset import CrossModalRetrievalDataset
import torch
import torch.utils.data as Data

from template import text_prompt, img_prompt, text_prompt_no_one_word, img_prompt_no_one_word
from model import MLLMRetrievalModel


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


def get_img_valid_tokens_values(tokenizer, logits, vocab_dict, data_args, filtered_ids):
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
    top_k = 128
    top_k_values, top_k_indices = logits.topk(top_k, dim=-1)
    # 原文中说，最后，通过对原始logits值乘以100并进行整数运算实现量化，所得结果表示对应token的权重，这里再四舍五入到最近整数(这是为什么呢)
    values = np.rint(top_k_values.cpu().detach().float().numpy() * 100).astype(int)
    # 把token id换成对应的单词，保存在tokens中
    tokens = [vocab_dict[i.item()] for i in top_k_indices.cpu().detach().float().numpy()]
    return tokens, values


def get_text_valid_tokens_values(text, tokenizer, logits, vocab_dict, data_args, filtered_ids):
    words = [i for i in word_tokenize(text.lower()) if i not in stopwords.words('english')]
    token_ids = set()
    for word in words:
        token_ids.update(tokenizer.encode(word, add_special_tokens=False))

    # top tokens in the text
    token_ids_in_text = torch.tensor(list(token_ids))
    if len(token_ids_in_text) == 0:  # if no tokens in the text (rare case), we use top 10 tokens
        top_k_values, top_k_indices = logits.topk(10, dim=-1)
        values = np.rint(top_k_values.cpu().detach().float().numpy() * 100).astype(int)
        tokens = [vocab_dict[i.item()] for i in top_k_indices.cpu().detach().float().numpy()]
    else:
        # 根据原文，他们遵循了SPLADE设置，为了加强logit离散性，只保留最多128个值
        # 这里应该是先获得了logit，然后再来筛选哪些值，正常来说词表所有位置上的logit都很难是0，
        # 但是这里就默认那些没有的单词还有排名128名之后的单词logit为0了
        top_k = min(len(token_ids_in_text), 128)
        top_k_values, top_k_indices = logits[token_ids_in_text].topk(top_k, dim=-1)
        # 原文中说，最后，通过对原始logits值乘以100并进行整数运算实现量化，所得结果表示对应token的权重，这里再四舍五入到最近整数(这是为什么呢)
        values = np.rint(top_k_values.cpu().detach().float().numpy() * 100).astype(int)
        # 把token id换成对应的单词，保存在tokens中
        tokens = [vocab_dict[i.item()] for i in token_ids_in_text[top_k_indices.cpu().detach().float().numpy()]]

    # top tokens not in the text for expansion.
    '''
    if data_args.num_expended_tokens > 0:
        token_ids_out_text = torch.tensor(list(filtered_ids - token_ids))
        top_k = min(data_args.num_expended_tokens, len(token_ids_out_text))
        top_k_values, top_k_indices = logits[token_ids_out_text].topk(top_k, dim=-1)
        values = np.append(values, np.rint(top_k_values.cpu().detach().float().numpy() * 100).astype(int))
        for i in token_ids_out_text[top_k_indices.cpu().detach().float().numpy()]:
            tokens.append(vocab_dict[i.item()])
    '''
    return tokens, values


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
    if model_args.model_name_or_path == './checkpoints/llava-hf-llava-1.5-7b-hf':
        encoder = LlavaForConditionalGeneration.from_pretrained(model_args.model_name_or_path, device_map=device_map,
                                                                torch_dtype=torch_type)
        processor = LlavaProcessor.from_pretrained(model_args.model_name_or_path)
    else:
        encoder = LlavaNextForConditionalGeneration.from_pretrained(model_args.model_name_or_path,
                                                                    device_map=device_map,
                                                                    torch_dtype=torch_type)
        processor = LlavaNextProcessor.from_pretrained(model_args.model_name_or_path)

    if training_args.encode_type == 'text':
        dataset = CrossModalRetrievalDataset(data_args.dataset_name, processor, 'test', 'full')
    else:
        dataset = CrossModalRetrievalDataset(data_args.dataset_name, processor, 'test', 'single')
    sampler = Data.DistributedSampler(dataset, num_replicas=dist.get_world_size(), shuffle=True, rank=dist.get_rank())
    test_dataloader = Data.DataLoader(dataset=dataset, sampler=sampler, pin_memory=True, batch_size=data_args.per_device_batch_size, shuffle=False)

    model = MLLMRetrievalModel(encoder)
    model = model.eval()
    print(model.is_ddp)

    encoded = []
    jsonl_data = []
    lookup_indices = []

    # 加载词表并获取过滤后的单词id，但目前尚不清楚filtered_ids是做什么的
    vocab_dict = processor.tokenizer.get_vocab()
    vocab_dict = {v: k for k, v in vocab_dict.items()}
    filtered_ids = get_filtered_ids(processor.tokenizer)

    with torch.no_grad():
        sampler.set_epoch(0)
        for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(test_dataloader),
                                                                     total=len(test_dataloader)):
            if len(texts) != data_args.per_device_batch_size:
                print(len(texts))
                print(dist.get_rank())
            if training_args.encode_type == 'text':
                logits, reps = model.encode_data(texts, 'text', processor, device)
            else:
                raw_images = [Image.open(path).convert('RGB') for path in imgs_path]
                img_inputs = processor(images=raw_images, text=[img_prompt] * len(imgs_path), return_tensors="pt",
                                       padding=True)
                imgs = img_inputs.to(device)
                logits, reps = model.encode_data(imgs, 'image', processor, device)

            if dist.is_initialized():
                # reps_list = [[None] for _ in range(dist.get_world_size())]
                # logits_list = [[None] for _ in range(dist.get_world_size())]

                reps_list = [torch.zeros_like(reps) for _ in range(dist.get_world_size())]
                logits_list = [torch.zeros_like(logits) for _ in range(dist.get_world_size())]
                texts_list = [[None] for _ in range(dist.get_world_size())]
                text_ids_list = [[None] for _ in range(dist.get_world_size())]
                image_ids_list = [[None] for _ in range(dist.get_world_size())]

                '''
                texts_list = [[None] for _ in range(dist.get_world_size())]
                text_ids_list = [[None] for _ in range(dist.get_world_size())]
                image_ids_list = [[None] for _ in range(dist.get_world_size())]
                '''

                dist.all_gather(tensor_list=reps_list, tensor=reps.contiguous())
                dist.all_gather(tensor_list=logits_list, tensor=logits.contiguous())
                dist.all_gather_object(object_list=text_ids_list, obj=text_ids)
                dist.all_gather_object(object_list=image_ids_list, obj=img_ids)
                dist.all_gather_object(object_list=texts_list, obj=texts)

                batch_reps = torch.cat(reps_list)
                batch_logits = torch.cat(logits_list)
                batch_text_ids = list(itertools.chain(*text_ids_list))
                batch_image_ids = list(itertools.chain(*image_ids_list))
                batch_texts = list(itertools.chain(*texts_list))

                if dist.get_rank() == 0:
                    if training_args.encode_type == 'text':
                        lookup_indices.extend(batch_text_ids)
                    else:
                        lookup_indices.extend(batch_image_ids)
                    encoded.append(batch_reps.cpu().detach().float().numpy())

                    ids = batch_text_ids if training_args.encode_type == 'text' else batch_image_ids
                    if training_args.encode_type == 'text':
                        for id, logits, text in zip(ids, batch_logits, batch_texts):
                            vector = dict()
                            tokens, values = get_text_valid_tokens_values(text, processor.tokenizer, logits, vocab_dict,
                                                                          data_args,
                                                                          filtered_ids)
                            for token, v in zip(tokens, values):
                                vector[token] = int(v)
                            jsonl_data.append(
                                dict(
                                    id=id,
                                    content="",
                                    vector=vector,
                                )
                            )
                    else:
                        for id, logits, text in zip(ids, batch_logits, batch_texts):
                            vector = dict()
                            tokens, values = get_img_valid_tokens_values(processor.tokenizer, logits, vocab_dict,
                                                                         data_args, filtered_ids)
                            for token, v in zip(tokens, values):
                                vector[token] = int(v)
                            jsonl_data.append(
                                dict(
                                    id=id,
                                    content="",
                                    vector=vector,
                                )
                            )

        if dist.get_rank() == 0:
            encoded = np.concatenate(encoded)

            print(len(encoded))
            print(len(lookup_indices))
            print(len(jsonl_data))

            os.makedirs(f'{data_args.dense_output_dir}/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{training_args.encode_type}', exist_ok=True)
            os.makedirs(f'{data_args.sparse_output_dir}/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{training_args.encode_type}', exist_ok=True)

            with open(os.path.join(f'{data_args.dense_output_dir}/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{training_args.encode_type}/', f'query.pkl') if data_args.encode_is_query else os.path.join(f'{data_args.dense_output_dir}/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{training_args.encode_type}/', f'corpus_{data_args.dataset_shard_index}.pkl'), 'wb') as f:
                pickle.dump((encoded, lookup_indices), f)

            with open(os.path.join(f'{data_args.sparse_output_dir}/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{training_args.encode_type}/', f'query.tsv') if data_args.encode_is_query else os.path.join(f'{data_args.sparse_output_dir}/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{training_args.encode_type}/', f'corpus_{data_args.dataset_shard_index}.jsonl'), 'w') as f:
                for data in jsonl_data:
                    if data_args.encode_is_query:
                        id = data['id']
                        vector = data['vector']
                        query = " ".join([" ".join([str(token)] * freq) for token, freq in vector.items()])
                        if len(query.strip()) == 0:
                            continue
                        f.write(f'{id}\t{query}\n')
                    else:
                        f.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    main()