import logging
import os
import sys

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from PIL import Image

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import (
    HfArgumentParser,
    set_seed,
    is_torch_xla_available,
    EarlyStoppingCallback,
    EvalPrediction
)
from transformers import LlavaProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, \
    LlavaNextForConditionalGeneration
from tevatron.retriever.arguments import ModelArguments, DataArguments
from tevatron.retriever.gc_trainer import GradCacheTrainer as GCTrainer
import torch.distributed as dist
from arguments import TrainingArguments
from dataset import CrossModalRetrievalDataset
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F

from template import text_prompt, img_prompt, text_prompt_no_one_word, img_prompt_no_one_word, img_prompt_no_special_llava_v1_5
from model import MLLMRetrievalModel


def get_filtered_ids(tokenizer):
    filtered_ids = set()
    for token, id in tokenizer.get_vocab().items():
        if token[0] == '▁' or token[0] == ' ':
            token = token[1:]
        if not token.isalpha() and not token.isdigit():
            continue
        if ord('a') <= ord(token[0]) and ord(token[0]) <= ord('z'):
            filtered_ids.add(id)
    return filtered_ids


def get_valid_tokens_values(text, tokenizer, logits, vocab_dict, data_args, filtered_ids):
    words = [i for i in word_tokenize(text.lower()) if i not in stopwords]
    token_ids = set()
    for word in words:
        # TODO： 这里encode的结果可能是有_符号的，不知对于稀疏特征是否有影响，需要想一想，如有可能需要改一改
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
在官方PromptReps中，有一个指定参数是multi_reps，目测是改取最后一个特征和logit为取多个特征和logit，
'''


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
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
        encoder = LlavaNextForConditionalGeneration.from_pretrained(model_args.model_name_or_path, device_map=device_map,
                                                                    torch_dtype=torch_type)
        processor = LlavaNextProcessor.from_pretrained(model_args.model_name_or_path)

    dataset = CrossModalRetrievalDataset('coco', processor, 'test', 'single')
    sampler = Data.DistributedSampler(dataset, num_replicas=world_size, shuffle=True, rank=rank)
    test_dataloader = Data.DataLoader(dataset=dataset, sampler=sampler, batch_size=4, shuffle=False)

    model = MLLMRetrievalModel(encoder)
    model = model.eval()
    print(model.is_ddp)

    datalist = []

    encoded = []
    jsonl_data = []
    lookup_indices = []

    # 加载词表并获取过滤后的单词id，但目前尚不清楚filtered_ids是做什么的
    vocab_dict = processor.tokenizer.get_vocab()
    vocab_dict = {v: k for k, v in vocab_dict.items()}
    filtered_ids = get_filtered_ids(processor.tokenizer)

    '''
    if dist.get_rank() == 0:
        print(vocab_dict)
        print(len(vocab_dict))
    '''

    with torch.no_grad():
        for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(test_dataloader),
                                                                     total=len(test_dataloader)):
            if model_args.model_name_or_path == './checkpoints/llava-hf-llava-1.5-7b-hf' or model_args.model_name_or_path == './checkpoints/llava-hf-llava-v1.6-vicuna-7b-hf':
                prompt = img_prompt_no_special_llava_v1_5
            else:
                prompt = img_prompt
            text_list = list(texts)
            print(imgs_path)
            # print(texts)
            # text_inputs = processor([text_prompt_no_one_word.replace('<sent>', text) for text in text_list],
            #                         return_tensors="pt",
            #                         padding=True).to(device)

            text_logits, text_embs = model.encode_data(texts, 'text', processor, device, model_args)
            # print(text_embs.shape)
            raw_images = [Image.open(path).convert('RGB') for path in imgs_path]
            img_inputs = processor(images=raw_images, text=[prompt] * len(imgs_path), return_tensors="pt",
                                   padding=True)
            print(img_inputs['pixel_values'].shape)
            imgs = img_inputs.to(device)
            # print(imgs.pixel_values.shape)
            image_logits, image_embs = model.encode_data(imgs, 'image', processor, device, model_args)
            # print(image_embs.shape)

            if dist.is_initialized():
                # print(dist.get_world_size())
                imgs_list = [[None] for _ in range(dist.get_world_size())]

                # print(imgs_list)
                # print(dist.get_rank())

                text_list = [torch.zeros_like(text_embs) for _ in range(dist.get_world_size())]
                image_list = [torch.zeros_like(image_embs) for _ in range(dist.get_world_size())]

                dist.all_gather(tensor_list=text_list, tensor=text_embs.contiguous())
                dist.all_gather(tensor_list=image_list, tensor=image_embs.contiguous())

                dist.all_gather_object(object_list=imgs_list, obj=img_ids)

                imgs_list[dist.get_rank()] = img_ids

                text_embeds = torch.cat(text_list, 0)
                image_embeds = torch.cat(image_list, 0)

                text_embeds = F.normalize(text_embeds, dim=-1)
                image_embeds = F.normalize(image_embeds, dim=-1)

                if dist.get_rank() == 0:
                    # print(imgs_list)
                    print(text_embeds.shape)
                    print(image_embeds.shape)
                    text_color = torch.zeros(text_embeds.shape[0])
                    image_color = torch.ones(image_embeds.shape[0])

                    datalist.extend(imgs_list)
                    text_tsne = TSNE(n_components=2, random_state=33, perplexity=15).fit_transform(text_embeds.to(torch.float16).cpu())
                    image_tsne = TSNE(n_components=2, random_state=33, perplexity=15).fit_transform(
                        image_embeds.to(torch.float16).cpu())
                    # plt.style.use("dark_background")
                    plt.figure(figsize=(8.5, 4))
                    plt.subplot(1, 2, 1)
                    plt.scatter(text_tsne[:, 0], text_tsne[:, 1], alpha=0.6,
                                cmap=plt.cm.get_cmap('rainbow', 10))
                    plt.scatter(image_tsne[:, 0], image_tsne[:, 1], alpha=0.6,
                                cmap=plt.cm.get_cmap('rainbow', 10))

                    plt.savefig(f"{model_args.model_name_or_path[14:]}_scatter.png")

                    plt.close()

                    # img = img.cuda()

            break

        # break

    '''
    if dist.get_rank() == 0:
        print(len(datalist))
        with open('output.txt', 'w') as file:
            file.write(str(datalist))
    '''


if __name__ == "__main__":
    main()
