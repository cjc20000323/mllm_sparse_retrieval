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
    LlavaNextForConditionalGeneration, Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration, AutoModel, \
    AutoProcessor, \
    AutoTokenizer, PhiForCausalLM, Phi3ForCausalLM, AutoModelForCausalLM
from tevatron.retriever.arguments import ModelArguments
from arguments import PromptRepsLLMDataArguments
import torch.distributed as dist
from arguments import TrainingArguments
from dataset import CrossModalRetrievalDataset
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from template import text_prompt, img_prompt, text_prompt_no_one_word, img_prompt_no_one_word, \
    img_prompt_no_special_llava_v1_5, text_prompt_no_special_llava_v1_5, text_prompt_qwen_v2_5, img_prompt_qwen_v2_5, \
    img_prompt_intern_vl_v2_5, text_prompt_intern_vl_v2_5
from model import MLLMRetrievalModel
from utils import split_model

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def filter_token(token):
    if ord(token[0]) < ord('a') or ord(token[0]) > ord('z'):
        token = token[1:]
    return token


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
    if data_args.sparse_manual:
        top_k_values, top_k_indices = logits.topk(data_args.sparse_length, dim=-1)
    else:
        top_k = 128
        top_k_values, top_k_indices = logits.topk(top_k, dim=-1)
    # print(top_k_indices)
    # 原文中说，最后，通过对原始logits值乘以100并进行整数运算实现量化，所得结果表示对应token的权重，这里再四舍五入到最近整数(这是为什么呢)
    values = np.rint(top_k_values.cpu().detach().float().numpy() * 100).astype(int)
    # 把token id换成对应的单词，保存在tokens中
    # e5-v模型会预测出超过词表长度的id，所以过滤一下，这个现象很普遍，目前不知道为什么会预测出这样的结果
    if data_args.is_filtered:
        tokens = [filter_token(vocab_dict[i.item()].lower()) for i in top_k_indices.cpu().detach().float().numpy() if
                  i < len(vocab_dict)]
    else:
        tokens = [vocab_dict[i.item()].lower() for i in top_k_indices.cpu().detach().float().numpy() if
                  i < len(vocab_dict)]

    # top tokens not in the text for expansion.
    if data_args.num_expended_tokens > 0:
        token_ids_out_text = torch.tensor(list(filtered_ids - set(top_k_indices)))
        top_k = min(data_args.num_expended_tokens, len(token_ids_out_text))
        top_k_values, top_k_indices = logits[token_ids_out_text].topk(top_k, dim=-1)
        values = np.append(values, np.rint(top_k_values.cpu().detach().float().numpy() * 100).astype(int))
        for i in token_ids_out_text[top_k_indices.cpu().detach().float().numpy()]:
            tokens.append(vocab_dict[i.item()].lower())
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
        if data_args.is_filtered:
            tokens = [filter_token(vocab_dict[i.item()].lower()) for i in top_k_indices.cpu().detach().float().numpy()]
        else:
            tokens = [vocab_dict[i.item()].lower() for i in top_k_indices.cpu().detach().float().numpy()]
    elif data_args.sparse_manual:
        top_k_values, top_k_indices = logits.topk(data_args.sparse_length, dim=-1)
        values = np.rint(top_k_values.cpu().detach().float().numpy() * 100).astype(int)
        if data_args.is_filtered:
            tokens = [filter_token(vocab_dict[i.item()].lower()) for i in top_k_indices.cpu().detach().float().numpy()]
        else:
            tokens = [vocab_dict[i.item()].lower() for i in top_k_indices.cpu().detach().float().numpy()]
    else:
        # 根据原文，他们遵循了SPLADE设置，为了加强logit离散性，只保留最多128个值
        # 这里应该是先获得了logit，然后再来筛选哪些值，正常来说词表所有位置上的logit都很难是0，
        # 但是这里就默认那些没有的单词还有排名128名之后的单词logit为0了
        top_k = min(len(token_ids_in_text), 128)
        top_k_values, top_k_indices = logits[token_ids_in_text].topk(top_k, dim=-1)
        # 原文中说，最后，通过对原始logits值乘以100并进行整数运算实现量化，所得结果表示对应token的权重，这里再四舍五入到最近整数(这是为什么呢)
        values = np.rint(top_k_values.cpu().detach().float().numpy() * 100).astype(int)
        # 把token id换成对应的单词，保存在tokens中
        if data_args.is_filtered:
            tokens = [filter_token(vocab_dict[i.item()].lower()) for i in
                      token_ids_in_text[top_k_indices.cpu().detach().float().numpy()]]
        else:
            tokens = [vocab_dict[i.item()].lower() for i in
                      token_ids_in_text[top_k_indices.cpu().detach().float().numpy()]]

    # top tokens not in the text for expansion.
    if data_args.num_expended_tokens > 0:
        token_ids_out_text = torch.tensor(list(filtered_ids - token_ids))
        top_k = min(data_args.num_expended_tokens, len(token_ids_out_text))
        top_k_values, top_k_indices = logits[token_ids_out_text].topk(top_k, dim=-1)
        values = np.append(values, np.rint(top_k_values.cpu().detach().float().numpy() * 100).astype(int))
        for i in token_ids_out_text[top_k_indices.cpu().detach().float().numpy()]:
            if data_args.is_filtered:
                tokens.append(filter_token(vocab_dict[i.item()].lower()))
            else:
                tokens.append(vocab_dict[i.item()].lower())
    return tokens, values


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def main():
    parser = HfArgumentParser((ModelArguments, PromptRepsLLMDataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: PromptRepsLLMDataArguments
    training_args: TrainingArguments

    # 下面这部分指定采用的模型精度
    if training_args.bf16:
        torch_type = torch.bfloat16
    elif training_args.fp16:
        torch_type = torch.float16
    else:
        torch_type = torch.float32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 指定模型
    device_map = split_model('InternVL2_5-8B')
    encoder = AutoModel.from_pretrained(model_args.model_name_or_path,
                                        device_map=device_map,
                                        torch_dtype=torch_type,
                                        trust_remote_code=True,
                                        low_cpu_mem_usage=True, )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, use_fast=False)

    if training_args.encode_type == 'text':
        dataset = CrossModalRetrievalDataset(data_args.dataset_name, None, 'test', 'full')
    else:
        dataset = CrossModalRetrievalDataset(data_args.dataset_name, None, 'test', 'single')
    test_dataloader = Data.DataLoader(dataset=dataset, pin_memory=True,
                                      batch_size=data_args.per_device_batch_size, shuffle=False, num_workers=0)

    encoder.eval()

    encoded = []
    jsonl_data = []
    lookup_indices = []

    # 加载词表并获取过滤后的单词id，但目前尚不清楚filtered_ids是做什么的
    vocab_dict = tokenizer.get_vocab()
    vocab_dict = {v: k for k, v in vocab_dict.items()}
    filtered_ids = get_filtered_ids(tokenizer)

    count = 0
    for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(test_dataloader),
                                                                 total=len(test_dataloader)):

        count += 1
        if training_args.encode_type == 'text':
            pass
        else:
            # Preparation for inference
            # raw_images = [Image.open(path).convert('RGB') for path in imgs_path]
            img_inputs = load_image('path', max_num=12).to(torch.bfloat16).cuda()

    print(count)


if __name__ == "__main__":
    main()
