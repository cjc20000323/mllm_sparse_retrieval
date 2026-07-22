import gc
import glob
import json
import logging
import os
import pickle
import string
import subprocess
import sys
import traceback
from contextlib import nullcontext
from itertools import chain

import dspy
import faiss
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data as Data
from PIL import Image
from nltk import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
)
from transformers import (LlavaProcessor,
                          LlavaForConditionalGeneration, LlavaNextProcessor, \
                          LlavaNextForConditionalGeneration, Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration,
                          AutoModel, \
                          AutoProcessor, Qwen3VLProcessor, Qwen3VLForConditionalGeneration)

from arguments import PromptRepsLLMDataArguments, ModelArguments
from arguments import TrainingArguments, PromptGenerationArguments, PromptRepsLLMSearchArguments
from dataset import CrossModalRetrievalDataset, TextPersonRetrievalDataset, ComposedTextImageRetrievalDataset, \
    Text2ImagetextRetrievalDataset, Imagetext2TextRetrievalDataset
from hybrid import fuse
from metrices import RecallMetrics
from model import MLLMRetrievalModel
from template import (llava_mistral_template_image_prefix, llava_mistral_template_content_element,
                      img_prompt_for_concat,
                      llama3_template_image_prefix, llama3_template_content_element, llava_mistral_template_text_prefix,
                      text_prompt_for_concat, llama3_template_text_prefix)

logger = logging.getLogger(__name__)

model_begin_indice = 28
path_prefix = '/root/autodl-fs/'


def pickle_load(path):
    with open(path, 'rb') as f:
        reps, lookup = pickle.load(f)
    return np.array(reps), lookup


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
    if token[0] == 'Ġ' or token[0] == 'ġ':
        token = token[1:]
    '''
    if ord(token[0]) < ord('a') or ord(token[0]) > ord('z'):
        token = token[1:]
    '''
    return token


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


def search_queries(retriever, q_reps, p_lookup, args):
    if args.retrieval_batch_size > 0:
        all_scores, all_indices = retriever.batch_search(q_reps, args.depth, args.retrieval_batch_size, args.quiet)
    else:
        all_scores, all_indices = retriever.search(q_reps, args.depth)

    psg_indices = [[str(p_lookup[x]) for x in q_dd] for q_dd in all_indices]
    psg_indices = np.array(psg_indices)
    return all_scores, psg_indices


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


def get_text_valid_disassemble_tokens_values(text, tokenizer, disassemble_logits, vocab_dict, data_args,
                                             filtered_ids, logits=None, model_args=None):
    word_set = set()
    word_values = dict()
    if data_args.sparse_manual:
        top_k = data_args.sparse_length
    else:
        top_k = data_args.sparse_length
    if model_args is not None and (
            model_args.eol_type == 'disassembleeol_separate_origin_text' or model_args.eol_type == 'all_disassembleeol_origin_text' or model_args.eol_type == 'disassembleeol_concrete_origin_text' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text'):
        words = [i for i in word_tokenize(text.lower()) if
                 i not in set(stopwords.words('english') + list(string.punctuation))]
        token_ids = set()
        for word in words:
            token_ids.update(tokenizer.encode(word, add_special_tokens=False))

        # top tokens in the text
        token_ids_in_text = torch.tensor(list(token_ids))
        top_k = min(len(token_ids_in_text), 128)

    if model_args is not None and (
            model_args.eol_type == 'disassembleeol_separate_origin_text' or model_args.eol_type == 'all_disassembleeol_origin_text'):
        top_k_values, top_k_indices = disassemble_logits[:, token_ids_in_text].topk(top_k, dim=-1)
        values = np.rint(top_k_values.cpu().detach().float().numpy() * 100).astype(int)
        for indice_list, value_list in zip(token_ids_in_text[top_k_indices.cpu().detach().float().numpy()], values):
            for indice, value in zip(indice_list, value_list):
                if int(indice.item()) < len(vocab_dict):
                    if vocab_dict[int(indice.item())] in word_values.keys():
                        if int(indice.item()) < len(vocab_dict):
                            if data_args.sparse_value_type == 'replace':
                                word_values[vocab_dict[int(indice.item())]] = value
                            elif data_args.sparse_value_type == 'sum':
                                word_values[vocab_dict[int(indice.item())]] += value
                            else:
                                if value > word_values[vocab_dict[int(indice.item())]]:
                                    word_values[vocab_dict[int(indice.item())]] = value
                    else:
                        if int(indice.item()) < len(vocab_dict):
                            word_values[vocab_dict[int(indice.item())]] = value
    else:
        top_k_values, top_k_indices = disassemble_logits.topk(top_k, dim=-1)
        for top_k_indice_list in top_k_indices:
            word_set.update(top_k_indice_list.tolist())
        if model_args is not None and (
                model_args.eol_type == 'disassembleeol_separate' or model_args.eol_type == 'all_disassembleeol'):
            values = np.rint(top_k_values.cpu().detach().float().numpy() * 100).astype(int)
            for indice_list, value_list in zip(top_k_indices.cpu().detach().float().numpy(), values):
                for indice, value in zip(indice_list, value_list):
                    if int(indice.item()) < len(vocab_dict):
                        if vocab_dict[int(indice.item())] in word_values.keys():
                            if int(indice.item()) < len(vocab_dict):
                                if data_args.sparse_value_type == 'replace':
                                    word_values[vocab_dict[int(indice.item())]] = value
                                elif data_args.sparse_value_type == 'sum':
                                    word_values[vocab_dict[int(indice.item())]] += value
                                else:
                                    if value > word_values[vocab_dict[int(indice.item())]]:
                                        word_values[vocab_dict[int(indice.item())]] = value
                        else:
                            if int(indice.item()) < len(vocab_dict):
                                word_values[vocab_dict[int(indice.item())]] = value

    if data_args.print_sparse:
        for top_k_indice_list, top_k_value_list in zip(top_k_indices, top_k_values):
            if dist.get_rank() == 2:
                print(
                    [{vocab_dict[i]: value} for i, value in zip(top_k_indice_list.tolist(), top_k_value_list.tolist())])

    if model_args is not None and (
            model_args.eol_type == 'disassembleeol_separate' or model_args.eol_type == 'all_disassembleeol'
            or model_args.eol_type == 'disassembleeol_separate_origin_text'
            or model_args.eol_type == 'all_disassembleeol_origin_text'):
        values = [word_values[key] for key in word_values.keys()]
        if data_args.is_filtered and data_args.sparse_lower_or_upper == 'lower':
            tokens = [filter_token(key.lower()) for key in word_values.keys()]
        elif data_args.sparse_lower_or_upper == 'lower':
            tokens = [key.lower() for key in word_values.keys()]
        else:
            tokens = [key for key in word_values.keys()]
    else:
        if model_args is not None and (
                model_args.eol_type == 'disassembleeol_concrete_origin_text' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text'):
            word_set = set(token_ids_in_text.tolist())
        values = [logits[indice].cpu().detach() for indice in word_set if indice < len(vocab_dict)]
        values = np.rint(np.array(values) * 100).astype(int)

        if data_args.is_filtered and data_args.sparse_lower_or_upper == 'lower':
            tokens = [filter_token(vocab_dict[i].lower()) for i in word_set if i < len(vocab_dict)]
        elif data_args.sparse_lower_or_upper == 'lower':
            tokens = [vocab_dict[i].lower() for i in word_set if i < len(vocab_dict)]
        else:
            tokens = [vocab_dict[i] for i in word_set if i < len(vocab_dict)]
    return tokens, values


def get_img_valid_disassemble_tokens_values(tokenizer, disassemble_logits, vocab_dict, data_args, filtered_ids,
                                            logits=None, model_args=None):
    word_set = set()
    word_values = dict()
    if data_args.sparse_manual:
        top_k = data_args.sparse_length
    else:
        top_k = data_args.sparse_length

    top_k_values, top_k_indices = disassemble_logits.topk(top_k, dim=-1)
    for top_k_indice_list in top_k_indices:
        word_set.update(top_k_indice_list.tolist())
    if data_args.print_sparse:
        for top_k_indice_list, top_k_value_list in zip(top_k_indices, top_k_values):
            if dist.get_rank() == 0:
                print(
                    [{vocab_dict[i]: value} for i, value in zip(top_k_indice_list.tolist(), top_k_value_list.tolist())])
    if model_args is not None and (
            model_args.eol_type == 'disassembleeol_separate' or model_args.eol_type == 'all_disassembleeol'
            or model_args.eol_type == 'disassembleeol_separate_origin_text'
            or model_args.eol_type == 'all_disassembleeol_origin_text'):
        values = np.rint(top_k_values.cpu().detach().float().numpy() * 100).astype(int)
        # 下面这里，是通过循环，将五个prompt预测logit结果给拿出来，保存到word_value字典里，先区分大小写，、
        # 在下面构造token和value时会统一转成小写，并在main中的json构造循环里面再次计算sparse_value_type
        for indice_list, value_list in zip(top_k_indices.cpu().detach().float().numpy(), values):
            for indice, value in zip(indice_list, value_list):
                if int(indice.item()) < len(vocab_dict):
                    if vocab_dict[int(indice.item())] in word_values.keys():
                        if int(indice.item()) < len(vocab_dict):
                            if data_args.sparse_value_type == 'replace':
                                word_values[vocab_dict[int(indice.item())]] = value
                            elif data_args.sparse_value_type == 'sum':
                                word_values[vocab_dict[int(indice.item())]] += value
                            else:
                                if value > word_values[vocab_dict[int(indice.item())]]:
                                    word_values[vocab_dict[int(indice.item())]] = value
                    else:
                        if int(indice.item()) < len(vocab_dict):
                            word_values[vocab_dict[int(indice.item())]] = value

    if model_args is not None and (
            model_args.eol_type == 'disassembleeol_separate' or model_args.eol_type == 'all_disassembleeol'
            or model_args.eol_type == 'disassembleeol_separate_origin_text'
            or model_args.eol_type == 'all_disassembleeol_origin_text'):
        values = [word_values[key] for key in word_values.keys()]
        if data_args.is_filtered and data_args.sparse_lower_or_upper == 'lower':
            tokens = [filter_token(key.lower()) for key in word_values.keys()]
        elif data_args.sparse_lower_or_upper == 'lower':
            tokens = [key.lower() for key in word_values.keys()]
        else:
            tokens = [key for key in word_values.keys()]
    else:
        values = [logits[indice].cpu().detach() for indice in word_set if indice < len(vocab_dict)]
        values = np.rint(np.array(values) * 100).astype(int)

        if data_args.is_filtered and data_args.sparse_lower_or_upper == 'lower':
            tokens = [filter_token(vocab_dict[i].lower()) for i in word_set if i < len(vocab_dict)]
        elif data_args.sparse_lower_or_upper == 'lower':
            tokens = [vocab_dict[i].lower() for i in word_set if i < len(vocab_dict)]
        else:
            tokens = [vocab_dict[i] for i in word_set if i < len(vocab_dict)]
    return tokens, values


def close_sparse_retriever(sparse_retriever, analyzer=None):
    for resource in (sparse_retriever, analyzer):
        if resource is None:
            continue
        close = getattr(resource, 'close', None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                logger.warning("Failed to close sparse retrieval resource %r: %s", resource, exc)
    gc.collect()


class GenerateSchemaAspects(dspy.Signature):
    """Generate 3 to 7 retrieval-useful ontology aspects from 20-30 dataset texts."""

    dataset_name: str = dspy.InputField()
    task_type: str = dspy.InputField()
    seed_texts: str = dspy.InputField()
    aspects: str = dspy.OutputField()


class SchemaAspectProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateSchemaAspects)

    def forward(self, dataset_name, task_type, seed_texts):
        return self.generate(
            dataset_name=dataset_name,
            task_type=task_type,
            seed_texts=seed_texts,
        )


class RetrievalAction:
    def __init__(self, training_args, data_args, model_args, search_args, prompt_generation_args, model, processor,
                 vocab_dict):
        super().__init__()
        self.training_args = training_args
        self.data_args = data_args
        self.model_args = model_args
        self.search_args = search_args
        self.prompt_generation_args = prompt_generation_args
        self.model = model
        self.processor = processor
        self.vocab_dict = vocab_dict
        self.encode_counter = 0

    def generate_concat_prompts(self, aspects_prompt_list, encode_type):
        if encode_type == 'text':
            if 'llava-hf-llava-v1.6-mistral-7b-hf' in self.model_args.model_name_or_path:
                prompt_template = llava_mistral_template_text_prefix
                if 'concrete' in self.model_args.eol_type or 'all' not in self.model_args.eol_type:
                    prompt_template += llava_mistral_template_content_element.format(text_prompt_for_concat)
                for llava_mistral_retrieval_disassemble_text_prompt in aspects_prompt_list:
                    content_element = llava_mistral_template_content_element.format(
                        llava_mistral_retrieval_disassemble_text_prompt)
                    prompt_template += content_element
            else:
                prompt_template = llama3_template_text_prefix
                if 'concrete' in self.model_args.eol_type or 'all' not in self.model_args.eol_type:
                    prompt_template += llama3_template_content_element.format(text_prompt_for_concat)
                for llama3_retrieval_disassemble_text_prompt in aspects_prompt_list:
                    content_element = llama3_template_content_element.format(
                        llama3_retrieval_disassemble_text_prompt)
                    prompt_template += content_element
        else:
            if 'llava-hf-llava-v1.6-mistral-7b-hf' in self.model_args.model_name_or_path:
                prompt_template = llava_mistral_template_image_prefix
                if 'concrete' in self.model_args.eol_type or 'all' not in self.model_args.eol_type:
                    prompt_template += llava_mistral_template_content_element.format(
                        img_prompt_for_concat)
                for llava_mistral_retrieval_disassemble_image_prompt in aspects_prompt_list:
                    content_element = llava_mistral_template_content_element.format(
                        llava_mistral_retrieval_disassemble_image_prompt)
                    prompt_template += content_element
            else:
                prompt_template = llama3_template_image_prefix
                if 'concrete' in self.model_args.eol_type or 'all' not in self.model_args.eol_type:
                    prompt_template += llama3_template_content_element.format(img_prompt_for_concat)
                for llama3_retrieval_disassemble_image_prompt in aspects_prompt_list:
                    content_element = llama3_template_content_element.format(
                        llama3_retrieval_disassemble_image_prompt)
                    prompt_template += content_element
        return prompt_template

    def encode(self, test_dataloader, aspects_prompt_list, filtered_ids, encode_type, device):
        encoded = []
        jsonl_data = []
        lookup_indices = []
        if self.training_args.task_type == 'tbpr':
            prompt_template = self.generate_concat_prompts(aspects_prompt_list, encode_type)
            for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(test_dataloader),
                                                                         total=len(test_dataloader)):
                raw_images = [Image.open(path).convert('RGB') for path in imgs_path]
                img_inputs = self.processor(images=raw_images, text=[prompt_template] * len(imgs_path),
                                            return_tensors="pt",
                                            padding=True)
                imgs = img_inputs.to(device)
                logits, reps = self.model.encode_data_concat_for_tbpr_dspy(imgs, prompt_template, aspects_prompt_list,
                                                                           'image', self.processor, device,
                                                                           self.model_args,
                                                                           self.data_args)
                disassemble_logits = logits

                reps = F.normalize(reps, dim=-1)

                lookup_indices.extend(img_ids)

                encoded.append(reps.cpu().detach().float().numpy())
                ids = img_ids

                for img_indice in range(len(ids)):
                    id = ids[img_indice]
                    length = len(aspects_prompt_list)
                    disassemble_logit = disassemble_logits[
                        img_indice * length:(img_indice + 1) * length]
                    vector = dict()
                    tokens, values = get_img_valid_disassemble_tokens_values(self.processor,
                                                                             disassemble_logit,
                                                                             self.vocab_dict,
                                                                             self.data_args,
                                                                             filtered_ids, None,
                                                                             self.model_args)
                    for token, v in zip(tokens, values):
                        if token in vector.keys():
                            if self.data_args.sparse_value_type == 'replace':
                                vector[token] = int(v)
                            elif self.data_args.sparse_value_type == 'sum':
                                vector[token] += int(v)
                            else:
                                if int(v) > vector[token]:
                                    vector[token] = int(v)
                        else:
                            vector[token] = int(v)
                    if self.data_args.sparse_value_mean:
                        for token in vector.keys():
                            vector[token] //= length
                    jsonl_data.append(
                        dict(
                            id=id,
                            content="",
                            vector=vector,
                        )
                    )

        else:
            prompt_template = self.generate_concat_prompts(aspects_prompt_list, encode_type)
            for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(test_dataloader),
                                                                         total=len(test_dataloader)):
                with torch.cuda.amp.autocast() if self.training_args.fp16 else nullcontext():

                    if encode_type == 'text':
                        logits, reps = self.model.encode_data_concat_dspy(texts, prompt_template, aspects_prompt_list,
                                                                          'text',
                                                                          self.processor, device, self.model_args,
                                                                          self.data_args)
                        disassemble_logits = logits
                    else:
                        raw_images = [Image.open(path).convert('RGB') for path in imgs_path]
                        img_inputs = self.processor(images=raw_images, text=[prompt_template] * len(imgs_path),
                                                    return_tensors="pt",
                                                    padding=True)
                        imgs = img_inputs.to(device)
                        logits, reps = self.model.encode_data_concat_dspy(imgs, prompt_template, aspects_prompt_list,
                                                                          'image',
                                                                          self.processor, device, self.model_args,
                                                                          self.data_args)
                        disassemble_logits = logits

                    reps = F.normalize(reps, dim=-1)
                    if encode_type == 'text':
                        lookup_indices.extend(text_ids)
                    else:
                        lookup_indices.extend(img_ids)
                    encoded.append(reps.cpu().detach().float().numpy())
                    ids = text_ids if self.training_args.encode_type == 'text' else img_ids
                    if encode_type == 'text':
                        for text_indice in range(len(ids)):
                            id = ids[text_indice]
                            text = texts[text_indice]
                            length = len(aspects_prompt_list)
                            disassemble_logit = disassemble_logits[
                                text_indice * length:(text_indice + 1) * length]
                            vector = dict()
                            tokens, values = get_text_valid_disassemble_tokens_values(text,
                                                                                      self.processor.tokenizer,
                                                                                      disassemble_logit,
                                                                                      self.vocab_dict,
                                                                                      self.data_args,
                                                                                      filtered_ids, None,
                                                                                      self.model_args)

                            for token, v in zip(tokens, values):
                                if token in vector.keys():
                                    if self.data_args.sparse_value_type == 'replace':
                                        vector[token] = int(v)
                                    elif self.data_args.sparse_value_type == 'sum':
                                        vector[token] += int(v)
                                    else:
                                        if int(v) > vector[token]:
                                            vector[token] = int(v)
                                else:
                                    vector[token] = int(v)
                            if self.data_args.sparse_value_mean:
                                for token in vector.keys():
                                    vector[token] //= length
                            jsonl_data.append(
                                dict(
                                    id=id,
                                    content="",
                                    vector=vector,
                                )
                            )
                    else:
                        for img_indice in range(len(ids)):
                            id = ids[img_indice]
                            length = len(aspects_prompt_list)
                            disassemble_logit = disassemble_logits[
                                img_indice * length:(img_indice + 1) * length]
                            vector = dict()
                            tokens, values = get_img_valid_disassemble_tokens_values(self.processor,
                                                                                     disassemble_logit,
                                                                                     self.vocab_dict,
                                                                                     self.data_args,
                                                                                     filtered_ids, None,
                                                                                     self.model_args)
                            for token, v in zip(tokens, values):
                                if token in vector.keys():
                                    if self.data_args.sparse_value_type == 'replace':
                                        vector[token] = int(v)
                                    elif self.data_args.sparse_value_type == 'sum':
                                        vector[token] += int(v)
                                    else:
                                        if int(v) > vector[token]:
                                            vector[token] = int(v)
                                else:
                                    vector[token] = int(v)
                            if self.data_args.sparse_value_mean:
                                for token in vector.keys():
                                    vector[token] //= length
                            jsonl_data.append(
                                dict(
                                    id=id,
                                    content="",
                                    vector=vector,
                                )
                            )

        return encoded, jsonl_data, lookup_indices

    def search(self, test_dataloader, aspects_prompt_list, filtered_ids, dense_retriever, sparse_retriever, analyzer,
               look_up, dataset, split, best_weight, query_type, device, is_dspy=False, output_dir=None):
        # 每次检索之前都需要调整search_args中的query_type，以适应RecallMetric类中的方法
        dense_run = {}
        sparse_run = {}
        fusion_run = [{}] * 9
        lookup_indices = []

        self.search_args.query_type = query_type

        if self.training_args.task_type == 'tbpr':
            with torch.no_grad(), torch.cuda.amp.autocast() if self.training_args.fp16 else nullcontext():
                prompt_template = self.generate_concat_prompts(aspects_prompt_list, self.training_args.encode_type)
                for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(test_dataloader),
                                                                             total=len(test_dataloader)):

                    lookup_indices.extend(text_ids)
                    query_logits, query_dense_reps = self.model.encode_data_concat_for_tbpr_dspy(texts, prompt_template,
                                                                                            aspects_prompt_list, 'text',
                                                                                            self.processor, device,
                                                                                            self.model_args,
                                                                                            self.data_args)
                    disassemble_logits = query_logits

                    batch_ids = text_ids

                    query_dense_reps = F.normalize(query_dense_reps, dim=-1)
                    query_dense_reps = query_dense_reps.cpu().detach().float().numpy()
                    dense_scores, dense_rankings = search_queries(dense_retriever, query_dense_reps, look_up,
                                                                  self.search_args)
                    dense_run.update(
                        get_run_dict(batch_ids, dense_scores, dense_rankings, self.search_args.remove_query))

                    batch_topics = []
                    for text_indice in range(len(batch_ids)):
                        text = texts[text_indice]

                        length = len(aspects_prompt_list)
                        disassemble_logit = disassemble_logits[
                            text_indice * length:(text_indice + 1) * length]
                        vector = dict()
                        tokens, values = get_text_valid_disassemble_tokens_values(text,
                                                                                  self.processor.tokenizer,
                                                                                  disassemble_logit,
                                                                                  self.vocab_dict,
                                                                                  self.data_args,
                                                                                  filtered_ids,
                                                                                  None,
                                                                                  self.model_args)

                        for token, v in zip(tokens, values):
                            if token in vector.keys():
                                if self.data_args.sparse_value_type == 'replace':
                                    vector[token] = int(v)
                                elif self.data_args.sparse_value_type == 'sum':
                                    vector[token] += int(v)
                                else:
                                    if int(v) > vector[token]:
                                        vector[token] = int(v)
                            else:
                                vector[token] = int(v)
                        if self.data_args.sparse_value_mean:
                            for token in vector.keys():
                                vector[token] //= length
                        query = ""
                        for token, v in vector.items():
                            query += (' ' + token) * v
                        batch_topics.append(query.strip())
                    sparse_scores, sparse_rankings = sparse_search(sparse_retriever, batch_topics,
                                                                   batch_ids,
                                                                   self.search_args)
                    sparse_run.update(
                        get_run_dict(batch_ids, sparse_scores, sparse_rankings,
                                     self.search_args.remove_query))
        else:
            with torch.no_grad(), torch.cuda.amp.autocast() if self.training_args.fp16 else nullcontext():
                for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(test_dataloader),
                                                                             total=len(test_dataloader)):
                    if self.search_args.query_type == 'text':
                        lookup_indices.extend(text_ids)
                    else:
                        lookup_indices.extend(img_ids)

                    prompt_template = self.generate_concat_prompts(aspects_prompt_list, self.training_args.encode_type)

                    if self.search_args.query_type == 'text':
                        query_logits, query_dense_reps = self.model.encode_data_concat_dspy(texts, prompt_template,
                                                                                            aspects_prompt_list, 'text',
                                                                                            self.processor, device,
                                                                                            self.model_args,
                                                                                            self.data_args)
                        disassemble_logits = query_logits

                    else:
                        raw_images = [Image.open(path).convert('RGB') for path in imgs_path]
                        img_inputs = self.processor(images=raw_images, text=[prompt_template] * len(imgs_path),
                                                    return_tensors="pt",
                                                    padding=True)
                        imgs = img_inputs.to(device)
                        query_logits, query_dense_reps = self.model.encode_data_concat_dspy(imgs, prompt_template,
                                                                                            aspects_prompt_list,
                                                                                            'image', self.processor,
                                                                                            device,
                                                                                            self.model_args,
                                                                                            self.data_args)
                        disassemble_logits = query_logits

                    if self.search_args.query_type == 'text':
                        batch_ids = text_ids
                    else:
                        batch_ids = img_ids

                    query_dense_reps = F.normalize(query_dense_reps, dim=-1)

                    query_dense_reps = query_dense_reps.cpu().detach().float().numpy()
                    dense_scores, dense_rankings = search_queries(dense_retriever, query_dense_reps, look_up,
                                                                  self.search_args)
                    dense_run.update(
                        get_run_dict(batch_ids, dense_scores, dense_rankings, self.search_args.remove_query))

                    batch_topics = []
                    if self.search_args.query_type == 'text':
                        for text_indice in range(len(batch_ids)):
                            text = texts[text_indice]
                            length = len(aspects_prompt_list)
                            disassemble_logit = disassemble_logits[
                                text_indice * length:(text_indice + 1) * length]
                            vector = dict()
                            tokens, values = get_text_valid_disassemble_tokens_values(text,
                                                                                      self.processor.tokenizer,
                                                                                      disassemble_logit,
                                                                                      self.vocab_dict,
                                                                                      self.data_args,
                                                                                      filtered_ids,
                                                                                      None,
                                                                                      self.model_args)

                            for token, v in zip(tokens, values):
                                if token in vector.keys():
                                    if self.data_args.sparse_value_type == 'replace':
                                        vector[token] = int(v)
                                    elif self.data_args.sparse_value_type == 'sum':
                                        vector[token] += int(v)
                                    else:
                                        if int(v) > vector[token]:
                                            vector[token] = int(v)
                                else:
                                    vector[token] = int(v)
                            if self.data_args.sparse_value_mean:
                                for token in vector.keys():
                                    vector[token] //= length
                            query = ""
                            for token, v in vector.items():
                                query += (' ' + token) * v
                            batch_topics.append(query.strip())
                        sparse_scores, sparse_rankings = sparse_search(sparse_retriever, batch_topics,
                                                                       batch_ids,
                                                                       self.search_args)
                        sparse_run.update(
                            get_run_dict(batch_ids, sparse_scores, sparse_rankings,
                                         self.search_args.remove_query))

                    else:
                        for img_indice in range(len(batch_ids)):
                            length = len(aspects_prompt_list)
                            disassemble_logit = disassemble_logits[
                                img_indice * length:(img_indice + 1) * length]
                            vector = dict()
                            tokens, values = get_img_valid_disassemble_tokens_values(self.processor,
                                                                                     disassemble_logit,
                                                                                     self.vocab_dict,
                                                                                     self.data_args,
                                                                                     filtered_ids, None,
                                                                                     self.model_args)
                            for token, v in zip(tokens, values):
                                if token in vector.keys():
                                    if self.data_args.sparse_value_type == 'replace':
                                        vector[token] = int(v)
                                    elif self.data_args.sparse_value_type == 'sum':
                                        vector[token] += int(v)
                                    else:
                                        if int(v) > vector[token]:
                                            vector[token] = int(v)
                                else:
                                    vector[token] = int(v)
                            if self.data_args.sparse_value_mean:
                                for token in vector.keys():
                                    vector[token] //= length
                            query = ""
                            for token, v in vector.items():
                                query += (' ' + token) * v
                            batch_topics.append(query.strip())
                        sparse_scores, sparse_rankings = sparse_search(sparse_retriever, batch_topics,
                                                                       batch_ids,
                                                                       self.search_args)
                        sparse_run.update(
                            get_run_dict(batch_ids, sparse_scores, sparse_rankings, self.search_args.remove_query))

        close_sparse_retriever(sparse_retriever, analyzer)
        gc.collect()
        torch.cuda.empty_cache()

        if split == 'val':
            max_val_fusion_metric = 0
            val_best_weight = 0.5
            for i in range(9):
                fusion_run[i].update(
                    fuse(
                        runs=[dense_run, sparse_run],
                        weights=[float((i + 1) / 10), 1 - float((i + 1) / 10)]
                    )
                )

                metric = RecallMetrics(dataset, dense_run, sparse_run, fusion_run[i], look_up, lookup_indices,
                                       self.search_args)
                metric.sort_and_count()

                metric.all_gather_object()
                fusion_recalls = {k: sum(metric.fusion_recall_lists[k]) for k in metric.recall_k_setting_list}
                if (fusion_recalls[1] + fusion_recalls[5] + fusion_recalls[10]) / 3 > max_val_fusion_metric:
                    max_val_fusion_metric = (fusion_recalls[1] + fusion_recalls[5] + fusion_recalls[10]) / 3
                    val_best_weight = float((i + 1) / 10)
                if not is_dspy:
                    output_path = os.path.join(output_dir, f'0_{i + 1}_0_{10 - i - 1}_val.xlsx')
                    metric.print_recall(output_path)

            best_test_fusion_run = {}
            best_test_fusion_run.update(
                fuse(
                    runs=[dense_run, sparse_run],
                    weights=[val_best_weight, 1 - val_best_weight]
                )
            )

            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            return dense_run, sparse_run, best_test_fusion_run, lookup_indices, val_best_weight, max_val_fusion_metric
        else:
            max_test_fusion_metric = 0
            test_best_weight = 0.5

            for i in range(9):
                fusion_run[i].update(
                    fuse(
                        runs=[dense_run, sparse_run],
                        weights=[float((i + 1) / 10), 1 - float((i + 1) / 10)]
                    )
                )

                metric = RecallMetrics(dataset, dense_run, sparse_run, fusion_run[i], look_up, lookup_indices,
                                       self.search_args)
                metric.sort_and_count()

                metric.all_gather_object()
                fusion_recalls = {k: sum(metric.fusion_recall_lists[k]) for k in metric.recall_k_setting_list}
                if (fusion_recalls[1] + fusion_recalls[5] + fusion_recalls[10]) / 3 > max_test_fusion_metric:
                    max_test_fusion_metric = (fusion_recalls[1] + fusion_recalls[5] + fusion_recalls[10]) / 3
                    test_best_weight = float((i + 1) / 10)
                output_path = os.path.join(output_dir, f'0_{i + 1}_0_{10 - i - 1}_test.xlsx')
                metric.print_recall(output_path)

            best_test_fusion_run = {}
            best_test_fusion_run.update(
                fuse(
                    runs=[dense_run, sparse_run],
                    weights=[best_weight, 1 - best_weight]
                )
            )

            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            return dense_run, sparse_run, best_test_fusion_run, lookup_indices

    def print_metric(self, output_path, dataset, dense_run, sparse_run, fusion_run, look_up, lookup_indices,
                     search_args):
        metric = RecallMetrics(dataset, dense_run, sparse_run, fusion_run, look_up, lookup_indices,
                               search_args)
        metric.sort_and_count()

        metric.all_gather_object()
        metric.print_recall(output_path)

    def generate_encode_files(self, encoded, jsonl_data, lookup_indices, encode_type, split, is_dspy):
        if self.data_args.is_filtered:
            filtered = "filter"
        else:
            filtered = "no_filter"

        if self.data_args.sparse_manual:
            manual = 'manual'
        else:
            manual = "no_manual"

        if self.model_args.use_output_embedding_cluster:
            cluster = f'cluster_{self.model_args.cluster_sum}'
        else:
            cluster = 'no_cluster'

        if self.data_args.sparse_value_mean:
            use_sparse_value_mean = 'mean'
        else:
            use_sparse_value_mean = 'no_mean'

        encoded = np.concatenate(encoded)

        if self.training_args.task_type == 'tbpr':
            if is_dspy:
                self.encode_counter += 1
                encode_counter = self.encode_counter
                os.makedirs(
                    f'{self.data_args.dense_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/image/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.tbpr_type}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}_{encode_counter}',
                    exist_ok=True)
                os.makedirs(
                    f'{self.data_args.sparse_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/image/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.tbpr_type}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}_{encode_counter}',
                    exist_ok=True)

                dense_output_dir = f'{self.data_args.dense_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/image/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.tbpr_type}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}_{encode_counter}'
                sparse_output_dir = f'{self.data_args.sparse_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/image/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.tbpr_type}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}_{encode_counter}'

                with open(os.path.join(
                        f'{self.data_args.dense_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/image/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.tbpr_type}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}_{encode_counter}',
                        f'query.pkl') if self.data_args.encode_is_query else os.path.join(
                    f'{self.data_args.dense_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/image/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.tbpr_type}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}_{encode_counter}',
                    f'corpus_{dist.get_rank()}.pkl'), 'wb') as f:
                    pickle.dump((encoded, lookup_indices), f)

                with open(os.path.join(
                        f'{self.data_args.sparse_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/image/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.tbpr_type}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}_{encode_counter}',
                        f'query.tsv') if self.data_args.encode_is_query else os.path.join(
                    f'{self.data_args.sparse_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/image/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.tbpr_type}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}_{encode_counter}',
                    f'corpus_{dist.get_rank()}.jsonl'), 'w') as f:
                    for data in jsonl_data:
                        if self.data_args.encode_is_query:
                            id = data['id']
                            vector = data['vector']
                            query = " ".join([" ".join([str(token)] * freq) for token, freq in vector.items()])
                            if len(query.strip()) == 0:
                                continue
                            f.write(f'{id}\t{query}\n')
                        else:
                            f.write(json.dumps(data) + "\n")
            else:
                os.makedirs(
                    f'{self.data_args.dense_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/image/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.tbpr_type}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}',
                    exist_ok=True)
                os.makedirs(
                    f'{self.data_args.sparse_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/image/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.tbpr_type}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}',
                    exist_ok=True)

                dense_output_dir = f'{self.data_args.dense_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/image/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.tbpr_type}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}'
                sparse_output_dir = f'{self.data_args.sparse_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/image/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.tbpr_type}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}'

                with open(os.path.join(
                        f'{self.data_args.dense_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/image/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.tbpr_type}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}',
                        f'query.pkl') if self.data_args.encode_is_query else os.path.join(
                    f'{self.data_args.dense_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/image/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.tbpr_type}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}',
                    f'corpus_{dist.get_rank()}.pkl'), 'wb') as f:
                    pickle.dump((encoded, lookup_indices), f)

                with open(os.path.join(
                        f'{self.data_args.sparse_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/image/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.tbpr_type}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}',
                        f'query.tsv') if self.data_args.encode_is_query else os.path.join(
                    f'{self.data_args.sparse_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/image/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.tbpr_type}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}',
                    f'corpus_{dist.get_rank()}.jsonl'), 'w') as f:
                    for data in jsonl_data:
                        if self.data_args.encode_is_query:
                            id = data['id']
                            vector = data['vector']
                            query = " ".join([" ".join([str(token)] * freq) for token, freq in vector.items()])
                            if len(query.strip()) == 0:
                                continue
                            f.write(f'{id}\t{query}\n')
                        else:
                            f.write(json.dumps(data) + "\n")
        else:
            if is_dspy:
                self.encode_counter += 1
                encode_counter = self.encode_counter
                os.makedirs(
                    f'{self.data_args.dense_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/{encode_type}/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}_{encode_counter}',
                    exist_ok=True)
                os.makedirs(
                    f'{self.data_args.sparse_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/{encode_type}/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}_{encode_counter}',
                    exist_ok=True)

                dense_output_dir = f'{self.data_args.dense_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/{encode_type}/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}_{encode_counter}'
                sparse_output_dir = f'{self.data_args.sparse_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/{encode_type}/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}_{encode_counter}'

                with open(os.path.join(
                        f'{self.data_args.dense_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/{encode_type}/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}_{encode_counter}',
                        f'query.pkl') if self.data_args.encode_is_query else os.path.join(
                    f'{self.data_args.dense_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/{encode_type}/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}_{encode_counter}',
                    f'corpus_{dist.get_rank()}.pkl'), 'wb') as f:
                    pickle.dump((encoded, lookup_indices), f)

                with open(os.path.join(
                        f'{self.data_args.sparse_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/{encode_type}/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}_{encode_counter}',
                        f'query.tsv') if self.data_args.encode_is_query else os.path.join(
                    f'{self.data_args.sparse_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/{encode_type}/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}_{encode_counter}',
                    f'corpus_{dist.get_rank()}.jsonl'), 'w') as f:
                    for data in jsonl_data:
                        if self.data_args.encode_is_query:
                            id = data['id']
                            vector = data['vector']
                            query = " ".join([" ".join([str(token)] * freq) for token, freq in vector.items()])
                            if len(query.strip()) == 0:
                                continue
                            f.write(f'{id}\t{query}\n')
                        else:
                            f.write(json.dumps(data) + "\n")
            else:
                os.makedirs(
                    f'{self.data_args.dense_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/{encode_type}/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}',
                    exist_ok=True)
                os.makedirs(
                    f'{self.data_args.sparse_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/{encode_type}/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}',
                    exist_ok=True)

                dense_output_dir = f'{self.data_args.dense_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/{encode_type}/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}'
                sparse_output_dir = f'{self.data_args.sparse_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/{encode_type}/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}'

                with open(os.path.join(
                        f'{self.data_args.dense_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/{encode_type}/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}',
                        f'query.pkl') if self.data_args.encode_is_query else os.path.join(
                    f'{self.data_args.dense_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/{encode_type}/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}',
                    f'corpus_{dist.get_rank()}.pkl'), 'wb') as f:
                    pickle.dump((encoded, lookup_indices), f)

                with open(os.path.join(
                        f'{self.data_args.sparse_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/{encode_type}/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}',
                        f'query.tsv') if self.data_args.encode_is_query else os.path.join(
                    f'{self.data_args.sparse_output_dir}/{self.model_args.model_name_or_path[model_begin_indice:]}/{self.data_args.dataset_name}/{encode_type}/{filtered}/{self.model_args.calculate_type}/{self.data_args.prompt_type}/{split}/{self.data_args.num_expended_tokens}_{manual}_{self.data_args.sparse_length}_{self.data_args.sparse_value_type}_{cluster}_{self.data_args.reps_loc}_{self.model_args.eol_type}_{self.data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{self.data_args.prompt_generation_model}_{self.prompt_generation_args.demonstration_num}_{self.prompt_generation_args.dspy_strength}',
                    f'corpus_{dist.get_rank()}.jsonl'), 'w') as f:
                    for data in jsonl_data:
                        if self.data_args.encode_is_query:
                            id = data['id']
                            vector = data['vector']
                            query = " ".join([" ".join([str(token)] * freq) for token, freq in vector.items()])
                            if len(query.strip()) == 0:
                                continue
                            f.write(f'{id}\t{query}\n')
                        else:
                            f.write(json.dumps(data) + "\n")

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        if dist.get_rank() == 0:
            command = f'''
                                python -m pyserini.index.lucene --collection JsonVectorCollection \
                                --input {sparse_output_dir} \
                                --index {sparse_output_dir}/index \
                                --generator DefaultLuceneDocumentGenerator \
                                --threads 16 \
                                --impact --pretokenized
                                '''
            subprocess.run(
                command,
                shell=True,
                executable="/bin/bash",
                cwd="/root/mllm_cross_modal_retrieval",
                check=True
            )

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        return dense_output_dir, sparse_output_dir


def construct_prompt(aspects, task_type):
    if task_type == 'itr':
        constructed_text_prompts = [
            f'Summary the {aspect} in above sentence in one word: ' for aspect in aspects
        ]

        constructed_img_prompts = [
            f'Summary the {aspect} in above image in one word: ' for aspect in aspects
        ]
    elif task_type == 'tbpr':
        constructed_text_prompts = [
            f'Summary the {aspect} of person in above sentence in one word: ' for aspect in aspects
        ]

        constructed_img_prompts = [
            f'Summary the {aspect} of person in above image in one word: ' for aspect in aspects
        ]
    else:
        constructed_text_prompts = [
            f'Summary the {aspect} in above sentence in one word: ' for aspect in aspects
        ]

        constructed_img_prompts = [
            f'Summary the {aspect} in above image in one word: ' for aspect in aspects
        ]

    return constructed_text_prompts, constructed_img_prompts


def dspy_metric(example, pred, trace=None):
    '''
        dataset_name=data_args.dataset_name,
        task_type='text-based person retrieval' if training_args.task_type == 'tbpr' else "image-text retrieval",
        seed_texts=seed_text,
        eval_split="dev_small",
        training_args=training_args,
        model_args=model_args,
        data_args=data_args,
        search_args=search_args,
        prompt_generation_args=prompt_generation_args,
        val_dataset_single=val_dataset_single,
        val_dataset_full=val_dataset_full,
        val_dataloader_single=val_dataloader_single,
        val_dataloader_full=val_dataloader_full
    '''
    with torch.inference_mode():
        try:
            val_dataloader_single = example.val_dataloader_single
            val_dataloader_full = example.val_dataloader_full
            val_dataset_single = example.val_dataset_single
            val_dataset_full = example.val_dataset_full
            training_args = example.training_args
            model_args = example.model_args
            data_args = example.data_args
            search_args = example.search_args
            prompt_generation_args = example.prompt_generation_args
            retrieval_action = example.retrieval_action
            device = example.device
            aspects_prompt_list = pred.aspects

            filtered_ids = get_filtered_ids(retrieval_action.processor.tokenizer)

            if training_args.task_type == 'tbpr':
                encoded, jsonl_data, lookup_indices = retrieval_action.encode(val_dataloader_single,
                                                                              aspects_prompt_list,
                                                                              filtered_ids, 'image', device)

                dense_output_dir, sparse_output_dir = retrieval_action.generate_encode_files(encoded, jsonl_data,
                                                                                             lookup_indices, 'image',
                                                                                             'val',
                                                                                             True)

                dense_retriever, sparse_retriever, analyzer, look_up = load_candidates(dense_output_dir,
                                                                                       sparse_output_dir,
                                                                                       use_gpu=True)

                dense_run, sparse_run, best_test_fusion_run, lookup_indices, best_weight, max_val_fusion_metric = retrieval_action.search(
                    val_dataloader_full, aspects_prompt_list, filtered_ids, dense_retriever,
                    sparse_retriever, analyzer, look_up, val_dataset_full, 'val', 0.5,
                    'text', device)
            else:
                encoded, jsonl_data, lookup_indices = retrieval_action.encode(val_dataloader_single,
                                                                              aspects_prompt_list,
                                                                              filtered_ids, 'image', device)

                dense_output_dir, sparse_output_dir = retrieval_action.generate_encode_files(encoded, jsonl_data,
                                                                                             lookup_indices, 'image',
                                                                                             'val',
                                                                                             True)

                dense_retriever, sparse_retriever, analyzer, look_up = load_candidates(dense_output_dir,
                                                                                       sparse_output_dir,
                                                                                       use_gpu=True)

                dense_run, sparse_run, best_test_fusion_run, lookup_indices, best_weight, max_val_fusion_metric_2 = retrieval_action.search(
                    val_dataloader_full, aspects_prompt_list, filtered_ids, dense_retriever,
                    sparse_retriever, analyzer, look_up, val_dataset_full, 'val', 0.5,
                    'text', device)

                encoded, jsonl_data, lookup_indices = retrieval_action.encode(val_dataloader_full, aspects_prompt_list,
                                                                              filtered_ids, 'text', device)

                dense_output_dir, sparse_output_dir = retrieval_action.generate_encode_files(encoded, jsonl_data,
                                                                                             lookup_indices, 'text',
                                                                                             'val',
                                                                                             True)

                dense_retriever, sparse_retriever, analyzer, look_up = load_candidates(dense_output_dir,
                                                                                       sparse_output_dir,
                                                                                       use_gpu=True)

                dense_run, sparse_run, best_test_fusion_run, lookup_indices, best_weight, max_val_fusion_metric_1 = retrieval_action.search(
                    val_dataloader_single, aspects_prompt_list, filtered_ids, dense_retriever,
                    sparse_retriever, analyzer, look_up, val_dataset_single, 'val', 0.5,
                    'image', device)

                max_val_fusion_metric = (max_val_fusion_metric_1 + max_val_fusion_metric_2) / 2

            return max_val_fusion_metric
        except Exception:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
            print(f"\n[dspy_metric traceback][rank={rank}]", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            print(f"pred.aspects={repr(getattr(pred, 'aspects', None))}", file=sys.stderr, flush=True)
            raise


def load_candidates(passage_reps, sparse_index, use_gpu):
    from tevatron.retriever.searcher import FaissFlatSearcher
    from pyserini.search.lucene import LuceneImpactSearcher
    from pyserini.analysis import JWhiteSpaceAnalyzer

    index_files = glob.glob(os.path.join(passage_reps, 'corpus*.pkl'))
    print(index_files)
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
        print(p_reps.shape)
        dense_retriever.add(p_reps)
        look_up += p_lookup
    if dist.get_rank() == 0:
        print(len(look_up))
    if use_gpu:
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

    sparse_retriever = LuceneImpactSearcher(os.path.join(sparse_index, 'index'), None)
    analyzer = JWhiteSpaceAnalyzer()
    sparse_retriever.set_analyzer(analyzer)

    return dense_retriever, sparse_retriever, analyzer, look_up


def main():
    parser = HfArgumentParser(
        (ModelArguments, PromptRepsLLMSearchArguments, PromptRepsLLMDataArguments, TrainingArguments,
         PromptGenerationArguments))

    model_args, search_args, data_args, training_args, prompt_generation_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: PromptRepsLLMDataArguments
    training_args: TrainingArguments
    prompt_generation_args: PromptGenerationArguments

    dspy_lm = dspy.LM(
        f"openai/" + model_args.dspy_model_path,
        api_base="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        temperature=0.0,
        max_tokens=256,
    )

    dspy.configure(lm=dspy_lm)

    program = SchemaAspectProgram()

    optimizer = dspy.MIPROv2(
        metric=dspy_metric,
        auto=prompt_generation_args.dspy_strength,
    )

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

    if 'llava-hf-llava-1.5-7b-hf' in model_args.model_name_or_path:
        encoder = LlavaForConditionalGeneration.from_pretrained(model_args.model_name_or_path, device_map=device_map,
                                                                torch_dtype=torch_type)
        processor = LlavaProcessor.from_pretrained(model_args.model_name_or_path)

    elif 'Qwen2.5-VL-7B-Instruct' in model_args.model_name_or_path or 'Qwen2.5-VL-3B-Instruct' in model_args.model_name_or_path:
        encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_args.model_name_or_path,
                                                                     device_map=device_map,
                                                                     torch_dtype=torch_type)
        processor = Qwen2_5_VLProcessor.from_pretrained(model_args.model_name_or_path)

    elif 'Qwen3-VL-8B-Instruct' in model_args.model_name_or_path:
        encoder = Qwen3VLForConditionalGeneration.from_pretrained(model_args.model_name_or_path,
                                                                  device_map=device_map,
                                                                  attn_implementation="sdpa",
                                                                  torch_dtype=torch_type)
        processor = Qwen3VLProcessor.from_pretrained(model_args.model_name_or_path)
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
        conversation = [
            {

                "role": "user",
                "content": [
                    {"type": "text", "text": "\nSummary above image in one word: "},
                    {"type": "image", "image": '{}'},
                ],
            },
        ]
        if 'royokong-e5-v' not in model_args.model_name_or_path:
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            if dist.get_rank() == 0:
                print(prompt)
            input_id = processor(text=prompt,
                                 return_tensors="pt",
                                 padding=True).input_ids
            if dist.get_rank() == 0:
                print(input_id)

    if data_args.reps_loc == 'after_pad':
        processor.tokenizer.padding_side = "left"
        processor.tokenizer.padding = True

    model = MLLMRetrievalModel(encoder)
    model = model.eval()

    # 加载词表并获取过滤后的单词id，但目前尚不清楚filtered_ids是做什么的
    if 'InternVL2_5-8B' in model_args.model_name_or_path or 'InternVL2_5-4B' in model_args.model_name_or_path:
        vocab_dict = processor.get_vocab()
        filtered_ids = get_filtered_ids(processor)
    else:
        vocab_dict = processor.tokenizer.get_vocab()
        filtered_ids = get_filtered_ids(processor.tokenizer)
    vocab_dict = {v: k for k, v in vocab_dict.items()}
    print(len(vocab_dict))

    if training_args.task_type == 'cir':
        val_dataset = ComposedTextImageRetrievalDataset(data_args.dataset_name, processor, 'val',
                                                        training_args.encode_type)
    elif training_args.task_type == 'tbpr':
        val_dataset_single = TextPersonRetrievalDataset(data_args.dataset_name, processor, 'val', 'single')
        val_dataset_full = TextPersonRetrievalDataset(data_args.dataset_name, processor, 'val', 'full')
    elif training_args.task_type == 't2it':
        val_dataset = Text2ImagetextRetrievalDataset(data_args.dataset_name, processor, 'val', 'corpus')
    elif training_args.task_type == 'it2t':
        val_dataset = Imagetext2TextRetrievalDataset(data_args.dataset_name, processor, 'val', 'corpus')
    else:
        val_dataset_full = CrossModalRetrievalDataset(data_args.dataset_name, processor, 'val', 'full')
        val_dataset_single = CrossModalRetrievalDataset(data_args.dataset_name, processor, 'val', 'single')
    sampler = Data.DistributedSampler(val_dataset_single, num_replicas=dist.get_world_size(), shuffle=True,
                                      rank=dist.get_rank())
    val_dataloader_single = Data.DataLoader(dataset=val_dataset_single, sampler=sampler, pin_memory=True,
                                            batch_size=data_args.per_device_batch_size, shuffle=False)
    sampler = Data.DistributedSampler(val_dataset_full, num_replicas=dist.get_world_size(), shuffle=True,
                                      rank=dist.get_rank())
    val_dataloader_full = Data.DataLoader(dataset=val_dataset_full, sampler=sampler, pin_memory=True,
                                          batch_size=data_args.per_device_batch_size, shuffle=False)

    if training_args.task_type == 'cir':
        dataset = ComposedTextImageRetrievalDataset(data_args.dataset_name, processor, 'test',
                                                    training_args.encode_type)
    elif training_args.task_type == 'tbpr':
        dataset_single = TextPersonRetrievalDataset(data_args.dataset_name, processor, 'test', 'single')
        dataset_full = TextPersonRetrievalDataset(data_args.dataset_name, processor, 'test', 'full')
    elif training_args.task_type == 't2it':
        dataset = Text2ImagetextRetrievalDataset(data_args.dataset_name, processor, 'test', 'corpus')
    elif training_args.task_type == 'it2t':
        dataset = Imagetext2TextRetrievalDataset(data_args.dataset_name, processor, 'test', 'corpus')
    else:
        dataset_full = CrossModalRetrievalDataset(data_args.dataset_name, processor, 'test', 'full')
        dataset_single = CrossModalRetrievalDataset(data_args.dataset_name, processor, 'test', 'single')
    sampler = Data.DistributedSampler(dataset_single, num_replicas=dist.get_world_size(), shuffle=True,
                                      rank=dist.get_rank())
    test_dataloader_single = Data.DataLoader(dataset=dataset_single, sampler=sampler, pin_memory=True,
                                             batch_size=data_args.per_device_batch_size, shuffle=False)
    sampler = Data.DistributedSampler(dataset_full, num_replicas=dist.get_world_size(), shuffle=True,
                                      rank=dist.get_rank())
    test_dataloader_full = Data.DataLoader(dataset=dataset_full, sampler=sampler, pin_memory=True,
                                           batch_size=data_args.per_device_batch_size, shuffle=False)

    seed_text = """
        'You are an experienced knowledge engineer and you are modeling schemas for knowledge graph construction. '
        'Given a set of sentences, you need to give several proper words or phrases for the abstract schemas of entities, relations and events in these sentences.'
        'You must return your answer in the following format: 1. phrases1\n2.phrases2\n3.phrases3\n...'
        'You can\'t return anything other than answers.'
        'These abstract intention words should fulfill the following requirements.'
        '1. The abstract schemas phrases can well represent the entities, relations and events, and it could be the type of the entities, relations and events or the related concepts of the entities, relations and events.'
        '2. Strictly follow the provided format, do not add extra characters or words.'
        '3. Write 3 to 7 word or phrase items at the highest possible abstract level if possible.'
        '4. Do not repeat the same word and the input in the answer.'
        '5. Stop immediately if you can\'t think of any more phrases, and no explanation is needed.'
        '6. Strictly limit the sum of answers between 3 and 7 items.'
        '\n'
        '\n'
        'Input sentences:\n<sent>\n'
        """

    counter = 0
    demonstration_string = ''
    for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(val_dataloader_single),
                                                                 total=len(val_dataloader_single)):
        # print(texts)
        counter += 1

        # itr_coco_demonstration += f'{counter}. '
        demonstration_string += texts[0]
        demonstration_string += '\n'

        if counter == prompt_generation_args.demonstration_num:
            break

    seed_text = seed_text.replace('<sent>', demonstration_string)

    retrieval_action = RetrievalAction(training_args, data_args, model_args, search_args, prompt_generation_args,
                                       model, processor, vocab_dict)

    trainset = [
        dspy.Example(
            dataset_name=data_args.dataset_name,
            task_type='text-based person retrieval' if training_args.task_type == 'tbpr' else "image-text retrieval",
            seed_texts=seed_text,
            eval_split="dev_small",
            training_args=training_args,
            model_args=model_args,
            data_args=data_args,
            search_args=search_args,
            prompt_generation_args=prompt_generation_args,
            val_dataset_single=val_dataset_single,
            val_dataset_full=val_dataset_full,
            val_dataloader_single=val_dataloader_single,
            val_dataloader_full=val_dataloader_full,
            retrieval_action=retrieval_action,
            device=device
        ).with_inputs("dataset_name", "task_type", "seed_texts"),
    ]

    compiled = optimizer.compile(
        program,
        trainset=trainset,
        valset=trainset,
    )

    prediction = compiled(
        dataset_name=data_args.dataset_name,
        task_type='text-based person retrieval' if training_args.task_type == 'tbpr' else "image-text retrieval",
        seed_texts=seed_text,
    )

    aspects_prompt_list = prediction.aspects

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
        encoded, jsonl_data, lookup_indices = retrieval_action.encode(test_dataloader_single, aspects_prompt_list,
                                                                      filtered_ids, 'image', device)

        dense_output_dir, sparse_output_dir = retrieval_action.generate_encode_files(encoded, jsonl_data,
                                                                                     lookup_indices, 'image', 'test',
                                                                                     False)

        encoded, jsonl_data, lookup_indices = retrieval_action.encode(val_dataloader_single, aspects_prompt_list,
                                                                      filtered_ids, 'image', device)

        dense_output_dir, sparse_output_dir = retrieval_action.generate_encode_files(encoded, jsonl_data,
                                                                                     lookup_indices, 'image', 'val',
                                                                                     False)

    else:
        encoded, jsonl_data, lookup_indices = retrieval_action.encode(test_dataloader_single, aspects_prompt_list,
                                                                      filtered_ids, 'text', device)
        dense_output_dir, sparse_output_dir = retrieval_action.generate_encode_files(encoded, jsonl_data,
                                                                                     lookup_indices, 'text', 'test',
                                                                                     False)

        encoded, jsonl_data, lookup_indices = retrieval_action.encode(val_dataloader_single, aspects_prompt_list,
                                                                      filtered_ids, 'text', device)

        dense_output_dir, sparse_output_dir = retrieval_action.generate_encode_files(encoded, jsonl_data,
                                                                                     lookup_indices, 'text', 'val',
                                                                                     False)

        encoded, jsonl_data, lookup_indices = retrieval_action.encode(test_dataloader_full, aspects_prompt_list,
                                                                      filtered_ids, 'image', device)

        dense_output_dir, sparse_output_dir = retrieval_action.generate_encode_files(encoded, jsonl_data,
                                                                                     lookup_indices, 'image', 'test',
                                                                                     False)

        encoded, jsonl_data, lookup_indices = retrieval_action.encode(val_dataloader_full, aspects_prompt_list,
                                                                      filtered_ids, 'image', device)

        dense_output_dir, sparse_output_dir = retrieval_action.generate_encode_files(encoded, jsonl_data,
                                                                                     lookup_indices, 'image', 'val',
                                                                                     False)

    '''
    下面这里将path，dataset，dataloader都组织成列表，并且让偶数索引是val设置，紧接着的奇数索引是对应的test设置，由val设置找到best_weight，
    然后用到test上，然后进入新的val设置，这样循环。
    '''

    if training_args.task_type == 'tbpr':
        val_passage_reps = f'{data_args.dense_output_dir}/{model_args.model_name_or_path[model_begin_indice:]}/{data_args.dataset_name}/image/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/val/{data_args.tbpr_type}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{data_args.prompt_generation_model}_{prompt_generation_args.demonstration_num}_{prompt_generation_args.dspy_strength}'
        val_sparse_index = f'{data_args.sparse_output_dir}/{model_args.model_name_or_path[model_begin_indice:]}/{data_args.dataset_name}/image/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/val/{data_args.tbpr_type}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{data_args.prompt_generation_model}_{prompt_generation_args.demonstration_num}_{prompt_generation_args.dspy_strength}'
        passage_reps = f'{data_args.dense_output_dir}/{model_args.model_name_or_path[model_begin_indice:]}/{data_args.dataset_name}/image/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/test/{data_args.tbpr_type}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{data_args.prompt_generation_model}_{prompt_generation_args.demonstration_num}_{prompt_generation_args.dspy_strength}'
        sparse_index = f'{data_args.sparse_output_dir}/{model_args.model_name_or_path[model_begin_indice:]}/{data_args.dataset_name}/image/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/test/{data_args.tbpr_type}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{data_args.prompt_generation_model}_{prompt_generation_args.demonstration_num}_{prompt_generation_args.dspy_strength}'
        path_list = [{'passage_reps': val_passage_reps, 'sparse_index': val_sparse_index},
                     {'passage_reps': passage_reps, 'sparse_index': sparse_index}]
        dataset_list = [val_dataset_full, dataset_full]
        dataloader_list = [val_dataloader_full, test_dataloader_full]
        query_type_list = ['text', 'text']
    else:
        val_text_passage_reps = f'{data_args.dense_output_dir}/{model_args.model_name_or_path[model_begin_indice:]}/{data_args.dataset_name}/text/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/val/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{data_args.prompt_generation_model}_{prompt_generation_args.demonstration_num}_{prompt_generation_args.dspy_strength}'
        val_image_passage_reps = f'{data_args.dense_output_dir}/{model_args.model_name_or_path[model_begin_indice:]}/{data_args.dataset_name}/image/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/val/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{data_args.prompt_generation_model}_{prompt_generation_args.demonstration_num}_{prompt_generation_args.dspy_strength}'
        val_text_sparse_index = f'{data_args.sparse_output_dir}/{model_args.model_name_or_path[model_begin_indice:]}/{data_args.dataset_name}/text/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/val/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{data_args.prompt_generation_model}_{prompt_generation_args.demonstration_num}_{prompt_generation_args.dspy_strength}'
        val_image_sparse_index = f'{data_args.sparse_output_dir}/{model_args.model_name_or_path[model_begin_indice:]}/{data_args.dataset_name}/image/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/val/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{data_args.prompt_generation_model}_{prompt_generation_args.demonstration_num}_{prompt_generation_args.dspy_strength}'
        text_passage_reps = f'{data_args.dense_output_dir}/{model_args.model_name_or_path[model_begin_indice:]}/{data_args.dataset_name}/text/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/test/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{data_args.prompt_generation_model}_{prompt_generation_args.demonstration_num}_{prompt_generation_args.dspy_strength}'
        image_passage_reps = f'{data_args.dense_output_dir}/{model_args.model_name_or_path[model_begin_indice:]}/{data_args.dataset_name}/image/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/test/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{data_args.prompt_generation_model}_{prompt_generation_args.demonstration_num}_{prompt_generation_args.dspy_strength}'
        text_sparse_index = f'{data_args.sparse_output_dir}/{model_args.model_name_or_path[model_begin_indice:]}/{data_args.dataset_name}/text/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/test/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{data_args.prompt_generation_model}_{prompt_generation_args.demonstration_num}_{prompt_generation_args.dspy_strength}'
        image_sparse_index = f'{data_args.sparse_output_dir}/{model_args.model_name_or_path[model_begin_indice:]}/{data_args.dataset_name}/image/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/test/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{data_args.prompt_generation_model}_{prompt_generation_args.demonstration_num}_{prompt_generation_args.dspy_strength}'
        path_list = [{'passage_reps': val_text_passage_reps, 'sparse_index': val_text_sparse_index},
                     {'passage_reps': text_passage_reps, 'sparse_index': text_sparse_index},
                     {'passage_reps': val_image_passage_reps, 'sparse_index': val_image_sparse_index},
                     {'passage_reps': image_passage_reps, 'sparse_index': image_sparse_index}]
        dataset_list = [val_dataset_single, dataset_single, val_dataset_full, dataset_full]
        dataloader_list = [val_dataloader_single, test_dataloader_single, val_dataloader_full, test_dataloader_full]
        query_type_list = ['image', 'image', 'text', 'text']

    global_best_weight = 0.5

    for index, path in enumerate(path_list):
        if training_args.task_type == 'tbpr':
            os.makedirs(
                path_prefix + f'search_results/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{query_type_list[index]}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.tbpr_type}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{data_args.sparse_type}_{data_args.prompt_generation_model}_{prompt_generation_args.demonstration_num}_{prompt_generation_args.dspy_strength}',
                exist_ok=True)

            output_dir = path_prefix + f'search_results/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{query_type_list[index]}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.tbpr_type}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{data_args.sparse_type}_{data_args.prompt_generation_model}_{prompt_generation_args.demonstration_num}_{prompt_generation_args.dspy_strength}'
        else:
            os.makedirs(
                path_prefix + f'search_results/{model_args.model_name_or_path[model_begin_indice:]}/{data_args.dataset_name}/{query_type_list[index]}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{data_args.sparse_type}_{data_args.prompt_generation_model}_{prompt_generation_args.demonstration_num}_{prompt_generation_args.dspy_strength}',
                exist_ok=True)

            output_dir = path_prefix + f'search_results/{model_args.model_name_or_path[model_begin_indice:]}/{data_args.dataset_name}/{query_type_list[index]}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}_{data_args.sparse_type}_{data_args.prompt_generation_model}_{prompt_generation_args.demonstration_num}_{prompt_generation_args.dspy_strength}'

        dense_retriever, sparse_retriever, analyzer, look_up = load_candidates(path['passage_reps'],
                                                                               path['sparse_index'],
                                                                               use_gpu=True)
        if index % 2 == 0:
            dense_run, sparse_run, best_test_fusion_run, lookup_indices, best_weight, max_val_fusion_metric = retrieval_action.search(
                dataloader_list[index], aspects_prompt_list, filtered_ids, dense_retriever,
                sparse_retriever, analyzer, look_up, dataset_list[index], 'val', global_best_weight,
                query_type_list[index], device, output_dir=output_dir)
            global_best_weight = best_weight
        else:
            dense_run, sparse_run, best_test_fusion_run, lookup_indices = retrieval_action.search(
                dataloader_list[index],
                aspects_prompt_list, filtered_ids, dense_retriever, sparse_retriever, analyzer, look_up,
                dataset_list[index], 'val',
                global_best_weight, query_type_list[index], device, output_dir=output_dir)

            output_path = os.path.join(output_dir, 'best.xlsx')
            search_args.query_type = query_type_list[index]
            retrieval_action.print_metric(output_path, dataset_list[index], dense_run, sparse_run, best_test_fusion_run,
                                          look_up, lookup_indices, search_args)

    # 训练结束后添加同步屏障
    dist.barrier()

    # 确保所有进程同步退出
    if dist.get_rank() == 0:
        # 主进程最后退出
        torch.distributed.destroy_process_group()
    else:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
