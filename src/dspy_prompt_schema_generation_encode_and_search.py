import os
from contextlib import nullcontext
import string

import select
import torch
import torch.distributed as dist
import torch.utils.data as Data
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
)
from transformers import (LlamaForCausalLM, MistralForCausalLM, LlamaTokenizer, AutoTokenizer, LlavaProcessor,
                          LlavaForConditionalGeneration, LlavaNextProcessor, \
                          LlavaNextForConditionalGeneration, Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration,
                          AutoModel, \
                          AutoProcessor, Qwen3VLProcessor, Qwen3VLForConditionalGeneration)

from arguments import PromptRepsLLMDataArguments, ModelArguments
from arguments import TrainingArguments, PromptGenerationArguments
from dataset import CrossModalRetrievalDataset, TextPersonRetrievalDataset, ComposedTextImageRetrievalDataset, \
    Text2ImagetextRetrievalDataset, Imagetext2TextRetrievalDataset
from template import (prompt_schema_generation_text_prompt, prompt_schema_generation_text_prompt_1, \
                      mistral_prompt_schema_generation_text_prompt, mistral_prompt_schema_generation_text_prompt_1, \
                      prompt_schema_generation_text_prompt_2, mistral_prompt_schema_generation_text_prompt_2,
                      tbpr_five_aspects, \
                      itr_five_aspects, llava_mistral_template_image_prefix, llava_mistral_template_content_element,
                      img_prompt_for_concat,
                      llama3_template_image_prefix, llama3_template_content_element, llava_mistral_template_text_prefix,
                      text_prompt_for_concat, llama3_template_text_prefix)
import dspy
from PIL import Image
import torch.nn.functional as F
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np


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


class RetrievalAction():
    def __init__(self, training_args, data_args, model_args, search_args, model, processor, vocab_dict):
        super().__init__()
        self.training_args = training_args
        self.data_args = data_args
        self.model_args = model_args
        self.search_args = search_args
        self.model = model
        self.processor = processor
        self.vocab_dict = vocab_dict

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

    def encode(self, test_dataloader, aspects_prompt_list, filtered_ids, device):
        encoded = []
        jsonl_data = []
        lookup_indices = []
        if self.training_args.task_type == 'tbpr':
            for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(test_dataloader),
                                                                         total=len(test_dataloader)):
                prompt_template = self.generate_concat_prompts(aspects_prompt_list, self.training_args.encode_type)
                raw_images = [Image.open(path).convert('RGB') for path in imgs_path]
                img_inputs = self.processor(images=raw_images, text=[prompt_template] * len(imgs_path),
                                            return_tensors="pt",
                                            padding=True)
                imgs = img_inputs.to(device)
                logits, reps = self.model.encode_data_concat_for_tbpr_dspy(imgs, 'image', self.processor, device,
                                                                           self.model_args,
                                                                           self.data_args)
                disassemble_logits = logits

                reps = F.normalize(reps, dim=-1)

                reps = reps.reshape(-1, len(aspects_prompt_list), reps.shape[1]).mean(1)
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
            for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(test_dataloader),
                                                                         total=len(test_dataloader)):
                with torch.cuda.amp.autocast() if self.training_args.fp16 else nullcontext():
                    prompt_template = self.generate_concat_prompts(aspects_prompt_list, self.training_args.encode_type)

                    if self.training_args.encode_type == 'text':
                        logits, reps = self.model.encode_data_concat_dspy(texts, prompt_template, 'text',
                                                                          self.processor, device, self.model_args,
                                                                          self.data_args)
                        disassemble_logits = logits
                    else:
                        raw_images = [Image.open(path).convert('RGB') for path in imgs_path]
                        img_inputs = self.processor(images=raw_images, text=[prompt_template] * len(imgs_path),
                                                    return_tensors="pt",
                                                    padding=True)
                        imgs = img_inputs.to(device)
                        logits, reps = self.model.encode_data_concat_dspy(imgs, prompt_template, 'image',
                                                                          self.processor, device, self.model_args,
                                                                          self.data_args)
                        disassemble_logits = logits

                    reps = F.normalize(reps, dim=-1)
                    reps = reps.reshape(-1, len(aspects_prompt_list), reps.shape[1]).mean(1)
                    if self.training_args.encode_type == 'text':
                        lookup_indices.extend(text_ids)
                    else:
                        lookup_indices.extend(img_ids)
                    encoded.append(reps.cpu().detach().float().numpy())
                    ids = text_ids if self.training_args.encode_type == 'text' else img_ids
                    if self.training_args.encode_type == 'text':
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

    def search(self, test_dataloader, aspects_prompt_list, filtered_ids, dense_retriever, sparse_retriever, look_up, device):
        dense_run = {}
        sparse_run = {}
        fusion_run = [{}] * 9
        lookup_indices = []

        if self.training_args.task_type == 'tbpr':
            with torch.no_grad(), torch.cuda.amp.autocast() if self.training_args.fp16 else nullcontext():
                for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(test_dataloader),
                                                                             total=len(test_dataloader)):
                    lookup_indices.extend(text_ids)
                    query_logits, query_dense_reps = self.model.encode_data_concat_for_tbpr(texts, 'text', self.processor, device,
                                                                                       self.model_args, self.data_args)
                    disassemble_logits = query_logits

                    batch_ids = text_ids

                    query_dense_reps = F.normalize(query_dense_reps, dim=-1)
                    if model_args.eol_type == 'all_disassembleeol' or model_args.eol_type == 'all_disassembleeol_origin_text' or model_args.eol_type == 'all_disassembleeol_concrete' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text':
                        prompt_length = 5
                        query_dense_reps = query_dense_reps.reshape(-1, prompt_length,
                                                                    query_dense_reps.shape[1]).mean(1)
                    query_dense_reps = query_dense_reps.cpu().detach().float().numpy()
                    dense_scores, dense_rankings = search_queries(dense_retriever, query_dense_reps, look_up,
                                                                  self.search_args)
                    dense_run.update(
                        get_run_dict(batch_ids, dense_scores, dense_rankings, self.search_args.remove_query))
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
                        query_logits, query_dense_reps = self.model.encode_data_concat_dspy(texts, 'text',
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
                        query_logits, query_dense_reps = self.model.encode_data_concat(imgs, 'image', self.processor,
                                                                                       device,
                                                                                       self.model_args,
                                                                                       self.data_args)
                        disassemble_logits = query_logits

                    if self.search_args.query_type == 'text':
                        batch_ids = text_ids
                    else:
                        batch_ids = img_ids

                    query_dense_reps = F.normalize(query_dense_reps, dim=-1)
                    query_dense_reps = query_dense_reps.reshape(-1, len(aspects_prompt_list),
                                                                query_dense_reps.shape[1]).mean(1)

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


def dspy_metric(pred, trace=None):
    pass


def main():
    parser = HfArgumentParser(
        (ModelArguments, PromptRepsLLMDataArguments, TrainingArguments, PromptGenerationArguments))

    model_args, data_args, training_args, prompt_generation_args = parser.parse_args_into_dataclasses()
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
        auto="light",
    )

    trainset = [
        dspy.Example(
            dataset_name="flickr",
            task_type="itr",
            seed_texts="sentence 1\nsentence 2\n...",
            eval_split="dev_small",
        ).with_inputs("dataset_name", "task_type", "seed_texts"),
    ]

    compiled = optimizer.compile(
        program,
        trainset=trainset,
        valset=devset,
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

    # 指定模型
    if 'Meta-Llama-3-8B-Instruct' in model_args.dspy_model_path:
        dspy_model = LlamaForCausalLM.from_pretrained(model_args.dspy_model_path,
                                                      device_map=device_map, torch_dtype=torch_type)
        tokenizer = AutoTokenizer.from_pretrained(model_args.dspy_model_path)
    elif 'Mistral-7B-Instruct-v0.3' in model_args.dspy_model_path:
        dspy_model = MistralForCausalLM.from_pretrained(model_args.dspy_model_path,
                                                        device_map=device_map, torch_dtype=torch_type)
        tokenizer = AutoTokenizer.from_pretrained(model_args.dspy_model_path)
    else:
        dspy_model = LlamaForCausalLM.from_pretrained(model_args.dspy_model_path,
                                                      device_map=device_map, torch_dtype=torch_type)
        tokenizer = LlamaTokenizer.from_pretrained(model_args.dspy_model_path)

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
        tokenizer.padding_side = "left"
        tokenizer.padding = True

    # 加载词表并获取过滤后的单词id，但目前尚不清楚filtered_ids是做什么的
    if 'InternVL2_5-8B' in model_args.model_name_or_path or 'InternVL2_5-4B' in model_args.model_name_or_path:
        vocab_dict = processor.get_vocab()
        filtered_ids = get_filtered_ids(processor)
    else:
        vocab_dict = processor.tokenizer.get_vocab()
        filtered_ids = get_filtered_ids(processor.tokenizer)
    vocab_dict = {v: k for k, v in vocab_dict.items()}
    print(len(vocab_dict))

    with torch.no_grad():
        if prompt_generation_args.prompt_generation_type == 'prompt_schema':
            if 'Mistral-7B-Instruct-v0.3' in model_args.model_name_or_path:
                prompt = mistral_prompt_schema_generation_text_prompt
            else:
                prompt = prompt_schema_generation_text_prompt
        elif prompt_generation_args.prompt_generation_type == 'prompt_schema_1':
            if 'Mistral-7B-Instruct-v0.3' in model_args.model_name_or_path:
                prompt = mistral_prompt_schema_generation_text_prompt_1
            else:
                prompt = prompt_schema_generation_text_prompt_1
        else:
            if 'Mistral-7B-Instruct-v0.3' in model_args.model_name_or_path:
                prompt = mistral_prompt_schema_generation_text_prompt_2
            else:
                prompt = prompt_schema_generation_text_prompt_2


if __name__ == "__main__":
    main()
