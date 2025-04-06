from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import nn, Tensor

from transformers import PreTrainedModel, AutoModel, AutoModelForCausalLM, LlamaForCausalLM
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoProcessor

from transformers.file_utils import ModelOutput
from template import text_prompt, img_prompt, img_prompt_no_one_word, text_prompt_no_one_word, \
    text_prompt_no_special_llava_v1_5, text_prompt_qwen_v2_5, text_prompt_intern_vl_v2_5, img_prompt_intern_vl_v2_5
import torch.nn.functional as F

import logging


class MLLMRetrievalModel(nn.Module):
    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 ):
        super().__init__()
        self.config = encoder.config
        self.encoder = encoder
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        # self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    # 这个函数中，input是输入的数据，input_type为输入的类型，指定输入是text还是image, transform是为了提供转换的函数, device
    def encode_data(self, input, input_type, processor, device, model_args, data_args):
        '''

        :param input: 输入的数据
        :param input_type: 输入的类型
        :param processor: 提供转换的函数
        :param device: 指定数据所在的硬件设备
        :return:
        '''
        if 'llava-hf-llava-1.5-7b-hf' in model_args.model_name_or_path or 'llava-hf-llava-v1.6-vicuna-7b-hf' in model_args.model_name_or_path:
            prompt = text_prompt_no_special_llava_v1_5
        elif 'Qwen2.5-VL-7B-Instruct' in model_args.model_name_or_path or 'Qwen2.5-VL-3B-Instruct' in model_args.model_name_or_path:
            prompt = text_prompt_qwen_v2_5
            prompt = processor.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        elif 'InternVL2_5-8B' in model_args.model_name_or_path or 'InternVL2_5-4B' in model_args.model_name_or_path:
            prompt = text_prompt_intern_vl_v2_5
            prompt = processor.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = text_prompt
        if input_type == 'text':
            if 'InternVL2_5-8B' in model_args.model_name_or_path or 'InternVL2_5-4B' in model_args.model_name_or_path:
                text_inputs = processor([prompt.replace('<sent>', text) for text in input], return_tensors='pt', padding=True)
                input_ids = text_inputs['input_ids'].to(device)
                attention_mask = text_inputs['attention_mask'].to(device)
                output = self.encoder.encode(processor, None, input_ids, attention_mask)
                if data_args.reps_loc == 'after_pad':
                    logits, embs = output.logits[:, -1, :], output.hidden_states[-1][:, -1, :]
                else:
                    # logits, embs = output.logits[:, -1, :], output.hidden_states[-1][:, -1, :]
                    logits = output.logits
                    # 由于每个批次数据长度不一定相同，为了批处理会有[pad]填充，这里是类似生成任务取next_token，因此不太好直接用最后一个logit和embedding结果，
                    # 所以使用注意力判断每个样本长度，然后把对应的logit和embedding取出来，这样才能排除[pad]的影响
                    sequence_lengths = text_inputs['attention_mask'].sum(dim=-1) - 1
                    batch_ids = torch.arange(len(text_inputs['input_ids']), device=logits.device)
                    logits, embs = output.logits[batch_ids, sequence_lengths], output.hidden_states[-1][
                        batch_ids, sequence_lengths]
                # 这里对应原文的log+relu操作
                logits = torch.log(1 + torch.relu(logits))
            else:
                text_inputs = processor(text=[prompt.replace('<sent>', text) for text in input], return_tensors="pt",
                                        padding=True).to('cuda')
                output = self.encoder(**text_inputs, output_hidden_states=True, return_dict=True)
                # print(text_inputs['input_ids'])
                # print(output.logits.shape)
                # print(output.hidden_states[-1].shape)
                if data_args.reps_loc == 'after_pad':
                    logits, embs = output.logits[:, -1, :], output.hidden_states[-1][:, -1, :]
                else:
                    # logits, embs = output.logits[:, -1, :], output.hidden_states[-1][:, -1, :]
                    logits = output.logits
                    # 由于每个批次数据长度不一定相同，为了批处理会有[pad]填充，这里是类似生成任务取next_token，因此不太好直接用最后一个logit和embedding结果，
                    # 所以使用注意力判断每个样本长度，然后把对应的logit和embedding取出来，这样才能排除[pad]的影响
                    sequence_lengths = text_inputs['attention_mask'].sum(dim=-1) - 1
                    batch_ids = torch.arange(len(text_inputs['input_ids']), device=logits.device)
                    logits, embs = output.logits[batch_ids, sequence_lengths], output.hidden_states[-1][
                        batch_ids, sequence_lengths]
                # 这里对应原文的log+relu操作
                logits = torch.log(1 + torch.relu(logits))
            return logits, embs
        elif input_type == 'image':
            if 'InternVL2_5-8B' in model_args.model_name_or_path or 'InternVL2_5-4B' in model_args.model_name_or_path:
                prompt = img_prompt_intern_vl_v2_5
                prompt = processor.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
                num_patches_list = [pixel_value.size(0) for pixel_value in input]
                pixel_values = torch.cat(input, dim=0)
                queries = []
                for idx, num_patches in enumerate(num_patches_list):
                    image_tokens = '<img>' + '<IMG_CONTEXT>' * self.encoder.num_image_token * num_patches + '</img>'
                    query = prompt.replace('<image>', image_tokens, 1)
                    queries.append(query)
                model_inputs = processor(queries, return_tensors='pt', padding=True)
                input_ids = model_inputs['input_ids'].to(device)
                attention_mask = model_inputs['attention_mask'].to(device)
                output = self.encoder.encode(processor, pixel_values, input_ids, attention_mask)
                if data_args.reps_loc == 'after_pad':
                    logits, embs = output.logits[:, -1, :], output.hidden_states[-1][:, -1, :]
                else:
                    logits = output.logits
                    # 由于每个批次数据长度不一定相同，为了批处理会有[pad]填充，这里是类似生成任务取next_token，因此不太好直接用最后一个logit和embedding结果，
                    # 所以使用注意力判断每个样本长度，然后把对应的logit和embedding取出来，这样才能排除[pad]的影响
                    sequence_lengths = model_inputs['attention_mask'].sum(dim=-1) - 1
                    batch_ids = torch.arange(len(model_inputs['input_ids']), device=logits.device)
                    logits, embs = output.logits[batch_ids, sequence_lengths], output.hidden_states[-1][
                        batch_ids, sequence_lengths]
                # 这里对应原文的log+relu操作
                logits = torch.log(1 + torch.relu(logits))
            else:
                length = len(input.pixel_values)
                # print('length is ', length)
                for key in input.keys():
                    input[key] = input[key].squeeze()  # 数据集读取的时候，是直接多了一个维度计数，因此会有一个维度是1，把这个维度去掉
                    # print(input[key].shape)
                if length == 1:
                    for key in input.keys():
                        input[key] = input[key].unsqueeze(0)  # 如果批次中数据只有1个，那么上面的操作同时将batch_size维度去掉了，这里是补充回来
                        # print(input[key].shape)
                output = self.encoder(**input, output_hidden_states=True, return_dict=True)
                if data_args.reps_loc == 'after_pad':
                    logits, embs = output.logits[:, -1, :], output.hidden_states[-1][:, -1, :]
                else:
                    logits = output.logits
                    # 由于每个批次数据长度不一定相同，为了批处理会有[pad]填充，这里是类似生成任务取next_token，因此不太好直接用最后一个logit和embedding结果，
                    # 所以使用注意力判断每个样本长度，然后把对应的logit和embedding取出来，这样才能排除[pad]的影响
                    sequence_lengths = input['attention_mask'].sum(dim=-1) - 1
                    batch_ids = torch.arange(len(input['input_ids']), device=logits.device)
                    logits, embs = output.logits[batch_ids, sequence_lengths], output.hidden_states[-1][
                        batch_ids, sequence_lengths]
                # 这里对应原文的log+relu操作
                logits = torch.log(1 + torch.relu(logits))
            return logits, embs
        else:
            return ValueError('Parameter input_type must be text or image, but the input is not either of them.')

    def compute_similarity(self, embs_1, embs_2):
        embs_1 = F.normalize(embs_1, dim=-1)
        embs_2 = F.normalize(embs_2, dim=-1)
        return embs_1 @ embs_2.t()

    # load方法，我对这个设计的理解是根据model_name_or_path来决定是什么模型，然后直接本类别赋值给encoder，也就是说，
    # 后面编码用的模型都是encoder，好处是比较简短，坏处是不太能直观的看到是哪个模型
    @classmethod
    def load(cls,
             model_name_or_path: str,
             pooling: str = 'cls',
             normalize: bool = False,
             lora_name_or_path: str = None,
             **hf_kwargs):
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if lora_name_or_path:
            lora_config = LoraConfig.from_pretrained(lora_name_or_path, **hf_kwargs)
            lora_model = PeftModel.from_pretrained(base_model, lora_name_or_path, config=lora_config)
            lora_model = lora_model.merge_and_unload()
            model = cls(
                encoder=lora_model,
                pooling=pooling,
                normalize=normalize
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=pooling,
                normalize=normalize
            )
        return model

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)
