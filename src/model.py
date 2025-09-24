import torch
import sys
torch.set_printoptions(threshold=sys.maxsize)  # 数字根据你的张量尺寸调整
import torch.distributed as dist
from torch import nn

from transformers import AutoModelForCausalLM
from peft import LoraConfig, PeftModel

from template import text_prompt, text_prompt_no_special_llava_v1_5, text_prompt_qwen_v2_5, text_prompt_intern_vl_v2_5, \
    img_prompt_intern_vl_v2_5, \
    llama3_template, task_text_prompts_copy, llama3_retrieval_disassemble_text_prompts, \
    llama3_template_text_prefix, llama3_template_content_element, text_prompt_for_concat, \
    retrieval_disassemble_text_prompts_3_for_concat, retrieval_disassemble_text_prompts_for_concat, \
    retrieval_disassemble_text_prompts_7_for_concat
import torch.nn.functional as F


class MLLMRetrievalModel(nn.Module):
    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(self,
                 encoder: nn.Module,
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

        if 'disassembleeol' in model_args.eol_type:
            if 'llava-hf-llava-1.5-7b-hf' in model_args.model_name_or_path or 'llava-hf-llava-v1.6-vicuna-7b-hf' in model_args.model_name_or_path:
                prompts = llama3_retrieval_disassemble_text_prompts
            else:
                prompts = llama3_retrieval_disassemble_text_prompts
        else:
            prompts = llama3_retrieval_disassemble_text_prompts
        if input_type == 'text':
            if 'InternVL2_5-8B' in model_args.model_name_or_path or 'InternVL2_5-4B' in model_args.model_name_or_path:
                text_inputs = processor([prompt.replace('<sent>', text) for text in input], return_tensors='pt',
                                        padding=True)
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
                if model_args.eol_type == 'all_disassembleeol' or model_args.eol_type == 'all_disassembleeol_origin_text':
                    disassemble_text_inputs = processor(
                        text=[prompt_text.replace('<sent>', text) for text in input for prompt_text in prompts],
                        return_tensors="pt",
                        padding=True).to(device)
                    disassemble_output = self.encoder(**disassemble_text_inputs, output_hidden_states=True,
                                                      return_dict=True)
                    if data_args.reps_loc == 'after_pad':
                        disassemble_logits = disassemble_output.logits[:, -1, :]
                        embs = disassemble_output.hidden_states[-1][:, -1, :]
                    else:
                        disassemble_logits = disassemble_output.logits
                        disassemble_sequence_lengths = disassemble_text_inputs['attention_mask'].sum(dim=-1) - 1
                        disassemble_batch_ids = torch.arange(len(disassemble_text_inputs['input_ids']),
                                                             device=disassemble_logits.device)
                        disassemble_logits = disassemble_output.logits[
                            disassemble_batch_ids, disassemble_sequence_lengths]
                        embs = disassemble_output.hidden_states[-1][disassemble_batch_ids, disassemble_sequence_lengths]
                    disassemble_logits = torch.log(1 + torch.relu(disassemble_logits))
                    return disassemble_logits, embs

                if model_args.eol_type == 'all_disassembleeol_concrete' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text':
                    text_inputs = processor(text=[prompt.replace('<sent>', text) for text in input],
                                            return_tensors="pt",
                                            padding=True).to(device)
                    output = self.encoder(**text_inputs, output_hidden_states=True, return_dict=True)
                    # print(text_inputs['input_ids'])
                    # print(output.logits.shape)
                    # print(output.hidden_states[-1].shape)
                    if data_args.reps_loc == 'after_pad':
                        logits = output.logits[:, -1, :]
                    else:
                        # logits, embs = output.logits[:, -1, :], output.hidden_states[-1][:, -1, :]
                        logits = output.logits
                        # 由于每个批次数据长度不一定相同，为了批处理会有[pad]填充，这里是类似生成任务取next_token，因此不太好直接用最后一个logit和embedding结果，
                        # 所以使用注意力判断每个样本长度，然后把对应的logit和embedding取出来，这样才能排除[pad]的影响
                        sequence_lengths = text_inputs['attention_mask'].sum(dim=-1) - 1
                        batch_ids = torch.arange(len(text_inputs['input_ids']), device=logits.device)
                        logits = output.logits[batch_ids, sequence_lengths]

                    disassemble_text_inputs = processor(
                        text=[prompt_text.replace('<sent>', text) for text in input for prompt_text in prompts],
                        return_tensors="pt",
                        padding=True).to(device)
                    disassemble_output = self.encoder(**disassemble_text_inputs, output_hidden_states=True,
                                                      return_dict=True)

                    if data_args.reps_loc == 'after_pad':
                        disassemble_logits = disassemble_output.logits[:, -1, :]
                        embs = disassemble_output.hidden_states[-1][:, -1, :]
                    else:
                        disassemble_logits = disassemble_output.logits
                        disassemble_sequence_lengths = disassemble_text_inputs['attention_mask'].sum(dim=-1) - 1
                        disassemble_batch_ids = torch.arange(len(disassemble_text_inputs['input_ids']),
                                                             device=disassemble_logits.device)
                        disassemble_logits = disassemble_output.logits[
                            disassemble_batch_ids, disassemble_sequence_lengths]
                        embs = disassemble_output.hidden_states[-1][disassemble_batch_ids, disassemble_sequence_lengths]
                    disassemble_logits = torch.log(1 + torch.relu(disassemble_logits))

                    # 这里对应原文的log+relu操作
                    logits = torch.log(1 + torch.relu(logits))
                    logits = torch.cat([logits, disassemble_logits], dim=0)

                    return logits, embs

                if model_args.eol_type == 'prompteol' or model_args.eol_type == 'prompteol_same_length':
                    text_inputs = processor(text=[prompt.replace('<sent>', text) for text in input],
                                            return_tensors="pt",
                                            padding=True).to(device)
                elif 'disassembleeol' in model_args.eol_type:
                    text_inputs = processor(text=[prompt.replace('<sent>', text) for text in input],
                                            return_tensors="pt",
                                            padding=True).to(device)
                    disassemble_text_inputs = processor(
                        text=[prompt_text.replace('<sent>', text) for text in input for prompt_text in prompts],
                        return_tensors="pt",
                        padding=True).to(device)
                    disassemble_output = self.encoder(**disassemble_text_inputs, output_hidden_states=True,
                                                      return_dict=True)
                    if data_args.reps_loc == 'after_pad':
                        disassemble_logits = disassemble_output.logits[:, -1, :]
                    else:
                        disassemble_logits = disassemble_output.logits
                        disassemble_sequence_lengths = disassemble_text_inputs['attention_mask'].sum(dim=-1) - 1
                        disassemble_batch_ids = torch.arange(len(disassemble_text_inputs['input_ids']),
                                                             device=disassemble_logits.device)
                        disassemble_logits = disassemble_output.logits[
                            disassemble_batch_ids, disassemble_sequence_lengths]
                    disassemble_logits = torch.log(1 + torch.relu(disassemble_logits))
                else:
                    prompts = [llama3_template.format(task_text_prompt) for task_text_prompt in
                               task_text_prompts_copy]
                    # 输入text的顺序是，对于每个input中的text，按照task_text_prompts中的顺序组装成列表
                    text_inputs = processor(
                        text=[task_text_prompt.replace('<sent>', text) for text in input for task_text_prompt in
                              prompts],
                        return_tensors="pt",
                        padding=True).to(device)
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
                if 'disassembleeol_concrete' in model_args.eol_type:
                    logits = torch.cat([logits, disassemble_logits], dim=0)
                if 'disassembleeol_separate' in model_args.eol_type:
                    logits = disassemble_logits

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
                output = self.encoder(**input, output_hidden_states=True, return_dict=True, use_cache=True)
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

    def encode_data_for_train(self, input, input_type, processor, device, model_args, data_args):
        '''

                :param input: 输入的数据
                :param input_type: 输入的类型
                :param processor: 提供转换的函数
                :param device: 指定数据所在的硬件设备
                :return:
                '''
        if 'llava-hf-llava-1.5-7b-hf' in model_args.model_name_or_path or 'llava-hf-llava-v1.6-vicuna-7b-hf' in model_args.model_name_or_path:
            prompt = text_prompt_no_special_llava_v1_5
        else:
            prompt = text_prompt
        if input_type == 'text':
            text_inputs = processor(text=[prompt.replace('<sent>', text) for text in input], return_tensors="pt",
                                    padding=True, ).to(device)
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

    def encode_data_for_logit_information_analysis(self, input, input_type, processor, device, model_args, data_args):
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

        if 'disassembleeol' in model_args.eol_type:
            if 'llava-hf-llava-1.5-7b-hf' in model_args.model_name_or_path or 'llava-hf-llava-v1.6-vicuna-7b-hf' in model_args.model_name_or_path:
                prompts = llama3_retrieval_disassemble_text_prompts
            else:
                prompts = llama3_retrieval_disassemble_text_prompts
        else:
            prompts = llama3_retrieval_disassemble_text_prompts
        if input_type == 'text':
            if model_args.eol_type == 'all_disassembleeol' or model_args.eol_type == 'all_disassembleeol_origin_text':
                disassemble_text_inputs = processor(
                    text=[prompt_text.replace('<sent>', text) for text in input for prompt_text in prompts],
                    return_tensors="pt",
                    padding=True).to(device)
                disassemble_output = self.encoder(**disassemble_text_inputs, output_hidden_states=True,
                                                  return_dict=True)
                if data_args.reps_loc == 'after_pad':
                    disassemble_logits = disassemble_output.logits[:, -1, :]
                    embs = disassemble_output.hidden_states[-1][:, -1, :]
                else:
                    disassemble_logits = disassemble_output.logits
                    disassemble_sequence_lengths = disassemble_text_inputs['attention_mask'].sum(dim=-1) - 1
                    disassemble_batch_ids = torch.arange(len(disassemble_text_inputs['input_ids']),
                                                         device=disassemble_logits.device)
                    disassemble_logits = disassemble_output.logits[
                        disassemble_batch_ids, disassemble_sequence_lengths]
                    embs = disassemble_output.hidden_states[-1][disassemble_batch_ids, disassemble_sequence_lengths]
                raw_disassemble_logits = disassemble_logits
                disassemble_logits = torch.log(1 + torch.relu(disassemble_logits))
                return disassemble_logits, raw_disassemble_logits

            if model_args.eol_type == 'all_disassembleeol_concrete' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text':
                text_inputs = processor(text=[prompt.replace('<sent>', text) for text in input],
                                        return_tensors="pt",
                                        padding=True).to(device)
                output = self.encoder(**text_inputs, output_hidden_states=True, return_dict=True)
                # print(text_inputs['input_ids'])
                # print(output.logits.shape)
                # print(output.hidden_states[-1].shape)
                if data_args.reps_loc == 'after_pad':
                    logits = output.logits[:, -1, :]
                else:
                    # logits, embs = output.logits[:, -1, :], output.hidden_states[-1][:, -1, :]
                    logits = output.logits
                    # 由于每个批次数据长度不一定相同，为了批处理会有[pad]填充，这里是类似生成任务取next_token，因此不太好直接用最后一个logit和embedding结果，
                    # 所以使用注意力判断每个样本长度，然后把对应的logit和embedding取出来，这样才能排除[pad]的影响
                    sequence_lengths = text_inputs['attention_mask'].sum(dim=-1) - 1
                    batch_ids = torch.arange(len(text_inputs['input_ids']), device=logits.device)
                    logits = output.logits[batch_ids, sequence_lengths]

                disassemble_text_inputs = processor(
                    text=[prompt_text.replace('<sent>', text) for text in input for prompt_text in prompts],
                    return_tensors="pt",
                    padding=True).to(device)
                disassemble_output = self.encoder(**disassemble_text_inputs, output_hidden_states=True,
                                                  return_dict=True)

                if data_args.reps_loc == 'after_pad':
                    disassemble_logits = disassemble_output.logits[:, -1, :]
                    embs = disassemble_output.hidden_states[-1][:, -1, :]
                else:
                    disassemble_logits = disassemble_output.logits
                    disassemble_sequence_lengths = disassemble_text_inputs['attention_mask'].sum(dim=-1) - 1
                    disassemble_batch_ids = torch.arange(len(disassemble_text_inputs['input_ids']),
                                                         device=disassemble_logits.device)
                    disassemble_logits = disassemble_output.logits[
                        disassemble_batch_ids, disassemble_sequence_lengths]
                    embs = disassemble_output.hidden_states[-1][disassemble_batch_ids, disassemble_sequence_lengths]
                raw_disassemble_logits = disassemble_logits
                disassemble_logits = torch.log(1 + torch.relu(disassemble_logits))

                # 这里对应原文的log+relu操作
                raw_logits = torch.cat([logits, raw_disassemble_logits])
                logits = torch.log(1 + torch.relu(logits))
                logits = torch.cat([logits, disassemble_logits], dim=0)

                return logits, raw_logits

            if model_args.eol_type == 'prompteol' or model_args.eol_type == 'prompteol_same_length':
                text_inputs = processor(text=[prompt.replace('<sent>', text) for text in input],
                                        return_tensors="pt",
                                        padding=True).to(device)
            elif 'disassembleeol' in model_args.eol_type:
                text_inputs = processor(text=[prompt.replace('<sent>', text) for text in input],
                                        return_tensors="pt",
                                        padding=True).to(device)
                disassemble_text_inputs = processor(
                    text=[prompt_text.replace('<sent>', text) for text in input for prompt_text in prompts],
                    return_tensors="pt",
                    padding=True).to(device)
                disassemble_output = self.encoder(**disassemble_text_inputs, output_hidden_states=True,
                                                  return_dict=True)
                if data_args.reps_loc == 'after_pad':
                    disassemble_logits = disassemble_output.logits[:, -1, :]
                else:
                    disassemble_logits = disassemble_output.logits
                    disassemble_sequence_lengths = disassemble_text_inputs['attention_mask'].sum(dim=-1) - 1
                    disassemble_batch_ids = torch.arange(len(disassemble_text_inputs['input_ids']),
                                                         device=disassemble_logits.device)
                    disassemble_logits = disassemble_output.logits[
                        disassemble_batch_ids, disassemble_sequence_lengths]
                raw_disassemble_logits = disassemble_logits
                disassemble_logits = torch.log(1 + torch.relu(disassemble_logits))
            else:
                prompts = [llama3_template.format(task_text_prompt) for task_text_prompt in
                           task_text_prompts_copy]
                # 输入text的顺序是，对于每个input中的text，按照task_text_prompts中的顺序组装成列表
                text_inputs = processor(
                    text=[task_text_prompt.replace('<sent>', text) for text in input for task_text_prompt in
                          prompts],
                    return_tensors="pt",
                    padding=True).to(device)
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
            raw_logits = logits
            logits = torch.log(1 + torch.relu(logits))
            if 'disassembleeol_concrete' in model_args.eol_type:
                raw_logits = torch.cat([raw_logits, raw_disassemble_logits], dim=0)
                logits = torch.cat([logits, disassemble_logits], dim=0)
            if 'disassembleeol_separate' in model_args.eol_type:
                logits = disassemble_logits
                raw_logits = raw_disassemble_logits

            return logits, raw_logits
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
                raw_logits = logits
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
                output = self.encoder(**input, output_hidden_states=True, return_dict=True, use_cache=True)
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
                raw_logits = logits
                logits = torch.log(1 + torch.relu(logits))
            return logits, raw_logits
        else:
            return ValueError('Parameter input_type must be text or image, but the input is not either of them.')

    def encode_data_at_same_time(self, text_input, image_input, processor, device, model_args, data_args):
        pass

    def encode_data_for_interface(self, input, input_type, embedding_type, processor, device, model_args, data_args):
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

        if 'disassembleeol' in model_args.eol_type:
            if 'llava-hf-llava-1.5-7b-hf' in model_args.model_name_or_path or 'llava-hf-llava-v1.6-vicuna-7b-hf' in model_args.model_name_or_path:
                prompts = llama3_retrieval_disassemble_text_prompts
            else:
                prompts = llama3_retrieval_disassemble_text_prompts
        else:
            prompts = llama3_retrieval_disassemble_text_prompts
        if input_type == 'text':
            if model_args.eol_type == 'all_disassembleeol' or model_args.eol_type == 'all_disassembleeol_origin_text':
                disassemble_text_inputs = processor(
                    text=[prompt_text.replace('<sent>', text) for text in input for prompt_text in prompts],
                    return_tensors="pt",
                    padding=True).to(device)
                disassemble_output = self.encoder(**disassemble_text_inputs, output_hidden_states=True,
                                                  return_dict=True)
                if embedding_type == 'dense':
                    if data_args.reps_loc == 'after_pad':
                        embs = disassemble_output.hidden_states[-1][:, -1, :]
                    else:
                        embs = disassemble_output.hidden_states
                        sequence_lengths = disassemble_text_inputs['attention_mask'].sum(dim=-1) - 1
                        batch_ids = torch.arange(len(disassemble_text_inputs['input_ids']), device=embs.device)
                        embs = disassemble_output.hidden_states[-1][batch_ids, sequence_lengths]
                    return embs
                elif embedding_type == 'sparse':
                    if data_args.reps_loc == 'after_pad':
                        logits = disassemble_output.logits[:, -1, :]
                    else:
                        # logits, embs = output.logits[:, -1, :], output.hidden_states[-1][:, -1, :]
                        logits = disassemble_output.logits
                        # 由于每个批次数据长度不一定相同，为了批处理会有[pad]填充，这里是类似生成任务取next_token，因此不太好直接用最后一个logit和embedding结果，
                        # 所以使用注意力判断每个样本长度，然后把对应的logit和embedding取出来，这样才能排除[pad]的影响
                        sequence_lengths = disassemble_text_inputs['attention_mask'].sum(dim=-1) - 1
                        batch_ids = torch.arange(len(disassemble_text_inputs['input_ids']), device=logits.device)
                        logits = disassemble_output.logits[batch_ids, sequence_lengths]
                    # 这里对应原文的log+relu操作
                    logits = torch.log(1 + torch.relu(logits))
                    return logits
                else:
                    if data_args.reps_loc == 'after_pad':
                        logits, embs = disassemble_output.logits[:, -1, :], disassemble_output.hidden_states[-1][:, -1,
                                                                            :]
                    else:
                        # logits, embs = output.logits[:, -1, :], output.hidden_states[-1][:, -1, :]
                        logits = disassemble_output.logits
                        # 由于每个批次数据长度不一定相同，为了批处理会有[pad]填充，这里是类似生成任务取next_token，因此不太好直接用最后一个logit和embedding结果，
                        # 所以使用注意力判断每个样本长度，然后把对应的logit和embedding取出来，这样才能排除[pad]的影响
                        sequence_lengths = disassemble_text_inputs['attention_mask'].sum(dim=-1) - 1
                        batch_ids = torch.arange(len(disassemble_text_inputs['input_ids']), device=logits.device)
                        logits, embs = disassemble_output.logits[batch_ids, sequence_lengths], \
                                       disassemble_output.hidden_states[-1][
                                           batch_ids, sequence_lengths]
                    # 这里对应原文的log+relu操作
                    logits = torch.log(1 + torch.relu(logits))
                    return logits, embs
            if model_args.eol_type == 'prompteol' or model_args.eol_type == 'prompteol_same_length':
                text_inputs = processor(text=[prompt.replace('<sent>', text) for text in input],
                                        return_tensors="pt",
                                        padding=True).to(device)
            elif 'disassembleeol' in model_args.eol_type:
                text_inputs = processor(text=[prompt.replace('<sent>', text) for text in input],
                                        return_tensors="pt",
                                        padding=True).to(device)
                disassemble_text_inputs = processor(
                    text=[prompt_text.replace('<sent>', text) for text in input for prompt_text in prompts],
                    return_tensors="pt",
                    padding=True).to(device)
                disassemble_output = self.encoder(**disassemble_text_inputs, output_hidden_states=True,
                                                  return_dict=True)

                if embedding_type != 'dense':
                    if data_args.reps_loc == 'after_pad':
                        disassemble_logits = disassemble_output.logits[:, -1, :]
                    else:
                        # logits, embs = output.logits[:, -1, :], output.hidden_states[-1][:, -1, :]
                        disassemble_logits = disassemble_output.logits
                        # 由于每个批次数据长度不一定相同，为了批处理会有[pad]填充，这里是类似生成任务取next_token，因此不太好直接用最后一个logit和embedding结果，
                        # 所以使用注意力判断每个样本长度，然后把对应的logit和embedding取出来，这样才能排除[pad]的影响
                        sequence_lengths = disassemble_text_inputs['attention_mask'].sum(dim=-1) - 1
                        batch_ids = torch.arange(len(disassemble_text_inputs['input_ids']),
                                                 device=disassemble_logits.device)
                        disassemble_logits = disassemble_output.logits[batch_ids, sequence_lengths]
                    # 这里对应原文的log+relu操作
                    disassemble_logits = torch.log(1 + torch.relu(disassemble_logits))
            else:
                prompts = [llama3_template.format(task_text_prompt) for task_text_prompt in
                           task_text_prompts_copy]
                # 输入text的顺序是，对于每个input中的text，按照task_text_prompts中的顺序组装成列表
                text_inputs = processor(
                    text=[task_text_prompt.replace('<sent>', text) for text in input for task_text_prompt in
                          prompts],
                    return_tensors="pt",
                    padding=True).to(device)
            output = self.encoder(**text_inputs, output_hidden_states=True, return_dict=True)
            # print(text_inputs['input_ids'])
            # print(output.logits.shape)
            # print(output.hidden_states[-1].shape)
            if embedding_type == 'dense':
                if data_args.reps_loc == 'after_pad':
                    embs = output.hidden_states[-1][:, -1, :]
                else:
                    embs = output.hidden_states
                    sequence_lengths = text_inputs['attention_mask'].sum(dim=-1) - 1
                    batch_ids = torch.arange(len(text_inputs['input_ids']), device=embs.device)
                    embs = output.hidden_states[-1][batch_ids, sequence_lengths]

                return embs

            elif embedding_type == 'sparse':
                if 'disassembleeol_separate' in model_args.eol_type:
                    logits = disassemble_logits
                    return logits
                if data_args.reps_loc == 'after_pad':
                    logits = output.logits[:, -1, :]
                else:
                    # logits, embs = output.logits[:, -1, :], output.hidden_states[-1][:, -1, :]
                    logits = output.logits
                    # 由于每个批次数据长度不一定相同，为了批处理会有[pad]填充，这里是类似生成任务取next_token，因此不太好直接用最后一个logit和embedding结果，
                    # 所以使用注意力判断每个样本长度，然后把对应的logit和embedding取出来，这样才能排除[pad]的影响
                    sequence_lengths = text_inputs['attention_mask'].sum(dim=-1) - 1
                    batch_ids = torch.arange(len(text_inputs['input_ids']), device=logits.device)
                    logits = output.logits[batch_ids, sequence_lengths]
                # 这里对应原文的log+relu操作
                logits = torch.log(1 + torch.relu(logits))
                if 'disassembleeol_concrete' in model_args.eol_type:
                    logits = torch.cat([logits, disassemble_logits], dim=0)
                return logits
            else:
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
                if 'disassembleeol_concrete' in model_args.eol_type:
                    logits = torch.cat([logits, disassemble_logits], dim=0)
                if 'disassembleeol_separate' in model_args.eol_type:
                    logits = disassemble_logits

                return logits, embs
        elif input_type == 'image':
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
            if embedding_type == 'dense':
                if data_args.reps_loc == 'after_pad':
                    embs = output.hidden_states[-1][:, -1, :]
                else:
                    embs = output.hidden_states
                    sequence_lengths = input['attention_mask'].sum(dim=-1) - 1
                    batch_ids = torch.arange(len(input['input_ids']), device=embs.device)
                    embs = output.hidden_states[-1][batch_ids, sequence_lengths]

                return embs
            elif embedding_type == 'sparse':
                if data_args.reps_loc == 'after_pad':
                    logits = output.logits[:, -1, :]
                else:
                    # logits, embs = output.logits[:, -1, :], output.hidden_states[-1][:, -1, :]
                    logits = output.logits
                    # 由于每个批次数据长度不一定相同，为了批处理会有[pad]填充，这里是类似生成任务取next_token，因此不太好直接用最后一个logit和embedding结果，
                    # 所以使用注意力判断每个样本长度，然后把对应的logit和embedding取出来，这样才能排除[pad]的影响
                    sequence_lengths = input['attention_mask'].sum(dim=-1) - 1
                    batch_ids = torch.arange(len(input['input_ids']), device=logits.device)
                    logits = output.logits[batch_ids, sequence_lengths]
                # 这里对应原文的log+relu操作
                logits = torch.log(1 + torch.relu(logits))
                return logits
            else:
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

    def encode_data_concat(self, input, input_type, processor, device, model_args, data_args):
        prompt_template = llama3_template_text_prefix
        if data_args.prompt_type == 'prompt_5':
            if 'concrete' in model_args.eol_type or 'all' not in model_args.eol_type:
                prompt_template += llama3_template_content_element.format(text_prompt_for_concat)
            for llama3_retrieval_disassemble_image_prompt in retrieval_disassemble_text_prompts_for_concat:
                content_element = llama3_template_content_element.format(llama3_retrieval_disassemble_image_prompt)
                prompt_template += content_element
        elif data_args.prompt_type == 'prompt_3':
            if 'concrete' in model_args.eol_type or 'all' not in model_args.eol_type:
                prompt_template += llama3_template_content_element.format(text_prompt_for_concat)
            for llama3_retrieval_disassemble_image_prompt in retrieval_disassemble_text_prompts_3_for_concat:
                content_element = llama3_template_content_element.format(llama3_retrieval_disassemble_image_prompt)
                prompt_template += content_element
        elif data_args.prompt_type == 'prompt_7':
            if 'concrete' in model_args.eol_type or 'all' not in model_args.eol_type:
                prompt_template += llama3_template_content_element.format(text_prompt_for_concat)
            for llama3_retrieval_disassemble_image_prompt in retrieval_disassemble_text_prompts_7_for_concat:
                content_element = llama3_template_content_element.format(llama3_retrieval_disassemble_image_prompt)
                prompt_template += content_element
        else:
            pass
        if input_type == 'text':
            text_inputs = processor(text=[prompt_template.replace('<sent>', text) for text in input],
                                    return_tensors="pt", padding=True).to(device)
            begin_of_text_id = processor.tokenizer.get_vocab()['<|begin_of_text|>']
            end_of_text_id = processor.tokenizer.get_vocab()['<|end_of_text|>']
            begin_of_text_indices = torch.where(text_inputs['input_ids'] == torch.tensor(begin_of_text_id))
            end_of_text_indices = torch.where(text_inputs['input_ids'] == torch.tensor(end_of_text_id))
            begin_col_list = []
            for i in range(len(begin_of_text_indices[1])):
                if 'concrete' in model_args.eol_type or 'all' not in model_args.eol_type:
                    if data_args.prompt_type == 'prompt_5':
                        if i % (len(retrieval_disassemble_text_prompts_for_concat) + 2) != 0:
                            begin_col_list.append(begin_of_text_indices[1][i].item())
                    elif data_args.prompt_type == 'prompt_3':
                        if i % (len(retrieval_disassemble_text_prompts_3_for_concat) + 2) != 0:
                            begin_col_list.append(begin_of_text_indices[1][i].item())
                    elif data_args.prompt_type == 'prompt_7':
                        if i % (len(retrieval_disassemble_text_prompts_7_for_concat) + 2) != 0:
                            begin_col_list.append(begin_of_text_indices[1][i].item())
                    else:
                        if i % (len(retrieval_disassemble_text_prompts_for_concat) + 2) != 0:
                            begin_col_list.append(begin_of_text_indices[1][i].item())
                else:
                    if data_args.prompt_type == 'prompt_5':
                        if i % (len(retrieval_disassemble_text_prompts_for_concat) + 1) != 0:
                            begin_col_list.append(begin_of_text_indices[1][i].item())
                    elif data_args.prompt_type == 'prompt_3':
                        if i % (len(retrieval_disassemble_text_prompts_3_for_concat) + 1) != 0:
                            begin_col_list.append(begin_of_text_indices[1][i].item())
                    elif data_args.prompt_type == 'prompt_7':
                        if i % (len(retrieval_disassemble_text_prompts_7_for_concat) + 1) != 0:
                            begin_col_list.append(begin_of_text_indices[1][i].item())
                    else:
                        if i % (len(retrieval_disassemble_text_prompts_for_concat) + 1) != 0:
                            begin_col_list.append(begin_of_text_indices[1][i].item())
            begin_col_list = sorted(list(set(begin_col_list)))
            end_col_list = sorted(list(set(end_of_text_indices[1].tolist())))

            text_inputs_embeds = self.encoder.get_input_embeddings()(text_inputs['input_ids'])
            dtype, device = text_inputs_embeds.dtype, text_inputs_embeds.device
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (text_inputs_embeds.shape[1], text_inputs['attention_mask'].shape[-1]),
                fill_value=min_dtype, dtype=dtype, device=device
            )
            causal_mask = torch.triu(causal_mask, diagonal=1)
            edit_causal_mask = causal_mask.clone()
            start_indice = 0
            for i in range(len(list(zip(begin_col_list, end_col_list)))):
                if i == 0:
                    start_indice = begin_col_list[i]
                else:
                    current_begin_col_indice = begin_col_list[i]
                    current_end_col_indice = end_col_list[i]
                    edit_causal_mask[current_begin_col_indice:current_end_col_indice + 1,
                    start_indice:current_begin_col_indice] = 1
            edit_causal_mask = edit_causal_mask[None, None, :, :].expand(text_inputs['attention_mask'].shape[0], 1, -1, -1)
            cache_position = torch.arange(
                0, 0 + text_inputs_embeds.shape[1],
                device=text_inputs_embeds.device
            )
            causal_mask *= torch.arange(text_inputs['attention_mask'].shape[-1],
                                        device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(text_inputs['attention_mask'].shape[0], 1, -1, -1)
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = text_inputs['attention_mask'].shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + text_inputs['attention_mask'][:, None, None, :].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
            edit_causal_mask = edit_causal_mask == 1
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                edit_causal_mask, min_dtype
            )

            text_inputs['attention_mask'] = causal_mask
            '''
            with open(f'tensor_values_{dist.get_rank()}.txt', 'w') as f:
                for mask in causal_mask:
                    f.write(str(mask.squeeze()))
            '''
            output = self.encoder(**text_inputs, output_hidden_states=True, return_dict=True)
            end_col_list = (torch.tensor(end_col_list) - 1).to(device)
            batch_size = text_inputs['input_ids'].shape[0]
            if model_args.eol_type == 'all_disassembleeol_concrete' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text':
                logits = output.logits[:, end_col_list[0], :]
                disassemble_logits = output.logits[:, end_col_list[1:], :].reshape(batch_size * len(end_col_list[1:]), -1)
                logits = torch.cat([logits, disassemble_logits], dim=0)
                logits = torch.log(1 + torch.relu(logits))
                embs = output.hidden_states[-1][:, end_col_list[1:], :].reshape(batch_size * len(end_col_list[1:]), -1)
            elif model_args.eol_type == 'all_disassembleeol' or model_args.eol_type == 'all_disassembleeol_origin_text':
                logits = output.logits[:, end_col_list, :].reshape(batch_size * len(end_col_list), -1)
                logits = torch.log(1 + torch.relu(logits))
                embs = output.hidden_states[-1][:, end_col_list, :].reshape(batch_size * len(end_col_list), -1)
            elif model_args.eol_type == 'disassembleeol_concrete' or model_args.eol_type == 'disassembleeol_concrete_origin_text':
                logits = output.logits[:, end_col_list[0], :]
                disassemble_logits = output.logits[:, end_col_list[1:], :].reshape(batch_size * len(end_col_list[1:]), -1)
                logits = torch.cat([logits, disassemble_logits], dim=0)
                logits = torch.log(1 + torch.relu(logits))
                embs = output.hidden_states[-1][:, end_col_list[0], :]
            else:
                logits = output.logits[:, end_col_list[1:], :].reshape(batch_size * len(end_col_list[1:]), -1)
                logits = torch.log(1 + torch.relu(logits))
                embs = output.hidden_states[-1][:, end_col_list[0], :]
            return logits, embs
        elif input_type == 'image':
            length = len(input.pixel_values)
            # print('length is ', length)
            for key in input.keys():
                input[key] = input[key].squeeze()  # 数据集读取的时候，是直接多了一个维度计数，因此会有一个维度是1，把这个维度去掉
                # print(input[key].shape)
            if length == 1:
                for key in input.keys():
                    input[key] = input[key].unsqueeze(0)  # 如果批次中数据只有1个，那么上面的操作同时将batch_size维度去掉了，这里是补充回来
                    # print(input[key].shape)

            begin_of_text_id = processor.tokenizer.get_vocab()['<|begin_of_text|>']
            end_of_text_id = processor.tokenizer.get_vocab()['<|end_of_text|>']
            begin_of_text_indices = torch.where(input['input_ids'] == torch.tensor(begin_of_text_id))
            end_of_text_indices = torch.where(input['input_ids'] == torch.tensor(end_of_text_id))
            begin_col_list = []
            for i in range(len(begin_of_text_indices[1])):
                if 'concrete' in model_args.eol_type or 'all' not in model_args.eol_type:
                    if data_args.prompt_type == 'prompt_5':
                        if i % (len(retrieval_disassemble_text_prompts_for_concat) + 2) != 0:
                            begin_col_list.append(begin_of_text_indices[1][i].item())
                    elif data_args.prompt_type == 'prompt_3':
                        if i % (len(retrieval_disassemble_text_prompts_3_for_concat) + 2) != 0:
                            begin_col_list.append(begin_of_text_indices[1][i].item())
                    elif data_args.prompt_type == 'prompt_7':
                        if i % (len(retrieval_disassemble_text_prompts_7_for_concat) + 2) != 0:
                            begin_col_list.append(begin_of_text_indices[1][i].item())
                    else:
                        if i % (len(retrieval_disassemble_text_prompts_for_concat) + 2) != 0:
                            begin_col_list.append(begin_of_text_indices[1][i].item())
                else:
                    if data_args.prompt_type == 'prompt_5':
                        if i % (len(retrieval_disassemble_text_prompts_for_concat) + 1) != 0:
                            begin_col_list.append(begin_of_text_indices[1][i].item())
                    elif data_args.prompt_type == 'prompt_3':
                        if i % (len(retrieval_disassemble_text_prompts_3_for_concat) + 1) != 0:
                            begin_col_list.append(begin_of_text_indices[1][i].item())
                    elif data_args.prompt_type == 'prompt_7':
                        if i % (len(retrieval_disassemble_text_prompts_7_for_concat) + 1) != 0:
                            begin_col_list.append(begin_of_text_indices[1][i].item())
                    else:
                        if i % (len(retrieval_disassemble_text_prompts_for_concat) + 1) != 0:
                            begin_col_list.append(begin_of_text_indices[1][i].item())
            begin_col_list = sorted(list(set(begin_col_list)))
            end_col_list = sorted(list(set(end_of_text_indices[1].tolist())))
            img_inputs_embeds = self.encoder.get_input_embeddings()(input['input_ids'])
            dtype, device = img_inputs_embeds.dtype, img_inputs_embeds.device
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (img_inputs_embeds.shape[1], input['attention_mask'].shape[-1]),
                fill_value=min_dtype, dtype=dtype, device=device
            )
            edit_causal_mask = causal_mask.clone()
            start_indice = 0
            for i in range(len(list(zip(begin_col_list, end_col_list)))):
                if i == 0:
                    start_indice = begin_col_list[i]
                else:
                    current_begin_col_indice = begin_col_list[i]
                    current_end_col_indice = end_col_list[i]
                    edit_causal_mask[current_begin_col_indice:current_end_col_indice + 1,
                    start_indice:current_begin_col_indice] = 1

            edit_causal_mask = edit_causal_mask[None, None, :, :].expand(input['attention_mask'].shape[0], 1, -1, -1)
            cache_position = torch.arange(
                0, 0 + img_inputs_embeds.shape[1],
                device=img_inputs_embeds.device
            )
            causal_mask *= torch.arange(input['attention_mask'].shape[-1],
                                        device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input['attention_mask'].shape[0], 1, -1, -1)
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = input['attention_mask'].shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + input['attention_mask'][:, None, None, :].to(
                causal_mask.device
            )
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
            edit_causal_mask = edit_causal_mask == 1
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                edit_causal_mask, min_dtype
            )

            input['attention_mask'] = causal_mask

            output = self.encoder(**input, output_hidden_states=True, return_dict=True, use_cache=True)
            # 这里对应原文的log+relu操作
            end_col_list = (torch.tensor(end_col_list) - 1).to(device)
            batch_size = input['input_ids'].shape[0]
            if model_args.eol_type == 'all_disassembleeol_concrete' or model_args.eol_type == 'all_disassembleeol_concrete_origin_text':
                logits = output.logits[:, end_col_list[0], :]
                disassemble_logits = output.logits[:, end_col_list[1:], :].reshape(batch_size * len(end_col_list[1:]),
                                                                                   -1)
                logits = torch.cat([logits, disassemble_logits], dim=0)
                logits = torch.log(1 + torch.relu(logits))
                embs = output.hidden_states[-1][:, end_col_list[1:], :].reshape(batch_size * len(end_col_list[1:]), -1)
            elif model_args.eol_type == 'all_disassembleeol' or model_args.eol_type == 'all_disassembleeol_origin_text':
                logits = output.logits[:, end_col_list, :].reshape(batch_size * len(end_col_list), -1)
                logits = torch.log(1 + torch.relu(logits))
                embs = output.hidden_states[-1][:, end_col_list, :].reshape(batch_size * len(end_col_list), -1)
            elif model_args.eol_type == 'disassembleeol_concrete' or model_args.eol_type == 'disassembleeol_concrete_origin_text':
                logits = output.logits[:, end_col_list[0], :]
                disassemble_logits = output.logits[:, end_col_list[1:], :].reshape(batch_size * len(end_col_list[1:]),
                                                                                   -1)
                logits = torch.cat([logits, disassemble_logits], dim=0)
                logits = torch.log(1 + torch.relu(logits))
                embs = output.hidden_states[-1][:, end_col_list[0], :]
            else:
                logits = output.logits[:, end_col_list[1:], :].reshape(batch_size * len(end_col_list[1:]), -1)
                logits = torch.log(1 + torch.relu(logits))
                embs = output.hidden_states[-1][:, end_col_list[0], :]
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

    def forward(self, texts, imgs, processor, device, model_args, data_args):
        text_logits, text_reps = self.encode_data(texts, 'text', processor, device, model_args,
                                                  data_args)

        img_logits, img_reps = self.encode_data(imgs, 'image', processor, device, model_args,
                                                data_args)

        return {'text_reps': text_reps, 'img_reps': img_reps}
