import torch.distributed as dist
import torch
from tqdm import tqdm
import random
import pandas as pd
from template import relevant_prompt, in_one_word_relevant_prompt, text_query_relevant_prompt, \
    image_query_relevant_prompt, precise_caption_prompt, please_relevant_prompt, old_text_query_relevant_prompt, \
    old_image_query_relevant_prompt, origin_old_text_query_relevant_prompt, origin_old_image_query_relevant_prompt, \
    role_relevant_prompt, role_precise_caption_prompt, role_old_image_query_relevant_prompt, \
    role_old_text_query_relevant_prompt, first_precise_caption_prompt, mistral_relevant_prompt, \
    mistral_in_one_word_relevant_prompt, mistral_text_query_relevant_prompt, mistral_image_query_relevant_prompt, \
    mistral_precise_caption_prompt, mistral_please_relevant_prompt, mistral_old_text_query_relevant_prompt, \
    mistral_old_image_query_relevant_prompt, mistral_origin_old_text_query_relevant_prompt, \
    mistral_origin_old_image_query_relevant_prompt, mistral_role_relevant_prompt, mistral_role_precise_caption_prompt, \
    mistral_role_old_text_query_relevant_prompt, mistral_role_old_image_query_relevant_prompt, \
    mistral_first_precise_caption_prompt, mistral_query_generation_paradigm_prompt, query_generation_paradigm_prompt, \
    mistral_query_generation_paradigm_prompt_1, query_generation_paradigm_prompt_1, \
    detailed_mistral_query_generation_paradigm_prompt, detailed_query_generation_paradigm_prompt, \
    detailed_query_generation_paradigm_prompt_1, detailed_mistral_query_generation_paradigm_prompt_1, \
    mistral_query_generation_paradigm_prompt_5, mistral_query_generation_paradigm_prompt_4, \
    query_generation_paradigm_prompt_4, query_generation_paradigm_prompt_5

from PIL import Image
import torch.nn.functional as F
from contextlib import nullcontext

from torch.utils.data import DataLoader

flickr_length_dict = {3: 3, 4: 5, 5: 26, 6: 83, 7: 196, 8: 316, 9: 376, 10: 447, 11: 446, 12: 455, 13: 399, 14: 403,
                          15: 343, 16: 287, 17: 213, 18: 179, 19: 134, 20: 127, 21: 82, 22: 78, 23: 83, 24: 45, 25: 40,
                          26: 40, 27: 27, 28: 27, 29: 30, 30: 20, 31: 16, 32: 8, 33: 14, 34: 3, 35: 7, 36: 9, 37: 2,
                          38: 4, 39: 3, 40: 3, 41: 3, 42: 1, 43: 2, 44: 1, 45: 2, 46: 2, 47: 1, 48: 1, 52: 1, 54: 1,
                          56: 1, 57: 1, 58: 2, 64: 1, 85: 1}

coco_length_dict = {7: 2, 8: 691, 9: 2878, 10: 4461, 11: 4937, 12: 4122, 13: 2872, 14: 1815, 15: 1183, 16: 690,
                        17: 445, 18: 298, 19: 183, 20: 118, 21: 85, 22: 48, 23: 35, 24: 35, 25: 26, 26: 21, 27: 15,
                        28: 3, 29: 10, 30: 4, 31: 6, 32: 6, 33: 1, 34: 4, 36: 2, 37: 3, 39: 1, 42: 3, 45: 1, 47: 1,
                        49: 1, 50: 3, 54: 1}

flickr_length_list_20 = [(3, 4, 5), (6,), (7,), (8,), (9,), (10,), (11,), (12,), (13,), (14,), (15,), (16,), (17,),
                             (18,), (19,),
                             (20,), (21,), (22,), (23,), (24,), (25,), (26,), (27,), (28,), (29,),
                             (30,), (
                             31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 52, 54, 56, 57, 58,
                             64, 85)]

flickr_length_list_30 = [(3, 4, 5), (6,), (7,), (8,), (9,), (10,), (11,), (12,), (13,), (14,), (15,), (16,), (17,),
                             (18,), (19,),
                             (20,), (21,), (22,), (23,), (24,), (25,), (26,), (27, 28, 29),
                             (
                                 30, 31),
                             (32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 52, 54, 56, 57,
                              58, 64, 85)]

coco_length_list_20 = [(7, 8), (9,), (10,), (11,), (12,), (13,), (14,), (15,), (16,), (17,), (18,), (19,), (20,),
                           (21,), (22,), (23,), (24,), (25,), (26,),
                           (27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 39, 42, 45, 47, 49, 50, 54)]

coco_length_list_30 = [(7, 8), (9,), (10,), (11,), (12,), (13,), (14,), (15,), (16,), (17,), (18,), (19,), (20,),
                           (21,), (22,), (23,), (24,),
                           (25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 39, 42, 45, 47, 49, 50, 54)]

class Reranker:

    def __init__(self, model, processor, data_name, query_type, text_map, img_filepath_map, img_path_map, vocab_dict, dataset=None):
        self.model = model
        self.rerank_run = {}
        self.query_type = query_type
        self.text_map = text_map
        self.img_filepath_map = img_filepath_map
        self.img_path_map = img_path_map
        self.data_name = data_name
        self.processor = processor
        self.vocab_dict = vocab_dict
        self.dataset = dataset

    def rerank(self, fusion_run, rerank_type, rerank_num, data_args, training_args, model_args, rerank_batch_size=1, rerank_prompt_type='relevant', log_likelihood=False):
        with torch.no_grad(), torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
            rerank_fusion_run = {}
            if 'llava-hf-llava-v1.6-mistral-7b-hf' in model_args.model_name_or_path:
                if rerank_prompt_type == 'relevant':
                    rerank_prompt_template = mistral_relevant_prompt
                elif rerank_prompt_type == 'old_relevant':
                    if self.query_type == 'image':
                        rerank_prompt_template = mistral_old_image_query_relevant_prompt
                    else:
                        rerank_prompt_template = mistral_old_text_query_relevant_prompt
                elif rerank_prompt_type == 'please_relevant':
                    rerank_prompt_template = mistral_please_relevant_prompt
                elif rerank_prompt_type == 'in_one_word_relevant':
                    rerank_prompt_template = mistral_in_one_word_relevant_prompt
                elif rerank_prompt_type == 'precise_caption':
                    rerank_prompt_template = mistral_precise_caption_prompt
                elif rerank_prompt_type == 'query_relevant':
                    if self.query_type == 'image':
                        rerank_prompt_template = mistral_image_query_relevant_prompt
                    else:
                        rerank_prompt_template = mistral_text_query_relevant_prompt
                elif rerank_prompt_type == 'origin_old_relevant':
                    if self.query_type == 'image':
                        rerank_prompt_template = mistral_origin_old_image_query_relevant_prompt
                    else:
                        rerank_prompt_template = mistral_origin_old_text_query_relevant_prompt
                elif rerank_prompt_type == 'role_relevant':
                    rerank_prompt_template = mistral_role_relevant_prompt
                elif rerank_prompt_type == 'role_precise_caption':
                    rerank_prompt_template = mistral_role_precise_caption_prompt
                elif rerank_prompt_type == 'role_old_relevant':
                    if self.query_type == 'image':
                        rerank_prompt_template = mistral_role_old_image_query_relevant_prompt
                    else:
                        rerank_prompt_template = mistral_role_old_text_query_relevant_prompt
                elif rerank_prompt_type == 'first_precise_caption':
                    rerank_prompt_template = mistral_first_precise_caption_prompt
                else:
                    rerank_prompt_template = mistral_relevant_prompt
            else:
                if rerank_prompt_type == 'relevant':
                    rerank_prompt_template = relevant_prompt
                elif rerank_prompt_type == 'old_relevant':
                    if self.query_type == 'image':
                        rerank_prompt_template = old_image_query_relevant_prompt
                    else:
                        rerank_prompt_template = old_text_query_relevant_prompt
                elif rerank_prompt_type == 'please_relevant':
                    rerank_prompt_template = please_relevant_prompt
                elif rerank_prompt_type == 'in_one_word_relevant':
                    rerank_prompt_template = in_one_word_relevant_prompt
                elif rerank_prompt_type == 'precise_caption':
                    rerank_prompt_template = precise_caption_prompt
                elif rerank_prompt_type == 'query_relevant':
                    if self.query_type == 'image':
                        rerank_prompt_template = image_query_relevant_prompt
                    else:
                        rerank_prompt_template = text_query_relevant_prompt
                elif rerank_prompt_type == 'origin_old_relevant':
                    if self.query_type == 'image':
                        rerank_prompt_template = origin_old_image_query_relevant_prompt
                    else:
                        rerank_prompt_template = origin_old_text_query_relevant_prompt
                elif rerank_prompt_type == 'role_relevant':
                    rerank_prompt_template = role_relevant_prompt
                elif rerank_prompt_type == 'role_precise_caption':
                    rerank_prompt_template = role_precise_caption_prompt
                elif rerank_prompt_type == 'role_old_relevant':
                    if self.query_type == 'image':
                        rerank_prompt_template = role_old_image_query_relevant_prompt
                    else:
                        rerank_prompt_template = role_old_text_query_relevant_prompt
                elif rerank_prompt_type == 'first_precise_caption':
                    rerank_prompt_template = first_precise_caption_prompt
                else:
                    rerank_prompt_template = relevant_prompt
            conversation = [
                {

                    "role": "user",
                    "content": [
                        {"type": "text", "text": 'For the following sentence and image, judge whether they are relevant. Output "Yes" or "No".\nSentence: <sent> Image: <image> Output: '},
                        {"type": "image"},
                    ],
                },
            ]
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            if rerank_type == 'pointwise':
                for k, v in tqdm(fusion_run.items()):
                    # k是查询的id，v是一个字典，key是候选的id，value是查询和候选的相似度
                    sorted_by_value = sorted(v.items(), key=lambda x: x[1], reverse=True)
                    candidate_pool = dict(sorted_by_value[:rerank_num])
                    rerank_run = {}
                    image_list = []
                    text_list = []
                    count = 0
                    if dist.get_rank() == 0:
                        print(k)
                        print(candidate_pool)
                    if self.query_type == 'image':
                        if self.img_filepath_map is not None:
                            img_file_path = self.img_filepath_map[k]
                            img_path = self.img_path_map[k]
                            image_path = f'./data/{self.data_name}/{img_file_path}/{img_path}'
                            raw_image = Image.open(image_path).convert('RGB')
                        else:
                            img_path = self.img_path_map[k]
                            image_path = f'./data/{self.data_name}/flickr30k-images/{img_path}'
                            raw_image = Image.open(image_path).convert('RGB')
                        # image_list = []
                        for text_id, sim_score in candidate_pool.items():
                            '''
                            count += 1
                            if count % rerank_num != 0 and count != len(candidate_pool):
                                text_list.append(relevant_prompt.replace('<sent>', text))
                            '''
                            text = self.text_map[text_id]
                            text_input = rerank_prompt_template.replace('<sent>', text)
                            inputs = self.processor(images=raw_image, text=text_input, return_tensors="pt").to(
                                self.model.device)
                            output = self.model(**inputs, output_hidden_states=True, return_dict=True)
                            if data_args.reps_loc == 'after_pad':
                                logits, embs = output.logits[:, -1, :], output.hidden_states[-1][:, -1, :]
                            else:
                                logits = output.logits
                                # 由于每个批次数据长度不一定相同，为了批处理会有[pad]填充，这里是类似生成任务取next_token，因此不太好直接用最后一个logit和embedding结果，
                                # 所以使用注意力判断每个样本长度，然后把对应的logit和embedding取出来，这样才能排除[pad]的影响
                                sequence_lengths = inputs['attention_mask'].sum(dim=-1) - 1
                                batch_ids = torch.arange(len(inputs['input_ids']), device=logits.device)
                                logits, embs = output.logits[batch_ids, sequence_lengths], output.hidden_states[-1][
                                    batch_ids, sequence_lengths]
                            yes_id = self.vocab_dict['Yes']
                            no_id = self.vocab_dict['No']
                            if log_likelihood:
                                logits = torch.log_softmax(logits, dim=-1)
                            logit_tensor = torch.cat([logits[:, yes_id].unsqueeze(0), logits[:, no_id].unsqueeze(0)],
                                                     dim=-1)
                            output_probs = F.softmax(logit_tensor, dim=1)  # 同样指定dim=1
                            yes_prob = output_probs.squeeze()[0]
                            rerank_run[text_id] = float(yes_prob)
                    else:
                        text = self.text_map[k]
                        for img_id, sim_score in candidate_pool.items():
                            if self.img_filepath_map is not None:
                                img_file_path = self.img_filepath_map[img_id]
                                img_path = self.img_path_map[img_id]
                                image_path = f'./data/{self.data_name}/{img_file_path}/{img_path}'
                                raw_image = Image.open(image_path).convert('RGB')
                            else:
                                img_path = self.img_path_map[img_id]
                                image_path = f'./data/{self.data_name}/flickr30k-images/{img_path}'
                                raw_image = Image.open(image_path).convert('RGB')
                            text_input = rerank_prompt_template.replace('<sent>', text)
                            inputs = self.processor(images=raw_image, text=text_input, return_tensors="pt").to(
                                self.model.device)
                            output = self.model(**inputs, output_hidden_states=True, return_dict=True)
                            if data_args.reps_loc == 'after_pad':
                                logits, embs = output.logits[:, -1, :], output.hidden_states[-1][:, -1, :]
                            else:
                                logits = output.logits
                                # 由于每个批次数据长度不一定相同，为了批处理会有[pad]填充，这里是类似生成任务取next_token，因此不太好直接用最后一个logit和embedding结果，
                                # 所以使用注意力判断每个样本长度，然后把对应的logit和embedding取出来，这样才能排除[pad]的影响
                                sequence_lengths = inputs['attention_mask'].sum(dim=-1) - 1
                                batch_ids = torch.arange(len(inputs['input_ids']), device=logits.device)
                                logits, embs = output.logits[batch_ids, sequence_lengths], output.hidden_states[-1][
                                    batch_ids, sequence_lengths]
                            yes_id = self.vocab_dict['Yes']
                            no_id = self.vocab_dict['No']
                            if log_likelihood:
                                logits = torch.log_softmax(logits, dim=-1)
                            logit_tensor = torch.cat([logits[:, yes_id].unsqueeze(0), logits[:, no_id].unsqueeze(0)], dim=-1)
                            output_probs = F.softmax(logit_tensor, dim=1)  # 同样指定dim=1
                            yes_prob = output_probs.squeeze()[0]
                            rerank_run[img_id] = float(yes_prob)
                    sorted_by_value_rerank_run = dict(sorted(rerank_run.items(), key=lambda x: x[1], reverse=True))
                    if dist.get_rank() == 0:
                        print(sorted_by_value_rerank_run)
                    rerank_fusion_run[k] = sorted_by_value_rerank_run

            elif rerank_type == 'listwise':
                pass

            else:
                pass

            return rerank_fusion_run

    def caption_generation_rerank(self, fusion_run, rerank_type, rerank_num, data_args, training_args, model_args, search_args, rerank_batch_size=1, rerank_prompt_type='caption_generation'):
        rerank_fusion_run = {}

        nll_sum_dict = {}

        if 'llava-hf-llava-v1.6-mistral-7b-hf' in model_args.model_name_or_path:
            if rerank_prompt_type == 'caption_generation':
                rerank_prompt_template = mistral_query_generation_paradigm_prompt
            elif rerank_prompt_type == 'what_caption_generation':
                rerank_prompt_template = mistral_query_generation_paradigm_prompt_1
            elif rerank_prompt_type == 'detailed_caption_generation':
                rerank_prompt_template = detailed_mistral_query_generation_paradigm_prompt
            elif rerank_prompt_type == 'detailed_caption_generation_1':
                rerank_prompt_template = detailed_mistral_query_generation_paradigm_prompt_1
            elif rerank_prompt_type == 'caption_generation_4':
                rerank_prompt_template = mistral_query_generation_paradigm_prompt_4
            elif rerank_prompt_type == 'caption_generation_5':
                rerank_prompt_template = mistral_query_generation_paradigm_prompt_5
            else:
                rerank_prompt_template = mistral_query_generation_paradigm_prompt
        else:
            if rerank_prompt_type == 'caption_generation':
                rerank_prompt_template = query_generation_paradigm_prompt
            elif rerank_prompt_type == 'what_caption_generation':
                rerank_prompt_template = query_generation_paradigm_prompt_1
            elif rerank_prompt_type == 'detailed_caption_generation':
                rerank_prompt_template = detailed_query_generation_paradigm_prompt
            elif rerank_prompt_type == 'detailed_caption_generation_1':
                rerank_prompt_template = detailed_query_generation_paradigm_prompt_1
            elif rerank_prompt_type == 'caption_generation_4':
                rerank_prompt_template = query_generation_paradigm_prompt_4
            elif rerank_prompt_type == 'caption_generation_5':
                rerank_prompt_template = query_generation_paradigm_prompt_5
            else:
                rerank_prompt_template = query_generation_paradigm_prompt

        choice_dataloader = DataLoader(
                self.dataset,
                batch_size=4,  # 批量大小
                shuffle=True,  # 是否打乱数据（通常训练集为True，验证/测试集为False）
                pin_memory=True  # 加速GPU数据传输（如果使用GPU）
            )

        if dist.get_rank() == 0:
            length_count_dict = {}  # 统计每个长度有多少句
            length_content_dict = {}  # 统计每个长度有哪些图文
            sharded_nll_dict = {}  # 统计每个长度的平均对数似然
            with torch.no_grad():
                for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(choice_dataloader),
                                                                             total=len(choice_dataloader)):
                    with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
                        for text, img_path, text_id, img_id in zip(texts, imgs_path, text_ids, img_ids):
                            input_id = self.processor(text=text, return_tensors="pt")['input_ids'].squeeze().tolist()[
                                       1:]
                            if len(input_id) not in length_count_dict.keys():
                                length_count_dict[len(input_id)] = 1
                            else:
                                length_count_dict[len(input_id)] += 1

                            if len(input_id) not in length_content_dict.keys():
                                length_content_dict[len(input_id)] = [(text, img_path, text_id, img_id)]
                            else:
                                length_content_dict[len(input_id)].append((text, img_path, text_id, img_id))
                print(length_content_dict)
                length_count_dict = dict(sorted(length_count_dict.items(), key=lambda item: item[0]))
                print(length_count_dict)
                if search_args.tuple_sum == 20:
                    for length_tuple in tqdm(flickr_length_list_20):
                        content_sub_set = set()
                        for length in length_tuple:
                            content_sub_set.update(length_content_dict[length])
                        selected_items = random.sample(content_sub_set, 20)
                        print(selected_items)
                        with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
                            nll_sum = 0
                            for item in selected_items:
                                text = item[0]
                                image_path = item[1]
                                raw_image = Image.open(image_path).convert('RGB')
                                text_input = rerank_prompt_template + text
                                inputs = self.processor(images=raw_image, text=text_input, return_tensors="pt").to(
                                    self.model.device)
                                labels = self.processor(text=text, return_tensors="pt")['input_ids'].squeeze().tolist()
                                max_inputs_sum = inputs['input_ids'].shape[1]
                                # 去掉label的第一个起始符
                                labels = [[-100] * (max_inputs_sum - len(labels[1:])) + labels[1:]]
                                labels_view = torch.tensor(labels).to(self.model.device)
                                output = self.model(**inputs, output_hidden_states=True, return_dict=True)
                                logits = output.logits
                                shift_logits = logits[..., :-1, :].contiguous()
                                shift_labels = labels_view[..., 1:].contiguous()
                                loss_func = torch.nn.CrossEntropyLoss(reduction='none')
                                nll = loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                                nll = nll.view(shift_labels.size())
                                # 这个为啥是sum呢？根据原论文，是要把各个token上预测结果概率的对数似然加和取平均，但这里似乎只是求了和
                                # upr的代码中，指定了每个batch_size是1，也就是每次只针对1个查询计算
                                avg_nll = torch.sum(nll, dim=1)
                                valid_tokens = (labels_view != -100).sum(dim=1).float()
                                avg_nll /= valid_tokens
                                # 目前暂时认为avg_nll的大小是[batch_size]，直接tolist后就是对应img_id的相似度
                                print(item)
                                print(avg_nll)
                                nll_sum += avg_nll
                            nll_sum /= 20
                            nll_sum_dict[length_tuple] = float(nll_sum)
                elif search_args.tuple_sum == 30:
                    for length_tuple in tqdm(flickr_length_list_30):
                        content_sub_set = set()
                        for length in length_tuple:
                            content_sub_set.update(length_content_dict[length])
                        selected_items = random.sample(content_sub_set, 30)
                        with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
                            nll_sum = 0
                            for item in selected_items:
                                text = item[0]
                                image_path = item[1]
                                raw_image = Image.open(image_path).convert('RGB')
                                text_input = rerank_prompt_template + text
                                inputs = self.processor(images=raw_image, text=text_input, return_tensors="pt").to(
                                    self.model.device)
                                labels = self.processor(text=text, return_tensors="pt")['input_ids'].squeeze().tolist()
                                max_inputs_sum = inputs['input_ids'].shape[1]
                                # 去掉label的第一个起始符
                                labels = [[-100] * (max_inputs_sum - len(labels[1:])) + labels[1:]]
                                labels_view = torch.tensor(labels).to(self.model.device)
                                output = self.model(**inputs, output_hidden_states=True, return_dict=True)
                                logits = output.logits
                                shift_logits = logits[..., :-1, :].contiguous()
                                shift_labels = labels_view[..., 1:].contiguous()
                                loss_func = torch.nn.CrossEntropyLoss(reduction='none')
                                nll = loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                                nll = nll.view(shift_labels.size())
                                # 这个为啥是sum呢？根据原论文，是要把各个token上预测结果概率的对数似然加和取平均，但这里似乎只是求了和
                                # upr的代码中，指定了每个batch_size是1，也就是每次只针对1个查询计算
                                avg_nll = torch.sum(nll, dim=1)
                                valid_tokens = (labels_view != -100).sum(dim=1).float()
                                avg_nll /= valid_tokens
                                # 目前暂时认为avg_nll的大小是[batch_size]，直接tolist后就是对应img_id的相似度
                                nll_sum += avg_nll
                            nll_sum /= 30
                            nll_sum_dict[length_tuple] = float(nll_sum)
                else:
                    for length_tuple in tqdm(flickr_length_list_20):
                        content_sub_set = set()
                        for length in length_tuple:
                            content_sub_set.update(length_content_dict[length])
                        selected_items = random.sample(content_sub_set, 20)
                        with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
                            nll_sum = 0
                            for item in selected_items:
                                text = item[0]
                                image_path = item[1]
                                raw_image = Image.open(image_path).convert('RGB')
                                text_input = rerank_prompt_template + text
                                inputs = self.processor(images=raw_image, text=text_input, return_tensors="pt").to(
                                    self.model.device)
                                labels = self.processor(text=text, return_tensors="pt")['input_ids'].squeeze().tolist()
                                max_inputs_sum = inputs['input_ids'].shape[1]
                                # 去掉label的第一个起始符
                                labels = [[-100] * (max_inputs_sum - len(labels[1:])) + labels[1:]]
                                labels_view = torch.tensor(labels).to(self.model.device)
                                output = self.model(**inputs, output_hidden_states=True, return_dict=True)
                                logits = output.logits
                                shift_logits = logits[..., :-1, :].contiguous()
                                shift_labels = labels_view[..., 1:].contiguous()
                                loss_func = torch.nn.CrossEntropyLoss(reduction='none')
                                nll = loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                                nll = nll.view(shift_labels.size())
                                # 这个为啥是sum呢？根据原论文，是要把各个token上预测结果概率的对数似然加和取平均，但这里似乎只是求了和
                                # upr的代码中，指定了每个batch_size是1，也就是每次只针对1个查询计算
                                avg_nll = torch.sum(nll, dim=1)
                                valid_tokens = (labels_view != -100).sum(dim=1).float()
                                avg_nll /= valid_tokens
                                # 目前暂时认为avg_nll的大小是[batch_size]，直接tolist后就是对应img_id的相似度
                                nll_sum += avg_nll
                            nll_sum /= 20
                            nll_sum_dict[length_tuple] = float(nll_sum)
                object_list = [nll_sum_dict]
        else:
            object_list = [None]
        dist.broadcast_object_list(object_list, src=0)
        received_nll_sum_dict = object_list[0]
        with torch.no_grad(), torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
            for k, v in tqdm(fusion_run.items()):
                # k是查询的id，v是一个字典，key是候选的id，value是查询和候选的相似度
                sorted_by_value = sorted(v.items(), key=lambda x: x[1], reverse=True)
                candidate_pool = dict(sorted_by_value[:rerank_num])
                rerank_run = {}
                if dist.get_rank() == 0:
                    print(k)
                    print(candidate_pool)
                if self.query_type == 'image':
                    if self.img_filepath_map is not None:
                        img_file_path = self.img_filepath_map[k]
                        img_path = self.img_path_map[k]
                        image_path = f'./data/{self.data_name}/{img_file_path}/{img_path}'
                        raw_image = Image.open(image_path).convert('RGB')
                    else:
                        img_path = self.img_path_map[k]
                        image_path = f'./data/{self.data_name}/flickr30k-images/{img_path}'
                        raw_image = Image.open(image_path).convert('RGB')
                    # image_list = []
                    text_id_list = []
                    sim_score_list = []
                    label_list = []
                    text_list = []

                    for text_id, sim_score in candidate_pool.items():
                        text_id_list.append(text_id)
                        text_list.append(self.text_map[text_id])
                        sim_score_list.append(sim_score_list)

                    sharded_nll_list = []

                    for indice in tqdm(range(0, len(text_id_list), rerank_batch_size)):
                        text_shard = text_list[indice: indice + rerank_batch_size]
                        text_input = [rerank_prompt_template + text for text in text_shard]
                        image_shard = [raw_image] * len(text_shard)
                        inputs = self.processor(images=image_shard, text=text_input, return_tensors="pt").to(
                            self.model.device)
                        max_inputs_sum = inputs['input_ids'].shape[1]
                        labels = [self.processor(text=text, return_tensors="pt")['input_ids'].squeeze().tolist() for text in text_shard]
                        # 去掉label的第一个起始符
                        labels = [[-100] * (max_inputs_sum - len(label[1:])) + label[1:] for label in labels]
                        labels_view = torch.tensor(labels).to(self.model.device)

                        output = self.model(**inputs, output_hidden_states=True, return_dict=True)
                        logits = output.logits
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels_view[..., 1:].contiguous()
                        loss_func = torch.nn.CrossEntropyLoss(reduction='none')
                        nll = loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        nll = nll.view(shift_labels.size())
                        avg_nll = torch.sum(nll, dim=1)

                        valid_tokens = (labels_view != -100).sum(dim=1).float()
                        avg_nll /= valid_tokens

                        sharded_nll_list.extend(avg_nll.tolist())

                    for text_id, nll in zip(text_id_list, sharded_nll_list):
                        rerank_run[text_id] = -float(nll)

                else:
                    text = self.text_map[k]
                    img_id_list = []
                    image_list = []
                    sim_score_list = []
                    for img_id, sim_score in candidate_pool.items():
                        if self.img_filepath_map is not None:
                            img_file_path = self.img_filepath_map[img_id]
                            img_path = self.img_path_map[img_id]
                            image_path = f'./data/{self.data_name}/{img_file_path}/{img_path}'
                            raw_image = Image.open(image_path).convert('RGB')
                        else:
                            img_path = self.img_path_map[img_id]
                            image_path = f'./data/{self.data_name}/flickr30k-images/{img_path}'
                            raw_image = Image.open(image_path).convert('RGB')
                        img_id_list.append(img_id)
                        image_list.append(raw_image)
                        sim_score_list.append(sim_score)

                    sharded_nll_list = []

                    for indice in tqdm(range(0, len(img_id_list), rerank_batch_size)):
                        image_shard = image_list[indice: indice + rerank_batch_size]
                        text_input = [rerank_prompt_template + text] * len(image_shard)
                        inputs = self.processor(images=image_shard, text=text_input, return_tensors="pt").to(
                            self.model.device)
                        max_inputs_sum = inputs['input_ids'].shape[1]
                        labels = [self.processor(text=text, return_tensors="pt")['input_ids'].squeeze().tolist()] * len(image_shard)
                        # 去掉label的第一个起始符
                        labels = [[-100] * (max_inputs_sum - len(label[1:])) + label[1:] for label in labels]
                        labels_view = torch.tensor(labels).to(self.model.device)
                        output = self.model(**inputs, output_hidden_states=True, return_dict=True)
                        logits = output.logits
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels_view[..., 1:].contiguous()

                        loss_func = torch.nn.CrossEntropyLoss(reduction='none')
                        nll = loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        nll = nll.view(shift_labels.size())
                        # 这个为啥是sum呢？根据原论文，是要把各个token上预测结果概率的对数似然加和取平均，但这里似乎只是求了和
                        # upr的代码中，指定了每个batch_size是1，也就是每次只针对1个查询计算
                        avg_nll = torch.sum(nll, dim=1)
                        valid_tokens = (labels_view != -100).sum(dim=1).float()
                        avg_nll /= valid_tokens
                        # 目前暂时认为avg_nll的大小是[batch_size]，直接tolist后就是对应img_id的相似度
                        sharded_nll_list.extend(avg_nll.tolist())

                    for img_id, nll in zip(img_id_list, sharded_nll_list):
                        rerank_run[img_id] = -float(nll)
                sorted_by_value_rerank_run = dict(sorted(rerank_run.items(), key=lambda x: x[1], reverse=True))
                if dist.get_rank() == 0:
                    print(sorted_by_value_rerank_run)
                rerank_fusion_run[k] = sorted_by_value_rerank_run

        return rerank_fusion_run

