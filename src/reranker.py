import torch.distributed as dist
import torch
from tqdm import tqdm
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
    mistral_first_precise_caption_prompt
from PIL import Image
import torch.nn.functional as F
from contextlib import nullcontext

class Reranker:

    def __init__(self, model, processor, data_name, query_type, text_map, img_filepath_map, img_path_map, vocab_dict):
        self.model = model
        self.rerank_run = {}
        self.query_type = query_type
        self.text_map = text_map
        self.img_filepath_map = img_filepath_map
        self.img_path_map = img_path_map
        self.data_name = data_name
        self.processor = processor
        self.vocab_dict = vocab_dict

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