import torch.distributed as dist
import torch
from tqdm import tqdm
import pandas as pd
from template import relevant_prompt
from PIL import Image
import torch.nn.functional as F

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

    def rerank(self, fusion_run, rerank_type, rerank_num, data_args):
        rerank_fusion_run = {}
        if rerank_type == 'pointwise':
            for k, v in fusion_run:
                # k是查询的id，v是一个字典，key是候选的id，value是查询和候选的相似度
                candidate_pool = v['docs'][:rerank_num]
                rerank_run = {}
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
                    for text_id, sim_score in candidate_pool:
                        text = self.text_map[text_id]
                        text_input = relevant_prompt.replace('<sent>', text)
                        inputs = self.processor(images=raw_image, text=text_input, return_tensors="pt").to(self.model.enocder.device)
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
                        logit_tensor = torch.tensor([logits[yes_id], logits[no_id]])
                        output_probs = F.softmax(logit_tensor, dim=1)  # 同样指定dim=1
                        yes_prob = output_probs[0]
                        rerank_run[text_id] = yes_prob
                else:
                    text = self.text_map[k]
                    for img_id, sim_score in candidate_pool:
                        if self.img_filepath_map is not None:
                            img_file_path = self.img_filepath_map[k]
                            img_path = self.img_path_map[k]
                            image_path = f'./data/{self.data_name}/{img_file_path}/{img_path}'
                            raw_image = Image.open(image_path).convert('RGB')
                        else:
                            img_path = self.img_path_map[k]
                            image_path = f'./data/{self.data_name}/flickr30k-images/{img_path}'
                            raw_image = Image.open(image_path).convert('RGB')
                        text_input = relevant_prompt.replace('<sent>', text)
                        inputs = self.processor(images=raw_image, text=text_input, return_tensors="pt").to(
                            self.model.enocder.device)
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
                        logit_tensor = torch.tensor([logits[yes_id], logits[no_id]])
                        output_probs = F.softmax(logit_tensor, dim=1)  # 同样指定dim=1
                        yes_prob = output_probs[0]
                        rerank_run[img_id] = yes_prob
                sorted_by_value_rerank_run = dict(sorted(rerank_run.items(), key=lambda x: x[1], reverse=True))
                rerank_fusion_run[k] = sorted_by_value_rerank_run

        elif rerank_type == 'listwise':
            pass

        else:
            pass

        return rerank_fusion_run