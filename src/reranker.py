from contextlib import nullcontext
from pathlib import Path
import time

import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

from rerank_batching import (
    RerankRequest, configure_rerank_processor, indexed_batches, last_valid_logits,
    mean_target_nll, move_batch, target_labels, validate_rerank_batch_size,
)
from rerank_prior_cache import (
    candidate_key, load_or_build_prior_cache, prior_metadata, validate_prior_cache_options,
)


def _is_main_process() -> bool:
    """Treat an uninitialized distributed runtime as a single rank-0 process."""
    return (
        not (dist.is_available() and dist.is_initialized())
        or dist.get_rank() == 0
    )


def _stage3_query_progress(items, progress_observer=None):
    if progress_observer is not None:
        return progress_observer.progress(
            items, total=len(items), total_items=len(items),
            item_count=lambda item: 1, progress_factory=tqdm,
            desc="[stage3] rerank queries (time estimate)", unit="query",
        )
    return tqdm(
        items,
        desc="[stage3] rerank queries",
        unit="query",
        dynamic_ncols=True,
    )


def _stage3_candidate_progress(items):
    return tqdm(
        items,
        desc="[stage3] candidate batches",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )


from template import (relevant_prompt, in_one_word_relevant_prompt, text_query_relevant_prompt, \
                      image_query_relevant_prompt, precise_caption_prompt, please_relevant_prompt,
                      old_text_query_relevant_prompt, \
                      old_image_query_relevant_prompt, origin_old_text_query_relevant_prompt,
                      origin_old_image_query_relevant_prompt, \
                      role_relevant_prompt, role_precise_caption_prompt, role_old_image_query_relevant_prompt, \
                      role_old_text_query_relevant_prompt, first_precise_caption_prompt, mistral_relevant_prompt, \
                      mistral_in_one_word_relevant_prompt, mistral_text_query_relevant_prompt,
                      mistral_image_query_relevant_prompt, \
                      mistral_precise_caption_prompt, mistral_please_relevant_prompt,
                      mistral_old_text_query_relevant_prompt, \
                      mistral_old_image_query_relevant_prompt, mistral_origin_old_text_query_relevant_prompt, \
                      mistral_origin_old_image_query_relevant_prompt, mistral_role_relevant_prompt,
                      mistral_role_precise_caption_prompt, \
                      mistral_role_old_text_query_relevant_prompt, mistral_role_old_image_query_relevant_prompt, \
                      mistral_first_precise_caption_prompt, mistral_query_generation_paradigm_prompt,
                      query_generation_paradigm_prompt, \
                      mistral_query_generation_paradigm_prompt_1, query_generation_paradigm_prompt_1, \
                      detailed_mistral_query_generation_paradigm_prompt, detailed_query_generation_paradigm_prompt, \
                      detailed_query_generation_paradigm_prompt_1, detailed_mistral_query_generation_paradigm_prompt_1, \
                      mistral_query_generation_paradigm_prompt_5, mistral_query_generation_paradigm_prompt_4, \
                      query_generation_paradigm_prompt_4, query_generation_paradigm_prompt_5,
                      query_generation_paradigm_prompt_2, \
                      query_generation_paradigm_prompt_3, mistral_query_generation_paradigm_prompt_2, \
                      mistral_query_generation_paradigm_prompt_3, query_generation_paradigm_prompt_6,
                      query_generation_paradigm_prompt_7, \
                      mistral_query_generation_paradigm_prompt_6, mistral_query_generation_paradigm_prompt_7, \
                      fashion_iq_query_relevant_prompt, fashion_iq_origin_old_query_relevant_prompt,
                      fashion_iq_relevant_prompt, \
                      fashion_iq_old_query_relevant_prompt, mistral_fashion_iq_relevant_prompt, \
                      mistral_fashion_iq_origin_old_query_relevant_prompt, mistral_fashion_iq_old_query_relevant_prompt, \
                      mistral_fashion_iq_query_relevant_prompt, mistral_person_retrieval_relevant_prompt, \
                      person_retrieval_relevant_prompt, mistral_person_retrieval_old_query_relevant_prompt, \
                      person_retrieval_old_query_relevant_prompt,
                      mistral_person_retrieval_origin_old_query_relevant_prompt, \
                      person_retrieval_origin_old_query_relevant_prompt, mistral_person_retrieval_query_relevant_prompt, \
                      person_retrieval_query_relevant_prompt, person_retrieval_query_generation_paradigm_prompt, \
                      person_retrieval_mistral_query_generation_paradigm_prompt,
                      person_retrieval_query_generation_paradigm_prompt_1, \
                      person_retrieval_mistral_query_generation_paradigm_prompt_1,
                      person_retrieval_query_generation_paradigm_prompt_2, \
                      person_retrieval_mistral_query_generation_paradigm_prompt_2,
                      mistral_fashion_iq_role_old_query_relevant_prompt, \
                      mistral_fashion_iq_role_relevant_prompt, mistral_fashion_iq_in_one_word_relevant_prompt, \
                      mistral_fashion_iq_please_relevant_prompt, mistral_origin_old_image_reverse_query_relevant_prompt, \
                      mistral_origin_old_text_reverse_query_relevant_prompt,
                      origin_old_image_reverse_query_relevant_prompt, \
                      origin_old_text_reverse_query_relevant_prompt, mistral_old_text_reverse_query_relevant_prompt, \
                      mistral_old_image_reverse_query_relevant_prompt, old_text_reverse_query_relevant_prompt, \
                      old_image_reverse_query_relevant_prompt, qwen2_5_relevant_prompt,
                      qwen2_5_old_text_query_relevant_prompt, \
                      qwen2_5_old_image_query_relevant_prompt, qwen2_5_text_query_relevant_prompt,
                      qwen2_5_image_query_relevant_prompt, \
                      qwen2_5_origin_old_text_query_relevant_prompt, qwen2_5_origin_old_image_query_relevant_prompt, \
                      qwen2_5_query_generation_paradigm_prompt_1, qwen3_query_generation_paradigm_prompt_1, \
                      qwen2_5_query_generation_paradigm_prompt, qwen3_query_generation_paradigm_prompt,
                      qwen3_relevant_prompt, \
                      qwen3_old_image_query_relevant_prompt, qwen3_old_text_query_relevant_prompt, \
                      qwen3_origin_old_image_query_relevant_prompt, qwen3_origin_old_text_query_relevant_prompt, \
                      qwen3_image_query_relevant_prompt, qwen3_text_query_relevant_prompt, qwen3_please_relevant_prompt, \
                      qwen3_in_one_word_relevant_prompt, qwen3_precise_caption_prompt, \
                      qwen3_origin_old_image_reverse_query_relevant_prompt,
                      qwen3_origin_old_text_reverse_query_relevant_prompt, \
                      qwen3_role_relevant_prompt, qwen3_role_precise_caption_prompt,
                      qwen3_role_old_image_query_relevant_prompt, \
                      qwen3_role_old_text_query_relevant_prompt, qwen3_first_precise_caption_prompt, \
                      qwen3_old_image_reverse_query_relevant_prompt, qwen3_old_text_reverse_query_relevant_prompt, \
                      qwen2_5_old_image_reverse_query_relevant_prompt, qwen2_5_old_text_reverse_query_relevant_prompt, \
                      qwen2_5_please_relevant_prompt, qwen2_5_in_one_word_relevant_prompt,
                      qwen2_5_precise_caption_prompt, \
                      qwen2_5_origin_old_image_reverse_query_relevant_prompt,
                      qwen2_5_origin_old_text_reverse_query_relevant_prompt, \
                      qwen2_5_role_relevant_prompt, qwen2_5_role_precise_caption_prompt,
                      qwen2_5_role_old_image_query_relevant_prompt, \
                      qwen2_5_role_old_text_query_relevant_prompt, qwen2_5_first_precise_caption_prompt, \
                      qwen3_person_retrieval_query_relevant_prompt, qwen3_person_retrieval_old_query_relevant_prompt, \
                      qwen3_person_retrieval_relevant_prompt, qwen3_person_retrieval_origin_old_query_relevant_prompt, \
                      person_retrieval_qwen3_query_generation_paradigm_prompt,
                      person_retrieval_qwen3_query_generation_paradigm_prompt_1, \
                      text_reverse_query_relevant_prompt, image_reverse_query_relevant_prompt,
                      mistral_text_reverse_query_relevant_prompt, \
                      mistral_image_reverse_query_relevant_prompt, person_retrieval_reverse_query_relevant_prompt, \
                      person_retrieval_reverse_old_query_relevant_prompt,
                      person_retrieval_reverse_origin_old_query_relevant_prompt, \
                      mistral_person_retrieval_reverse_old_query_relevant_prompt,
                      mistral_person_retrieval_reverse_query_relevant_prompt, \
                      mistral_person_retrieval_reverse_origin_old_query_relevant_prompt,
                      vicuna_query_generation_paradigm_prompt, \
                      vicuna_query_generation_paradigm_prompt_1,
                      person_retrieval_vicuna_query_generation_paradigm_prompt_1, \
                      person_retrieval_vicuna_query_generation_paradigm_prompt,
                      vicuna_person_retrieval_old_query_relevant_prompt, \
                      vicuna_person_retrieval_query_relevant_prompt,
                      vicuna_person_retrieval_origin_old_query_relevant_prompt, \
                      vicuna_person_retrieval_reverse_query_relevant_prompt, \
                      vicuna_person_retrieval_reverse_origin_old_query_relevant_prompt, \
                      vicuna_person_retrieval_reverse_old_query_relevant_prompt,
                      vicuna_old_image_reverse_query_relevant_prompt, \
                      vicuna_old_text_reverse_query_relevant_prompt, vicuna_old_image_query_relevant_prompt, \
                      vicuna_old_text_query_relevant_prompt, \
                      vicuna_origin_old_image_query_relevant_prompt, vicuna_origin_old_text_query_relevant_prompt, \
                      vicuna_text_query_relevant_prompt, \
                      vicuna_image_query_relevant_prompt, vicuna_image_reverse_query_relevant_prompt, \
                      vicuna_text_reverse_query_relevant_prompt, \
                      vicuna_origin_old_text_reverse_query_relevant_prompt,
                      vicuna_origin_old_image_reverse_query_relevant_prompt, \
                      vicuna_freeret_image_rerank_prompt, vicuna_freeret_text_rerank_prompt,
                      mistral_freeret_image_rerank_prompt, \
                      mistral_freeret_text_rerank_prompt, freeret_text_rerank_prompt, freeret_image_rerank_prompt, \
                      person_retrieval_llava_34b_query_generation_paradigm_prompt_1, \
                      person_retrieval_llava_34b_query_generation_paradigm_prompt, \
                      llava_34b_query_generation_paradigm_prompt, llava_34b_query_generation_paradigm_prompt_1, \
                      llava_34b_person_retrieval_query_relevant_prompt,
                      llava_34b_person_retrieval_reverse_query_relevant_prompt, \
                      llava_34b_text_query_relevant_prompt, llava_34b_image_query_relevant_prompt, \
                      llava_34b_text_reverse_query_relevant_prompt, \
                      llava_34b_image_reverse_query_relevant_prompt, t2it_retrieval_old_query_relevant_prompt, \
                      t2it_retrieval_query_relevant_prompt, t2it_retrieval_origin_old_query_relevant_prompt, \
                      mistral_t2it_retrieval_origin_old_query_relevant_prompt,
                      mistral_t2it_retrieval_old_query_relevant_prompt, \
                      mistral_t2it_retrieval_query_relevant_prompt, t2it_retrieval_query_generation_paradigm_prompt, \
                      t2it_retrieval_mistral_query_generation_paradigm_prompt,
                      t2it_retrieval_mistral_query_generation_paradigm_prompt_1, \
                      t2it_retrieval_query_generation_paradigm_prompt_1, it2t_retrieval_query_relevant_prompt, \
                      it2t_retrieval_old_query_relevant_prompt, it2t_retrieval_origin_old_query_relevant_prompt, \
                      mistral_it2t_retrieval_query_relevant_prompt, mistral_it2t_retrieval_old_query_relevant_prompt, \
                      mistral_it2t_retrieval_origin_old_query_relevant_prompt,
                      it2t_retrieval_query_generation_paradigm_prompt, \
                      it2t_retrieval_mistral_query_generation_paradigm_prompt,
                      it2t_retrieval_mistral_query_generation_paradigm_prompt_1, \
                      it2t_retrieval_query_generation_paradigm_prompt_1, lamra_2_text_relevant_prompt,
                      lamra_2_image_relevant_prompt,
                      it2t_retrieval_answer_relevant_prompt, mistral_it2t_retrieval_answer_relevant_prompt,
                      lamra_2_t2it_relevant_prompt, lamra_2_5_t2it_relevant_prompt, lamra_2_5_it2t_relevant_prompt,
                      lamra_2_it2t_relevant_prompt, lamra_2_5_image_relevant_prompt, lamra_2_5_text_relevant_prompt)
from io import BytesIO

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

flickr_length_list_100 = [(2, 3, 4), (5,), (6,), (7,), (8,), (9,), (10,), (11,), (12,), (13,), (14,), (15,), (16,),
                          (17,), (18,), (19,),
                          (20,), (21,), (22,), (23,), (24,), (25,), (26,), (27,), (28,), (29,),
                          (30,), (31,), (32,), (33,), (34,), (35,), (36,), (37,), (38,), (39, 40), (41, 42, 43),
                          (44, 45, 46, 47),
                          (48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 67, 68, 70, 71, 72, 73,
                           74, 77, 79, 85)]

coco_length_list_20 = [(7, 8), (9,), (10,), (11,), (12,), (13,), (14,), (15,), (16,), (17,), (18,), (19,), (20,),
                       (21,), (22,), (23,), (24,), (25,), (26,),
                       (27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 39, 42, 45, 47, 49, 50, 54)]

coco_length_list_30 = [(7, 8), (9,), (10,), (11,), (12,), (13,), (14,), (15,), (16,), (17,), (18,), (19,), (20,),
                       (21,), (22,), (23,), (24,),
                       (25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 39, 42, 45, 47, 49, 50, 54)]

coco_length_list_100 = [(2, 3, 4), (5,), (6,), (7,), (8,), (9,), (10,), (11,), (12,), (13,), (14,), (15,), (16,), (17,),
                        (18,), (19,),
                        (20,), (21,), (22,), (23,), (24,), (25,), (26,), (27,), (28,), (29,),
                        (30,), (31,), (32,), (33,), (34,), (35,), (36,), (37,), (38,), (39, 40), (41, 42, 43),
                        (44, 45, 46, 47),
                        (48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 67, 68, 70, 71, 72, 73, 74,
                         77, 79, 85)]

path_prefix = '/root/autodl-tmp/'


class Reranker:

    def __init__(self, model, processor, data_name, query_type, text_map, img_filepath_map, img_path_map, vocab_dict,
                 dataset=None):
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

    @staticmethod
    def _read_image(source):
        if isinstance(source, Image.Image):
            return source.convert('RGB')
        if isinstance(source, dict):
            if source.get('bytes') is not None:
                source = BytesIO(source['bytes'])
            else:
                source = source.get('path')
        elif isinstance(source, (bytes, bytearray, memoryview)):
            source = BytesIO(bytes(source))
        with Image.open(source) as image:
            return image.convert('RGB').copy()

    def _ir_image_path(self, image_id):
        base = Path(path_prefix) / 'data' / self.data_name
        if self.img_filepath_map is not None:
            return base / self.img_filepath_map[image_id] / self.img_path_map[image_id]
        return base / 'flickr30k-images' / self.img_path_map[image_id]

    def _requests(self, query_id, candidate_ids, task_type, template, prompt_type, *, caption):
        """Construct lightweight requests; decode images only for the current batch."""
        if caption and task_type == 'cir':
            raise ValueError('CIR caption-generation reranking has no implemented scoring definition')
        if task_type == 'cir':
            reference_id, text_id = query_id.split('_', 1)
            reference = Path(path_prefix) / 'data' / self.data_name / 'images/images' / self.img_path_map[reference_id]
            query_text = self.dataset.get_text(text_id)
            dress = self.dataset.get_dress_type(reference_id)
            template = template.replace('{}', 'shirt' if dress == 'toptee' else dress)
        elif task_type == 'tbpr':
            query_text = self.text_map[query_id]
        elif task_type == 't2it':
            query_text = self.dataset.get_query(query_id)
        elif task_type == 'it2t':
            query = self.dataset.get_query(query_id)
            query_text, query_image = query['text'], query['image']
        elif self.query_type == 'image':
            query_image = self._ir_image_path(query_id)
        else:
            query_text = self.text_map[query_id]

        requests = []
        for candidate_id in candidate_ids:
            baseline_prefix = None
            if task_type == 'cir':
                candidate = Path(path_prefix) / 'data' / self.data_name / 'images/images' / f'{candidate_id}.png'
                images = (candidate, reference) if prompt_type == 'origin_old_relevant' else (reference, candidate)
                text = template.replace('<sent>', query_text)
                prefix = None
            elif task_type == 'tbpr':
                images = (Path(path_prefix) / 'data' / self.data_name / 'imgs' / self.img_path_map[candidate_id],)
                prefix = template if caption else None
                text = template + query_text if caption else template.replace('<sent>', query_text)
            elif task_type == 't2it':
                candidate = self.dataset.get_candidate(candidate_id)
                images = (candidate['image'],)
                if caption:
                    prefix = template.replace('<sent>', candidate['text'])
                    text = prefix + query_text
                else:
                    prefix = None
                    text = template.replace('<sent>', candidate['text'], 1).replace('<sent>', query_text, 1)
            elif task_type == 'it2t':
                candidate_text = self.dataset.get_candidate(candidate_id)
                images = (query_image,)
                if caption:
                    prefix = template.replace('<sent>', query_text)
                    text = prefix + candidate_text
                    # Preserve the existing modify_single text-only context.
                    baseline_prefix = template
                else:
                    prefix = None
                    text = template.replace('<sent>', query_text, 1).replace('<sent>', candidate_text, 1)
            elif self.query_type == 'image':
                candidate_text = self.text_map[candidate_id]
                images = (query_image,)
                prefix = template if caption else None
                text = template + candidate_text if caption else template.replace('<sent>', candidate_text)
                baseline_prefix = template if caption else None
            else:
                images = (self._ir_image_path(candidate_id),)
                prefix = template if caption else None
                text = template + query_text if caption else template.replace('<sent>', query_text)
            if caption and not text[len(prefix):].strip():
                raise ValueError(f'Empty caption target for candidate {candidate_id!r}')
            requests.append(RerankRequest(candidate_id, text, images, prefix, baseline_prefix))
        return requests

    def _batch_images(self, requests):
        cache = {}
        groups = []
        for request in requests:
            group = []
            for source in request.images:
                key = ('path', str(source)) if isinstance(source, (str, Path)) else ('object', id(source))
                if key not in cache:
                    cache[key] = self._read_image(source)
                group.append(cache[key])
            groups.append(group)
        return groups

    def _processor_batch(self, texts, image_groups):
        kwargs = dict(text=texts, padding=True, return_tensors='pt')
        if image_groups and any(image_groups):
            # Native LLaVA/Qwen processors consume images in placeholder order.
            # OneVision additionally requires explicit per-sample grouping.
            kwargs['images'] = image_groups if 'onevision' in type(self.processor).__name__.lower() else [
                image for images in image_groups for image in images
            ]
        inputs = self.processor(**kwargs)
        if 'input_ids' not in inputs or 'attention_mask' not in inputs:
            raise ValueError('Batch reranking requires processor input_ids and attention_mask')
        if len(inputs['input_ids']) != len(texts):
            raise ValueError('Processor lost the sample-to-image mapping while building a rerank batch')
        return inputs

    def _pointwise_scores(self, requests, prompt_type, log_likelihood):
        inputs = move_batch(self._processor_batch([item.text for item in requests], self._batch_images(requests)), self.model.device)
        output = self.model(**inputs, use_cache=False, output_hidden_states=False, return_dict=True)
        logits = last_valid_logits(output.logits, inputs['attention_mask']).float()
        yes, no = ('A', 'B') if prompt_type == 'freeret' else ('Yes', 'No')
        from qwen_vl_prompts import is_native_qwen_model
        if is_native_qwen_model(self.model):
            encoded = [self.processor.tokenizer.encode(answer, add_special_tokens=False) for answer in (yes, no)]
            if any(len(ids) != 1 for ids in encoded):
                raise ValueError('Qwen pointwise scoring requires single-token Yes/No or A/B answers')
            answer_ids = [ids[0] for ids in encoded]
        else:
            answer_ids = [self.vocab_dict[yes], self.vocab_dict[no]]
        if log_likelihood:
            logits = logits.log_softmax(dim=-1)
        scores = logits[:, answer_ids].softmax(dim=-1)[:, 0]
        return scores.detach().cpu().tolist()

    def _caption_scores(self, requests, *, text_only=False):
        prefixes = [item.text_only_prefix if text_only else item.prefix for item in requests]
        texts = [prefix + item.text[len(item.prefix):] for prefix, item in zip(prefixes, requests)]
        groups = [] if text_only else self._batch_images(requests)
        inputs = self._processor_batch(texts, groups)
        prefix_inputs = self._processor_batch(prefixes, groups)
        labels = target_labels(inputs['input_ids'], inputs['attention_mask'], prefix_inputs['input_ids'], prefix_inputs['attention_mask'])
        del prefix_inputs
        inputs = move_batch(inputs, self.model.device)
        output = self.model(**inputs, use_cache=False, output_hidden_states=False, return_dict=True)
        scores = mean_target_nll(output.logits, labels, inputs['attention_mask'])
        return scores.detach().cpu().tolist()

    def _batch_rerank(self, fusion_run, task_type, template, batch_size, prompt_type,
                      data_args, *, rerank_num, caption=False, modify_type='no', log_likelihood=False,
                      progress_observer=None, prior_scores=None):
        validate_rerank_batch_size(batch_size)
        if not fusion_run and progress_observer is None:
            return {}
        if fusion_run:
            configure_rerank_processor(self.model, self.processor, data_args.reps_loc)
        # Empty distributed shards must still enter this observer-controlled
        # loop while peers score their nonempty shards.
        result = {}
        for query_id, run in _stage3_query_progress(fusion_run.items(), progress_observer):
            run = run['docs'] if isinstance(run.get('docs'), dict) else run
            candidates = sorted(run, key=run.get, reverse=True)[:max(0, int(rerank_num))]
            if not candidates:
                result[query_id] = {}
                continue
            requests = self._requests(query_id, candidates, task_type, template, prompt_type, caption=caption)
            scores = [None] * len(requests)
            for batch in _stage3_candidate_progress(list(indexed_batches(requests, batch_size, key=lambda item: len(item.images)))):
                batch_requests = [item for _, item in batch]
                if caption:
                    values = [-score for score in self._caption_scores(batch_requests)]
                    if modify_type == 'modify_single' and all(item.text_only_prefix is not None for item in batch_requests):
                        if prior_scores is None:
                            baseline = self._caption_scores(batch_requests, text_only=True)
                        else:
                            for item in batch_requests:
                                if item.text_only_prefix != template or item.text[len(item.prefix):] != self._caption_prior_texts[item.candidate_id]:
                                    raise ValueError('Cached prior input differs from the existing text-only scoring input')
                            baseline = [prior_scores[candidate_key(item.candidate_id)] for item in batch_requests]
                        values = [score + text_score for score, text_score in zip(values, baseline)]
                else:
                    score_type = 'relevant' if task_type == 'cir' else prompt_type
                    values = self._pointwise_scores(batch_requests, score_type, log_likelihood)
                if len(values) != len(batch):
                    raise ValueError('Reranker must return exactly one score for every candidate in its batch')
                for (position, _), score in zip(batch, values):
                    scores[position] = float(score)
            run = dict(zip(candidates, scores))
            result[query_id] = dict(sorted(run.items(), key=lambda item: item[1], reverse=True))
        return result

    def rerank(self, fusion_run, rerank_type, rerank_num, data_args, training_args, model_args, rerank_batch_size=1,
               rerank_prompt_type='relevant', log_likelihood=False, progress_observer=None):
        from qwen_vl_prompts import is_native_qwen_model, rerank_template
        if is_native_qwen_model(getattr(self, 'model', None)) or any(name in model_args.model_name_or_path for name in ('Qwen2-VL-', 'Qwen2.5-VL-', 'Qwen3-VL-')):
            if rerank_type != 'pointwise':
                raise ValueError('Qwen listwise reranking has no implemented scoring definition')
            template = rerank_template(self.processor, training_args.task_type, self.query_type, rerank_prompt_type)
            with torch.no_grad():
                return self._batch_rerank(fusion_run, training_args.task_type, template,
                                          rerank_batch_size, rerank_prompt_type, data_args,
                                          rerank_num=rerank_num, log_likelihood=log_likelihood,
                                          progress_observer=progress_observer)
        with torch.no_grad(), torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
            rerank_fusion_run = {}
            if training_args.task_type == 'cir':
                if 'llava-hf-llava-v1.6-mistral-7b-hf' in model_args.model_name_or_path:
                    if rerank_prompt_type == 'relevant':
                        rerank_prompt_template = mistral_fashion_iq_relevant_prompt
                    elif rerank_prompt_type == 'old_relevant':
                        rerank_prompt_template = mistral_fashion_iq_old_query_relevant_prompt
                    elif rerank_prompt_type == 'query_relevant':
                        rerank_prompt_template = mistral_fashion_iq_query_relevant_prompt
                    elif rerank_prompt_type == 'origin_old_relevant':
                        rerank_prompt_template = mistral_fashion_iq_origin_old_query_relevant_prompt
                    elif rerank_prompt_type == 'please_relevant':
                        rerank_prompt_template = mistral_fashion_iq_please_relevant_prompt
                    elif rerank_prompt_type == 'in_one_word_relevant':
                        rerank_prompt_template = mistral_fashion_iq_in_one_word_relevant_prompt
                    elif rerank_prompt_type == 'role_relevant':
                        rerank_prompt_template = mistral_fashion_iq_role_relevant_prompt
                    elif rerank_prompt_type == 'role_old_relevant':
                        rerank_prompt_template = mistral_fashion_iq_role_old_query_relevant_prompt
                    else:
                        rerank_prompt_template = mistral_fashion_iq_relevant_prompt
                else:
                    if rerank_prompt_type == 'relevant':
                        rerank_prompt_template = fashion_iq_relevant_prompt
                    elif rerank_prompt_type == 'old_relevant':
                        rerank_prompt_template = fashion_iq_old_query_relevant_prompt
                    elif rerank_prompt_type == 'query_relevant':
                        rerank_prompt_template = fashion_iq_query_relevant_prompt
                    elif rerank_prompt_type == 'origin_old_relevant':
                        rerank_prompt_template = fashion_iq_origin_old_query_relevant_prompt
                    else:
                        rerank_prompt_template = fashion_iq_relevant_prompt
            elif training_args.task_type == 'tbpr':
                if 'llava-hf-llava-v1.6-mistral-7b-hf' in model_args.model_name_or_path:
                    if rerank_prompt_type == 'relevant':
                        rerank_prompt_template = mistral_person_retrieval_relevant_prompt
                    elif rerank_prompt_type == 'old_relevant':
                        rerank_prompt_template = mistral_person_retrieval_old_query_relevant_prompt
                    elif rerank_prompt_type == 'query_relevant':
                        rerank_prompt_template = mistral_person_retrieval_query_relevant_prompt
                    elif rerank_prompt_type == 'origin_old_relevant':
                        rerank_prompt_template = mistral_person_retrieval_origin_old_query_relevant_prompt
                    elif rerank_prompt_type == 'reverse_old_relevant':
                        rerank_prompt_template = mistral_person_retrieval_reverse_old_query_relevant_prompt
                    elif rerank_prompt_type == 'reverse_query_relevant':
                        rerank_prompt_template = mistral_person_retrieval_reverse_query_relevant_prompt
                    elif rerank_prompt_type == 'reverse_origin_old_relevant':
                        rerank_prompt_template = mistral_person_retrieval_reverse_origin_old_query_relevant_prompt
                    elif rerank_prompt_type == 'freeret':
                        rerank_prompt_template = mistral_freeret_text_rerank_prompt
                    else:
                        rerank_prompt_template = mistral_person_retrieval_relevant_prompt
                elif 'llava-hf-llava-v1.6-vicuna-7b-hf' in model_args.model_name_or_path or 'llava-hf-llava-v1.6-vicuna-13b-hf' in model_args.model_name_or_path:
                    if rerank_prompt_type == 'old_relevant':
                        rerank_prompt_template = vicuna_person_retrieval_old_query_relevant_prompt
                    elif rerank_prompt_type == 'query_relevant':
                        rerank_prompt_template = vicuna_person_retrieval_query_relevant_prompt
                    elif rerank_prompt_type == 'origin_old_relevant':
                        rerank_prompt_template = vicuna_person_retrieval_origin_old_query_relevant_prompt
                    elif rerank_prompt_type == 'reverse_old_relevant':
                        rerank_prompt_template = vicuna_person_retrieval_reverse_old_query_relevant_prompt
                    elif rerank_prompt_type == 'reverse_query_relevant':
                        rerank_prompt_template = vicuna_person_retrieval_reverse_query_relevant_prompt
                    elif rerank_prompt_type == 'reverse_origin_old_relevant':
                        rerank_prompt_template = vicuna_person_retrieval_reverse_origin_old_query_relevant_prompt
                    elif rerank_prompt_type == 'freeret':
                        rerank_prompt_template = vicuna_freeret_text_rerank_prompt
                    else:
                        rerank_prompt_template = vicuna_person_retrieval_query_relevant_prompt
                elif 'llava-hf-llava-v1.6-34b-hf' in model_args.model_name_or_path:
                    if rerank_prompt_type == 'query_relevant':
                        rerank_prompt_template = llava_34b_person_retrieval_query_relevant_prompt
                    elif rerank_prompt_type == 'reverse_query_relevant':
                        rerank_prompt_template = llava_34b_person_retrieval_reverse_query_relevant_prompt
                elif 'Qwen-Qwen3-VL-8B-Instruct' in model_args.model_name_or_path:
                    if rerank_prompt_type == 'relevant':
                        rerank_prompt_template = qwen3_person_retrieval_relevant_prompt
                    elif rerank_prompt_type == 'old_relevant':
                        rerank_prompt_template = qwen3_person_retrieval_old_query_relevant_prompt
                    elif rerank_prompt_type == 'query_relevant':
                        rerank_prompt_template = qwen3_person_retrieval_query_relevant_prompt
                    elif rerank_prompt_type == 'origin_old_relevant':
                        rerank_prompt_template = qwen3_person_retrieval_origin_old_query_relevant_prompt
                    else:
                        rerank_prompt_template = qwen3_person_retrieval_relevant_prompt
                elif 'LamRA' in model_args.model_name_or_path:
                    if 'Qwen' in model_args.model_name_or_path:
                        rerank_prompt_template = lamra_2_5_text_relevant_prompt
                    else:
                        rerank_prompt_template = lamra_2_text_relevant_prompt
                else:
                    if rerank_prompt_type == 'relevant':
                        rerank_prompt_template = person_retrieval_relevant_prompt
                    elif rerank_prompt_type == 'old_relevant':
                        rerank_prompt_template = person_retrieval_old_query_relevant_prompt
                    elif rerank_prompt_type == 'query_relevant':
                        rerank_prompt_template = person_retrieval_query_relevant_prompt
                    elif rerank_prompt_type == 'origin_old_relevant':
                        rerank_prompt_template = person_retrieval_origin_old_query_relevant_prompt
                    elif rerank_prompt_type == 'reverse_old_relevant':
                        rerank_prompt_template = person_retrieval_reverse_old_query_relevant_prompt
                    elif rerank_prompt_type == 'reverse_query_relevant':
                        rerank_prompt_template = person_retrieval_reverse_query_relevant_prompt
                    elif rerank_prompt_type == 'reverse_origin_old_relevant':
                        rerank_prompt_template = person_retrieval_reverse_origin_old_query_relevant_prompt
                    elif rerank_prompt_type == 'freeret':
                        rerank_prompt_template = freeret_text_rerank_prompt
                    else:
                        rerank_prompt_template = person_retrieval_relevant_prompt
            elif training_args.task_type == 't2it':
                if 'llava-hf-llava-v1.6-mistral-7b-hf' in model_args.model_name_or_path:
                    if rerank_prompt_type == 'old_relevant':
                        rerank_prompt_template = mistral_t2it_retrieval_old_query_relevant_prompt
                    elif rerank_prompt_type == 'query_relevant':
                        rerank_prompt_template = mistral_t2it_retrieval_query_relevant_prompt
                    elif rerank_prompt_type == 'origin_old_relevant':
                        rerank_prompt_template = mistral_t2it_retrieval_origin_old_query_relevant_prompt
                    else:
                        rerank_prompt_template = mistral_t2it_retrieval_old_query_relevant_prompt
                elif 'LamRA' in model_args.model_name_or_path:
                    if 'Qwen' in model_args.model_name_or_path:
                        rerank_prompt_template = lamra_2_5_t2it_relevant_prompt
                    else:
                        rerank_prompt_template = lamra_2_t2it_relevant_prompt
                else:
                    if rerank_prompt_type == 'old_relevant':
                        rerank_prompt_template = t2it_retrieval_old_query_relevant_prompt
                    elif rerank_prompt_type == 'query_relevant':
                        rerank_prompt_template = t2it_retrieval_query_relevant_prompt
                    elif rerank_prompt_type == 'origin_old_relevant':
                        rerank_prompt_template = t2it_retrieval_origin_old_query_relevant_prompt
                    else:
                        rerank_prompt_template = t2it_retrieval_old_query_relevant_prompt
            elif training_args.task_type == 'it2t':
                if 'llava-hf-llava-v1.6-mistral-7b-hf' in model_args.model_name_or_path:
                    if rerank_prompt_type == 'old_relevant':
                        rerank_prompt_template = mistral_it2t_retrieval_old_query_relevant_prompt
                    elif rerank_prompt_type == 'query_relevant':
                        rerank_prompt_template = mistral_it2t_retrieval_query_relevant_prompt
                    elif rerank_prompt_type == 'origin_old_relevant':
                        rerank_prompt_template = mistral_it2t_retrieval_origin_old_query_relevant_prompt
                    elif rerank_prompt_type == 'answer_relevant':
                        rerank_prompt_template = mistral_it2t_retrieval_answer_relevant_prompt
                    else:
                        rerank_prompt_template = mistral_it2t_retrieval_old_query_relevant_prompt
                elif 'LamRA' in model_args.model_name_or_path:
                    if 'Qwen' in model_args.model_name_or_path:
                        rerank_prompt_template = lamra_2_5_it2t_relevant_prompt
                    else:
                        rerank_prompt_template = lamra_2_it2t_relevant_prompt
                else:
                    if rerank_prompt_type == 'old_relevant':
                        rerank_prompt_template = it2t_retrieval_old_query_relevant_prompt
                    elif rerank_prompt_type == 'query_relevant':
                        rerank_prompt_template = it2t_retrieval_query_relevant_prompt
                    elif rerank_prompt_type == 'origin_old_relevant':
                        rerank_prompt_template = it2t_retrieval_origin_old_query_relevant_prompt
                    elif rerank_prompt_type == 'answer_relevant':
                        rerank_prompt_template = it2t_retrieval_answer_relevant_prompt
                    else:
                        rerank_prompt_template = it2t_retrieval_old_query_relevant_prompt
            else:
                if 'llava-hf-llava-v1.6-mistral-7b-hf' in model_args.model_name_or_path:
                    if rerank_prompt_type == 'relevant':
                        rerank_prompt_template = mistral_relevant_prompt
                    elif rerank_prompt_type == 'old_relevant':
                        if self.query_type == 'image':
                            rerank_prompt_template = mistral_old_image_query_relevant_prompt
                        else:
                            rerank_prompt_template = mistral_old_text_query_relevant_prompt
                    elif rerank_prompt_type == 'reverse_old_relevant':
                        if self.query_type == 'image':
                            rerank_prompt_template = mistral_old_image_reverse_query_relevant_prompt
                        else:
                            rerank_prompt_template = mistral_old_text_reverse_query_relevant_prompt
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
                    elif rerank_prompt_type == 'reverse_query_relevant':
                        if self.query_type == 'image':
                            rerank_prompt_template = mistral_image_reverse_query_relevant_prompt
                        else:
                            rerank_prompt_template = mistral_text_reverse_query_relevant_prompt
                    elif rerank_prompt_type == 'origin_old_relevant':
                        if self.query_type == 'image':
                            rerank_prompt_template = mistral_origin_old_image_query_relevant_prompt
                        else:
                            rerank_prompt_template = mistral_origin_old_text_query_relevant_prompt
                    elif rerank_prompt_type == 'reverse_origin_old_relevant':
                        if self.query_type == 'image':
                            rerank_prompt_template = mistral_origin_old_image_reverse_query_relevant_prompt
                        else:
                            rerank_prompt_template = mistral_origin_old_text_reverse_query_relevant_prompt
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
                    elif rerank_prompt_type == 'freeret':
                        if self.query_type == 'image':
                            rerank_prompt_template = mistral_freeret_image_rerank_prompt
                        else:
                            rerank_prompt_template = mistral_freeret_text_rerank_prompt
                    else:
                        rerank_prompt_template = mistral_relevant_prompt
                elif 'llava-hf-llava-v1.6-vicuna-7b-hf' in model_args.model_name_or_path or 'llava-hf-llava-v1.6-vicuna-13b-hf' in model_args.model_name_or_path:
                    if rerank_prompt_type == 'old_relevant':
                        if self.query_type == 'image':
                            rerank_prompt_template = vicuna_old_image_query_relevant_prompt
                        else:
                            rerank_prompt_template = vicuna_old_text_query_relevant_prompt
                    elif rerank_prompt_type == 'reverse_old_relevant':
                        if self.query_type == 'image':
                            rerank_prompt_template = vicuna_old_image_reverse_query_relevant_prompt
                        else:
                            rerank_prompt_template = vicuna_old_text_reverse_query_relevant_prompt
                    elif rerank_prompt_type == 'query_relevant':
                        if self.query_type == 'image':
                            rerank_prompt_template = vicuna_image_query_relevant_prompt
                        else:
                            rerank_prompt_template = vicuna_text_query_relevant_prompt
                    elif rerank_prompt_type == 'reverse_query_relevant':
                        if self.query_type == 'image':
                            rerank_prompt_template = vicuna_image_reverse_query_relevant_prompt
                        else:
                            rerank_prompt_template = vicuna_text_reverse_query_relevant_prompt
                    elif rerank_prompt_type == 'origin_old_relevant':
                        if self.query_type == 'image':
                            rerank_prompt_template = vicuna_origin_old_image_query_relevant_prompt
                        else:
                            rerank_prompt_template = vicuna_origin_old_text_query_relevant_prompt
                    elif rerank_prompt_type == 'reverse_origin_old_relevant':
                        if self.query_type == 'image':
                            rerank_prompt_template = vicuna_origin_old_image_reverse_query_relevant_prompt
                        else:
                            rerank_prompt_template = vicuna_origin_old_text_reverse_query_relevant_prompt
                    elif rerank_prompt_type == 'freeret':
                        if self.query_type == 'image':
                            rerank_prompt_template = vicuna_freeret_image_rerank_prompt
                        else:
                            rerank_prompt_template = vicuna_freeret_text_rerank_prompt
                    else:
                        rerank_prompt_template = mistral_relevant_prompt

                elif 'llava-hf-llava-v1.6-34b-hf' in model_args.model_name_or_path:
                    if rerank_prompt_type == 'query_relevant':
                        if self.query_type == 'image':
                            rerank_prompt_template = llava_34b_image_query_relevant_prompt
                        else:
                            rerank_prompt_template = llava_34b_text_query_relevant_prompt
                    elif rerank_prompt_type == 'reverse_query_relevant':
                        if self.query_type == 'image':
                            rerank_prompt_template = llava_34b_image_reverse_query_relevant_prompt
                        else:
                            rerank_prompt_template = llava_34b_text_reverse_query_relevant_prompt

                elif 'Qwen2.5-VL-7B-Instruct' in model_args.model_name_or_path:
                    if rerank_prompt_type == 'relevant':
                        rerank_prompt_template = qwen2_5_relevant_prompt
                    elif rerank_prompt_type == 'old_relevant':
                        if self.query_type == 'image':
                            rerank_prompt_template = qwen2_5_old_image_query_relevant_prompt
                        else:
                            rerank_prompt_template = qwen2_5_old_text_query_relevant_prompt
                    elif rerank_prompt_type == 'reverse_old_relevant':
                        if self.query_type == 'image':
                            rerank_prompt_template = qwen2_5_old_image_reverse_query_relevant_prompt
                        else:
                            rerank_prompt_template = qwen2_5_old_text_reverse_query_relevant_prompt
                    elif rerank_prompt_type == 'please_relevant':
                        rerank_prompt_template = qwen2_5_please_relevant_prompt
                    elif rerank_prompt_type == 'in_one_word_relevant':
                        rerank_prompt_template = qwen2_5_in_one_word_relevant_prompt
                    elif rerank_prompt_type == 'precise_caption':
                        rerank_prompt_template = qwen2_5_precise_caption_prompt
                    elif rerank_prompt_type == 'query_relevant':
                        if self.query_type == 'image':
                            rerank_prompt_template = qwen2_5_image_query_relevant_prompt
                        else:
                            rerank_prompt_template = qwen2_5_text_query_relevant_prompt
                    elif rerank_prompt_type == 'origin_old_relevant':
                        if self.query_type == 'image':
                            rerank_prompt_template = qwen2_5_origin_old_image_query_relevant_prompt
                        else:
                            rerank_prompt_template = qwen2_5_origin_old_text_query_relevant_prompt
                    elif rerank_prompt_type == 'reverse_origin_old_relevant':
                        if self.query_type == 'image':
                            rerank_prompt_template = qwen2_5_origin_old_image_reverse_query_relevant_prompt
                        else:
                            rerank_prompt_template = qwen2_5_origin_old_text_reverse_query_relevant_prompt
                    elif rerank_prompt_type == 'role_relevant':
                        rerank_prompt_template = qwen2_5_role_relevant_prompt
                    elif rerank_prompt_type == 'role_precise_caption':
                        rerank_prompt_template = qwen2_5_role_precise_caption_prompt
                    elif rerank_prompt_type == 'role_old_relevant':
                        if self.query_type == 'image':
                            rerank_prompt_template = qwen2_5_role_old_image_query_relevant_prompt
                        else:
                            rerank_prompt_template = qwen2_5_role_old_text_query_relevant_prompt
                    elif rerank_prompt_type == 'first_precise_caption':
                        rerank_prompt_template = qwen2_5_first_precise_caption_prompt
                    else:
                        rerank_prompt_template = qwen2_5_relevant_prompt
                elif 'Qwen3-VL-8B-Instruct' in model_args.model_name_or_path:
                    if rerank_prompt_type == 'relevant':
                        rerank_prompt_template = qwen3_relevant_prompt
                    elif rerank_prompt_type == 'old_relevant':
                        if self.query_type == 'image':
                            rerank_prompt_template = qwen3_old_image_query_relevant_prompt
                        else:
                            rerank_prompt_template = qwen3_old_text_query_relevant_prompt
                    elif rerank_prompt_type == 'reverse_old_relevant':
                        if self.query_type == 'image':
                            rerank_prompt_template = qwen3_old_image_reverse_query_relevant_prompt
                        else:
                            rerank_prompt_template = qwen3_old_text_reverse_query_relevant_prompt
                    elif rerank_prompt_type == 'please_relevant':
                        rerank_prompt_template = qwen3_please_relevant_prompt
                    elif rerank_prompt_type == 'in_one_word_relevant':
                        rerank_prompt_template = qwen3_in_one_word_relevant_prompt
                    elif rerank_prompt_type == 'precise_caption':
                        rerank_prompt_template = qwen3_precise_caption_prompt
                    elif rerank_prompt_type == 'query_relevant':
                        if self.query_type == 'image':
                            rerank_prompt_template = qwen3_image_query_relevant_prompt
                        else:
                            rerank_prompt_template = qwen3_text_query_relevant_prompt
                    elif rerank_prompt_type == 'origin_old_relevant':
                        if self.query_type == 'image':
                            rerank_prompt_template = qwen3_origin_old_image_query_relevant_prompt
                        else:
                            rerank_prompt_template = qwen3_origin_old_text_query_relevant_prompt
                    elif rerank_prompt_type == 'reverse_origin_old_relevant':
                        if self.query_type == 'image':
                            rerank_prompt_template = qwen3_origin_old_image_reverse_query_relevant_prompt
                        else:
                            rerank_prompt_template = qwen3_origin_old_text_reverse_query_relevant_prompt
                    elif rerank_prompt_type == 'role_relevant':
                        rerank_prompt_template = qwen3_role_relevant_prompt
                    elif rerank_prompt_type == 'role_precise_caption':
                        rerank_prompt_template = qwen3_role_precise_caption_prompt
                    elif rerank_prompt_type == 'role_old_relevant':
                        if self.query_type == 'image':
                            rerank_prompt_template = qwen3_role_old_image_query_relevant_prompt
                        else:
                            rerank_prompt_template = qwen3_role_old_text_query_relevant_prompt
                    elif rerank_prompt_type == 'first_precise_caption':
                        rerank_prompt_template = qwen3_first_precise_caption_prompt
                    else:
                        rerank_prompt_template = qwen3_relevant_prompt
                elif 'LamRA' in model_args.model_name_or_path:
                    if self.query_type == 'image':
                        rerank_prompt_template = lamra_2_image_relevant_prompt
                    else:
                        rerank_prompt_template = lamra_2_text_relevant_prompt
                else:
                    if rerank_prompt_type == 'relevant':
                        rerank_prompt_template = relevant_prompt
                    elif rerank_prompt_type == 'old_relevant':
                        if self.query_type == 'image':
                            rerank_prompt_template = old_image_query_relevant_prompt
                        else:
                            rerank_prompt_template = old_text_query_relevant_prompt
                    elif rerank_prompt_type == 'reverse_old_relevant':
                        if self.query_type == 'image':
                            rerank_prompt_template = old_image_reverse_query_relevant_prompt
                        else:
                            rerank_prompt_template = old_text_reverse_query_relevant_prompt
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
                    elif rerank_prompt_type == 'reverse_query_relevant':
                        if self.query_type == 'image':
                            rerank_prompt_template = image_reverse_query_relevant_prompt
                        else:
                            rerank_prompt_template = text_reverse_query_relevant_prompt
                    elif rerank_prompt_type == 'origin_old_relevant':
                        if self.query_type == 'image':
                            rerank_prompt_template = origin_old_image_query_relevant_prompt
                        else:
                            rerank_prompt_template = origin_old_text_query_relevant_prompt
                    elif rerank_prompt_type == 'reverse_origin_old_relevant':
                        if self.query_type == 'image':
                            rerank_prompt_template = origin_old_image_reverse_query_relevant_prompt
                        else:
                            rerank_prompt_template = origin_old_text_reverse_query_relevant_prompt
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
                    elif rerank_prompt_type == 'freeret':
                        if self.query_type == 'image':
                            rerank_prompt_template = freeret_image_rerank_prompt
                        else:
                            rerank_prompt_template = freeret_text_rerank_prompt
                    else:
                        rerank_prompt_template = relevant_prompt
            '''
            conversation = [
                {

                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": 'For the following sentence and image, judge whether they are relevant. Output "Yes" or "No".\nSentence: <sent> Image: <image> Output: '},
                        {"type": "image"},
                    ],
                },
            ]
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            '''
            if _is_main_process():
                print(rerank_prompt_template)
            if rerank_type == 'pointwise':
                return self._batch_rerank(
                    fusion_run, training_args.task_type, rerank_prompt_template,
                    rerank_batch_size, rerank_prompt_type, data_args,
                    rerank_num=rerank_num, log_likelihood=log_likelihood,
                    progress_observer=progress_observer,
                )
            elif rerank_type == 'listwise':
                pass
            else:
                pass
            return rerank_fusion_run

    def _caption_template(self, training_args, model_args, rerank_prompt_type):
        from qwen_vl_prompts import is_native_qwen_model, rerank_template
        if is_native_qwen_model(getattr(self, 'model', None)) or any(name in model_args.model_name_or_path for name in ('Qwen2-VL-', 'Qwen2.5-VL-', 'Qwen3-VL-')):
            return rerank_template(self.processor, training_args.task_type, self.query_type,
                                   rerank_prompt_type, caption=True)
        if 'llava-hf-llava-v1.6-mistral-7b-hf' in model_args.model_name_or_path:
            if training_args.task_type == 'tbpr':
                if rerank_prompt_type == 'caption_generation':
                    rerank_prompt_template = person_retrieval_mistral_query_generation_paradigm_prompt
                elif rerank_prompt_type == 'what_caption_generation':
                    rerank_prompt_template = person_retrieval_mistral_query_generation_paradigm_prompt_1
                elif rerank_prompt_type == 'describe_caption_generation':
                    rerank_prompt_template = person_retrieval_mistral_query_generation_paradigm_prompt_2
            elif training_args.task_type == 't2it':
                if rerank_prompt_type == 'caption_generation':
                    rerank_prompt_template = t2it_retrieval_mistral_query_generation_paradigm_prompt
                else:
                    rerank_prompt_template = t2it_retrieval_mistral_query_generation_paradigm_prompt_1
            elif training_args.task_type == 'it2t':
                if rerank_prompt_type == 'caption_generation':
                    rerank_prompt_template = it2t_retrieval_mistral_query_generation_paradigm_prompt
                else:
                    rerank_prompt_template = it2t_retrieval_mistral_query_generation_paradigm_prompt_1
            else:
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
                elif rerank_prompt_type == 'caption_generation_2':
                    rerank_prompt_template = mistral_query_generation_paradigm_prompt_2
                elif rerank_prompt_type == 'caption_generation_3':
                    rerank_prompt_template = mistral_query_generation_paradigm_prompt_3
                elif rerank_prompt_type == 'caption_generation_6':
                    rerank_prompt_template = mistral_query_generation_paradigm_prompt_6
                elif rerank_prompt_type == 'caption_generation_7':
                    rerank_prompt_template = mistral_query_generation_paradigm_prompt_7
                else:
                    rerank_prompt_template = mistral_query_generation_paradigm_prompt
        elif 'llava-hf-llava-v1.6-vicuna-7b-hf' in model_args.model_name_or_path or 'llava-hf-llava-v1.6-vicuna-13b-hf' in model_args.model_name_or_path:
            if training_args.task_type == 'tbpr':
                if rerank_prompt_type == 'caption_generation':
                    rerank_prompt_template = person_retrieval_vicuna_query_generation_paradigm_prompt
                elif rerank_prompt_type == 'what_caption_generation':
                    rerank_prompt_template = person_retrieval_vicuna_query_generation_paradigm_prompt_1
            else:
                if rerank_prompt_type == 'caption_generation':
                    rerank_prompt_template = vicuna_query_generation_paradigm_prompt
                elif rerank_prompt_type == 'what_caption_generation':
                    rerank_prompt_template = vicuna_query_generation_paradigm_prompt_1

        elif 'llava-hf-llava-v1.6-34b-hf' in model_args.model_name_or_path:
            if training_args.task_type == 'tbpr':
                if rerank_prompt_type == 'caption_generation':
                    rerank_prompt_template = person_retrieval_llava_34b_query_generation_paradigm_prompt
                elif rerank_prompt_type == 'what_caption_generation':
                    rerank_prompt_template = person_retrieval_llava_34b_query_generation_paradigm_prompt_1
            else:
                if rerank_prompt_type == 'caption_generation':
                    rerank_prompt_template = llava_34b_query_generation_paradigm_prompt
                elif rerank_prompt_type == 'what_caption_generation':
                    rerank_prompt_template = llava_34b_query_generation_paradigm_prompt_1
        elif 'Qwen2.5-VL-7B-Instruct' in model_args.model_name_or_path:
            if rerank_prompt_type == 'caption_generation':
                rerank_prompt_template = qwen2_5_query_generation_paradigm_prompt
            elif rerank_prompt_type == 'what_caption_generation':
                rerank_prompt_template = qwen2_5_query_generation_paradigm_prompt_1
        elif 'Qwen3-VL-8B-Instruct' in model_args.model_name_or_path:
            if training_args.task_type == 'tbpr':
                if rerank_prompt_type == 'caption_generation':
                    rerank_prompt_template = person_retrieval_qwen3_query_generation_paradigm_prompt
                elif rerank_prompt_type == 'what_caption_generation':
                    rerank_prompt_template = person_retrieval_qwen3_query_generation_paradigm_prompt_1
            else:
                if rerank_prompt_type == 'caption_generation':
                    rerank_prompt_template = qwen3_query_generation_paradigm_prompt
                elif rerank_prompt_type == 'what_caption_generation':
                    rerank_prompt_template = qwen3_query_generation_paradigm_prompt_1
        else:
            if training_args.task_type == 'tbpr':
                if rerank_prompt_type == 'caption_generation':
                    rerank_prompt_template = person_retrieval_query_generation_paradigm_prompt
                elif rerank_prompt_type == 'what_caption_generation':
                    rerank_prompt_template = person_retrieval_query_generation_paradigm_prompt_1
                elif rerank_prompt_type == 'describe_caption_generation':
                    rerank_prompt_template = person_retrieval_query_generation_paradigm_prompt_2
            elif training_args.task_type == 't2it':
                if rerank_prompt_type == 'caption_generation':
                    rerank_prompt_template = t2it_retrieval_query_generation_paradigm_prompt
                else:
                    rerank_prompt_template = t2it_retrieval_query_generation_paradigm_prompt_1
            elif training_args.task_type == 'it2t':
                if rerank_prompt_type == 'caption_generation':
                    rerank_prompt_template = it2t_retrieval_query_generation_paradigm_prompt
                else:
                    rerank_prompt_template = it2t_retrieval_query_generation_paradigm_prompt_1
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
                elif rerank_prompt_type == 'caption_generation_2':
                    rerank_prompt_template = query_generation_paradigm_prompt_2
                elif rerank_prompt_type == 'caption_generation_3':
                    rerank_prompt_template = query_generation_paradigm_prompt_3
                elif rerank_prompt_type == 'caption_generation_6':
                    rerank_prompt_template = query_generation_paradigm_prompt_6
                elif rerank_prompt_type == 'caption_generation_7':
                    rerank_prompt_template = query_generation_paradigm_prompt_7
                else:
                    rerank_prompt_template = query_generation_paradigm_prompt

        return rerank_prompt_template

    def prepare_caption_prior_cache(self, data_args, training_args, model_args, search_args,
                                    rerank_batch_size=1, rerank_prompt_type='caption_generation',
                                    rerank_type='pointwise'):
        enabled = getattr(search_args, 'use_rerank_prior_cache', False)
        if not enabled:
            return None
        prepare_started = time.perf_counter()
        validate_prior_cache_options(enabled, search_args.modify_type, rerank_prompt_type,
                                     training_args.task_type, self.query_type, rerank_type)
        validate_rerank_batch_size(rerank_batch_size)
        configure_rerank_processor(self.model, self.processor, data_args.reps_loc)
        template = self._caption_template(training_args, model_args, rerank_prompt_type)
        # text_map is the actual current candidate pool. self.dataset can be a
        # separate training/choice dataset in the original rerank entrypoint.
        candidates = []
        for key, text in self.text_map.items():
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f'Prior cache requires nonempty candidate text for {key!r}')
            candidates.append((candidate_key(key), text))
        candidates.sort(key=lambda item: item[0])
        if not candidates:
            raise ValueError('Cannot precompute a prior for an empty candidate pool')
        metadata = prior_metadata(
            self.model, self.processor, model_args, training_args, data_args,
            dataset=self.data_name, query_type=self.query_type, template_name=rerank_prompt_type,
            prefix=template, candidates=candidates,
        )

        def score_batch(batch):
            requests = [RerankRequest(key, template + text, (), template, template) for key, text in batch]
            return self._caption_scores(requests, text_only=True)

        with torch.no_grad(), torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
            self._caption_prior_scores, self.prior_cache_report = load_or_build_prior_cache(
                metadata, candidates, score_batch, rerank_batch_size,
                cache_dir=getattr(search_args, 'rerank_prior_cache_dir', None), progress_factory=tqdm,
            )
        self._caption_prior_template = template
        self._caption_prior_texts = dict(self.text_map)
        self.prior_cache_report['prepare_seconds'] = time.perf_counter() - prepare_started
        return self.prior_cache_report

    def caption_generation_rerank(self, fusion_run, rerank_type, rerank_num, data_args, training_args, model_args,
                                  search_args, rerank_batch_size=1, rerank_prompt_type='caption_generation',
                                  progress_observer=None, prior_cache_prepared=False):
        from qwen_vl_prompts import is_native_qwen_model
        if (is_native_qwen_model(getattr(self, 'model', None)) or any(name in model_args.model_name_or_path for name in ('Qwen2-VL-', 'Qwen2.5-VL-', 'Qwen3-VL-'))) and rerank_type != 'pointwise':
            raise ValueError('Qwen listwise reranking has no implemented scoring definition')
        template = self._caption_template(training_args, model_args, rerank_prompt_type)
        prior_scores = None
        if getattr(search_args, 'use_rerank_prior_cache', False):
            validate_prior_cache_options(True, search_args.modify_type, rerank_prompt_type,
                                         training_args.task_type, self.query_type, rerank_type)
            if not prior_cache_prepared:
                self.prepare_caption_prior_cache(data_args, training_args, model_args, search_args,
                                                 rerank_batch_size, rerank_prompt_type, rerank_type)
            if getattr(self, '_caption_prior_template', None) != template:
                raise ValueError('Caption prior cache was not prepared for the current template')
            prior_scores = self._caption_prior_scores
        with torch.no_grad(), torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
            return self._batch_rerank(
                fusion_run, training_args.task_type, template,
                rerank_batch_size, rerank_prompt_type, data_args,
                rerank_num=rerank_num, caption=True,
                modify_type=search_args.modify_type,
                progress_observer=progress_observer,
                prior_scores=prior_scores,
            )
