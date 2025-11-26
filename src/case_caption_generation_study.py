import os
import gc
import torch
import torch.distributed as dist
from PIL import Image
from transformers import (
    HfArgumentParser,
)
from transformers import LlavaProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, \
    LlavaNextForConditionalGeneration, Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration, AutoModel, \
    AutoProcessor, LlamaForCausalLM, MistralForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM

from arguments import PromptRepsLLMDataArguments, ModelArguments
from arguments import TrainingArguments, PromptGenerationArguments, PromptRepsLLMSearchArguments
from encode import get_filtered_ids
from model import MLLMRetrievalModel
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
    query_generation_paradigm_prompt_4, query_generation_paradigm_prompt_5, query_generation_paradigm_prompt_2, \
    query_generation_paradigm_prompt_3, mistral_query_generation_paradigm_prompt_2, \
    mistral_query_generation_paradigm_prompt_3, query_generation_paradigm_prompt_6, query_generation_paradigm_prompt_7, \
    mistral_query_generation_paradigm_prompt_6, mistral_query_generation_paradigm_prompt_7


def main():
    parser = HfArgumentParser(
        (ModelArguments, PromptRepsLLMDataArguments, PromptRepsLLMSearchArguments, TrainingArguments))

    model_args, data_args, search_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: PromptRepsLLMDataArguments
    training_args: TrainingArguments
    search_args: PromptRepsLLMSearchArguments

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
    if 'llava-hf-llava-1.5-7b-hf' in model_args.model_name_or_path:
        encoder = LlavaForConditionalGeneration.from_pretrained(model_args.model_name_or_path, device_map=device_map,
                                                                torch_dtype=torch_type)
        processor = LlavaProcessor.from_pretrained(model_args.model_name_or_path)

    elif 'Qwen2.5-VL-7B-Instruct' in model_args.model_name_or_path or 'Qwen2.5-VL-3B-Instruct' in model_args.model_name_or_path:
        encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_args.model_name_or_path,
                                                                     device_map=device_map,
                                                                     torch_dtype=torch_type)
        processor = Qwen2_5_VLProcessor.from_pretrained(model_args.model_name_or_path)
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

    if data_args.reps_loc == 'after_pad':
        processor.tokenizer.padding_side = "left"
        processor.tokenizer.padding = True

    model = MLLMRetrievalModel(encoder)
    model = model.eval()
    print(model.is_ddp)

    image_path = './data/flickr/flickr30k-images/58803866.jpg'
    text1 = 'Three dogs playing together in a grassy field with trees in the background'
    text2 = 'A crowd of people on the street gathering to watch several young men put on a show.'

    with torch.no_grad():
        if 'llava-hf-llava-v1.6-mistral-7b-hf' in model_args.model_name_or_path:
            if search_args.rerank_template == 'caption_generation':
                rerank_prompt_template = mistral_query_generation_paradigm_prompt
            elif search_args.rerank_template == 'what_caption_generation':
                rerank_prompt_template = mistral_query_generation_paradigm_prompt_1
            elif search_args.rerank_template == 'detailed_caption_generation':
                rerank_prompt_template = detailed_mistral_query_generation_paradigm_prompt
            elif search_args.rerank_template == 'detailed_caption_generation_1':
                rerank_prompt_template = detailed_mistral_query_generation_paradigm_prompt_1
            elif search_args.rerank_template == 'caption_generation_4':
                rerank_prompt_template = mistral_query_generation_paradigm_prompt_4
            elif search_args.rerank_template == 'caption_generation_5':
                rerank_prompt_template = mistral_query_generation_paradigm_prompt_5
            elif search_args.rerank_template == 'caption_generation_2':
                rerank_prompt_template = mistral_query_generation_paradigm_prompt_2
            elif search_args.rerank_template == 'caption_generation_3':
                rerank_prompt_template = mistral_query_generation_paradigm_prompt_3
            elif search_args.rerank_template == 'caption_generation_6':
                rerank_prompt_template = mistral_query_generation_paradigm_prompt_6
            elif search_args.rerank_template == 'caption_generation_7':
                rerank_prompt_template = mistral_query_generation_paradigm_prompt_7
            else:
                rerank_prompt_template = mistral_query_generation_paradigm_prompt
        else:
            if search_args.rerank_template == 'caption_generation':
                rerank_prompt_template = query_generation_paradigm_prompt
            elif search_args.rerank_template == 'what_caption_generation':
                rerank_prompt_template = query_generation_paradigm_prompt_1
            elif search_args.rerank_template == 'detailed_caption_generation':
                rerank_prompt_template = detailed_query_generation_paradigm_prompt
            elif search_args.rerank_template == 'detailed_caption_generation_1':
                rerank_prompt_template = detailed_query_generation_paradigm_prompt_1
            elif search_args.rerank_template == 'caption_generation_4':
                rerank_prompt_template = query_generation_paradigm_prompt_4
            elif search_args.rerank_template == 'caption_generation_5':
                rerank_prompt_template = query_generation_paradigm_prompt_5
            elif search_args.rerank_template == 'caption_generation_2':
                rerank_prompt_template = query_generation_paradigm_prompt_2
            elif search_args.rerank_template == 'caption_generation_3':
                rerank_prompt_template = query_generation_paradigm_prompt_3
            elif search_args.rerank_template == 'caption_generation_6':
                rerank_prompt_template = query_generation_paradigm_prompt_6
            elif search_args.rerank_template == 'caption_generation_7':
                rerank_prompt_template = query_generation_paradigm_prompt_7
            else:
                rerank_prompt_template = query_generation_paradigm_prompt

        text_input1 = rerank_prompt_template + text1
        text_input2 = rerank_prompt_template + text2

        raw_image = Image.open(image_path).convert('RGB')

        inputs_1 = processor(images=raw_image, text=text_input1, return_tensors="pt").to(
            encoder.device)

        inputs_2 = processor(images=raw_image, text=text_input2, return_tensors="pt").to(
            encoder.device)

        max_inputs_sum_1 = inputs_1['input_ids'].shape[1]
        labels_1 = processor(text=text_input1, return_tensors="pt")['input_ids'].squeeze().tolist()
        # 去掉label的第一个起始符
        labels_1 = [-100] * (max_inputs_sum_1 - len(labels_1[1:])) + labels_1[1:]

        max_inputs_sum_2 = inputs_2['input_ids'].shape[1]
        labels_2 = processor(text=text_input2, return_tensors="pt")['input_ids'].squeeze().tolist()
        # 去掉label的第一个起始符
        labels_2 = [-100] * (max_inputs_sum_2 - len(labels_2[1:])) + labels_2[1:]

        labels_view_1 = torch.tensor(labels_1).to(encoder.device)
        labels_view_2 = torch.tensor(labels_2).to(encoder.device)

        output_1 = encoder(**inputs_1, output_hidden_states=True, return_dict=True)
        logits_1 = output_1.logits
        shift_logits_1 = logits_1[..., :-1, :].contiguous()
        shift_labels_1 = labels_view_1[..., 1:].contiguous()

        output_2 = encoder(**inputs_2, output_hidden_states=True, return_dict=True)
        logits_2 = output_2.logits
        shift_logits_2 = logits_2[..., :-1, :].contiguous()
        shift_labels_2 = labels_view_2[..., 1:].contiguous()

        loss_func = torch.nn.CrossEntropyLoss(reduction='none')
        nll_1 = loss_func(shift_logits_1.view(-1, shift_logits_1.size(-1)), shift_labels_1.view(-1))
        nll_1 = nll_1.view(shift_labels_1.size())

        nll_2 = loss_func(shift_logits_2.view(-1, shift_logits_2.size(-1)), shift_labels_2.view(-1))
        nll_2 = nll_2.view(shift_labels_2.size())

        print(nll_1)
        print(nll_2)



