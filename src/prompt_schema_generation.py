import os

import torch
import torch.distributed as dist
import torch.utils.data as Data
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
)
from transformers import LlamaForCausalLM, MistralForCausalLM, LlamaTokenizer, AutoTokenizer

from arguments import PromptRepsLLMDataArguments, ModelArguments
from arguments import TrainingArguments, PromptGenerationArguments
from dataset import CrossModalRetrievalDataset, TextPersonRetrievalDataset, ComposedTextImageRetrievalDataset, \
    Text2ImagetextRetrievalDataset, Imagetext2TextRetrievalDataset
from template import prompt_schema_generation_text_prompt, prompt_schema_generation_text_prompt_1, \
    mistral_prompt_schema_generation_text_prompt, mistral_prompt_schema_generation_text_prompt_1, \
    prompt_schema_generation_text_prompt_2, mistral_prompt_schema_generation_text_prompt_2, tbpr_five_aspects, \
    itr_five_aspects


def main():
    parser = HfArgumentParser(
        (ModelArguments, PromptRepsLLMDataArguments, TrainingArguments, PromptGenerationArguments))

    model_args, data_args, training_args, prompt_generation_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: PromptRepsLLMDataArguments
    training_args: TrainingArguments
    prompt_generation_args: PromptGenerationArguments

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
    if 'Meta-Llama-3-8B-Instruct' in model_args.model_name_or_path:
        model = LlamaForCausalLM.from_pretrained(model_args.model_name_or_path,
                                                device_map=device_map, torch_dtype=torch_type)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    elif 'Mistral-7B-Instruct-v0.3' in model_args.model_name_or_path:
        model = MistralForCausalLM.from_pretrained(model_args.model_name_or_path,
                                                device_map=device_map, torch_dtype=torch_type)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    else:
        model = LlamaForCausalLM.from_pretrained(model_args.model_name_or_path,
                                                 device_map=device_map, torch_dtype=torch_type)
        tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path)

    if data_args.reps_loc == 'after_pad':
        tokenizer.padding_side = "left"
        tokenizer.padding = True

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

        tbpr_cuhk_pedes_dataset = TextPersonRetrievalDataset('CUHK-PEDES', tokenizer, 'test', 'full')
        tbpr_icfg_pedes_dataset = TextPersonRetrievalDataset('ICFG-PEDES', tokenizer, 'test', 'full')
        itr_flickr_dataset = CrossModalRetrievalDataset('flickr', tokenizer, 'test', 'single')
        itr_coco_dataset = CrossModalRetrievalDataset('coco', tokenizer, 'test', 'single')
        cir_dataset = ComposedTextImageRetrievalDataset('fashion-iq', tokenizer, 'val', 'composed')
        t2it_webqa_dataset = Text2ImagetextRetrievalDataset('webqa', tokenizer, 'test', 'query')

        tbpr_cuhk_pedes_dataloader = Data.DataLoader(dataset=tbpr_cuhk_pedes_dataset, batch_size=1, shuffle=False)
        tbpr_icfg_pedes_dataloader = Data.DataLoader(dataset=tbpr_icfg_pedes_dataset, batch_size=1, shuffle=False)
        itr_flickr_dataloader = Data.DataLoader(dataset=itr_flickr_dataset, batch_size=1, shuffle=False)
        itr_coco_dataloader = Data.DataLoader(dataset=itr_coco_dataset, batch_size=1, shuffle=False)
        cir_dataloader = Data.DataLoader(dataset=cir_dataset, batch_size=1, shuffle=False)
        t2it_webqa_dataloader = Data.DataLoader(dataset=t2it_webqa_dataset, batch_size=1, shuffle=False)

        counter = 0
        tbpr_cuhk_pedes_demonstration = ''
        tbpr_icfg_pedes_demonstration = ''
        itr_flickr_demonstration = ''
        itr_coco_demonstration = ''
        cir_demonstration = ''
        t2it_webqa_demonstration = ''
        for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(tbpr_cuhk_pedes_dataloader),
                                                                     total=len(tbpr_cuhk_pedes_dataloader)):
            # print(texts)

            # tbpr_cuhk_pedes_demonstration += f'{counter}. '
            tbpr_cuhk_pedes_demonstration += texts[0]
            tbpr_cuhk_pedes_demonstration += '\n'

            if counter == prompt_generation_args.demonstration_num:
                break
            counter += 1

        counter = 0
        for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(tbpr_icfg_pedes_dataloader),
                                                                     total=len(tbpr_icfg_pedes_dataloader)):
            # print(texts)

            # tbpr_icfg_pedes_demonstration += f'{counter}. '
            tbpr_icfg_pedes_demonstration += texts[0]
            tbpr_icfg_pedes_demonstration += '\n'

            if counter == prompt_generation_args.demonstration_num:
                break
            counter += 1

        counter = 0
        for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(itr_flickr_dataloader),
                                                                     total=len(itr_flickr_dataloader)):
            # print(texts)

            # itr_flickr_demonstration += f'{counter}. '
            itr_flickr_demonstration += texts[0]
            itr_flickr_demonstration += '\n'

            if counter == prompt_generation_args.demonstration_num:
                break
            counter += 1
        # print(itr_flickr_demonstration)

        counter = 0
        for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(itr_coco_dataloader),
                                                                     total=len(itr_coco_dataloader)):
            # print(texts)
            counter += 1

            # itr_coco_demonstration += f'{counter}. '
            itr_coco_demonstration += texts[0]
            itr_coco_demonstration += '\n'

            if counter == prompt_generation_args.demonstration_num:
                break
        # print(itr_flickr_demonstration)

        counter = 0
        for batch_idx, (texts, imgs_path, target_path, text_ids, img_ids, composed_ids, dress_type) in tqdm(
                enumerate(cir_dataloader),
                total=len(cir_dataloader)):
            # print(texts)

            # cir_demonstration += f'{counter}. '
            cir_demonstration += texts[0]
            cir_demonstration += '\n'

            if counter == prompt_generation_args.demonstration_num:
                break
            counter += 1

        counter = 0
        for batch_idx, (query_texts, query_ids) in tqdm(
                enumerate(t2it_webqa_dataloader),
                total=len(t2it_webqa_dataloader)):
            # print(texts)

            # cir_demonstration += f'{counter}. '
            print(query_texts)
            t2it_webqa_demonstration += query_texts[0]
            t2it_webqa_demonstration += '\n'

            if counter == prompt_generation_args.demonstration_num:
                break
            counter += 1

        # print(cir_demonstration)

        if prompt_generation_args.prompt_generation_type == 'prompt_schema':
            text_input = prompt.replace('<sent>', itr_flickr_demonstration, 1)
        elif prompt_generation_args.prompt_generation_type == 'prompt_schema_1':
            text_input = prompt.replace('<sent>', tbpr_cuhk_pedes_demonstration, 1)
            text_input = text_input.replace('<sent>', tbpr_five_aspects, 1)
            text_input = text_input.replace('<sent>', itr_flickr_demonstration, 1)
        else:
            text_input = prompt.replace('<sent>', tbpr_cuhk_pedes_demonstration, 1)
            text_input = text_input.replace('<sent>', tbpr_five_aspects, 1)
            text_input = text_input.replace('<sent>', itr_coco_demonstration, 1)
            text_input = text_input.replace('<sent>', itr_five_aspects, 1)
            text_input = text_input.replace('<sent>', itr_flickr_demonstration, 1)
        inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=100)

        print('itr')
        print('Here is the original output')
        print(tokenizer.decode(output[0], skip_special_tokens=True))
        print('Here is the filtered output')
        print(tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True))

        if prompt_generation_args.prompt_generation_type == 'prompt_schema':
            text_input = prompt.replace('<sent>', tbpr_icfg_pedes_demonstration, 1)
        elif prompt_generation_args.prompt_generation_type == 'prompt_schema_1':
            text_input = prompt.replace('<sent>', itr_coco_demonstration, 1)
            text_input = text_input.replace('<sent>', itr_five_aspects, 1)
            text_input = text_input.replace('<sent>', tbpr_icfg_pedes_demonstration, 1)
        else:
            text_input = prompt.replace('<sent>', itr_coco_demonstration, 1)
            text_input = text_input.replace('<sent>', itr_five_aspects, 1)
            text_input = text_input.replace('<sent>', tbpr_cuhk_pedes_demonstration, 1)
            text_input = text_input.replace('<sent>', tbpr_five_aspects, 1)
            text_input = text_input.replace('<sent>', tbpr_icfg_pedes_demonstration, 1)
        inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=100)

        print('tbpr')
        print('Here is the original output')
        print(tokenizer.decode(output[0], skip_special_tokens=True))
        print('Here is the filtered output')
        print(tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True))

        if prompt_generation_args.prompt_generation_type == 'prompt_schema':
            text_input = prompt.replace('<sent>', t2it_webqa_demonstration, 1)
        elif prompt_generation_args.prompt_generation_type == 'prompt_schema_1':
            text_input = prompt.replace('<sent>', itr_coco_demonstration, 1)
            text_input = text_input.replace('<sent>', itr_five_aspects, 1)
            text_input = text_input.replace('<sent>', t2it_webqa_demonstration, 1)
        else:
            text_input = prompt.replace('<sent>', itr_coco_demonstration, 1)
            text_input = text_input.replace('<sent>', itr_five_aspects, 1)
            text_input = text_input.replace('<sent>', tbpr_cuhk_pedes_demonstration, 1)
            text_input = text_input.replace('<sent>', tbpr_five_aspects, 1)
            text_input = text_input.replace('<sent>', t2it_webqa_demonstration, 1)
        inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=100)

        print('t2it')
        print(t2it_webqa_demonstration)
        print('Here is the original output')
        print(tokenizer.decode(output[0], skip_special_tokens=True))
        print('Here is the filtered output')
        print(tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True))

if __name__ == "__main__":
    main()
