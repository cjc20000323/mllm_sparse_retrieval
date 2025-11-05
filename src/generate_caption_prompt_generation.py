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
from arguments import TrainingArguments, PromptGenerationArguments
from encode import get_filtered_ids
from model import MLLMRetrievalModel
from template import new_prompt_generation_from_pair_prompt, new_prompt_generation_from_pair_prompt_1, \
    new_prompt_generation_from_pair_prompt_2, new_prompt_generation_from_pair_prompt_4, \
    mistral_new_prompt_generation_from_pair_prompt, mistral_new_prompt_generation_from_pair_prompt_1, \
    mistral_new_prompt_generation_from_pair_prompt_2, mistral_new_prompt_generation_from_pair_prompt_4, \
    llava_mistral_caption_generation_prompt_1, llava_mistral_caption_generation_prompt_2, \
    llava_llama_caption_generation_prompt_1, llava_llama_caption_generation_prompt_2, \
    llama_prompt_generation_text_modal_only_prompt, llama_prompt_generation_text_modal_only_prompt_1, \
    llama_prompt_generation_text_modal_only_prompt_2, llama_prompt_generation_text_modal_only_prompt_3, \
    llama_prompt_generation_text_modal_only_prompt_4, mistral_prompt_generation_text_modal_only_prompt, \
    mistral_prompt_generation_text_modal_only_prompt_1, mistral_prompt_generation_text_modal_only_prompt_2, \
    mistral_prompt_generation_text_modal_only_prompt_3, mistral_prompt_generation_text_modal_only_prompt_4


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

    demonstrations_image_path_list = ['./data/flickr/flickr30k-images/101654506.jpg', './data/flickr/flickr30k-images/100207720.jpg', './data/flickr/flickr30k-images/1018148011.jpg', './data/flickr/flickr30k-images/1021439420.jpg']
    demonstrations_text_list = ['The white and brown dog is running over the surface of the snow.', 'Girl in black jacket sifting powdered sugar over a chocolate cake.', 'A group of people stand in the back of a truck filled with cotton.', 'Two guys sitting on the floor, with the guy in the green jacket reading a piece of paper.']
    demonstrations_answer_list = ['1. people or objects\n2. environment\n3. actions\n4. appearance', '1. people or objects\n2. environment\n3. actions\n4. appearance', '1. people or objects\n2. relations\n3. environment\n4. actions', '1. people or objects\n2. relations\n3. environment\n4. actions\n5. appearance']
    demonstrations_caption_list = []

    with torch.no_grad():
        if prompt_generation_args.prompt_generation_type == 'caption_generation_prompt_1':
            if 'llava-hf-llava-v1.6-mistral-7b-hf' in model_args.model_name_or_path:
                prompt = llava_mistral_caption_generation_prompt_1
            else:
                prompt = llava_llama_caption_generation_prompt_1
        else:
            if 'llava-hf-llava-v1.6-mistral-7b-hf' in model_args.model_name_or_path:
                prompt = llava_mistral_caption_generation_prompt_2
            else:
                prompt = llava_llama_caption_generation_prompt_2

        for demonstration_image_path in demonstrations_image_path_list:
            demonstration_image = Image.open(demonstration_image_path).convert('RGB')
            demonstration_input = processor(images=demonstration_image, text=prompt, return_tensors="pt").to(encoder.device)
            demonstration_output = model.encoder.generate(**demonstration_input, max_new_tokens=200)
            demonstrations_caption_list.append(processor.decode(demonstration_output[0][demonstration_input['input_ids'].shape[1]:], skip_special_tokens=True))

        image_path = prompt_generation_args.prompt_generation_image
        image = Image.open(image_path).convert('RGB')
        image_list = [image]
        inputs = processor(images=image_list, text=prompt, return_tensors="pt").to(encoder.device)
        output = model.encoder.generate(**inputs, max_new_tokens=200)
        # generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]  # 这一行是关键！
        caption = processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        if dist.get_rank() == 0:
            print(caption)
            print(demonstrations_caption_list)

    del model
    del encoder
    del processor

    gc.collect()
    torch.cuda.empty_cache()

    print('Now loading prompt generation model.')

    if 'llama' in prompt_generation_args.prompt_generation_model:
        model = LlamaForCausalLM.from_pretrained(prompt_generation_args.prompt_generation_model, device_map=device_map,
                                                                torch_dtype=torch_type)
        tokenizer = LlamaTokenizer.from_pretrained(prompt_generation_args.prompt_generation_model)
    else:
        model = MistralForCausalLM.from_pretrained(prompt_generation_args.prompt_generation_model, device_map=device_map,
                                                   torch_dtype=torch_type)
        tokenizer = AutoTokenizer.from_pretrained(prompt_generation_args.prompt_generation_model)

    with torch.no_grad():
        sent = prompt_generation_args.prompt_generation_text
        if prompt_generation_args.demonstration_num == 0:
            if 'Mistral' in prompt_generation_args.prompt_generation_model:
                prompt = mistral_prompt_generation_text_modal_only_prompt
            else:
                prompt = llama_prompt_generation_text_modal_only_prompt
            if prompt_generation_args.case_type == 'caption':
                text_input = prompt.replace('<sent>', caption, 1)
            else:
                text_input = prompt.replace('<sent>', sent, 1)
        elif prompt_generation_args.demonstration_num == 1:
            if 'Mistral' in prompt_generation_args.prompt_generation_model:
                prompt = mistral_prompt_generation_text_modal_only_prompt_1
            else:
                prompt = llama_prompt_generation_text_modal_only_prompt_1
            if prompt_generation_args.case_type == 'caption':
                text_input = prompt.replace('<sent>', demonstrations_caption_list[0], 1)
                text_input = text_input.replace('<sent>', demonstrations_answer_list[0], 1)
                text_input = text_input.replace('<sent>', caption, 1)
            else:
                text_input = prompt.replace('<sent>', demonstrations_text_list[0], 1)
                text_input = text_input.replace('<sent>', demonstrations_answer_list[0], 1)
                text_input = text_input.replace('<sent>', sent, 1)
        elif prompt_generation_args.demonstration_num == 2:
            if 'Mistral' in prompt_generation_args.prompt_generation_model:
                prompt = mistral_prompt_generation_text_modal_only_prompt_2
            else:
                prompt = llama_prompt_generation_text_modal_only_prompt_2
            if prompt_generation_args.case_type == 'caption':
                text_input = prompt.replace('<sent>', demonstrations_caption_list[0], 1)
                text_input = text_input.replace('<sent>', demonstrations_answer_list[0], 1)
                text_input = text_input.replace('<sent>', demonstrations_caption_list[3], 1)
                text_input = text_input.replace('<sent>', demonstrations_answer_list[3], 1)
                text_input = text_input.replace('<sent>', caption, 1)
            else:
                text_input = prompt.replace('<sent>', demonstrations_text_list[0], 1)
                text_input = text_input.replace('<sent>', demonstrations_answer_list[0], 1)
                text_input = text_input.replace('<sent>', demonstrations_text_list[3], 1)
                text_input = text_input.replace('<sent>', demonstrations_answer_list[3], 1)
                text_input = text_input.replace('<sent>', sent, 1)
        elif prompt_generation_args.demonstration_num == 3:
            if 'Mistral' in prompt_generation_args.prompt_generation_model:
                prompt = mistral_prompt_generation_text_modal_only_prompt_3
            else:
                prompt = llama_prompt_generation_text_modal_only_prompt_3
            if prompt_generation_args.case_type == 'caption':
                text_input = prompt.replace('<sent>', demonstrations_caption_list[0], 1)
                text_input = text_input.replace('<sent>', demonstrations_answer_list[0], 1)
                text_input = text_input.replace('<sent>', demonstrations_caption_list[1], 1)
                text_input = text_input.replace('<sent>', demonstrations_answer_list[1], 1)
                text_input = text_input.replace('<sent>', demonstrations_caption_list[2], 1)
                text_input = text_input.replace('<sent>', demonstrations_answer_list[2], 1)
                text_input = text_input.replace('<sent>', caption, 1)
            else:
                text_input = prompt.replace('<sent>', demonstrations_caption_list[0], 1)
                text_input = text_input.replace('<sent>', demonstrations_answer_list[0], 1)
                text_input = text_input.replace('<sent>', demonstrations_caption_list[1], 1)
                text_input = text_input.replace('<sent>', demonstrations_answer_list[1], 1)
                text_input = text_input.replace('<sent>', demonstrations_caption_list[2], 1)
                text_input = text_input.replace('<sent>', demonstrations_answer_list[2], 1)
                text_input = text_input.replace('<sent>', sent, 1)
        elif prompt_generation_args.demonstration_num == 4:
            if 'Mistral' in prompt_generation_args.prompt_generation_model:
                prompt = mistral_prompt_generation_text_modal_only_prompt_4
            else:
                prompt = llama_prompt_generation_text_modal_only_prompt_4
            if prompt_generation_args.case_type == 'caption':
                text_input = prompt.replace('<sent>', demonstrations_caption_list[0], 1)
                text_input = text_input.replace('<sent>', demonstrations_answer_list[0], 1)
                text_input = text_input.replace('<sent>', demonstrations_caption_list[1], 1)
                text_input = text_input.replace('<sent>', demonstrations_answer_list[1], 1)
                text_input = text_input.replace('<sent>', demonstrations_caption_list[2], 1)
                text_input = text_input.replace('<sent>', demonstrations_answer_list[2], 1)
                text_input = text_input.replace('<sent>', demonstrations_caption_list[3], 1)
                text_input = text_input.replace('<sent>', demonstrations_answer_list[3], 1)
                text_input = text_input.replace('<sent>', caption, 1)
            else:
                text_input = prompt.replace('<sent>', demonstrations_caption_list[0], 1)
                text_input = text_input.replace('<sent>', demonstrations_answer_list[0], 1)
                text_input = text_input.replace('<sent>', demonstrations_caption_list[1], 1)
                text_input = text_input.replace('<sent>', demonstrations_answer_list[1], 1)
                text_input = text_input.replace('<sent>', demonstrations_caption_list[2], 1)
                text_input = text_input.replace('<sent>', demonstrations_answer_list[2], 1)
                text_input = text_input.replace('<sent>', demonstrations_caption_list[3], 1)
                text_input = text_input.replace('<sent>', demonstrations_answer_list[3], 1)
                text_input = text_input.replace('<sent>', sent, 1)
        print(text_input)
        inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=500)
        if dist.get_rank() == 0:
            print('Here is the original output')
            print(tokenizer.decode(output[0], skip_special_tokens=True))
            print('Here is the filtered output')
            print(tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True))


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