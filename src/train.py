import os

import transformers
from transformers import (
    HfArgumentParser,
    BitsAndBytesConfig
)
from transformers import LlavaProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, \
    LlavaNextForConditionalGeneration, Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration, AutoModel, \
    AutoProcessor, \
    AutoTokenizer, PhiForCausalLM, Phi3ForCausalLM, AutoModelForCausalLM, LlavaNextConfig
from arguments import PromptRepsLLMDataArguments, ModelArguments
import torch.distributed as dist
from arguments import TrainingArguments
from dataset import CrossModalRetrievalDataset, PromptRepsTrainCollator
import torch
import torch.utils.data as Data
import torch.nn.functional as F

from template import text_prompt, img_prompt, text_prompt_no_one_word, img_prompt_no_one_word, \
    img_prompt_no_special_llava_v1_5, text_prompt_no_special_llava_v1_5, text_prompt_qwen_v2_5, img_prompt_qwen_v2_5, \
    img_prompt_intern_vl_v2_5, text_prompt_intern_vl_v2_5
from model import MLLMRetrievalModel
from utils import split_model, load_image, find_all_linear_names
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from constant import llava_next_llama_8b_constant
from trainer import DenseEmbTrainer


def main():
    parser = HfArgumentParser((ModelArguments, PromptRepsLLMDataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: PromptRepsLLMDataArguments
    training_args: TrainingArguments

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

    # accelerator = Accelerator()

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
        if training_args.load_kbit == 4:
            encoder = LlavaNextForConditionalGeneration.from_pretrained(model_args.model_name_or_path,
                                                                        quantization_config=BitsAndBytesConfig(
                                                                            load_in_4bit=True,
                                                                            bnb_4bit_compute_dtype=torch_type,
                                                                            bnb_4bit_use_double_quant=True,
                                                                            bnb_4bit_quant_type='nf4'
                                                                        ),
                                                                        device_map=device_map,
                                                                        torch_dtype=torch_type)
        else:
            encoder = LlavaNextForConditionalGeneration.from_pretrained(model_args.model_name_or_path,
                                                                        load_in_8bit=training_args.load_kbit == 8,
                                                                        load_in_4bit=training_args.load_kbit == 4,
                                                                        device_map=device_map,
                                                                        torch_dtype=torch_type)
        processor = LlavaNextProcessor.from_pretrained(model_args.model_name_or_path)
        if 'royokong-e5-v' in model_args.model_name_or_path:
            setattr(processor, "patch_size", 14)  # hack for pass

    lora_modules = []
    full_modules = []
    if model_args.lora:
        if training_args.load_kbit == 4 or training_args.load_kbit == 8:
            encoder = prepare_model_for_kbit_training(encoder)
        if 'llama3-llava-next-8b' in model_args.model_name_or_path:
            target_modules = find_all_linear_names(encoder, llava_next_llama_8b_constant['llm'])
            lora_modules.extend(target_modules)
        else:
            target_modules = find_all_linear_names(encoder, llava_next_llama_8b_constant['llm'])
            lora_modules.extend(target_modules)

        if training_args.train_vision_lora:
            if 'llama3-llava-next-8b' in model_args.model_name_or_path:
                target_modules = find_all_linear_names(encoder, llava_next_llama_8b_constant['vision_encoder'])
                lora_modules.extend(target_modules)
            else:
                target_modules = find_all_linear_names(encoder, llava_next_llama_8b_constant['vision_encoder'])
                lora_modules.extend(target_modules)

        if training_args.train_vision_lora:
            if 'llama3-llava-next-8b' in model_args.model_name_or_path:
                target_modules = find_all_linear_names(encoder, llava_next_llama_8b_constant['projector'])
                lora_modules.extend(target_modules)
            else:
                target_modules = find_all_linear_names(encoder, llava_next_llama_8b_constant['projector'])
                lora_modules.extend(target_modules)

        '''
        if dist.get_rank() == 0:
            print(lora_modules)
        '''

        config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=lora_modules,
            lora_dropout=model_args.lora_dropout,
            bias=model_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        encoder = get_peft_model(encoder, config)


    else:
        pass

    model = MLLMRetrievalModel(encoder)

    if dist.get_rank() == 0:
        '''
        for name, param in model.named_parameters():
            print(f"\t{name} {param.requires_grad}")
        '''

        for name, param in model.named_parameters():
            print(f"Param ID: {id(param)}, Name: {name}")

    train_dataset = CrossModalRetrievalDataset(data_args.dataset_name, processor, 'train', 'single', data_args)

    data_collator = PromptRepsTrainCollator(processor, model_args, device)

    if training_args.train_mode == 'dense_emb':
        trainer = DenseEmbTrainer(
            model=model,
            train_dataset=train_dataset,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=training_args.per_device_train_batch_size,
                gradient_accumulation_steps=training_args.gradient_accumulation_steps,
                warmup_steps=10,
                num_train_epochs=training_args.num_train_epochs,
                learning_rate=training_args.learning_rate,
                fp16=True if training_args.fp16 else False,
                bf16=True if training_args.bf16 else False,
                eval_strategy="no",
                save_strategy="no",
                eval_steps=None,
                output_dir=training_args.output_dir,
                save_total_limit=100,
                load_best_model_at_end=False,
                # ddp_find_unused_parameters=False if ddp else None,
                ddp_find_unused_parameters=False if ddp else None,
                report_to=None,
                deepspeed=training_args.deepspeed,
                logging_steps=1,
                gradient_checkpointing_kwargs={"use_reentrant": False}
            ),
            data_collator=data_collator,
        )
        if dist.get_rank() == 0:
            print('Trainer has been created.')
    else:
        trainer = DenseEmbTrainer(
            model=model,
            train_dataset=train_dataset,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=training_args.per_device_train_batch_size,
                gradient_accumulation_steps=training_args.gradient_accumulation_steps,
                warmup_steps=10,
                num_train_epochs=training_args.num_train_epochs,
                learning_rate=training_args.learning_rate,
                fp16=True if training_args.fp16 else False,
                bf16=True if training_args.bf16 else False,
                eval_strategy="no",
                save_strategy="steps",
                eval_steps=None,
                output_dir=training_args.output_dir,
                save_total_limit=100,
                load_best_model_at_end=False,
                # ddp_find_unused_parameters=False if ddp else None,
                ddp_find_unused_parameters=False if ddp else None,
                report_to=None,
                deepspeed=training_args.deepspeed,
                logging_steps=1,
                gradient_checkpointing_kwargs={"use_reentrant": False}
            ),
            data_collator=data_collator,
        )
        if dist.get_rank() == 0:
            print('Trainer has been created.')

    trainer.model_args = model_args
    trainer.data_args = data_args
    trainer.device = device
    trainer.processor = processor
    trainer.gather_save_gradient = training_args.gather_save_gradient
    trainer.tau = training_args.tau
    trainer.train()

    model.encoder.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
