import torch
from matplotlib import pyplot as plt
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

encoder = LlavaNextForConditionalGeneration.from_pretrained(model_args.model_name_or_path,
                                                                    device_map=device_map,
                                                                    torch_dtype=torch.float16)
processor = LlavaNextProcessor.from_pretrained(model_args.model_name_or_path)