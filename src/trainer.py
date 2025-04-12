import os
import sys
from typing import List

import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset, load_from_disk
import transformers
from transformers import Trainer
import torch.distributed as dist

NIL_DATASET = True

from transformers import LlamaTokenizer, LlamaConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import set_seed
from transformers import BitsAndBytesConfig

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass

from transformers.utils import logging
from transformers.trainer_callback import TrainerCallback


class DenseEmbTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        texts = inputs['texts']

        loss_fct = nn.CrossEntropyLoss()


        return 0