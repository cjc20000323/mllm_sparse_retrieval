import logging
import os
from tevatron.retriever.dataset import EncodeDataset, TrainDataset
from typing import Tuple, List
from nltk.corpus import stopwords
import string
from dataclasses import dataclass, field
stopwords = set(stopwords.words('english') + list(string.punctuation))
from tevatron.retriever.arguments import ModelArguments, DataArguments, TevatronTrainingArguments

logger = logging.getLogger(__name__)


class PromptRepsEncodeDataset(EncodeDataset):
    def __getitem__(self, item) -> Tuple[str, str]:
        text = self.encode_data[item]
        if self.data_args.encode_is_query:
            text_id = text['query_id']
            formated_text = text['query'].strip()
        else:
            text_id = text['docid']
            formated_text = f"{text['title']} {text['text']}".strip()
        return text_id, formated_text

@dataclass
class PromptRepsDataArguments(DataArguments):
    query_suffix: str = field(
        default='', metadata={"help": "suffix or instruction for query"}
    )
    passage_suffix: str = field(
        default='', metadata={"help": "suffix or instruction for passage"}
    )
    dense_output_dir: str = field(default=None, metadata={"help": "where to save the encode dense vectors"})
    sparse_output_dir: str = field(default=None, metadata={"help": "where to save the encode dense vectors"})
    num_expended_tokens: int = field(default=0, metadata={"help": "Number of expended tokens. Default is 0, "
                                                                  "meaning exact term matching only."})
    num_pooled_tokens: int = field(default=0, metadata={"help": "Number of tokens to form the embeddings."})
    multi_reps: bool = field(default=False, metadata={"help": "Whether to use multiple representations for retrieval (ColBERT style)"})
    word_level_reps: bool = field(default=False, metadata={"help": "Whether to use word level representations for retrieval"})

    def __post_init__(self):
        if os.path.exists(self.query_prefix):
            with open(self.query_prefix, 'r') as f:
                self.query_prefix = f.read().strip()

        if os.path.exists(self.query_suffix):
            with open(self.query_suffix, 'r') as f:
                self.query_suffix = f.read().strip()

        if os.path.exists(self.passage_prefix):
            with open(self.passage_prefix, 'r') as f:
                self.passage_prefix = f.read().strip()

        if os.path.exists(self.passage_suffix):
            with open(self.passage_suffix, 'r') as f:
                self.passage_suffix = f.read().strip()


data_args = PromptRepsDataArguments()
data_args.dataset_name == 'Tevatron/beir-corpus'
data_args.dataset_config == 'nfcorpus'
encode_dataset = PromptRepsEncodeDataset(
        data_args=data_args,
    )