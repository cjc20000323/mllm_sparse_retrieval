from transformers import LlavaProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, \
    LlavaNextForConditionalGeneration, Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration, AutoModel, \
    AutoProcessor, LlamaForCausalLM
from dataset import CrossModalRetrievalDataset
import torch.utils.data as Data
from tqdm import tqdm
from nltk import word_tokenize
from nltk.corpus import stopwords
import string

processor = LlavaNextProcessor.from_pretrained('./checkpoints/llava-hf-llama3-llava-next-8b-hf')

dataset = CrossModalRetrievalDataset('flickr', processor, 'test', 'full')

test_dataloader = Data.DataLoader(dataset=dataset, pin_memory=True,
                                  batch_size=4, shuffle=False)

length = 0
count = 0
for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(test_dataloader),
                                                             total=len(test_dataloader)):
    for text in texts:
        words = [i for i in word_tokenize(text.lower()) if
                 i not in set(stopwords.words('english') + list(string.punctuation))]
        token_ids = set()
        for word in words:
            token_ids.update(processor.tokenizer.encode(word, add_special_tokens=False))
        length += len(token_ids)
        count += 1

print(length / count)
