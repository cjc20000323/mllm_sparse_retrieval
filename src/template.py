llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'
llama3_template_image_prefix = '<|start_header_id|>user<|end_header_id|>\n\n<image>\n'
llama3_template_text_prefix = '<|start_header_id|>user<|end_header_id|>\n\n<sent>\n'
llama3_template_content_element = '<|begin_of_text|>{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n<|end_of_text|>'
llama3_template_fashion_iq_composed_image_prefix = '<|start_header_id|>user<|end_header_id|>\n\n<image> change the style of this {} to <sent>\n'
llama3_template_fashion_iq_image_prefix = '<|start_header_id|>user<|end_header_id|>\n\n<image>\n'
llama3_template_fashion_iq_text_prefix = '<|start_header_id|>user<|end_header_id|>\n\n<sent>\n'
llava_mistral_template = '[INST]{}[/INST]'
llava_mistral_template_image_prefix = '[INST]<image>\n'
llava_mistral_template_text_prefix = '[INST]<sent>\n'
llava_mistral_template_content_element = '<s>{}[/INST]</s>'
llava_mistral_template_fashion_iq_composed_image_prefix = '[INST]<image> change the style of this {} to <sent>\n'
llava_mistral_template_fashion_iq_image_prefix = '[INST]<image>\n'
llava_mistral_template_fashion_iq_text_prefix = '[INST]<sent>\n'

qwen2_5_template = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant'
qwen2_5_template_image_prefix = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n'
qwen2_5_template_text_prefix = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<sent>\n'
qwen2_5_template_content_element = '<tool_call>{}<|im_end|>\n<|im_start|>assistant</tool_call>'
qwen3_template = '<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant'
qwen3_template_image_prefix = '<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n'
qwen3_template_text_prefix = '<|im_start|>user\n<sent>\n'
qwen3_template_content_element = '<think>{}<|im_end|>\n<|im_start|>assistant</think>'
llava_v1_5_template = '<s>user\n\n{}</s><s>assistant\n\n \n'
img_prompt = llama3_template.format('<image>\n<|begin_of_text|>Summary above image in one word: ')
text_prompt = llama3_template.format('<sent>\n<|begin_of_text|>Summary above sentence in one word: ')
mistral_img_prompt = llava_mistral_template.format('<image>\n<s>Summary above image in one word: ')
mistral_text_prompt = llava_mistral_template.format('<sent>\n<s>Summary above sentence in one word: ')
qwen2_5_img_prompt = qwen2_5_template.format('<image>\n<tool_call>Summary above image in one word: ')
qwen2_5_text_prompt = qwen2_5_template.format('<sent>\n</tool_call>Summary above sentence in one word: ')
qwen3_img_prompt = qwen3_template.format('<image>\n<think>Summary above image in one word: ')
qwen3_text_prompt = qwen3_template.format('<sent>\n<think>Summary above sentence in one word: ')

person_retrieval_img_prompt = llama3_template.format('<image>\n<|begin_of_text|>Summary the person in above image in one word: ')
person_retrieval_text_prompt = llama3_template.format('<sent>\n<|begin_of_text|>Summary the person in above sentence in one word: ')
mistral_person_retrieval_img_prompt = llava_mistral_template.format('<image>\n<s>Summary the person in above image in one word: ')
mistral_person_retrieval_text_prompt = llava_mistral_template.format('<sent>\n<s>Summary the person in above sentence in one word: ')
person_retrieval_img_prompt_1 = llama3_template.format('<image>\n<|begin_of_text|>Describe this person in one word based on the image: ')
person_retrieval_text_prompt_1 = llama3_template.format('<sent>\n<|begin_of_text|>Describe this person in one word based on the sentence: ')
mistral_person_retrieval_img_prompt_1 = llava_mistral_template.format('<image>\n<s>Describe this person in one word based on its image: ')
mistral_person_retrieval_text_prompt_1 = llava_mistral_template.format('<sent>\n<s>Describe this person in one word based on its sentence: ')
person_retrieval_img_prompt_2 = llama3_template.format('<image>\n<|begin_of_text|>Describe wearing style of this person in one word based on the image: ')
person_retrieval_text_prompt_2 = llama3_template.format('<sent>\n<|begin_of_text|>Describe wearing style of this person in one word based on the sentence: ')
mistral_person_retrieval_img_prompt_2 = llava_mistral_template.format('<image>\n<s>Describe wearing style of this person in one word based on the image: ')
mistral_person_retrieval_text_prompt_2 = llava_mistral_template.format('<sent>\n<s>Describe wearing style of this person in one word based on the sentence: ')


llama3_fashion_iq_composed_image_prompt = llama3_template.format('<image> change the style of this {} to <sent>\n<|begin_of_text|>Describe this modified {} in one word based on its style: ')
mistral_fashion_iq_composed_image_prompt = llava_mistral_template.format('<image> change the style of this {} to <sent>\n<s>Describe this modified {} in one word based on its style: ')
llama3_fashion_iq_image_prompt = llama3_template.format('<image>\n<|begin_of_text|>Describe this {} in one word based on its style: ')
mistral_fashion_iq_image_prompt = llava_mistral_template.format('<image>\n<s>Describe this {} in one word based on its style: ')

relevant_prompt = llama3_template.format("For the following sentence and image, judge whether they are relevant. Output 'Yes' or 'No'.\nSentence: <sent> Image: <image> Output: ")
in_one_word_relevant_prompt = llama3_template.format("For the following sentence and image, judge whether they are relevant. Output 'Yes' or 'No'.\nSentence: <sent> Image: <image> Output in one word: ")
please_relevant_prompt = llama3_template.format("For the following sentence and image, judge whether they are relevant. Please output 'Yes' or 'No'.\nSentence: <sent> Image: <image> Output: ")
text_query_relevant_prompt = llama3_template.format("For the following query sentence and candidate image, judge whether they are relevant. Output 'Yes' or 'No'.\nQuery Sentence: <sent> Candidate Image: <image> Output: ")
image_query_relevant_prompt = llama3_template.format("For the following query image and candidate sentence, judge whether they are relevant. Output 'Yes' or 'No'.\nQuery Image: <image> Candidate Sentence: <sent> Output: ")
old_text_query_relevant_prompt = llama3_template.format("Query: <sent>\nCandidate: <image>\n Does the candidate answer the query?  Answer 'Yes' or 'No'.  Answer: ")
old_image_query_relevant_prompt = llama3_template.format("Query: <image>\nCandidate: <sent>\n Does the candidate answer the query?  Answer 'Yes' or 'No'.  Answer: ")
old_text_reverse_query_relevant_prompt = llama3_template.format("Candidate: <image>\nQuery: <sent>\n Does the candidate answer the query?  Answer 'Yes' or 'No'.  Answer: ")
old_image_reverse_query_relevant_prompt = llama3_template.format("Candidate: <sent>\nQuery: <image>\n Does the candidate answer the query?  Answer 'Yes' or 'No'.  Answer: ")
origin_old_text_query_relevant_prompt = llama3_template.format("Given a candidate and a query, predict whether the candidate includes an answer to the query by producing either ‘Yes‘ or ‘No‘.\nCandidate: <image>\nQuery: <sent>\nDoes the candidate answer the query? Answer: ")
origin_old_image_query_relevant_prompt = llama3_template.format("Given a candidate and a query, predict whether the candidate includes an answer to the query by producing either ‘Yes‘ or ‘No‘.\nCandidate: <sent>\nQuery: <image>\nDoes the candidate answer the query? Answer: ")
origin_old_text_reverse_query_relevant_prompt = llama3_template.format("Given a candidate and a query, predict whether the candidate includes an answer to the query by producing either ‘Yes‘ or ‘No‘.\nQuery: <sent>\nCandidate: <image>\nDoes the candidate answer the query? Answer: ")
origin_old_image_reverse_query_relevant_prompt = llama3_template.format("Given a candidate and a query, predict whether the candidate includes an answer to the query by producing either ‘Yes‘ or ‘No‘.\nQuery: <image>\nCandidate: <sent>\nDoes the candidate answer the query? Answer: ")
precise_caption_prompt = llama3_template.format("For the following sentence and image, judge whether the sentence is the precise caption of the image. Output 'Yes' or 'No'.\nSentence: <sent> Image: <image> Output: ")
first_precise_caption_prompt = llama3_template.format("Sentence: <sent>\n Image: <image>\n For the following sentence and image, judge whether the sentence is the precise caption of the image. Output 'Yes' or 'No'. Output: ")
role_relevant_prompt = llama3_template.format("You are RankGPT, an intelligent assistant that can rank candidates based on their relevancy to the query. \n For the following sentence and image, judge whether they are relevant. Output 'Yes' or 'No'.\nSentence: <sent> Image: <image> Output: ")
role_precise_caption_prompt = llama3_template.format("You are RankGPT, an intelligent assistant that can rank candidates based on their relevancy to the query. \n For the following sentence and image, judge whether the sentence is the precise caption of the image. Output 'Yes' or 'No'.\nSentence: <sent> Image: <image> Output: ")
role_old_text_query_relevant_prompt = llama3_template.format("You are RankGPT, an intelligent assistant that can rank candidates based on their relevancy to the query. \nQuery: <sent>\nCandidate: <image>\n Does the candidate answer the query?  Answer 'Yes' or 'No'.  Answer: ")
role_old_image_query_relevant_prompt = llama3_template.format("You are RankGPT, an intelligent assistant that can rank candidates based on their relevancy to the query. \nQuery: <image>\nCandidate: <sent>\n Does the candidate answer the query?  Answer 'Yes' or 'No'.  Answer: ")
mistral_relevant_prompt = llava_mistral_template.format("For the following sentence and image, judge whether they are relevant. Output 'Yes' or 'No'.\nSentence: <sent> Image: <image> Output: ")
mistral_in_one_word_relevant_prompt = llava_mistral_template.format("For the following sentence and image, judge whether they are relevant. Output 'Yes' or 'No'.\nSentence: <sent> Image: <image> Output in one word: ")
mistral_please_relevant_prompt = llava_mistral_template.format("For the following sentence and image, judge whether they are relevant. Please output 'Yes' or 'No'.\nSentence: <sent> Image: <image> Output: ")
mistral_text_query_relevant_prompt = llava_mistral_template.format("For the following query sentence and candidate image, judge whether they are relevant. Output 'Yes' or 'No'.\nQuery Sentence: <sent> Candidate Image: <image> Output: ")
mistral_image_query_relevant_prompt = llava_mistral_template.format("For the following query image and candidate sentence, judge whether they are relevant. Output 'Yes' or 'No'.\nQuery Image: <image> Candidate Sentence: <sent> Output: ")
mistral_old_text_query_relevant_prompt = llava_mistral_template.format("Query: <sent>\nCandidate: <image>\n Does the candidate answer the query?  Answer 'Yes' or 'No'.  Answer: ")
mistral_old_image_query_relevant_prompt = llava_mistral_template.format("Query: <image>\nCandidate: <sent>\n Does the candidate answer the query?  Answer 'Yes' or 'No'.  Answer: ")
mistral_old_text_reverse_query_relevant_prompt = llava_mistral_template.format("Candidate: <image>\nQuery: <sent>\n Does the candidate answer the query?  Answer 'Yes' or 'No'.  Answer: ")
mistral_old_image_reverse_query_relevant_prompt = llava_mistral_template.format("Candidate: <sent>\nQuery: <image>\n Does the candidate answer the query?  Answer 'Yes' or 'No'.  Answer: ")
mistral_origin_old_text_query_relevant_prompt = llava_mistral_template.format("Given a candidate and a query, predict whether the candidate includes an answer to the query by producing either ‘Yes‘ or ‘No‘.\nCandidate: <image>\nQuery: <sent>\nDoes the candidate answer the query? Answer: ")
mistral_origin_old_image_query_relevant_prompt = llava_mistral_template.format("Given a candidate and a query, predict whether the candidate includes an answer to the query by producing either ‘Yes‘ or ‘No‘.\nCandidate: <sent>\nQuery: <image>\nDoes the candidate answer the query? Answer: ")
mistral_origin_old_image_reverse_query_relevant_prompt = llava_mistral_template.format("Given a candidate and a query, predict whether the candidate includes an answer to the query by producing either ‘Yes‘ or ‘No‘.\nQuery: <image>\nCandidate: <sent>\nDoes the candidate answer the query? Answer: ")
mistral_origin_old_text_reverse_query_relevant_prompt = llava_mistral_template.format("Given a candidate and a query, predict whether the candidate includes an answer to the query by producing either ‘Yes‘ or ‘No‘.\nQuery: <sent>\nCandidate: <image>\nDoes the candidate answer the query? Answer: ")
mistral_precise_caption_prompt = llava_mistral_template.format("For the following sentence and image, judge whether the sentence is the precise caption of the image. Output 'Yes' or 'No'.\nSentence: <sent> Image: <image> Output: ")
mistral_first_precise_caption_prompt = llava_mistral_template.format("Sentence: <sent>\n Image: <image>\n For the following sentence and image, judge whether the sentence is the precise caption of the image. Output 'Yes' or 'No'. Output: ")
mistral_role_relevant_prompt = llava_mistral_template.format("You are RankGPT, an intelligent assistant that can rank candidates based on their relevancy to the query. \n For the following sentence and image, judge whether they are relevant. Output 'Yes' or 'No'.\nSentence: <sent> Image: <image> Output: ")
mistral_role_precise_caption_prompt = llava_mistral_template.format("You are RankGPT, an intelligent assistant that can rank candidates based on their relevancy to the query. \n For the following sentence and image, judge whether the sentence is the precise caption of the image. Output 'Yes' or 'No'.\nSentence: <sent> Image: <image> Output: ")
mistral_role_old_text_query_relevant_prompt = llava_mistral_template.format("You are RankGPT, an intelligent assistant that can rank candidates based on their relevancy to the query. \nQuery: <sent>\nCandidate: <image>\n Does the candidate answer the query?  Answer 'Yes' or 'No'.  Answer: ")
mistral_role_old_image_query_relevant_prompt = llava_mistral_template.format("You are RankGPT, an intelligent assistant that can rank candidates based on their relevancy to the query. \nQuery: <image>\nCandidate: <sent>\n Does the candidate answer the query?  Answer 'Yes' or 'No'.  Answer: ")

qwen2_5_relevant_prompt = qwen2_5_template.format("For the following sentence and image, judge whether they are relevant. Output 'Yes' or 'No'.\nSentence: <sent> Image: <image> Output: ")
qwen2_5_in_one_word_relevant_prompt = qwen2_5_template.format("For the following sentence and image, judge whether they are relevant. Output 'Yes' or 'No'.\nSentence: <sent> Image: <image> Output in one word: ")
qwen2_5_please_relevant_prompt = qwen2_5_template.format("For the following sentence and image, judge whether they are relevant. Please output 'Yes' or 'No'.\nSentence: <sent> Image: <image> Output: ")
qwen2_5_text_query_relevant_prompt = qwen2_5_template.format("For the following query sentence and candidate image, judge whether they are relevant. Output 'Yes' or 'No'.\nQuery Sentence: <sent> Candidate Image: <image> Output: ")
qwen2_5_image_query_relevant_prompt = qwen2_5_template.format("For the following query image and candidate sentence, judge whether they are relevant. Output 'Yes' or 'No'.\nQuery Image: <image> Candidate Sentence: <sent> Output: ")
qwen2_5_old_text_query_relevant_prompt = qwen2_5_template.format("Query: <sent>\nCandidate: <image>\n Does the candidate answer the query?  Answer 'Yes' or 'No'.  Answer: ")
qwen2_5_old_image_query_relevant_prompt = qwen2_5_template.format("Query: <image>\nCandidate: <sent>\n Does the candidate answer the query?  Answer 'Yes' or 'No'.  Answer: ")
qwen2_5_old_text_reverse_query_relevant_prompt = qwen2_5_template.format("Candidate: <image>\nQuery: <sent>\n Does the candidate answer the query?  Answer 'Yes' or 'No'.  Answer: ")
qwen2_5_old_image_reverse_query_relevant_prompt = qwen2_5_template.format("Candidate: <sent>\nQuery: <image>\n Does the candidate answer the query?  Answer 'Yes' or 'No'.  Answer: ")
qwen2_5_origin_old_text_query_relevant_prompt = qwen2_5_template.format("Given a candidate and a query, predict whether the candidate includes an answer to the query by producing either ‘Yes‘ or ‘No‘.\nCandidate: <image>\nQuery: <sent>\nDoes the candidate answer the query? Answer: ")
qwen2_5_origin_old_image_query_relevant_prompt = qwen2_5_template.format("Given a candidate and a query, predict whether the candidate includes an answer to the query by producing either ‘Yes‘ or ‘No‘.\nCandidate: <sent>\nQuery: <image>\nDoes the candidate answer the query? Answer: ")
qwen2_5_origin_old_image_reverse_query_relevant_prompt = qwen2_5_template.format("Given a candidate and a query, predict whether the candidate includes an answer to the query by producing either ‘Yes‘ or ‘No‘.\nQuery: <image>\nCandidate: <sent>\nDoes the candidate answer the query? Answer: ")
qwen2_5_origin_old_text_reverse_query_relevant_prompt = qwen2_5_template.format("Given a candidate and a query, predict whether the candidate includes an answer to the query by producing either ‘Yes‘ or ‘No‘.\nQuery: <sent>\nCandidate: <image>\nDoes the candidate answer the query? Answer: ")
qwen2_5_precise_caption_prompt = qwen2_5_template.format("For the following sentence and image, judge whether the sentence is the precise caption of the image. Output 'Yes' or 'No'.\nSentence: <sent> Image: <image> Output: ")
qwen2_5_first_precise_caption_prompt = qwen2_5_template.format("Sentence: <sent>\n Image: <image>\n For the following sentence and image, judge whether the sentence is the precise caption of the image. Output 'Yes' or 'No'. Output: ")
qwen2_5_role_relevant_prompt = qwen2_5_template.format("You are RankGPT, an intelligent assistant that can rank candidates based on their relevancy to the query. \n For the following sentence and image, judge whether they are relevant. Output 'Yes' or 'No'.\nSentence: <sent> Image: <image> Output: ")
qwen2_5_role_precise_caption_prompt = qwen2_5_template.format("You are RankGPT, an intelligent assistant that can rank candidates based on their relevancy to the query. \n For the following sentence and image, judge whether the sentence is the precise caption of the image. Output 'Yes' or 'No'.\nSentence: <sent> Image: <image> Output: ")
qwen2_5_role_old_text_query_relevant_prompt = qwen2_5_template.format("You are RankGPT, an intelligent assistant that can rank candidates based on their relevancy to the query. \nQuery: <sent>\nCandidate: <image>\n Does the candidate answer the query?  Answer 'Yes' or 'No'.  Answer: ")
qwen2_5_role_old_image_query_relevant_prompt = qwen2_5_template.format("You are RankGPT, an intelligent assistant that can rank candidates based on their relevancy to the query. \nQuery: <image>\nCandidate: <sent>\n Does the candidate answer the query?  Answer 'Yes' or 'No'.  Answer: ")


qwen3_relevant_prompt = qwen3_template.format("For the following sentence and image, judge whether they are relevant. Output 'Yes' or 'No'.\nSentence: <sent> Image: <image> Output: ")
qwen3_in_one_word_relevant_prompt = qwen3_template.format("For the following sentence and image, judge whether they are relevant. Output 'Yes' or 'No'.\nSentence: <sent> Image: <image> Output in one word: ")
qwen3_please_relevant_prompt = qwen3_template.format("For the following sentence and image, judge whether they are relevant. Please output 'Yes' or 'No'.\nSentence: <sent> Image: <image> Output: ")
qwen3_text_query_relevant_prompt = qwen3_template.format("For the following query sentence and candidate image, judge whether they are relevant. Output 'Yes' or 'No'.\nQuery Sentence: <sent> Candidate Image: <image> Output: ")
qwen3_image_query_relevant_prompt = qwen3_template.format("For the following query image and candidate sentence, judge whether they are relevant. Output 'Yes' or 'No'.\nQuery Image: <image> Candidate Sentence: <sent> Output: ")
qwen3_old_text_query_relevant_prompt = qwen3_template.format("Query: <sent>\nCandidate: <image>\n Does the candidate answer the query?  Answer 'Yes' or 'No'.  Answer: ")
qwen3_old_image_query_relevant_prompt = qwen3_template.format("Query: <image>\nCandidate: <sent>\n Does the candidate answer the query?  Answer 'Yes' or 'No'.  Answer: ")
qwen3_old_text_reverse_query_relevant_prompt = qwen3_template.format("Candidate: <image>\nQuery: <sent>\n Does the candidate answer the query?  Answer 'Yes' or 'No'.  Answer: ")
qwen3_old_image_reverse_query_relevant_prompt = qwen3_template.format("Candidate: <sent>\nQuery: <image>\n Does the candidate answer the query?  Answer 'Yes' or 'No'.  Answer: ")
qwen3_origin_old_text_query_relevant_prompt = qwen3_template.format("Given a candidate and a query, predict whether the candidate includes an answer to the query by producing either ‘Yes‘ or ‘No‘.\nCandidate: <image>\nQuery: <sent>\nDoes the candidate answer the query? Answer: ")
qwen3_origin_old_image_query_relevant_prompt = qwen3_template.format("Given a candidate and a query, predict whether the candidate includes an answer to the query by producing either ‘Yes‘ or ‘No‘.\nCandidate: <sent>\nQuery: <image>\nDoes the candidate answer the query? Answer: ")
qwen3_origin_old_image_reverse_query_relevant_prompt = qwen3_template.format("Given a candidate and a query, predict whether the candidate includes an answer to the query by producing either ‘Yes‘ or ‘No‘.\nQuery: <image>\nCandidate: <sent>\nDoes the candidate answer the query? Answer: ")
qwen3_origin_old_text_reverse_query_relevant_prompt = qwen3_template.format("Given a candidate and a query, predict whether the candidate includes an answer to the query by producing either ‘Yes‘ or ‘No‘.\nQuery: <sent>\nCandidate: <image>\nDoes the candidate answer the query? Answer: ")
qwen3_precise_caption_prompt = qwen3_template.format("For the following sentence and image, judge whether the sentence is the precise caption of the image. Output 'Yes' or 'No'.\nSentence: <sent> Image: <image> Output: ")
qwen3_first_precise_caption_prompt = qwen3_template.format("Sentence: <sent>\n Image: <image>\n For the following sentence and image, judge whether the sentence is the precise caption of the image. Output 'Yes' or 'No'. Output: ")
qwen3_role_relevant_prompt = qwen3_template.format("You are RankGPT, an intelligent assistant that can rank candidates based on their relevancy to the query. \n For the following sentence and image, judge whether they are relevant. Output 'Yes' or 'No'.\nSentence: <sent> Image: <image> Output: ")
qwen3_role_precise_caption_prompt = qwen3_template.format("You are RankGPT, an intelligent assistant that can rank candidates based on their relevancy to the query. \n For the following sentence and image, judge whether the sentence is the precise caption of the image. Output 'Yes' or 'No'.\nSentence: <sent> Image: <image> Output: ")
qwen3_role_old_text_query_relevant_prompt = qwen3_template.format("You are RankGPT, an intelligent assistant that can rank candidates based on their relevancy to the query. \nQuery: <sent>\nCandidate: <image>\n Does the candidate answer the query?  Answer 'Yes' or 'No'.  Answer: ")
qwen3_role_old_image_query_relevant_prompt = qwen3_template.format("You are RankGPT, an intelligent assistant that can rank candidates based on their relevancy to the query. \nQuery: <image>\nCandidate: <sent>\n Does the candidate answer the query?  Answer 'Yes' or 'No'.  Answer: ")


fashion_iq_relevant_prompt = llama3_template.format("For the following modified {} and target {}, judge whether they are relevant. Output 'Yes' or 'No'.\nModified {}: <image> change the style of this {} to <sent> Target {}: <image> Output: ")
fashion_iq_old_query_relevant_prompt = llama3_template.format("Query Modified {}: <image> change the style of this {} to <sent>\nCandidate {}: <image>\n Does the candidate answer the query?  Answer 'Yes' or 'No'.  Answer: ")
fashion_iq_origin_old_query_relevant_prompt = llama3_template.format("Given a candidate {} and a query modified {}, predict whether the candidate includes an answer to the query by producing either ‘Yes‘ or ‘No‘.\nCandidate {}: <image>\nQuery Modified {}: <image> change the style of this {} to <sent>\nDoes the candidate answer the query? Answer: ")
fashion_iq_query_relevant_prompt = llama3_template.format("For the following query modified {} and candidate {}, judge whether they are relevant. Output 'Yes' or 'No'.\nQuery Modified {}: <image> change the style of this {} to <sent> Candidate {}: <image> Output: ")
mistral_fashion_iq_relevant_prompt = llava_mistral_template.format("For the following modified {} and target {}, judge whether they are relevant. Output 'Yes' or 'No'.\nModified {}: <image> change the style of this {} to <sent> Target {}: <image> Output: ")
mistral_fashion_iq_old_query_relevant_prompt = llava_mistral_template.format("Query Modified {}: <image> change the style of this {} to <sent>\nCandidate {}: <image>\n Does the candidate answer the query?  Answer 'Yes' or 'No'.  Answer: ")
mistral_fashion_iq_origin_old_query_relevant_prompt = llava_mistral_template.format("Given a candidate {} and a query modified {}, predict whether the candidate includes an answer to the query by producing either ‘Yes‘ or ‘No‘.\nCandidate {}: <image>\nQuery Modified {}: <image> change the style of this {} to <sent>\nDoes the candidate answer the query? Answer: ")
mistral_fashion_iq_query_relevant_prompt = llava_mistral_template.format("For the following query modified {} and candidate {}, judge whether they are relevant. Output 'Yes' or 'No'.\nQuery Modified {}: <image> change the style of this {} to <sent> Candidate {}: <image> Output: ")
mistral_fashion_iq_role_relevant_prompt = llava_mistral_template.format("You are RankGPT, an intelligent assistant that can rank candidates based on their relevancy to the query. \n For the following modified {} and target {}, judge whether they are relevant. Output 'Yes' or 'No'.\nModified {}: <image> change the style of this {} to <sent> Target {}: <image> Output: ")
mistral_fashion_iq_role_old_query_relevant_prompt = llava_mistral_template.format("You are RankGPT, an intelligent assistant that can rank candidates based on their relevancy to the query. \nQuery Modified {}: <image> change the style of this {} to <sent>\nCandidate {}: <image>\n Does the candidate answer the query?  Answer 'Yes' or 'No'.  Answer: ")
mistral_fashion_iq_in_one_word_relevant_prompt = llava_mistral_template.format("For the following modified {} and target {}, judge whether they are relevant. Output 'Yes' or 'No'.\nModified {}: <image> change the style of this {} to <sent> Target {}: <image> Output: ")
mistral_fashion_iq_please_relevant_prompt = llava_mistral_template.format("For the following modified {} and target {}, judge whether they are relevant. Please output 'Yes' or 'No'.\nModified {}: <image> change the style of this {} to <sent> Target {}: <image> Output: ")

fashion_iq_modify_class_prompt = llama3_template.format("<sent>\nGiven a text that aims to modify the reference image, please classify the text into one of the seven modified perspectives: {}")
mistral_fashion_iq_modify_class_prompt = llava_mistral_template.format("<sent>\nGiven a text that aims to modify the reference image, please classify the text into one of the seven modified perspectives: {}")

person_retrieval_relevant_prompt = llama3_template.format("For the following sentence and person image, judge whether they are relevant. Output 'Yes' or 'No'.\nSentence: <sent> Person Image: <image> Output: ")
person_retrieval_old_query_relevant_prompt = llama3_template.format("Query: <sent>\nCandidate: <image>\n Does the candidate answer the query?  Answer 'Yes' or 'No'.  Answer: ")
person_retrieval_origin_old_query_relevant_prompt = llama3_template.format("Given a candidate and a query, predict whether the candidate includes an answer to the query by producing either ‘Yes‘ or ‘No‘.\nCandidate: <image>\nQuery: <sent>\nDoes the candidate answer the query? Answer: ")
person_retrieval_query_relevant_prompt = llama3_template.format("For the following query sentence and candidate person image, judge whether they are relevant. Output 'Yes' or 'No'.\nQuery Sentence: <sent> Candidate Person Image: <image> Output: ")
mistral_person_retrieval_relevant_prompt = llava_mistral_template.format("For the following sentence and person image, judge whether they are relevant. Output 'Yes' or 'No'.\nSentence: <sent> Person Image: <image> Output: ")
mistral_person_retrieval_old_query_relevant_prompt = llava_mistral_template.format("Query: <sent>\nCandidate: <image>\n Does the candidate answer the query?  Answer 'Yes' or 'No'.  Answer: ")
mistral_person_retrieval_origin_old_query_relevant_prompt = llava_mistral_template.format("Given a candidate and a query, predict whether the candidate includes an answer to the query by producing either ‘Yes‘ or ‘No‘.\nCandidate: <image>\nQuery: <sent>\nDoes the candidate answer the query? Answer: ")
mistral_person_retrieval_query_relevant_prompt = llava_mistral_template.format("For the following query sentence and candidate image, judge whether they are relevant. Output 'Yes' or 'No'.\nQuery Sentence: <sent> Candidate Image: <image> Output: ")

mistral_query_generation_paradigm_prompt = llava_mistral_template.format("Image: <image>\nPlease write a caption based on this image.")
query_generation_paradigm_prompt = llama3_template.format("Image: <image>\nPlease write a caption based on this image.")
qwen2_5_query_generation_paradigm_prompt = qwen2_5_template.format("Image: <image>\nPlease write a caption based on this image.")
qwen3_query_generation_paradigm_prompt = qwen3_template.format("Image: <image>\nPlease write a caption based on this image.")
mistral_query_generation_paradigm_prompt_1 = llava_mistral_template.format("Image: <image>\nWhat is the caption of the above image?")
query_generation_paradigm_prompt_1 = llama3_template.format("Image: <image>\nWhat is the caption of the above image?")
qwen2_5_query_generation_paradigm_prompt_1 = qwen2_5_template.format("Image: <image>\nWhat is the caption of the above image?")
qwen3_query_generation_paradigm_prompt_1 = qwen3_template.format("Image: <image>\nWhat is the caption of the above image?")
mistral_query_generation_paradigm_prompt_2 = llava_mistral_template.format("<image>\nPlease write a caption based on this image.")
query_generation_paradigm_prompt_2 = llama3_template.format("<image>\nPlease write a caption based on this image.")
mistral_query_generation_paradigm_prompt_3 = llava_mistral_template.format("<image>\nWhat is the caption of the above image?")
query_generation_paradigm_prompt_3 = llama3_template.format("<image>\nWhat is the caption of the above image?")
mistral_query_generation_paradigm_prompt_4 = llava_mistral_template.format("Image: <image>\nDescribe the image concisely.")
query_generation_paradigm_prompt_4 = llama3_template.format("Image: <image>\nDescribe the image concisely.")
mistral_query_generation_paradigm_prompt_5 = llava_mistral_template.format("Image: <image>\nProvide a brief description of the given image.")
query_generation_paradigm_prompt_5 = llama3_template.format("Image: <image>\nProvide a brief description of the given image.")
query_generation_paradigm_prompt_6 = llama3_template.format("<image>\n<s>Please write a caption based on this image.")
mistral_query_generation_paradigm_prompt_6 = llava_mistral_template.format("<image>\n<s>Please write a caption based on this image.")
query_generation_paradigm_prompt_7 = llama3_template.format("<image>\n<s>What is the caption of the above image?")
mistral_query_generation_paradigm_prompt_7 = llava_mistral_template.format("<image>\n<s>What is the caption of the above image?")
detailed_mistral_query_generation_paradigm_prompt = llava_mistral_template.format("Image: <image>\nProvide a detailed description of the given image.")
detailed_query_generation_paradigm_prompt = llama3_template.format("Image: <image>\nProvide a detailed description of the given image.")
detailed_mistral_query_generation_paradigm_prompt_1 = llava_mistral_template.format("Image: <image>\nGive an elaborate explanation of the image you see.")
detailed_query_generation_paradigm_prompt_1 = llama3_template.format("Image: <image>\nGive an elaborate explanation of the image you see.")

fashion_iq_mistral_query_generation_paradigm_prompt = llava_mistral_template.format("<image> change the style of this {} to <sent>\nPlease write a caption based on this image.")
fashion_iq_query_generation_paradigm_prompt = llama3_template.format("Image: <image>\nPlease write a caption based on this image.")
fashion_iq_mistral_query_generation_paradigm_prompt_1 = llava_mistral_template.format("Image: <image>\nWhat is the caption of the above image?")
fashion_iq_query_generation_paradigm_prompt_1 = llama3_template.format("Image: <image>\nWhat is the caption of the above image?")

person_retrieval_mistral_query_generation_paradigm_prompt = llava_mistral_template.format("Person Image: <image>\nPlease write a caption based on this person image.")
person_retrieval_query_generation_paradigm_prompt = llama3_template.format("Person Image: <image>\nPlease write a caption based on this person image.")
person_retrieval_mistral_query_generation_paradigm_prompt_1 = llava_mistral_template.format("Person Image: <image>\nWhat is the caption of the above person image?")
person_retrieval_query_generation_paradigm_prompt_1 = llama3_template.format("Person Image: <image>\nWhat is the caption of the above person image?")
person_retrieval_mistral_query_generation_paradigm_prompt_2 = llava_mistral_template.format("Person Image: <image>\nPlease describe this person write a caption for this person image.")
person_retrieval_query_generation_paradigm_prompt_2 = llama3_template.format("Person Image: <image>\nPlease describe this person write a caption for this person image.")

img_prompt_for_concat = 'Summary above image in one word: '
text_prompt_for_concat = 'Summary above sentence in one word: '
fashion_iq_composed_image_for_concat = 'Describe this modified {} in one word based on its style: '
fashion_iq_img_prompt_for_concat = 'Describe this {} in one word based on its style: '
person_retrieval_img_prompt_for_concat = 'Summary the person in above image in one word: '
person_retrieval_text_prompt_for_concat = 'Summary the person in above sentence in one word: '
person_retrieval_img_prompt_for_concat_1 = 'Describe this person in one word based on the image: '
person_retrieval_text_prompt_for_concat_1 = 'Describe this person in one word based on the sentence: '
person_retrieval_img_prompt_for_concat_2 = 'Describe wearing style of this person in one word based on the image: '
person_retrieval_text_prompt_for_concat_2 = 'Describe wearing style of this person in one word based on the sentence: '

img_prompt_no_one_word = llama3_template.format('<image>\n<|begin_of_text|>Summary above image: ')
text_prompt_no_one_word = llama3_template.format('<sent>\n<|begin_of_text|>Summary above sentence: ')
img_prompt_no_special_llava_v1_5 = llava_v1_5_template.format('<image>\n<|begin_of_text|>Summary above image in one word: ')
text_prompt_no_special_llava_v1_5 = llava_v1_5_template.format('<sent>\n<|begin_of_text|>Summary above sentence in one word: ')
img_prompt_qwen_v2_5 = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": '{}',
            },
            {"type": "text", "text": '\n<|im_start|>Summary above image in one word: '},
        ],
    }
]
text_prompt_qwen_v2_5 = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": '<sent>',
            },
            {"type": "text", "text": '\n<|im_start|>Summary above sentence in one word: '},
        ],
    }
]
img_prompt_qwen_v3 = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": '{}',
            },
            {"type": "text", "text": '\n<|im_start|>Summary above image in one word: '},
        ],
    }
]

text_prompt_qwen_v3 = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": '<sent>',
            },
            {"type": "text", "text": '\n<|im_start|>Summary above sentence in one word: '},
        ],
    }
]

img_prompt_intern_vl_v2_5 = [
    {
        "role": "user",
        "content": "<image>\nSummary above image in one word: ",
    }
]

text_prompt_intern_vl_v2_5 = [
    {
        "role": "user",
        "content": '<sent>\nSummary above sentence in one word: '
    }
]

task_text_prompts = [
    "In this task, you're presented with a text excerpt. Your task is to categorize the excerpt into a broad category such as 'Education', 'Technology', 'Health', 'Business', 'Environment', 'Politics', or 'Culture'. These categories help in organizing content for better accessibility and targeting.\n\n<sent>\nFor this task, this above sentence should be classified under one general category in one word: ",
    "In this task, you're given a statement and you need to determine whether it's presenting an 'Opinion' or a 'Fact'. This distinction is vital for information verification, educational purposes, and content analysis.\n\n<sent>\nFor this task, this above sentence discriminates between opinion and fact in one word: ",
    "In this task, you're given a review from an online platform. Your task is to generate a rating for the product based on the review on a scale of 1-5, where 1 means 'extremely negative' and 5 means 'extremely positive'.\n\n<sent>\nFor this task, this above sentence reflects the sentiment in one word: ",
    "In this task, you're reading a personal diary entry. Your task is to identify the predominant emotion expressed, such as joy, sadness, anger, fear, or love.\n\n<sent>\nFor this task, this above sentence conveys the emotion in one word: ",
    "In this task, you're presented with two sentences. Your task is to assess whether the sentences convey the same meaning. Use 'identical', 'similar', 'different', or 'unrelated' to describe the relationship.\n\n<sent>\nTo enhance the performance of this task, this above sentence means in one word: ",
    "In this task, you're given a sentence and a phrase. Your task is to determine if the phrase can be a contextual synonym within the given sentence. Options include 'yes', 'no', or 'partially'.\n\n<sent>\nTo enhance the performance of this task, this above sentence means in one word: ",
    "In this task, you're examining a news article. Your task is to extract the most critical fact from the article.\n\n<sent>\nFor this task, this above sentence encapsulates the key fact in one word: ",
    "In this task, you're reviewing a scientific abstract. Your task is to identify the main entities (e.g., proteins, diseases) and their relations (e.g., causes, treats).\n\n<sent>\nFor this task, this above sentence highlights the primary entity or relation in one word: ",
    ]

task_text_prompts_copy = [
    "In this task, you're presented with a text excerpt. Your task is to categorize the excerpt into a broad category such as 'Education', 'Technology', 'Health', 'Business', 'Environment', 'Politics', or 'Culture'. These categories help in organizing content for better accessibility and targeting. For this task, this sentence : <sent> should be classified under one general category in one word: ",
    "In this task, you're given a statement and you need to determine whether it's presenting an 'Opinion' or a 'Fact'. This distinction is vital for information verification, educational purposes, and content analysis. For this task, this sentence : <sent> discriminates between opinion and fact in one word: ",
    "In this task, you're given a review from an online platform. Your task is to generate a rating for the product based on the review on a scale of 1-5, where 1 means 'extremely negative' and 5 means 'extremely positive'. For this task, this sentence : <sent> reflects the sentiment in one word: ",
    "In this task, you're reading a personal diary entry. Your task is to identify the predominant emotion expressed, such as joy, sadness, anger, fear, or love. For this task, this sentence : <sent> conveys the emotion in one word: ",
    "In this task, you're presented with two sentences. Your task is to assess whether the sentences convey the same meaning. Use 'identical', 'similar', 'different', or 'unrelated' to describe the relationship. To enhance the performance of this task, this sentence : <sent> means in one word: ",
    "In this task, you're given a sentence and a phrase. Your task is to determine if the phrase can be a contextual synonym within the given sentence. Options include 'yes', 'no', or 'partially'. To enhance the performance of this task, this sentence : <sent> means in one word: ",
    "In this task, you're examining a news article. Your task is to extract the most critical fact from the article. For this task, this sentence : <sent> encapsulates the key fact in one word: ",
    "In this task, you're reviewing a scientific abstract. Your task is to identify the main entities (e.g., proteins, diseases) and their relations (e.g., causes, treats). For this task, this sentence : <sent> highlights the primary entity or relation in one word: ",
    ]

task_image_prompts = [
    "In this task, you're presented with an image. Your task is to categorize the image into a broad category such as 'Education', 'Technology', 'Health', 'Business', 'Environment', 'Politics', or 'Culture'. These categories help in organizing content for better accessibility and targeting.\n\n<image>\nFor this task, this above image should be classified under one general category in one word: ",
    "In this task, you're given an image and you need to determine whether it's presenting an 'Opinion' or a 'Fact'. This distinction is vital for information verification, educational purposes, and content analysis.\n\n<image>\nFor this task, this above image discriminates between opinion and fact in one word: ",
    "In this task, you're given an image from an online platform. Your task is to generate a rating for the product based on the image on a scale of 1-5, where 1 means 'extremely negative' and 5 means 'extremely positive'.\n\n<image>\nFor this task, this above image reflects the sentiment in one word: ",
    "In this task, you're reading a personal diary image. Your task is to identify the predominant emotion expressed, such as joy, sadness, anger, fear, or love.\n\n<image>\nFor this task, this above image conveys the emotion in one word: ",
    "In this task, you're presented with two images. Your task is to assess whether the images convey the same meaning. Use 'identical', 'similar', 'different', or 'unrelated' to describe the relationship.\n\n<image>\nTo enhance the performance of this task, this above image means in one word: ",
    "In this task, you're given an image and a phrase. Your task is to determine if the phrase can be a contextual synonym within the given image. Options include 'yes', 'no', or 'partially'.\n\n<image>\nTo enhance the performance of this task, this above image means in one word: ",
    "In this task, you're examining a news image. Your task is to extract the most critical fact from the image.\n\n<image>\nFor this task, this above image encapsulates the key fact in one word: ",
    "In this task, you're reviewing a scientific image. Your task is to identify the main entities (e.g., proteins, diseases) and their relations (e.g., causes, treats).\n\n<image>\nFor this task, this above image highlights the primary entity or relation in one word: ",
    ]

task_image_prompts_copy = [
    "In this task, you're presented with an image. Your task is to categorize the image into a broad category such as 'Education', 'Technology', 'Health', 'Business', 'Environment', 'Politics', or 'Culture'. These categories help in organizing content for better accessibility and targeting. For this task, this image : <image> should be classified under one general category in one word: ",
    "In this task, you're given an image and you need to determine whether it's presenting an 'Opinion' or a 'Fact'. This distinction is vital for information verification, educational purposes, and content analysis. For this task, this image : <image> discriminates between opinion and fact in one word: ",
    "In this task, you're given an image from an online platform. Your task is to generate a rating for the product based on the image on a scale of 1-5, where 1 means 'extremely negative' and 5 means 'extremely positive'. For this task, this image : <image> reflects the sentiment in one word: ",
    "In this task, you're reading a personal diary image. Your task is to identify the predominant emotion expressed, such as joy, sadness, anger, fear, or love. For this task, this image : <image> conveys the emotion in one word: ",
    "In this task, you're presented with two images. Your task is to assess whether the images convey the same meaning. Use 'identical', 'similar', 'different', or 'unrelated' to describe the relationship. To enhance the performance of this task, this image : <image> means in one word: ",
    "In this task, you're given an image and a phrase. Your task is to determine if the phrase can be a contextual synonym within the given image. Options include 'yes', 'no', or 'partially'. To enhance the performance of this task, this image : <image> means in one word: ",
    "In this task, you're examining a news image. Your task is to extract the most critical fact from the image. For this task, this image : <image> encapsulates the key fact in one word: ",
    "In this task, you're reviewing a scientific image. Your task is to identify the main entities (e.g., proteins, diseases) and their relations (e.g., causes, treats). For this task, this image : <image> highlights the primary entity or relation in one word: ",
    ]


retrieval_disassemble_text_prompts = [
    '<sent>\nSummary the people or objects in above sentence in one word: ',
    '<sent>\nSummary the relations, such as belongings or spatial position, between main people or objects in above sentence in one word: ',
    '<sent>\nSummary the environment, weather or places in above sentence in one word: ',
    '<sent>\nSummary the actions or movements of main people or objects in above sentence in one word: ',
    '<sent>\nSummary the appearance, such as color, material, decoration and so on, of main people or objects in above sentence in one word: '
]

retrieval_disassemble_text_prompts_for_concat = [
    'Summary the people or objects in above sentence in one word: ',
    'Summary the relations, such as belongings or spatial position, between main people or objects in above sentence in one word: ',
    'Summary the environment, weather or places in above sentence in one word: ',
    'Summary the actions or movements of main people or objects in above sentence in one word: ',
    'Summary the appearance, such as color, material, decoration and so on, of main people or objects in above sentence in one word: '
]

retrieval_disassemble_text_prompts_3 = [
    '<sent>\nSummary the people or objects in above sentence in one word: ',
    '<sent>\nSummary the environment, weather or places in above sentence in one word: ',
    '<sent>\nSummary the actions or movements of main people or objects in above sentence in one word: ',
]

retrieval_disassemble_text_prompts_3_for_concat = [
    'Summary the people or objects in above sentence in one word: ',
    'Summary the environment, weather or places in above sentence in one word: ',
    'Summary the actions or movements of main people or objects in above sentence in one word: ',
]

retrieval_disassemble_text_prompts_7_for_concat = [
    'Summary the people or objects in above sentence in one word: ',
    'Summary the relations, such as belongings or spatial position, between main people or objects in above sentence in one word: ',
    'Summary the environment, weather or places in above sentence in one word: ',
    'Summary the actions or movements of main people or objects in above sentence in one word: ',
    'Summary the color of main people or objects in above sentence in one word: ',
    'Summary the reason why main people or objects might be in this position and doing this thing in above sentence in one word: ',
    'Summary the material and decoration of main people or objects in above sentence in one word: '
]

fashion_iq_perspective = "\'color\', \'pattern\', \'design style\', \'length\'"

retrieval_disassemble_composed_image_prompts_fashion_iq_for_concat = [
    'Describe the clothes type of this modified {} in one word based on its style: ',
    'Describe the color of this modified {} in one word based on its style: ',
    'Describe the pattern of this modified {} in one word based on its style: ',
    'Describe the design style of this modified {} in one word based on its style: ',
    'Describe the length characteristics of different part, such as sleeve, neck, shoulder and so on, of this {} in one word based on its style: '
]

retrieval_disassemble_image_prompts_fashion_iq_for_concat = [
    'Describe the clothes type of this {} in one word based on its style: ',
    'Describe the color of this {} in one word based on its style: ',
    'Describe the pattern of this {} in one word based on its style: ',
    'Describe the design style of this {} in one word based on its style: ',
    'Describe the length characteristics of different part, such as sleeve, neck, shoulder and so on, of this {} in one word based on its style: '
]

fashion_iq_perspective_1 = "\'color\', \'pattern\', \'sleeve\', \'neck\', \'shoulder\', \'design style\', \'length of whole clothes\'"

retrieval_disassemble_composed_image_prompts_fashion_iq_for_concat_1 = [
    'Describe the clothes type of this modified {} with one of the three clothes types: shirt, dress, and toptee in one word base on its style: ',
    'Describe the color of this modified {} in one word based on its style: ',
    'Describe the graphic pattern of this modified {} in one word based on its style: ',
    'Describe the details of sleeves of this modified {} with one of the three types: long sleeves, short sleeves and sleeveless in one word based on its style: ',
    'Describe the details of neck of this modified {} with one of the ten types: V-neck, u-neck, round, ovel, broad, scoop, crew, turtle, high, and tight in one word based on its style: ',
    'Describe the details of shoulder strap of this modified {} with one of the six types: thick, thin, loose, one shoulder, no strap and off-shoulder in one word based on its style: ',
    'Describe the design style of this modified {} with one of the eight styles: elegant, sporty, formal, revealing, casual, sculptural, flowy, and sexy in one word based on its style: ',
    'Describe the length of this modified {} with one of the two words: long and short in one word based on its style: '
]

retrieval_disassemble_image_prompts_fashion_iq_for_concat_1 = [
    'Describe the clothes type of this {} with one of the three clothes types: shirt, dress, and toptee in one word base on its style: ',
    'Describe the color of this {} in one word based on its style: ',
    'Describe the graphic pattern of this {} in one word based on its style: ',
    'Describe the details of sleeves of this {} with one of the three types: long sleeves, short sleeves and sleeveless in one word based on its style: ',
    'Describe the details of neck of this {} with one of the ten types: V-neck, u-neck, round, ovel, broad, scoop, crew, turtle, high, and tight in one word based on its style: ',
    'Describe the details of shoulder strap of this {} with one of the six types: thick, thin, loose, one shoulder, strapless and off-shoulder in one word based on its style: ',
    'Describe the design style of this {} with one of the eight styles: elegant, sporty, formal, revealing, casual, sculptural, flowy, and sexy in one word based on its style: ',
    'Describe the length of this {} with one of the two words: long and short in one word based on its style: '
]

retrieval_disassemble_text_prompts_fashion_iq_for_concat_1 = [
    'Describe the clothes type of this {} in above sentence with one of the three clothes types: shirt, dress, and toptee in one word base on its style: ',
    'Describe the color of this {} in above sentence in one word based on its style: ',
    'Describe the graphic pattern of this {} in above sentence in one word based on its style: ',
    'Describe the details of sleeves of this {} in above sentence with one of the three types: long sleeves, short sleeves and sleeveless in one word based on its style: ',
    'Describe the details of neck of this {} in above sentence with one of the ten types: V-neck, u-neck, round, ovel, broad, scoop, crew, turtle, high, and tight in one word based on its style: ',
    'Describe the details of shoulder strap of this {} in above sentence with one of the six types: thick, thin, loose, one shoulder, strapless and off-shoulder in one word based on its style: ',
    'Describe the design style of this {} in above sentence with one of the eight styles: elegant, sporty, formal, revealing, casual, sculptural, flowy, and sexy in one word based on its style: ',
    'Describe the length of this {} in above sentence with one of the two words: long and short in one word based on its style: '
]


retrieval_disassemble_text_origin_prompts_person_retrieval_for_concat = [
    'Summary the people or objects in above sentence in one word: ',
    'Summary the gender in above sentence in one word: ',
    'Summary the actions or movements of main people or objects in above sentence in one word: ',
    'Summary the wearing of people or objects in above sentence in one word: ',
    'Summary the appearance, such as color, material, decoration and so on, of main people or objects in above sentence in one word: '
]

retrieval_disassemble_image_origin_prompts_person_retrieval_for_concat = [
    'Summary the people or objects in above image in one word: ',
    'Summary the gender in above image in one word: ',
    'Summary the actions or movements of main people or objects in above image in one word: ',
    'Summary the wearing of people or objects in above image in one word: ',
    'Summary the appearance, such as color, material, decoration and so on, of main people or objects in above image in one word: '
]

retrieval_disassemble_text_prompts_person_retrieval_for_concat = [
    'Summary the gender of person in above sentence in one word: ',
    'Summary the actions or movements of person in above sentence in one word: ',
    'Summary the objects in above sentence in one word: ',
    'Summary the wearing of person in above sentence in one word: ',
    'Summary the appearance and decoration details of person, such as color, pattern and so on, in above sentence in one word: '
]

retrieval_disassemble_image_prompts_person_retrieval_for_concat = [
    'Summary the gender of person in above image in one word: ',
    'Summary the actions or movements of person in above image in one word: ',
    'Summary the objects in above image in one word: ',
    'Summary the wearing of person in above image in one word: ',
    'Summary the appearance and decoration details of person, such as color, pattern and so on, in above image in one word: '
]

retrieval_disassemble_image_prompts_person_retrieval_for_concat_1 = [
    'Describe the gender of this person in one word based on the image: ',
    'Describe the actions or movements this person in one word based on the image: ',
    'Describe the objects in one word based on the image: ',
    'Describe the wearing of this person in one word based on the image: '
    'Describe the appearance and decoration details of this person, such as color, pattern and so on, in one word based on the image: ',
]

retrieval_disassemble_text_prompts_person_retrieval_for_concat_1 = [
    'Describe the gender of this person in one word based on the sentence: ',
    'Describe the actions or movements this person in one word based on the sentence: ',
    'Describe the objects in one word based on the sentence: ',
    'Describe the wearing of this person in one word based on the sentence: '
    'Describe the appearance and decoration details of this person, such as color, pattern and so on, in one word based on the sentence: ',
]

retrieval_disassemble_image_prompts = [
    '<image>\nSummary the people or objects in above image in one word: ',
    '<image>\nSummary the relations, such as belongings or spatial position, between main people or objects in above image in one word: ',
    '<image>\nSummary the environment, weather or places in above image in one word: ',
    '<image>\nSummary the actions or movements of main people or objects in above image in one word: ',
    '<image>\nSummary the appearance, such as color, material, decoration and so on, of main people or objects in above image in one word: '
]

retrieval_disassemble_image_prompts_for_concat = [
    'Summary the people or objects in above image in one word: ',
    'Summary the relations, such as belongings or spatial position, between main people or objects in above image in one word: ',
    'Summary the environment, weather or places in above image in one word: ',
    'Summary the actions or movements of main people or objects in above image in one word: ',
    'Summary the appearance, such as color, material, decoration and so on, of main people or objects in above image in one word: ',
]

retrieval_disassemble_image_prompts_3 = [
    '<image>\nSummary the people or objects in above image in one word: ',
    '<image>\nSummary the environment, weather or places in above image in one word: ',
    '<image>\nSummary the actions or movements of main people or objects in above image in one word: ',
]

retrieval_disassemble_image_prompts_3_for_concat = [
    'Summary the people or objects in above image in one word: ',
    'Summary the environment, weather or places in above image in one word: ',
    'Summary the actions or movements of main people or objects in above image in one word: ',
]

retrieval_disassemble_image_prompts_7_for_concat = [
    'Summary the people or objects in above image in one word: ',
    'Summary the relations, such as belongings or spatial position, between main people or objects in above image in one word: ',
    'Summary the environment, weather or places in above image in one word: ',
    'Summary the actions or movements of main people or objects in above image in one word: ',
    'Summary the color of main people or objects in above image in one word: ',
    'Summary the reason why main people or objects might be in this position and doing this thing in above image in one word:  ',
    'Summary the material and decoration of main people or objects in above image in one word: '
]

llama3_retrieval_disassemble_text_prompts = [llama3_template.format(prompt) for prompt in retrieval_disassemble_text_prompts]
llama3_retrieval_disassemble_image_prompts = [llama3_template.format(prompt) for prompt in retrieval_disassemble_image_prompts]

prompt_generation_text_prompt = llama3_template.format(
    'Your mission is to generate proper summary tasks from 3 to 5 aspects or perspectives for the new sentence. You need to ensure that formats of all summary tasks like \'Summary the people or objects in above sentence in one word.\'.\n\n'
    '<sent>\n'
    'Summary tasks:\n'
)

prompt_generation_image_prompt = llama3_template.format(
    'Your mission is to generate proper summary tasks from 3 to 5 aspects or perspectives for the new image. You need to ensure that formats of all summary tasks like \'Summary the people or objects in above sentence in one word.\'.\n\n'
    '<image>\n'
    'Summary tasks:\n'
)

prompt_generation_from_text_prompt = llama3_template.format(
    'We will provide a sentence and some corresponding summary tasks that can describe the content of sentence from different perspectives as examples. Your mission is to refer to format of the examples and generate proper summary tasks from three to five aspects or perspectives for the new sentence. You need to ensure that formats of all summary tasks like \'Summary the people or objects in above sentence in one word.\' and you do not need to answer these tasks.\n\n'
    '<sent>\n'
    'Summary tasks:\n1. Summary the people or objects in above sentence in one word.\n2. Summary the relations, such as belongings or spatial position, between main people or objects in above sentence in one word.\n3. Summary the environment, weather or places in above sentence in one word.\n4. Summary the actions or movements of main people or objects in above sentence in one word.\n5. Summary the appearance, such as color, material, decoration and so on, of main people or objects in above sentence in one word.\n\n'
    '<sent>\n'
    'Summary tasks:\n'
)
prompt_generation_from_image_prompt = llama3_template.format(
    'We will provide an image and some corresponding summary tasks that can describe the content of image from different perspectives as examples. Your mission is to refer to format of the examples and generate proper summary tasks from three to five aspects or perspectives for the new image. You need to ensure that formats of all summary tasks like \'Summary the people or objects in above image in one word.\' and you do not need to answer these tasks.\n\n'
    '<image>\n'
    'Summary tasks:\n1. Summary the people or objects in above image in one word.\n2. Summary the relations, such as belongings or spatial position, between main people or objects in above image in one word.\n3. Summary the environment, weather or places in above image in one word.\n4. Summary the actions or movements of main people or objects in above image in one word.\n5. Summary the appearance, such as color, material, decoration and so on, of main people or objects in above image in one word.\n\n'
    '<image>\n'
    'Summary tasks:\n'
)

prompt_generation_image_from_text_prompt = llama3_template.format(
    'We will provide a sentence and some corresponding summary tasks that can describe the content of sentence from different perspectives as examples. Your mission is to refer to format of the examples and generate proper summary tasks from three to five aspects or perspectives for the new sentence. You need to ensure that formats of all summary tasks like \'Summary the people or objects in above image in one word.\' and you do not need to answer these tasks.\n\n'
    '<sent>\n'
    'Summary tasks:\n1. Summary the people or objects in above sentence in one word.\n2. Summary the relations, such as belongings or spatial position, between main people or objects in above sentence in one word.\n3. Summary the environment, weather or places in above sentence in one word.\n4. Summary the actions or movements of main people or objects in above sentence in one word.\n5. Summary the appearance, such as color, material, decoration and so on, of main people or objects in above sentence in one word.\n\n'
    '<image>\n'
    'Summary tasks:\n'
)

prompt_generation_from_text_prompt_2 = llama3_template.format(
    'We will provide two sentences and some corresponding summary tasks that can describe the content of sentences from different perspectives as examples. Your mission is to refer to format of the examples and generate proper summary tasks from three to five aspects or perspectives for the new sentence. You need to ensure that formats of all summary tasks like \'Summary the people or objects in above sentence in one word.\' and you do not need to answer these tasks.\n\n'
    '<sent>\n'
    'Summary tasks:\n1. Summary the people or objects in above sentence in one word.\n2. Summary the relations, such as belongings or spatial position, between main people or objects in above sentence in one word.\n3. Summary the environment, weather or places in above sentence in one word.\n4. Summary the actions or movements of main people or objects in above sentence in one word.\n5. Summary the appearance, such as color, material, decoration and so on, of main people or objects in above sentence in one word.\n\n'
    '<sent>\n'
    'Summary tasks:\n1. Summary the people or objects in above sentence in one word.\n2. Summary the relations, such as belongings or spatial position, between main people or objects in above sentence in one word.\n3. Summary the environment, weather or places in above sentence in one word.\n4. Summary the actions or movements of main people or objects in above sentence in one word.\n5. Summary the appearance, such as color, material, decoration and so on, of main people or objects in above sentence in one word.\n\n'
    '<sent>\n'
    'Summary tasks:\n'
)

prompt_generation_from_image_prompt_2 = llama3_template.format(
    'We will provide two images and some corresponding summary tasks that can describe the content of images from different perspectives as examples. Your mission is to refer to format of the examples and generate proper summary tasks from three to five aspects or perspectives for the new image. You need to ensure that formats of all summary tasks like \'Summary the people or objects in above image in one word.\' and you do not need to answer these tasks.\n\n'
    '<image>\n'
    'Summary tasks:\n1. Summary the people or objects in above image in one word.\n2. Summary the relations, such as belongings or spatial position, between main people or objects in above image in one word.\n3. Summary the environment, weather or places in above image in one word.\n4. Summary the actions or movements of main people or objects in above image in one word.\n5. Summary the appearance, such as color, material, decoration and so on, of main people or objects in above image in one word.\n\n'
    '<image>\n'
    'Summary tasks:\n1. Summary the people or objects in above image in one word.\n2. Summary the relations, such as belongings or spatial position, between main people or objects in above image in one word.\n3. Summary the environment, weather or places in above image in one word.\n4. Summary the actions or movements of main people or objects in above image in one word.\n5. Summary the appearance, such as color, material, decoration and so on, of main people or objects in above image in one word.\n\n'
    '<image>\n'
    'Summary tasks:\n'
)

prompt_generation_image_from_text_prompt_2 = llama3_template.format(
    'We will provide two sentences and some corresponding summary tasks that can describe the content of sentences from different perspectives as examples. Your mission is to refer to format of the examples and generate proper summary tasks from three to five aspects or perspectives for the new sentence. You need to ensure that formats of all summary tasks like \'Summary the people or objects in above image in one word.\' and you do not need to answer these tasks.\n\n'
    '<sent>\n'
    'Summary tasks:\n1. Summary the people or objects in above sentence in one word.\n2. Summary the relations, such as belongings or spatial position, between main people or objects in above sentence in one word.\n3. Summary the environment, weather or places in above sentence in one word.\n4. Summary the actions or movements of main people or objects in above sentence in one word.\n5. Summary the appearance, such as color, material, decoration and so on, of main people or objects in above sentence in one word.\n\n'
    '<sent>\n'
    'Summary tasks:\n1. Summary the people or objects in above sentence in one word.\n2. Summary the relations, such as belongings or spatial position, between main people or objects in above sentence in one word.\n3. Summary the environment, weather or places in above sentence in one word.\n4. Summary the actions or movements of main people or objects in above sentence in one word.\n5. Summary the appearance, such as color, material, decoration and so on, of main people or objects in above sentence in one word.\n\n'
    '<image>\n'
    'Summary tasks:\n'
)

prompt_generation_from_pair_prompt = llama3_template.format(
    'Your mission is to generate proper summary tasks from 3 to 7 aspects or perspectives for the input image-sentence pair. You need to provide the results in list format and ensure all summary tasks like \'Summary xxx in above sentence in one word.\'.\n\n'
    'Input image-sentence pair: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Summary tasks:\n'
)

mistral_prompt_generation_from_pair_prompt = llava_mistral_template.format(
    'Your mission is to generate proper summary tasks from 3 to 7 aspects or perspectives for the input image-sentence pair. You need to provide the results in list format and ensure all summary tasks like \'Summary xxx in above sentence in one word.\'.\n\n'
    'Input image-sentence pair: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Summary tasks:\n'
)

prompt_generation_from_pair_prompt_1 = llama3_template.format(
    'Your mission is to generate proper summary tasks from 3 to 7 aspects or perspectives for the input image-sentence pair. You need to provide the results in list format and ensure all summary tasks like \'Summary xxx in above sentence in one word.\' and you do not need to answer these tasks. We will provide an image-sentence pair and some corresponding summary tasks that can describe the content of image-sentence pair from different perspectives as examples.\n\n'
    'Example 1: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Summary tasks:\n<sent>\n\n'
    'Input image-sentence pair: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Summary tasks:\n'
)

mistral_prompt_generation_from_pair_prompt_1 = llava_mistral_template.format(
    'Your mission is to generate proper summary tasks from 3 to 7 aspects or perspectives for the input image-sentence pair. You need to provide the results in list format and ensure all summary tasks like \'Summary xxx in above sentence in one word.\' and you do not need to answer these tasks. We will provide an image-sentence pair and some corresponding summary tasks that can describe the content of image-sentence pair from different perspectives as examples.\n\n'
    'Example 1: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Summary tasks:\n<sent>\n\n'
    'Input image-sentence pair: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Summary tasks:\n'
)

five_prompt = '1. Summary the people or objects in above sentence in one word.\n2. Summary the relations, such as belongings or spatial position, between main people or objects in above sentence in one word.\n3. Summary the environment, weather or places in above sentence in one word.\n4. Summary the actions or movements of main people or objects in above sentence in one word.\n5. Summary the appearance, such as color, material, decoration and so on, of main people or objects in above sentence in one word.'

prompt_generation_from_pair_prompt_2 = llama3_template.format(
    'Your mission is to generate proper summary tasks from 3 to 7 aspects or perspectives for the input image-sentence pair. You need to provide the results in list format and ensure all summary tasks like \'Summary xxx in above sentence in one word.\' and you do not need to answer these tasks. We will provide two image-sentence pairs and some corresponding summary tasks that can describe the content of image-sentence pairs from different perspectives as examples.\n\n'
    'Example 1: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Summary tasks:\n<sent>\n\n'
    'Example 2: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Summary tasks:\n<sent>\n\n'
    'Input image-sentence pair: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Summary tasks:\n'
)

mistral_prompt_generation_from_pair_prompt_2 = llava_mistral_template.format(
    'Your mission is to generate proper summary tasks from 3 to 7 aspects or perspectives for the input image-sentence pair. You need to provide the results in list format and ensure all summary tasks like \'Summary xxx in above sentence in one word.\' and you do not need to answer these tasks. We will provide two image-sentence pairs and some corresponding summary tasks that can describe the content of image-sentence pairs from different perspectives as examples.\n\n'
    'Example 1: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Summary tasks:\n<sent>\n\n'
    'Example 2: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Summary tasks:\n<sent>\n\n'
    'Input image-sentence pair: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Summary tasks:\n'
)

prompt_generation_from_pair_prompt_3 = llama3_template.format(
    'Your mission is to generate proper summary tasks from 3 to 7 aspects or perspectives for the input image-sentence pair. You need to provide the results in list format and ensure all summary tasks like \'Summary xxx in above sentence in one word.\' and you do not need to answer these tasks. We will provide three image-sentence pairs and some corresponding summary tasks that can describe the content of image-sentence pairs from different perspectives as examples.\n\n'
    'Example 1: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Summary tasks:\n<sent>\n\n'
    'Example 2: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Summary tasks:\n<sent>\n\n'
    'Example 3: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Summary tasks:\n<sent>\n\n'
    'Input image-sentence pair: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Summary tasks:\n'
)

mistral_prompt_generation_from_pair_prompt_3 = llava_mistral_template.format(
    'Your mission is to generate proper summary tasks from 3 to 7 aspects or perspectives for the input image-sentence pair. You need to provide the results in list format and ensure all summary tasks like \'Summary xxx in above sentence in one word.\' and you do not need to answer these tasks. We will provide three image-sentence pairs and some corresponding summary tasks that can describe the content of image-sentence pairs from different perspectives as examples.\n\n'
    'Example 1: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Summary tasks:\n<sent>\n\n'
    'Example 2: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Summary tasks:\n<sent>\n\n'
    'Example 3: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Summary tasks:\n<sent>\n\n'
    'Input image-sentence pair: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Summary tasks:\n'
)

prompt_generation_from_pair_prompt_4 = llama3_template.format(
    'Your mission is to generate proper summary tasks from 3 to 7 aspects or perspectives for the input image-sentence pair. You need to provide the results in list format and ensure all summary tasks like \'Summary xxx in above sentence in one word.\' and you do not need to answer these tasks. We will provide four image-sentence pairs and some corresponding summary tasks that can describe the content of image-sentence pairs from different perspectives as examples.\n\n'                                                          
    'Example 1: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Summary tasks:\n<sent>\n\n'
    'Example 2: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Summary tasks:\n<sent>\n\n'
    'Example 3: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Summary tasks:\n<sent>\n\n' 
    'Example 4: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Summary tasks:\n<sent>\n\n' 
    'Input image-sentence pair: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Summary tasks:\n'
)

mistral_prompt_generation_from_pair_prompt_4 = llava_mistral_template.format(
    'Your mission is to generate proper summary tasks from 3 to 7 aspects or perspectives for the input image-sentence pair. You need to provide the results in list format and ensure all summary tasks like \'Summary xxx in above sentence in one word.\' and you do not need to answer these tasks. We will provide four image-sentence pairs and some corresponding summary tasks that can describe the content of image-sentence pairs from different perspectives as examples.\n\n'                                                          
    'Example 1: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Summary tasks:\n<sent>\n\n'
    'Example 2: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Summary tasks:\n<sent>\n\n'
    'Example 3: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Summary tasks:\n<sent>\n\n' 
    'Example 4: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Summary tasks:\n<sent>\n\n' 
    'Input image-sentence pair: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Summary tasks:\n'
)

new_prompt_generation_from_pair_prompt = llama3_template.format(
    'Your mission is to generate 3 to 7 proper aspects or perspectives that can basically contain all information for the input image-sentence pair. You can be only permitted to predict 1 to 3 words for each aspects and output them in list format.\n\n'
    'Input image-sentence pair: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Proper aspects or perspectives:\n'
)

mistral_new_prompt_generation_from_pair_prompt = llava_mistral_template.format(
    'Your mission is to generate 3 to 7 proper aspects or perspectives that can basically contain all information for the input image-sentence pair. You can be only permitted to predict 1 to 3 words for each aspects and output them in list format.\n\n'
    'Input image-sentence pair: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Proper aspects or perspectives:\n'
)

new_prompt_generation_from_pair_prompt_1 = llama3_template.format(
    'Your mission is to generate 3 to 7 proper aspects or perspectives that can basically contain all information for the input image-sentence pair. You can be only permitted to predict 1 to 3 words for each aspects and output them in list format. We will provide an image-sentence pair and some corresponding aspects that can summary the content information from different perspectives as examples.\n\n'
    'Example 1: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Proper aspects or perspectives:\n<sent>\n\n'
    'Input image-sentence pair: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Proper aspects or perspectives:\n'
)

mistral_new_prompt_generation_from_pair_prompt_1 = llava_mistral_template.format(
    'Your mission is to generate 3 to 7 proper aspects or perspectives that can basically contain all information for the input image-sentence pair. You can be only permitted to predict 1 to 3 words for each aspects and output them in list format. We will provide an image-sentence pair and some corresponding aspects that can summary the content information from different perspectives as examples.\n\n'
    'Example 1: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Proper aspects or perspectives:\n<sent>\n\n'
    'Input image-sentence pair: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Proper aspects or perspectives:\n'
)

new_prompt_generation_from_pair_prompt_2 = llama3_template.format(
    'Your mission is to generate 3 to 7 proper aspects or perspectives that can basically contain all information for the input image-sentence pair. You can be only permitted to predict 1 to 3 words for each aspects and output them in list format. We will provide two image-sentence pairs and some corresponding aspects that can summary the content information from different perspectives as examples.\n\n'
    'Example 1: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Proper aspects or perspectives:\n<sent>\n\n'
    'Example 2: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Proper aspects or perspectives:\n<sent>\n\n'
    'Input image-sentence pair: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Proper aspects or perspectives:\n'
)

mistral_new_prompt_generation_from_pair_prompt_2 = llava_mistral_template.format(
    'Your mission is to generate 3 to 7 proper aspects or perspectives that can basically contain all information for the input image-sentence pair. You can be only permitted to predict 1 to 3 words for each aspects and output them in list format. We will provide two image-sentence pairs and some corresponding aspects that can summary the content information from different perspectives as examples.\n\n'
    'Example 1: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Proper aspects or perspectives:\n<sent>\n\n'
    'Example 2: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Proper aspects or perspectives:\n<sent>\n\n'
    'Input image-sentence pair: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Proper aspects or perspectives:\n'
)

new_prompt_generation_from_pair_prompt_3 = llama3_template.format(
    'Your mission is to generate 3 to 7 proper aspects or perspectives that can basically contain all information for the input image-sentence pair. You can be only permitted to predict 1 to 3 words for each aspects and output them in list format. We will provide three image-sentence pairs and some corresponding aspects that can summary the content information from different perspectives as examples.\n\n'
    'Example 1: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Proper aspects or perspectives:\n<sent>\n\n'
    'Example 2: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Proper aspects or perspectives:\n<sent>\n\n'
    'Example 3: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Proper aspects or perspectives:\n<sent>\n\n'
    'Input image-sentence pair: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Proper aspects or perspectives:\n'
)

new_prompt_generation_from_pair_prompt_4 = llama3_template.format(
    'Your mission is to generate 3 to 7 proper aspects or perspectives that can basically contain all information for the input image-sentence pair. You can be only permitted to predict 1 to 3 words for each aspects and output them in list format. We will provide four image-sentence pairs and some corresponding aspects that can summary the content information from different perspectives as examples.\n\n'
    'Example 1: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Proper aspects or perspectives:\n<sent>\n\n'
    'Example 2: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Proper aspects or perspectives:\n<sent>\n\n'
    'Example 3: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Proper aspects or perspectives:\n<sent>\n\n'
    'Example 4: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Proper aspects or perspectives:\n<sent>\n\n'
    'Input image-sentence pair: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Proper aspects or perspectives:\n'
)

mistral_new_prompt_generation_from_pair_prompt_4 = llava_mistral_template.format(
    'Your mission is to generate 3 to 7 proper aspects or perspectives that can basically contain all information for the input image-sentence pair. You can be only permitted to predict 1 to 3 words for each aspects and output them in list format. We will provide four image-sentence pairs and some corresponding aspects that can summary the content information from different perspectives as examples.\n\n'
    'Example 1: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Proper aspects or perspectives:\n<sent>\n\n'
    'Example 2: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Proper aspects or perspectives:\n<sent>\n\n'
    'Example 3: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Proper aspects or perspectives:\n<sent>\n\n'
    'Example 4: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Proper aspects or perspectives:\n<sent>\n\n'
    'Input image-sentence pair: \n'
    'Sentence: <sent>\n'
    'Image: <image>\n'
    'Proper aspects or perspectives:\n'
)

llava_llama_caption_generation_prompt_1 = llama3_template.format('<image>\nPlease write a caption based on this image.')
llava_llama_caption_generation_prompt_2 = llama3_template.format('<image>\nWhat is the caption of the above image?')
llava_mistral_caption_generation_prompt_1 = llava_mistral_template.format('<image>\nPlease write a caption based on this image.')
llava_mistral_caption_generation_prompt_2 = llava_mistral_template.format('<image>\nWhat is the caption of the above image?')

llama_prompt_generation_text_modal_only_prompt = llama3_template.format(
    'Your mission is to generate 3 to 7 proper aspects or perspectives that can basically contain all information for the input sentence. You can be only permitted to predict 1 to 3 words for each aspects and output them in list format.\n\n'
    'Input sentence: <sent>\n'
    'Proper aspects or perspectives:\n'
)

llama_prompt_generation_text_modal_only_prompt_1 = llama3_template.format(
    'Your mission is to generate 3 to 7 proper aspects or perspectives that can basically contain all information for the input sentence. You can be only permitted to predict 1 to 3 words for each aspects and output them in list format. Please be advised, you must not give answers for these aspects. We will provide a sentence and some corresponding aspects that can summary the content information from different perspectives as examples.\n\n'
    'Input sentence: <sent>\n'
    'Proper aspects or perspectives:\n<sent>\n\n'
    'Input sentence: <sent>\n'
    'Proper aspects or perspectives:\n'
)

llama_prompt_generation_text_modal_only_prompt_2 = llama3_template.format(
    'Your mission is to generate 3 to 7 proper aspects or perspectives that can basically contain all information for the input sentence. You can be only permitted to predict 1 to 3 words for each aspects and output them in list format. Please be advised, you must not give answers for these aspects. We will provide two sentences and some corresponding aspects that can summary the content information from different perspectives as examples.\n\n'
    'Input sentence: <sent>\n'
    'Proper aspects or perspectives:\n<sent>\n\n'
    'Input sentence: <sent>\n'
    'Proper aspects or perspectives:<sent>\n'
    'Input sentence: <sent>\n'
    'Proper aspects or perspectives:\n'
)

llama_prompt_generation_text_modal_only_prompt_3 = llama3_template.format(
    'Your mission is to generate 3 to 7 proper aspects or perspectives that can basically contain all information for the input sentence. You can be only permitted to predict 1 to 3 words for each aspects and output them in list format. Please be advised, you must not give answers for these aspects. We will provide three sentences and some corresponding aspects that can summary the content information from different perspectives as examples.\n\n'
    'Input sentence: <sent>\n'
    'Proper aspects or perspectives:\n<sent>\n\n'
    'Input sentence: <sent>\n'
    'Proper aspects or perspectives:<sent>\n'
    'Input sentence: <sent>\n'
    'Proper aspects or perspectives:<sent>\n' 
    'Input sentence: <sent>\n'
    'Proper aspects or perspectives:\n'
)

llama_prompt_generation_text_modal_only_prompt_4 = llama3_template.format(
    'Your mission is to generate 3 to 7 proper aspects or perspectives that can basically contain all information for the input sentence. You can be only permitted to predict 1 to 3 words for each aspects and output them in list format. Please be advised, you must not give answers for these aspects. We will provide four sentences and some corresponding aspects that can summary the content information from different perspectives as examples.\n\n'
    'Input sentence: <sent>\n'
    'Proper aspects or perspectives:\n<sent>\n\n'
    'Input sentence: <sent>\n'
    'Proper aspects or perspectives:<sent>\n'
    'Input sentence: <sent>\n'
    'Proper aspects or perspectives:<sent>\n' 
    'Input sentence: <sent>\n'
    'Proper aspects or perspectives:<sent>\n'
    'Input sentence: <sent>\n'
    'Proper aspects or perspectives:\n'
)

mistral_prompt_generation_text_modal_only_prompt = llava_mistral_template.format(
    'Your mission is to generate 3 to 7 proper aspects or perspectives that can basically contain all information for the input sentence. You can be only permitted to predict 1 to 3 words for each aspects and output them in list format.\n\n'
    'Input sentence: <sent>\n'
    'Proper aspects or perspectives:\n'
)

mistral_prompt_generation_text_modal_only_prompt_1 = llava_mistral_template.format(
    'Your mission is to generate 3 to 7 proper aspects or perspectives that can basically contain all information for the input sentence. You can be only permitted to predict 1 to 3 words for each aspects and output them in list format. Please be advised, you must not give answers for these aspects. We will provide a sentence and some corresponding aspects that can summary the content information from different perspectives as examples.\n\n'
    'Input sentence: <sent>\n'
    'Proper aspects or perspectives:\n<sent>\n\n'
    'Input sentence: <sent>\n'
    'Proper aspects or perspectives:\n'
)

mistral_prompt_generation_text_modal_only_prompt_2 = llava_mistral_template.format(
    'Your mission is to generate 3 to 7 proper aspects or perspectives that can basically contain all information for the input sentence. You can be only permitted to predict 1 to 3 words for each aspects and output them in list format. Please be advised, you must not give answers for these aspects. We will provide two sentences and some corresponding aspects that can summary the content information from different perspectives as examples.\n\n'
    'Input sentence: <sent>\n'
    'Proper aspects or perspectives:\n<sent>\n\n'
    'Input sentence: <sent>\n'
    'Proper aspects or perspectives:<sent>\n'
    'Input sentence: <sent>\n'
    'Proper aspects or perspectives:\n'
)

mistral_prompt_generation_text_modal_only_prompt_3 = llava_mistral_template.format(
    'Your mission is to generate 3 to 7 proper aspects or perspectives that can basically contain all information for the input sentence. You can be only permitted to predict 1 to 3 words for each aspects and output them in list format. Please be advised, you must not give answers for these aspects. We will provide three sentences and some corresponding aspects that can summary the content information from different perspectives as examples.\n\n'
    'Input sentence: <sent>\n'
    'Proper aspects or perspectives:\n<sent>\n\n'
    'Input sentence: <sent>\n'
    'Proper aspects or perspectives:<sent>\n'
    'Input sentence: <sent>\n'
    'Proper aspects or perspectives:<sent>\n' 
    'Input sentence: <sent>\n'
    'Proper aspects or perspectives:\n'
)

mistral_prompt_generation_text_modal_only_prompt_4 = llava_mistral_template.format(
    'Your mission is to generate 3 to 7 proper aspects or perspectives that can basically contain all information for the input sentence. You can be only permitted to predict 1 to 3 words for each aspects and output them in list format. Please be advised, you must not give answers for these aspects. We will provide four sentences and some corresponding aspects that can summary the content information from different perspectives as examples.\n\n'
    'Input sentence: <sent>\n'
    'Proper aspects or perspectives:\n<sent>\n\n'
    'Input sentence: <sent>\n'
    'Proper aspects or perspectives:<sent>\n'
    'Input sentence: <sent>\n'
    'Proper aspects or perspectives:<sent>\n' 
    'Input sentence: <sent>\n'
    'Proper aspects or perspectives:<sent>\n'
    'Input sentence: <sent>\n'
    'Proper aspects or perspectives:\n'
)

five_aspects = '1. people or objects\n2. relations\n3. environment\n4. actions\n5. appearance\n'

prompt_schema_generation_text_prompt = llama3_template.format(
    'You are an experienced knowledge engineer and you are modeling schemas for knowledge graph construction. '
    'Given a set of sentences, you need to give several proper words or phrases for the abstract schemas of entities, relations and events in these sentences.'
    'You must return your answer in the following format: 1. phrases1\n2.phrases2\n3.phrases3\n...'
    'You can\'t return anything other than answers.'
    'These abstract intention words should fulfill the following requirements.'
    '1. The abstract schemas phrases can well represent the entities, relations and events, and it could be the type of the entities, relations and events or the related concepts of the entities, relations and events.'
    '2. Strictly follow the provided format, do not add extra characters or words.'
    '3. Write 3 to 7 words or phrases at the highest possible abstract level if possible.'
    '4. Do not repeat the same word and the input in the answer.'
    '5. Stop immediately if you can\'t think of any more phrases, and no explanation is needed.'

    'Input sentences: <sent>\n'
    'Your answer:'

)

prompt_schema_generation_text_prompt_1 = llama3_template.format(
    'You are an experienced knowledge engineer and you are modeling schemas for knowledge graph construction. '
    'Given a set of sentences, you need to give several proper words or phrases for the abstract schemas of entities, relations and events in these sentences.'
    'You must return your answer in the following format: 1. phrases1\n2.phrases2\n3.phrases3\n...'
    'You can\'t return anything other than answers.'
    'These abstract intention words should fulfill the following requirements.'
    '1. The abstract schemas phrases can well represent the entities, relations and events, and it could be the type of the entities, relations and events or the related concepts of the entities, relations and events.'
    '2. Strictly follow the provided format, do not add extra characters or words.'
    '3. Write 3 to 7 words or phrases at the highest possible abstract level if possible.'
    '4. Do not repeat the same word and the input in the answer.'
    '5. Stop immediately if you can\'t think of any more phrases, and no explanation is needed.'

    'Input sentences: <sent>\n'
    'Your answer: <sent>\n'
    'Input sentences: <sent>\n'
    'Your answer:'
)

mistral_prompt_schema_generation_text_prompt = llava_mistral_template.format(
    'You are an experienced knowledge engineer and you are modeling schemas for knowledge graph construction. '
    'Given a set of sentences, you need to give several proper words or phrases for the abstract schemas of entities, relations and events in these sentences.'
    'You must return your answer in the following format: 1. phrases1\n2.phrases2\n3.phrases3\n...'
    'You can\'t return anything other than answers.'
    'These abstract intention words should fulfill the following requirements.'
    '1. The abstract schemas phrases can well represent the entities, relations and events, and it could be the type of the entities, relations and events or the related concepts of the entities, relations and events.'
    '2. Strictly follow the provided format, do not add extra characters or words.'
    '3. Write 3 to 7 words or phrases at the highest possible abstract level if possible.'
    '4. Do not repeat the same word and the input in the answer.'
    '5. Stop immediately if you can\'t think of any more phrases, and no explanation is needed.'

    'Input sentences: <sent>\n'
    'Your answer:'
)