llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'
img_prompt = llama3_template.format('<image>\nSummary above image in one word: ')
text_prompt = llama3_template.format('<sent>\nSummary above sentence in one word: ')
img_prompt_no_one_word = llama3_template.format('<image>\nSummary above image: ')
text_prompt_no_one_word = llama3_template.format('<sent>\nSummary above sentence: ')