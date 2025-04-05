llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'
llava_v1_5_template = '<s>user\n\n{}</s><s>assistant\n\n \n'
img_prompt = llama3_template.format('<image>\nSummary above image in one word: ')
text_prompt = llama3_template.format('<sent>\nSummary above sentence in one word: ')
img_prompt_no_one_word = llama3_template.format('<image>\nSummary above image: ')
text_prompt_no_one_word = llama3_template.format('<sent>\nSummary above sentence: ')
img_prompt_no_special_llava_v1_5 = llava_v1_5_template.format('<image>\nSummary above image in one word: ')
text_prompt_no_special_llava_v1_5 = llava_v1_5_template.format('<sent>\nSummary above sentence in one word: ')
img_prompt_qwen_v2_5 = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": '{}',
                        },
                        {"type": "text", "text": '\nSummary above image in one word: '},
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
                        {"type": "text", "text": '\nSummary above sentence in one word: '},
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