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


retreival_disassemble_text_prompts = [
    '<sent>\nSummary the people or objects in above sentence in one word: ',
    '<sent>\nSummary the relations, such as belongings or spatial position, between main people or objects in above sentence in one word: ',
    '<sent>\nSummary the environment, weather or places in above sentence in one word: ',
    '<sent>\nSummary the actions or movements of main people or objects in above sentence in one word: ',
    '<sent>\nSummary the appearance, such as color, meterial, decoration and so on, of main people or objects in above sentence in one word: '
]

retrieval_disassemble_image_prompts = [
    '<image>\nSummary the people or objects in above image in one word: ',
    '<image>\nSummary the relations, such as belongings or spatial position, between main people or objects in above image in one word: ',
    '<image>\nSummary the environment, weather or places in above image in one word: ',
    '<image>\nSummary the actions or movements of main people or objects in above image in one word: ',
    '<image>\nSummary the appearance, such as color, meterial, decoration and so on, of main people or objects in above image in one word: '
]

llama3_retrieval_disassemble_text_prompts = [llama3_template.format(prompt) for prompt in retreival_disassemble_text_prompts]
llama3_retrieval_disassemble_image_prompts = [llama3_template.format(prompt) for prompt in retrieval_disassemble_image_prompts]

prompt_generation_from_text_prompt = llama3_template.format(
    'We will provide a sentence and some summary tasks as demonstrations. Your mission is to refer to the demonstration and generate proper summary tasks for the new sentence.\n\n<sent>\n'
    'Summary tasks:\nSummary the people or objects in above sentence in one word.\nSummary the relations, such as belongings or spatial position, between main people or objects in above sentence in one word.\nSummary the environment, weather or places in above image in one word.\nSummary the actions or movements of main people or objects in above sentence in one word.\nSummary the appearance, such as color, meterial, decoration and so on, of main people or objects in above sentence in one word.\n\n'
    '<sent>\n'
    'Summary tasks:\n'
)
prompt_generation_from_image_prompt = llama3_template.format(
    'We will provide an image and some summary tasks as demonstrations. Your mission is to refer to the demonstration and generate proper summary tasks for the new image.\n\n<image>\n'
    'Summary tasks:\nSummary the people or objects in above image in one word.\nSummary the relations, such as belongings or spatial position, between main people or objects in above image in one word.\nSummary the environment, weather or places in above image in one word.\nSummary the actions or movements of main people or objects in above image in one word.\nSummary the appearance, such as color, meterial, decoration and so on, of main people or objects in above image in one word.\n\n'
    '<image>\n'
    'Summary tasks:\n'
)

# prompt_generation_text_2_image_prompt = llama3_template.format()
# prompt_generation_image_2_text_prompt = llama3_template.format()