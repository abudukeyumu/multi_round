import asyncio
import logging
from client import custom_model_if_cache,custom_httpx
import json
import re
import random


def prompt_generate(text):
    prompt=f"""
You are a helpful, pattern-following assistant.

role-user:
Given the following conversation between user and assistant:

user: I’ve heard about adobo sauce in Mexican cuisine. Can you tell me more about it?
assistant: Adobo sauce is a dark red sauce made from ground chiles, herbs, and vinegar. It’s used as a marinade and cooking sauce in Mexican and Southwestern cuisines.[0]

user: What are the main ingredients in adobo sauce?
assistant: The main ingredients are typically chile peppers, garlic, and vinegar.However, there can be variations in the recipe.[1]

user: How can I make adobo sauce at home?
assistant: One way to make adobo sauce at home is by using smoked jalapeno peppers called chipotles. You can stew the chipotles in a sauce with tomatoes, garlic,vinegar, salt, and spices. It’s also possible to find canned adobo sauce in the Hispanic foods section of many grocery stores.[2]

user: Are there any other uses for adobo sauce?
assistant: Adobo sauce can be used as a condiment or as a marinade for various dishes. It enhances the flavor of meats and can be used as a cooking and serving sauce. Additionally, adobo is also used in a stew-like presentation of meat prepared in this sauce.[3]

user: Is adobo sauce easy to preserve?
assistant: Yes, you can preserve adobo sauce by freezing or canning it when you’re done making it. This allows you to store it for up to a year and enjoy its flavors whenever you want.[4]

Modify user’s questions to use pronouns or common nouns to refer back to words or phrases in user’s previous questions or assistant’s previous responses.

role-assistant:
user: I’ve heard about adobo sauce in Mexican cuisine. Can you provide more information about it?
assistant: Adobo sauce is a dark red sauce made from ground chiles, herbs, and vinegar. It’s used as a marinade and cooking sauce in Mexican and Southwestern cuisines.[0]

user: What are the main ingredients in the sauce?
assistant: The main ingredients are typically chile peppers, garlic, and vinegar.However, there can be variations in the recipe.[1]

user: How can I make the sauce at home?
assistant: One way to make it at home is by using smoked jalapeno peppers called chipotles. You can stew them in a sauce with tomatoes, garlic, vinegar, salt, and spices. It’s also possible to find canned adobo sauce in the Hispanic foods section of many grocery stores.[2]

user: Are there any other uses for it?
assistant: It can be used as a condiment or as a marinade for various dishes. It enhances the flavor of meats and can be used as a cooking and serving sauce. Additionally, it is also used in a stew-like presentation of meat prepared in this sauce.[3]

user: Is it easy to preserve?
assistant: Yes, you can preserve it by freezing or canning it when you’re done making it. This allows you to store it for up to a year and enjoy its flavors whenever you want.[4]

role-user:
Given the following conversation between user and assistant:

user: I’ve heard about a test called urinalysis. What is it used for?
assistant: Urinalysis is a test that evaluates a sample of your urine. It is used to detect and assess a wide range of disorders, such as urinary tract infection,kidney disease, and diabetes. It examines the appearance, concentration, and content of urine.[0]

user: Can it help diagnose bladder cancer?
assistant: Yes, urinalysis can be used as a screening and diagnostic tool for bladder cancer. One way to test for bladder cancer is to check for blood in the urine, which is called hematuria. This can be done during a urinalysis by examining the urine cells under a special microscope.[1]

user: What other conditions or diseases can be detected through urinalysis?
assistant: Urinalysis is commonly used to diagnose urinary tract or kidney infections,evaluate causes of kidney failure, screen for the progression of chronic conditions like diabetes mellitus and high blood pressure, and it can be used in combination with other tests to diagnose certain diseases. It is a versatile test that helps detect abnormalities in the urinary tract.[9]

user: Where can I get a urinalysis done?
assistant: Urinalysis can be performed in various healthcare settings, including doctors’ offices, urgent care facilities, laboratories, and hospitals. You canprovide a urine sample in a specimen cup, and the sample can be analyzed on-site or sent to a laboratory for testing.[15]

user: How much urine is typically needed for a urinalysis?
assistant: Usually, only small amounts of urine, ranging from 30 to 60 ml, are required for urinalysis testing. It’s a non-invasive and relatively simple test that provides valuable information about your health.[17]

Modify user’s questions to use pronouns or common nouns to refer back to words or phrases in user’s previous questions or assistant’s previous responses.

role-assistant:
user: I’ve heard about a test called urinalysis. What is it used for?
assistant: Urinalysis is a test that evaluates a sample of your urine. It’s used to detect and assess a wide range of disorders, such as urinary tract infection, kidney disease, and diabetes. It examines the appearance, concentration, and content of urine.[0]

user: Can it help diagnose bladder cancer?
assistant: Yes, urinalysis can be used as a screening and diagnostic tool for bladder cancer. One way to test for bladder cancer is to check for blood in the urine, which is called hematuria. This can be done during a urinalysis by examining the urine cells under a special microscope.[1]

user: What other conditions or diseases can be detected through this test?
assistant: Urinalysis is commonly used to diagnose urinary tract or kidney infections, evaluate causes of kidney failure, screen for the progression of chronic conditions like diabetes mellitus and high blood pressure, and it can be used in combination with other tests to diagnose certain diseases. It is a versatile test that helps detect abnormalities in the urinary tract.[9]

user: Where can I go to get this test done?
assistant: Urinalysis can be performed in various healthcare settings, including doctors’ offices, urgent care facilities, laboratories, and hospitals. You can provide a urine sample in a specimen cup, and the sample can be analyzed on-site or sent to a laboratory for testing.[15]

user: How much urine is typically needed for the test?
assistant: Usually, only small amounts of urine, ranging from 30 to 60 ml, are required for urinalysis testing. It’s a non-invasive and relatively simple test that provides valuable information about your health.[17]

role-user:
Given the following conversation between user and assistant:

{text}

Modify user’s questions to use pronouns or common nouns to refer back to words or phrases in user’s previous questions or assistant’s previous responses.

role-assistant:
"""
    return prompt

def read_from_jsonl(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                item = json.loads(line)
                data.append(item)
    return data


def save_to_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')  

async def send_request():
    tasks = []
    text = read_from_jsonl("/data/abudukeyumu/test-c/wiki_generate/output2_results.jsonl")
    # import pdb
    # pdb.set_trace()
    TOTAL_REQUESTS= len(text)
    print(TOTAL_REQUESTS)
    # import pdb
    # pdb.set_trace()
    for i in range(TOTAL_REQUESTS):

        prompt =prompt_generate(text[i])
        # 每次请求的不同参数
        task = custom_model_if_cache(prompt=prompt)  # 调用受限的异步函数
        tasks.append(task)
        # 并发执行所有请求
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results


# TOTAL_REQUESTS = 5
if __name__ == "__main__":

    # 运行异步函数
    results = asyncio.run(send_request())
    # print("\n输出内容:",results)
    # import pdb
    # pdb.set_trace()
    save_to_jsonl(results, '/data/abudukeyumu/test-c/wiki_generate/2step/4o_gaixie_output2_results.jsonl')
    print("结果已保存到 gaixie_output2_results.jsonl 文件中")

