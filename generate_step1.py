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
assistant has access to information in the following passages:

[1]There are certain important aspects of college preparation. The student preparing for college should define their goals and interests. The student should also plan financially and academically.

[2]We have a checklist that helps students get ready financially and academically.

[3]Students should consider the reasons for choosing between college and career schools. Each option provides different benefits for future career paths.

[4]Students should explore various career options before making college decisions. Understanding career paths helps guide educational choices.

Generate a conversation of around four turns between assistant and user where user needs information from assistant that is in the passages. Use only information found in the passages.
user starts the conversation and only asks questions. Most of user's questions are follow-up or continuations of previous questions.

For each response from assistant, indicate which passage(s) the information comes from by adding the reference number in square brackets at the end of the response. Each response should only use information from a single passage.

role-assistant:
user: I need to know how to prepare for college.
assistant: You should first define your goals and interests, and make sure to plan both financially and academically for college. [1]

user: Is there something I can use as a guide to get ready?
assistant: Yes, we have a checklist that can help you get ready both financially and academically. [2]

user: What else should I consider before making my decision?
assistant: You should consider the different benefits offered by colleges and career schools, as each option provides different advantages for your future career path. [3]

user: How can I make sure I'm making the right choice?
assistant: It's important to explore various career options before making your college decisions, as understanding potential career paths will help guide your educational choices. [4]

role-user:
assistant has access to information in the following passages:

[1]Vascepa (icosapent ethyl) is a prescription fish oil medication approved for treating adults with extremely high triglycerides, defined as a triglyceride level of 500 mg/dL or higher. While using this medicine, you may need frequent blood tests at your doctor's office. Vascepa is only part of a treatment program that may also include diet, exercise, and weight control.

[2]You should not use Vascepa if you are allergic to icosapent. To make sure Vascepa is safe for you, tell your doctor if you have: 1 liver disease; 2 diabetes; 3 a thyroid disorder; 4 problems with your pancreas; 5 a bleeding or blood-clotting disorder; 6 if you are allergic to fish or shellfish; or 7 if you drink large amounts of alcohol.

[3]Vascepa (icosapent) works in the liver and bloodstream to reduce very low-density triglycerides. Follow your doctor's instructions very closely.

Generate a conversation of around four turns between assistant and user where user needs information from assistant that is in the passages. Use only information found in the passages.
user starts the conversation and only asks questions. Most of user's questions are follow-up or continuations of previous questions.

For each response from assistant, indicate which passage(s) the information comes from by adding the reference number in square brackets at the end of the response. Each response should only use information from a single passage.

role-assistant:
user: I heard about a medication called Vascepa. What is it used for?
assistant: Vascepa is a prescription fish oil medication approved for treating adults with extremely high triglycerides, defined as a triglyceride level of 500 mg/dL or higher. [1]

user: Are there any side effects or precautions I should be aware of?
assistant: You should not use Vascepa if you are allergic to icosapent. You need to tell your doctor if you have liver disease, diabetes, thyroid disorder, pancreas problems, bleeding disorders, fish/shellfish allergies, or if you drink large amounts of alcohol. [2]

user: How does Vascepa work?
assistant: Vascepa works in the liver and bloodstream to reduce very low-density triglycerides. [3]

role-user:
assistant has access to information in the following passages:
{text}
Generate a conversation of around four turns between assistant and user where user needs information from assistant that is in the passages. Use only information found in the passages.
user starts the conversation and only asks questions. Most of user's questions are follow-up or continuations of previous questions.

For each response from assistant, indicate which passage(s) the information comes from by adding the reference number in square brackets at the end of the response. Each response should only use information from a single passage.

role-assistant:
"""
    return prompt

def process_data(file_path):

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    text = []
    for cluster_id, cluster_info in data['clusters'].items():
        documents = cluster_info['documents']
    # 将每个cluster的文档组合成一个字符串，每个文档前加上编号
        cluster_text = ""
        for i, doc in enumerate(documents):
            cluster_text += f"[{cluster_id}{i}]{doc}\n"

        text.append(cluster_text)
    return text


def save_to_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')  

async def send_request():
    tasks = []
    text = process_data("/data/abudukeyumu/11_7/zifu_512_over_0_11_9/11_11/non_fuhao_style_filtered_output2.json")
    # import pdb
    # pdb.set_trace()
    TOTAL_REQUESTS= 5
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
    save_to_jsonl(results, '/data/abudukeyumu/test-c/wan/output2_results.jsonl')
    print("结果已保存到output2_results.jsonl 文件中")

