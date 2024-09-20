from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import random
import json
from tqdm import tqdm

def model_generate(model_path,prompt):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=2048)

    llm = LLM(model=model_path,
              tensor_parallel_size=8,
              gpu_memory_utilization=0.94,  # 将GPU内存利用率
              )

    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # generate outputs
    outputs = llm.generate([text], sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        return generated_text

def prompt_generate(random_data):
    prompt = """
    You are required to generate English multi-turn dialogue data to evaluate the model's understanding of referential relationships, specifically focusing on anaphora, which reflect realistic inquiries that users might pose to large-scale models. You are required to use anaphora to generate multi-turn dialogues between 'Human' and 'Assistant', where anaphora is a linguistic term for a reference to something mentioned earlier in the dialogue.
    Step 1: 'Human' poses a question.
    Step 2: 'Assistant' answers the question.
    Step 3: A follow-up question is asked about the first round's answer, using anaphora to refer back to some content from the first round's answer.
    Step 4: The question is answered.
    Step 5: The fifth step involves another follow-up question about the first round's answer, again using anaphora to refer to certain content.
    Step 6: In the sixth step, the question is answered.

    You can refer to these examples:
    """
    for i, d in enumerate(random_data):
        prompt += f"\n# Example {i+1} #\n"
        for turn in d["history"]:
            prompt += f"Human: {turn['user']}\n"
            prompt += f"Assistant: {turn['bot']}\n"

    prompt += """Please ensure that the anaphoric references in the third and fifth steps effectively demonstrate the model's capability to understand and resolve referential expressions. Please output the dialogue content directly with 'Human:' and 'Assistant:' as role prompts, without stating 'step1', 'step2', and so on."""

    return prompt

def main():
    model_path = "/mnt/abu/models/Qwen2.5-72B-Instruct"
    example_path = "/mnt/abu/mt-bench-101/data/AR.json"
    output_path = "/mnt/abu/test/generated_dialogues_qwen2.5_500.json "

    # 读取JSON文件
    with open(example_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_dialogues = []

    for i in tqdm(range(500), desc="Processing conversations"):
        random_data = random.sample(data, 3)
        prompt = prompt_generate(random_data)
        response = model_generate(model_path,prompt)

        turns = response.split('\n')
        processed_dialogue = {"conversation": []}

        for j in range(0, len(turns), 2):
            if j+1 < len(turns):
                human_turn = turns[j].replace("Human: ", "").strip()
                assistant_turn = turns[j+1].replace("Assistant: ", "").strip()
                processed_dialogue["conversation"].append({
                    "human": human_turn,
                    "assistant": assistant_turn
                })

        all_dialogues.append(processed_dialogue)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_dialogues, f, ensure_ascii=False, indent=4)

    print("All dialogues have been generated and saved to ",output_path)

if __name__ == "__main__":
    main()