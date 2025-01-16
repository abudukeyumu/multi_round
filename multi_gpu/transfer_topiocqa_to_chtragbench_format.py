
import csv
import json
from tqdm import tqdm

# 文件路径
tsv_file_path = "/mnt/abu/topiocqa/downloads/data/wikipedia_split/full_wiki_segments.tsv"
output_json_path = "/mnt/abu/evaluation/xiugai/topiocqa_output.json"
input_json_path = "/mnt/abu/data/ChatRAG-Bench/data/topiocqa/dev.json"
output_with_index_path = "/mnt/abu/evaluation/xiugai/dev_with_idx.json"


print(f"读取 TSV 文件：{tsv_file_path}")
text_list = []


with open(tsv_file_path, 'r', encoding='utf-8') as file:
    tsv_reader = csv.reader(file, delimiter='\t')  
    header = next(tsv_reader)  
    text_index = header.index("text")  

    for row in tsv_reader:

        text_list.append(row[text_index])


output_data = {"wiki": text_list}
with open(output_json_path, 'w', encoding='utf-8') as json_file:
    json.dump(output_data, json_file, ensure_ascii=False, indent=4)

print(f"TSV 数据已成功保存到 JSON 文件：{output_json_path}")


print(f"加载输入 JSON 文件：{input_json_path}")
with open(input_json_path, "r", encoding="utf-8") as f:
    input_list = json.load(f)


wiki_dict = {text: idx for idx, text in enumerate(text_list)}  


print("匹配 ground_truth_ctx 的下标...")
unmatched_count = 0  

for item in tqdm(input_list, desc="Processing input list"):
    item["document"] = "wiki"
    ground_truth_ctx = item.get("ground_truth_ctx", {})
    ctx = ground_truth_ctx.get("ctx", "")

    idx = wiki_dict.get(ctx, -1)  
    ground_truth_ctx["index"] = idx
    if idx == -1:
        unmatched_count += 1  


print(f"未匹配的 ground_truth_ctx 数量: {unmatched_count}")


with open(output_with_index_path, "w", encoding="utf-8") as f:
    json.dump(input_list, f, ensure_ascii=False, indent=4)

print(f"更新后的数据已成功保存到：{output_with_index_path}")