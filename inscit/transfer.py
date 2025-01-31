import csv
import json
from tqdm import tqdm
import sys
from collections import defaultdict

maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

# 文件路径
tsv_file_path = "/mnt/abu/INSCIT/data/wikipedia/full_wiki_segments.tsv"    #维基百科数据
input_json_path = "/mnt/abu/data/ChatRAG-Bench/data/inscit/dev.json"        #chatrag bench里的inscit数据
output_with_index_path = "/mnt/abu/evaluation/inscit/test/dev_with_idx.json"    #添加index到inscit数据里
unmatched_output_path = "/mnt/abu/evaluation/inscit/test/unmatched_data.json"   #找不到的数据
wiki_texts_path = "/mnt/abu/evaluation/inscit/test/inscit_texts.json"           #把维基百科数据转换成如下格式然后输出的json文件：{"wiki":[文本列表]}

# 读取TSV文件并同时建立索引和保存文本
print("读取TSV文件并建立索引...")
id_to_positions = defaultdict(list)
wiki_texts = []

with open(tsv_file_path, 'r', encoding='utf-8') as file:
    tsv_reader = csv.reader(file, delimiter='\t')  
    header = next(tsv_reader)  
    idx_index = header.index("id")
    text_index = header.index("text")
    
    for position, row in tqdm(enumerate(tsv_reader), desc="Processing TSV"):
        current_id = row[idx_index]
        id_to_positions[current_id].append(position)
        wiki_texts.append(row[text_index])

# 保存文本到JSON文件
print(f"保存文本到JSON文件: {wiki_texts_path}")
wiki_dict = {"wiki": wiki_texts}
with open(wiki_texts_path, 'w', encoding='utf-8') as f:
    json.dump(wiki_dict, f, ensure_ascii=False)
print(f"总共保存了 {len(wiki_texts)} 条文本")

# 找出有多个位置的ID
duplicate_ids = {id_: positions for id_, positions in id_to_positions.items() if len(positions) > 1}
print(f"发现{len(duplicate_ids)}个重复的passage_id：")
for id_, positions in duplicate_ids.items():
    print(f"passage_id: {id_} 出现在位置: {positions}")

# 处理输入JSON文件
print(f"加载输入 JSON 文件：{input_json_path}")
with open(input_json_path, "r", encoding="utf-8") as f:
    input_list = json.load(f)

print("处理 passage_id...")
duplicate_passages = []  # 存储包含重复passage_id的数据
unmatched_items = []    # 存储未匹配的数据

for item in tqdm(input_list, desc="Processing input list"):
    item["document"] = "wiki"
    if len(item["ground_truth_ctx"]) != 0:
        has_duplicate = False
        has_unmatched = False
        
        for ground_truth_ctx in item["ground_truth_ctx"]:
            passage_id = ground_truth_ctx["passage_id"]
            positions = id_to_positions.get(passage_id, [])
            
            if len(positions) == 0:
                # passage_id 未找到
                has_unmatched = True
                ground_truth_ctx["index"] = -1
            elif len(positions) > 1:
                # passage_id 有多个位置
                has_duplicate = True
                ground_truth_ctx["index"] = -2  # 使用特殊值标记重复
            else:
                # passage_id 唯一
                ground_truth_ctx["index"] = positions[0]
        
        if has_duplicate:
            duplicate_passages.append(item)
        elif has_unmatched:
            unmatched_items.append(item)

print(f"包含重复passage_id的数据项数量: {len(duplicate_passages)}")
print(f"包含未匹配passage_id的数据项数量: {len(unmatched_items)}")

# 保存包含重复passage_id的数据
duplicate_output_path = "/mnt/abu/evaluation/inscit/test/duplicate_data.json"
with open(duplicate_output_path, "w", encoding="utf-8") as f:
    json.dump(duplicate_passages, f, ensure_ascii=False, indent=4)
print(f"包含重复passage_id的数据已保存到：{duplicate_output_path}")

# 保存未匹配的数据
with open(unmatched_output_path, "w", encoding="utf-8") as f:
    json.dump(unmatched_items, f, ensure_ascii=False, indent=4)
print(f"未匹配的数据已保存到：{unmatched_output_path}")

# 保存所有处理后的数据
with open(output_with_index_path, "w", encoding="utf-8") as f:
    json.dump(input_list, f, ensure_ascii=False, indent=4)
print(f"更新后的数据已成功保存到：{output_with_index_path}")
