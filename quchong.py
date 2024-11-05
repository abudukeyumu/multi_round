import os
import json
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from datasketch import MinHash, MinHashLSH

def load_split_documents(file_path):
    print(f"从 {file_path} 加载分割后的文档")

    with open(file_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    all_splits = []
    for doc in tqdm(documents, desc="处理文档"):
        all_splits.extend(doc['split_text'])
        
    print(f"成功加载 {len(documents)} 个文档，共 {len(all_splits)} 个文本块")
    return all_splits

def create_minhash(text, num_perm):
    m = MinHash(num_perm=num_perm)
    for d in text:
        m.update(d.encode('utf8'))
    return m

def worker(text):
    return create_minhash(text, 256)

def deduplicate_texts_parallel(texts, threshold=0.9, num_perm=256, batch_size=1000000):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    unique_indices = []
    print("开始分批处理文本\n")
    
    for start in tqdm(range(0, len(texts), batch_size), desc="建minhashes"):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        minhashes = process_map(worker, batch_texts, max_workers=os.cpu_count(), chunksize=1000)

        for i, minhash in enumerate(minhashes):
            result = lsh.query(minhash)
            if not result:
                lsh.insert(f'doc_{start + i}', minhash)
                unique_indices.append(start + i)

    print("原始数据大小：", len(texts))
    print("\n去重后的数据大小", len(unique_indices))
    
    return [texts[i] for i in unique_indices]
    
# def deduplicate_texts_parallel(texts, embeddings, threshold=0.9, num_perm=128):
#     lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
#     unique_indices = []

#     # 使用 process_map 来并行处理数据并显示进度条
#     minhashes = process_map(worker, texts, max_workers=os.cpu_count(), chunksize=1000)

#     for i, minhash in tqdm(enumerate(minhashes)):
#         result = lsh.query(minhash)
#         if not result:
#             lsh.insert(f'doc_{i}', minhash)
#             unique_indices.append(i)
    
#     print("原始数据大小：", len(texts))
#     print("\n去重后的数据大小", len(unique_indices))
    
#     unique_embeddings = embeddings[unique_indices]
#     return [texts[i] for i in unique_indices], unique_embeddings

def save_data(texts, text_base_path, batch_size=1000000):
    num_batches = len(texts) // batch_size + (1 if len(texts) % batch_size != 0 else 0)
    
    for i in range(num_batches):
        batch_texts = texts[i * batch_size:(i + 1) * batch_size]
        
        text_path = os.path.join(text_base_path, f"{i}.json")
        
        with open(text_path, 'w', encoding='utf-8') as f:
            json.dump(batch_texts, f, ensure_ascii=False, indent=2)

def main():
    texts_dir = '/mnt/abu/multi/cluster/zifu_512_over_0/split_documents_no_references.json'
    output_text_path = "/mnt/abu/multi/cluster/zifu_512_over_0/texts_no_references/quchong"

    all_texts = load_split_documents(texts_dir)

    unique_texts = deduplicate_texts_parallel(all_texts)

    save_data(unique_texts, output_text_path)
    
    print("数据保存完毕")

if __name__ == "__main__":
    main()