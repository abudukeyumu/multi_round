import faiss
import numpy as np
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import json


def get_query(messages, num_turns=5):
    query = ""
    for item in messages[-num_turns:]:
        item['role'] = item['role'].replace("assistant", "agent")
        query += "{}: {}\n".format(item['role'], item['content'])
    query = query.strip()
    
    return query

def load_index_and_texts(save_dir):
    """加载FAISS索引和文本数据"""
    # 加载FAISS索引
    index_path = os.path.join(save_dir, "faiss_flatl2.index")
    index = faiss.read_index(index_path)
    
    # 加载文本数据
    texts_path = os.path.join(save_dir, "texts.json")
    with open(texts_path, 'r', encoding='utf-8') as f:
        texts = json.load(f)
    
    return index, texts


def create_embeddings(texts, tokenizer, encoder, batch_size=2048,device='cuda' if torch.cuda.is_available() else 'cpu') :
    embeddings = []
    
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU")
        encoder = torch.nn.DataParallel(encoder)
    
    encoder = encoder.to(device)
    
    for i in tqdm(range(0, len(texts), batch_size), desc="生成文本向量"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts,padding=True,truncation=True,max_length=512,return_tensors='pt')
  
        with torch.no_grad():
            outputs = encoder(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)
            
    return np.vstack(embeddings)


def search_index(index, query_embedding , k= 20):
    return index.search(query_embedding, k)

def main():
    # 配置参数
    query_model_path = "/mnt/abu/models/dragon-multiturn-query-encoder"
    faiss_path = "/mnt/abu/12_9/llama_3.3/query_context/faiss"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = "/mnt/abu/12_9/llama_3.3/data/wiki_test.json"
    output_path = "/mnt/abu/12_9/llama_3.3/query_context/output.json"
    batch_size = 1024
    top_k = 20

    print("正在加载查询数据...")
    with open(data_path, 'r', encoding='utf-8') as f:
        queries_data = json.load(f)
    print(f"已加载 {len(queries_data)} 条查询数据")

    # 准备查询
    print("正在准备查询...")
    formatted_queries = []
    for item in tqdm(queries_data, desc="准备查询"):
        query = get_query(item['messages'])
        formatted_queries.append(query)

    # 加载模型
    print("正在加载查询编码器...")
    tokenizer = AutoTokenizer.from_pretrained(query_model_path)
    query_encoder = AutoModel.from_pretrained(query_model_path)

    # 生成查询向量
    print("正在生成查询向量...")
    query_embeddings = create_embeddings(formatted_queries, tokenizer, query_encoder, batch_size=batch_size,device=device)

    # 加载FAISS索引
    print("正在加载FAISS索引...")
    index, texts = load_index_and_texts(faiss_path)
    
    # 搜索相似向量
    print(f"正在搜索Top-{top_k}相似结果...")
    distances, indices = search_index(index, query_embeddings, k=top_k)  

    # 保存结果
    print("正在保存搜索结果...")

    results = []
    for ind in indices:
        res = []
        for i in ind:
            res.append(texts[i])
        results.append(res)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"搜索结果已保存至: {output_path}")

if __name__ == "__main__":
    main()