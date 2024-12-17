import faiss
import numpy as np
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import json

def load_data(split_docs_path) :
    """加载文本数据"""
    texts = []
    if os.path.isdir(split_docs_path):
        for filename in os.listdir(split_docs_path):
            if filename.endswith('.json'):
                with open(os.path.join(split_docs_path, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    texts.extend(data)
    elif split_docs_path.endswith('.json'):
        with open(split_docs_path, 'r', encoding='utf-8') as f:
            texts = json.load(f)
    return texts

def create_embeddings(texts, tokenizer, context_encoder, batch_size= 2048,device: str = 'cuda' if torch.cuda.is_available() else 'cpu') :
    embeddings = []

    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU")
        context_encoder = torch.nn.DataParallel(context_encoder)
    
    context_encoder = context_encoder.to(device)
    
    for i in tqdm(range(0, len(texts), batch_size), desc="生成文本向量"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts,padding=True,truncation=True,max_length=512,return_tensors='pt')
        
        with torch.no_grad():
            outputs = context_encoder(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)
            
    return np.vstack(embeddings)

def build_index(embeddings, dimension= 1024):
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def save_index_and_texts(index, texts, save_dir):
    """保存FAISS索引和文本数据"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存FAISS索引
    index_save_path = os.path.join(save_dir, "faiss_flatl2.index")
    faiss.write_index(index, index_save_path)
    print(f"索引已保存至: {index_save_path}")
    
    # 保存文本数据
    texts_save_path = os.path.join(save_dir, "texts.json")
    with open(texts_save_path, 'w', encoding='utf-8') as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    print(f"文本数据已保存至: {texts_save_path}")

def main():
    # 配置参数
    context_model_path = "/mnt/abu/models/dragon-multiturn-context-encoder"
    query_model_path = "/mnt/abu/models/dragon-multiturn-query-encoder"
    split_docs_path = "/mnt/abu/multi/cluster/zifu_512_over_0_11_9/quchong"
    save_dir = "/mnt/abu/12_9/llama_3.3/query_context/faiss"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("正在加载数据集...")
    texts = load_data(split_docs_path)
    
    print("正在加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(context_model_path)
    context_encoder = AutoModel.from_pretrained(context_model_path)
    
    print("正在生成文档向量...")
    embeddings = create_embeddings(texts, tokenizer, context_encoder, device=device)
    
    print("正在构建FAISS索引...")
    dimension = embeddings.shape[1]
    index = build_index(embeddings, dimension)
    
    save_index_and_texts(index, texts, save_dir)

if __name__ == "__main__":
    main()