import os
import numpy as np
import faiss
from datasets import load_dataset
from langchain.vectorstores import FAISS
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import json
import torch



def load_split_documents(file_path):
    print(f"从 {file_path} 加载分割后的文档")
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取文档"):
            document = json.loads(line)
            texts.append(document['text'])
    # print(f"成功加载 {len(texts)} 个文本块")
    return texts

def save_split_documents(texts, save_path):
    print(f"保存分割后的文档到 {save_path}")
    with open(save_path, 'w', encoding='utf-8') as f:
        for i, text in enumerate(texts):
            document = {
                "id": i,
                "text": text
            }
            json.dump(document, f, ensure_ascii=False)
            f.write('\n')
    print("分割后的文档保存完毕。")
    
def process_in_batches(texts, embeddings, batch_size=1000000, save_dir='/mnt/abu/multi_round/embeddings'):
    file_counter = 52

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    for i in tqdm(range(51*batch_size, len(texts), batch_size), desc="Processing embeddings"):
        batch = texts[i:i+batch_size]
        batch_embeddings = embeddings.embed_documents(batch)

        # 构建文件路径
        file_path = os.path.join(save_dir, f"{file_counter}.npy")

        # 将批次嵌入保存为 .npy 文件
        np.save(file_path, np.array(batch_embeddings))

        print(f"Saved batch {file_counter} to {file_path}")
        file_counter += 1

def load_all_embeddings(directory='/mnt/abu/multi_round/embeddings'):
    all_embeddings = []
    
    # 确保目录存在
    if not os.path.isdir(directory):
        raise ValueError(f"The specified path is not a directory: {directory}")
    
    # 获取目录中的所有 .npy 文件并排序
    npy_files = sorted([f for f in os.listdir(directory) if f.endswith('.npy')])
    
    # 使用tqdm创建进度条
    for filename in tqdm(npy_files, desc="加载嵌入向量", unit="file"):
        file_path = os.path.join(directory, filename)
        embeddings = np.load(file_path)
        all_embeddings.extend(embeddings)
    
    print(f"总共加载了 {len(all_embeddings)} 个嵌入向量")
    return all_embeddings

def main():
    # 配置参数
    model_path = "/mnt/abu/models/gte-large-en-v1.5"  
    dataset_path = "/mnt/abu/data/wiki"        
    TEXT_COLUMN = "text"  
    chunk_size = 512
    chunk_overlap = 50    # 重叠部分
    faiss_index_path = "/mnt/abu/multi_round/faiss_index"  # FAISS索引保存路径
    split_docs_path = "/mnt/abu/multi_round/split_documents.jsonl"  # 分割后文档保存路径
    embeddings_path = "/mnt/abu/multi_round/embeddings.json"

    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"trust_remote_code": True}  # 设置 trust_remote_code=True
    )

    # # 使用 load_dataset 加载数据
    # print("加载数据集...")
    # dataset = load_dataset("parquet", data_files=f"{dataset_path}/*.parquet")
    # print(f"成功加载数据集，共有 {len(dataset['train'])} 条记录。")

    # # 提取所有文本内容
    # texts_raw = dataset['train'][TEXT_COLUMN]
    # print(f"共有 {len(texts_raw)} 条文本记录。")

    # # 初始化文本分割器
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=chunk_size,
    #     chunk_overlap=chunk_overlap,
    #     separators=["\n\n", "\n", " ", ""]
    # )
    
    # texts = []
    # for content in tqdm(texts_raw, desc="处理文档"):
    #     if content:  # 检查非空文本
    #         splits = text_splitter.split_text(content)
    #         texts.extend(splits)

    # # 保存分割后的文档
    # save_split_documents(texts, split_docs_path)

    texts = load_split_documents(split_docs_path)

    # print(f"总共分割为 {len(texts)} 个文本块。")


    # 批量处理
    # process_in_batches(texts, embeddings)
    import pdb
    pdb.set_trace()
    print("开始加载所有embeddings........\n")
    all_embeddings = load_all_embeddings()

    # 使用累积的嵌入构建索引
    print("开始构建索引........\n")
    text_embedding_pairs = zip(texts, all_embeddings)
    print("开始构建索引2........\n")
    vector_store = FAISS.from_embeddings(text_embedding_pairs, embeddings)
    # vector_store = FAISS.from_embeddings(all_embeddings, embeddings, texts)

    print(f"保存 FAISS 索引到 {faiss_index_path} 中...")
    vector_store.save_local(faiss_index_path)
    print("FAISS 索引保存完毕。")

if __name__ == "__main__":
    main()
