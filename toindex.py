import os
from datasets import load_dataset
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

def main():
    # 配置参数
    model_path = "/mnt/abu/models/gte-large-en-v1.5"  
    dataset_path = "/mnt/abu/data/wiki"        
    TEXT_COLUMN = "text"  
    chunk_size = 512
    chunk_overlap = 50    # 重叠部分
    faiss_index_path = "/mnt/abu/multi_round/faiss_index"  # FAISS索引保存路径

    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"trust_remote_code": True}  # 设置 trust_remote_code=True
    )

    # 使用 load_dataset 加载数据
    print("加载数据集...")
    dataset = load_dataset("parquet", data_files=f"{dataset_path}/*.parquet")
    print(f"成功加载数据集，共有 {len(dataset['train'])} 条记录。")

    # 提取所有文本内容
    texts_raw = dataset['train'][TEXT_COLUMN]
    print(f"共有 {len(texts_raw)} 条文本记录。")

    # 初始化文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    texts = []
    for content in tqdm(texts_raw, desc="处理文档"):
        if content:  # 检查非空文本
            splits = text_splitter.split_text(content)
            texts.extend(splits)

    print(f"总共分割为 {len(texts)} 个文本块。")

    print("创建 FAISS 向量数据库...")
    vector_store = FAISS.from_texts(texts, embeddings)

    print(f"保存 FAISS 索引到 {faiss_index_path} 中...")
    vector_store.save_local(faiss_index_path)
    print("FAISS 索引保存完毕。")

if __name__ == "__main__":
    main()