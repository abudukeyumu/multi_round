import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import faiss
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import os
from functools import partial
from niuload import balanced_load

class MTRAG:
    def __init__(self, context_model_name_or_path, query_model_name_or_path=None):
        """初始化 MTRAG 模型，支持双塔架构"""
        self.context_encoder = AutoModel.from_pretrained(context_model_name_or_path, torch_dtype="auto").cuda()
        if query_model_name_or_path is None:
            self.query_encoder = self.context_encoder
        else:
            self.query_encoder = AutoModel.from_pretrained(query_model_name_or_path, torch_dtype="auto").cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(context_model_name_or_path)
        self.index = None  # 初始化时不设置索引
        self.norm = None

    # def encode_queries(self, messages, k=5):
    #     """编码查询并进行检索"""
    #     query_embeddings = self.encode_corpus(messages, use_query_encoder=True)

    #     # 使用 FAISS 查找最相似的 k 个文档
    #     D, I = self.index.search(query_embeddings, k=5 * k)  # 获取前 5*k 个候选结果
    #     D, I = D[0], I[0]

    #     return I

    def faiss_search(self, embeddings, k=5):

        D, I = self.index.search(embeddings, k=k)
        return I.tolist()  



    def initialize_index(self, texts,index_path, use_gpu=False, use_query_encoder=False):
        """初始化 FAISS 索引"""
        print("开始编码文本并初始化索引...")
        embeddings = self.encode_corpus(texts, use_query_encoder=use_query_encoder)
        embedding_dim = embeddings.shape[-1]
        self.index = faiss.IndexFlatIP(embedding_dim)
        if use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        self.index.add(embeddings)
        print(f"索引初始化完成，包含 {self.index.ntotal} 个向量。")
        
        if use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, index_path)
        else:
            faiss.write_index(self.index, index_path)

    def load_faiss_index(self,index_path,use_gpu=False):
        """加载已保存的 FAISS 索引"""
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"索引文件未找到：{index_path}")

        # 从磁盘加载索引
        self.index = faiss.read_index(index_path)
        print("成功加载 FAISS 索引。")

        # 如果需要在 GPU 上使用索引
        if use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            print("索引已转移到 GPU。")

    def encode_corpus(self, texts, use_query_encoder=False, rank=0, world_size=1):
        """分布式编码文本"""
        class DocumentDataset(Dataset):
            def __init__(self, texts):
                self.texts = texts

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, index):
                return self.texts[index]

        def collate_fn(batch, tokenizer):
            return tokenizer(batch, padding=True, max_length=512, truncation=True, return_tensors='pt')

        # 将数据集分配给不同的 GPU
        dataset = DocumentDataset(texts)
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        dataloader = DataLoader(
            dataset,
            batch_size=2048,
            shuffle=False,
            num_workers=0,
            collate_fn=partial(collate_fn, tokenizer=self.tokenizer),
            pin_memory=True,
            sampler=sampler
        )

        encoder = self.query_encoder if use_query_encoder else self.context_encoder
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            encoder = torch.nn.DataParallel(encoder)

            
        doc_embeddings = []

        for encoded_input in tqdm(dataloader, desc="编码文本"):
            encoded_input = encoded_input.to("cuda")
            with torch.inference_mode():
                model_output = encoder(**encoded_input).last_hidden_state[:, 0, :]
                embeddings = model_output.cpu().numpy()
                doc_embeddings.append(embeddings)

        doc_embeddings = np.vstack(doc_embeddings)

        return doc_embeddings
