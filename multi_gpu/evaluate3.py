from transformers import AutoModel, AutoTokenizer
from dataset import get_data_for_evaluation
from arguments import get_args
from tqdm import tqdm
import torch
import os
import numpy as np
import faiss
import math

def load_faiss_index(index_path,use_gpu=False):
    """加载已保存的 FAISS 索引"""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"索引文件未找到：{index_path}")
    
    # 从磁盘加载索引
    index = faiss.read_index(index_path)
    print("成功加载 FAISS 索引。")
    print(f"索引初始化完成，包含 {index.ntotal} 个向量。")
    # 如果需要在 GPU 上使用索引
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        print("索引已转移到 GPU。")
    
    return index
def initialize_index(embeddings, use_gpu=False):
    """初始化 FAISS 索引"""
    print("开始编码文本并初始化索引...")
    embedding_dim = embeddings.shape[-1]
    index = faiss.IndexFlatIP(embedding_dim)
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(embeddings)
    print(f"索引初始化完成，包含 {index.ntotal} 个向量。")
    
    # index_path = os.path.join(save_path, "index.faiss")
    # if use_gpu:
    #     cpu_index = faiss.index_gpu_to_cpu(index)
    #     faiss.write_index(cpu_index, index_path)
    # else:
    #     faiss.write_index(index, index_path)
    return index


    
def run_retrieval(eval_data, documents, query_encoder, context_encoder, tokenizer, max_seq_len=512):
    # index_path = "/mnt/abu/evaluation/topiocqa/embeddings/dragon-plus"
    ranked_indices_list = []
    gold_index_list = []
    for doc_id in tqdm(eval_data):

        # context_embeddings = np.load("/mnt/abu/evaluation/topiocqa/embeddings/dragon-plus/final_embeddings.pt.npy")
        context_embeddings = np.load("/mnt/abu/evaluation/topiocqa/embeddings/dragon-plus-automodel/final_embeddings.pt.npy")


        index  =initialize_index(context_embeddings)
        # index =load_faiss_index("/mnt/abu/evaluation/xiugai/faiss/index.faiss") 


        sample_list = eval_data[doc_id]
        queries = []
        for item in sample_list:
            gold_idx = item['gold_idx']
            gold_index_list.append(gold_idx)
            queries.append(item['query'])
        # import pdb
        # pdb.set_trace()
        with torch.no_grad():
            # 获取所有query embeddings
            batch_size =512
            query_embs = []
            for i in tqdm(range(0, len(queries), batch_size), desc="Query embeddings", position=1, leave=False):
                batch_queries = queries[i:i + batch_size]
                query_inputs = tokenizer(batch_queries, max_length=max_seq_len, padding=True, truncation=True, return_tensors="pt").to("cuda")
                q_emb = query_encoder(**query_inputs)
                q_emb = q_emb.last_hidden_state[:, 0, :].cpu().numpy()
                query_embs.append(q_emb)

        query_embs = np.vstack(query_embs)
        
        D, I = index.search(query_embs, k=100)
        import pdb
        pdb.set_trace()  
        ranked_indices_list = [indices.tolist() for indices in I]
        # import pdb
        # pdb.set_trace()
       
            
    return ranked_indices_list, gold_index_list

def calculate_recall(ranked_indices_list, gold_index_list, topk):
    hit = 0
    for ranked_indices, gold_index in zip(ranked_indices_list, gold_index_list):
        for idx in ranked_indices[:topk]:
            if idx == gold_index:
                hit += 1
                break
    recall = hit / len(ranked_indices_list)

    print("top-%d recall score: %.4f" % (topk, recall))

def calculate_mrr(ranked_indices_list, gold_index_list):
    reciprocal_rank_sum = 0.0
    for ranked_indices, gold_index in zip(ranked_indices_list, gold_index_list):
        try:
            rank = ranked_indices.index(gold_index) + 1  # ranks start at 1
            reciprocal_rank_sum += 1.0 / rank
        except ValueError:
            # gold_index not found in ranked_indices
            reciprocal_rank_sum += 0.0
    mrr = reciprocal_rank_sum / len(ranked_indices_list)
    print("Mean Reciprocal Rank (MRR): %.4f" % mrr)

def calculate_ndcg(ranked_indices_list, gold_index_list, k=20):
    dcg = 0.0
    idcg = 0.0
    for ranked_indices, gold_index in zip(ranked_indices_list, gold_index_list):
        try:
            rank = ranked_indices.index(gold_index) + 1
            if rank <= k:
                dcg += 1.0 / math.log2(rank + 1)
        except ValueError:
            dcg += 0.0
        # 理想情况下，相关文档排名在第一位
        idcg += 1.0 / math.log2(1 + 1) if k >= 1 else 0.0
    ndcg = dcg / idcg if idcg > 0 else 0.0
    print("Normalized Discounted Cumulative Gain (NDCG@%d): %.4f" % (k, ndcg))
def main():
    args = get_args()

    tokenizer = AutoTokenizer.from_pretrained(args.query_encoder_path)

    ## get retriever model
    query_encoder = AutoModel.from_pretrained(args.query_encoder_path)
    context_encoder = AutoModel.from_pretrained(args.context_encoder_path)
    # from sentence_transformers import SentenceTransformer
    # query_encoder = SentenceTransformer(args.query_encoder_path).to("cuda")
    # context_encoder = SentenceTransformer(args.context_encoder_path).to("cuda")
    query_encoder.to("cuda"), query_encoder.eval()
    context_encoder.to("cuda"), context_encoder.eval()


    ## get evaluation data
    if args.eval_dataset == "doc2dial":
        input_datapath = os.path.join(args.data_folder, args.doc2dial_datapath)
        input_docpath = os.path.join(args.data_folder, args.doc2dial_docpath)
    elif args.eval_dataset == "quac":
        input_datapath = os.path.join(args.data_folder, args.quac_datapath)
        input_docpath = os.path.join(args.data_folder, args.quac_docpath)
    elif args.eval_dataset == "qrecc":
        input_datapath = os.path.join(args.data_folder, args.qrecc_datapath)
        input_docpath = os.path.join(args.data_folder, args.qrecc_docpath)
    elif args.eval_dataset == "topiocqa":
        input_datapath =  args.topiocqa_datapath
        input_docpath =  args.topiocqa_docpath
    elif args.eval_dataset in [ "inscit"]:
        raise Exception("我们已经准备好了获取查询的函数，但需要下载维基百科语料库。")
    else:
        raise Exception("请输入正确的 eval_dataset 名称！")

    eval_data, documents = get_data_for_evaluation(input_datapath, input_docpath, args.eval_dataset)
    # import pdb
    # pdb.set_trace()
    # truncated_documents = {}
    # for doc_id, context_list in documents.items():
    #     # 只保留前2048个chunks
    #     truncated_documents[doc_id] = context_list[:4096]
    ## run retrieval
    # ranked_indices_list, gold_index_list = run_retrieval(eval_data, documents, query_encoder, context_encoder, tokenizer=None)
    ranked_indices_list, gold_index_list = run_retrieval(eval_data, documents, query_encoder, context_encoder, tokenizer)
    print("number of the total test samples: %d" % len(ranked_indices_list))

    ## calculate recall scores
    print("evaluating on %s" % args.eval_dataset)
    topk_list = [1, 5, 20]
    for topk in topk_list:
        calculate_recall(ranked_indices_list, gold_index_list, topk=topk)
    calculate_mrr(ranked_indices_list, gold_index_list)
    
    # 计算 NDCG@20
    calculate_ndcg(ranked_indices_list, gold_index_list, k=20)


if __name__ == "__main__":
    main()