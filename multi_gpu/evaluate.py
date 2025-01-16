from transformers import AutoModel, AutoTokenizer
from dataset import get_data_for_evaluation
from arguments import get_args
from tqdm import tqdm
import torch
import os
from mtrag11 import MTRAG
import math

def run_retrieval(eval_data, documents, query_encoder_path, context_encoder_path, max_seq_len=512):
    ranked_indices_list = []
    gold_index_list = []
    for doc_id in tqdm(eval_data, desc="Running retrieval"):
        mtrag = MTRAG(context_encoder_path, query_encoder_path)
        context_list = documents[doc_id]

        root_faiss_file = "/mnt/abu/evaluation/xiugai/faiss"
        model_name = os.path.basename(context_encoder_path)
        save_path = os.path.join(root_faiss_file, f"{model_name}-faiss.index")
        # import pdb
        # pdb.set_trace()
        mtrag.initialize_index(context_list,index_path =save_path)
        with torch.no_grad():
            # get chunk embeddings
            sample_list = eval_data[doc_id]
            queries = []
            for item in sample_list:
                gold_idx = item['gold_idx']
                gold_index_list.append(gold_idx)
                queries.append(item['query'])

            query_embeddings = mtrag.encode_corpus(queries,use_query_encoder =True)
            ranked_indices_list = mtrag.faiss_search(query_embeddings,k=20)

    return ranked_indices_list, gold_index_list

def calculate_recall(ranked_indices_list, gold_index_list, topk):
    hit = 0
    for ranked_indices, gold_index in zip(ranked_indices_list, gold_index_list):
        if gold_index in ranked_indices[:topk]:
            hit += 1
    recall = hit / len(ranked_indices_list)
    print("Top-%d Recall: %.4f" % (topk, recall))

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

    ## 获取评估数据路径
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

    ## 运行检索
    ranked_indices_list, gold_index_list = run_retrieval(
        eval_data,
        documents,
        args.query_encoder_path,
        args.context_encoder_path
    )
    print("总测试样本数: %d" % len(ranked_indices_list))

    ## 计算评估指标
    print("正在评估数据集: %s" % args.eval_dataset)
    
    # 计算 Recall@K
    topk_list = [1, 5, 20]
    for topk in topk_list:
        calculate_recall(ranked_indices_list, gold_index_list, topk=topk)
    
    # 计算 MRR
    calculate_mrr(ranked_indices_list, gold_index_list)
    
    # 计算 NDCG@20
    calculate_ndcg(ranked_indices_list, gold_index_list, k=20)

if __name__ == "__main__":
    main()
