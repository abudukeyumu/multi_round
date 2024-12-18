from transformers import AutoModel, AutoTokenizer
from dataset import get_data_for_evaluation
from arguments import get_args
from tqdm import tqdm
import torch
import os
import numpy as np


def run_retrieval(eval_data, documents, query_encoder, context_encoder, tokenizer, max_seq_len=512):
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        query_encoder = torch.nn.DataParallel(query_encoder)
        context_encoder = torch.nn.DataParallel(context_encoder)
    batch_size = 4096
    ranked_indices_list = []
    gold_index_list = []
    
    for doc_id in tqdm(eval_data, desc="Documents", position=0):
        context_list = documents[doc_id]
        sample_list = eval_data[doc_id]
        
        # 准备queries
        queries = []
        for item in sample_list:
            gold_idx = item['gold_idx']
            gold_index_list.append(gold_idx)
            queries.append(item['query'])
            
        with torch.no_grad():
            # 获取所有query embeddings

            query_embs = []
            for i in tqdm(range(0, len(queries), batch_size), desc="Query embeddings", position=1, leave=False):
                batch_queries = queries[i:i + batch_size]
                query_inputs = tokenizer(batch_queries, max_length=max_seq_len, padding=True, truncation=True, return_tensors="pt").to("cuda")
                q_emb = query_encoder(**query_inputs)
                q_emb = q_emb.last_hidden_state[:, 0, :].cpu().numpy()
                query_embs.append(q_emb)

                del query_inputs, q_emb
                torch.cuda.empty_cache()
                
            query_embs = np.vstack(query_embs)
            torch.cuda.empty_cache()
            # 获取所有context embeddings
            context_embs = []
            for i in tqdm(range(0, len(context_list), batch_size), desc="Context embeddings", position=1, leave=False):
                batch_chunks = context_list[i:i + batch_size]
                chunk_inputs = tokenizer(batch_chunks, max_length=max_seq_len, padding=True, truncation=True, return_tensors="pt").to("cuda")
                c_emb = context_encoder(**chunk_inputs)
                c_emb = c_emb.last_hidden_state[:, 0, :].cpu().numpy()
                context_embs.append(c_emb)

                # 清理本次循环的变量
                del chunk_inputs, c_emb
                torch.cuda.empty_cache()

            context_embs = np.vstack(context_embs)
            
            # 分批计算相似度

            all_similarities = []
            for i in tqdm(range(0, len(query_embs), batch_size), desc="Computing similarities (queries)", position=1, leave=False):
                query_batch = torch.from_numpy(query_embs[i:i+batch_size]).cuda()
                
                batch_similarities = []
                for j in tqdm(range(0, len(context_embs), batch_size), desc="Computing similarities (contexts)", position=2, leave=False):
                    context_batch = torch.from_numpy(context_embs[j:j+batch_size]).cuda()
                    sim = torch.matmul(query_batch, context_batch.t())
                    batch_similarities.append(sim.cpu())

                    del sim
                    torch.cuda.empty_cache()
                
                # 对当前query batch的所有结果进行合并和排序
                
                query_similarities = torch.cat(batch_similarities, dim=1)
                k = min(100, query_similarities.size(1))
                _, top_indices = torch.topk(query_similarities, k=k, dim=-1)
                ranked_indices_list.extend(top_indices.tolist())
                
                del query_similarities, top_indices, batch_similarities
                torch.cuda.empty_cache()
            #     query_similarities = torch.cat(batch_similarities, dim=1)
            #     all_similarities.append(query_similarities)
            
            # similarities = torch.cat(all_similarities, dim=0)
            # ranked_results = torch.argsort(similarities, dim=-1, descending=True)
            # ranked_indices_list.extend(ranked_results.tolist())
            
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


def main():
    args = get_args()

    tokenizer = AutoTokenizer.from_pretrained(args.query_encoder_path)

    ## get retriever model
    query_encoder = AutoModel.from_pretrained(args.query_encoder_path)
    context_encoder = AutoModel.from_pretrained(args.context_encoder_path)
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
    elif args.eval_dataset == "wiki":
        input_datapath = os.path.join(args.data_folder, args.wiki_datapath)
        input_docpath = os.path.join(args.data_folder, args.wiki_docpath)
    elif args.eval_dataset == "topiocqa" or args.eval_dataset == "inscit":
        raise Exception("We have prepared the function to get queries, but a Wikipedia corpus needs to be downloaded")
    else:
        raise Exception("Please input a correct eval_dataset name!")

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


if __name__ == "__main__":
    main()