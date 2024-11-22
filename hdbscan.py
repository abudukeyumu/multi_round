
import os
import numpy as np
import json
from tqdm import tqdm
from cuml.cluster.hdbscan import HDBSCAN
from collections import defaultdict
import cupy as cp

def load_embeddings_and_texts(embeddings_dir, texts_dir):
    embeddings_list = []
    texts_list = []

    embedding_files = sorted([f for f in os.listdir(embeddings_dir) if f.endswith('.npy')])
    text_files = sorted([f for f in os.listdir(texts_dir) if f.endswith('.json')])

    assert len(embedding_files) == len(text_files), "嵌入文件和文本文件的批次数不匹配"

    for emb_file, txt_file in tqdm(zip(embedding_files, text_files), total=len(embedding_files), desc="加载批次数据"):
        # 加载嵌入
        embeddings = np.load(os.path.join(embeddings_dir, emb_file))
        embeddings_list.append(embeddings)

        # 加载文本
        with open(os.path.join(texts_dir, txt_file), 'r', encoding='utf-8') as f:
            texts = json.load(f)
            texts_list.extend(texts)

    # 合并所有嵌入为一个大数组
    all_embeddings = np.vstack(embeddings_list)
    all_texts = texts_list

    print(f"总共加载了 {len(all_texts)} 个文档的嵌入")
    return all_embeddings, all_texts

def sample_and_cluster(embeddings, texts, sample_size=2000000, min_cluster_size=3, max_cluster_size=5):
    # 随机采样
    if embeddings.shape[0] > sample_size:
        sampled_indices = np.random.choice(embeddings.shape[0], sample_size, replace=False)
        sampled_embeddings = embeddings[sampled_indices]
        sampled_texts = [texts[i] for i in sampled_indices]
    else:
        sampled_embeddings = embeddings
        sampled_texts = texts
    print("\n采样完成,采样的数据数为:", len(sampled_texts))
    
    # 在 GPU 上进行聚类
    # sampled_embeddings_gpu = cp.asarray(sampled_embeddings).astype(cp.float32)
    sampled_embeddings_gpu = sampled_embeddings
    # import pdb
    # pdb.set_trace()
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_cluster_size,
        max_cluster_size=max_cluster_size,
        prediction_data=True,
        gen_min_span_tree=True,
        cluster_selection_epsilon=0.0
    )
    import time
    start_time = time.time()
    print("\n开始进行采样数据的 GPU 加速的 HDBSCAN 聚类...")
    sampled_labels = clusterer.fit_predict(sampled_embeddings_gpu)
    print("采样数据的 GPU 加速的 HDBSCAN 聚类完成。")
    sampled_labels = cp.asnumpy(sampled_labels)
    end_time = time.time()
    print("\nGPU 聚类时间：", end_time - start_time)
    return sampled_labels, sampled_texts

def save_raw_clusters(cluster_labels, texts, save_path='raw_clusters.json'):
    # 使用字典存储簇信息
    clusters = defaultdict(list)
    
    # 将样本按簇组织
    for idx, label in enumerate(cluster_labels):
        if label != -1:  # 排除噪声点
            clusters[int(label)].append(idx)
    
    # 准备保存的数据
    cluster_data = {}
    for cluster_id, indices in clusters.items():
        cluster_data[cluster_id] = {
            "documents": [texts[idx] for idx in indices],
            "size": len(indices)
        }
    
    # 生成一些统计信息
    stats = {
        "total_clusters": len(clusters),
        "noise_points": len([l for l in cluster_labels if l == -1]),
        "cluster_sizes": {k: len(v) for k, v in clusters.items()},
        "avg_cluster_size": sum(len(v) for v in clusters.values()) / len(clusters) if clusters else 0
    }
    
    # 保存结果
    output_data = {
        "clusters": cluster_data,
        "statistics": stats
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"原始聚类结果已保存到 {save_path}")
    print(f"总簇数: {stats['total_clusters']}")
    print(f"噪声点数: {stats['noise_points']}")
    print(f"平均簇大小: {stats['avg_cluster_size']:.2f}")

def main():
    embeddings_dir = '/data/abudukeyumu/11_7/zifu_512_over_0_11_9/quchong/embeddings_no_references'
    texts_dir = '/data/abudukeyumu/11_7/zifu_512_over_0_11_9/quchong/texts_no_references'

    # 第一步：加载嵌入和文本
    all_embeddings, all_texts = load_embeddings_and_texts(embeddings_dir, texts_dir)

    # 第二步：采样并进行 GPU 加速的 HDBSCAN 聚类
    cluster_labels, sampled_texts = sample_and_cluster(
        all_embeddings,
        all_texts,
        sample_size=6000000,  # 根据 GPU 内存调整采样大小
        min_cluster_size=3,
        max_cluster_size=10
    )

    # 第三步：保存聚类结果
    save_raw_clusters(cluster_labels, sampled_texts, save_path='/data/abudukeyumu/11_7/zifu_512_over_0_11_9/min3_50/sample_no_references_clusters_gpu.json')

if __name__ == "__main__":
    main()
