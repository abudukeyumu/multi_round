#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python /mnt/abu/12_9/llama_3.3/eval/evaluate.py \
    --eval-dataset wiki \
    --query-encoder-path /mnt/abu/models/dragon-multiturn-query-encoder \
    --context-encoder-path /mnt/abu/models/dragon-multiturn-context-encoder \
    --wiki-datapath /mnt/abu/12_9/llama_3.3/data/wiki_test.json \
    --wiki-docpath /mnt/abu/12_9/llama_3.3/data/documents.json
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python /mnt/abu/12_9/llama_3.3/eval/evaluate.py \
#     --eval-dataset wiki \
#     --query-encoder-path /mnt/abu/models/dragon-plus-query-encoder \
#     --context-encoder-path /mnt/abu/models/dragon-plus-context-encoder \
#     --wiki-datapath /mnt/abu/12_9/llama_3.3/data/wiki_test.json \
#     --wiki-docpath /mnt/abu/12_9/llama_3.3/data/documents.json

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python /mnt/abu/12_9/llama_3.3/eval/evaluate.py \
#     --eval-dataset doc2dial \
#     --query-encoder-path /mnt/abu/models/dragon-multiturn-query-encoder \
#     --context-encoder-path /mnt/abu/models/dragon-multiturn-context-encoder \
#     --doc2dial-datapath /mnt/abu/data/ChatRAG-Bench/data/doc2dial/test.json \
#     --doc2dial-docpath /mnt/abu/data/ChatRAG-Bench/data/doc2dial/documents.json

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python /mnt/abu/12_9/llama_3.3/eval/evaluate.py \
#     --eval-dataset qrecc \
#     --query-encoder-path /mnt/abu/models/dragon-multiturn-query-encoder \
#     --context-encoder-path /mnt/abu/models/dragon-multiturn-context-encoder \
#     --qrecc-datapath /mnt/abu/data/ChatRAG-Bench/data/qrecc/test.json \
#     --qrecc-docpath /mnt/abu/data/ChatRAG-Bench/data/qrecc/documents.json

# CUDA_VISIBLE_DEVICES=0,1,2,3 python /mnt/abu/12_9/llama_3.3/eval/evaluate.py \
#     --eval-dataset quac \
#     --query-encoder-path /mnt/abu/models/dragon-multiturn-query-encoder \
#     --context-encoder-path /mnt/abu/models/dragon-multiturn-context-encoder \
#     --quac-datapath /mnt/abu/data/ChatRAG-Bench/data/quac/test.json \
#     --quac-docpath /mnt/abu/data/ChatRAG-Bench/data/quac/documents.json

# python /mnt/abu/12_9/llama_3.3/eval/evaluate.py \
#     --eval-dataset wiki \
#     --query-encoder-path /mnt/abu/models/gte-large-en-v1.5 \
#     --context-encoder-path /mnt/abu/models/gte-large-en-v1.5 \
#     --wiki-datapath /mnt/abu/12_9/llama_3.3/data/wiki_test.json \
#     --wiki-docpath /mnt/abu/12_9/llama_3.3/data/documents.json