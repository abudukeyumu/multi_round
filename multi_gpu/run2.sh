CUDA_VISIBLE_DEVICES=0,1,2,3,4 python /mnt/abu/evaluation/xiugai/evaluate.py \
    --eval-dataset topiocqa \
    --query-encoder-path /mnt/abu/models/dragon-plus-query-encoder\
    --context-encoder-path /mnt/abu/models/dragon-plus-context-encoder