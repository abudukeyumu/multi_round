#!/bin/bash


export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/mnt/abu/evaluation:$PYTHONPATH

# 训练数据和模型路径
TRAIN_DATA="/mnt/abu/evaluation/train/output/train_data.pt"
MODEL_PATH="/mnt/abu/models/dragon-plus"
# MODEL_PATH="/mnt/abu/models/bge-large-en-v1.5"
OUTPUT_DIR="/mnt/abu/evaluation/train/dragon-batch-8-gradient-4"

# 训练参数
NUM_EPOCHS=3
BATCH_SIZE=8
LEARNING_RATE=3e-5
MAX_LENGTH=512
WARMUP_STEPS=450
GRADIENT_ACCUMULATION_STEPS=4

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 启动训练
python /mnt/abu/evaluation/train/train.py \
    --train_data ${TRAIN_DATA} \
    --model_path ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --max_length ${MAX_LENGTH} \
    --warmup_steps ${WARMUP_STEPS} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --device cuda