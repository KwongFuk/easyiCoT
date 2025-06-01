#!/bin/bash

# 运行第一个命令，并将输出保存到第一个日志文件
echo "Starting training with token1..."
CUDA_VISIBLE_DEVICES=4,5,6,7 DISABLE_VERSION_CHECK=1 llamafactory-cli train examples/train_full/ablation/qwen2vl_interleave_token1.yaml > token1.log 2>&1
echo "Training with token1 completed."

# 运行第二个命令，并将输出保存到第二个日志文件
echo "Starting training with token2..."
CUDA_VISIBLE_DEVICES=4,5,6,7 DISABLE_VERSION_CHECK=1 llamafactory-cli train examples/train_full/ablation/qwen2vl_interleave_token2.yaml > token2.log 2>&1
echo "Training with token2 completed."

# 运行第三个命令，并将输出保存到第三个日志文件
echo "Starting training with token3..."
CUDA_VISIBLE_DEVICES=4,5,6,7 DISABLE_VERSION_CHECK=1 llamafactory-cli train examples/train_full/ablation/qwen2vl_interleave_token3.yaml > token3.log 2>&1
echo "Training with token3 completed."

# 运行第四个命令，并将输出保存到第四个日志文件
echo "Starting training with token4..."
CUDA_VISIBLE_DEVICES=4,5,6,7 DISABLE_VERSION_CHECK=1 llamafactory-cli train examples/train_full/ablation/qwen2vl_interleave_token4.yaml > token4.log 2>&1
echo "Training with token4 completed."

echo "All training completed."
