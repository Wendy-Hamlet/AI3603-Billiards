#!/bin/bash
# 并行训练启动脚本
# 使用方法: bash start_parallel_training.sh [--resume checkpoint_path]

echo "============================================================"
echo "🚀 启动并行 SAC 训练"
echo "============================================================"

# 检查参数
if [ "$1" == "--resume" ]; then
    echo "📂 恢复训练: $2"
    python train/train_sac_parallel.py --resume "$2"
else
    echo "🆕 从头开始训练"
    python train/train_sac_parallel.py
fi
