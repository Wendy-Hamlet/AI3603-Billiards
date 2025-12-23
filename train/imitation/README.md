# 模仿学习训练框架

本目录包含基于MCTS数据的模仿学习训练框架。

## 目录结构

```
train/imitation/
├── collect_data.py        # 数据收集脚本（增量收集，多核并行）
├── model.py               # 连续动作模型定义
├── model_discrete.py      # 离散 phi 模型定义
├── train.py               # 连续动作训练脚本
├── train_discrete.py      # 离散 phi 训练脚本
├── train_augmented.py     # 带数据增强的训练脚本
├── agent_imitation.py     # 连续动作推理 Agent
├── agent_discrete.py      # 离散 phi 推理 Agent（支持 Top-K + 模拟验证）
├── evaluate_imitation.py  # 连续动作评估脚本
├── evaluate_discrete.py   # 离散 phi 评估脚本
├── data/                  # 10k 数据集
├── data_100k/             # 100k 数据集（124万样本）
├── checkpoints/           # 连续动作 large 模型
├── checkpoints_discrete_v2/  # 离散 phi large 模型 (best: 50.19% phi acc)
├── checkpoints_aug_small/ # 带增强的 small 模型
└── checkpoints_100k_small/# 100k 数据 small 模型
```

## 训练结果

| 模型 | Phi Acc | Val Loss | 胜率 | 备注 |
|------|---------|----------|------|------|
| discrete_v2 (epoch 60) | 50.19% | 3.7579 | ~32% | 36 bins, 100k 数据 |
| aug_small (epoch 101) | - | 0.65 | ~30% | 连续 phi, 带增强 |

## 使用流程

### 1. 数据收集

```bash
# 增量收集（追加到已有数据）
python collect_data.py \
    --num_games 100000 \
    --num_workers 64 \
    --cpu_cores 64 \
    --mcts_simulations 200 \
    --mcts_candidates 32 \
    --output_dir ./data_200k \
    --append \
    --checkpoint_interval 2000
```

### 2. 离散 phi 训练（推荐）

```bash
# 72 bins = 5° 精度
python train_discrete.py \
    --data_dir ./data_100k \
    --output_dir ./checkpoints_discrete_72bins \
    --model_type large \
    --num_phi_bins 72 \
    --batch_size 256 \
    --lr 0.0003 \
    --num_epochs 200 \
    --no_amp  # 禁用 AMP 防止 NaN
```

### 3. 评估

```bash
# 基础评估
python evaluate_discrete.py \
    --checkpoint ./checkpoints_discrete_v2/checkpoint_best.pt \
    --model_type large \
    --num_games 40

# 带 Top-K + 模拟验证
python evaluate_discrete.py \
    --checkpoint ./checkpoints_discrete_v2/checkpoint_best.pt \
    --model_type large \
    --num_games 40 \
    --use_simulation \
    --top_k 5 \
    --sim_per_candidate 3
```

## 模型架构

### 离散 Phi 模型 (推荐)

- **优势**: 避免多模态平均化问题
- **phi 预测**: 分类任务（36/72 bins）
- **其他参数**: 回归任务（V0, theta, a, b）

### 输入特征 (80维)

| 特征 | 维度 | 说明 |
|------|------|------|
| 白球位置 | 3 | x, y, pocketed |
| 15球位置 | 45 | 每球 x, y, pocketed |
| 目标球 mask | 15 | 是否为己方目标 |
| 袋口位置 | 12 | 6 袋口 × 2 |
| 统计特征 | 5 | 剩余球数等 |

## 关键发现

1. **Phi 是主要难点**: 击球方向的多模态性导致回归困难
2. **离散化有效**: 将 phi 转为分类任务可避免平均化
3. **数据质量 > 数量**: MCTS 胜率高的样本更有价值
4. **AMP 需谨慎**: 混合精度训练可能导致 NaN

## 下一步改进

- [ ] 增加 phi bins 到 72 (5° 精度)
- [ ] 收集更多数据 (200k+)
- [ ] Top-K + 模拟验证推理
- [ ] 分层训练（先 phi 后其他）
