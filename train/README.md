# SAC 自对弈训练

## 概述

使用 Soft Actor-Critic (SAC) 算法进行自对弈训练的台球 AI。

### 特点

- **自对弈训练**: 双方使用同一策略进行对弈，从输赢两个角度学习
- **课程学习**: 从简单环境（1球 vs 1球）逐步过渡到完整游戏（7球 vs 7球）
- **并行数据收集**: 使用 64 个 CPU 核心并行运行游戏
- **GPU 训练**: 使用 Ada 6000 GPU 加速网络更新
- **对称编码**: 己方/对方球使用相同的编码方式

## 目录结构

```
train/
├── sac/
│   ├── networks.py         # Actor, Critic 网络定义
│   ├── sac_agent.py        # SAC 算法实现
│   ├── replay_buffer.py    # 经验回放缓冲区
│   └── reward.py           # 奖励函数
├── environment/
│   ├── pool_wrapper.py     # 环境封装器
│   └── state_encoder.py    # 状态编码器
├── parallel/
│   └── worker.py           # 数据收集 Worker
├── train_selfplay.py       # 主训练脚本
├── config.yaml             # 配置文件
└── README.md               # 本文件
```

## 环境配置

```bash
# 激活 conda 环境
conda activate poolenv

# 安装额外依赖
pip install torch pyyaml tensorboard
```

## 训练

### 快速开始

```bash
cd /path/to/AI3603-Billiards

# 使用默认配置训练
python train/train_selfplay.py

# 指定配置文件
python train/train_selfplay.py --config train/config.yaml

# 指定 Worker 数量
python train/train_selfplay.py --workers 32

# 从检查点恢复
python train/train_selfplay.py --resume checkpoints/sac_selfplay/xxx/sac_ep10000.pt
```

### 配置说明

主要配置项在 `config.yaml` 中:

```yaml
training:
  num_workers: 64          # 并行 Worker 数量
  games_per_batch: 64      # 每批游戏数
  batch_size: 2048         # GPU 训练批量大小
  lr_actor: 3.0e-4         # Actor 学习率
  lr_critic: 3.0e-4        # Critic 学习率
  gamma: 0.99              # 折扣因子

curriculum:
  enabled: true
  stages:
    - name: "stage_1"
      own_balls: 1
      enemy_balls: 1
      episodes: 50000
    # ... 更多阶段
```

## 奖励函数设计

### 终局奖励
| 场景 | 奖励 |
|------|------|
| 我方正常获胜 | +100 |
| 对方犯规导致我方获胜 | 0 |
| 我方犯规导致失败 | -1000 |
| 对方正常获胜 | -50 |
| 平局 | 0 |

### 进球奖励
- 己方进球: 递增奖励（第 k 球得 5k 分）
- 清台进度加成: 剩余球越少，每球价值越高
- 维持球权: +8 分
- 帮对手进球: 递减惩罚

### 走位奖励
- 评估白球到最佳目标球的击球可行性
- 考虑距离、进袋角度、路径遮挡

### 犯规惩罚
- 白球进袋: -25
- 首球犯规: -15
- 未碰库犯规: -12
- 未击中球: -20

## 状态编码

64 维状态向量:
- 白球位置 (2维)
- 黑8球状态 (3维)
- 己方球状态 (7×3=21维, 按距离排序)
- 对方球状态 (7×3=21维, 按距离排序)
- 球袋位置 (6×2=12维)
- 游戏信息 (5维)

## 动作空间

5 维连续动作:
- V0: 初速度 [0.5, 8.0] m/s
- phi: 水平角度 [0, 360]°
- theta: 垂直角度 [0, 45]°
- a: 横向偏移 [-0.4, 0.4]
- b: 纵向偏移 [-0.4, 0.4]

## 监控训练

```bash
# 使用 TensorBoard (如果启用)
tensorboard --logdir logs/sac_selfplay/
```

## 评估

训练完成后，将模型转换为可用于 `evaluate.py` 的 Agent:

```python
# 在 agent.py 中的 NewAgent 类中加载训练好的模型
class NewAgent(Agent):
    def __init__(self):
        # 加载 SAC 模型
        from train.sac.sac_agent import SACAgent
        self.sac = SACAgent(state_dim=64, action_dim=5, device='cpu')
        self.sac.load('checkpoints/sac_selfplay/xxx/sac_final.pt')
        self.sac.eval_mode()
        
    def decision(self, balls, my_targets, table):
        # 编码状态
        from train.environment.state_encoder import StateEncoder, ActionSpace
        encoder = StateEncoder()
        action_space = ActionSpace()
        
        state = encoder.encode(balls, my_targets, table, 0)
        action = self.sac.select_action(state, deterministic=True)
        return action_space.from_normalized(action)
```

