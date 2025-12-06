# SAC (Soft Actor-Critic) 训练系统

## 概述

本目录包含使用 Soft Actor-Critic (SAC) 强化学习算法训练台球 AI 的完整实现。

## 系统架构

### 核心组件

1. **config.py** - 配置文件
   - SAC 超参数
   - 训练阶段配置（渐进式课程学习）
   - 奖励函数参数
   - 动作空间定义

2. **state_encoder.py** - 状态编码器
   - 实现 Option B：分组语义编码
   - 状态维度：53 维（母球3 + 我方球21 + 对方球21 + 8号球3 + 全局信息5）

3. **reward_shaper.py** - 奖励函数
   - 比例价值奖励：进球价值 = (进球数/剩余球数) × 常数
   - 防守奖励：对手失误时的延迟奖励
   - 犯规惩罚：固定值惩罚

4. **networks.py** - 神经网络
   - GaussianPolicy：Actor 网络（高斯策略）
   - TwinQNetwork：Critic 网络（Twin Q，减少过估计）

5. **replay_buffer.py** - 经验回放池
   - 支持防守奖励的延迟追溯
   - EpisodeTracker：追踪 episode 中的 transitions

6. **sac_agent.py** - SAC Agent
   - SAC 算法核心实现
   - 自动调节温度参数 alpha
   - SACAgentWrapper：提供与 poolenv 兼容的接口

7. **opponent_pool.py** - 对手池管理
   - 管理 BasicAgent、PhysicsAgent、MCTSAgent
   - Self-play checkpoint 池管理

8. **train_sac.py** - 训练脚本
   - 渐进式课程学习训练流程
   - 自动评估和 checkpoint 保存

9. **eval_sac.py** - 评估脚本
   - 评估训练好的模型
   - 对阵不同对手的胜率统计

## 训练策略

采用**渐进式课程学习**（Curriculum Learning）策略：

### Stage 1: Foundation (15k episodes, ~2-3小时)
- **对手**: 100% BasicAgent
- **目标**: 学习基础击球，避免犯规
- **退出条件**: 对 BasicAgent 胜率 > 70%

### Stage 2: Intermediate (25k episodes, ~4-5小时)
- **对手**: 60% BasicAgent + 30% PhysicsAgent + 10% Self-play
- **目标**: 学习对抗稳定对手
- **退出条件**: 对 PhysicsAgent 胜率 > 40%

### Stage 3: Advanced (30k episodes, ~5-6小时)
- **对手**: 20% BasicAgent + 30% PhysicsAgent + 20% MCTSAgent + 30% Self-play
- **目标**: 挑战顶级对手，自我进化
- **退出条件**: 对 MCTSAgent 胜率 > 30%

**总训练时间**: 约 11-14 小时（A100 GPU）

## 奖励函数设计

### 即时奖励

```python
即时奖励 = 己方进球价值 - 对方进球价值 + 球权价值 + 犯规惩罚

其中：
- 己方进球价值 = (进球数 / 剩余球数) × 100
- 对方进球价值 = (进球数 / 剩余球数) × 100
- 球权价值 = +10（保持）或 -5（失去）
- 犯规惩罚 = -20（白球）, -15（首次接触）, -10（无碰库）, -25（完全未击中）
```

### 防守奖励（延迟奖励）

当对手在下一回合失误时，追溯给我方上一杆增加奖励：
- 对手犯规：+5
- 对手未进球：+2

### 终局奖励

- 胜利：+1000
- 失败：-1000

## 使用方法

### 1. 训练模型

```bash
# 标准训练
python train_sac.py

# 从 checkpoint 恢复训练
python train_sac.py --resume checkpoints/sac/checkpoint_ep10000.pth

# 快速测试模式（少量 episodes）
python train_sac.py --test
```

### 2. 评估模型

```bash
# 评估所有对手
python eval_sac.py --checkpoint checkpoints/sac/final_model.pth

# 只评估某个对手
python eval_sac.py --checkpoint checkpoints/sac/final_model.pth --opponent basic

# 增加对局数
python eval_sac.py --checkpoint checkpoints/sac/final_model.pth --games 100

# 显示详细信息
python eval_sac.py --checkpoint checkpoints/sac/final_model.pth --verbose
```

### 3. 单元测试

每个模块都包含测试代码，可以单独运行：

```bash
# 测试状态编码器
python state_encoder.py

# 测试奖励函数
python reward_shaper.py

# 测试神经网络
python networks.py

# 测试 Replay Buffer
python replay_buffer.py

# 测试 SAC Agent
python sac_agent.py

# 测试对手池
python opponent_pool.py
```

## 配置说明

### 修改超参数

编辑 `config.py` 中的 `SAC_CONFIG`：

```python
SAC_CONFIG = {
    'batch_size': 512,        # 批大小（A100 可用更大值）
    'lr_actor': 3e-4,         # Actor 学习率
    'lr_critic': 3e-4,        # Critic 学习率
    'gamma': 0.99,            # 折扣因子
    'tau': 0.005,             # 软更新系数
    'alpha': 0.2,             # 温度参数（自动调节）
    # ...
}
```

### 修改奖励权重

编辑 `config.py` 中的 `REWARD_CONFIG`：

```python
REWARD_CONFIG = {
    'C1': 100.0,              # 己方进球基础价值
    'C2': 100.0,              # 对方进球基础价值
    'C3': 10.0,               # 球权维持价值
    'defense_foul': 5.0,      # 防守奖励
    # ...
}
```

### 修改训练阶段

编辑 `config.py` 中的 `TRAINING_STAGES`：

```python
TRAINING_STAGES = {
    'stage1': {
        'name': 'Foundation',
        'episodes': 15000,
        'opponents': {'basic': 1.0},
        'target_metrics': {'basic_winrate': 0.70}
    },
    # ...
}
```

## 目录结构

```
train/
├── config.py              # 配置文件
├── state_encoder.py       # 状态编码器
├── reward_shaper.py       # 奖励函数
├── networks.py           # 神经网络
├── replay_buffer.py      # 经验回放池
├── sac_agent.py          # SAC Agent
├── opponent_pool.py      # 对手池管理
├── train_sac.py          # 训练脚本
├── eval_sac.py           # 评估脚本
├── README_SAC.md         # 本文档
├── checkpoints/          # 模型保存目录
│   └── sac/
└── logs/                 # 日志目录
    └── sac/
```

## 性能优化

### GPU 加速

代码自动检测并使用 GPU（CUDA）：

```python
# 在 config.py 中
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### 混合精度训练（A100 专用）

```python
USE_MIXED_PRECISION = True  # 在 config.py 中启用
```

### 批量大小调整

A100 GPU 可以使用更大的 batch size：

```python
'batch_size': 512,  # 或尝试 1024
```

## 预期结果

根据设计和测试：

- **对 BasicAgent**: 胜率 > 70% (Stage 1 结束)
- **对 PhysicsAgent**: 胜率 > 40% (Stage 2 结束)
- **对 MCTSAgent**: 胜率 > 30% (Stage 3 结束)

最终模型预期能够：
1. 稳定避免犯规
2. 学会基本的进攻策略
3. 理解防守的价值
4. 在不同局面下做出合理决策

## 常见问题

### 1. MCTS Agent 不可用

如果 MCTS Agent 未实现或不可用，系统会自动使用 PhysicsAgent 代替。

### 2. 训练速度慢

- 确认使用了 GPU
- 考虑减少评估频率（`eval_frequency`）
- 减少 checkpoint 保存频率（`checkpoint_frequency`）

### 3. 内存不足

- 减少 `replay_buffer_size`
- 减少 `batch_size`
- 减少网络层数或隐藏层维度

### 4. 训练不稳定

- 检查奖励函数是否合理
- 降低学习率
- 增加 warmup steps

## 技术细节

### 状态编码

- **输入**: 台球游戏状态（球的位置、类型、剩余信息等）
- **输出**: 53 维向量
- **特点**: 按距离排序，提供优先级提示

### 动作空间

- **维度**: 5 维连续动作
- **范围**: [-1, 1]（经过归一化）
- **映射**: V0 [0.5, 2.5], phi [-180°, 180°], theta [0°, 20°], a/b [-R, R]

### SAC 算法

- **Twin Q**: 减少 Q 值过估计
- **自动温度调节**: 自适应探索-利用平衡
- **软更新**: 稳定目标网络

## 引用

如果使用本代码，请引用：

- Haarnoja et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor" (ICML 2018)
- 本项目课程：AI3603 人工智能理论及应用

## 作者

- 实现者：GitHub Copilot AI Assistant
- 课程：AI3603 - 上海交通大学
- 时间：2025年12月

## 许可

本代码仅供学习和研究使用。
