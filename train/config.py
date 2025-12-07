"""
SAC Training Configuration
包含超参数、训练阶段配置和网络参数
"""

import torch

# ==================== 设备配置 ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_MIXED_PRECISION = True  # A100 支持混合精度训练

# ==================== SAC 超参数 ====================
SAC_CONFIG = {
    # 网络结构
    'state_dim': 53,          # 状态维度（分组语义编码）
    'action_dim': 5,          # 动作维度 (V0, phi, theta, a, b)
    'hidden_dims': [256, 256, 256],  # 隐藏层维度
    
    # 学习率
    'lr_actor': 3e-4,
    'lr_critic': 3e-4,
    'lr_alpha': 3e-4,         # 温度参数学习率
    
    # 折扣和软更新
    'gamma': 0.99,            # 折扣因子
    'tau': 0.005,             # 目标网络软更新系数
    
    # 熵正则化
    'alpha': 0.2,             # 初始温度参数
    'auto_alpha': True,       # 是否自动调节 alpha
    'target_entropy': -5.0,   # 目标熵（-action_dim）
    
    # 训练参数
    'batch_size': 512,        # A100 可用更大 batch
    'gradient_steps': 1,      # 每步环境交互后的梯度更新次数
    'warmup_steps': 5000,     # 随机策略预热步数（从10000减少，加速启动）
    
    # 并行训练
    'num_parallel_envs': 16,  # 并行环境数量（同时进行的对局数）
    'update_frequency': 16,   # 每收集N个episodes后更新一次网络
    
    # 经验回放
    'replay_buffer_size': 500000,
    
    # 网络初始化
    'init_weight_uniform': True,
    'init_weight_scale': 1e-3,
}

# ==================== 动作空间配置 ====================
ACTION_SPACE = {
    'V0': {'min': 0.5, 'max': 2.5, 'scale': 1.0, 'bias': 1.5},      # 速度
    'phi': {'min': -180, 'max': 180, 'scale': 180, 'bias': 0},      # 水平角度
    'theta': {'min': 0, 'max': 20, 'scale': 10, 'bias': 10},        # 仰角
    'a': {'min': -0.028575, 'max': 0.028575, 'scale': 0.028575, 'bias': 0},  # 球半径
    'b': {'min': -0.028575, 'max': 0.028575, 'scale': 0.028575, 'bias': 0},
}

# 动作空间转换函数
def denormalize_action(action):
    """
    将神经网络输出的 [-1, 1] 动作转换为实际动作空间
    
    Args:
        action: numpy array of shape (5,), values in [-1, 1]
    
    Returns:
        dict: {'V0': float, 'phi': float, 'theta': float, 'a': float, 'b': float}
    """
    return {
        'V0': float(ACTION_SPACE['V0']['bias'] + ACTION_SPACE['V0']['scale'] * action[0]),
        'phi': float(ACTION_SPACE['phi']['bias'] + ACTION_SPACE['phi']['scale'] * action[1]),
        'theta': float(ACTION_SPACE['theta']['bias'] + ACTION_SPACE['theta']['scale'] * action[2]),
        'a': float(ACTION_SPACE['a']['bias'] + ACTION_SPACE['a']['scale'] * action[3]),
        'b': float(ACTION_SPACE['b']['bias'] + ACTION_SPACE['b']['scale'] * action[4]),
    }

# ==================== 奖励函数配置 ====================
REWARD_CONFIG = {
    # 进球奖励系数
    'C1': 100.0,              # 己方进球基础价值
    'C2': 100.0,              # 对方进球基础价值
    'C3': 10.0,               # 球权维持价值
    
    # 防守奖励
    'defense_foul': 5.0,      # 对手犯规的防守奖励
    'defense_no_pocket': 2.0, # 对手未进球的防守奖励
    
    # 犯规惩罚
    'foul_white_ball': -20.0,
    'foul_first_hit': -15.0,
    'foul_no_rail': -10.0,
    'foul_no_hit': -25.0,
    
    # 终局奖励
    'win_reward': 1000.0,
    'loss_reward': -1000.0,
    
    # 失去球权惩罚（未进己方球）
    'lose_turn': -5.0,
}

# ==================== 训练阶段配置 ====================
TRAINING_STAGES = {
    'stage1': {
        'name': 'Foundation',
        'episodes': 15000,
        'opponents': {'self': 0.7, 'basic': 0.3},  # 70%自对弈，30%BasicAgent（减少CPU负载）
        'target_metrics': {
            'basic_winrate': 0.70
        },
        'description': '自对弈为主，学习基础击球和避免犯规'
    },
    'stage2': {
        'name': 'Intermediate',
        'episodes': 25000,
        'opponents': {'self': 0.6, 'basic': 0.2, 'physics': 0.2},  # 提高自对弈比例
        'target_metrics': {
            'physics_winrate': 0.40
        },
        'description': '自对弈为主，逐步引入PhysicsAgent'
    },
    'stage3': {
        'name': 'Advanced',
        'episodes': 30000,
        'opponents': {'self': 0.7, 'mcts': 0.3},  # 高比例自对弈，减少MCTS负载
        'target_metrics': {
            'mcts_winrate': 0.30
        },
        'description': '自对弈为主，挑战MCTS顶级对手'
    }
}

# ==================== 评估配置 ====================
EVAL_CONFIG = {
    'eval_frequency': 1000,    # 每 N episodes 评估一次
    'eval_games': 40,          # 每次评估的对局数（保证统计显著性）
    'checkpoint_frequency': 5000,  # 每 N episodes 保存 checkpoint
    'log_frequency': 1,        # 每 N episodes 记录日志（1=每轮都显示）
    'detailed_log_frequency': 100,  # 每 N episodes 显示详细统计
}

# ==================== Checkpoint 管理 ====================
CHECKPOINT_CONFIG = {
    'save_dir': './train/checkpoints/sac',
    'max_keep': 20,            # 最多保留的 checkpoint 数量
    'save_best': True,         # 是否额外保存最佳模型
}

# ==================== 日志配置 ====================
LOG_CONFIG = {
    'log_dir': './train/logs/sac',
    'tensorboard': True,       # 是否使用 TensorBoard
    'console_log_level': 'INFO',
    'metrics_to_log': [
        'episode',
        'total_reward',
        'episode_length',
        'winrate_vs_basic',
        'winrate_vs_physics',
        'winrate_vs_mcts',
        'avg_q_value',
        'policy_loss',
        'critic_loss',
        'alpha',
        'foul_rate',
    ]
}

# ==================== 物理环境配置 ====================
TABLE_CONFIG = {
    'width': 0.9906,           # 台面宽度（米）
    'length': 1.9812,          # 台面长度（米）
    'ball_radius': 0.028575,   # 球半径（米）
}

# ==================== 调试配置 ====================
DEBUG_CONFIG = {
    'verbose': False,          # 是否打印详细信息
    'save_replay': False,      # 是否保存重放
    'render': False,           # 是否可视化
}

# ==================== 快速测试配置 ====================
def get_quick_test_config():
    """返回用于快速测试的配置（减少 episodes）"""
    test_config = TRAINING_STAGES.copy()
    test_config['stage1']['episodes'] = 100
    test_config['stage2']['episodes'] = 200
    test_config['stage3']['episodes'] = 300
    return test_config
