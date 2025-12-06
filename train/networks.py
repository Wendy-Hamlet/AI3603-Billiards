"""
Neural Networks for SAC
包含 Actor 网络和 Twin Critic 网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class MLP(nn.Module):
    """多层感知机基础模块"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU, 
                 use_layer_norm=True, output_activation=None):
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class GaussianPolicy(nn.Module):
    """
    高斯策略网络 (Actor)
    输出动作的均值和标准差，用于采样动作
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256, 256],
                 log_std_min=-20, log_std_max=2):
        super(GaussianPolicy, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_dim = action_dim
        
        # 共享特征提取层
        self.feature_net = MLP(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=hidden_dims[-1],
            use_layer_norm=True
        )
        
        # 均值和标准差头
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state):
        """
        前向传播
        
        Args:
            state: torch.Tensor of shape (batch_size, state_dim)
        
        Returns:
            mean: torch.Tensor of shape (batch_size, action_dim)
            log_std: torch.Tensor of shape (batch_size, action_dim)
        """
        features = self.feature_net(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        
        # 限制 log_std 的范围
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state, deterministic=False, return_log_prob=True):
        """
        采样动作
        
        Args:
            state: torch.Tensor
            deterministic: bool, 是否使用确定性策略（测试时用）
            return_log_prob: bool, 是否返回 log 概率
        
        Returns:
            action: torch.Tensor of shape (batch_size, action_dim), in [-1, 1]
            log_prob: torch.Tensor of shape (batch_size, 1) or None
        """
        mean, log_std = self.forward(state)
        
        if deterministic:
            # 确定性策略：直接使用均值
            action = torch.tanh(mean)
            log_prob = None if not return_log_prob else torch.zeros(mean.shape[0], 1, device=state.device)
        else:
            # 随机策略：从高斯分布采样
            std = log_std.exp()
            normal = Normal(mean, std)
            
            # Reparameterization trick
            x_t = normal.rsample()
            
            # Squash to [-1, 1]
            action = torch.tanh(x_t)
            
            # 计算 log 概率（需要考虑 tanh 的变换）
            if return_log_prob:
                log_prob = normal.log_prob(x_t)
                # 修正 tanh 变换的 log 概率
                log_prob -= torch.log(1 - action.pow(2) + 1e-6)
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                log_prob = None
        
        return action, log_prob
    
    def get_action(self, state, deterministic=False):
        """
        便捷方法：获取单个动作（用于推理）
        
        Args:
            state: numpy array of shape (state_dim,)
            deterministic: bool
        
        Returns:
            action: numpy array of shape (action_dim,), in [-1, 1]
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
            action, _ = self.sample(state_tensor, deterministic=deterministic, return_log_prob=False)
            return action.cpu().numpy()[0]


class QNetwork(nn.Module):
    """
    Q 网络 (Critic)
    输入状态和动作，输出 Q 值
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256, 256]):
        super(QNetwork, self).__init__()
        
        self.q_net = MLP(
            input_dim=state_dim + action_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            use_layer_norm=True
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state, action):
        """
        前向传播
        
        Args:
            state: torch.Tensor of shape (batch_size, state_dim)
            action: torch.Tensor of shape (batch_size, action_dim)
        
        Returns:
            q_value: torch.Tensor of shape (batch_size, 1)
        """
        x = torch.cat([state, action], dim=1)
        return self.q_net(x)


class TwinQNetwork(nn.Module):
    """
    Twin Q 网络（用于减少过估计）
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256, 256]):
        super(TwinQNetwork, self).__init__()
        
        self.q1 = QNetwork(state_dim, action_dim, hidden_dims)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dims)
    
    def forward(self, state, action):
        """
        返回两个 Q 值
        
        Returns:
            q1: torch.Tensor of shape (batch_size, 1)
            q2: torch.Tensor of shape (batch_size, 1)
        """
        q1 = self.q1(state, action)
        q2 = self.q2(state, action)
        return q1, q2
    
    def q1_forward(self, state, action):
        """只使用 Q1（用于策略更新）"""
        return self.q1(state, action)


# ==================== 测试代码 ====================
if __name__ == '__main__':
    """测试神经网络"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 参数
    state_dim = 53
    action_dim = 5
    batch_size = 32
    
    # 测试 Actor
    print("\n" + "=" * 50)
    print("测试 Gaussian Policy (Actor)")
    actor = GaussianPolicy(state_dim, action_dim).to(device)
    
    # 随机状态
    state = torch.randn(batch_size, state_dim).to(device)
    
    # 采样动作
    action, log_prob = actor.sample(state, deterministic=False)
    print(f"Action shape: {action.shape}")
    print(f"Action range: [{action.min().item():.3f}, {action.max().item():.3f}]")
    print(f"Log prob shape: {log_prob.shape}")
    
    # 确定性动作
    det_action, _ = actor.sample(state, deterministic=True)
    print(f"Deterministic action shape: {det_action.shape}")
    
    # 单个动作
    single_state = state[0].cpu().numpy()
    single_action = actor.get_action(single_state, deterministic=False)
    print(f"Single action shape: {single_action.shape}")
    
    # 测试 Critic
    print("\n" + "=" * 50)
    print("测试 Twin Q Network (Critic)")
    critic = TwinQNetwork(state_dim, action_dim).to(device)
    
    q1, q2 = critic(state, action)
    print(f"Q1 shape: {q1.shape}")
    print(f"Q2 shape: {q2.shape}")
    print(f"Q1 range: [{q1.min().item():.3f}, {q1.max().item():.3f}]")
    print(f"Q2 range: [{q2.min().item():.3f}, {q2.max().item():.3f}]")
    
    # 参数数量
    print("\n" + "=" * 50)
    print("网络参数统计")
    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    print(f"Actor parameters: {actor_params:,}")
    print(f"Critic parameters: {critic_params:,}")
    print(f"Total parameters: {actor_params + critic_params:,}")
    
    print("\n✅ 神经网络测试通过！")
