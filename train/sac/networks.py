"""
networks.py - SAC 神经网络定义

实现 Actor 和 Critic 网络：
- Actor: 输出动作的均值和标准差（高斯策略）
- Critic: 双Q网络（减少过估计）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple, List


def build_mlp(input_dim: int, 
              hidden_dims: List[int], 
              output_dim: int,
              activation: str = 'relu',
              output_activation: str = None) -> nn.Sequential:
    """
    构建多层感知机
    
    Args:
        input_dim: 输入维度
        hidden_dims: 隐藏层维度列表
        output_dim: 输出维度
        activation: 激活函数 ('relu', 'tanh', 'leaky_relu')
        output_activation: 输出层激活函数
    
    Returns:
        nn.Sequential: MLP网络
    """
    activation_fn = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'leaky_relu': nn.LeakyReLU,
        'elu': nn.ELU
    }
    
    layers = []
    prev_dim = input_dim
    
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation_fn[activation]())
        prev_dim = hidden_dim
    
    layers.append(nn.Linear(prev_dim, output_dim))
    
    if output_activation is not None:
        layers.append(activation_fn[output_activation]())
    
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    """
    高斯策略 Actor 网络
    
    输出动作的均值和对数标准差，用于采样连续动作
    """
    
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [512, 512, 256],
                 activation: str = 'relu'):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dims: 隐藏层维度
            activation: 激活函数
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 共享特征提取层
        self.backbone = build_mlp(
            input_dim=state_dim,
            hidden_dims=hidden_dims[:-1],
            output_dim=hidden_dims[-1],
            activation=activation
        )
        
        # 均值和对数标准差输出头
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        # 输出层使用更小的初始化
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 状态张量 [batch, state_dim]
        
        Returns:
            Tuple[Tensor, Tensor]: (均值, 对数标准差)
        """
        features = self.backbone(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样动作并计算对数概率
        
        使用重参数化技巧和 tanh 压缩
        
        Args:
            state: 状态张量
        
        Returns:
            Tuple[Tensor, Tensor]: (动作 [-1,1], 对数概率)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # 重参数化采样
        normal = Normal(mean, std)
        x_t = normal.rsample()  # 重参数化
        
        # tanh 压缩到 [-1, 1]
        action = torch.tanh(x_t)
        
        # 计算对数概率（考虑 tanh 变换的雅可比行列式）
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        获取动作（用于推理）
        
        Args:
            state: 状态张量
            deterministic: 是否使用确定性策略（均值）
        
        Returns:
            Tensor: 动作 [-1, 1]
        """
        mean, log_std = self.forward(state)
        
        if deterministic:
            action = torch.tanh(mean)
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.sample()
            action = torch.tanh(x_t)
        
        return action


class TwinQCritic(nn.Module):
    """
    双Q网络 Critic
    
    使用两个独立的Q网络减少过估计
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [512, 512, 256],
                 activation: str = 'relu'):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dims: 隐藏层维度
            activation: 激活函数
        """
        super().__init__()
        
        input_dim = state_dim + action_dim
        
        # Q1 网络
        self.q1 = build_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=activation
        )
        
        # Q2 网络
        self.q2 = build_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=activation
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 状态张量 [batch, state_dim]
            action: 动作张量 [batch, action_dim]
        
        Returns:
            Tuple[Tensor, Tensor]: (Q1值, Q2值)
        """
        x = torch.cat([state, action], dim=-1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2
    
    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """只计算 Q1（用于策略更新）"""
        x = torch.cat([state, action], dim=-1)
        return self.q1(x)


class ValueNetwork(nn.Module):
    """
    价值网络 V(s)
    
    可选组件，用于某些 SAC 变体
    """
    
    def __init__(self,
                 state_dim: int,
                 hidden_dims: List[int] = [512, 512, 256],
                 activation: str = 'relu'):
        super().__init__()
        
        self.net = build_mlp(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=activation
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """
    软更新目标网络
    
    target = tau * source + (1 - tau) * target
    
    Args:
        target: 目标网络
        source: 源网络
        tau: 更新系数
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1 - tau) * target_param.data
        )


def hard_update(target: nn.Module, source: nn.Module):
    """
    硬更新目标网络
    
    target = source
    """
    target.load_state_dict(source.state_dict())

