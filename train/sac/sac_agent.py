"""
sac_agent.py - SAC 算法实现

Soft Actor-Critic 算法：
- 自动温度调节
- 双Q网络
- 软策略更新
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional
import os

from .networks import GaussianActor, TwinQCritic, soft_update, hard_update
from .replay_buffer import ReplayBuffer


class SACAgent:
    """
    Soft Actor-Critic 智能体
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: list = [512, 512, 256],
                 lr_actor: float = 3e-4,
                 lr_critic: float = 3e-4,
                 lr_alpha: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 initial_alpha: float = 0.2,
                 target_entropy: Optional[float] = None,
                 device: str = 'cuda'):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dims: 隐藏层维度
            lr_actor: Actor 学习率
            lr_critic: Critic 学习率
            lr_alpha: 温度参数学习率
            gamma: 折扣因子
            tau: 目标网络软更新系数
            initial_alpha: 初始熵系数
            target_entropy: 目标熵（None表示自动计算）
            device: 设备
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # 创建网络
        self.actor = GaussianActor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(device)
        
        self.critic = TwinQCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(device)
        
        self.critic_target = TwinQCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(device)
        
        # 初始化目标网络
        hard_update(self.critic_target, self.critic)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 自动温度调节
        self.log_alpha = torch.tensor(
            np.log(initial_alpha), 
            requires_grad=True, 
            device=device,
            dtype=torch.float32
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
        
        # 目标熵
        if target_entropy is None:
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = target_entropy
        
        # 训练统计
        self.training_step = 0
    
    @property
    def alpha(self) -> torch.Tensor:
        """当前温度参数"""
        return self.log_alpha.exp()
    
    def select_action(self, 
                      state: np.ndarray, 
                      deterministic: bool = False) -> np.ndarray:
        """
        选择动作
        
        Args:
            state: 状态
            deterministic: 是否确定性策略
        
        Returns:
            np.ndarray: 动作 [-1, 1]
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor.get_action(state_tensor, deterministic)
            return action.cpu().numpy().flatten()
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        更新网络
        
        Args:
            batch: 经验批次 {'states', 'actions', 'rewards', 'next_states', 'dones'}
        
        Returns:
            Dict: 训练统计信息
        """
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        # ===== 更新 Critic =====
        with torch.no_grad():
            # 采样下一个动作
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            # 计算目标Q值
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 当前Q值
        current_q1, current_q2 = self.critic(states, actions)
        
        # Critic 损失
        critic_loss = nn.functional.mse_loss(current_q1, target_q) + \
                      nn.functional.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # ===== 更新 Actor =====
        new_actions, log_probs = self.actor.sample(states)
        q1, q2 = self.critic(states, new_actions)
        q = torch.min(q1, q2)
        
        # Actor 损失
        actor_loss = (self.alpha.detach() * log_probs - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # ===== 更新温度参数 =====
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # ===== 软更新目标网络 =====
        soft_update(self.critic_target, self.critic, self.tau)
        
        self.training_step += 1
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item(),
            'q_mean': q.mean().item(),
            'log_prob_mean': log_probs.mean().item()
        }
    
    def save(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'training_step': self.training_step
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        self.training_step = checkpoint['training_step']
        print(f"Model loaded from {path}")
    
    def get_policy_state_dict(self) -> dict:
        """获取策略网络状态（用于 Worker 同步）"""
        return self.actor.state_dict()
    
    def set_policy_state_dict(self, state_dict: dict):
        """设置策略网络状态（从 Learner 同步）"""
        self.actor.load_state_dict(state_dict)
    
    def train_mode(self):
        """设置为训练模式"""
        self.actor.train()
        self.critic.train()
    
    def eval_mode(self):
        """设置为评估模式"""
        self.actor.eval()
        self.critic.eval()


class SACTrainer:
    """
    SAC 训练器
    
    管理训练循环和日志
    """
    
    def __init__(self,
                 agent: SACAgent,
                 replay_buffer: ReplayBuffer,
                 batch_size: int = 2048,
                 updates_per_batch: int = 8,
                 min_buffer_size: int = 10000):
        """
        Args:
            agent: SAC 智能体
            replay_buffer: 经验回放
            batch_size: 批量大小
            updates_per_batch: 每批经验的更新次数
            min_buffer_size: 开始训练前的最小经验数
        """
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.updates_per_batch = updates_per_batch
        self.min_buffer_size = min_buffer_size
        
        # 统计信息
        self.total_updates = 0
        self.stats_history = []
    
    def can_train(self) -> bool:
        """检查是否可以开始训练"""
        return self.replay_buffer.is_ready(self.min_buffer_size)
    
    def train_step(self) -> Dict[str, float]:
        """
        执行一次训练步骤
        
        Returns:
            Dict: 平均训练统计
        """
        if not self.can_train():
            return {}
        
        stats_list = []
        
        for _ in range(self.updates_per_batch):
            batch = self.replay_buffer.sample(self.batch_size)
            stats = self.agent.update(batch)
            stats_list.append(stats)
            self.total_updates += 1
        
        # 计算平均统计
        avg_stats = {}
        for key in stats_list[0].keys():
            avg_stats[key] = np.mean([s[key] for s in stats_list])
        
        self.stats_history.append(avg_stats)
        return avg_stats
    
    def get_recent_stats(self, n: int = 100) -> Dict[str, float]:
        """获取最近n次更新的平均统计"""
        if len(self.stats_history) == 0:
            return {}
        
        recent = self.stats_history[-n:]
        avg_stats = {}
        for key in recent[0].keys():
            avg_stats[key] = np.mean([s[key] for s in recent])
        return avg_stats

