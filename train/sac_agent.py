"""
SAC Agent - Soft Actor-Critic 强化学习智能体
实现 SAC 算法的核心逻辑
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy

from networks import GaussianPolicy, TwinQNetwork
from config import SAC_CONFIG, DEVICE, denormalize_action


class SACAgent:
    """
    Soft Actor-Critic Agent
    
    实现特性：
    1. Twin Q 网络减少过估计
    2. 自动调节温度参数 alpha
    3. 支持确定性和随机策略
    """
    
    def __init__(self, state_dim=None, action_dim=None, config=None):
        """
        Args:
            state_dim: int, 状态维度
            action_dim: int, 动作维度  
            config: dict, 配置字典（可选，默认使用 SAC_CONFIG）
        """
        self.config = config or SAC_CONFIG
        self.state_dim = state_dim or self.config['state_dim']
        self.action_dim = action_dim or self.config['action_dim']
        self.device = DEVICE
        
        # 超参数
        self.gamma = self.config['gamma']
        self.tau = self.config['tau']
        self.lr_actor = self.config['lr_actor']
        self.lr_critic = self.config['lr_critic']
        self.lr_alpha = self.config['lr_alpha']
        self.auto_alpha = self.config['auto_alpha']
        self.target_entropy = self.config['target_entropy']
        
        # 初始化网络
        self.actor = GaussianPolicy(
            self.state_dim,
            self.action_dim,
            hidden_dims=self.config['hidden_dims']
        ).to(self.device)
        
        self.critic = TwinQNetwork(
            self.state_dim,
            self.action_dim,
            hidden_dims=self.config['hidden_dims']
        ).to(self.device)
        
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        # 温度参数 alpha（熵正则化系数）
        if self.auto_alpha:
            self.log_alpha = torch.tensor(
                np.log(self.config['alpha']), 
                requires_grad=True, 
                device=self.device
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_alpha)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(self.config['alpha'], device=self.device)
        
        # 训练统计
        self.update_count = 0
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.alpha_history = []
    
    def select_action(self, state, deterministic=False):
        """
        选择动作
        
        Args:
            state: numpy array of shape (state_dim,)
            deterministic: bool, 是否使用确定性策略
        
        Returns:
            action: numpy array of shape (action_dim,), in [-1, 1]
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, _ = self.actor.sample(state_tensor, deterministic=deterministic, return_log_prob=False)
        
        return action.cpu().numpy()[0]
    
    def decision(self, balls, my_type, table, deterministic=False):
        """
        决策接口（与 poolenv 兼容）
        
        Args:
            balls: dict, 球的状态
            my_type: str, 'solid' or 'stripe'
            table: table object
            deterministic: bool
        
        Returns:
            dict: {'V0': float, 'phi': float, 'theta': float, 'a': float, 'b': float}
        """
        # 需要 state_encoder，这里先占位
        # 实际使用时需要传入 encoder
        raise NotImplementedError("请使用 SACAgentWrapper 调用 decision")
    
    def update(self, replay_buffer, batch_size=256):
        """
        更新网络参数
        
        Args:
            replay_buffer: ReplayBuffer instance
            batch_size: int
        
        Returns:
            dict: 损失统计
        """
        # 采样 batch
        batch = replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)
        
        # ========== 更新 Critic ==========
        with torch.no_grad():
            # 采样下一个动作
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            # 计算目标 Q 值（使用 target network）
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * q_next
        
        # 当前 Q 值
        q1, q2 = self.critic(states, actions)
        
        # Critic 损失（MSE）
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        # 更新 Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # ========== 更新 Actor ==========
        # 采样当前动作
        actions_new, log_probs = self.actor.sample(states)
        
        # 计算 Q 值
        q1_new = self.critic.q1_forward(states, actions_new)
        
        # Actor 损失（最大化 Q - alpha * entropy）
        actor_loss = (self.alpha * log_probs - q1_new).mean()
        
        # 更新 Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # ========== 更新 Alpha（温度参数）==========
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # ========== 软更新目标网络 ==========
        self._soft_update(self.critic, self.critic_target)
        
        # 记录统计
        self.update_count += 1
        self.actor_loss_history.append(actor_loss.item())
        self.critic_loss_history.append(critic_loss.item())
        self.alpha_history.append(self.alpha.item())
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'alpha': self.alpha.item(),
            'q1_mean': q1.mean().item(),
            'q2_mean': q2.mean().item(),
        }
    
    def _soft_update(self, source, target):
        """
        软更新目标网络: target = tau * source + (1 - tau) * target
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_alpha else None,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.auto_alpha else None,
            'update_count': self.update_count,
        }, filepath)
    
    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        if self.auto_alpha and checkpoint['log_alpha'] is not None:
            self.log_alpha.data = checkpoint['log_alpha'].data
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            self.alpha = self.log_alpha.exp()
        
        self.update_count = checkpoint['update_count']
    
    def get_statistics(self):
        """获取训练统计信息"""
        if len(self.actor_loss_history) == 0:
            return {}
        
        # 最近 100 次更新的平均
        recent = min(100, len(self.actor_loss_history))
        
        return {
            'update_count': self.update_count,
            'actor_loss_mean': np.mean(self.actor_loss_history[-recent:]),
            'critic_loss_mean': np.mean(self.critic_loss_history[-recent:]),
            'alpha_mean': np.mean(self.alpha_history[-recent:]),
        }


class SACAgentWrapper:
    """
    SAC Agent 的包装器，提供与 poolenv 兼容的接口
    """
    
    def __init__(self, sac_agent, state_encoder):
        """
        Args:
            sac_agent: SACAgent instance
            state_encoder: StateEncoder instance
        """
        self.sac_agent = sac_agent
        self.state_encoder = state_encoder
        self.deterministic = False  # 默认随机策略
    
    def decision(self, balls, my_type, table):
        """
        决策接口（与 BasicAgent 兼容）
        
        Args:
            balls: dict
            my_type: str
            table: table object
        
        Returns:
            dict: {'V0': float, 'phi': float, 'theta': float, 'a': float, 'b': float}
        """
        # 编码状态
        # 需要构造 game_info
        from reward_shaper import get_ball_ids_by_type, count_remaining_balls
        
        my_ball_ids = get_ball_ids_by_type(my_type)
        opponent_type = 'stripe' if my_type == 'solid' else 'solid'
        opponent_ball_ids = get_ball_ids_by_type(opponent_type)
        
        my_remaining = count_remaining_balls(balls, my_ball_ids)
        enemy_remaining = count_remaining_balls(balls, opponent_ball_ids)
        
        game_info = {
            'turn': 0,  # 这个值在决策时不重要
            'my_balls_remaining': my_ball_ids[:my_remaining],
            'enemy_balls_remaining': opponent_ball_ids[:enemy_remaining]
        }
        
        state = self.state_encoder.encode(balls, my_type, game_info)
        
        # 选择动作（归一化的 [-1, 1]）
        action = self.sac_agent.select_action(state, deterministic=self.deterministic)
        
        # 转换为实际动作空间
        return denormalize_action(action)
    
    def set_deterministic(self, deterministic):
        """设置是否使用确定性策略"""
        self.deterministic = deterministic


# ==================== 测试代码 ====================
if __name__ == '__main__':
    """测试 SAC Agent"""
    
    print("=" * 50)
    print("测试 SAC Agent")
    
    # 初始化
    from replay_buffer import ReplayBuffer
    
    agent = SACAgent()
    buffer = ReplayBuffer(capacity=10000)
    
    print(f"Device: {agent.device}")
    print(f"State dim: {agent.state_dim}")
    print(f"Action dim: {agent.action_dim}")
    
    # 模拟填充 buffer
    print("\n1. 填充 replay buffer")
    for i in range(1000):
        state = np.random.randn(agent.state_dim)
        action = np.random.uniform(-1, 1, agent.action_dim)
        reward = np.random.randn()
        next_state = np.random.randn(agent.state_dim)
        done = False
        
        buffer.push(state, action, reward, next_state, done)
    
    print(f"Buffer size: {len(buffer)}")
    
    # 测试选择动作
    print("\n2. 测试动作选择")
    test_state = np.random.randn(agent.state_dim)
    
    random_action = agent.select_action(test_state, deterministic=False)
    print(f"Random action: {random_action}")
    print(f"Action range: [{random_action.min():.3f}, {random_action.max():.3f}]")
    
    det_action = agent.select_action(test_state, deterministic=True)
    print(f"Deterministic action: {det_action}")
    
    # 测试更新
    print("\n3. 测试网络更新")
    for i in range(10):
        losses = agent.update(buffer, batch_size=128)
        if i == 0 or i == 9:
            print(f"Update {i+1}: {losses}")
    
    # 测试统计
    print("\n4. 训练统计")
    stats = agent.get_statistics()
    print(stats)
    
    # 测试保存和加载
    print("\n5. 测试保存/加载")
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'test_model.pth')
        agent.save(save_path)
        print(f"Model saved to {save_path}")
        
        # 创建新 agent 并加载
        new_agent = SACAgent()
        new_agent.load(save_path)
        print(f"Model loaded, update count: {new_agent.update_count}")
    
    print("\n✅ SAC Agent 测试通过！")
