"""
replay_buffer.py - 分类优先级经验回放缓冲区

实现基于奖励的分类存储和优先级采样：
- 经验分类：按奖励范围分为多个类别，每类有容量上限
- 淘汰策略：每类内FIFO（先进先出），保证各类平衡
- 采样策略：按优先级采样（高奖励经验更常被采样）
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
from collections import deque
import threading


class ReplayBuffer:
    """
    分类优先级经验回放缓冲区
    
    特性：
    - 经验按奖励分为多个类别，每类有容量上限
    - 同类内FIFO淘汰（先进先出）
    - 采样时按优先级加权
    """
    
    # 经验类别定义（按奖励范围，适配紧凑奖励 [-100, +100]）
    CATEGORY_BOUNDS = [
        ('win', 80, float('inf')),           # 胜利 (+100)
        ('lose', float('-inf'), -60),        # 失败/平局消极 (-80 到 -100)
        ('pocket', 15, 80),                  # 进球 (+8 到 +30)
        ('foul', -60, -5),                   # 犯规 (-5 到 -15)
        ('neutral', -5, 15),                 # 中性（走位奖励等）
    ]
    
    # 每类容量占比（总和应为1.0）
    CATEGORY_RATIOS = {
        'win': 0.15,        # 15% - 胜利经验（最重要）
        'lose': 0.15,       # 15% - 失败经验（也重要）
        'pocket': 0.25,     # 25% - 进球经验
        'foul': 0.15,       # 15% - 犯规经验
        'neutral': 0.30,    # 30% - 中性经验
    }
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 buffer_size: int = 500000,
                 device: str = 'cpu',
                 priority_alpha: float = 0.6):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            buffer_size: 缓冲区总大小
            device: 设备
            priority_alpha: 优先级采样系数 [0,1]
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.device = device
        self.priority_alpha = priority_alpha
        
        # 预分配内存
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)
        
        # 优先级数组
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        
        # 时间戳数组（用于FIFO）
        self.timestamps = np.zeros(buffer_size, dtype=np.int64)
        self.global_timestamp = 0
        
        # 分类队列：每类维护一个FIFO队列（存储索引）
        self.category_queues: Dict[str, deque] = {}
        self.category_limits: Dict[str, int] = {}
        for name, _, _ in self.CATEGORY_BOUNDS:
            ratio = self.CATEGORY_RATIOS.get(name, 0.1)
            self.category_limits[name] = int(buffer_size * ratio)
            self.category_queues[name] = deque()
        
        # 空闲位置
        self.free_indices = list(range(buffer_size))
        self.used_indices = set()
        
        self.size = 0
        self.lock = threading.Lock()
        self.min_priority = 1.0
    
    def _get_category(self, reward: float) -> str:
        """根据奖励确定经验类别"""
        for name, low, high in self.CATEGORY_BOUNDS:
            if low <= reward < high:
                return name
        return 'low'  # 默认归入低奖励类
    
    def _compute_priority(self, reward: float) -> float:
        """计算经验的优先级"""
        return abs(reward) + self.min_priority
    
    def _get_free_index(self, category: str) -> int:
        """
        获取一个空闲索引
        
        如果buffer满了，从该类别的队列中FIFO淘汰最旧的
        """
        if self.free_indices:
            return self.free_indices.pop()
        
        # Buffer满，需要从某个类别淘汰
        queue = self.category_queues[category]
        
        if len(queue) > 0:
            # 从本类别淘汰最旧的
            old_idx = queue.popleft()
            self.used_indices.discard(old_idx)
            return old_idx
        
        # 本类别为空，从最大的类别淘汰
        max_category = max(self.category_queues.keys(), 
                          key=lambda c: len(self.category_queues[c]))
        if len(self.category_queues[max_category]) > 0:
            old_idx = self.category_queues[max_category].popleft()
            self.used_indices.discard(old_idx)
            return old_idx
        
        # 不应该到达这里
        return 0
    
    def _enforce_category_limit(self, category: str):
        """确保类别不超过容量限制"""
        queue = self.category_queues[category]
        limit = self.category_limits[category]
        
        while len(queue) > limit:
            # FIFO淘汰最旧的
            old_idx = queue.popleft()
            self.used_indices.discard(old_idx)
            self.free_indices.append(old_idx)
    
    def add(self,
            state: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_state: np.ndarray,
            done: bool):
        """添加单条经验"""
        category = self._get_category(reward)
        priority = self._compute_priority(reward)
        
        with self.lock:
            idx = self._get_free_index(category)
            
            self.states[idx] = state
            self.actions[idx] = action
            self.rewards[idx] = reward
            self.next_states[idx] = next_state
            self.dones[idx] = float(done)
            self.priorities[idx] = priority
            self.timestamps[idx] = self.global_timestamp
            self.global_timestamp += 1
            
            self.used_indices.add(idx)
            self.category_queues[category].append(idx)
            
            # 确保不超过类别限制
            self._enforce_category_limit(category)
            
            self.size = len(self.used_indices)
    
    def add_batch(self,
                  states: np.ndarray,
                  actions: np.ndarray,
                  rewards: np.ndarray,
                  next_states: np.ndarray,
                  dones: np.ndarray):
        """批量添加经验"""
        batch_size = len(states)
        
        with self.lock:
            for i in range(batch_size):
                reward = float(rewards[i]) if rewards.ndim == 1 else float(rewards[i, 0])
                category = self._get_category(reward)
                priority = self._compute_priority(reward)
                
                idx = self._get_free_index(category)
                
                self.states[idx] = states[i]
                self.actions[idx] = actions[i]
                self.rewards[idx] = reward
                self.next_states[idx] = next_states[i]
                self.dones[idx] = float(dones[i]) if dones.ndim == 1 else float(dones[i, 0])
                self.priorities[idx] = priority
                self.timestamps[idx] = self.global_timestamp
                self.global_timestamp += 1
                
                self.used_indices.add(idx)
                self.category_queues[category].append(idx)
                
                self._enforce_category_limit(category)
            
            self.size = len(self.used_indices)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """按优先级采样一批经验"""
        used_list = list(self.used_indices)
        
        if len(used_list) < batch_size:
            indices = np.random.choice(used_list, size=batch_size, replace=True)
        else:
            priorities = self.priorities[used_list]
            
            if self.priority_alpha > 0:
                probs = priorities ** self.priority_alpha
                probs = probs / probs.sum()
                
                sample_indices = np.random.choice(
                    len(used_list), size=batch_size, replace=False, p=probs
                )
                indices = np.array(used_list)[sample_indices]
            else:
                indices = np.random.choice(used_list, size=batch_size, replace=False)
        
        return {
            'states': torch.FloatTensor(self.states[indices]).to(self.device),
            'actions': torch.FloatTensor(self.actions[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'next_states': torch.FloatTensor(self.next_states[indices]).to(self.device),
            'dones': torch.FloatTensor(self.dones[indices]).to(self.device)
        }
    
    def get_stats(self) -> Dict[str, float]:
        """获取缓冲区统计信息"""
        if self.size == 0:
            return {'size': 0}
        
        used_list = list(self.used_indices)
        rewards = self.rewards[used_list].flatten()
        
        stats = {
            'size': self.size,
            'min_reward': float(rewards.min()),
            'max_reward': float(rewards.max()),
            'mean_reward': float(rewards.mean()),
        }
        
        # 各类别数量
        for name in self.category_queues:
            stats[f'cat_{name}'] = len(self.category_queues[name])
        
        return stats
    
    def get_category_summary(self) -> str:
        """获取类别分布摘要"""
        lines = []
        for name in self.category_queues:
            count = len(self.category_queues[name])
            limit = self.category_limits[name]
            pct = count / max(self.size, 1) * 100
            lines.append(f"  {name}: {count}/{limit} ({pct:.1f}%)")
        return "\n".join(lines)
    
    def __len__(self) -> int:
        return self.size
    
    def is_ready(self, min_size: int) -> bool:
        """检查是否有足够的经验开始训练"""
        return self.size >= min_size


class EpisodeBuffer:
    """
    Episode 级别的经验缓冲
    
    用于收集完整的 episode 数据
    """
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
    
    def add(self,
            state: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_state: np.ndarray,
            done: bool):
        """添加一步经验"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def get_all(self) -> Tuple[np.ndarray, ...]:
        """获取所有经验"""
        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.actions, dtype=np.float32),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.next_states, dtype=np.float32),
            np.array(self.dones, dtype=np.float32)
        )
    
    def clear(self):
        """清空缓冲区"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
    
    def __len__(self) -> int:
        return len(self.states)


class SharedReplayBuffer:
    """
    共享内存经验回放缓冲区
    
    用于多进程环境
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 buffer_size: int = 500000):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            buffer_size: 缓冲区大小
        """
        import multiprocessing as mp
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        
        # 使用共享内存
        self.states = mp.Array('f', buffer_size * state_dim)
        self.actions = mp.Array('f', buffer_size * action_dim)
        self.rewards = mp.Array('f', buffer_size)
        self.next_states = mp.Array('f', buffer_size * state_dim)
        self.dones = mp.Array('f', buffer_size)
        
        self.ptr = mp.Value('i', 0)
        self.size = mp.Value('i', 0)
        self.lock = mp.Lock()
    
    def add(self,
            state: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_state: np.ndarray,
            done: bool):
        """添加单条经验"""
        with self.lock:
            ptr = self.ptr.value
            
            # 写入状态
            start_idx = ptr * self.state_dim
            for i, v in enumerate(state.flatten()):
                self.states[start_idx + i] = v
            
            # 写入动作
            start_idx = ptr * self.action_dim
            for i, v in enumerate(action.flatten()):
                self.actions[start_idx + i] = v
            
            # 写入奖励
            self.rewards[ptr] = reward
            
            # 写入下一状态
            start_idx = ptr * self.state_dim
            for i, v in enumerate(next_state.flatten()):
                self.next_states[start_idx + i] = v
            
            # 写入结束标志
            self.dones[ptr] = float(done)
            
            # 更新指针
            self.ptr.value = (ptr + 1) % self.buffer_size
            self.size.value = min(self.size.value + 1, self.buffer_size)
    
    def sample(self, batch_size: int, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """随机采样"""
        with self.lock:
            current_size = self.size.value
        
        indices = np.random.randint(0, current_size, size=batch_size)
        
        # 读取数据
        states = np.zeros((batch_size, self.state_dim), dtype=np.float32)
        actions = np.zeros((batch_size, self.action_dim), dtype=np.float32)
        rewards = np.zeros((batch_size, 1), dtype=np.float32)
        next_states = np.zeros((batch_size, self.state_dim), dtype=np.float32)
        dones = np.zeros((batch_size, 1), dtype=np.float32)
        
        for i, idx in enumerate(indices):
            # 读取状态
            start_idx = idx * self.state_dim
            states[i] = self.states[start_idx:start_idx + self.state_dim]
            
            # 读取动作
            start_idx = idx * self.action_dim
            actions[i] = self.actions[start_idx:start_idx + self.action_dim]
            
            # 读取奖励
            rewards[i] = self.rewards[idx]
            
            # 读取下一状态
            start_idx = idx * self.state_dim
            next_states[i] = self.next_states[start_idx:start_idx + self.state_dim]
            
            # 读取结束标志
            dones[i] = self.dones[idx]
        
        return {
            'states': torch.FloatTensor(states).to(device),
            'actions': torch.FloatTensor(actions).to(device),
            'rewards': torch.FloatTensor(rewards).to(device),
            'next_states': torch.FloatTensor(next_states).to(device),
            'dones': torch.FloatTensor(dones).to(device)
        }
    
    def __len__(self) -> int:
        with self.lock:
            return self.size.value
    
    def is_ready(self, min_size: int) -> bool:
        with self.lock:
            return self.size.value >= min_size

