"""
replay_buffer.py - 经验回放缓冲区

实现高效的经验存储和采样
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
import threading


class ReplayBuffer:
    """
    经验回放缓冲区
    
    支持多进程写入和采样
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 buffer_size: int = 500000,
                 device: str = 'cpu'):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            buffer_size: 缓冲区大小
            device: 设备
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.device = device
        
        # 预分配内存
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
        
        # 线程锁（用于多线程安全）
        self.lock = threading.Lock()
    
    def add(self,
            state: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_state: np.ndarray,
            done: bool):
        """
        添加单条经验
        
        Args:
            state: 状态
            action: 动作
            reward: 奖励
            next_state: 下一状态
            done: 是否结束
        """
        with self.lock:
            self.states[self.ptr] = state
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.next_states[self.ptr] = next_state
            self.dones[self.ptr] = float(done)
            
            self.ptr = (self.ptr + 1) % self.buffer_size
            self.size = min(self.size + 1, self.buffer_size)
    
    def add_batch(self,
                  states: np.ndarray,
                  actions: np.ndarray,
                  rewards: np.ndarray,
                  next_states: np.ndarray,
                  dones: np.ndarray):
        """
        批量添加经验
        
        Args:
            states: 状态数组 [batch, state_dim]
            actions: 动作数组 [batch, action_dim]
            rewards: 奖励数组 [batch]
            next_states: 下一状态数组 [batch, state_dim]
            dones: 结束标志数组 [batch]
        """
        batch_size = len(states)
        
        with self.lock:
            # 计算插入位置
            if self.ptr + batch_size <= self.buffer_size:
                # 可以直接插入
                self.states[self.ptr:self.ptr + batch_size] = states
                self.actions[self.ptr:self.ptr + batch_size] = actions
                self.rewards[self.ptr:self.ptr + batch_size] = rewards.reshape(-1, 1)
                self.next_states[self.ptr:self.ptr + batch_size] = next_states
                self.dones[self.ptr:self.ptr + batch_size] = dones.reshape(-1, 1)
            else:
                # 需要分两段插入（环形缓冲区）
                first_part = self.buffer_size - self.ptr
                second_part = batch_size - first_part
                
                self.states[self.ptr:] = states[:first_part]
                self.states[:second_part] = states[first_part:]
                
                self.actions[self.ptr:] = actions[:first_part]
                self.actions[:second_part] = actions[first_part:]
                
                self.rewards[self.ptr:] = rewards[:first_part].reshape(-1, 1)
                self.rewards[:second_part] = rewards[first_part:].reshape(-1, 1)
                
                self.next_states[self.ptr:] = next_states[:first_part]
                self.next_states[:second_part] = next_states[first_part:]
                
                self.dones[self.ptr:] = dones[:first_part].reshape(-1, 1)
                self.dones[:second_part] = dones[first_part:].reshape(-1, 1)
            
            self.ptr = (self.ptr + batch_size) % self.buffer_size
            self.size = min(self.size + batch_size, self.buffer_size)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        随机采样一批经验
        
        Args:
            batch_size: 采样数量
        
        Returns:
            Dict: {'states', 'actions', 'rewards', 'next_states', 'dones'}
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            'states': torch.FloatTensor(self.states[indices]).to(self.device),
            'actions': torch.FloatTensor(self.actions[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'next_states': torch.FloatTensor(self.next_states[indices]).to(self.device),
            'dones': torch.FloatTensor(self.dones[indices]).to(self.device)
        }
    
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

