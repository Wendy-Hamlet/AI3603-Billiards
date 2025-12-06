"""
Replay Buffer with Defense Reward Retrospection
支持防守奖励追溯的经验回放池
"""

import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """
    经验回放池
    
    特性：
    1. 存储 (state, action, reward, next_state, done) transitions
    2. 支持防守奖励的延迟追溯
    3. 高效的随机采样
    """
    
    def __init__(self, capacity=500000):
        """
        Args:
            capacity: int, buffer 最大容量
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
        
        # 用于追踪需要补充防守奖励的 transitions
        self.pending_defense_reward_indices = []
    
    def push(self, state, action, reward, next_state, done, meta_info=None):
        """
        存储一个 transition
        
        Args:
            state: numpy array
            action: numpy array
            reward: float
            next_state: numpy array
            done: bool
            meta_info: dict, 额外信息（如：是否是 SAC agent 的回合）
        
        Returns:
            int: 当前 transition 在 buffer 中的索引
        """
        if meta_info is None:
            meta_info = {}
        
        transition = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'meta': meta_info,
            'pending_defense': True,  # 标记为待补充防守奖励
        }
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            idx = len(self.buffer) - 1
        else:
            idx = self.position % self.capacity
            self.buffer[idx] = transition
        
        self.position += 1
        
        # 记录需要补充防守奖励的索引
        if meta_info.get('is_sac_turn', False):
            self.pending_defense_reward_indices.append(idx)
        
        return idx
    
    def add_defense_reward(self, transition_idx, defense_reward):
        """
        追溯添加防守奖励
        
        Args:
            transition_idx: int, transition 索引
            defense_reward: float, 防守奖励值
        """
        if transition_idx < len(self.buffer):
            self.buffer[transition_idx]['reward'] += defense_reward
            self.buffer[transition_idx]['pending_defense'] = False
            
            # 从待处理列表中移除
            if transition_idx in self.pending_defense_reward_indices:
                self.pending_defense_reward_indices.remove(transition_idx)
    
    def sample(self, batch_size):
        """
        随机采样一批 transitions
        
        Args:
            batch_size: int
        
        Returns:
            dict: {
                'states': numpy array of shape (batch_size, state_dim),
                'actions': numpy array of shape (batch_size, action_dim),
                'rewards': numpy array of shape (batch_size, 1),
                'next_states': numpy array of shape (batch_size, state_dim),
                'dones': numpy array of shape (batch_size, 1)
            }
        """
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([t['state'] for t in batch])
        actions = np.array([t['action'] for t in batch])
        rewards = np.array([[t['reward']] for t in batch])
        next_states = np.array([t['next_state'] for t in batch])
        dones = np.array([[float(t['done'])] for t in batch])
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """清空 buffer"""
        self.buffer.clear()
        self.position = 0
        self.pending_defense_reward_indices.clear()
    
    def get_statistics(self):
        """获取 buffer 统计信息（用于调试）"""
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'avg_reward': 0.0,
                'pending_defense': 0
            }
        
        rewards = [t['reward'] for t in self.buffer]
        pending_count = sum(1 for t in self.buffer if t.get('pending_defense', False))
        
        return {
            'size': len(self.buffer),
            'avg_reward': np.mean(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'pending_defense': pending_count
        }


class EpisodeTracker:
    """
    Episode 追踪器：记录一局游戏中的所有 transitions
    用于管理防守奖励的追溯
    """
    
    def __init__(self):
        self.episode_transitions = []
        self.sac_turn_indices = []  # SAC agent 出杆的 buffer 索引
    
    def add_transition(self, buffer_idx, is_sac_turn):
        """
        添加一个 transition 记录
        
        Args:
            buffer_idx: int, 在 replay buffer 中的索引
            is_sac_turn: bool, 是否是 SAC agent 的回合
        """
        self.episode_transitions.append({
            'buffer_idx': buffer_idx,
            'is_sac_turn': is_sac_turn
        })
        
        if is_sac_turn:
            self.sac_turn_indices.append(buffer_idx)
    
    def get_last_sac_turn_idx(self):
        """
        获取最后一次 SAC agent 出杆的 buffer 索引
        
        Returns:
            int or None
        """
        if self.sac_turn_indices:
            return self.sac_turn_indices[-1]
        return None
    
    def reset(self):
        """重置追踪器（新的一局开始）"""
        self.episode_transitions.clear()
        self.sac_turn_indices.clear()


# ==================== 测试代码 ====================
if __name__ == '__main__':
    """测试 Replay Buffer"""
    
    print("=" * 50)
    print("测试 Replay Buffer")
    
    # 初始化
    buffer = ReplayBuffer(capacity=1000)
    state_dim = 53
    action_dim = 5
    
    # 模拟存储 transitions
    print("\n1. 存储 transitions")
    for i in range(100):
        state = np.random.randn(state_dim)
        action = np.random.randn(action_dim)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = False
        
        idx = buffer.push(state, action, reward, next_state, done,
                         meta_info={'is_sac_turn': i % 2 == 0})
    
    print(f"Buffer size: {len(buffer)}")
    stats = buffer.get_statistics()
    print(f"Statistics: {stats}")
    
    # 测试采样
    print("\n2. 采样 batch")
    batch = buffer.sample(batch_size=32)
    print(f"Batch states shape: {batch['states'].shape}")
    print(f"Batch actions shape: {batch['actions'].shape}")
    print(f"Batch rewards shape: {batch['rewards'].shape}")
    
    # 测试防守奖励追溯
    print("\n3. 测试防守奖励追溯")
    test_idx = 10
    original_reward = buffer.buffer[test_idx]['reward']
    print(f"Original reward at index {test_idx}: {original_reward:.3f}")
    
    buffer.add_defense_reward(test_idx, 5.0)
    updated_reward = buffer.buffer[test_idx]['reward']
    print(f"Updated reward at index {test_idx}: {updated_reward:.3f}")
    print(f"Difference: {updated_reward - original_reward:.3f}")
    
    # 测试 Episode Tracker
    print("\n" + "=" * 50)
    print("测试 Episode Tracker")
    
    tracker = EpisodeTracker()
    
    # 模拟一局游戏
    for turn in range(10):
        is_sac = turn % 2 == 0
        buffer_idx = len(buffer)
        tracker.add_transition(buffer_idx, is_sac)
        
        state = np.random.randn(state_dim)
        action = np.random.randn(action_dim)
        reward = 10.0 if is_sac else -5.0
        next_state = np.random.randn(state_dim)
        done = turn == 9
        
        buffer.push(state, action, reward, next_state, done,
                   meta_info={'is_sac_turn': is_sac})
    
    print(f"Episode transitions: {len(tracker.episode_transitions)}")
    print(f"SAC turns: {len(tracker.sac_turn_indices)}")
    
    last_sac_idx = tracker.get_last_sac_turn_idx()
    print(f"Last SAC turn index: {last_sac_idx}")
    
    # 模拟对手犯规，追溯奖励
    if last_sac_idx is not None:
        print(f"\n对手犯规，给 SAC 上一杆追加奖励")
        before = buffer.buffer[last_sac_idx]['reward']
        buffer.add_defense_reward(last_sac_idx, 5.0)
        after = buffer.buffer[last_sac_idx]['reward']
        print(f"Reward before: {before:.2f}, after: {after:.2f}")
    
    print("\n✅ Replay Buffer 测试通过！")
