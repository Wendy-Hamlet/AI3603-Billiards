"""
Opponent Pool - 对手池管理
管理不同类型的对手：BasicAgent, PhysicsAgent, MCTSAgent, Self-play checkpoints
"""

import copy
import random
import os
import numpy as np
import torch
from agent import BasicAgent  # 导入 BasicAgent
from external_agents import PhysicsAgent, MCTSAgent  # 导入外部 agents


class RandomAgent:
    """快速随机 Agent - 用于预热阶段，比 BasicAgent 快 100 倍"""
    
    def decision(self, balls=None, my_targets=None, table=None):
        """返回随机动作"""
        return {
            'V0': np.random.uniform(0.5, 3.0),  # 低速，更容易控制
            'phi': np.random.uniform(0, 360),
            'theta': np.random.uniform(0, 30),  # 低仰角
            'a': np.random.uniform(-0.2, 0.2),
            'b': np.random.uniform(-0.2, 0.2)
        }


class OpponentPool:
    """
    对手池管理器
    
    功能：
    1. 管理不同类型的对手
    2. 根据训练阶段选择对手
    3. 管理 self-play checkpoints
    """
    
    def __init__(self, enable_mcts=True, verbose=True):
        """
        Args:
            enable_mcts: bool, 是否启用 MCTS Agent（需要安装 bayesian-optimization）
            verbose: bool, 是否打印初始化信息（worker 进程应设为 False）
        """
        self.verbose = verbose
        
        # 快速对手（用于预热）
        self.random_agent = RandomAgent()
        
        # 固定对手（训练时禁用详细输出）
        self.basic_agent = BasicAgent(verbose=False)
        self.physics_agent = PhysicsAgent()  # 物理模拟 agent
        self.mcts_agent = None
        
        # 尝试初始化 MCTS Agent
        if enable_mcts:
            self.mcts_agent = self._init_mcts_agent()
        
        # Self-play checkpoint 池
        self.checkpoint_pool = []
        self.max_checkpoints = 20
        
        # 缓存最近使用的 checkpoint agent，避免重复创建
        self._cached_checkpoint_agent = None
        self._cached_checkpoint_episode = None
    
    def _init_mcts_agent(self):
        """
        初始化 MCTS Agent
        
        Returns:
            MCTS Agent instance or None
        """
        try:
            mcts_agent = MCTSAgent(num_simulations=120, num_candidates=20, max_depth=2)
            if self.verbose:
                print("✅ MCTS Agent 初始化成功")
            return mcts_agent
        except ImportError as e:
            if self.verbose:
                print(f"⚠️  MCTS Agent 不可用: {e}")
                print(f"   提示: 运行 'pip install bayesian-optimization' 来启用 MCTS Agent")
            return None
        except Exception as e:
            if self.verbose:
                print(f"⚠️  MCTS Agent 初始化失败: {e}")
            return None
    
    def get_opponent(self, opponent_type):
        """
        获取指定类型的对手
        
        Args:
            opponent_type: str, 'random', 'basic', 'physics', 'mcts' 之一
        
        Returns:
            Agent 实例
        """
        if opponent_type == 'random':
            return self.random_agent
        elif opponent_type == 'basic':
            return self.basic_agent
        elif opponent_type == 'physics':
            return self.physics_agent
        elif opponent_type == 'mcts':
            if self.mcts_agent is None:
                if self.verbose:
                    print("⚠️  MCTS Agent 不可用，使用 Physics Agent 代替")
                return self.physics_agent
            return self.mcts_agent
        elif opponent_type == 'self':
            return self.sample_checkpoint()
        else:
            raise ValueError(f"未知的对手类型: {opponent_type}")
    
    def sample_opponent(self, stage_config):
        """
        根据阶段配置采样对手
        
        Args:
            stage_config: dict, 包含 'opponents' 字段，如 {'basic': 0.6, 'physics': 0.3, 'self': 0.1}
        
        Returns:
            opponent agent
        """
        opponents = stage_config['opponents']
        opponent_types = list(opponents.keys())
        weights = list(opponents.values())
        
        opponent_type = random.choices(opponent_types, weights=weights)[0]
        
        # Debug: 如果采样到 self 但池为空，打印警告
        if opponent_type == 'self' and len(self.checkpoint_pool) == 0 and self.verbose:
            print(f"⚠️  采样到 self-play 但 checkpoint 池为空（池大小: {len(self.checkpoint_pool)}）")
        
        return self.get_opponent(opponent_type)
    
    def add_checkpoint(self, sac_agent_wrapper, episode, metrics):
        """
        添加 self-play checkpoint
        
        Args:
            sac_agent_wrapper: SACAgentWrapper instance
            episode: int
            metrics: dict, 评估指标
        """
        # 保存模型状态字典而不是 deepcopy 对象
        # 这样可以避免 PyTorch 的 deepcopy 问题
        checkpoint = {
            'episode': episode,
            'state_dict': {
                'actor': sac_agent_wrapper.sac_agent.actor.state_dict(),
                'critic': sac_agent_wrapper.sac_agent.critic.state_dict(),
            },
            'state_encoder': sac_agent_wrapper.state_encoder,  # StateEncoder 是无状态的，可以共享
            'metrics': metrics
        }
        
        self.checkpoint_pool.append(checkpoint)
        
        # 保持池大小限制
        if len(self.checkpoint_pool) > self.max_checkpoints:
            self.checkpoint_pool.pop(0)  # 移除最旧的
        
        if self.verbose:
            print(f"✅ Checkpoint 添加成功 (episode {episode}), 池大小: {len(self.checkpoint_pool)}")
    
    def sample_checkpoint(self):
        """
        从 checkpoint 池中采样一个对手
        
        策略：70% 采样最强，30% 随机采样历史
        
        Returns:
            opponent agent or None
        """
        if len(self.checkpoint_pool) == 0:
            if self.verbose:
                print("⚠️  Checkpoint 池为空，使用 BasicAgent")
            return self.basic_agent
        
        # 选择一个 checkpoint（70% 最新，30% 随机历史）
        if random.random() < 0.7:
            checkpoint = self.checkpoint_pool[-1]
        else:
            checkpoint = random.choice(self.checkpoint_pool)
        
        # 使用缓存避免重复创建相同的 agent
        if (self._cached_checkpoint_agent is not None and 
            self._cached_checkpoint_episode == checkpoint['episode']):
            return self._cached_checkpoint_agent
        
        # 从 state_dict 创建新的 agent wrapper
        from sac_agent import SACAgent, SACAgentWrapper
        
        # 创建新的 SAC agent 并加载权重
        agent = SACAgent()
        agent.actor.load_state_dict(checkpoint['state_dict']['actor'])
        agent.critic.load_state_dict(checkpoint['state_dict']['critic'])
        
        # 设置为评估模式
        agent.actor.eval()
        agent.critic.eval()
        
        # 创建 wrapper
        wrapper = SACAgentWrapper(agent, checkpoint['state_encoder'])
        
        # 缓存
        self._cached_checkpoint_agent = wrapper
        self._cached_checkpoint_episode = checkpoint['episode']
        
        return wrapper
    
    def get_available_opponents(self):
        """
        获取可用对手列表
        
        Returns:
            list of str
        """
        opponents = ['basic', 'physics']
        
        if self.mcts_agent is not None:
            opponents.append('mcts')
        
        if len(self.checkpoint_pool) > 0:
            opponents.append('self')
        
        return opponents
    
    def get_checkpoint_info(self):
        """
        获取 checkpoint 池信息
        
        Returns:
            list of dict
        """
        return [
            {
                'episode': cp['episode'],
                'metrics': cp['metrics']
            }
            for cp in self.checkpoint_pool
        ]


# ==================== 测试代码 ====================
if __name__ == '__main__':
    """测试对手池"""
    
    print("=" * 50)
    print("测试 Opponent Pool")
    
    # 初始化
    pool = OpponentPool()
    
    print(f"\n可用对手: {pool.get_available_opponents()}")
    
    # 测试获取基础对手
    print("\n1. 测试获取固定对手")
    basic = pool.get_opponent('basic')
    print(f"Basic Agent: {type(basic).__name__}")
    
    physics = pool.get_opponent('physics')
    print(f"Physics Agent: {type(physics).__name__}")
    
    # 测试阶段采样
    print("\n2. 测试阶段采样")
    stage_config = {
        'opponents': {'basic': 0.6, 'physics': 0.4}
    }
    
    sample_counts = {'basic': 0, 'physics': 0}
    for _ in range(100):
        opponent = pool.sample_opponent(stage_config)
        if isinstance(opponent, BasicAgent):
            sample_counts['basic'] += 1
        else:
            sample_counts['physics'] += 1
    
    print(f"采样结果 (100次): {sample_counts}")
    print(f"期望: basic ≈ 60, physics ≈ 40")
    
    # 测试 checkpoint 管理
    print("\n3. 测试 Checkpoint 管理")
    
    # 模拟添加 checkpoints
    for i in range(5):
        # 这里用 BasicAgent 模拟 SACAgentWrapper
        mock_agent = BasicAgent()
        metrics = {'winrate': 0.5 + i * 0.1}
        pool.add_checkpoint(mock_agent, episode=i*1000, metrics=metrics)
    
    print(f"Checkpoint 池大小: {len(pool.checkpoint_pool)}")
    print(f"Checkpoint 信息: {pool.get_checkpoint_info()}")
    
    # 测试采样 checkpoint
    print("\n4. 测试 Checkpoint 采样")
    sample_episodes = []
    for _ in range(20):
        checkpoint_agent = pool.sample_checkpoint()
        # 找到对应的 episode
        for cp in pool.checkpoint_pool:
            if cp['agent'] == checkpoint_agent:
                sample_episodes.append(cp['episode'])
                break
    
    print(f"采样的 episodes: {sample_episodes}")
    print(f"最新 episode 被采样次数: {sample_episodes.count(4000)}")
    
    print("\n✅ Opponent Pool 测试通过！")
