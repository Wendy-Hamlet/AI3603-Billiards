"""
pool_wrapper.py - 台球环境封装器

将 PoolEnv 封装为适合 SAC 训练的接口：
- 自对弈模式
- 课程学习支持
- 简化环境设置
"""

import numpy as np
import copy
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from poolenv import PoolEnv
from train.environment.state_encoder import StateEncoder, ActionSpace

import pooltool as pt


class PoolEnvWrapper:
    """
    台球环境封装器
    
    特点：
    - 自对弈模式：双方使用同一策略
    - 课程学习：可设置初始球数
    - 统一的状态编码和动作解码
    """
    
    def __init__(self,
                 own_balls: int = 7,
                 enemy_balls: int = 7,
                 enable_noise: bool = True):
        """
        Args:
            own_balls: 己方球数量（1-7）
            enemy_balls: 对方球数量（1-7）
            enable_noise: 是否启用动作噪声
        """
        self.env = PoolEnv()
        self.env.enable_noise = enable_noise
        
        self.own_balls = own_balls
        self.enemy_balls = enemy_balls
        
        self.state_encoder = StateEncoder()
        self.action_space = ActionSpace()
        
        self.state_dim = self.state_encoder.get_state_dim()
        self.action_dim = self.action_space.get_action_dim()
        
        # 当前回合信息
        self.current_player = None
        self.my_targets = None
        self.enemy_targets = None
        self.episode_step = 0
        self.balls_before_shot = None
    
    def reset(self, target_ball: str = 'solid') -> np.ndarray:
        """
        重置环境
        
        Args:
            target_ball: Player A 的目标球类型 ('solid' 或 'stripe')
        
        Returns:
            np.ndarray: 初始状态
        """
        self.env.reset(target_ball=target_ball)
        
        # 设置课程学习的初始球数
        self._setup_curriculum()
        
        self.current_player = self.env.get_curr_player()
        self.episode_step = 0
        
        # 获取目标球信息
        balls, my_targets, table = self.env.get_observation()
        self.my_targets = my_targets.copy()
        
        # 确定对方目标球
        if '1' in my_targets:
            self.enemy_targets = [str(i) for i in range(9, 16)]
        else:
            self.enemy_targets = [str(i) for i in range(1, 8)]
        
        # 编码状态
        state = self.state_encoder.encode(
            balls, my_targets, table, self.env.hit_count
        )
        
        return state
    
    def _setup_curriculum(self):
        """
        设置课程学习的初始球数
        
        将多余的球标记为已进袋
        """
        if self.own_balls == 7 and self.enemy_balls == 7:
            return  # 完整游戏，不需要修改
        
        # 获取当前玩家的目标球
        player = self.env.get_curr_player()
        my_targets = self.env.player_targets[player]
        
        # 确定己方和对方球ID
        if '1' in my_targets:
            own_ids = [str(i) for i in range(1, 8)]
            enemy_ids = [str(i) for i in range(9, 16)]
        else:
            own_ids = [str(i) for i in range(9, 16)]
            enemy_ids = [str(i) for i in range(1, 8)]
        
        # 计算需要"移除"的球数
        own_to_remove = 7 - self.own_balls
        enemy_to_remove = 7 - self.enemy_balls
        
        # 将球标记为已进袋（state.s = 4）
        # 随机选择要移除的球
        import random
        
        if own_to_remove > 0:
            balls_to_remove = random.sample(own_ids, own_to_remove)
            for bid in balls_to_remove:
                # 将球移出台面（设置进袋状态）
                self.env.balls[bid].state.s = 4
                # 更新 last_state
                self.env.last_state[bid].state.s = 4
        
        if enemy_to_remove > 0:
            balls_to_remove = random.sample(enemy_ids, enemy_to_remove)
            for bid in balls_to_remove:
                self.env.balls[bid].state.s = 4
                self.env.last_state[bid].state.s = 4
    
    def step(self, action: np.ndarray):
        """
        执行一步动作
        
        Args:
            action: 归一化动作 [-1, 1]^5
        
        Returns:
            Tuple: (next_state, reward, done, info)
        """
        # 保存击球前的状态
        self.balls_before_shot = copy.deepcopy(self.env.balls)
        
        # 解码动作
        action_dict = self.action_space.from_normalized(action)
        
        # 记录击球前的信息
        player_before = self.env.get_curr_player()
        
        # 执行击球
        shot_result = self.env.take_shot(action_dict)
        
        # 获取击球后的状态
        done, game_info = self.env.get_done()
        balls_after = copy.deepcopy(self.env.balls)
        
        # 获取新的观测
        if not done:
            balls, my_targets, table = self.env.get_observation()
            next_state = self.state_encoder.encode(
                balls, my_targets, table, self.env.hit_count
            )
        else:
            # 游戏结束，返回最终状态
            next_state = self.state_encoder.encode(
                balls_after, self.my_targets, self.env.table, self.env.hit_count
            )
        
        self.episode_step += 1
        
        # 构建完整信息
        info = {
            'shot_result': shot_result,
            'balls_before': self.balls_before_shot,
            'balls_after': balls_after,
            'my_targets': self.my_targets,
            'enemy_targets': self.enemy_targets,
            'player': player_before,
            'game_info': game_info if done else {},
            'done': done,
            'winner': game_info.get('winner') if done else None,
            'hit_count': self.env.hit_count
        }
        
        return next_state, info
    
    def get_state(self) -> np.ndarray:
        """获取当前状态编码"""
        balls, my_targets, table = self.env.get_observation()
        return self.state_encoder.encode(
            balls, my_targets, table, self.env.hit_count
        )
    
    def get_current_player(self) -> str:
        """获取当前玩家"""
        return self.env.get_curr_player()
    
    def get_table(self):
        """获取球桌对象"""
        return self.env.table


class SelfPlayEnv:
    """
    自对弈环境
    
    双方使用同一策略进行对弈
    记录双方的经验
    """
    
    def __init__(self,
                 own_balls: int = 7,
                 enemy_balls: int = 7,
                 enable_noise: bool = True):
        """
        Args:
            own_balls: 初始己方球数
            enemy_balls: 初始对方球数
            enable_noise: 是否启用动作噪声
        """
        self.env = PoolEnvWrapper(
            own_balls=own_balls,
            enemy_balls=enemy_balls,
            enable_noise=enable_noise
        )
        
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim
        
        # 双方的经验缓存
        self.player_a_experiences = []
        self.player_b_experiences = []
        
        # 当前状态
        self.current_state = None
        self.current_player = None
    
    def reset(self, target_ball: str = None) -> tuple:
        """
        重置环境
        
        Args:
            target_ball: Player A 的目标球类型（None表示随机）
        
        Returns:
            Tuple: (初始状态, 当前玩家)
        """
        if target_ball is None:
            target_ball = np.random.choice(['solid', 'stripe'])
        
        self.current_state = self.env.reset(target_ball=target_ball)
        self.current_player = self.env.get_current_player()
        
        # 清空经验缓存
        self.player_a_experiences.clear()
        self.player_b_experiences.clear()
        
        return self.current_state, self.current_player
    
    def step(self, action: np.ndarray):
        """
        执行一步
        
        Args:
            action: 归一化动作
        
        Returns:
            Tuple: (next_state, reward, done, info, next_player)
        """
        from train.sac.reward import RewardCalculator
        
        player_before = self.current_player
        state_before = self.current_state.copy()
        
        # 执行动作
        next_state, info = self.env.step(action)
        
        done = info['done']
        winner = info['winner']
        
        # 计算奖励
        reward_calc = RewardCalculator()
        reward, reward_details = reward_calc.compute_reward(
            shot_result=info['shot_result'],
            balls_before=info['balls_before'],
            balls_after=info['balls_after'],
            my_targets=info['my_targets'],
            enemy_targets=info['enemy_targets'],
            table=self.env.get_table(),
            game_done=done,
            winner=winner,
            my_player=player_before,
            is_my_shot=True
        )
        
        # 记录经验
        experience = {
            'state': state_before,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'reward_details': reward_details
        }
        
        if player_before == 'A':
            self.player_a_experiences.append(experience)
        else:
            self.player_b_experiences.append(experience)
        
        # 更新当前状态
        self.current_state = next_state
        if not done:
            self.current_player = self.env.get_current_player()
        
        info['reward'] = reward
        info['reward_details'] = reward_details
        
        return next_state, reward, done, info, self.current_player
    
    def get_all_experiences(self):
        """
        获取所有经验
        
        返回双方的经验，用于训练
        """
        all_exp = []
        all_exp.extend(self.player_a_experiences)
        all_exp.extend(self.player_b_experiences)
        return all_exp
    
    def set_curriculum(self, own_balls: int, enemy_balls: int):
        """设置课程学习的球数"""
        self.env.own_balls = own_balls
        self.env.enemy_balls = enemy_balls


def create_env(own_balls: int = 7, 
               enemy_balls: int = 7,
               enable_noise: bool = True) -> SelfPlayEnv:
    """
    创建自对弈环境的工厂函数
    
    Args:
        own_balls: 己方球数
        enemy_balls: 对方球数
        enable_noise: 是否启用噪声
    
    Returns:
        SelfPlayEnv: 自对弈环境
    """
    return SelfPlayEnv(
        own_balls=own_balls,
        enemy_balls=enemy_balls,
        enable_noise=enable_noise
    )

