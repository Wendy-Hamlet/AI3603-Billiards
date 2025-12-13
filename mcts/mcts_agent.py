"""
mcts_agent.py - MCTS台球智能体

蒙特卡洛树搜索实现，利用物理引擎精确模拟
"""

import numpy as np
import copy
import math
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poolenv import PoolEnv
from mcts.evaluator import StateEvaluator


@dataclass
class ActionSample:
    """一个击球动作样本"""
    V0: float      # 初速度 m/s
    phi: float     # 水平角度 度
    theta: float   # 垂直角度 度
    a: float       # 横向偏移
    b: float       # 纵向偏移
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'V0': self.V0,
            'phi': self.phi,
            'theta': self.theta,
            'a': self.a,
            'b': self.b
        }


@dataclass
class MCTSNode:
    """MCTS节点"""
    action: Optional[ActionSample] = None  # 到达此节点的动作
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    
    visits: int = 0
    total_value: float = 0.0
    
    # 即时奖励（击球结果）
    immediate_reward: float = 0.0
    
    # 是否终止节点
    is_terminal: bool = False
    
    @property
    def value(self) -> float:
        """平均价值"""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits
    
    @property
    def ucb1(self) -> float:
        """UCB1值（用于selection）"""
        if self.visits == 0:
            return float('inf')
        
        if self.parent is None or self.parent.visits == 0:
            return self.value
        
        exploration = math.sqrt(2 * math.log(self.parent.visits) / self.visits)
        return self.value + 1.41 * exploration
    
    def best_child(self, exploration_weight: float = 1.41) -> 'MCTSNode':
        """选择最佳子节点（UCB1）"""
        if not self.children:
            return self
        
        best = None
        best_ucb = float('-inf')
        
        for child in self.children:
            if child.visits == 0:
                return child  # 优先访问未访问的节点
            
            ucb = child.value + exploration_weight * math.sqrt(
                2 * math.log(self.visits) / child.visits
            )
            
            if ucb > best_ucb:
                best_ucb = ucb
                best = child
        
        return best
    
    def best_action(self) -> Optional[ActionSample]:
        """返回访问次数最多的子节点对应的动作"""
        if not self.children:
            return None
        
        best_child = max(self.children, key=lambda c: c.visits)
        return best_child.action


class ActionSampler:
    """
    动作采样器
    
    智能采样击球动作，而不是简单的均匀采样
    """
    
    def __init__(self):
        # 速度采样范围
        self.v0_range = (1.5, 5.0)
        
        # 垂直角度（大多数情况用小角度）
        self.theta_values = [0, 5, 10, 20]
        
        # 旋转参数（大多数情况用中心击球）
        self.spin_values = [0, -0.2, 0.2]
    
    def sample_actions(self,
                       cue_pos: np.ndarray,
                       target_balls: List[Tuple[str, np.ndarray]],
                       pockets: Dict[str, np.ndarray],
                       n_samples: int = 50) -> List[ActionSample]:
        """
        智能采样动作
        
        策略：
        1. 针对每个目标球，计算理想击球方向
        2. 针对每个袋口，计算进袋角度
        3. 在最佳方向附近采样
        
        Args:
            cue_pos: 白球位置
            target_balls: [(ball_id, ball_pos), ...]
            pockets: 袋口位置字典
            n_samples: 采样数量
        
        Returns:
            List[ActionSample]: 采样的动作列表
        """
        actions = []
        
        if len(target_balls) == 0:
            # 没有目标球，随机采样
            return self._random_samples(n_samples)
        
        # 计算每个目标球的击球机会
        opportunities = []
        for ball_id, ball_pos in target_balls:
            for pocket_name, pocket_pos in pockets.items():
                # 计算进袋方向
                ball_to_pocket = pocket_pos - ball_pos
                pocket_dist = np.linalg.norm(ball_to_pocket)
                
                if pocket_dist < 0.05:
                    continue
                
                # 理想击球方向（让球朝袋口移动）
                hit_direction = ball_to_pocket / pocket_dist
                
                # 计算白球应该击打的角度
                cue_to_ball = ball_pos - cue_pos
                cue_dist = np.linalg.norm(cue_to_ball)
                
                if cue_dist < 0.05:
                    continue
                
                # 击球角度（phi）
                # 需要让白球击中目标球后，目标球朝袋口方向运动
                # 简化处理：直接瞄准目标球，加上进袋修正
                phi = np.degrees(np.arctan2(cue_to_ball[1], cue_to_ball[0]))
                if phi < 0:
                    phi += 360
                
                # 计算进袋角度质量
                cos_angle = np.dot(cue_to_ball / cue_dist, hit_direction)
                angle_quality = max(0, (cos_angle + 1) / 2)
                
                # 距离因素
                if pocket_dist < 0.4:
                    dist_quality = 1.0
                elif pocket_dist < 0.8:
                    dist_quality = 0.7
                else:
                    dist_quality = 0.4
                
                quality = angle_quality * dist_quality
                
                opportunities.append({
                    'phi': phi,
                    'quality': quality,
                    'distance': cue_dist,
                    'pocket_dist': pocket_dist
                })
        
        if not opportunities:
            return self._random_samples(n_samples)
        
        # 按质量排序
        opportunities.sort(key=lambda x: x['quality'], reverse=True)
        
        # 在最佳机会附近采样
        samples_per_opportunity = max(5, n_samples // len(opportunities[:5]))
        
        for opp in opportunities[:5]:  # 最多考虑5个机会
            base_phi = opp['phi']
            base_dist = opp['distance']
            
            for _ in range(samples_per_opportunity):
                # 在最佳角度附近采样
                phi = base_phi + np.random.normal(0, 10)
                phi = phi % 360
                
                # 根据距离选择速度
                if base_dist < 0.3:
                    v0 = np.random.uniform(1.5, 3.0)
                elif base_dist < 0.8:
                    v0 = np.random.uniform(2.0, 4.0)
                else:
                    v0 = np.random.uniform(3.0, 5.0)
                
                # 垂直角度（大多数用小角度）
                theta = np.random.choice(self.theta_values, p=[0.5, 0.3, 0.15, 0.05])
                
                # 旋转（大多数用中心）
                a = np.random.choice(self.spin_values, p=[0.7, 0.15, 0.15])
                b = np.random.choice(self.spin_values, p=[0.7, 0.15, 0.15])
                
                actions.append(ActionSample(
                    V0=v0,
                    phi=phi,
                    theta=theta,
                    a=a,
                    b=b
                ))
        
        # 补充随机样本
        remaining = n_samples - len(actions)
        if remaining > 0:
            actions.extend(self._random_samples(remaining))
        
        return actions[:n_samples]
    
    def _random_samples(self, n: int) -> List[ActionSample]:
        """纯随机采样"""
        actions = []
        for _ in range(n):
            actions.append(ActionSample(
                V0=np.random.uniform(1.5, 5.0),
                phi=np.random.uniform(0, 360),
                theta=np.random.choice(self.theta_values),
                a=np.random.choice(self.spin_values),
                b=np.random.choice(self.spin_values)
            ))
        return actions


class MCTSAgent:
    """
    MCTS台球智能体
    
    使用蒙特卡洛树搜索寻找最佳击球策略
    """
    
    def __init__(self,
                 simulation_budget: int = 100,
                 time_limit: float = 5.0,
                 exploration_weight: float = 1.41,
                 rollout_depth: int = 3,
                 n_action_samples: int = 30):
        """
        Args:
            simulation_budget: 最大模拟次数
            time_limit: 时间限制（秒）
            exploration_weight: UCB1探索权重
            rollout_depth: rollout深度
            n_action_samples: 每层采样的动作数
        """
        self.simulation_budget = simulation_budget
        self.time_limit = time_limit
        self.exploration_weight = exploration_weight
        self.rollout_depth = rollout_depth
        self.n_action_samples = n_action_samples
        
        self.evaluator = StateEvaluator()
        self.action_sampler = ActionSampler()
        
        # 统计信息
        self.stats = {
            'simulations': 0,
            'time_used': 0.0,
            'best_value': 0.0
        }
    
    def select_action(self,
                      env: PoolEnv,
                      my_player: str,
                      verbose: bool = False) -> Dict[str, float]:
        """
        选择最佳击球动作
        
        Args:
            env: 台球环境（将被复制用于模拟）
            my_player: 己方标识
            verbose: 是否输出详细信息
        
        Returns:
            Dict[str, float]: 击球参数
        """
        start_time = time.time()
        
        # 获取当前状态信息
        balls, my_targets, table = env.get_observation()
        cue_pos = balls['cue'].state.rvw[0][:2]
        
        # 收集目标球位置
        target_balls = []
        for bid in my_targets:
            if bid != '8' and balls[bid].state.s != 4:
                target_balls.append((bid, balls[bid].state.rvw[0][:2]))
        
        # 如果没有普通目标球，考虑黑8
        if len(target_balls) == 0 and balls['8'].state.s != 4:
            target_balls.append(('8', balls['8'].state.rvw[0][:2]))
        
        # 收集袋口位置
        pockets = {}
        for pid, pocket in table.pockets.items():
            pockets[pid] = pocket.center[:2]
        
        # 创建根节点
        root = MCTSNode()
        
        # 初始动作采样
        initial_actions = self.action_sampler.sample_actions(
            cue_pos, target_balls, pockets, self.n_action_samples
        )
        
        # MCTS主循环
        simulations = 0
        while simulations < self.simulation_budget:
            # 检查时间限制
            if time.time() - start_time > self.time_limit:
                break
            
            # 1. Selection + Expansion
            node = self._select_and_expand(root, initial_actions)
            
            # 2. Simulation
            value = self._simulate(env, node, my_player, my_targets)
            
            # 3. Backpropagation
            self._backpropagate(node, value)
            
            simulations += 1
        
        # 选择最佳动作
        best_action = root.best_action()
        
        # 更新统计
        self.stats['simulations'] = simulations
        self.stats['time_used'] = time.time() - start_time
        self.stats['best_value'] = root.value
        
        if verbose:
            print(f"MCTS: {simulations} simulations in {self.stats['time_used']:.2f}s")
            print(f"  Best value: {root.value:.1f}")
            if best_action:
                print(f"  Action: V0={best_action.V0:.1f}, phi={best_action.phi:.1f}")
        
        if best_action is None:
            # fallback: 随机动作
            return {
                'V0': 3.0,
                'phi': np.random.uniform(0, 360),
                'theta': 5.0,
                'a': 0.0,
                'b': 0.0
            }
        
        return best_action.to_dict()
    
    def _select_and_expand(self,
                           root: MCTSNode,
                           actions: List[ActionSample]) -> MCTSNode:
        """Selection + Expansion"""
        node = root
        
        # 如果根节点没有子节点，先扩展
        if not node.children:
            for action in actions:
                child = MCTSNode(action=action, parent=node)
                node.children.append(child)
        
        # Selection: 使用UCB1选择子节点
        while node.children:
            # 如果有未访问的子节点，优先选择
            unvisited = [c for c in node.children if c.visits == 0]
            if unvisited:
                return np.random.choice(unvisited)
            
            # 否则用UCB1选择
            node = node.best_child(self.exploration_weight)
            
            # 如果是终止节点，返回
            if node.is_terminal:
                return node
        
        return node
    
    def _simulate(self,
                  env: PoolEnv,
                  node: MCTSNode,
                  my_player: str,
                  my_targets: List[str]) -> float:
        """
        模拟击球并评估结果
        
        使用物理引擎精确模拟
        """
        if node.action is None:
            return 0.0
        
        # 复制环境状态
        env_copy = self._copy_env(env)
        
        # 执行击球
        action_dict = node.action.to_dict()
        result = env_copy.take_shot(action_dict)
        
        # 获取击球后的状态
        balls_after, _, table = env_copy.get_observation()
        balls_before = env.last_state
        
        # 检查游戏是否结束
        game_done, game_info = env_copy.get_done()
        winner = game_info.get('winner') if game_done else None
        
        # 确定对方目标球
        if '1' in my_targets:
            enemy_targets = [str(i) for i in range(9, 16)]
        else:
            enemy_targets = [str(i) for i in range(1, 8)]
        
        # 评估结果
        value = self.evaluator.evaluate(
            balls_before=balls_before,
            balls_after=balls_after,
            shot_result=result,
            my_targets=my_targets,
            enemy_targets=enemy_targets,
            game_done=game_done,
            winner=winner,
            my_player=my_player
        )
        
        # 记录即时奖励
        node.immediate_reward = value
        node.is_terminal = game_done
        
        return value
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """反向传播更新节点价值"""
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent
    
    def _copy_env(self, env: PoolEnv) -> PoolEnv:
        """
        复制环境状态
        
        创建一个新的环境，复制所有球的状态
        """
        # 创建新环境并初始化
        new_env = PoolEnv(verbose=False)
        new_env.reset(target_ball='solid')  # 先初始化
        
        # 复制球的状态
        for bid, ball in env.balls.items():
            if bid in new_env.balls:
                # 复制位置和速度状态
                new_env.balls[bid].state.rvw = ball.state.rvw.copy()
                new_env.balls[bid].state.s = ball.state.s
        
        # 复制游戏状态
        new_env.hit_count = env.hit_count
        new_env.curr_player = env.curr_player
        new_env.player_targets = copy.deepcopy(env.player_targets)
        new_env.last_state = copy.deepcopy(env.last_state)
        
        return new_env
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()

