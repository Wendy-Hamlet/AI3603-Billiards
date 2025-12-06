"""
External Agents Module
整合来自其他分支的 Agent 实现
- PhysicsAgent (NewAgent): 来自 feature/physics-simulation 分支
- MCTSAgent: 来自 MCTS 分支
"""

import math
import pooltool as pt
import numpy as np
import copy
import random
import warnings
from typing import Dict, List, Optional, Callable

# 尝试导入 MCTS 相关依赖（可选）
try:
    from bayes_opt import BayesianOptimization
    MCTS_AVAILABLE = True
except ImportError:
    MCTS_AVAILABLE = False
    print("⚠️  bayesian-optimization 未安装，MCTSAgent 将不可用")


# ==================== Physics Agent (NewAgent) ====================

class PhysicsAgent:
    """
    物理模拟 Agent (来自 feature/physics-simulation 分支)
    使用几何计算和物理模拟来选择最佳击球方案
    """
    
    def __init__(self):
        self.ball_radius = 0.028575  # 2.25 inch
        self.table_width = 0.9906
        self.table_length = 1.9812
        self.pocket_radius_corner = 0.062
        self.pocket_radius_side = 0.0645
        
        # 噪声安全边距
        self.NOISE_SAFETY_MARGIN = 3 * self.ball_radius
        
        self._init_physics_params()
    
    def _init_physics_params(self):
        """初始化物理参数"""
        self.pockets = None  # 将在首次使用时初始化
    
    def decision(self, balls, my_type, table):
        """决策接口"""
        # 初始化 pockets
        if self.pockets is None:
            self.pockets = {name: pocket.center for name, pocket in table.pockets.items()}
        
        # 获取母球和目标球
        cue_ball = balls.get('cue') or balls.get(0)
        if cue_ball is None or cue_ball.state.s == 4:
            return self._random_action()
        
        cue_pos = np.array(cue_ball.xyz[:2])
        
        # 获取我方球
        my_ball_ids = self._get_my_ball_ids(my_type, balls)
        
        if not my_ball_ids:
            return self._random_action()
        
        # 评估所有可能的击球
        best_score = -float('inf')
        best_action = None
        
        for ball_id in my_ball_ids:
            ball = balls.get(ball_id)
            if ball is None or ball.state.s == 4:
                continue
            
            ball_pos = np.array(ball.xyz[:2])
            
            # 对每个球袋尝试击球
            for pocket_name, pocket_pos in self.pockets.items():
                pocket_pos_2d = np.array(pocket_pos[:2])
                
                # 计算目标点
                action, score = self._calculate_shot(
                    cue_pos, ball_pos, pocket_pos_2d, 
                    balls, ball_id, pocket_name
                )
                
                if score > best_score:
                    best_score = score
                    best_action = action
        
        if best_action is None:
            return self._random_action()
        
        return best_action
    
    def _calculate_shot(self, cue_pos, ball_pos, pocket_pos, balls, ball_id, pocket_name):
        """计算击球参数"""
        # 计算 ghost ball 位置
        direction = pocket_pos - ball_pos
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            return None, -float('inf')
        
        direction = direction / distance
        ghost_ball_pos = ball_pos - direction * (2 * self.ball_radius)
        
        # 计算击球角度
        aim_vector = ghost_ball_pos - cue_pos
        phi = np.degrees(np.arctan2(aim_vector[1], aim_vector[0]))
        
        # 计算切角
        cue_to_ball = ball_pos - cue_pos
        cue_to_ball_norm = cue_to_ball / (np.linalg.norm(cue_to_ball) + 1e-9)
        cut_angle = np.degrees(np.arccos(np.clip(np.dot(cue_to_ball_norm, direction), -1, 1)))
        
        # 噪声鲁棒性检查：切角不能太大
        if cut_angle > 60:  # 降低阈值以提高成功率
            return None, -float('inf')
        
        # 检查路径是否有障碍
        if self._path_has_obstacle(cue_pos, ghost_ball_pos, balls, ignore_ball=ball_id):
            return None, -float('inf')
        
        # 估算速度
        shot_distance = np.linalg.norm(aim_vector)
        V0 = self._estimate_velocity(shot_distance, cut_angle)
        
        # 构造动作
        action = {
            'V0': float(V0),
            'phi': float(phi % 360),
            'theta': 0.0,  # 平击
            'a': 0.0,
            'b': 0.0
        }
        
        # 评分
        score = self._calculate_shot_score(shot_distance, cut_angle, distance)
        
        return action, score
    
    def _path_has_obstacle(self, start, end, balls, ignore_ball=None):
        """检查路径上是否有障碍球"""
        path_vec = end - start
        path_len = np.linalg.norm(path_vec)
        
        if path_len < 1e-6:
            return False
        
        path_dir = path_vec / path_len
        
        # 噪声鲁棒：增加安全边距
        safe_dist = 2 * self.ball_radius + self.NOISE_SAFETY_MARGIN
        
        for ball_id, ball in balls.items():
            if ball_id == 'cue' or ball_id == ignore_ball:
                continue
            if ball.state.s == 4:  # 已入袋
                continue
            
            ball_pos = np.array(ball.xyz[:2])
            
            # 点到线段的距离
            start_to_ball = ball_pos - start
            proj_len = np.dot(start_to_ball, path_dir)
            
            if proj_len < 0 or proj_len > path_len:
                continue
            
            proj_point = start + path_dir * proj_len
            dist = np.linalg.norm(ball_pos - proj_point)
            
            if dist < safe_dist:
                return True
        
        return False
    
    def _estimate_velocity(self, distance, cut_angle):
        """估算所需速度"""
        # 基础速度：根据距离
        base_v = 1.0 + distance * 0.8
        
        # 切角补偿：大切角需要更大速度
        angle_factor = 1.0 + (cut_angle / 90.0) * 0.3
        
        V0 = base_v * angle_factor
        return np.clip(V0, 0.8, 2.5)
    
    def _calculate_shot_score(self, shot_distance, cut_angle, pocket_distance):
        """计算击球方案的评分"""
        # 距离因素：越近越好
        distance_score = 100 / (1 + shot_distance)
        
        # 切角因素：越小越好
        angle_score = 100 * (1 - cut_angle / 90.0)
        
        # 目标球到球袋距离：越近越好
        pocket_score = 50 / (1 + pocket_distance)
        
        return distance_score + angle_score + pocket_score
    
    def _get_my_ball_ids(self, my_type, balls):
        """获取我方球的ID列表"""
        if my_type == 'solid':
            candidate_ids = ['1', '2', '3', '4', '5', '6', '7', 1, 2, 3, 4, 5, 6, 7]
        else:  # stripe
            candidate_ids = ['9', '10', '11', '12', '13', '14', '15', 9, 10, 11, 12, 13, 14, 15]
        
        my_ball_ids = []
        for ball_id in candidate_ids:
            ball = balls.get(ball_id)
            if ball is not None and ball.state.s != 4:
                my_ball_ids.append(ball_id)
        
        # 如果所有己方球都进了，目标是8号球
        if not my_ball_ids:
            ball_8 = balls.get('8') or balls.get(8)
            if ball_8 and ball_8.state.s != 4:
                my_ball_ids = ['8']
        
        return my_ball_ids
    
    def _random_action(self):
        """随机动作"""
        return {
            'V0': float(np.random.uniform(0.8, 2.5)),
            'phi': float(np.random.uniform(0, 360)),
            'theta': 0.0,
            'a': 0.0,
            'b': 0.0
        }


# ==================== MCTS Agent ====================

if MCTS_AVAILABLE:
    
    class MCTSNode:
        """MCTS 树节点"""
        def __init__(self, balls, table, my_targets, parent=None, action=None, depth=0):
            self.balls = balls
            self.table = table
            self.my_targets = list(my_targets)
            self.parent = parent
            self.action = action
            self.depth = depth
            self.children: List['MCTSNode'] = []
            self.visits = 0
            self.value = 0.0

        def is_leaf(self):
            return len(self.children) == 0

        def ucb_score(self, c=1.4):
            if self.visits == 0:
                return float('inf')
            return self.value / self.visits + c * math.sqrt(math.log(self.parent.visits + 1) / self.visits)


    class MCTSSolver:
        """MCTS 求解器"""
        def __init__(
            self,
            pbounds: Dict[str, tuple],
            reward_fn: Callable,
            num_simulations: int = 120,
            max_depth: int = 2,
            exploration_c: float = 1.4,
            rollout_per_leaf: int = 1,
            enable_noise: bool = True,
            noise_std: Optional[Dict[str, float]] = None,
        ):
            self.pbounds = pbounds
            self.reward_fn = reward_fn
            self.num_simulations = num_simulations
            self.max_depth = max_depth
            self.exploration_c = exploration_c
            self.rollout_per_leaf = rollout_per_leaf
            self.enable_noise = enable_noise
            self.noise_std = noise_std or {
                'V0': 0.1,
                'phi': 0.1,
                'theta': 0.1,
                'a': 0.003,
                'b': 0.003,
            }

        def search(self, balls, my_targets, table, candidate_actions: List[dict]):
            root = MCTSNode(copy.deepcopy(balls), copy.deepcopy(table), my_targets, None, None, 0)
            root.children = [
                MCTSNode(copy.deepcopy(balls), copy.deepcopy(table), my_targets, root, act, 1)
                for act in candidate_actions
            ]

            for _ in range(self.num_simulations):
                leaf = self._select(root)
                value = self._simulate_from_node(leaf)
                self._backpropagate(leaf, value)

            if not root.children:
                return None
            best_child = max(root.children, key=lambda n: n.visits)
            return best_child.action

        def _select(self, node: MCTSNode) -> MCTSNode:
            curr = node
            while not curr.is_leaf() and curr.depth < self.max_depth:
                curr = max(curr.children, key=lambda ch: ch.ucb_score(self.exploration_c))
            return curr

        def _simulate_from_node(self, node: MCTSNode) -> float:
            total = 0.0
            for _ in range(self.rollout_per_leaf):
                total += self._rollout_once(node)
            return total / float(self.rollout_per_leaf)

        def _rollout_once(self, node: MCTSNode) -> float:
            balls = copy.deepcopy(node.balls)
            table = copy.deepcopy(node.table)
            my_targets = node.my_targets
            depth = node.depth
            value = 0.0

            if node.action is not None:
                reward, balls, table = self._simulate_action(balls, table, my_targets, node.action)
                value += reward

            while depth < self.max_depth:
                action = self._random_action()
                reward, balls, table = self._simulate_action(balls, table, my_targets, action)
                value += reward * (0.9 ** depth)
                depth += 1
            return value

        def _backpropagate(self, node: MCTSNode, value: float):
            curr = node
            while curr is not None:
                curr.visits += 1
                curr.value += value
                curr = curr.parent

        def _simulate_action(self, balls, table, my_targets, action):
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            sim_table = copy.deepcopy(table)
            cue = pt.Cue(cue_ball_id='cue')
            shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
            act = self._apply_noise(action) if self.enable_noise else action
            try:
                cue.set_state(
                    V0=act['V0'],
                    phi=act['phi'],
                    theta=act['theta'],
                    a=act['a'],
                    b=act['b'],
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=RuntimeWarning, module=r"pooltool\.ptmath\.roots\.core")
                    pt.simulate(shot, inplace=True)
            except Exception:
                return -500.0, balls, table
            reward = self.reward_fn(shot=shot, last_state=balls, player_targets=my_targets)
            return reward, shot.balls, sim_table

        def _apply_noise(self, action: dict) -> dict:
            noisy = dict(action)
            for k, std in self.noise_std.items():
                noisy[k] = float(np.clip(noisy[k] + np.random.normal(0, std), *self.pbounds[k]))
            if 'phi' in noisy:
                noisy['phi'] = noisy['phi'] % 360.0
            return noisy

        def _random_action(self):
            return {
                'V0': random.uniform(max(0.8, self.pbounds['V0'][0]), min(6.5, self.pbounds['V0'][1])),
                'phi': random.uniform(*self.pbounds['phi']),
                'theta': random.uniform(0.0, min(12.0, self.pbounds['theta'][1])),
                'a': random.uniform(max(-0.2, self.pbounds['a'][0]), min(0.2, self.pbounds['a'][1])),
                'b': random.uniform(max(-0.2, self.pbounds['b'][0]), min(0.2, self.pbounds['b'][1])),
            }


    def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
        """奖励函数（用于MCTS）"""
        new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
        own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
        enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
        cue_pocketed = "cue" in new_pocketed
        eight_pocketed = "8" in new_pocketed

        first_contact_ball_id = None
        foul_first_hit = False
        for e in shot.events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
                other_ids = [i for i in ids if i != 'cue']
                if other_ids:
                    first_contact_ball_id = other_ids[0]
                    break
        
        if first_contact_ball_id is None:
            if len(last_state) > 2:
                foul_first_hit = True
        else:
            remaining_own_before = [bid for bid in player_targets if last_state[bid].state.s != 4]
            opponent_plus_eight = [bid for bid in last_state.keys() if bid not in player_targets and bid not in ['cue']]
            if '8' not in opponent_plus_eight:
                opponent_plus_eight.append('8')
            if len(remaining_own_before) > 0 and first_contact_ball_id in opponent_plus_eight:
                foul_first_hit = True

        score = 0
        if cue_pocketed and eight_pocketed:
            score -= 150
        elif cue_pocketed:
            score -= 100
        elif eight_pocketed:
            is_targeting_eight_ball_legally = (len(player_targets) == 1 and player_targets[0] == "8")
            score += 100 if is_targeting_eight_ball_legally else -150
        if foul_first_hit:
            score -= 30
        score += len(own_pocketed) * 50
        score -= len(enemy_pocketed) * 20
        if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit:
            score = 10
        return score


    class MCTSAgent:
        """
        MCTS Agent (来自 MCTS 分支)
        使用蒙特卡洛树搜索来规划最优击球策略
        """
        def __init__(self, num_simulations=120, num_candidates=20, max_depth=2):
            self.num_simulations = num_simulations
            self.num_candidates = num_candidates
            self.max_depth = max_depth
            
            self.pbounds = {
                "V0": (0.6, 8.0),
                "phi": (0.0, 360.0),
                "theta": (0.0, 90.0),
                "a": (-0.5, 0.5),
                "b": (-0.5, 0.5),
            }
            
            self.mcts = MCTSSolver(
                pbounds=self.pbounds,
                reward_fn=analyze_shot_for_reward,
                num_simulations=num_simulations,
                max_depth=max_depth,
                exploration_c=1.4,
                rollout_per_leaf=1,
                enable_noise=True,
            )

        def decision(self, balls, my_type, table):
            """决策接口"""
            try:
                # 转换 my_type 为目标球列表
                if my_type == 'solid':
                    my_targets = [str(i) for i in range(1, 8)]
                else:
                    my_targets = [str(i) for i in range(9, 16)]
                
                # 检查是否需要打8号球
                remaining_own = [bid for bid in my_targets if bid in balls and getattr(balls[bid].state, "s", 0) != 4]
                if len(remaining_own) == 0:
                    my_targets = ["8"]
                
                # 生成候选动作
                candidates = self._generate_candidate_actions(balls, my_targets, table, self.num_candidates)
                if not candidates:
                    return self._random_action()
                
                # MCTS 搜索
                action = self.mcts.search(balls=balls, my_targets=my_targets, table=table, candidate_actions=candidates)
                if action is None:
                    return self._random_action()
                
                return action
            except Exception as e:
                print(f"[MCTSAgent] Error: {e}")
                return self._random_action()

        def _generate_candidate_actions(self, balls, my_targets, table, num_candidates: int):
            """生成候选动作"""
            candidates = []
            cue_ball = balls.get("cue", None)
            if cue_ball is None:
                return candidates

            cue_pos = np.array(cue_ball.state.rvw[0], dtype=float)
            cue_xy = cue_pos[:2]
            ball_radius = 0.028575

            for bid in my_targets:
                if bid == "cue":
                    continue
                ball = balls.get(bid, None)
                if ball is None or getattr(ball.state, "s", 0) == 4:
                    continue
                    
                ball_xy = np.array(ball.state.rvw[0], dtype=float)[:2]
                
                for pocket in table.pockets.values():
                    pocket_xy = np.array(pocket.center, dtype=float)[:2]
                    vec_bp = pocket_xy - ball_xy
                    dist_bp = np.linalg.norm(vec_bp)
                    if dist_bp < 1e-6:
                        continue
                        
                    dir_bp = vec_bp / dist_bp
                    aim_xy = ball_xy - dir_bp * (ball_radius * 2.0)
                    vec_ca = aim_xy - cue_xy
                    
                    if np.linalg.norm(vec_ca) < 1e-6:
                        continue
                        
                    base_phi_deg = math.degrees(math.atan2(vec_ca[1], vec_ca[0])) % 360.0
                    base_v0 = float(np.clip(dist_bp * 4.5, 1.0, 6.5))
                    
                    for off in [-4.0, -2.0, 0.0, 2.0, 4.0]:
                        phi = (base_phi_deg + off) % 360.0
                        action = {
                            "V0": float(np.random.uniform(base_v0 * 0.9, base_v0 * 1.1)),
                            "phi": float(phi),
                            "theta": float(np.random.uniform(0.0, 8.0)),
                            "a": float(np.random.uniform(-0.12, 0.12)),
                            "b": float(np.random.uniform(-0.12, 0.12)),
                        }
                        candidates.append((dist_bp, action))

            candidates.sort(key=lambda x: x[0])
            geom_actions = [act for _, act in candidates]
            keep_geom = geom_actions[: max(1, int(num_candidates * 0.7))]

            final_candidates = list(keep_geom)
            while len(final_candidates) < num_candidates:
                final_candidates.append(self._random_action())

            if len(final_candidates) > num_candidates:
                idx = np.random.choice(len(final_candidates), size=num_candidates, replace=False)
                final_candidates = [final_candidates[i] for i in idx]
                
            return final_candidates

        def _random_action(self):
            """随机动作"""
            return {
                'V0': float(np.random.uniform(0.6, 8.0)),
                'phi': float(np.random.uniform(0, 360)),
                'theta': float(np.random.uniform(0, 90)),
                'a': float(np.random.uniform(-0.5, 0.5)),
                'b': float(np.random.uniform(-0.5, 0.5))
            }

else:
    # 如果 MCTS 依赖不可用，创建一个占位类
    class MCTSAgent:
        def __init__(self, *args, **kwargs):
            raise ImportError("MCTSAgent 需要 bayesian-optimization 包。请运行: pip install bayesian-optimization")
        
        def decision(self, *args, **kwargs):
            raise ImportError("MCTSAgent 不可用")


# ==================== 测试代码 ====================
if __name__ == '__main__':
    print("=" * 60)
    print("测试外部 Agents")
    print("=" * 60)
    
    # 测试 PhysicsAgent
    print("\n1. 测试 PhysicsAgent")
    physics_agent = PhysicsAgent()
    print(f"   PhysicsAgent 初始化成功")
    print(f"   Ball radius: {physics_agent.ball_radius}")
    
    # 测试 MCTSAgent
    print("\n2. 测试 MCTSAgent")
    if MCTS_AVAILABLE:
        try:
            mcts_agent = MCTSAgent(num_simulations=10)  # 少量模拟用于测试
            print(f"   MCTSAgent 初始化成功")
            print(f"   Simulations: {mcts_agent.num_simulations}")
        except Exception as e:
            print(f"   MCTSAgent 初始化失败: {e}")
    else:
        print(f"   MCTSAgent 不可用（缺少依赖）")
    
    print("\n✅ 外部 Agents 模块测试完成")
