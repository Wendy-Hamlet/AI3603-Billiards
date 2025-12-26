"""
agent.py - Agent 决策模块

定义 Agent 基类和具体实现：
- Agent: 基类，定义决策接口
- BasicAgent: 基于贝叶斯优化的参考实现
- NewAgent: 学生自定义实现模板
- analyze_shot_for_reward: 击球结果评分函数
"""

import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import os
from datetime import datetime
import random
import signal
# from poolagent.pool import Pool as CuetipEnv, State as CuetipState
# from poolagent import FunctionAgent

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from mcts import MCTSSolver
import time

# ============ 超时安全模拟机制 ============
class SimulationTimeoutError(Exception):
    """物理模拟超时异常"""
    pass

def _timeout_handler(signum, frame):
    """超时信号处理器"""
    raise SimulationTimeoutError("物理模拟超时")

def simulate_with_timeout(shot, timeout=3):
    """带超时保护的物理模拟
    
    参数：
        shot: pt.System 对象
        timeout: 超时时间（秒），默认3秒
    
    返回：
        bool: True 表示模拟成功，False 表示超时或失败
    
    说明：
        使用 signal.SIGALRM 实现超时机制（仅支持 Unix/Linux）
        超时后自动恢复，不会导致程序卡死
    """
    # 设置超时信号处理器
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)  # 设置超时时间
    
    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)  # 取消超时
        return True
    except SimulationTimeoutError:
        print(f"[WARNING] 物理模拟超时（>{timeout}秒），跳过此次模拟")
        return False
    except Exception as e:
        signal.alarm(0)  # 取消超时
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)  # 恢复原处理器

# ============================================



def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数（完全对齐台球规则）
    
    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...] 或 ['8']
    
    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -500（非法黑8/白球+黑8）, -30（首球/碰库犯规）
    
    规则核心：
        - 清台前：player_targets = ['1'-'7'] 或 ['9'-'15']，黑8不属于任何人
        - 清台后：player_targets = ['8']，黑8成为唯一目标球
    """
    
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    # 根据 player_targets 判断进球归属（黑8只有在清台后才算己方球）
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞（定义合法的球ID集合）
    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            # 过滤掉 'cue' 和非球对象（如 'cue stick'），只保留合法的球ID
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    # 首球犯规判定：完全对齐 player_targets
    if first_contact_ball_id is None:
        # 未击中任何球（但若只剩白球和黑8且已清台，则不算犯规）
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        # 首次击打的球必须是 player_targets 中的球
        if first_contact_ball_id not in player_targets:
            foul_first_hit = True
    
    # 3. 分析碰库
    cue_hit_cushion = False
    target_hit_cushion = False
    foul_no_rail = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if 'cushion' in et:
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
        foul_no_rail = True
        
    # 计算奖励分数
    score = 0
    
    if cue_pocketed and eight_pocketed:
        score -= 500
    elif cue_pocketed:
        score -= 100
    elif eight_pocketed:
        is_targeting_eight_ball_legally = (len(player_targets) == 1 and player_targets[0] == "8")
        score += 150 if is_targeting_eight_ball_legally else -500
            
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10
        
    return score

class Agent():
    """Agent 基类"""
    def __init__(self):
        pass
    
    def decision(self, *args, **kwargs):
        """决策方法（子类需实现）
        
        返回：dict, 包含 'V0', 'phi', 'theta', 'a', 'b'
        """
        pass
    
    def _random_action(self,):
        """生成随机击球动作
        
        返回：dict
            V0: [0.5, 8.0] m/s
            phi: [0, 360] 度
            theta: [0, 90] 度
            a, b: [-0.5, 0.5] 球半径比例
        """
        action = {
            'V0': round(random.uniform(0.5, 8.0), 2),   # 初速度 0.5~8.0 m/s
            'phi': round(random.uniform(0, 360), 2),    # 水平角度 (0°~360°)
            'theta': round(random.uniform(0, 90), 2),   # 垂直角度
            'a': round(random.uniform(-0.5, 0.5), 3),   # 杆头横向偏移（单位：球半径比例）
            'b': round(random.uniform(-0.5, 0.5), 3)    # 杆头纵向偏移
        }
        return action



class BasicAgent(Agent):
    def __init__(self,
                 n_simulations=50,       # 仿真次数
                 c_puct=1.414):          # 探索系数
        super().__init__()
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.ball_radius = 0.028575
        
        # 定义噪声水平 (与 poolenv 保持一致或略大)
        self.sim_noise = {
            'V0': 0.1, 'phi': 0.15, 'theta': 0.1, 'a': 0.005, 'b': 0.005
        }

    def _calc_angle_degrees(self, v):
        angle = math.degrees(math.atan2(v[1], v[0]))
        return angle % 360

    def _get_ghost_ball_target(self, cue_pos, obj_pos, pocket_pos):
        vec_obj_to_pocket = np.array(pocket_pos) - np.array(obj_pos)
        dist_obj_to_pocket = np.linalg.norm(vec_obj_to_pocket)
        if dist_obj_to_pocket == 0: return 0, 0
        unit_vec = vec_obj_to_pocket / dist_obj_to_pocket
        ghost_pos = np.array(obj_pos) - unit_vec * (2 * self.ball_radius)
        vec_cue_to_ghost = ghost_pos - np.array(cue_pos)
        dist_cue_to_ghost = np.linalg.norm(vec_cue_to_ghost)
        phi = self._calc_angle_degrees(vec_cue_to_ghost)
        return phi, dist_cue_to_ghost

    def generate_heuristic_actions(self, balls, my_targets, table):
        """
        生成候选动作列表
        """
        actions = []
        
        cue_ball = balls.get('cue')
        if not cue_ball: return [self._random_action()]
        cue_pos = cue_ball.state.rvw[0]

        # 获取所有目标球的ID
        target_ids = [bid for bid in my_targets if balls[bid].state.s != 4]
        
        # 如果没有目标球了（理论上外部会处理转为8号，这里兜底）
        if not target_ids:
            target_ids = ['8']

        # 遍历每一个目标球
        for tid in target_ids:
            obj_ball = balls[tid]
            obj_pos = obj_ball.state.rvw[0]

            # 遍历每一个袋口
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center

                # 1. 计算理论进球角度
                phi_ideal, dist = self._get_ghost_ball_target(cue_pos, obj_pos, pocket_pos)

                # 2. 根据距离简单的估算力度 (距离越远力度越大，基础力度2.0)
                v_base = 1.5 + dist * 1.5
                v_base = np.clip(v_base, 1.0, 6.0)

                # 3. 生成几个变种动作加入候选池
                # 变种1：精准一击
                actions.append({
                    'V0': v_base, 'phi': phi_ideal, 'theta': 0, 'a': 0, 'b': 0
                })
                # 变种2：力度稍大
                actions.append({
                    'V0': min(v_base + 1.5, 7.5), 'phi': phi_ideal, 'theta': 0, 'a': 0, 'b': 0
                })
                # 变种3：角度微调 (左右偏移 0.5 度，应对噪声)
                actions.append({
                    'V0': v_base, 'phi': (phi_ideal + 0.5) % 360, 'theta': 0, 'a': 0, 'b': 0
                })
                actions.append({
                    'V0': v_base, 'phi': (phi_ideal - 0.5) % 360, 'theta': 0, 'a': 0, 'b': 0
                })

        # 如果通过启发式没有生成任何动作（极罕见），补充随机动作
        if len(actions) == 0:
            for _ in range(5):
                actions.append(self._random_action())
        
        # 随机打乱顺序
        random.shuffle(actions)
        return actions[:30]

    def simulate_action(self, balls, table, action):
        """
        [修改点1] 执行带噪声的物理仿真
        让 Agent 意识到由于误差的存在，某些“极限球”是不可打的
        """
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        
        try:
            # --- 注入高斯噪声 ---
            noisy_V0 = np.clip(action['V0'] + np.random.normal(0, self.sim_noise['V0']), 0.5, 8.0)
            noisy_phi = (action['phi'] + np.random.normal(0, self.sim_noise['phi'])) % 360
            noisy_theta = np.clip(action['theta'] + np.random.normal(0, self.sim_noise['theta']), 0, 90)
            noisy_a = np.clip(action['a'] + np.random.normal(0, self.sim_noise['a']), -0.5, 0.5)
            noisy_b = np.clip(action['b'] + np.random.normal(0, self.sim_noise['b']), -0.5, 0.5)

            cue.set_state(V0=noisy_V0, phi=noisy_phi, theta=noisy_theta, a=noisy_a, b=noisy_b)
            pt.simulate(shot, inplace=True)
            return shot
        except Exception:
            return None

    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None: return self._random_action()
        
        # 预处理
        remaining = [bid for bid in my_targets if balls[bid].state.s != 4]
        if len(remaining) == 0: my_targets = ["8"]
        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

        # 生成候选动作
        candidate_actions = self.generate_heuristic_actions(balls, my_targets, table)
        n_candidates = len(candidate_actions)
        
        N = np.zeros(n_candidates)
        Q = np.zeros(n_candidates)
        
        # MCTS 循环
        for i in range(self.n_simulations):
            # Selection (UCB)
            if i < n_candidates:
                idx = i
            else:
                total_n = np.sum(N)
                # 使用归一化后的 Q 进行计算
                ucb_values = (Q / (N + 1e-6)) + self.c_puct * np.sqrt(np.log(total_n + 1) / (N + 1e-6))
                idx = np.argmax(ucb_values)
            
            # Simulation (带噪声)
            shot = self.simulate_action(balls, table, candidate_actions[idx])

            # Evaluation
            if shot is None:
                raw_reward = -500.0
            else:
                raw_reward = analyze_shot_for_reward(shot, last_state_snapshot, my_targets)
            
            # 映射公式: (val - min) / (max - min)
            normalized_reward = (raw_reward - (-500)) / 650.0
            # 截断一下防止越界
            normalized_reward = np.clip(normalized_reward, 0.0, 1.0)

            # Backpropagation
            N[idx] += 1
            Q[idx] += normalized_reward # 累加归一化后的分数

        # Final Decision
        # 选平均分最高的 (Robust Child)
        avg_rewards = Q / (N + 1e-6)
        best_idx = np.argmax(avg_rewards)
        best_action = candidate_actions[best_idx]
        
        # 简单打印一下当前最好的预测胜率
        print(f"[BasicAgent] Best Avg Score: {avg_rewards[best_idx]:.3f} (Sims: {self.n_simulations})")
        
        return best_action


def analyze_shot_for_reward_mcts(
    shot: pt.System,
    last_state: dict,
    player_targets: list,
    candidate_type: str = None,
):

    new_pocketed = [
        bid for bid, b in shot.balls.items()
        if b.state.s == 4 and last_state[bid].state.s != 4
    ]
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [
        bid for bid in new_pocketed
        if bid not in player_targets and bid not in ["cue", "8"]
    ]
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    first_contact_ball_id = None
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, "ids") else []
        if ("cushion" not in et) and ("pocket" not in et) and ("cue" in ids):
            other_ids = [i for i in ids if i != "cue"]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break

    foul_first_hit = False
    if first_contact_ball_id is None:
        if len(last_state) > 2:
            foul_first_hit = True
    else:
        remaining_own_before = [
            bid for bid in player_targets if last_state[bid].state.s != 4
        ]
        opponent_plus_eight = [
            bid for bid in last_state.keys()
            if bid not in player_targets and bid not in ["cue"]
        ]
        if "8" not in opponent_plus_eight:
            opponent_plus_eight.append("8")
        if len(remaining_own_before) > 0 and first_contact_ball_id in opponent_plus_eight:
            foul_first_hit = True

    cue_hit_cushion = False
    target_hit_cushion = False
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, "ids") else []
        if "cushion" in et:
            if "cue" in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    foul_no_rail = False
    if (
        len(new_pocketed) == 0
        and first_contact_ball_id is not None
        and (not cue_hit_cushion)
        and (not target_hit_cushion)
    ):
        foul_no_rail = True

    contact_bonus = 0
    if first_contact_ball_id is not None:
        contact_bonus += 5
        if first_contact_ball_id in player_targets:
            contact_bonus += 8

    score = float(contact_bonus)

    if cue_pocketed and eight_pocketed:
        score -= 1000.0
    elif cue_pocketed:
        score -= 100.0
    elif eight_pocketed:
        is_targeting_eight_ball_legally = (len(player_targets) == 1 and player_targets[0] == "8")
        score += 100.0 if is_targeting_eight_ball_legally else -1000.0

    if foul_no_rail:
        score -= 30.0

    if foul_first_hit:
        score -= 150.0

    if len(own_pocketed) > 0 and not foul_first_hit:
        score += 70.0

    score += len(own_pocketed) * 50.0
    score -= len(enemy_pocketed) * 20.0

    if (
        score == 0.0
        and not cue_pocketed
        and not eight_pocketed
        and not foul_first_hit
        and not foul_no_rail
    ):
        score = 10.0

    return float(score)


class NewAgent(Agent):
    """Bandit-style Monte Carlo agent (单杆多次蒙特卡洛评估 + 几何候选 + 绕球候选)."""

    def __init__(
        self,
        num_candidates: int = 64,
        num_simulations: int = 400,
        max_depth: int = 2,           # 保留参数，占位
        exploration_c: float = 1.4,     
        rollout_per_leaf: int = 2,    # 保留参数，占位
        risk_aversion: float = 0.4,   # 均值-λ*方差 中的 λ
        num_workers=8,                # CPU 并行线程数，None 表示自动
    ):
        super().__init__()
        self.num_candidates = num_candidates
        self.pbounds = {
            "V0": (0.5, 8.0),
            "phi": (0.0, 360.0),
            "theta": (0.0, 90.0),
            "a": (-0.5, 0.5),
            "b": (-0.5, 0.5),
        }
        self.noise_std = {
            "V0": 0.1,
            "phi": 0.1,
            "theta": 0.1,
            "a": 0.003,
            "b": 0.003,
        }
        self.mcts = MCTSSolver(
            pbounds=self.pbounds,
            reward_fn=analyze_shot_for_reward_mcts,
            num_simulations=num_simulations,
            max_depth=max_depth,
            exploration_c=exploration_c,
            rollout_per_leaf=rollout_per_leaf,
            enable_noise=True,
            noise_std=self.noise_std,
            risk_aversion=risk_aversion,
            num_workers=num_workers,
        )

    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None or my_targets is None or table is None:
            return self._random_action()
        try:
            # 若己方普通球已全部进袋，则只剩黑8
            remaining_own = [
                bid for bid in my_targets
                if bid in balls and getattr(balls[bid].state, "s", 0) != 4 and bid != "8"
            ]
            if len(remaining_own) == 0 and "8" not in my_targets:
                my_targets = ["8"]
                print("[NewAgent] 目标球已清空，切换为黑8")
            remaining_report = [
                bid for bid in my_targets
                if bid in balls and getattr(balls[bid].state, "s", 0) != 4
            ]
            print(f"[NewAgent] 剩余目标球: {remaining_report}")

            candidates = self._generate_candidate_actions(
                balls, my_targets, table, self.num_candidates
            )
            if not candidates:
                return self._random_action()

            action = self.mcts.search(
                balls=balls,
                my_targets=my_targets,
                table=table,
                candidate_actions=candidates,
            )
            if action is None:
                return self._random_action()
            ctype = action.get("candidate_type", "unknown")
            print(
                f"[NewAgent] action({ctype}): "
                f"V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                f"theta={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}"
            )
            return action
        except Exception as e:
            print(f"[NewAgent] error, fallback to random. Reason: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()

    def _generate_candidate_actions(self, balls, my_targets, table, num_candidates: int):
        """
        Generate candidate actions:
        - geom_basic / geom_pro
        - detour_basic / detour_pro
        """
        cue_ball = balls.get("cue", None)
        if cue_ball is None:
            return []

        cue_pos = np.array(cue_ball.state.rvw[0], dtype=float)
        cue_xy = cue_pos[:2]
        ball_radius = 0.028575
        clearance = ball_radius * 2.1   # centerline-to-ball clearance
        angle_offsets = [-4.0, -2.0, 0.0, 2.0, 4.0]
        own_set = set(my_targets)

        base_cost_max = 200.0
        cost_scale = base_cost_max / 30.0
        table_scale = max(float(table.w), float(table.l))

        # --- geometry helpers ---

        def segment_distance(a_xy, b_xy, p_xy):
            ab = b_xy - a_xy
            ab_len2 = float(np.dot(ab, ab))
            if ab_len2 < 1e-9:
                return float(np.linalg.norm(p_xy - a_xy)), 0.0, 0.0
            t = float(np.clip(np.dot(p_xy - a_xy, ab) / ab_len2, 0.0, 1.0))
            proj = a_xy + t * ab
            dist = float(np.linalg.norm(p_xy - proj))
            cross_z = ab[0] * (p_xy[1] - a_xy[1]) - ab[1] * (p_xy[0] - a_xy[0])
            side = 0.0
            if abs(cross_z) > 1e-9:
                side = math.copysign(1.0, cross_z)
            return dist, t, side

        def find_blockers(target_xy):
            blockers = []
            line_vec = target_xy - cue_xy
            for oid, ob in balls.items():
                if oid == "cue" or getattr(ob.state, "s", 0) == 4:
                    continue
                if oid in own_set:
                    continue
                pos_xy = np.array(ob.state.rvw[0], dtype=float)[:2]
                dist, t, side = segment_distance(cue_xy, target_xy, pos_xy)
                if 0.0 < t < 1.0 and dist < clearance:
                    along = float(np.linalg.norm(line_vec) * t)
                    blockers.append({
                        "id": oid,
                        "along": along,
                        "dist_to_line": dist,
                        "side": side
                    })
            return sorted(blockers, key=lambda x: x["along"])

        def has_blocker_on_segment(p_start: np.ndarray, p_end: np.ndarray, allow_ids=None) -> bool:
            allow_ids = allow_ids or set()
            for oid, ob in balls.items():
                if oid == "cue" or getattr(ob.state, "s", 0) == 4:
                    continue
                if oid in allow_ids:
                    continue
                pos_xy = np.array(ob.state.rvw[0], dtype=float)[:2]
                dist, t, _ = segment_distance(p_start, p_end, pos_xy)
                if 0.0 < t < 1.0 and dist < clearance:
                    return True
            return False

        def pocket_radius(pocket):
            r = getattr(pocket, "r", None)
            if r is None:
                r = getattr(pocket, "radius", None)
            if r is None:
                params = getattr(pocket, "params", None)
                if params is not None:
                    r = getattr(params, "R", None)
            if r is None:
                r = ball_radius * 2.6
            return float(r)

        def min_clearance_on_segment(p_start, p_end, ignore_ids=None):
            ignore_ids = ignore_ids or set()
            min_dist = float("inf")
            for oid, ob in balls.items():
                if oid in ignore_ids or getattr(ob.state, "s", 0) == 4:
                    continue
                pos_xy = np.array(ob.state.rvw[0], dtype=float)[:2]
                dist, t, _ = segment_distance(p_start, p_end, pos_xy)
                if 0.0 < t < 1.0:
                    min_dist = min(min_dist, dist)
            if min_dist == float("inf"):
                return 999.0
            return float(min_dist)

        def compute_side_hint(aim_xy, bid):
            score = 0.0
            for oid, ob in balls.items():
                if oid in ("cue", bid) or getattr(ob.state, "s", 0) == 4:
                    continue
                pos_xy = np.array(ob.state.rvw[0], dtype=float)[:2]
                dist, t, side = segment_distance(cue_xy, aim_xy, pos_xy)
                if 0.0 < t < 1.0 and dist < ball_radius * 3.0:
                    score -= side * (ball_radius / max(dist, ball_radius))
            if abs(score) < 0.2:
                return 0.0
            return math.copysign(1.0, score)

        def score_geom_pair(bid, ball_xy, pocket_xy, pocket, aim_xy, cue_pos=None):
            cue_ref = cue_xy if cue_pos is None else cue_pos
            vec_bp = pocket_xy - ball_xy
            dist_bp = float(np.linalg.norm(vec_bp))
            vec_cb = ball_xy - cue_ref
            dist_cb = float(np.linalg.norm(vec_cb))
            if dist_bp < 1e-6 or dist_cb < 1e-6:
                return 0.0, 180.0, 0.0, 0.0, 0.0, dist_bp, dist_cb
            cos_cut = float(np.dot(vec_cb, vec_bp) / (dist_cb * dist_bp))
            cos_cut = max(-1.0, min(1.0, cos_cut))
            cut_deg = float(math.degrees(math.acos(cos_cut)))
            clear_cue = min_clearance_on_segment(cue_ref, aim_xy, ignore_ids={"cue", bid})
            clear_obj = min_clearance_on_segment(ball_xy, pocket_xy, ignore_ids={"cue", bid})
            p_rad = pocket_radius(pocket)
            mouth = max(1e-4, p_rad - ball_radius * 0.9)
            window_deg = float(math.degrees(math.atan2(mouth, dist_bp)))
            f_cut = max(0.0, 1.0 - cut_deg / 60.0)
            f_clear = min(1.0, min(clear_cue, clear_obj) / (2.6 * ball_radius))
            f_window = min(1.0, window_deg / 4.0)
            f_dist = 1.0 / (1.0 + dist_bp / (0.6 * table_scale))
            score = base_cost_max * (0.4 * f_cut + 0.3 * f_clear + 0.2 * f_window + 0.1 * f_dist)
            score = float(max(0.0, min(base_cost_max, score)))
            return score, cut_deg, window_deg, clear_cue, clear_obj, dist_bp, dist_cb

        def _reflect_point(point_xy, axis: str, c_val: float):
            if axis == "x":
                return np.array([2.0 * c_val - point_xy[0], point_xy[1]], dtype=float)
            return np.array([point_xy[0], 2.0 * c_val - point_xy[1]], dtype=float)

        # --- basic/pro profiles ---
        spin_profiles = [
            {"name": "plain", "a": 0.0, "b": 0.0, "theta": (0.0, 6.0), "v": (0.9, 1.1)},
            {"name": "follow_soft", "a": 0.0, "b": 0.15, "theta": (0.0, 7.0), "v": (0.95, 1.1)},
            {"name": "draw_soft", "a": 0.0, "b": -0.15, "theta": (0.0, 8.0), "v": (0.95, 1.1)},
            {"name": "side_left", "a": -0.15, "b": 0.0, "theta": (0.0, 7.0), "v": (0.95, 1.1)},
            {"name": "side_right", "a": 0.15, "b": 0.0, "theta": (0.0, 7.0), "v": (0.95, 1.1)},
        ]
        plain_prof = spin_profiles[0]
        follow_draw_profiles = [spin_profiles[1], spin_profiles[2]]
        side_profiles = [spin_profiles[3], spin_profiles[4]]

        def build_geom_pro_actions_for_pair(pair, per_pair=3):
            actions = []
            base_cost = pair["pair_score"]
            if base_cost <= 0.0:
                return actions
            phi = pair["base_phi"]
            dist_ca = pair["dist_ca"]
            v_base = float(np.clip(1.5 + dist_ca * 1.5, 1.0, 6.0))

            offsets = [-0.8, 0.8]
            for off in offsets:
                phi_pro = (phi + off) % 360.0
                theta = float(np.random.uniform(0.0, 3.0))
                a_val = float(np.random.uniform(-0.02, 0.02))
                b_val = float(np.random.uniform(-0.02, 0.02))
                action = {
                    "V0": float(np.clip(v_base, 0.8, 6.5)),
                    "phi": float(phi_pro),
                    "theta": theta,
                    "a": a_val,
                    "b": b_val,
                    "candidate_type": "geom_pro",
                    "target_bid": pair["bid"],
                }
                spin_mag = abs(a_val) + abs(b_val)
                adj = (abs(off) * 2.0 + spin_mag * 15.0 + theta * 0.3) * cost_scale
                actions.append((base_cost - adj, action))

            if per_pair >= 3 and base_cost > 0.0:
                b_val = float(random.choice([-0.06, 0.06]))
                theta = float(np.random.uniform(0.0, 4.0))
                action = {
                    "V0": float(np.clip(v_base, 0.8, 6.5)),
                    "phi": float(phi),
                    "theta": theta,
                    "a": 0.0,
                    "b": b_val,
                    "candidate_type": "geom_pro",
                    "target_bid": pair["bid"],
                }
                adj = (1.5 + abs(b_val) * 10.0 + theta * 0.3) * cost_scale
                actions.append((base_cost - adj, action))

            return actions[:per_pair]

        def build_geom_basic_actions_for_pair(pair, max_actions=2):
            actions = []
            base_cost = pair["pair_score"]
            if base_cost <= 0.0:
                return actions
            phi = pair["base_phi"]
            dist_ca = pair["dist_ca"]
            v_base = float(np.clip(1.5 + dist_ca * 1.5, 1.0, 6.0))

            # basic: one random offset
            off = random.choice(angle_offsets)
            phi_basic = (phi + off) % 360.0
            a_val = float(np.random.uniform(-0.12, 0.12))
            b_val = float(np.random.uniform(-0.12, 0.12))
            action = {
                "V0": float(np.random.uniform(v_base * 0.9, v_base * 1.1)),
                "phi": float(phi_basic),
                "theta": float(np.random.uniform(0.0, 8.0)),
                "a": a_val,
                "b": b_val,
                "candidate_type": "geom_basic",
                "target_bid": pair["bid"],
            }
            actions.append((base_cost, action))

            if max_actions >= 2:
                prof = random.choice([plain_prof] + follow_draw_profiles + side_profiles)
                vmin, vmax = prof["v"]
                tmin, tmax = prof["theta"]
                a_center = prof["a"]
                b_center = prof["b"]
                phi_t = (phi + random.choice(angle_offsets)) % 360.0
                action = {
                    "V0": float(np.clip(np.random.uniform(v_base * vmin, v_base * vmax), 0.5, 8.0)),
                    "phi": float(phi_t),
                    "theta": float(np.random.uniform(tmin, tmax)),
                    "a": float(np.random.uniform(a_center - 0.04, a_center + 0.04)),
                    "b": float(np.random.uniform(b_center - 0.04, b_center + 0.04)),
                    "candidate_type": "geom_basic",
                    "target_bid": pair["bid"],
                }
                extra_cost = 0.0
                if prof["name"] in ("follow_soft", "draw_soft"):
                    extra_cost = 0.08
                elif prof["name"] in ("side_left", "side_right"):
                    extra_cost = 0.12
                actions.append((base_cost - extra_cost * cost_scale, action))

            return actions[:max_actions]

        # --- detour (basic/pro) ---

        def detour_basic_actions_for_pair(pair, max_actions=2):
            actions = []
            C = cue_xy
            G = pair["aim_xy"]
            v = G - C
            L = float(np.linalg.norm(v))
            if L < 1e-4:
                return actions
            e_para = v / L
            e_perp = np.array([-e_para[1], e_para[0]], dtype=float)
            t_min = 0.15 * L
            t_max = 0.85 * L
            s_max = min(max(2.4 * ball_radius, 0.25 * L), 0.8 * L)

            obstacles = []
            for oid, ob in balls.items():
                if oid in ("cue", pair["bid"]):
                    continue
                if getattr(ob.state, "s", 0) == 4:
                    continue
                pos_xy = np.array(ob.state.rvw[0], dtype=float)[:2]
                obstacles.append((oid, pos_xy))

            prefer_side = float(pair.get("side_hint", 0.0))
            num_samples = 60
            best_list = []

            for _ in range(num_samples):
                t = float(np.random.uniform(t_min, t_max))
                if prefer_side == 0.0:
                    side = random.choice([-1.0, 1.0])
                else:
                    side = prefer_side if random.random() < 0.7 else -prefer_side
                s_abs = float(np.random.uniform(0.4 * ball_radius, s_max))
                s = side * s_abs
                X = C + t * e_para + s * e_perp
                path_len = float(np.linalg.norm(X - C) + np.linalg.norm(G - X))
                min_clear = float("inf")
                penalty = 0.0
                for _, O in obstacles:
                    d1, tau1, _ = segment_distance(C, X, O)
                    d2, tau2, _ = segment_distance(X, G, O)
                    d_candidates = []
                    if 0.0 < tau1 < 1.0:
                        d_candidates.append(d1)
                    if 0.0 < tau2 < 1.0:
                        d_candidates.append(d2)
                    if not d_candidates:
                        continue
                    d = min(d_candidates)
                    min_clear = min(min_clear, d)
                    if d < ball_radius * 2.2:
                        penalty += 1.0 + (ball_radius * 2.2 - d) / max(ball_radius * 2.2, 1e-4)
                if min_clear == float("inf"):
                    min_clear = 999.0
                obj = path_len + 200.0 * penalty
                best_list.append((obj, min_clear, t, s, X, path_len))

            best_list.sort(key=lambda x: x[0])
            best_list = best_list[:max_actions]

            for _, min_clear, _, s_val, X, path_len in best_list:
                diff = X - C
                s_comp = float(np.dot(diff, e_perp))
                curvature = abs(s_comp) / max(L, 1e-4)
                length_ratio = path_len / max(L, 1e-4)
                f_clear = min(1.0, min_clear / (2.8 * ball_radius))
                f_len = 1.0 / (1.0 + path_len / (1.1 * max(L, 0.4)))
                f_curve = max(0.0, 1.0 - curvature / 0.9)
                base_cost = base_cost_max * f_clear * f_len * f_curve

                if curvature < 0.15:
                    a_min, a_max = 0.04, 0.10
                    theta_min, theta_max = 2.5, 6.0
                elif curvature < 0.3:
                    a_min, a_max = 0.08, 0.16
                    theta_min, theta_max = 4.0, 9.0
                else:
                    a_min, a_max = 0.12, 0.22
                    theta_min, theta_max = 6.0, 12.0

                side_spin_sign = math.copysign(1.0, s_comp) if abs(s_comp) > 1e-4 else random.choice([-1.0, 1.0])
                v0_scale = float(np.clip(0.9 + 0.3 * (length_ratio - 1.0), 0.8, 1.4))
                detour_phi = float(math.degrees(math.atan2(diff[1], diff[0])) % 360.0)
                detour_action = {
                    "V0": float(np.clip(pair["base_v0"] * v0_scale, 0.8, 7.0)),
                    "phi": detour_phi,
                    "theta": float(np.random.uniform(theta_min, theta_max)),
                    "a": float(np.random.uniform(a_min, a_max) * side_spin_sign),
                    "b": float(np.random.uniform(-0.05, 0.05)),
                    "candidate_type": "detour_basic",
                    "target_bid": pair["bid"],
                }
                actions.append((base_cost, detour_action))

            return actions

        def detour_pro_actions_for_pair(pair, max_actions=2):
            actions = []
            ball_xy = pair["ball_xy"]
            pocket_xy = pair["pocket_xy"]
            bid = pair["bid"]
            for axis, c_val in (("x", 0.0), ("x", float(table.w)), ("y", 0.0), ("y", float(table.l))):
                if axis == "x":
                    if abs(pocket_xy[0] - c_val) < 1e-4:
                        continue
                else:
                    if abs(pocket_xy[1] - c_val) < 1e-4:
                        continue
                mirror = _reflect_point(pocket_xy, axis, c_val)
                dir_vec = mirror - ball_xy
                dir_len = float(np.linalg.norm(dir_vec))
                if dir_len < 1e-6:
                    continue
                if axis == "x":
                    if abs(dir_vec[0]) < 1e-9:
                        continue
                    t = (c_val - ball_xy[0]) / dir_vec[0]
                    if t <= 0.0:
                        continue
                    y = ball_xy[1] + dir_vec[1] * t
                    if not (0.0 <= y <= table.l):
                        continue
                    bank_pt = np.array([c_val, y], dtype=float)
                else:
                    if abs(dir_vec[1]) < 1e-9:
                        continue
                    t = (c_val - ball_xy[1]) / dir_vec[1]
                    if t <= 0.0:
                        continue
                    x = ball_xy[0] + dir_vec[0] * t
                    if not (0.0 <= x <= table.w):
                        continue
                    bank_pt = np.array([x, c_val], dtype=float)

                bank_dir = bank_pt - ball_xy
                dist_bb = float(np.linalg.norm(bank_dir))
                if dist_bb < 2.0 * ball_radius:
                    continue
                bank_unit = bank_dir / dist_bb
                aim_xy = ball_xy - bank_unit * (2.0 * ball_radius)
                dist_ca = float(np.linalg.norm(aim_xy - cue_xy))
                if dist_ca < 1e-6:
                    continue
                dist_bp = float(np.linalg.norm(pocket_xy - bank_pt))
                if dist_bp < 2.0 * ball_radius:
                    continue

                clear_cue = min_clearance_on_segment(cue_xy, aim_xy, ignore_ids={"cue", bid})
                clear_obj = min_clearance_on_segment(ball_xy, bank_pt, ignore_ids={"cue", bid})
                clear_bank = min_clearance_on_segment(bank_pt, pocket_xy, ignore_ids={"cue", bid})
                min_clear = min(clear_cue, clear_obj, clear_bank)
                if min_clear < ball_radius * 2.6:
                    continue

                bank_to_pocket = pocket_xy - bank_pt
                btp_len = float(np.linalg.norm(bank_to_pocket))
                if btp_len < 1e-6:
                    continue
                cos_turn = float(np.dot(bank_dir / dist_bb, bank_to_pocket / btp_len))
                cos_turn = max(-1.0, min(1.0, cos_turn))
                turn_angle = float(math.degrees(math.acos(cos_turn)))
                f_turn = max(0.0, math.cos(math.radians(turn_angle)))
                total_len = dist_ca + dist_bb + dist_bp
                f_len = 1.0 / (1.0 + total_len / (1.1 * table_scale))
                f_clear = min(1.0, min_clear / (3.2 * ball_radius))
                base_cost = base_cost_max * f_turn * f_len * f_clear
                if base_cost <= 0.0:
                    continue

                v0 = float(np.clip(1.2 + total_len * 1.1, 1.0, 7.0))
                detour_phi = float(math.degrees(math.atan2(aim_xy[1] - cue_xy[1], aim_xy[0] - cue_xy[0])) % 360.0)
                detour_action = {
                    "V0": v0,
                    "phi": detour_phi,
                    "theta": float(np.random.uniform(0.0, 4.0)),
                    "a": float(np.random.uniform(-0.03, 0.03)),
                    "b": float(np.random.uniform(-0.03, 0.03)),
                    "candidate_type": "detour_pro",
                    "target_bid": bid,
                }
                actions.append((base_cost, detour_action))

            actions.sort(key=lambda x: x[0], reverse=True)
            return actions[:max_actions]

        # --- collect pair infos ---
        pair_infos = []
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
                dist_bp = float(np.linalg.norm(vec_bp))
                if dist_bp < 1e-6:
                    continue
                dir_bp = vec_bp / dist_bp
                aim_xy = ball_xy - dir_bp * (ball_radius * 2.0)
                vec_ca = aim_xy - cue_xy
                dist_ca = float(np.linalg.norm(vec_ca))
                if dist_ca < 1e-6:
                    continue
                score, cut_deg, window_deg, clear_cue, clear_obj, dist_bp, dist_cb = score_geom_pair(
                    bid, ball_xy, pocket_xy, pocket, aim_xy
                )
                base_phi = float(math.degrees(math.atan2(vec_ca[1], vec_ca[0])) % 360.0)
                pair_infos.append({
                    "bid": bid,
                    "ball_xy": ball_xy,
                    "pocket_xy": pocket_xy,
                    "pocket": pocket,
                    "aim_xy": aim_xy,
                    "dist_ca": dist_ca,
                    "dist_bp": dist_bp,
                    "dist_cb": dist_cb,
                    "base_phi": base_phi,
                    "base_v0": float(np.clip(1.5 + dist_ca * 1.5, 1.0, 6.0)),
                    "pair_score": score,
                    "cut_deg": cut_deg,
                    "window_deg": window_deg,
                    "clear_cue": clear_cue,
                    "clear_obj": clear_obj,
                    "side_hint": compute_side_hint(aim_xy, bid),
                })

        if not pair_infos:
            return []

        pair_infos.sort(key=lambda x: x["pair_score"], reverse=True)

        target_detour = max(0, int(round(num_candidates * 0.2)))
        target_geom = max(1, num_candidates - target_detour)
        geom_pro_target = int(round(target_geom * 0.6))
        geom_basic_target = max(0, target_geom - geom_pro_target)

        pro_pairs_needed = int(math.ceil(geom_pro_target / 3.0)) if geom_pro_target > 0 else 0
        geom_pro_pairs = pair_infos[:pro_pairs_needed]
        geom_basic_pairs = pair_infos[pro_pairs_needed:]

        geom_pro_candidates = []
        for pair in geom_pro_pairs:
            geom_pro_candidates.extend(build_geom_pro_actions_for_pair(pair, per_pair=3))

        geom_basic_candidates = []
        for pair in geom_basic_pairs:
            geom_basic_candidates.extend(build_geom_basic_actions_for_pair(pair, max_actions=2))

        geom_pro_candidates.sort(key=lambda x: x[0], reverse=True)
        geom_basic_candidates.sort(key=lambda x: x[0], reverse=True)

        geom_actions = []
        if geom_pro_target > 0:
            geom_actions.extend(geom_pro_candidates[:geom_pro_target])
        if geom_basic_target > 0:
            geom_actions.extend(geom_basic_candidates[:geom_basic_target])

        if len(geom_actions) < target_geom:
            leftover = geom_pro_candidates[geom_pro_target:] + geom_basic_candidates[geom_basic_target:]
            leftover.sort(key=lambda x: x[0], reverse=True)
            for item in leftover:
                if len(geom_actions) >= target_geom:
                    break
                geom_actions.append(item)

        detour_basic_candidates = []
        detour_pro_candidates = []
        if target_detour > 0:
            for pair in pair_infos:
                detour_basic_candidates.extend(detour_basic_actions_for_pair(pair, max_actions=2))
                detour_pro_candidates.extend(detour_pro_actions_for_pair(pair, max_actions=2))

        detour_basic_candidates.sort(key=lambda x: x[0], reverse=True)
        detour_pro_candidates.sort(key=lambda x: x[0], reverse=True)

        detour_actions = []
        if target_detour > 0:
            detour_basic_quota = int(round(target_detour * 0.4))
            detour_basic_quota = max(0, min(detour_basic_quota, target_detour))
            detour_pro_quota = max(0, target_detour - detour_basic_quota)
            detour_actions.extend(detour_pro_candidates[:detour_pro_quota])
            detour_actions.extend(detour_basic_candidates[:detour_basic_quota])
            if len(detour_actions) < target_detour:
                remain = target_detour - len(detour_actions)
                extra = detour_pro_candidates[detour_pro_quota:] + detour_basic_candidates[detour_basic_quota:]
                extra.sort(key=lambda x: x[0], reverse=True)
                detour_actions.extend(extra[:remain])

        final_candidates = [act for _, act in geom_actions] + [act for _, act in detour_actions]
        used = len(final_candidates)

        if used < num_candidates:
            geom_leftover = geom_pro_candidates[geom_pro_target:] + geom_basic_candidates[geom_basic_target:]
            geom_leftover.sort(key=lambda x: x[0], reverse=True)
            for _, act in geom_leftover:
                if used >= num_candidates:
                    break
                final_candidates.append(act)
                used += 1

        if used < num_candidates:
            filler_offsets = [-1.5, 1.5, -2.5, 2.5]
            fillers = []
            for pair in pair_infos:
                base_cost = pair["pair_score"]
                if base_cost <= 0.0:
                    continue
                phi = pair["base_phi"]
                dist_ca = pair["dist_ca"]
                v_base = float(np.clip(1.5 + dist_ca * 1.5, 1.0, 6.0))
                for off in filler_offsets:
                    phi_fill = (phi + off) % 360.0
                    action = {
                        "V0": float(np.clip(v_base, 0.8, 6.5)),
                        "phi": float(phi_fill),
                        "theta": float(np.random.uniform(0.0, 2.5)),
                        "a": float(np.random.uniform(-0.015, 0.015)),
                        "b": float(np.random.uniform(-0.015, 0.015)),
                        "candidate_type": "geom_basic",
                        "target_bid": pair["bid"],
                    }
                    adj = abs(off) * 2.2 * cost_scale
                    fillers.append((base_cost - adj, action))
            fillers.sort(key=lambda x: x[0], reverse=True)
            for _, act in fillers:
                if used >= num_candidates:
                    break
                final_candidates.append(act)
                used += 1

        if len(final_candidates) > num_candidates:
            final_candidates = final_candidates[:num_candidates]

        return final_candidates
