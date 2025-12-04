import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import os
from datetime import datetime
import random

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from mcts import MCTSSolver


def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """Score a shot result (same as original agent)."""
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
    if foul_no_rail:
        score -= 30
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10
    return score


class Agent:
    def __init__(self):
        pass

    def decision(self, *args, **kwargs):
        pass

    def _random_action(self):
        # 避免极小速度导致物理求解退化，略微抬高下限
        v0_min = 0.6
        return {
            'V0': round(random.uniform(v0_min, 8.0), 2),
            'phi': round(random.uniform(0, 360), 2),
            'theta': round(random.uniform(0, 90), 2),
            'a': round(random.uniform(-0.5, 0.5), 3),
            'b': round(random.uniform(-0.5, 0.5), 3)
        }


class BasicAgent(Agent):
    def __init__(self, target_balls=None):
        super().__init__()
        self.pbounds = {
            'V0': (0.5, 8.0),
            'phi': (0, 360),
            'theta': (0, 90),
            'a': (-0.5, 0.5),
            'b': (-0.5, 0.5)
        }
        self.INITIAL_SEARCH = 20
        self.OPT_SEARCH = 10
        self.ALPHA = 1e-2
        self.noise_std = {'V0': 0.1, 'phi': 0.1, 'theta': 0.1, 'a': 0.003, 'b': 0.003}
        self.enable_noise = False
        print("BasicAgent (Smart, pooltool-native) 已初始化")

    def _create_optimizer(self, reward_function, seed):
        gpr = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=self.ALPHA, n_restarts_optimizer=10, random_state=seed)
        bounds_transformer = SequentialDomainReductionTransformer(gamma_osc=0.8, gamma_pan=1.0)
        optimizer = BayesianOptimization(f=reward_function, pbounds=self.pbounds, random_state=seed, verbose=0, bounds_transformer=bounds_transformer)
        optimizer._gp = gpr
        return optimizer

    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None:
            print("[BasicAgent] 未收到balls，使用随机动作")
            return self._random_action()
        try:
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                print("[BasicAgent] 目标球清空，切换为8号球")

            def reward_fn_wrapper(V0, phi, theta, a, b):
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")
                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                try:
                    if self.enable_noise:
                        V0_noisy = np.clip(V0 + np.random.normal(0, self.noise_std['V0']), 0.5, 8.0)
                        phi_noisy = (phi + np.random.normal(0, self.noise_std['phi'])) % 360
                        theta_noisy = np.clip(theta + np.random.normal(0, self.noise_std['theta']), 0, 90)
                        a_noisy = np.clip(a + np.random.normal(0, self.noise_std['a']), -0.5, 0.5)
                        b_noisy = np.clip(b + np.random.normal(0, self.noise_std['b']), -0.5, 0.5)
                        shot.cue.set_state(V0=V0_noisy, phi=phi_noisy, theta=theta_noisy, a=a_noisy, b=b_noisy)
                    else:
                        shot.cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
                    pt.simulate(shot, inplace=True)
                except Exception:
                    return -500
                return analyze_shot_for_reward(shot=shot, last_state=last_state_snapshot, player_targets=my_targets)

            print(f"[BasicAgent] 搜索最佳击球，targets: {my_targets}")
            seed = np.random.randint(1e6)
            optimizer = self._create_optimizer(reward_fn_wrapper, seed)
            optimizer.maximize(init_points=self.INITIAL_SEARCH, n_iter=self.OPT_SEARCH)
            best_result = optimizer.max
            best_params = best_result['params']
            best_score = best_result['target']
            if best_score < 10:
                print(f"[BasicAgent] 未找到好解，score={best_score:.2f}，随机动作")
                return self._random_action()
            action = {k: float(best_params[k]) for k in ['V0', 'phi', 'theta', 'a', 'b']}
            print(f"[BasicAgent] 决策 score={best_score:.2f}: V0={action['V0']:.2f}, phi={action['phi']:.2f}, theta={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}")
            return action
        except Exception as e:
            print(f"[BasicAgent] 决策错误，使用随机动作。原因: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()


class NewAgent(Agent):
    """MCTS-based agent (UCT + short rollout)."""

    def __init__(
        self,
        num_candidates: int = 64,
        num_simulations: int = 100,
        max_depth: int = 2,
        exploration_c: float = 1.4,
        rollout_per_leaf: int = 2,
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
            reward_fn=analyze_shot_for_reward,
            num_simulations=num_simulations,
            max_depth=max_depth,
            exploration_c=exploration_c,
            rollout_per_leaf=rollout_per_leaf,
            enable_noise=True,
            noise_std=self.noise_std,
        )

    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None or my_targets is None or table is None:
            return self._random_action()
        try:
            remaining_own = [bid for bid in my_targets if bid in balls and getattr(balls[bid].state, "s", 0) != 4 and bid != "8"]
            if len(remaining_own) == 0 and "8" not in my_targets:
                my_targets = ["8"]
                print("[MCTS NewAgent] 目标球已清空，切换为黑8")
            remaining_report = [bid for bid in my_targets if bid in balls and getattr(balls[bid].state, "s", 0) != 4]
            print(f"[MCTS NewAgent] 剩余目标球: {remaining_report}")
            candidates = self._generate_candidate_actions(balls, my_targets, table, self.num_candidates)
            if not candidates:
                return self._random_action()
            action = self.mcts.search(balls=balls, my_targets=my_targets, table=table, candidate_actions=candidates)
            if action is None:
                return self._random_action()
            print(f"[MCTS NewAgent] action: V0={action['V0']:.2f}, phi={action['phi']:.2f}, theta={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}")
            return action
        except Exception as e:
            print(f"[MCTS NewAgent] error, fallback to random. Reason: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()

    def _generate_candidate_actions(self, balls, my_targets, table, num_candidates: int):
        candidates = []
        cue_ball = balls.get("cue", None)
        if cue_ball is None:
            return candidates

        cue_pos = np.array(cue_ball.state.rvw[0], dtype=float)
        cue_xy = cue_pos[:2]
        ball_radius = 0.028575  # 2.25 inch in meters
        angle_offsets = [-4.0, -2.0, 0.0, 2.0, 4.0]

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
                for off in angle_offsets:
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
