# MCTS core for billiards

import copy
import math
import random
import warnings
from typing import Dict, List, Optional, Callable

import numpy as np
import pooltool as pt


class MCTSNode:
    def __init__(self, balls, table, my_targets, parent=None, action=None, depth=0):
        self.balls = balls
        self.table = table
        self.my_targets = list(my_targets)
        self.parent = parent
        self.action = action
        self.depth = depth
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value = 0.0

    def is_leaf(self):
        return len(self.children) == 0

    def ucb_score(self, c=1.4):
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + c * math.sqrt(math.log(self.parent.visits + 1) / self.visits)


class MCTSSolver:
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
