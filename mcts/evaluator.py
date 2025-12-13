"""
evaluator.py - 局面评估器

评估台球局面的得分，用于MCTS的节点评估
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class StateEvaluator:
    """
    台球局面评估器
    
    评估维度：
    1. 进球情况（最重要）
    2. 走位质量
    3. 袋口机会
    4. 防守安全性
    """
    
    # 固定的袋口位置
    POCKET_POSITIONS = {
        'lb': np.array([-0.03, -0.03]),
        'lc': np.array([-0.069, 0.99]),
        'lt': np.array([-0.03, 2.01]),
        'rb': np.array([1.02, -0.03]),
        'rc': np.array([1.069, 0.99]),
        'rt': np.array([1.02, 2.01]),
    }
    
    def __init__(self, table_w: float = 0.99, table_l: float = 1.98):
        self.table_w = table_w
        self.table_l = table_l
        self.ball_radius = 0.02625
        self.max_dist = np.sqrt(table_w**2 + table_l**2)
    
    def evaluate(self,
                 balls_before: Dict,
                 balls_after: Dict,
                 shot_result: Dict,
                 my_targets: List[str],
                 enemy_targets: List[str],
                 game_done: bool,
                 winner: Optional[str],
                 my_player: str) -> float:
        """
        评估一次击球的得分
        
        Args:
            balls_before: 击球前的球状态
            balls_after: 击球后的球状态
            shot_result: 击球结果
            my_targets: 己方目标球
            enemy_targets: 对方目标球
            game_done: 游戏是否结束
            winner: 胜者
            my_player: 己方标识
        
        Returns:
            float: 评估得分 [-1000, +1000]
        """
        score = 0.0
        
        # 1. 终局评估（最高优先级）
        if game_done:
            if winner == my_player:
                return 1000.0  # 胜利
            elif winner == 'SAME':
                return -100.0  # 平局
            else:
                return -1000.0  # 失败
        
        # 2. 进球评估
        own_pocketed = shot_result.get('ME_INTO_POCKET', [])
        enemy_pocketed = shot_result.get('ENEMY_INTO_POCKET', [])
        
        # 己方进球：大奖励
        score += len(own_pocketed) * 100.0
        
        # 连续进球额外奖励
        if len(own_pocketed) >= 2:
            score += (len(own_pocketed) - 1) * 50.0
        
        # 帮对方进球：惩罚
        score -= len(enemy_pocketed) * 30.0
        
        # 3. 犯规评估
        if shot_result.get('WHITE_BALL_INTO_POCKET', False):
            score -= 80.0
        if shot_result.get('FOUL_FIRST_HIT', False):
            score -= 40.0
        if shot_result.get('NO_HIT', False):
            score -= 50.0
        if shot_result.get('NO_POCKET_NO_RAIL', False):
            score -= 20.0
        
        # 4. 走位评估（只有进球后才重要）
        if len(own_pocketed) > 0:
            position_score = self._evaluate_position(
                balls_after, my_targets
            )
            score += position_score * 0.5
        
        # 5. 机会评估（如果没进球）
        if len(own_pocketed) == 0:
            opportunity_score = self._evaluate_opportunities(
                balls_after, my_targets
            )
            score += opportunity_score * 0.3
        
        return score
    
    def evaluate_state(self,
                       balls: Dict,
                       my_targets: List[str],
                       my_player: str) -> float:
        """
        评估当前局面的静态得分
        
        用于MCTS rollout结束时的评估
        
        Args:
            balls: 当前球状态
            my_targets: 己方目标球
            my_player: 己方标识
        
        Returns:
            float: 局面得分 [-500, +500]
        """
        score = 0.0
        
        # 1. 己方剩余球数（越少越好）
        my_remaining = sum(1 for bid in my_targets 
                          if bid != '8' and balls[bid].state.s != 4)
        score += (7 - my_remaining) * 50.0
        
        # 2. 是否可以打黑8
        if my_remaining == 0 and balls['8'].state.s != 4:
            score += 100.0
        
        # 3. 机会评估
        opportunity_score = self._evaluate_opportunities(balls, my_targets)
        score += opportunity_score
        
        return score
    
    def _evaluate_position(self, balls: Dict, my_targets: List[str]) -> float:
        """评估走位质量"""
        cue_pos = self._get_ball_pos(balls['cue'])[:2]
        
        # 获取剩余的己方球
        remaining = [bid for bid in my_targets 
                     if bid != '8' and balls[bid].state.s != 4]
        
        if len(remaining) == 0:
            # 只剩黑8
            if balls['8'].state.s != 4:
                remaining = ['8']
            else:
                return 50.0  # 全部清台
        
        if len(remaining) == 0:
            return 0.0
        
        # 找最佳击球机会
        best_score = 0.0
        for bid in remaining:
            ball_pos = self._get_ball_pos(balls[bid])[:2]
            score = self._evaluate_shot_opportunity(cue_pos, ball_pos, balls, bid)
            best_score = max(best_score, score)
        
        return best_score
    
    def _evaluate_opportunities(self, balls: Dict, my_targets: List[str]) -> float:
        """评估当前局面的击球机会"""
        cue_pos = self._get_ball_pos(balls['cue'])[:2]
        
        remaining = [bid for bid in my_targets 
                     if bid != '8' and balls[bid].state.s != 4]
        
        if len(remaining) == 0:
            if balls['8'].state.s != 4:
                remaining = ['8']
            else:
                return 100.0
        
        if len(remaining) == 0:
            return 0.0
        
        # 评估每个球的击球机会
        total_score = 0.0
        for bid in remaining:
            ball_pos = self._get_ball_pos(balls[bid])[:2]
            score = self._evaluate_shot_opportunity(cue_pos, ball_pos, balls, bid)
            total_score += score
        
        return total_score / len(remaining)
    
    def _evaluate_shot_opportunity(self,
                                    cue_pos: np.ndarray,
                                    ball_pos: np.ndarray,
                                    balls: Dict,
                                    ball_id: str) -> float:
        """评估单个击球机会"""
        score = 0.0
        
        # 1. 路径是否清晰
        path_clear = self._is_path_clear(cue_pos, ball_pos, balls, ball_id)
        if not path_clear:
            return -20.0  # 被遮挡
        
        # 2. 距离评分
        dist = np.linalg.norm(ball_pos - cue_pos)
        if dist < 0.3:
            score += 30.0
        elif dist < 0.6:
            score += 20.0
        elif dist < 1.0:
            score += 10.0
        else:
            score += 5.0
        
        # 3. 进袋角度评分
        best_pocket_score = 0.0
        for pocket_pos in self.POCKET_POSITIONS.values():
            angle_score = self._evaluate_pocket_angle(cue_pos, ball_pos, pocket_pos)
            pocket_dist = np.linalg.norm(ball_pos - pocket_pos)
            
            if pocket_dist < 0.3:
                dist_factor = 1.0
            elif pocket_dist < 0.6:
                dist_factor = 0.8
            elif pocket_dist < 1.0:
                dist_factor = 0.5
            else:
                dist_factor = 0.2
            
            pocket_score = angle_score * dist_factor * 40.0
            best_pocket_score = max(best_pocket_score, pocket_score)
        
        score += best_pocket_score
        
        return score
    
    def _evaluate_pocket_angle(self,
                               cue_pos: np.ndarray,
                               ball_pos: np.ndarray,
                               pocket_pos: np.ndarray) -> float:
        """评估进袋角度 [0, 1]"""
        cue_to_ball = ball_pos - cue_pos
        ball_to_pocket = pocket_pos - ball_pos
        
        cue_dist = np.linalg.norm(cue_to_ball)
        pocket_dist = np.linalg.norm(ball_to_pocket)
        
        if cue_dist < 0.01 or pocket_dist < 0.01:
            return 0.5
        
        cos_angle = np.dot(cue_to_ball / cue_dist, ball_to_pocket / pocket_dist)
        
        # 角度越接近直线（cos接近1），得分越高
        return max(0, (cos_angle + 1) / 2)
    
    def _is_path_clear(self,
                       start: np.ndarray,
                       end: np.ndarray,
                       balls: Dict,
                       exclude_id: str) -> bool:
        """检查路径是否清晰"""
        direction = end - start
        length = np.linalg.norm(direction)
        
        if length < 0.01:
            return True
        
        direction = direction / length
        
        for bid, ball in balls.items():
            if bid in ['cue', exclude_id] or ball.state.s == 4:
                continue
            
            ball_pos = self._get_ball_pos(ball)[:2]
            to_ball = ball_pos - start
            proj_length = np.dot(to_ball, direction)
            
            if proj_length < 0 or proj_length > length:
                continue
            
            perp_dist = np.linalg.norm(to_ball - proj_length * direction)
            
            if perp_dist < self.ball_radius * 2.5:
                return False
        
        return True
    
    def _get_ball_pos(self, ball) -> np.ndarray:
        return ball.state.rvw[0]

