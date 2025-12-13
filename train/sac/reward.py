"""
reward.py - 奖励计算器

实现 SAC 训练的奖励函数，采用紧凑奖励范围设计：
- 总奖励范围控制在 [-100, +100] 避免 Critic Loss 爆炸
- 只在关键事件（进球、犯规、终局）给予奖励
- 避免每步惩罚累积导致的负奖励主导

奖励设计原则：
1. 稀疏但有意义：不进球不惩罚，进球给奖励
2. 终局奖励适中：胜利 +100，失败 -100
3. 过程奖励辅助：进球、走位提供引导
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class RewardCalculator:
    """
    SAC 奖励计算器 - 紧凑范围版本
    
    奖励范围：[-100, +100]
    
    组成：
    - R_terminal: 终局奖励 [±100]
    - R_pocket: 进球奖励 [0, +30]
    - R_position: 走位奖励 [-2, +5]
    - R_foul: 犯规惩罚 [-15, 0]
    """
    
    def __init__(self,
                 w_terminal: float = 1.0,
                 w_pocket: float = 1.0,
                 w_position: float = 0.3,
                 w_foul: float = 1.0,
                 w_defense: float = 0.0,
                 table_w: float = 0.99,
                 table_l: float = 1.98):
        """
        Args:
            w_terminal: 终局奖励权重
            w_pocket: 进球奖励权重
            w_position: 走位奖励权重
            w_foul: 犯规惩罚权重
            w_defense: 防守奖励权重
            table_w: 球桌宽度
            table_l: 球桌长度
        """
        self.w_terminal = w_terminal
        self.w_pocket = w_pocket
        self.w_position = w_position
        self.w_foul = w_foul
        self.w_defense = w_defense
        
        self.table_w = table_w
        self.table_l = table_l
        
        # 球半径（用于遮挡检查）
        self.ball_radius = 0.02625
    
    def compute_reward(self,
                       shot_result: Dict,
                       balls_before: Dict,
                       balls_after: Dict,
                       my_targets: List[str],
                       enemy_targets: List[str],
                       table,
                       game_done: bool,
                       winner: Optional[str],
                       my_player: str,
                       is_my_shot: bool = True) -> Tuple[float, Dict]:
        """
        计算单步奖励（紧凑范围版本）
        
        设计原则：
        - 不进球时无惩罚（避免累积负奖励）
        - 进球时给予正奖励（引导学习）
        - 犯规时给予适度惩罚（避免恶习）
        - 终局奖励控制在 ±100
        
        Returns:
            Tuple[float, Dict]: (总奖励, 奖励详情)
        """
        reward_details = {}
        total_reward = 0.0
        
        # 计算球数信息
        my_total_balls = len([bid for bid in my_targets if bid != '8'])
        my_remaining_before = sum(1 for bid in my_targets 
                                  if bid != '8' and balls_before[bid].state.s != 4)
        enemy_remaining_before = sum(1 for bid in enemy_targets 
                                     if balls_before[bid].state.s != 4)
        
        is_foul = self._is_foul(shot_result)
        own_pocketed = shot_result.get('ME_INTO_POCKET', [])
        
        # 1. 终局奖励 [±100]
        r_terminal = self._compute_terminal_reward(
            game_done, winner, my_player, is_my_shot, shot_result,
            my_remaining_before, my_total_balls
        )
        reward_details['terminal'] = r_terminal
        total_reward += self.w_terminal * r_terminal
        
        # 如果游戏结束，直接返回终局奖励
        if game_done:
            reward_details['pocket'] = 0.0
            reward_details['position'] = 0.0
            reward_details['foul'] = 0.0
            reward_details['total'] = total_reward
            return total_reward, reward_details
        
        # 2. 进球奖励 [0, +30]（只有正奖励，无惩罚）
        r_pocket = self._compute_pocket_reward(
            shot_result, my_remaining_before, enemy_remaining_before
        )
        reward_details['pocket'] = r_pocket
        total_reward += self.w_pocket * r_pocket
        
        # 3. 走位奖励 [-2, +5]（只在进球时给予，避免噪声）
        if len(own_pocketed) > 0 and not is_foul:
            r_position = self._compute_position_reward(
                balls_after, my_targets, table
            )
            reward_details['position'] = r_position
            total_reward += self.w_position * r_position
        else:
            reward_details['position'] = 0.0
        
        # 4. 犯规惩罚 [-15, 0]
        r_foul = self._compute_foul_penalty(shot_result)
        reward_details['foul'] = r_foul
        total_reward += self.w_foul * r_foul
        
        reward_details['total'] = total_reward
        return total_reward, reward_details
    
    def _compute_terminal_reward(self,
                                  game_done: bool,
                                  winner: Optional[str],
                                  my_player: str,
                                  is_my_shot: bool,
                                  shot_result: Dict,
                                  my_remaining_before: int = 7,
                                  my_total_balls: int = 7) -> float:
        """
        终局奖励（紧凑范围版本）
        
        范围控制在 [-100, +100]：
        - 正常胜利（打进黑8）: +100
        - 对方犯规获胜: +30（运气成分）
        - 我方犯规失败: -100
        - 对方正常获胜: -50
        - 平局: -20 到 -80（根据进度动态调整）
        """
        if not game_done:
            return 0.0
        
        i_won = (winner == my_player)
        is_draw = (winner == 'SAME')
        
        if is_draw:
            # 基于清台进度计算平局惩罚
            if my_total_balls == 0:
                # stage_0：只有黑八，平局 = 未打进黑八
                return -80.0
            
            own_pocketed = my_total_balls - my_remaining_before
            progress = own_pocketed / my_total_balls
            
            # 进度越高，平局惩罚越轻
            if progress >= 1.0:
                # 已清台只剩黑八，但没打进
                return -60.0
            elif progress >= 0.7:
                return -30.0
            elif progress >= 0.5:
                return -40.0
            elif progress >= 0.3:
                return -50.0
            else:
                # 进度很低，严厉惩罚
                return -80.0
        
        if i_won:
            black_ball_pocketed = shot_result.get('BLACK_BALL_INTO_POCKET', False)
            
            if is_my_shot and black_ball_pocketed:
                # 正常胜利：打进黑8
                return 100.0
            elif is_my_shot and not black_ball_pocketed:
                # 超时胜利：运气成分
                return 20.0
            else:
                # 对方犯规获胜
                return 30.0
        else:
            # 我方输了
            if is_my_shot:
                # 我方犯规导致失败
                return -100.0
            else:
                # 对方正常获胜
                return -50.0
    
    def _compute_pocket_reward(self,
                                shot_result: Dict,
                                my_remaining_before: int,
                                enemy_remaining_before: int) -> float:
        """
        进球奖励（紧凑范围版本）
        
        范围控制在 [0, +30]：
        - 只有正奖励，无惩罚（帮对手进球由终局结果体现）
        - 每球 +8，连续进球有小加成
        - 清台最后几球价值更高
        """
        reward = 0.0
        
        own_pocketed = shot_result.get('ME_INTO_POCKET', [])
        n_own = len(own_pocketed)
        
        if n_own > 0:
            # 1. 基础进球奖励：每球 +8
            base_reward = 8.0 * n_own
            
            # 2. 连续进球加成：+3 per extra ball
            combo_bonus = 3.0 * max(0, n_own - 1)
            
            # 3. 清台进度加成：剩余球越少，奖励越高
            progress_bonus = 0.0
            for i in range(n_own):
                remaining = my_remaining_before - i
                if remaining > 0 and remaining <= 3:
                    # 最后3球价值更高
                    progress_bonus += 2.0 * (4 - remaining)
            
            reward = base_reward + combo_bonus + progress_bonus
            
            # 限制在合理范围
            reward = min(reward, 30.0)
        
        return reward
    
    def _compute_position_reward(self,
                                  balls: Dict,
                                  my_targets: List[str],
                                  table) -> float:
        """
        走位奖励（紧凑范围版本）
        
        范围控制在 [-2, +5]：
        - 只在进球后计算（减少噪声）
        - 评估下一杆的击球质量
        """
        cue_pos = balls['cue'].state.rvw[0][:2]
        
        # 获取还在台上的己方目标球（不包括黑8）
        remaining = [bid for bid in my_targets 
                     if balls[bid].state.s != 4 and bid != '8']
        
        if len(remaining) == 0:
            # 检查是否可以打黑8
            if balls['8'].state.s != 4:
                remaining = ['8']
            else:
                return 0.0
        
        if len(remaining) == 0:
            return 0.0
        
        # 计算每个目标球的击球质量
        shot_qualities = []
        for target_id in remaining:
            target_pos = balls[target_id].state.rvw[0][:2]
            quality = self._evaluate_shot_quality(
                cue_pos, target_pos, balls, table, target_id
            )
            shot_qualities.append(quality)
        
        # 取最佳击球质量
        best_quality = max(shot_qualities)
        
        # 映射到奖励范围 [-2, +5]
        # quality 范围是 [0, 1]
        reward = (best_quality - 0.3) * 10.0
        reward = np.clip(reward, -2.0, 5.0)
        
        return reward
    
    def _evaluate_shot_quality(self,
                                cue_pos: np.ndarray,
                                target_pos: np.ndarray,
                                balls: Dict,
                                table,
                                target_id: str) -> float:
        """
        综合评估一个击球的质量 [0, 1]
        """
        quality = 0.0
        
        # 1. 距离评分 [0, 0.3]
        dist = np.linalg.norm(cue_pos - target_pos)
        if dist < 0.1:
            dist_score = 0.1
        elif dist < 0.2:
            dist_score = 0.2
        elif dist < 0.5:
            dist_score = 0.3
        elif dist < 0.8:
            dist_score = 0.25
        elif dist < 1.2:
            dist_score = 0.15
        else:
            dist_score = 0.05
        quality += dist_score
        
        # 2. 进袋角度评分 [0, 0.5]
        best_pocket_score = 0.0
        for pocket_id, pocket in table.pockets.items():
            pocket_pos = pocket.center[:2]
            angle_score = self._compute_pocket_angle_score(
                cue_pos, target_pos, pocket_pos
            )
            pocket_dist = np.linalg.norm(target_pos - pocket_pos)
            dist_factor = max(0, 1 - pocket_dist / 1.5)
            combined = angle_score * (0.6 + 0.4 * dist_factor)
            best_pocket_score = max(best_pocket_score, combined)
        quality += best_pocket_score * 0.5
        
        # 3. 路径遮挡检查 [0, 0.2]
        blocked = self._check_path_blocked(
            cue_pos, target_pos, balls, exclude=['cue', target_id]
        )
        if blocked:
            quality -= 0.15
        else:
            quality += 0.2
        
        return max(0, min(1, quality))
    
    def _compute_pocket_angle_score(self,
                                     cue_pos: np.ndarray,
                                     target_pos: np.ndarray,
                                     pocket_pos: np.ndarray) -> float:
        """计算白球-目标球-袋口的角度得分"""
        v1 = pocket_pos - target_pos
        v1_norm = np.linalg.norm(v1)
        if v1_norm < 0.01:
            return 1.0
        v1 = v1 / v1_norm
        
        v2 = target_pos - cue_pos
        v2_norm = np.linalg.norm(v2)
        if v2_norm < 0.01:
            return 0.5
        v2 = v2 / v2_norm
        
        cos_angle = np.dot(v1, v2)
        
        if cos_angle > 0.95:
            return 1.0
        elif cos_angle > 0.85:
            return 0.85
        elif cos_angle > 0.7:
            return 0.65
        elif cos_angle > 0.5:
            return 0.4
        elif cos_angle > 0.0:
            return 0.15
        else:
            return 0.0
    
    def _check_path_blocked(self,
                             start: np.ndarray,
                             end: np.ndarray,
                             balls: Dict,
                             exclude: List[str] = None) -> bool:
        """检查路径是否被遮挡"""
        if exclude is None:
            exclude = []
        
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 0.01:
            return False
        direction = direction / length
        
        for bid, ball in balls.items():
            if bid in exclude or ball.state.s == 4:
                continue
            
            ball_pos = ball.state.rvw[0][:2]
            to_ball = ball_pos - start
            proj_length = np.dot(to_ball, direction)
            
            if proj_length < 0 or proj_length > length:
                continue
            
            perp_dist = np.linalg.norm(to_ball - proj_length * direction)
            
            if perp_dist < self.ball_radius * 2.5:
                return True
        
        return False
    
    def _compute_foul_penalty(self, shot_result: Dict) -> float:
        """
        犯规惩罚（紧凑范围版本）
        
        范围控制在 [-15, 0]：
        - 白球进袋：-8
        - 首先击中对方球：-5
        - 无球进袋且无球触边：-3
        - 空杆：-6
        """
        penalty = 0.0
        
        if shot_result.get('WHITE_BALL_INTO_POCKET', False):
            penalty -= 8.0
        
        if shot_result.get('FOUL_FIRST_HIT', False):
            penalty -= 5.0
        
        if shot_result.get('NO_POCKET_NO_RAIL', False):
            penalty -= 3.0
        
        if shot_result.get('NO_HIT', False):
            penalty -= 6.0
        
        # 限制最大惩罚
        return max(penalty, -15.0)
    
    def _is_foul(self, shot_result: Dict) -> bool:
        """检查是否犯规"""
        return (shot_result.get('WHITE_BALL_INTO_POCKET', False) or
                shot_result.get('FOUL_FIRST_HIT', False) or
                shot_result.get('NO_POCKET_NO_RAIL', False) or
                shot_result.get('NO_HIT', False))
    
    def add_defense_reward(self,
                           opponent_shot_result: Dict,
                           enabled: bool = False) -> float:
        """
        防守奖励：对手犯规时给予奖励
        
        需要等到对手击球后调用
        """
        if not enabled or opponent_shot_result is None:
            return 0.0
        
        reward = 0.0
        
        if opponent_shot_result.get('WHITE_BALL_INTO_POCKET', False):
            reward += 10.0
        
        if opponent_shot_result.get('FOUL_FIRST_HIT', False):
            reward += 8.0
        
        if opponent_shot_result.get('NO_HIT', False):
            reward += 12.0
        
        return self.w_defense * reward

