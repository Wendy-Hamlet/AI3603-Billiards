"""
reward.py - 奖励计算器

实现 SAC 训练的奖励函数，包含：
- 终局奖励（区分主动/被动胜负）
- 进球奖励（递增奖励 + 球权逻辑）
- 走位奖励
- 犯规惩罚
- 防守奖励（可选）
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class RewardCalculator:
    """
    SAC 奖励计算器
    
    奖励组成：
    - R_terminal: 终局奖励（胜/负/平）
    - R_pocket: 进球奖励
    - R_position: 走位奖励
    - R_foul: 犯规惩罚
    - R_defense: 防守奖励（可选）
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
        计算单步奖励
        
        Args:
            shot_result: take_shot() 的返回值
            balls_before: 击球前的球状态
            balls_after: 击球后的球状态
            my_targets: 我方目标球列表
            enemy_targets: 对方目标球列表
            table: 球桌对象
            game_done: 游戏是否结束
            winner: 胜者 ('A', 'B', 'SAME')
            my_player: 我方玩家标识 ('A' 或 'B')
            is_my_shot: 这一杆是否是我方击球
        
        Returns:
            Tuple[float, Dict]: (总奖励, 奖励详情)
        """
        reward_details = {}
        total_reward = 0.0
        
        # 计算击球前的剩余球数
        my_remaining_before = sum(1 for bid in my_targets 
                                  if bid != '8' and balls_before[bid].state.s != 4)
        enemy_remaining_before = sum(1 for bid in enemy_targets 
                                     if balls_before[bid].state.s != 4)
        
        # 1. 终局奖励
        r_terminal = self._compute_terminal_reward(
            game_done, winner, my_player, is_my_shot, shot_result
        )
        reward_details['terminal'] = r_terminal
        total_reward += self.w_terminal * r_terminal
        
        # 如果游戏结束且是致命犯规，不再计算其他奖励
        if game_done and r_terminal < -500:
            reward_details['total'] = total_reward
            return total_reward, reward_details
        
        # 2. 进球奖励
        r_pocket = self._compute_pocket_reward(
            shot_result, my_remaining_before, enemy_remaining_before
        )
        reward_details['pocket'] = r_pocket
        total_reward += self.w_pocket * r_pocket
        
        # 3. 走位奖励（只有在没犯规且游戏未结束时计算）
        if not game_done and not self._is_foul(shot_result):
            r_position = self._compute_position_reward(
                balls_after, my_targets, table
            )
            reward_details['position'] = r_position
            total_reward += self.w_position * r_position
        else:
            reward_details['position'] = 0.0
        
        # 4. 犯规惩罚（游戏未结束时的犯规）
        if not game_done:
            r_foul = self._compute_foul_penalty(shot_result)
            reward_details['foul'] = r_foul
            total_reward += self.w_foul * r_foul
        else:
            reward_details['foul'] = 0.0
        
        # 5. 防守奖励（需要下一步对手信息，这里预留）
        reward_details['defense'] = 0.0
        
        reward_details['total'] = total_reward
        return total_reward, reward_details
    
    def _compute_terminal_reward(self,
                                  game_done: bool,
                                  winner: Optional[str],
                                  my_player: str,
                                  is_my_shot: bool,
                                  shot_result: Dict) -> float:
        """
        终局奖励
        
        区分主动胜负和被动胜负：
        - 我方正常获胜: +100
        - 对方犯规导致我方获胜: 0 (不是我们的功劳)
        - 我方犯规导致失败: -1000 (严厉惩罚)
        - 对方正常获胜: -50 (适度惩罚)
        """
        if not game_done:
            return 0.0
        
        i_won = (winner == my_player)
        is_draw = (winner == 'SAME')
        
        if is_draw:
            return 0.0
        
        if i_won:
            if is_my_shot:
                # 我方击球导致获胜（正常清台打进黑8）
                return 100.0
            else:
                # 对方犯规导致我方获胜
                return 0.0
        else:
            # 我方输了
            if is_my_shot:
                # 我方犯规导致失败
                return -1000.0
            else:
                # 对方正常获胜
                return -50.0
    
    def _compute_pocket_reward(self,
                                shot_result: Dict,
                                my_remaining_before: int,
                                enemy_remaining_before: int) -> float:
        """
        进球奖励
        
        - 己方球：递增奖励（第k球得 5*k 分）
        - 对方球：递减惩罚（第k球扣 5*k 分）
        - 维持球权是核心目标
        """
        reward = 0.0
        
        own_pocketed = shot_result.get('ME_INTO_POCKET', [])
        enemy_pocketed = shot_result.get('ENEMY_INTO_POCKET', [])
        
        n_own = len(own_pocketed)
        n_enemy = len(enemy_pocketed)
        
        # === 己方进球奖励（递增） ===
        # 第1球: 5分, 第2球: 10分, 第3球: 15分, ...
        # 总分 = 5 * (1 + 2 + ... + n) = 5 * n * (n+1) / 2
        own_reward = 5.0 * n_own * (n_own + 1) / 2
        
        # 清台进度加成：剩余球越少，每球价值越高
        if n_own > 0 and my_remaining_before > 0:
            for i in range(n_own):
                remaining_at_this_ball = my_remaining_before - i
                if remaining_at_this_ball > 0:
                    progress_bonus = 10.0 / remaining_at_this_ball
                    own_reward += progress_bonus
        
        reward += own_reward
        
        # === 对方球惩罚（递减） ===
        enemy_penalty = -5.0 * n_enemy * (n_enemy + 1) / 2
        
        # === 球权维持逻辑 ===
        if n_own > 0:
            # 维持球权奖励
            reward += 8.0
            
            # 如果同时帮对手进球，减少惩罚
            if n_enemy > 0:
                enemy_penalty *= 0.5
                # 对手快清台时，帮他进球更严重
                if enemy_remaining_before <= 2:
                    enemy_penalty *= 1.5
        
        reward += enemy_penalty
        
        return reward
    
    def _compute_position_reward(self,
                                  balls: Dict,
                                  my_targets: List[str],
                                  table) -> float:
        """
        走位奖励
        
        评估击球后白球位置的击球质量
        """
        cue_pos = balls['cue'].state.rvw[0][:2]
        
        # 获取还在台上的己方目标球
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
        
        # 考虑第二选择（鲁棒性）
        if len(shot_qualities) >= 2:
            sorted_q = sorted(shot_qualities, reverse=True)
            second_quality = sorted_q[1]
            if second_quality > 0.5:
                best_quality += 0.1 * second_quality
        
        # 映射到奖励范围 [-5, +10]
        reward = (best_quality - 0.3) * 20.0
        reward = np.clip(reward, -5.0, 10.0)
        
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
        犯规惩罚（游戏未结束时的犯规）
        """
        penalty = 0.0
        
        if shot_result.get('WHITE_BALL_INTO_POCKET', False):
            penalty -= 25.0
        
        if shot_result.get('FOUL_FIRST_HIT', False):
            penalty -= 15.0
        
        if shot_result.get('NO_POCKET_NO_RAIL', False):
            penalty -= 12.0
        
        if shot_result.get('NO_HIT', False):
            penalty -= 20.0
        
        return penalty
    
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

