"""
Reward Shaper - 奖励函数
实现比例价值奖励 + 防守奖励 + 犯规惩罚
"""

import numpy as np
from config import REWARD_CONFIG


class RewardShaper:
    """
    奖励塑形器：将游戏结果转换为训练信号
    
    奖励组成：
    1. 即时奖励：进球价值 + 球权价值 + 犯规惩罚
    2. 防守奖励：对手失误时的延迟奖励（需要在下一回合补充）
    3. 终局奖励：胜利/失败的大额奖励
    """
    
    def __init__(self, config=None):
        self.config = config or REWARD_CONFIG
        
        # 奖励系数
        self.C1 = self.config['C1']
        self.C2 = self.config['C2']
        self.C3 = self.config['C3']
        
        # 防守奖励
        self.defense_foul = self.config['defense_foul']
        self.defense_no_pocket = self.config['defense_no_pocket']
        
        # 犯规惩罚
        self.foul_penalties = {
            'WHITE_BALL_INTO_POCKET': self.config['foul_white_ball'],
            'FOUL_FIRST_HIT': self.config['foul_first_hit'],
            'NO_POCKET_NO_RAIL': self.config['foul_no_rail'],
            'NO_HIT': self.config['foul_no_hit'],
        }
        
        # 终局奖励
        self.win_reward = self.config['win_reward']  # 主动胜利
        self.win_passive = self.config.get('win_passive', 30.0)  # 被动胜利
        self.loss_reward = self.config['loss_reward']  # 主动失败  
        self.loss_passive = self.config.get('loss_passive', -30.0)  # 被动失败
        self.lose_turn = self.config['lose_turn']
    
    def calculate_immediate_reward(self, shot_result, my_balls_before, enemy_balls_before, 
                                   game_done=False, i_won=None, win_reason=None):
        """
        计算即时奖励（不包含防守奖励）
        
        Args:
            shot_result: dict, 包含击球结果信息
                - ME_INTO_POCKET: list, 我方进球
                - ENEMY_INTO_POCKET: list, 对方进球
                - BLACK_BALL_INTO_POCKET: bool, 黑八是否进袋
                - WHITE_BALL_INTO_POCKET: bool
                - FOUL_FIRST_HIT: bool
                - NO_POCKET_NO_RAIL: bool
                - NO_HIT: bool
            my_balls_before: int, 击球前我方剩余球数
            enemy_balls_before: int, 击球前对方剩余球数
            game_done: bool, 游戏是否结束
            i_won: bool or None, 如果游戏结束，我是否获胜（True/False/None）
            win_reason: str, 胜利原因 ('active' 打进黑八 / 'passive' 对手失误)
        
        Returns:
            float: 即时奖励
        """
        reward = 0.0
        
        # 1. 终局奖励（最高优先级）
        if game_done:
            if i_won is True:
                # 区分主动胜利和被动胜利
                if win_reason == 'active':  # 打进黑八
                    return self.win_reward
                else:  # 对手失误导致胜利
                    return self.win_passive
            elif i_won is False:
                # 区分主动失败和被动失败
                if win_reason == 'active':  # 自己失误导致输球
                    return self.loss_reward
                else:  # 对手打进黑八
                    return self.loss_passive
            else:  # 平局或超时
                return 0.0
        
        # 2. 己方进球价值（比例奖励）
        my_pocketed = shot_result.get('ME_INTO_POCKET', [])
        if my_pocketed and my_balls_before > 0:
            my_value = (len(my_pocketed) / my_balls_before) * self.C1
            reward += my_value
        
        # 3. 对方进球价值（比例惩罚）
        enemy_pocketed = shot_result.get('ENEMY_INTO_POCKET', [])
        if enemy_pocketed and enemy_balls_before > 0:
            enemy_value = (len(enemy_pocketed) / enemy_balls_before) * self.C2
            reward -= enemy_value
        
        # 4. 球权价值
        if my_pocketed:  # 保持球权
            reward += self.C3
        else:  # 失去球权
            reward += self.lose_turn  # 负值
        
        # 5. 犯规惩罚
        for foul_type, penalty in self.foul_penalties.items():
            if shot_result.get(foul_type, False):
                reward += penalty  # penalty 是负值
        
        return reward
    
    def calculate_defense_reward(self, opponent_shot_result):
        """
        计算防守奖励（对手失误时的奖励）
        
        Args:
            opponent_shot_result: dict, 对手的击球结果
        
        Returns:
            float: 防守奖励（正值）
        """
        reward = 0.0
        
        # 对手犯规
        fouls = ['WHITE_BALL_INTO_POCKET', 'FOUL_FIRST_HIT', 'NO_POCKET_NO_RAIL', 'NO_HIT']
        if any(opponent_shot_result.get(foul, False) for foul in fouls):
            reward += self.defense_foul
        
        # 对手未进球（较小的防守奖励）
        elif not opponent_shot_result.get('ME_INTO_POCKET', []):
            reward += self.defense_no_pocket
        
        return reward
    
    def get_reward_breakdown(self, shot_result, my_balls_before, enemy_balls_before):
        """
        获取奖励的详细分解（用于调试和分析）
        
        Returns:
            dict: 奖励分解
        """
        breakdown = {
            'my_ball_value': 0.0,
            'enemy_ball_value': 0.0,
            'turn_value': 0.0,
            'foul_penalty': 0.0,
            'total': 0.0
        }
        
        # 己方进球
        my_pocketed = shot_result.get('ME_INTO_POCKET', [])
        if my_pocketed and my_balls_before > 0:
            breakdown['my_ball_value'] = (len(my_pocketed) / my_balls_before) * self.C1
        
        # 对方进球
        enemy_pocketed = shot_result.get('ENEMY_INTO_POCKET', [])
        if enemy_pocketed and enemy_balls_before > 0:
            breakdown['enemy_ball_value'] = -(len(enemy_pocketed) / enemy_balls_before) * self.C2
        
        # 球权
        if my_pocketed:
            breakdown['turn_value'] = self.C3
        else:
            breakdown['turn_value'] = self.lose_turn
        
        # 犯规
        for foul_type, penalty in self.foul_penalties.items():
            if shot_result.get(foul_type, False):
                breakdown['foul_penalty'] += penalty
        
        breakdown['total'] = sum([
            breakdown['my_ball_value'],
            breakdown['enemy_ball_value'],
            breakdown['turn_value'],
            breakdown['foul_penalty']
        ])
        
        return breakdown


# ==================== 辅助函数 ====================

def count_remaining_balls(balls, ball_ids):
    """
    统计剩余球数
    
    Args:
        balls: dict, 球状态字典
        ball_ids: list, 要统计的球ID列表
    
    Returns:
        int: 剩余球数
    """
    count = 0
    for bid in ball_ids:
        ball = balls.get(bid)
        if ball and hasattr(ball, 'state') and ball.state.s != 4:
            count += 1
    return count


def get_ball_ids_by_type(ball_type):
    """
    根据球型返回球ID列表
    
    Args:
        ball_type: str, 'solid' or 'stripe'
    
    Returns:
        list: 球ID列表
    """
    if ball_type == 'solid':
        return [1, 2, 3, 4, 5, 6, 7]
    else:  # stripe
        return [9, 10, 11, 12, 13, 14, 15]


# ==================== 测试代码 ====================
if __name__ == '__main__':
    """测试奖励函数"""
    
    reward_shaper = RewardShaper()
    
    # 测试场景1：标准进攻（进1个己方球）
    print("=" * 50)
    print("测试场景1：标准进攻")
    shot_result = {
        'ME_INTO_POCKET': [1],
        'ENEMY_INTO_POCKET': [],
        'WHITE_BALL_INTO_POCKET': False,
        'FOUL_FIRST_HIT': False,
    }
    reward = reward_shaper.calculate_immediate_reward(shot_result, 7, 7)
    breakdown = reward_shaper.get_reward_breakdown(shot_result, 7, 7)
    print(f"奖励: {reward:.2f}")
    print(f"分解: {breakdown}")
    
    # 测试场景2：同时进己方和对方球
    print("\n" + "=" * 50)
    print("测试场景2：同时进己方和对方球（各剩5球）")
    shot_result = {
        'ME_INTO_POCKET': [1],
        'ENEMY_INTO_POCKET': [9],
        'WHITE_BALL_INTO_POCKET': False,
    }
    reward = reward_shaper.calculate_immediate_reward(shot_result, 5, 5)
    breakdown = reward_shaper.get_reward_breakdown(shot_result, 5, 5)
    print(f"奖励: {reward:.2f}")
    print(f"分解: {breakdown}")
    
    # 测试场景3：清台关键球
    print("\n" + "=" * 50)
    print("测试场景3：清台关键球（我方剩1球）")
    shot_result = {
        'ME_INTO_POCKET': [7],
        'ENEMY_INTO_POCKET': [],
    }
    reward = reward_shaper.calculate_immediate_reward(shot_result, 1, 6)
    breakdown = reward_shaper.get_reward_breakdown(shot_result, 1, 6)
    print(f"奖励: {reward:.2f}")
    print(f"分解: {breakdown}")
    
    # 测试场景4：白球犯规
    print("\n" + "=" * 50)
    print("测试场景4：白球犯规")
    shot_result = {
        'ME_INTO_POCKET': [],
        'ENEMY_INTO_POCKET': [],
        'WHITE_BALL_INTO_POCKET': True,
    }
    reward = reward_shaper.calculate_immediate_reward(shot_result, 5, 5)
    breakdown = reward_shaper.get_reward_breakdown(shot_result, 5, 5)
    print(f"奖励: {reward:.2f}")
    print(f"分解: {breakdown}")
    
    # 测试场景5：防守成功（对手犯规）
    print("\n" + "=" * 50)
    print("测试场景5：防守成功（对手犯规）")
    opponent_result = {
        'WHITE_BALL_INTO_POCKET': True,
    }
    defense_reward = reward_shaper.calculate_defense_reward(opponent_result)
    print(f"防守奖励: {defense_reward:.2f}")
    
    # 测试场景6：胜利
    print("\n" + "=" * 50)
    print("测试场景6：胜利")
    reward = reward_shaper.calculate_immediate_reward({}, 0, 5, game_done=True, i_won=True)
    print(f"奖励: {reward:.2f}")
    
    print("\n✅ 奖励函数测试通过！")
