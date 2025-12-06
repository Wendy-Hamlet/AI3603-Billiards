"""
State Encoder - Option B: 分组语义编码
将游戏状态编码为神经网络输入
"""

import numpy as np
from config import TABLE_CONFIG


class StateEncoder:
    """
    状态编码器：将台球游戏状态编码为固定维度的向量
    
    编码结构（总计 53 维）：
    - 母球信息: 3 维 (x, y, on_table)
    - 我方球: 7×3=21 维 (每个球: x, y, on_table)
    - 对方球: 7×3=21 维
    - 8号球: 3 维
    - 全局信息: 5 维 (我方剩余比例, 对方剩余比例, 我方清台标志, 对方清台标志, 回合数)
    """
    
    def __init__(self):
        self.table_width = TABLE_CONFIG['width']
        self.table_length = TABLE_CONFIG['length']
        self.state_dim = 53
        
    def encode(self, balls, my_type, game_info):
        """
        编码状态
        
        Args:
            balls: dict, 球的状态字典 {ball_id: ball_object}
            my_type: str, 'solid' or 'stripe'
            game_info: dict, 包含 {'turn': int, 'my_balls_remaining': list, 'enemy_balls_remaining': list}
        
        Returns:
            numpy array of shape (53,)
        """
        state = []
        
        # 1. 母球信息 (3维)
        cue_ball = balls.get('cue') or balls.get(0)
        state.extend(self._encode_ball(cue_ball))
        
        # 2. 我方球 (7×3=21维)
        my_ball_ids = self._get_my_ball_ids(my_type)
        my_balls_features = [self._encode_ball(balls.get(bid)) for bid in my_ball_ids]
        
        # 按离母球距离排序（给网络提供优先级提示）
        cue_pos = self._get_ball_position(cue_ball)
        my_balls_features = sorted(
            my_balls_features,
            key=lambda feat: self._distance_to_cue(feat, cue_pos)
        )
        
        for feat in my_balls_features:
            state.extend(feat)
        
        # 3. 对方球 (7×3=21维)
        opponent_type = 'stripe' if my_type == 'solid' else 'solid'
        opponent_ball_ids = self._get_my_ball_ids(opponent_type)
        opponent_balls_features = [self._encode_ball(balls.get(bid)) for bid in opponent_ball_ids]
        
        # 同样按距离排序
        opponent_balls_features = sorted(
            opponent_balls_features,
            key=lambda feat: self._distance_to_cue(feat, cue_pos)
        )
        
        for feat in opponent_balls_features:
            state.extend(feat)
        
        # 4. 8号球 (3维)
        black_ball = balls.get('8') or balls.get(8)
        state.extend(self._encode_ball(black_ball))
        
        # 5. 全局信息 (5维)
        my_remaining = len(game_info.get('my_balls_remaining', []))
        enemy_remaining = len(game_info.get('enemy_balls_remaining', []))
        turn = game_info.get('turn', 0)
        
        state.extend([
            my_remaining / 7.0,           # 我方剩余球比例
            enemy_remaining / 7.0,        # 对方剩余球比例
            float(my_remaining == 0),     # 我方是否清台
            float(enemy_remaining == 0),  # 对方是否清台
            min(turn / 100.0, 1.0)        # 回合数归一化
        ])
        
        return np.array(state, dtype=np.float32)
    
    def _encode_ball(self, ball):
        """
        编码单个球的信息
        
        Args:
            ball: ball object or None
        
        Returns:
            list of 3 floats: [x_norm, y_norm, on_table]
        """
        if ball is None:
            return [0.0, 0.0, 0.0]
        
        # 检查球是否在台面上（state.s == 4 表示入袋）
        on_table = 1.0 if (hasattr(ball, 'state') and ball.state.s != 4) else 0.0
        
        if on_table == 0.0:
            return [0.0, 0.0, 0.0]
        
        # 归一化位置到 [0, 1]
        x_norm = ball.xyz[0] / self.table_width
        y_norm = ball.xyz[1] / self.table_length
        
        # 裁剪到有效范围（处理可能的越界）
        x_norm = np.clip(x_norm, 0.0, 1.0)
        y_norm = np.clip(y_norm, 0.0, 1.0)
        
        return [x_norm, y_norm, on_table]
    
    def _get_ball_position(self, ball):
        """获取球的物理位置"""
        if ball is None:
            return np.array([self.table_width / 2, self.table_length / 2])
        return np.array([ball.xyz[0], ball.xyz[1]])
    
    def _distance_to_cue(self, ball_features, cue_pos):
        """
        计算球特征到母球的距离（用于排序）
        
        Args:
            ball_features: [x_norm, y_norm, on_table]
            cue_pos: numpy array [x, y] in physical coordinates
        
        Returns:
            float: distance (如果球不在台面返回无穷大)
        """
        if ball_features[2] == 0.0:  # 不在台面
            return float('inf')
        
        # 反归一化
        x = ball_features[0] * self.table_width
        y = ball_features[1] * self.table_length
        
        return np.sqrt((x - cue_pos[0])**2 + (y - cue_pos[1])**2)
    
    def _get_my_ball_ids(self, ball_type):
        """
        根据球型返回球ID列表
        
        Args:
            ball_type: 'solid' or 'stripe'
        
        Returns:
            list of ball IDs (int or str)
        """
        if ball_type == 'solid':
            return [1, 2, 3, 4, 5, 6, 7]
        else:  # stripe
            return [9, 10, 11, 12, 13, 14, 15]
    
    def encode_from_env(self, env, player):
        """
        从环境对象直接编码状态（便捷方法）
        
        Args:
            env: PoolEnv instance
            player: 'A' or 'B'
        
        Returns:
            numpy array of shape (53,)
        """
        balls = env.balls
        my_type = env.player_targets[player][0]  # 获取球型
        
        # 构建 game_info
        my_ball_ids = self._get_my_ball_ids(my_type)
        opponent_type = 'stripe' if my_type == 'solid' else 'solid'
        opponent_ball_ids = self._get_my_ball_ids(opponent_type)
        
        my_balls_remaining = [
            bid for bid in my_ball_ids 
            if balls.get(bid) and balls[bid].state.s != 4
        ]
        
        enemy_balls_remaining = [
            bid for bid in opponent_ball_ids
            if balls.get(bid) and balls[bid].state.s != 4
        ]
        
        game_info = {
            'turn': env.hit_count,
            'my_balls_remaining': my_balls_remaining,
            'enemy_balls_remaining': enemy_balls_remaining
        }
        
        return self.encode(balls, my_type, game_info)


# ==================== 测试代码 ====================
if __name__ == '__main__':
    """测试状态编码器"""
    from poolenv import PoolEnv
    
    # 初始化环境
    env = PoolEnv()
    env.reset(target_ball='solid')
    
    # 创建编码器
    encoder = StateEncoder()
    
    # 编码状态
    state = encoder.encode_from_env(env, 'A')
    
    print(f"State dimension: {state.shape}")
    print(f"Expected dimension: {encoder.state_dim}")
    print(f"State sample (first 10): {state[:10]}")
    print(f"State range: [{state.min():.3f}, {state.max():.3f}]")
    print(f"All values finite: {np.all(np.isfinite(state))}")
    
    # 验证维度
    assert state.shape == (encoder.state_dim,), f"维度不匹配: {state.shape} != ({encoder.state_dim},)"
    assert np.all(np.isfinite(state)), "状态包含无效值"
    
    print("\n✅ 状态编码器测试通过！")
