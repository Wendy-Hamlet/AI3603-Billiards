"""
state_encoder.py - 状态编码器

将台球状态编码为神经网络输入，采用对称设计：
- 己方7球使用相同的编码方式
- 对方7球使用相同的编码方式
- 球的顺序按到白球距离排序，确保置换不变性
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class StateEncoder:
    """
    台球状态编码器
    
    状态维度 (64维):
    - 白球: 2维 (x, y 归一化)
    - 黑8: 3维 (x, y, pocketed)
    - 己方球: 7 × 3 = 21维 (x, y, pocketed) - 按距离排序
    - 对方球: 7 × 3 = 21维 (x, y, pocketed) - 按距离排序
    - 球袋: 6 × 2 = 12维 (x, y)
    - 游戏信息: 5维 (己方剩余/7, 对方剩余/7, 击球数/60, 是否清台, 阶段)
    
    总计: 2 + 3 + 21 + 21 + 12 + 5 = 64维
    """
    
    def __init__(self, table_w: float = 0.99, table_l: float = 1.98):
        """
        Args:
            table_w: 球桌宽度（米）
            table_l: 球桌长度（米）
        """
        self.table_w = table_w
        self.table_l = table_l
        self.state_dim = 64
        
        # 球ID定义
        self.solid_ids = [str(i) for i in range(1, 8)]    # 1-7 实心球
        self.stripe_ids = [str(i) for i in range(9, 16)]  # 9-15 条纹球
        
        # 球袋ID
        self.pocket_ids = ['lb', 'lc', 'lt', 'rb', 'rc', 'rt']
    
    def encode(self, 
               balls: Dict, 
               my_targets: List[str], 
               table, 
               hit_count: int = 0) -> np.ndarray:
        """
        编码当前状态
        
        Args:
            balls: 球状态字典 {ball_id: Ball}
            my_targets: 我方目标球ID列表
            table: 球桌对象
            hit_count: 当前击球数
            
        Returns:
            np.ndarray: 64维状态向量
        """
        state = []
        
        # 获取白球位置
        cue_pos = self._get_ball_pos(balls['cue'])
        
        # 1. 白球位置 (2维)
        state.extend([
            cue_pos[0] / self.table_w,
            cue_pos[1] / self.table_l
        ])
        
        # 2. 黑8球 (3维)
        eight_pos = self._get_ball_pos(balls['8'])
        eight_pocketed = 1.0 if balls['8'].state.s == 4 else 0.0
        state.extend([
            eight_pos[0] / self.table_w,
            eight_pos[1] / self.table_l,
            eight_pocketed
        ])
        
        # 确定己方和对方球
        if '1' in my_targets or my_targets == ['8']:
            # 我方打实心球
            my_ball_ids = self.solid_ids
            enemy_ball_ids = self.stripe_ids
        else:
            # 我方打条纹球
            my_ball_ids = self.stripe_ids
            enemy_ball_ids = self.solid_ids
        
        # 3. 己方球 (7 × 3 = 21维) - 按到白球距离排序
        my_ball_features = self._encode_balls_symmetric(
            balls, my_ball_ids, cue_pos
        )
        state.extend(my_ball_features)
        
        # 4. 对方球 (7 × 3 = 21维) - 按到白球距离排序
        enemy_ball_features = self._encode_balls_symmetric(
            balls, enemy_ball_ids, cue_pos
        )
        state.extend(enemy_ball_features)
        
        # 5. 球袋位置 (6 × 2 = 12维)
        for pocket_id in self.pocket_ids:
            pocket_pos = table.pockets[pocket_id].center
            state.extend([
                pocket_pos[0] / self.table_w,
                pocket_pos[1] / self.table_l
            ])
        
        # 6. 游戏信息 (5维)
        my_remaining = sum(1 for bid in my_ball_ids 
                          if balls[bid].state.s != 4)
        enemy_remaining = sum(1 for bid in enemy_ball_ids 
                             if balls[bid].state.s != 4)
        
        # 是否已清台（可以打黑8）
        can_shoot_eight = 1.0 if my_remaining == 0 else 0.0
        
        # 游戏进度
        progress = min(1.0, hit_count / 60.0)
        
        state.extend([
            my_remaining / 7.0,
            enemy_remaining / 7.0,
            progress,
            can_shoot_eight,
            0.0  # 预留：课程阶段标识
        ])
        
        return np.array(state, dtype=np.float32)
    
    def _encode_balls_symmetric(self, 
                                 balls: Dict, 
                                 ball_ids: List[str], 
                                 cue_pos: np.ndarray) -> List[float]:
        """
        对称编码一组球
        
        按到白球的距离排序，确保置换不变性
        已进袋的球放在最后，位置设为(0, 0)
        
        Args:
            balls: 球状态字典
            ball_ids: 球ID列表
            cue_pos: 白球位置
            
        Returns:
            List[float]: 21维特征 (7球 × 3特征)
        """
        ball_features = []
        
        for bid in ball_ids:
            ball = balls[bid]
            pos = self._get_ball_pos(ball)
            pocketed = ball.state.s == 4
            
            if pocketed:
                # 已进袋的球
                dist = float('inf')
                features = (dist, 0.0, 0.0, 1.0)
            else:
                # 计算到白球的距离
                dist = np.linalg.norm(pos[:2] - cue_pos[:2])
                features = (
                    dist,
                    pos[0] / self.table_w,
                    pos[1] / self.table_l,
                    0.0  # not pocketed
                )
            ball_features.append(features)
        
        # 按距离排序（已进袋的球在最后）
        ball_features.sort(key=lambda x: x[0])
        
        # 展平特征（去掉距离，只保留位置和进袋标志）
        result = []
        for _, x, y, pocketed in ball_features:
            result.extend([x, y, pocketed])
        
        return result
    
    def _get_ball_pos(self, ball) -> np.ndarray:
        """获取球的位置"""
        return ball.state.rvw[0]
    
    def get_state_dim(self) -> int:
        """返回状态维度"""
        return self.state_dim


class ActionSpace:
    """
    动作空间处理
    
    SAC输出 tanh 范围 [-1, 1]，映射到实际击球参数
    """
    
    def __init__(self):
        self.action_dim = 5
        
        # 实际参数范围
        self.ranges = {
            'V0': (0.5, 8.0),       # 初速度 m/s
            'phi': (0.0, 360.0),    # 水平角度
            'theta': (0.0, 45.0),   # 垂直角度（限制避免跳球）
            'a': (-0.4, 0.4),       # 横向偏移
            'b': (-0.4, 0.4)        # 纵向偏移
        }
        
        # 参数顺序
        self.param_order = ['V0', 'phi', 'theta', 'a', 'b']
    
    def from_normalized(self, action: np.ndarray) -> Dict[str, float]:
        """
        将归一化动作 [-1, 1] 转换为实际击球参数
        
        Args:
            action: 5维归一化动作向量
            
        Returns:
            Dict: 击球参数 {'V0', 'phi', 'theta', 'a', 'b'}
        """
        result = {}
        for i, param in enumerate(self.param_order):
            low, high = self.ranges[param]
            # [-1, 1] -> [0, 1] -> [low, high]
            normalized_value = (action[i] + 1) * 0.5
            result[param] = low + normalized_value * (high - low)
        return result
    
    def to_normalized(self, action_dict: Dict[str, float]) -> np.ndarray:
        """
        将实际击球参数转换为归一化动作
        
        Args:
            action_dict: 击球参数字典
            
        Returns:
            np.ndarray: 5维归一化动作向量
        """
        result = []
        for param in self.param_order:
            low, high = self.ranges[param]
            value = action_dict[param]
            # [low, high] -> [0, 1] -> [-1, 1]
            normalized = (value - low) / (high - low)
            result.append(normalized * 2 - 1)
        return np.array(result, dtype=np.float32)
    
    def get_action_dim(self) -> int:
        """返回动作维度"""
        return self.action_dim
    
    def clip_action(self, action: np.ndarray) -> np.ndarray:
        """裁剪动作到有效范围"""
        return np.clip(action, -1.0, 1.0)

