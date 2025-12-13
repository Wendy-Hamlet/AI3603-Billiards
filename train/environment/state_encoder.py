"""
state_encoder.py - 状态编码器

将台球状态编码为神经网络输入：
- V1: 基础编码（64维）- 绝对坐标
- V2: 增强编码（84维）- 相对坐标 + 进袋评分 + 路径检测

动作空间：
- Full: 5维 (V0, phi, theta, a, b)
- Simple: 2维 (V0, phi) - 固定其他参数
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
            # 确保转换为 Python 原生 float (float64)，避免 numba 类型错误
            normalized_value = (float(action[i]) + 1) * 0.5
            result[param] = float(low + normalized_value * (high - low))
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


class ActionSpaceSimple:
    """
    简化动作空间（2维）
    
    只学习力度和方向，固定其他参数：
    - V0: 初速度
    - phi: 水平角度
    - theta: 固定为 5°（轻微仰角）
    - a, b: 固定为 0（中心击球）
    """
    
    def __init__(self):
        self.action_dim = 2
        
        # 只有力度和角度是可学习的
        self.ranges = {
            'V0': (1.0, 6.0),       # 限制速度范围，避免过轻或过重
            'phi': (0.0, 360.0),    # 水平角度
        }
        
        # 固定参数
        self.fixed_params = {
            'theta': 5.0,   # 轻微仰角
            'a': 0.0,       # 中心击球
            'b': 0.0        # 中心击球
        }
        
        self.param_order = ['V0', 'phi']
    
    def from_normalized(self, action: np.ndarray) -> Dict[str, float]:
        """将归一化动作转换为实际击球参数"""
        result = {}
        
        # 可学习参数
        for i, param in enumerate(self.param_order):
            low, high = self.ranges[param]
            normalized_value = (float(action[i]) + 1) * 0.5
            result[param] = float(low + normalized_value * (high - low))
        
        # 固定参数
        result.update(self.fixed_params)
        
        return result
    
    def to_normalized(self, action_dict: Dict[str, float]) -> np.ndarray:
        """将实际击球参数转换为归一化动作"""
        result = []
        for param in self.param_order:
            low, high = self.ranges[param]
            value = action_dict[param]
            normalized = (value - low) / (high - low)
            result.append(normalized * 2 - 1)
        return np.array(result, dtype=np.float32)
    
    def get_action_dim(self) -> int:
        return self.action_dim
    
    def clip_action(self, action: np.ndarray) -> np.ndarray:
        return np.clip(action, -1.0, 1.0)


class StateEncoderV2:
    """
    增强状态编码器（84维）
    
    改进：
    1. 使用相对白球的极坐标
    2. 预计算进袋评分
    3. 路径遮挡检测
    4. 移除固定的袋口坐标
    
    状态组成：
    - 白球位置: 2维
    - 黑8: 6维 (位置, 距离, 角度, 进袋评分, 路径清晰)
    - 己方球: 7 × 6 = 42维
    - 对方球: 7 × 4 = 28维
    - 游戏信息: 6维
    """
    
    # 固定的袋口位置（米）
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
        self.state_dim = 84
        self.ball_radius = 0.02625
        
        self.solid_ids = [str(i) for i in range(1, 8)]
        self.stripe_ids = [str(i) for i in range(9, 16)]
        
        self.max_dist = np.sqrt(table_w**2 + table_l**2)
    
    def encode(self, 
               balls: Dict, 
               my_targets: List[str], 
               table, 
               hit_count: int = 0) -> np.ndarray:
        """编码当前状态"""
        state = []
        
        cue_pos = self._get_ball_pos(balls['cue'])[:2]
        
        # 1. 白球位置 (2维)
        state.extend([
            cue_pos[0] / self.table_w,
            cue_pos[1] / self.table_l
        ])
        
        # 确定己方和对方球
        if '1' in my_targets or my_targets == ['8']:
            my_ball_ids = self.solid_ids
            enemy_ball_ids = self.stripe_ids
        else:
            my_ball_ids = self.stripe_ids
            enemy_ball_ids = self.solid_ids
        
        # 收集所有在台球的位置
        all_ball_positions = {}
        for bid, ball in balls.items():
            if bid != 'cue' and ball.state.s != 4:
                all_ball_positions[bid] = self._get_ball_pos(ball)[:2]
        
        # 2. 黑8球 (6维)
        eight_features = self._encode_target_ball(
            balls['8'], cue_pos, all_ball_positions, '8'
        )
        state.extend(eight_features)
        
        # 3. 己方球 (7 × 6 = 42维)
        my_ball_features = self._encode_my_balls(
            balls, my_ball_ids, cue_pos, all_ball_positions
        )
        state.extend(my_ball_features)
        
        # 4. 对方球 (7 × 4 = 28维)
        enemy_ball_features = self._encode_enemy_balls(
            balls, enemy_ball_ids, cue_pos
        )
        state.extend(enemy_ball_features)
        
        # 5. 游戏信息 (6维)
        my_remaining = sum(1 for bid in my_ball_ids 
                          if balls[bid].state.s != 4)
        enemy_remaining = sum(1 for bid in enemy_ball_ids 
                             if balls[bid].state.s != 4)
        
        can_shoot_eight = 1.0 if my_remaining == 0 else 0.0
        progress = min(1.0, hit_count / 60.0)
        
        # 己方球的平均进袋评分
        avg_pocket_score = 0.0
        count = 0
        for bid in my_ball_ids:
            if balls[bid].state.s != 4:
                pos = self._get_ball_pos(balls[bid])[:2]
                score = self._get_best_pocket_score(pos, cue_pos)
                avg_pocket_score += score
                count += 1
        if count > 0:
            avg_pocket_score /= count
        
        state.extend([
            my_remaining / 7.0,
            enemy_remaining / 7.0,
            progress,
            can_shoot_eight,
            avg_pocket_score,
            0.0
        ])
        
        return np.array(state, dtype=np.float32)
    
    def _encode_target_ball(self, ball, cue_pos: np.ndarray, 
                            all_positions: Dict, ball_id: str) -> List[float]:
        """编码目标球（6维）"""
        if ball.state.s == 4:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        
        pos = self._get_ball_pos(ball)[:2]
        
        delta = pos - cue_pos
        dist = np.linalg.norm(delta)
        angle = np.arctan2(delta[1], delta[0])
        
        pocket_score = self._get_best_pocket_score(pos, cue_pos)
        path_clear = self._is_path_clear(cue_pos, pos, all_positions, ball_id)
        
        return [
            pos[0] / self.table_w,
            pos[1] / self.table_l,
            min(dist / self.max_dist, 1.0),
            np.sin(angle),
            pocket_score,
            1.0 if path_clear else 0.0
        ]
    
    def _encode_my_balls(self, balls: Dict, ball_ids: List[str],
                         cue_pos: np.ndarray, all_positions: Dict) -> List[float]:
        """编码己方球（7 × 6 = 42维）"""
        ball_features = []
        
        for bid in ball_ids:
            ball = balls[bid]
            features = self._encode_target_ball(ball, cue_pos, all_positions, bid)
            
            if ball.state.s != 4:
                pos = self._get_ball_pos(ball)[:2]
                dist = np.linalg.norm(pos - cue_pos)
            else:
                dist = float('inf')
            
            ball_features.append((dist, features))
        
        ball_features.sort(key=lambda x: x[0])
        
        result = []
        for _, features in ball_features:
            result.extend(features)
        
        return result
    
    def _encode_enemy_balls(self, balls: Dict, ball_ids: List[str],
                            cue_pos: np.ndarray) -> List[float]:
        """编码对方球（7 × 4 = 28维）"""
        ball_features = []
        
        for bid in ball_ids:
            ball = balls[bid]
            
            if ball.state.s == 4:
                features = (float('inf'), [0.0, 0.0, 0.0, 1.0])
            else:
                pos = self._get_ball_pos(ball)[:2]
                delta = pos - cue_pos
                dist = np.linalg.norm(delta)
                angle = np.arctan2(delta[1], delta[0])
                
                features = (dist, [
                    min(dist / self.max_dist, 1.0),
                    np.sin(angle),
                    np.cos(angle),
                    0.0
                ])
            
            ball_features.append(features)
        
        ball_features.sort(key=lambda x: x[0])
        
        result = []
        for _, features in ball_features:
            result.extend(features)
        
        return result
    
    def _get_best_pocket_score(self, ball_pos: np.ndarray, 
                               cue_pos: np.ndarray) -> float:
        """计算最佳进袋评分 [0, 1]"""
        best_score = 0.0
        
        for pocket_pos in self.POCKET_POSITIONS.values():
            cue_to_ball = ball_pos - cue_pos
            ball_to_pocket = pocket_pos - ball_pos
            
            cue_dist = np.linalg.norm(cue_to_ball)
            pocket_dist = np.linalg.norm(ball_to_pocket)
            
            if cue_dist < 0.01 or pocket_dist < 0.01:
                continue
            
            cue_to_ball_norm = cue_to_ball / cue_dist
            ball_to_pocket_norm = ball_to_pocket / pocket_dist
            
            cos_angle = np.dot(cue_to_ball_norm, ball_to_pocket_norm)
            angle_score = max(0, (cos_angle + 1) / 2)
            
            if pocket_dist < 0.2:
                dist_score = 0.8 + 0.2 * (pocket_dist / 0.2)
            elif pocket_dist < 0.8:
                dist_score = 1.0
            else:
                dist_score = max(0, 1.0 - (pocket_dist - 0.8) / 1.0)
            
            score = angle_score * 0.7 + dist_score * 0.3
            best_score = max(best_score, score)
        
        return best_score
    
    def _is_path_clear(self, start: np.ndarray, end: np.ndarray,
                       all_positions: Dict, exclude_id: str) -> bool:
        """检查路径是否清晰"""
        direction = end - start
        length = np.linalg.norm(direction)
        
        if length < 0.01:
            return True
        
        direction = direction / length
        
        for bid, pos in all_positions.items():
            if bid == exclude_id:
                continue
            
            to_ball = pos - start
            proj_length = np.dot(to_ball, direction)
            
            if proj_length < 0 or proj_length > length:
                continue
            
            perp_dist = np.linalg.norm(to_ball - proj_length * direction)
            
            if perp_dist < self.ball_radius * 2.2:
                return False
        
        return True
    
    def _get_ball_pos(self, ball) -> np.ndarray:
        return ball.state.rvw[0]
    
    def get_state_dim(self) -> int:
        return self.state_dim

