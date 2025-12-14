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
            -100（白球进袋）, -150（非法黑8/白球+黑8）, -30（首球/碰库犯规）
    
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
        
    # 4. 计算奖励分数
    score = 0
    
    # 白球进袋处理
    if cue_pocketed and eight_pocketed:
        score -= 150  # 白球+黑8同时进袋，严重犯规
    elif cue_pocketed:
        score -= 100  # 白球进袋
    elif eight_pocketed:
        # 黑8进袋：只有清台后（player_targets == ['8']）才合法
        if player_targets == ['8']:
            score += 100  # 合法打进黑8
        else:
            score -= 150  # 清台前误打黑8，判负
            
    # 首球犯规和碰库犯规
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    # 进球得分（own_pocketed 已根据 player_targets 正确分类）
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    # 合法无进球小奖励
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
    """基于贝叶斯优化的智能 Agent"""
    
    def __init__(self, target_balls=None):
        """初始化 Agent
        
        参数：
            target_balls: 保留参数，暂未使用
        """
        super().__init__()
        
        # 搜索空间
        self.pbounds = {
            'V0': (0.5, 8.0),
            'phi': (0, 360),
            'theta': (0, 90), 
            'a': (-0.5, 0.5),
            'b': (-0.5, 0.5)
        }
        
        # 优化参数
        self.INITIAL_SEARCH = 20
        self.OPT_SEARCH = 10
        self.ALPHA = 1e-2
        
        # 模拟噪声（可调整以改变训练难度）
        self.noise_std = {
            'V0': 0.1,
            'phi': 0.1,
            'theta': 0.1,
            'a': 0.003,
            'b': 0.003
        }
        self.enable_noise = False
        
        print("BasicAgent (Smart, pooltool-native) 已初始化。")

    
    def _create_optimizer(self, reward_function, seed):
        """创建贝叶斯优化器
        
        参数：
            reward_function: 目标函数，(V0, phi, theta, a, b) -> score
            seed: 随机种子
        
        返回：
            BayesianOptimization对象
        """
        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=self.ALPHA,
            n_restarts_optimizer=10,
            random_state=seed
        )
        
        bounds_transformer = SequentialDomainReductionTransformer(
            gamma_osc=0.8,
            gamma_pan=1.0
        )
        
        optimizer = BayesianOptimization(
            f=reward_function,
            pbounds=self.pbounds,
            random_state=seed,
            verbose=0,
            bounds_transformer=bounds_transformer
        )
        optimizer._gp = gpr
        
        return optimizer


    def decision(self, balls=None, my_targets=None, table=None):
        """使用贝叶斯优化搜索最佳击球参数
        
        参数：
            balls: 球状态字典，{ball_id: Ball}
            my_targets: 目标球ID列表，['1', '2', ...]
            table: 球桌对象
        
        返回：
            dict: 击球动作 {'V0', 'phi', 'theta', 'a', 'b'}
                失败时返回随机动作
        """
        if balls is None:
            print(f"[BasicAgent] Agent decision函数未收到balls关键信息，使用随机动作。")
            return self._random_action()
        try:
            
            # 保存一个击球前的状态快照，用于对比
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                print("[BasicAgent] 我的目标球已全部清空，自动切换目标为：8号球")

            # 1.动态创建“奖励函数” (Wrapper)
            # 贝叶斯优化器会调用此函数，并传入参数
            def reward_fn_wrapper(V0, phi, theta, a, b):
                # 创建一个用于模拟的沙盒系统
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")

                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                
                try:
                    if self.enable_noise:
                        V0_noisy = V0 + np.random.normal(0, self.noise_std['V0'])
                        phi_noisy = phi + np.random.normal(0, self.noise_std['phi'])
                        theta_noisy = theta + np.random.normal(0, self.noise_std['theta'])
                        a_noisy = a + np.random.normal(0, self.noise_std['a'])
                        b_noisy = b + np.random.normal(0, self.noise_std['b'])
                        
                        V0_noisy = np.clip(V0_noisy, 0.5, 8.0)
                        phi_noisy = phi_noisy % 360
                        theta_noisy = np.clip(theta_noisy, 0, 90)
                        a_noisy = np.clip(a_noisy, -0.5, 0.5)
                        b_noisy = np.clip(b_noisy, -0.5, 0.5)
                        
                        shot.cue.set_state(V0=V0_noisy, phi=phi_noisy, theta=theta_noisy, a=a_noisy, b=b_noisy)
                    else:
                        shot.cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
                    
                    # 关键：使用带超时保护的物理模拟（3秒上限）
                    if not simulate_with_timeout(shot, timeout=3):
                        return 0  # 超时是物理引擎问题，不惩罚agent
                except Exception as e:
                    # 模拟失败，给予极大惩罚
                    return -500
                
                # 使用我们的“裁判”来打分
                score = analyze_shot_for_reward(
                    shot=shot,
                    last_state=last_state_snapshot,
                    player_targets=my_targets
                )


                return score

            print(f"[BasicAgent] 正在为 Player (targets: {my_targets}) 搜索最佳击球...")
            
            seed = np.random.randint(1e6)
            optimizer = self._create_optimizer(reward_fn_wrapper, seed)
            optimizer.maximize(
                init_points=self.INITIAL_SEARCH,
                n_iter=self.OPT_SEARCH
            )
            
            best_result = optimizer.max
            best_params = best_result['params']
            best_score = best_result['target']

            if best_score < 10:
                print(f"[BasicAgent] 未找到好的方案 (最高分: {best_score:.2f})。使用随机动作。")
                return self._random_action()
            action = {
                'V0': float(best_params['V0']),
                'phi': float(best_params['phi']),
                'theta': float(best_params['theta']),
                'a': float(best_params['a']),
                'b': float(best_params['b']),
            }

            print(f"[BasicAgent] 决策 (得分: {best_score:.2f}): "
                  f"V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                  f"θ={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}")
            return action

        except Exception as e:
            print(f"[BasicAgent] 决策时发生严重错误，使用随机动作。原因: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()


class NewAgent(Agent):
    """增强版Agent - 混合策略 + 噪声鲁棒优化
    
    核心改进：
    1. 战略层：根据局面选择进攻/防守/清台策略
    2. 噪声模拟：在优化时主动添加多次噪声采样，筛选鲁棒动作
    3. 黑8保护：严格过滤可能误打黑8的危险动作
    4. 搜索空间优化：根据策略缩小搜索范围，提升速度和质量
    """
    
    def __init__(self):
        super().__init__()
        
        # 基础搜索空间（会根据策略动态调整）
        self.pbounds = {
            'V0': (0.5, 8.0),
            'phi': (0, 360),
            'theta': (0, 90), 
            'a': (-0.3, 0.3),  # 减小偏移范围，提升精准度
            'b': (-0.3, 0.3)
        }
        
        # 优化参数（比BasicAgent更快）
        self.INITIAL_SEARCH = 15  # 减少初始搜索
        self.OPT_SEARCH = 8       # 减少优化迭代
        self.ALPHA = 1e-2
        
        # 噪声鲁棒性参数
        self.noise_std = {
            'V0': 0.1,
            'phi': 0.1,
            'theta': 0.1,
            'a': 0.003,
            'b': 0.003
        }
        self.N_NOISE_SAMPLES = 5  # 每个动作测试5次噪声采样
        
        print("NewAgent (Enhanced Hybrid) 已初始化。")
    
    def _analyze_game_state(self, balls, my_targets):
        """分析当前局面，制定战略
        
        返回：
            strategy: 'aggressive' | 'defensive' | 'finish' | 'safe'
            pbounds: 调整后的搜索空间
        """
        # 统计球数
        my_remaining = [bid for bid in my_targets if balls[bid].state.s != 4]
        all_balls_on_table = [bid for bid, b in balls.items() if b.state.s != 4 and bid != 'cue']
        enemy_remaining = [bid for bid in all_balls_on_table if bid not in my_targets and bid != '8']
        
        n_my = len(my_remaining)
        n_enemy = len(enemy_remaining)
        is_finish_phase = (my_targets == ['8'])  # 清台阶段
        
        print(f"[NewAgent] 局面分析: 己方球={n_my}, 对方球={n_enemy}, 清台阶段={is_finish_phase}")
        
        # 战略决策
        if is_finish_phase:
            strategy = 'finish'
            # 清台阶段：中等速度，高精度
            pbounds = {
                'V0': (2.0, 5.0),      # 稳定速度
                'phi': (0, 360),
                'theta': (0, 60),      # 减小跳球角度
                'a': (-0.2, 0.2),      # 更小的偏移
                'b': (-0.2, 0.2)
            }
        elif n_my <= 2:
            strategy = 'aggressive'
            # 己方优势：快速清台
            pbounds = {
                'V0': (3.0, 7.0),      # 高速
                'phi': (0, 360),
                'theta': (0, 75),
                'a': (-0.3, 0.3),
                'b': (-0.3, 0.3)
            }
        elif n_enemy <= 2:
            strategy = 'defensive'
            # 对方优势：保守打法，制造障碍
            pbounds = {
                'V0': (0.5, 3.5),      # 低速精准
                'phi': (0, 360),
                'theta': (0, 50),
                'a': (-0.2, 0.2),
                'b': (-0.2, 0.2)
            }
        else:
            strategy = 'safe'
            # 中局：平衡策略
            pbounds = {
                'V0': (1.5, 6.0),
                'phi': (0, 360),
                'theta': (0, 70),
                'a': (-0.25, 0.25),
                'b': (-0.25, 0.25)
            }
        
        print(f"[NewAgent] 采用策略: {strategy}")
        return strategy, pbounds
    
    def _is_action_safe(self, action, balls, my_targets, table):
        """安全性检查：防止误打黑8（关键！）
        
        返回：
            bool: True表示安全，False表示危险
        """
        if my_targets == ['8']:
            return True  # 清台阶段，黑8就是目标
        
        # 快速模拟：只检查是否会打进黑8或白球
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        
        try:
            shot.cue.set_state(
                V0=action['V0'],
                phi=action['phi'],
                theta=action['theta'],
                a=action['a'],
                b=action['b']
            )
            
            if not simulate_with_timeout(shot, timeout=2):
                return False  # 超时也算不安全
            
            # 检查黑8和白球是否进袋
            eight_ball = shot.balls.get('8')
            cue_ball = shot.balls.get('cue')
            
            if eight_ball and eight_ball.state.s == 4:
                return False  # 黑8进袋，危险！
            if cue_ball and cue_ball.state.s == 4:
                return False  # 白球进袋，危险！
            
            return True
            
        except Exception as e:
            return False  # 模拟失败也算不安全
    
    def _robust_reward_evaluation(self, V0, phi, theta, a, b, balls, my_targets, table, last_state_snapshot):
        """噪声鲁棒性评估：多次采样取平均
        
        这是关键改进：不再只看无噪声情况，而是测试真实环境噪声下的表现
        """
        scores = []
        
        for _ in range(self.N_NOISE_SAMPLES):
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            sim_table = copy.deepcopy(table)
            cue = pt.Cue(cue_ball_id="cue")
            shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
            
            try:
                # 添加噪声（模拟真实环境）
                V0_noisy = V0 + np.random.normal(0, self.noise_std['V0'])
                phi_noisy = phi + np.random.normal(0, self.noise_std['phi'])
                theta_noisy = theta + np.random.normal(0, self.noise_std['theta'])
                a_noisy = a + np.random.normal(0, self.noise_std['a'])
                b_noisy = b + np.random.normal(0, self.noise_std['b'])
                
                # 参数裁剪
                V0_noisy = np.clip(V0_noisy, 0.5, 8.0)
                phi_noisy = phi_noisy % 360
                theta_noisy = np.clip(theta_noisy, 0, 90)
                a_noisy = np.clip(a_noisy, -0.5, 0.5)
                b_noisy = np.clip(b_noisy, -0.5, 0.5)
                
                shot.cue.set_state(V0=V0_noisy, phi=phi_noisy, theta=theta_noisy, 
                                  a=a_noisy, b=b_noisy)
                
                if not simulate_with_timeout(shot, timeout=2):
                    scores.append(-100)  # 超时惩罚
                    continue
                
                # 计算得分
                score = analyze_shot_for_reward(
                    shot=shot,
                    last_state=last_state_snapshot,
                    player_targets=my_targets
                )
                
                # 额外惩罚误打黑8（即使概率低也要严厉惩罚）
                eight_ball = shot.balls.get('8')
                if eight_ball and eight_ball.state.s == 4 and my_targets != ['8']:
                    score -= 200  # 严重惩罚
                
                scores.append(score)
                
            except Exception as e:
                scores.append(-500)  # 模拟失败
        
        # 返回平均分（鲁棒性指标）
        avg_score = np.mean(scores)
        min_score = np.min(scores)  # 最坏情况
        
        # 综合评分：70%平均 + 30%最坏情况（避免高风险动作）
        return 0.7 * avg_score + 0.3 * min_score
    
    def _create_optimizer(self, reward_function, seed, pbounds):
        """创建优化器（复用BasicAgent的方法）"""
        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=self.ALPHA,
            n_restarts_optimizer=10,
            random_state=seed
        )
        
        bounds_transformer = SequentialDomainReductionTransformer(
            gamma_osc=0.8,
            gamma_pan=1.0
        )
        
        optimizer = BayesianOptimization(
            f=reward_function,
            pbounds=pbounds,
            random_state=seed,
            verbose=0,
            bounds_transformer=bounds_transformer
        )
        optimizer._gp = gpr
        
        return optimizer
    
    def decision(self, balls=None, my_targets=None, table=None):
        """主决策函数
        
        流程：
        1. 分析局面 → 选择策略
        2. 调整搜索空间
        3. 贝叶斯优化（带噪声鲁棒性测试）
        4. 安全性二次验证
        """
        if balls is None:
            print(f"[NewAgent] 缺少关键信息，使用随机动作。")
            return self._random_action()
        
        try:
            # 保存状态快照
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            
            # 检查是否清台
            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                print("[NewAgent] 己方球已清空，切换目标为黑8")
            
            # 步骤1：战略分析
            strategy, pbounds = self._analyze_game_state(balls, my_targets)
            
            # 步骤2：定义鲁棒性奖励函数
            def reward_fn_wrapper(V0, phi, theta, a, b):
                return self._robust_reward_evaluation(
                    V0, phi, theta, a, b, 
                    balls, my_targets, table, last_state_snapshot
                )
            
            # 步骤3：贝叶斯优化
            print(f"[NewAgent] 开始优化（策略={strategy}）...")
            seed = np.random.randint(1e6)
            optimizer = self._create_optimizer(reward_fn_wrapper, seed, pbounds)
            
            optimizer.maximize(
                init_points=self.INITIAL_SEARCH,
                n_iter=self.OPT_SEARCH
            )
            
            best_result = optimizer.max
            best_params = best_result['params']
            best_score = best_result['target']
            
            # 步骤4：构造动作并进行安全性检查
            action = {
                'V0': float(best_params['V0']),
                'phi': float(best_params['phi']),
                'theta': float(best_params['theta']),
                'a': float(best_params['a']),
                'b': float(best_params['b'])
            }
            
            # 关键：二次安全验证（防止误打黑8）
            if not self._is_action_safe(action, balls, my_targets, table):
                print(f"[NewAgent] 警告：最佳动作可能误打黑8，重新搜索...")
                # 使用更保守的参数重试一次
                pbounds_safe = {
                    'V0': (1.0, 4.0),  # 降低速度
                    'phi': (0, 360),
                    'theta': (0, 50),  # 减小角度
                    'a': (-0.15, 0.15),
                    'b': (-0.15, 0.15)
                }
                optimizer2 = self._create_optimizer(reward_fn_wrapper, seed+1, pbounds_safe)
                optimizer2.maximize(init_points=10, n_iter=5)
                
                best_result = optimizer2.max
                best_params = best_result['params']
                best_score = best_result['target']
                
                action = {
                    'V0': float(best_params['V0']),
                    'phi': float(best_params['phi']),
                    'theta': float(best_params['theta']),
                    'a': float(best_params['a']),
                    'b': float(best_params['b'])
                }
            
            # 如果得分太低，使用随机动作（但要安全）
            if best_score < 5:
                print(f"[NewAgent] 未找到好方案（最高分={best_score:.2f}），尝试安全随机动作")
                for _ in range(10):
                    action = self._random_action()
                    if self._is_action_safe(action, balls, my_targets, table):
                        break
            
            print(f"[NewAgent] 决策 (鲁棒得分={best_score:.2f}): "
                  f"V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                  f"θ={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}")
            
            return action
            
        except Exception as e:
            print(f"[NewAgent] 决策失败，使用随机动作。错误: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()
