"""
worker.py - 数据收集 Worker

多进程环境中的数据收集器：
- 运行自对弈游戏
- 收集经验数据
- 与 Learner 同步策略
"""

import numpy as np
import torch
import multiprocessing as mp
from multiprocessing import Queue, Event
import time
import traceback
from typing import Optional, Dict, List
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def worker_process(worker_id: int,
                   experience_queue: Queue,
                   policy_queue: Queue,
                   stop_event: Event,
                   config: Dict):
    """
    Worker 进程主函数
    
    Args:
        worker_id: Worker ID
        experience_queue: 经验发送队列
        policy_queue: 策略接收队列
        stop_event: 停止信号
        config: 配置字典
    """
    try:
        # 限制每个 worker 的线程数（避免占用过多 CPU）
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        torch.set_num_threads(1)
        
        # 设置随机种子
        seed = config.get('seed', 42) + worker_id
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 延迟导入（避免多进程问题）
        from train.environment.pool_wrapper import create_env
        from train.sac.networks import GaussianActor
        from train.environment.state_encoder import ActionSpace, ActionSpaceSimple
        
        # 创建环境
        own_balls = config.get('own_balls', 7)
        enemy_balls = config.get('enemy_balls', 7)
        enable_noise = config.get('enable_noise', True)
        verbose = config.get('verbose', False)
        state_encoder_version = config.get('state_encoder_version', 'v2')
        action_space_type = config.get('action_space_type', 'simple')
        
        env = create_env(
            own_balls=own_balls,
            enemy_balls=enemy_balls,
            enable_noise=enable_noise,
            verbose=verbose,
            state_encoder_version=state_encoder_version,
            action_space_type=action_space_type
        )
        
        # 创建本地策略网络（CPU）
        state_dim = env.state_dim
        action_dim = env.action_dim
        hidden_dims = config.get('hidden_dims', [512, 512, 256])
        
        actor = GaussianActor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        )
        actor.eval()
        
        # 选择动作空间
        if action_space_type == 'simple':
            action_space = ActionSpaceSimple()
        else:
            action_space = ActionSpace()
        
        # 统计信息
        games_played = 0
        total_reward = 0.0
        
        while not stop_event.is_set():
            # 检查是否有新策略
            try:
                while not policy_queue.empty():
                    policy_state = policy_queue.get_nowait()
                    actor.load_state_dict(policy_state)
            except Exception as e:
                pass
            
            # 运行一局游戏
            try:
                game_result = run_game(env, actor, action_space)
                games_played += 1
                
                # 发送经验和统计
                experience_queue.put({
                    'worker_id': worker_id,
                    'experiences': game_result['experiences'],
                    'winner': game_result['winner'],
                    'reward_a': game_result['reward_a'],
                    'reward_b': game_result['reward_b'],
                    'steps_a': game_result['steps_a'],
                    'steps_b': game_result['steps_b'],
                    'total_steps': game_result['total_steps'],
                    'pockets_a': game_result['pockets_a'],
                    'pockets_b': game_result['pockets_b'],
                    'games_played': games_played
                })
                
            except Exception as e:
                # 静默处理错误，避免刷屏
                continue
        
    except Exception as e:
        print(f"[Worker {worker_id}] Fatal error: {e}")
        traceback.print_exc()


def run_game(env, actor, action_space) -> Dict:
    """
    运行一局完整的自对弈游戏
    
    Args:
        env: 自对弈环境
        actor: 策略网络
        action_space: 动作空间
    
    Returns:
        Dict: 包含经验和游戏统计的字典
    """
    # 随机选择先手方的球型
    target_ball = np.random.choice(['solid', 'stripe'])
    state, current_player = env.reset(target_ball=target_ball)
    
    done = False
    winner = None
    
    while not done:
        # 使用策略网络选择动作
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = actor.get_action(state_tensor, deterministic=False)
            # 确保转换为 float64，避免 numba 类型错误
            action = action.numpy().astype(np.float64).flatten()
        
        # 裁剪动作
        action = action_space.clip_action(action)
        
        # 执行动作
        next_state, reward, done, info, next_player = env.step(action)
        
        if done:
            winner = info.get('winner')
        
        state = next_state
        current_player = next_player
    
    # 收集所有经验，分开A和B
    all_experiences = env.get_all_experiences()
    
    # 计算各方总奖励
    reward_a = sum(exp['reward'] for exp in env.player_a_experiences)
    reward_b = sum(exp['reward'] for exp in env.player_b_experiences)
    steps_a = len(env.player_a_experiences)
    steps_b = len(env.player_b_experiences)
    
    # 进球统计（不含黑8）
    pockets_a = env.pockets_a
    pockets_b = env.pockets_b
    
    return {
        'experiences': all_experiences,
        'winner': winner,
        'reward_a': reward_a,
        'reward_b': reward_b,
        'steps_a': steps_a,
        'steps_b': steps_b,
        'total_steps': steps_a + steps_b,
        'pockets_a': pockets_a,
        'pockets_b': pockets_b
    }


class WorkerManager:
    """
    Worker 管理器
    
    管理多个 Worker 进程的生命周期
    """
    
    def __init__(self,
                 num_workers: int,
                 config: Dict):
        """
        Args:
            num_workers: Worker 数量
            config: 配置字典
        """
        self.num_workers = num_workers
        self.config = config
        
        # 进程间通信
        self.experience_queue = mp.Queue(maxsize=num_workers * 10)
        self.policy_queues = [mp.Queue(maxsize=2) for _ in range(num_workers)]
        self.stop_event = mp.Event()
        
        # Worker 进程
        self.workers = []
        
        # 统计信息
        self.total_games = 0
        self.total_experiences = 0
    
    def start(self):
        """启动所有 Worker"""
        for i in range(self.num_workers):
            p = mp.Process(
                target=worker_process,
                args=(i, self.experience_queue, self.policy_queues[i], 
                      self.stop_event, self.config)
            )
            p.start()
            self.workers.append(p)
    
    def stop(self):
        """停止所有 Worker"""
        self.stop_event.set()
        
        for p in self.workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
    
    def broadcast_policy(self, policy_state: dict):
        """
        广播新策略给所有 Worker
        
        Args:
            policy_state: 策略网络状态字典
        """
        for q in self.policy_queues:
            try:
                # 清空旧策略
                while not q.empty():
                    q.get_nowait()
                q.put(policy_state)
            except Exception as e:
                pass
    
    def collect_experiences(self, 
                           target_games: int,
                           timeout: float = 300.0) -> Dict:
        """
        收集指定数量的游戏经验
        
        Args:
            target_games: 目标游戏数
            timeout: 超时时间（秒）
        
        Returns:
            Dict: 包含经验和统计信息
        """
        all_experiences = []
        games_collected = 0
        start_time = time.time()
        
        # 游戏统计
        stats = {
            'wins_a': 0,
            'wins_b': 0,
            'draws': 0,
            'total_reward_a': 0.0,
            'total_reward_b': 0.0,
            'total_steps': 0,
            'total_pockets_a': 0,
            'total_pockets_b': 0
        }
        
        while games_collected < target_games:
            if time.time() - start_time > timeout:
                print(f"Warning: Timeout collecting experiences, got {games_collected}/{target_games}")
                break
            
            try:
                data = self.experience_queue.get(timeout=1.0)
                all_experiences.extend(data['experiences'])
                games_collected += 1
                self.total_games += 1
                self.total_experiences += len(data['experiences'])
                
                # 更新统计
                winner = data.get('winner')
                if winner == 'A':
                    stats['wins_a'] += 1
                elif winner == 'B':
                    stats['wins_b'] += 1
                else:
                    stats['draws'] += 1
                
                stats['total_reward_a'] += data.get('reward_a', 0)
                stats['total_reward_b'] += data.get('reward_b', 0)
                stats['total_steps'] += data.get('total_steps', 0)
                stats['total_pockets_a'] += data.get('pockets_a', 0)
                stats['total_pockets_b'] += data.get('pockets_b', 0)
                
            except Exception as e:
                continue
        
        return {
            'experiences': all_experiences,
            'stats': stats,
            'games_collected': games_collected
        }
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_games': self.total_games,
            'total_experiences': self.total_experiences
        }


def test_worker():
    """测试 Worker 功能"""
    config = {
        'seed': 42,
        'own_balls': 1,
        'enemy_balls': 1,
        'enable_noise': True,
        'hidden_dims': [512, 512, 256]
    }
    
    # 创建管理器
    manager = WorkerManager(num_workers=2, config=config)
    
    try:
        # 启动 Worker
        manager.start()
        
        # 收集一些经验
        experiences = manager.collect_experiences(target_games=4, timeout=60.0)
        
        print(f"Collected {len(experiences)} experiences")
        
    finally:
        manager.stop()


if __name__ == '__main__':
    test_worker()

