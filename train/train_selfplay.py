"""
train_selfplay.py - SAC 自对弈训练主脚本

使用多进程并行收集数据，GPU 训练
支持课程学习
"""

import os
import sys
import time
import yaml
import argparse
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Optional

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.sac.sac_agent import SACAgent, SACTrainer
from train.sac.replay_buffer import ReplayBuffer
from train.parallel.worker import WorkerManager


class CurriculumScheduler:
    """
    课程学习调度器
    
    根据训练进度调整环境难度
    """
    
    def __init__(self, stages: List[Dict]):
        """
        Args:
            stages: 课程阶段列表
                [{'name': str, 'own_balls': int, 'enemy_balls': int, 'episodes': int}, ...]
        """
        self.stages = stages
        self.current_stage_idx = 0
        self.episodes_in_stage = 0
        self.total_episodes = 0
    
    def get_current_config(self) -> Dict:
        """获取当前阶段配置"""
        stage = self.stages[self.current_stage_idx]
        return {
            'own_balls': stage['own_balls'],
            'enemy_balls': stage['enemy_balls'],
            'stage_name': stage['name'],
            'stage_idx': self.current_stage_idx
        }
    
    def step(self, episodes: int = 1) -> bool:
        """
        更新进度
        
        Args:
            episodes: 完成的 episode 数
        
        Returns:
            bool: 是否切换到新阶段
        """
        self.episodes_in_stage += episodes
        self.total_episodes += episodes
        
        stage = self.stages[self.current_stage_idx]
        
        if self.episodes_in_stage >= stage['episodes']:
            if self.current_stage_idx < len(self.stages) - 1:
                self.current_stage_idx += 1
                self.episodes_in_stage = 0
                print(f"\n{'='*50}")
                print(f"Curriculum: Advancing to stage {self.current_stage_idx + 1}")
                print(f"  {self.stages[self.current_stage_idx]['name']}")
                print(f"  own_balls: {self.stages[self.current_stage_idx]['own_balls']}")
                print(f"  enemy_balls: {self.stages[self.current_stage_idx]['enemy_balls']}")
                print(f"{'='*50}\n")
                return True
        
        return False
    
    def get_progress(self) -> Dict:
        """获取进度信息"""
        stage = self.stages[self.current_stage_idx]
        return {
            'stage': self.current_stage_idx + 1,
            'total_stages': len(self.stages),
            'stage_name': stage['name'],
            'stage_progress': self.episodes_in_stage / stage['episodes'],
            'total_episodes': self.total_episodes
        }


class SelfPlayTrainer:
    """
    自对弈训练器
    
    协调 Worker 数据收集和 Learner 训练
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 设置设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # 设置随机种子
        seed = config['training'].get('seed', 42)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # 状态和动作维度
        self.state_dim = config['state']['state_dim']
        self.action_dim = config['action']['action_dim']
        
        # 创建 SAC Agent
        self.agent = SACAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=config['network']['actor']['hidden_dims'],
            lr_actor=config['training']['lr_actor'],
            lr_critic=config['training']['lr_critic'],
            lr_alpha=config['training']['lr_alpha'],
            gamma=config['training']['gamma'],
            tau=config['training']['tau'],
            initial_alpha=config['training']['initial_alpha'],
            target_entropy=config['training'].get('target_entropy'),
            device=self.device
        )
        
        # 创建经验回放
        self.replay_buffer = ReplayBuffer(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            buffer_size=config['training']['buffer_size'],
            device=self.device
        )
        
        # 创建训练器
        self.trainer = SACTrainer(
            agent=self.agent,
            replay_buffer=self.replay_buffer,
            batch_size=config['training']['batch_size'],
            updates_per_batch=config['training']['updates_per_batch'],
            min_buffer_size=config['training']['min_buffer_size']
        )
        
        # 课程学习
        if config['curriculum']['enabled']:
            self.curriculum = CurriculumScheduler(config['curriculum']['stages'])
        else:
            # 默认完整游戏
            self.curriculum = CurriculumScheduler([{
                'name': 'full_game',
                'own_balls': 7,
                'enemy_balls': 7,
                'episodes': config['training']['total_episodes']
            }])
        
        # 训练参数
        self.num_workers = config['training']['num_workers']
        # games_per_batch 自动根据 worker 数调整，至少为 worker 数
        self.games_per_batch = max(config['training']['games_per_batch'], self.num_workers)
        # 如果 worker 数较少，减少每批游戏数加快迭代
        if self.num_workers <= 4:
            self.games_per_batch = self.num_workers * 2
        self.total_episodes = config['training']['total_episodes']
        self.save_freq = config['training']['save_freq']
        self.eval_freq = config['training']['eval_freq']
        self.log_freq = config['training']['log_freq']
        
        # 日志和保存路径
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = os.path.join(config['logging']['log_dir'], timestamp)
        self.save_dir = os.path.join(config['logging']['save_dir'], timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 保存配置
        with open(os.path.join(self.log_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)
        
        # 统计信息
        self.episode_rewards = []
        self.training_stats = []
        self.start_time = None
        
        # Worker 管理器
        self.worker_manager = None
    
    def train(self):
        """主训练循环"""
        self.start_time = time.time()
        total_games = 0
        
        print(f"\n{'='*60}")
        print(f"Starting Self-Play Training")
        print(f"  Workers: {self.num_workers}")
        print(f"  Games per batch: {self.games_per_batch}")
        print(f"  Total episodes: {self.total_episodes}")
        print(f"  Device: {self.device}")
        print(f"{'='*60}\n")
        
        try:
            while total_games < self.total_episodes:
                # 获取当前课程配置
                curriculum_config = self.curriculum.get_current_config()
                
                # 创建/更新 Worker 配置
                worker_config = {
                    'seed': self.config['training']['seed'],
                    'own_balls': curriculum_config['own_balls'],
                    'enemy_balls': curriculum_config['enemy_balls'],
                    'enable_noise': True,
                    'hidden_dims': self.config['network']['actor']['hidden_dims']
                }
                
                # 如果需要，重新创建 Worker
                if self.worker_manager is None:
                    self.worker_manager = WorkerManager(
                        num_workers=self.num_workers,
                        config=worker_config
                    )
                    self.worker_manager.start()
                    
                    # 广播初始策略
                    policy_state = self.agent.get_policy_state_dict()
                    # 转换到 CPU
                    policy_state = {k: v.cpu() for k, v in policy_state.items()}
                    self.worker_manager.broadcast_policy(policy_state)
                
                # 收集一批游戏经验
                experiences = self.worker_manager.collect_experiences(
                    target_games=self.games_per_batch,
                    timeout=300.0
                )
                
                # 处理经验
                batch_rewards = []
                for exp in experiences:
                    self.replay_buffer.add(
                        state=exp['state'],
                        action=exp['action'],
                        reward=exp['reward'],
                        next_state=exp['next_state'],
                        done=exp['done']
                    )
                    batch_rewards.append(exp['reward'])
                
                total_games += self.games_per_batch
                self.episode_rewards.extend(batch_rewards)
                
                # 更新课程
                stage_changed = self.curriculum.step(self.games_per_batch)
                if stage_changed:
                    # 重新创建 Worker 以适应新难度
                    self.worker_manager.stop()
                    self.worker_manager = None
                    continue
                
                # 训练
                if self.trainer.can_train():
                    stats = self.trainer.train_step()
                    self.training_stats.append(stats)
                    
                    # 广播新策略
                    policy_state = self.agent.get_policy_state_dict()
                    policy_state = {k: v.cpu() for k, v in policy_state.items()}
                    self.worker_manager.broadcast_policy(policy_state)
                
                # 日志
                if total_games % self.log_freq == 0:
                    self._log_progress(total_games)
                
                # 保存
                if total_games % self.save_freq == 0:
                    self._save_checkpoint(total_games)
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        finally:
            # 清理
            if self.worker_manager is not None:
                self.worker_manager.stop()
            
            # 保存最终模型
            self._save_checkpoint(total_games, final=True)
        
        print(f"\nTraining completed!")
        print(f"  Total games: {total_games}")
        print(f"  Total time: {(time.time() - self.start_time) / 3600:.2f} hours")
    
    def _log_progress(self, total_games: int):
        """打印训练进度"""
        elapsed = time.time() - self.start_time
        games_per_sec = total_games / max(elapsed, 1)
        
        # 计算最近的平均奖励
        recent_rewards = self.episode_rewards[-1000:]
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        
        # 训练统计
        if self.training_stats:
            recent_stats = self.training_stats[-100:]
            avg_critic_loss = np.mean([s.get('critic_loss', 0) for s in recent_stats])
            avg_actor_loss = np.mean([s.get('actor_loss', 0) for s in recent_stats])
            avg_alpha = np.mean([s.get('alpha', 0) for s in recent_stats])
        else:
            avg_critic_loss = avg_actor_loss = avg_alpha = 0
        
        progress = self.curriculum.get_progress()
        
        print(f"\n[Games: {total_games:,} | Time: {elapsed/60:.1f}min | Speed: {games_per_sec:.2f} g/s]")
        print(f"  Curriculum: Stage {progress['stage']}/{progress['total_stages']} ({progress['stage_name']})")
        print(f"  Stage Progress: {progress['stage_progress']*100:.1f}%")
        print(f"  Buffer Size: {len(self.replay_buffer):,}")
        print(f"  Avg Reward: {avg_reward:.2f}")
        print(f"  Critic Loss: {avg_critic_loss:.4f} | Actor Loss: {avg_actor_loss:.4f} | Alpha: {avg_alpha:.4f}")
    
    def _save_checkpoint(self, total_games: int, final: bool = False):
        """保存检查点"""
        suffix = 'final' if final else f'ep{total_games}'
        path = os.path.join(self.save_dir, f'sac_{suffix}.pt')
        self.agent.save(path)
        
        # 保存训练统计
        stats_path = os.path.join(self.save_dir, f'stats_{suffix}.npz')
        np.savez(stats_path,
                 episode_rewards=np.array(self.episode_rewards),
                 total_games=total_games)


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='SAC Self-Play Training')
    parser.add_argument('--config', type=str, default='train/config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--workers', type=int, default=None,
                        help='Override number of workers')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖配置
    if args.workers is not None:
        config['training']['num_workers'] = args.workers
    
    # 创建训练器
    trainer = SelfPlayTrainer(config)
    
    # 恢复训练
    if args.resume is not None:
        trainer.agent.load(args.resume)
        print(f"Resumed from {args.resume}")
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    # 多进程启动方式
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    main()

