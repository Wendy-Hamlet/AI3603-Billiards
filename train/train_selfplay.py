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
        
        # 状态和动作配置
        self.state_encoder_version = config['state'].get('encoder_version', 'v2')
        self.action_space_type = config['action'].get('action_type', 'simple')
        
        # 根据配置动态确定维度
        if self.state_encoder_version == 'v2':
            self.state_dim = 84
        else:
            self.state_dim = 64
        
        if self.action_space_type == 'simple':
            self.action_dim = 2
        else:
            self.action_dim = 5
        
        print(f"State encoder: {self.state_encoder_version} ({self.state_dim}D)")
        print(f"Action space: {self.action_space_type} ({self.action_dim}D)")
        
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
        
        # 创建经验回放（优先级缓冲区）
        self.replay_buffer = ReplayBuffer(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            buffer_size=config['training']['buffer_size'],
            device=self.device,
            priority_alpha=config['training'].get('priority_alpha', 0.6)
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
        # 限制最大 worker 数为 64
        self.num_workers = min(config['training']['num_workers'], 64)
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
        
        # 游戏统计
        self.game_stats = {
            'wins_a': 0,
            'wins_b': 0,
            'draws': 0,
            'total_reward_a': 0.0,
            'total_reward_b': 0.0,
            'total_steps': 0,
            'total_pockets_a': 0,
            'total_pockets_b': 0
        }
        
        # Worker 管理器
        self.worker_manager = None
    
    def train(self):
        """主训练循环"""
        self.start_time = time.time()
        total_games = 0
        
        print(f"\n{'='*60}")
        print(f"Starting Self-Play Training")
        print(f"  Workers: {self.num_workers} (max CPUs: {os.environ.get('OMP_NUM_THREADS', 'unlimited')})")
        print(f"  Games per batch: {self.games_per_batch}")
        print(f"  Total episodes: {self.total_episodes}")
        print(f"  Device: {self.device}")
        print(f"  Verbose: {self.config['training'].get('verbose', False)}")
        curriculum_config = self.curriculum.get_current_config()
        print(f"  Starting stage: {curriculum_config['stage_idx']+1} ({curriculum_config['stage_name']})")
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
                    'hidden_dims': self.config['network']['actor']['hidden_dims'],
                    'verbose': self.config['training'].get('verbose', False),
                    'state_encoder_version': self.state_encoder_version,
                    'action_space_type': self.action_space_type
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
                batch_start_time = time.time()
                batch_result = self.worker_manager.collect_experiences(
                    target_games=self.games_per_batch,
                    timeout=600.0  # 10分钟超时
                )
                collect_time = time.time() - batch_start_time
                
                experiences = batch_result['experiences']
                batch_stats = batch_result['stats']
                
                # 处理经验
                for exp in experiences:
                    self.replay_buffer.add(
                        state=exp['state'],
                        action=exp['action'],
                        reward=exp['reward'],
                        next_state=exp['next_state'],
                        done=exp['done']
                    )
                
                total_games += self.games_per_batch
                
                # 累积统计
                self.game_stats['wins_a'] += batch_stats['wins_a']
                self.game_stats['wins_b'] += batch_stats['wins_b']
                self.game_stats['draws'] += batch_stats['draws']
                self.game_stats['total_reward_a'] += batch_stats['total_reward_a']
                self.game_stats['total_reward_b'] += batch_stats['total_reward_b']
                self.game_stats['total_steps'] += batch_stats['total_steps']
                self.game_stats['total_pockets_a'] += batch_stats['total_pockets_a']
                self.game_stats['total_pockets_b'] += batch_stats['total_pockets_b']
                
                # 更新课程
                stage_changed = self.curriculum.step(self.games_per_batch)
                if stage_changed:
                    # 重新创建 Worker 以适应新难度
                    self.worker_manager.stop()
                    self.worker_manager = None
                    continue
                
                # 训练
                train_stats = None
                if self.trainer.can_train():
                    train_start_time = time.time()
                    train_stats = self.trainer.train_step()
                    train_time = time.time() - train_start_time
                    self.training_stats.append(train_stats)
                    
                    # 广播新策略
                    policy_state = self.agent.get_policy_state_dict()
                    policy_state = {k: v.cpu() for k, v in policy_state.items()}
                    self.worker_manager.broadcast_policy(policy_state)
                else:
                    train_time = 0
                
                # 每批次统一输出汇总
                self._log_batch_summary(
                    total_games, 
                    len(experiences), 
                    batch_stats,
                    train_stats,
                    collect_time,
                    train_time
                )
                
                # 详细日志（按频率）
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
    
    def _log_batch_summary(self, 
                           total_games: int, 
                           num_experiences: int,
                           batch_stats: dict,
                           train_stats: dict,
                           collect_time: float,
                           train_time: float):
        """每批次训练后的统一汇总输出"""
        # 计算本批次统计
        games_in_batch = batch_stats['wins_a'] + batch_stats['wins_b'] + batch_stats['draws']
        if games_in_batch > 0:
            avg_reward_a = batch_stats['total_reward_a'] / games_in_batch
            avg_reward_b = batch_stats['total_reward_b'] / games_in_batch
            avg_steps = batch_stats['total_steps'] / games_in_batch
            avg_pockets = (batch_stats['total_pockets_a'] + batch_stats['total_pockets_b']) / games_in_batch
            draw_rate = batch_stats['draws'] / games_in_batch * 100
            # A胜率 = A赢的场次 / 总场次（包括平局）
            win_rate_a = batch_stats['wins_a'] / games_in_batch * 100
            win_rate_b = batch_stats['wins_b'] / games_in_batch * 100
        else:
            avg_reward_a = avg_reward_b = avg_steps = avg_pockets = draw_rate = 0
            win_rate_a = win_rate_b = 0
        
        # 构建输出
        progress = self.curriculum.get_progress()
        
        # 单行紧凑输出
        train_info = ""
        if train_stats:
            train_info = f"| CriticL: {train_stats.get('critic_loss', 0):.1f} ActorL: {train_stats.get('actor_loss', 0):.3f} α: {train_stats.get('alpha', 0):.3f}"
        
        # 显示：A胜/B胜/平局 百分比 | A/B平均奖励 | 平均进球 | 平均步数
        print(f"[{total_games:>7,}] Stage {progress['stage']}/{progress['total_stages']} | "
              f"A:{win_rate_a:>2.0f}% B:{win_rate_b:>2.0f}% D:{draw_rate:>2.0f}% | R_A: {avg_reward_a:>6.1f} R_B: {avg_reward_b:>6.1f} | "
              f"Pkt: {avg_pockets:>4.1f} | Steps: {avg_steps:>4.0f} {train_info}")
    
    def _log_progress(self, total_games: int):
        """打印详细训练进度"""
        elapsed = time.time() - self.start_time
        games_per_sec = total_games / max(elapsed, 1)
        
        # 计算累计统计
        total_all_games = self.game_stats['wins_a'] + self.game_stats['wins_b'] + self.game_stats['draws']
        if total_all_games > 0:
            overall_win_rate_a = self.game_stats['wins_a'] / total_all_games * 100
            overall_win_rate_b = self.game_stats['wins_b'] / total_all_games * 100
            overall_draw_rate = self.game_stats['draws'] / total_all_games * 100
        else:
            overall_win_rate_a = overall_win_rate_b = overall_draw_rate = 0
        
        if total_games > 0:
            avg_reward_a = self.game_stats['total_reward_a'] / total_games
            avg_reward_b = self.game_stats['total_reward_b'] / total_games
            avg_steps = self.game_stats['total_steps'] / total_games
            avg_pockets = (self.game_stats['total_pockets_a'] + self.game_stats['total_pockets_b']) / total_games
        else:
            avg_reward_a = avg_reward_b = avg_steps = avg_pockets = 0
        
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
        print(f"  Win Stats: A={self.game_stats['wins_a']} B={self.game_stats['wins_b']} Draw={self.game_stats['draws']} (A:{overall_win_rate_a:.0f}% B:{overall_win_rate_b:.0f}% D:{overall_draw_rate:.0f}%)")
        print(f"  Avg Reward: A={avg_reward_a:.2f} B={avg_reward_b:.2f} | Avg Pockets: {avg_pockets:.2f} | Avg Steps: {avg_steps:.1f}")
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
    parser.add_argument('--cpus', type=int, default=64,
                        help='Maximum number of CPU cores to use (default: 64)')
    parser.add_argument('--stage', type=int, default=None,
                        help='Start from specific curriculum stage (1-4)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose game output')
    args = parser.parse_args()
    
    # 限制 CPU 核数 - 更全面的限制
    max_cpus = min(args.cpus, 64)  # 最多64核
    
    # 设置各种线程库的限制
    os.environ['OMP_NUM_THREADS'] = str(max_cpus)
    os.environ['MKL_NUM_THREADS'] = str(max_cpus)
    os.environ['OPENBLAS_NUM_THREADS'] = str(max_cpus)
    os.environ['NUMEXPR_NUM_THREADS'] = str(max_cpus)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(max_cpus)
    
    # 限制 numpy 线程（需要在导入 numpy 之前设置，但我们可以尝试）
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(limits=max_cpus)
    except ImportError:
        pass
    
    # PyTorch 线程限制
    torch.set_num_threads(max_cpus)
    torch.set_num_interop_threads(min(max_cpus, 4))
    
    # 设置 CPU 亲和性（只使用前 max_cpus 个核心）
    try:
        os.sched_setaffinity(0, set(range(max_cpus)))
        print(f"CPU affinity set to cores 0-{max_cpus-1}")
    except (AttributeError, OSError) as e:
        print(f"Could not set CPU affinity: {e}")
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖配置
    if args.workers is not None:
        # workers 数量不能超过 CPU 核数
        config['training']['num_workers'] = min(args.workers, max_cpus)
    else:
        config['training']['num_workers'] = min(config['training']['num_workers'], max_cpus)
    
    # 添加 verbose 配置
    config['training']['verbose'] = args.verbose
    
    # 创建训练器
    trainer = SelfPlayTrainer(config)
    
    # 设置起始阶段
    if args.stage is not None:
        if 1 <= args.stage <= len(config['curriculum']['stages']):
            trainer.curriculum.current_stage_idx = args.stage - 1
            trainer.curriculum.episodes_in_stage = 0
            print(f"Starting from stage {args.stage}: {config['curriculum']['stages'][args.stage-1]['name']}")
        else:
            print(f"Invalid stage {args.stage}, using stage 1")
    
    # 恢复训练
    if args.resume is not None:
        trainer.agent.load(args.resume)
        print(f"Resumed model from {args.resume}")
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    # 多进程启动方式
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    main()

