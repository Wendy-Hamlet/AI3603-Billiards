"""
Train SAC Parallel - 并行 SAC 训练主脚本
实现批量并行对局以加速训练
"""

import os
import sys

# ==================== CUDA 环境设置 ====================
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import time
import numpy as np
import torch
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# 检查 CUDA 环境
print("="*60)
print("环境检查")
print("="*60)
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
print()

# 导入自定义模块
from config import (
    TRAINING_STAGES, EVAL_CONFIG, CHECKPOINT_CONFIG, LOG_CONFIG,
    DEVICE, SAC_CONFIG, denormalize_action
)
from state_encoder import StateEncoder
from reward_shaper import RewardShaper, get_ball_ids_by_type, count_remaining_balls
from sac_agent import SACAgent, SACAgentWrapper
from replay_buffer import ReplayBuffer, EpisodeTracker
from opponent_pool import OpponentPool
from poolenv import PoolEnv


class ParallelSACTrainer:
    """并行 SAC 训练器 - 批量运行多个环境实例"""
    
    def __init__(self, resume_from=None):
        """
        Args:
            resume_from: str, checkpoint 路径（可选，用于恢复训练）
        """
        print("=" * 60)
        print("初始化并行 SAC 训练器")
        print("=" * 60)
        
        # 初始化组件（共享）
        self.state_encoder = StateEncoder()
        self.reward_shaper = RewardShaper()
        self.sac_agent = SACAgent()
        self.sac_wrapper = SACAgentWrapper(self.sac_agent, self.state_encoder)
        self.replay_buffer = ReplayBuffer(capacity=SAC_CONFIG['replay_buffer_size'])
        self.opponent_pool = OpponentPool()
        
        # 并行配置
        self.num_parallel_envs = SAC_CONFIG['num_parallel_envs']
        self.update_frequency = SAC_CONFIG['update_frequency']
        
        # 训练状态
        self.global_episode = 0
        self.current_stage = 'stage1'
        self.stage_episode = 0
        
        # 统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_counts = {'basic': 0, 'physics': 0, 'mcts': 0, 'self': 0}
        self.game_counts = {'basic': 0, 'physics': 0, 'mcts': 0, 'self': 0}
        
        # 创建保存目录
        os.makedirs(CHECKPOINT_CONFIG['save_dir'], exist_ok=True)
        os.makedirs(LOG_CONFIG['log_dir'], exist_ok=True)
        
        # 恢复训练（如果指定）
        if resume_from:
            self._load_checkpoint(resume_from)
        
        print(f"✅ 初始化完成")
        print(f"   Device: {DEVICE}")
        print(f"   并行环境数: {self.num_parallel_envs}")
        print(f"   更新频率: 每 {self.update_frequency} episodes")
        print(f"   Replay Buffer Capacity: {SAC_CONFIG['replay_buffer_size']}")
        print(f"   Training Stages: {len(TRAINING_STAGES)}")
    
    def train(self):
        """主训练循环"""
        print("\n" + "=" * 60)
        print("开始并行训练")
        print("=" * 60)
        
        # 预热阶段
        if len(self.replay_buffer) < SAC_CONFIG['warmup_steps']:
            self._warmup_parallel()
        
        # 遍历训练阶段
        for stage_name, stage_config in TRAINING_STAGES.items():
            if self.current_stage != stage_name:
                continue  # 跳过已完成的阶段
            
            self._train_stage(stage_name, stage_config)
            
            # 完成当前阶段，进入下一阶段
            self.current_stage = self._get_next_stage(stage_name)
            self.stage_episode = 0
        
        print("\n" + "=" * 60)
        print("🎉 训练完成！")
        print("=" * 60)
    
    def _warmup_parallel(self):
        """并行预热 - 使用随机策略填充 buffer"""
        print("\n" + "-" * 60)
        print(f"预热阶段：并行随机策略填充 buffer 到 {SAC_CONFIG['warmup_steps']} 个 transitions")
        print(f"   环境噪声: 禁用")
        print(f"   对手: RandomAgent (快速随机)")
        print(f"   并行环境数: {self.num_parallel_envs}")
        print("-" * 60)
        
        while len(self.replay_buffer) < SAC_CONFIG['warmup_steps']:
            # 并行运行多个episode
            batch_results = self._run_parallel_episodes_warmup(self.num_parallel_envs)
            
            # 存储所有transitions
            for transitions in batch_results:
                for transition in transitions:
                    self.replay_buffer.push(*transition)
            
            if len(self.replay_buffer) % 1000 == 0 or len(self.replay_buffer) >= SAC_CONFIG['warmup_steps']:
                progress = min(100.0, (len(self.replay_buffer) / SAC_CONFIG['warmup_steps']) * 100)
                print(f"  📦 Buffer: {len(self.replay_buffer):5d}/{SAC_CONFIG['warmup_steps']} [{progress:5.1f}%]")
        
        print(f"✅ 预热完成，buffer size: {len(self.replay_buffer)}")
    
    def _run_parallel_episodes_warmup(self, num_episodes):
        """并行运行多个warmup episodes"""
        with ThreadPoolExecutor(max_workers=self.num_parallel_envs) as executor:
            futures = [
                executor.submit(self._run_single_episode_warmup)
                for _ in range(num_episodes)
            ]
            
            results = []
            for future in as_completed(futures):
                try:
                    transitions = future.result()
                    results.append(transitions)
                except Exception as e:
                    print(f"❌ Warmup episode error: {e}")
            
            return results
    
    def _run_single_episode_warmup(self):
        """运行单个warmup episode并返回transitions"""
        # 每个线程创建自己的环境实例
        env = PoolEnv(verbose=False)
        env.enable_noise = False
        opponent = self.opponent_pool.get_opponent('random')
        
        transitions = []
        env.reset(target_ball='solid')
        done = False
        
        while not done:
            current_player = env.get_curr_player()
            
            if current_player == 'A':  # SAC agent
                state = self.state_encoder.encode_from_env(env, 'A')
                action = np.random.uniform(-1, 1, SAC_CONFIG['action_dim'])
                
                action_dict = denormalize_action(action)
                shot_result = env.take_shot(action_dict)
                
                my_type = env.player_targets['A'][0]
                enemy_type = 'stripe' if my_type == 'solid' else 'solid'
                my_balls_before = count_remaining_balls(
                    env.balls, get_ball_ids_by_type(my_type)
                )
                enemy_balls_before = count_remaining_balls(
                    env.balls, get_ball_ids_by_type(enemy_type)
                )
                
                reward = self.reward_shaper.calculate_immediate_reward(
                    shot_result, my_balls_before, enemy_balls_before
                )
                
                next_state = self.state_encoder.encode_from_env(env, 'A')
                done = env.get_done()[0]
                
                transitions.append((state, action, reward, next_state, done))
            
            else:  # 对手
                balls, my_type, table = env.get_observation()
                action_dict = opponent.decision(balls, my_type, table)
                env.take_shot(action_dict)
                done = env.get_done()[0]
        
        return transitions
    
    def _train_stage(self, stage_name, stage_config):
        """训练一个阶段"""
        print("\n" + "=" * 80)
        print(f"🎯 阶段 {stage_name}: {stage_config['name']}")
        print("=" * 80)
        print(f"  📈 目标 Episodes: {stage_config['episodes']}")
        print(f"  🤖 对手分布: {stage_config['opponents']}")
        print(f"  🏆 完成条件: {stage_config['target_metrics']}")
        print("=" * 80)
        print(f"Episode |     Stage Progress |                  Reward |   Steps |          Result")
        print("-" * 80)
        
        stage_start_episode = self.global_episode
        target_episodes = stage_config['episodes']
        
        # 收集一批episodes后再更新
        episode_batch = []
        
        while self.stage_episode < target_episodes:
            # 并行运行一批episodes
            batch_results = self._run_parallel_episodes_train(
                self.update_frequency,
                stage_config
            )
            
            # 处理结果并存储到buffer
            for ep_info in batch_results:
                self.global_episode += 1
                self.stage_episode += 1
                
                # 更新统计
                self.episode_rewards.append(ep_info['reward'])
                self.episode_lengths.append(ep_info['length'])
                self.game_counts[ep_info['opponent']] += 1
                if ep_info['won']:
                    self.win_counts[ep_info['opponent']] += 1
                
                # 存储transitions
                for transition in ep_info['transitions']:
                    self.replay_buffer.push(*transition)
                
                # 打印进度
                self._log_episode(ep_info, stage_name, target_episodes)
                
                # 详细统计
                if self.global_episode % LOG_CONFIG['detailed_log_frequency'] == 0:
                    self._print_detailed_stats()
                
                # 评估
                if self.global_episode % EVAL_CONFIG['eval_frequency'] == 0:
                    self._evaluate()
                
                # 保存checkpoint
                if self.global_episode % EVAL_CONFIG['checkpoint_frequency'] == 0:
                    self._save_checkpoint()
            
            # 批量更新网络（收集完一批episodes后）
            self._batch_update_network(len(batch_results))
        
        print(f"\n✅ 阶段 {stage_name} 完成")
    
    def _run_parallel_episodes_train(self, num_episodes, stage_config):
        """并行运行多个训练episodes"""
        with ThreadPoolExecutor(max_workers=self.num_parallel_envs) as executor:
            futures = [
                executor.submit(self._run_single_episode_train, stage_config)
                for _ in range(num_episodes)
            ]
            
            results = []
            for future in as_completed(futures):
                try:
                    ep_info = future.result()
                    results.append(ep_info)
                except Exception as e:
                    print(f"❌ Training episode error: {e}")
                    import traceback
                    traceback.print_exc()
            
            return results
    
    def _run_single_episode_train(self, stage_config):
        """运行单个训练episode"""
        # 每个线程创建自己的环境实例
        env = PoolEnv(verbose=False)
        env.enable_noise = False
        
        # 选择对手
        opponent = self.opponent_pool.sample_opponent(stage_config)
        opponent_type = self._identify_opponent_type(opponent)
        
        transitions = []
        target_ball = 'solid' if np.random.rand() < 0.5 else 'stripe'
        env.reset(target_ball=target_ball)
        
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            current_player = env.get_curr_player()
            
            if current_player == 'A':  # SAC agent
                state = self.state_encoder.encode_from_env(env, 'A')
                
                # 使用共享的SAC agent（线程安全）
                with torch.no_grad():
                    action = self.sac_agent.select_action(state, deterministic=False)
                
                my_type = env.player_targets['A'][0]
                enemy_type = 'stripe' if my_type == 'solid' else 'solid'
                my_balls_before = count_remaining_balls(
                    env.balls, get_ball_ids_by_type(my_type)
                )
                enemy_balls_before = count_remaining_balls(
                    env.balls, get_ball_ids_by_type(enemy_type)
                )
                
                action_dict = denormalize_action(action)
                shot_result = env.take_shot(action_dict)
                
                reward = self.reward_shaper.calculate_immediate_reward(
                    shot_result, my_balls_before, enemy_balls_before
                )
                
                next_state = self.state_encoder.encode_from_env(env, 'A')
                done = env.get_done()[0]
                
                transitions.append((state, action, reward, next_state, done))
                episode_reward += reward
                episode_length += 1
            
            else:  # 对手
                balls, my_type, table = env.get_observation()
                action_dict = opponent.decision(balls, my_type, table)
                env.take_shot(action_dict)
                done = env.get_done()[0]
        
        # 检查胜负
        winner = env.get_winner()
        won = (winner == 'A')
        
        return {
            'transitions': transitions,
            'reward': episode_reward,
            'length': episode_length,
            'opponent': opponent_type,
            'won': won
        }
    
    def _batch_update_network(self, num_episodes):
        """批量更新网络 - 在收集完一批episodes后执行"""
        if len(self.replay_buffer) < SAC_CONFIG['batch_size']:
            return
        
        # 根据episode数量决定更新次数
        # 每个episode平均10-20步，所以更新次数 = num_episodes * 15 * gradient_steps
        update_steps = num_episodes * 15 * SAC_CONFIG['gradient_steps']
        
        for _ in range(update_steps):
            batch = self.replay_buffer.sample(SAC_CONFIG['batch_size'])
            self.sac_agent.update(batch)
    
    def _identify_opponent_type(self, opponent):
        """识别对手类型"""
        class_name = opponent.__class__.__name__
        if 'Random' in class_name:
            return 'random'
        elif 'Basic' in class_name:
            return 'basic'
        elif 'Physics' in class_name:
            return 'physics'
        elif 'MCTS' in class_name:
            return 'mcts'
        elif 'SAC' in class_name or 'Wrapper' in class_name:
            return 'self'
        else:
            return 'unknown'
    
    def _log_episode(self, ep_info, stage_name, target_episodes):
        """记录episode信息"""
        if self.global_episode % LOG_CONFIG['log_frequency'] != 0:
            return
        
        progress = (self.stage_episode / target_episodes) * 100
        avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
        result_symbol = '✓' if ep_info['won'] else '✗'
        
        print(f"Ep {self.global_episode:5d} | Stage {stage_name} [{progress:5.1f}%] | "
              f"Reward: {ep_info['reward']:7.2f} (avg100: {avg_reward:7.2f}) | "
              f"Steps: {ep_info['length']:2d} | {result_symbol} vs {ep_info['opponent']:7s}")
    
    def _print_detailed_stats(self):
        """打印详细统计"""
        print("\n" + "=" * 80)
        print(f"📊 详细统计 (Episode {self.global_episode})")
        print("=" * 80)
        
        recent_rewards = self.episode_rewards[-100:]
        print(f"  🎯 奖励统计:")
        print(f"     - 最近100轮平均: {np.mean(recent_rewards):.2f}")
        print(f"     - 最近100轮标准差: {np.std(recent_rewards):.2f}")
        print(f"     - 最近100轮最大: {np.max(recent_rewards):.2f}")
        print(f"     - 最近100轮最小: {np.min(recent_rewards):.2f}")
        
        print(f"  🎮 训练参数:")
        print(f"     - Alpha (温度): {self.sac_agent.alpha.item():.4f}")
        print(f"     - Buffer 大小: {len(self.replay_buffer)}")
        
        print(f"  🏆 胜率统计:")
        for opponent_type in ['basic', 'physics', 'mcts', 'self']:
            if self.game_counts[opponent_type] > 0:
                winrate = self.win_counts[opponent_type] / self.game_counts[opponent_type]
                print(f"     - vs {opponent_type:7s}: {winrate:5.1%} "
                      f"({self.win_counts[opponent_type]}/{self.game_counts[opponent_type]})")
        print("=" * 80 + "\n")
    
    def _evaluate(self):
        """评估当前策略"""
        # TODO: 实现评估逻辑
        pass
    
    def _save_checkpoint(self):
        """保存checkpoint"""
        checkpoint_path = os.path.join(
            CHECKPOINT_CONFIG['save_dir'],
            f"checkpoint_ep{self.global_episode}.pth"
        )
        
        self.sac_agent.save(checkpoint_path)
        
        # 保存训练状态
        state_path = checkpoint_path.replace('.pth', '_state.pth')
        torch.save({
            'global_episode': self.global_episode,
            'current_stage': self.current_stage,
            'stage_episode': self.stage_episode,
            'episode_rewards': self.episode_rewards,
            'win_counts': self.win_counts,
            'game_counts': self.game_counts,
        }, state_path)
        
        # 保存 replay buffer
        buffer_path = checkpoint_path.replace('.pth', '_buffer.pkl')
        self.replay_buffer.save(buffer_path)
        
        print(f"💾 Checkpoint 已保存: {checkpoint_path}")
        print(f"💾 Buffer 已保存: {len(self.replay_buffer)} transitions")
    
    def _load_checkpoint(self, checkpoint_path):
        """加载checkpoint"""
        self.sac_agent.load(checkpoint_path)
        
        # 加载训练状态
        state_path = checkpoint_path.replace('.pth', '_state.pth')
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=DEVICE, weights_only=True)
            self.global_episode = state['global_episode']
            self.current_stage = state['current_stage']
            self.stage_episode = state['stage_episode']
            self.episode_rewards = state['episode_rewards']
            self.win_counts = state['win_counts']
            self.game_counts = state['game_counts']
            
            print(f"📋 训练状态已恢复: Episode {self.global_episode}, Stage {self.current_stage}")
        
        # 加载 replay buffer
        buffer_path = checkpoint_path.replace('.pth', '_buffer.pkl')
        self.replay_buffer.load(buffer_path)
        
        print(f"📂 Checkpoint 已加载: {checkpoint_path}")
    
    def _get_next_stage(self, current_stage):
        """获取下一个训练阶段"""
        stages = list(TRAINING_STAGES.keys())
        current_idx = stages.index(current_stage)
        if current_idx < len(stages) - 1:
            return stages[current_idx + 1]
        return None  # 所有阶段完成


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='并行 SAC 训练')
    parser.add_argument('--resume', type=str, default=None,
                        help='从 checkpoint 恢复训练')
    args = parser.parse_args()
    
    try:
        trainer = ParallelSACTrainer(resume_from=args.resume)
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        if 'trainer' in locals():
            trainer._save_checkpoint()
            print("模型已保存")
    except Exception as e:
        print(f"\n\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
        if 'trainer' in locals():
            trainer._save_checkpoint()
            print("模型已保存")


if __name__ == '__main__':
    main()
