"""
Train SAC - SAC训练主脚本
实现渐进式课程学习训练流程
"""

import os
import sys
import time
import numpy as np
import torch
from datetime import datetime

# 导入自定义模块
from config import (
    TRAINING_STAGES, EVAL_CONFIG, CHECKPOINT_CONFIG, LOG_CONFIG,
    DEVICE, SAC_CONFIG
)
from state_encoder import StateEncoder
from reward_shaper import RewardShaper, get_ball_ids_by_type, count_remaining_balls
from sac_agent import SACAgent, SACAgentWrapper
from replay_buffer import ReplayBuffer, EpisodeTracker
from opponent_pool import OpponentPool
from poolenv import PoolEnv


class SACTrainer:
    """SAC 训练器"""
    
    def __init__(self, resume_from=None):
        """
        Args:
            resume_from: str, checkpoint 路径（可选，用于恢复训练）
        """
        print("=" * 60)
        print("初始化 SAC 训练器")
        print("=" * 60)
        
        # 初始化组件
        self.state_encoder = StateEncoder()
        self.reward_shaper = RewardShaper()
        self.sac_agent = SACAgent()
        self.sac_wrapper = SACAgentWrapper(self.sac_agent, self.state_encoder)
        self.replay_buffer = ReplayBuffer(capacity=SAC_CONFIG['replay_buffer_size'])
        self.opponent_pool = OpponentPool()
        self.env = PoolEnv()
        
        # 训练状态
        self.global_episode = 0
        self.current_stage = 'stage1'
        self.stage_episode = 0
        
        # 统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_counts = {'basic': 0, 'physics': 0, 'mcts': 0}
        self.game_counts = {'basic': 0, 'physics': 0, 'mcts': 0}
        
        # 创建保存目录
        os.makedirs(CHECKPOINT_CONFIG['save_dir'], exist_ok=True)
        os.makedirs(LOG_CONFIG['log_dir'], exist_ok=True)
        
        # 恢复训练（如果指定）
        if resume_from:
            self._load_checkpoint(resume_from)
        
        print(f"✅ 初始化完成")
        print(f"   Device: {DEVICE}")
        print(f"   Replay Buffer Capacity: {SAC_CONFIG['replay_buffer_size']}")
        print(f"   Training Stages: {len(TRAINING_STAGES)}")
    
    def train(self):
        """主训练循环"""
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)
        
        # 预热阶段：随机策略填充 buffer
        if len(self.replay_buffer) < SAC_CONFIG['warmup_steps']:
            self._warmup()
        
        # 渐进式训练
        for stage_name, stage_config in TRAINING_STAGES.items():
            if self._should_skip_stage(stage_name):
                continue
            
            self.current_stage = stage_name
            self.stage_episode = 0
            
            print("\n" + "=" * 60)
            print(f"阶段: {stage_config['name']}")
            print(f"目标 Episodes: {stage_config['episodes']}")
            print(f"对手分布: {stage_config['opponents']}")
            print("=" * 60)
            
            # 阶段训练循环
            while self.stage_episode < stage_config['episodes']:
                # 训练一个 episode
                episode_info = self._train_episode(stage_config)
                
                self.global_episode += 1
                self.stage_episode += 1
                
                # 记录统计
                self._log_episode(episode_info)
                
                # 定期评估
                if self.global_episode % EVAL_CONFIG['eval_frequency'] == 0:
                    eval_results = self._evaluate()
                    self._log_evaluation(eval_results)
                    
                    # 检查是否提前完成阶段
                    if self._check_stage_completion(stage_config, eval_results):
                        print(f"\n✅ 阶段 {stage_name} 提前完成！")
                        break
                
                # 保存 checkpoint
                if self.global_episode % EVAL_CONFIG['checkpoint_frequency'] == 0:
                    self._save_checkpoint()
                    
                    # 添加到 self-play 池
                    if self.global_episode >= 5000:  # 训练足够久后才加入
                        self.opponent_pool.add_checkpoint(
                            self.sac_wrapper,
                            self.global_episode,
                            {'episode': self.global_episode}
                        )
        
        print("\n" + "=" * 60)
        print("🎉 训练完成！")
        print("=" * 60)
        
        # 最终评估
        final_eval = self._evaluate()
        self._log_evaluation(final_eval, final=True)
        
        # 保存最终模型
        self._save_checkpoint(is_final=True)
    
    def _warmup(self):
        """预热阶段：使用随机策略填充 replay buffer"""
        print("\n" + "-" * 60)
        print(f"预热阶段：随机策略填充 buffer 到 {SAC_CONFIG['warmup_steps']} 个 transitions")
        print("-" * 60)
        
        while len(self.replay_buffer) < SAC_CONFIG['warmup_steps']:
            # 随机对手
            opponent = self.opponent_pool.get_opponent('basic')
            
            # 玩一局游戏
            self.env.reset(target_ball='solid')
            done = False
            
            while not done:
                current_player = self.env.get_curr_player()
                
                if current_player == 'A':  # SAC agent
                    # 编码状态
                    state = self.state_encoder.encode_from_env(self.env, 'A')
                    
                    # 随机动作
                    action = np.random.uniform(-1, 1, SAC_CONFIG['action_dim'])
                    
                    # 执行动作
                    from config import denormalize_action
                    action_dict = denormalize_action(action)
                    shot_result = self.env.take_shot(**action_dict)
                    
                    # 计算奖励
                    my_balls_before = count_remaining_balls(
                        self.env.balls,
                        get_ball_ids_by_type(self.env.player_targets['A'][0])
                    )
                    enemy_balls_before = count_remaining_balls(
                        self.env.balls,
                        get_ball_ids_by_type(self.env.player_targets['B'][0])
                    )
                    
                    reward = self.reward_shaper.calculate_immediate_reward(
                        shot_result, my_balls_before, enemy_balls_before
                    )
                    
                    next_state = self.state_encoder.encode_from_env(self.env, 'A')
                    done = self.env.get_done()[0]
                    
                    # 存储
                    self.replay_buffer.push(state, action, reward, next_state, done)
                
                else:  # 对手
                    balls, my_type, table = self.env.get_observation()
                    action_dict = opponent.decision(balls, my_type, table)
                    self.env.take_shot(**action_dict)
                    done = self.env.get_done()[0]
            
            if len(self.replay_buffer) % 1000 == 0:
                print(f"  Buffer size: {len(self.replay_buffer)}/{SAC_CONFIG['warmup_steps']}")
        
        print(f"✅ 预热完成，buffer size: {len(self.replay_buffer)}")
    
    def _train_episode(self, stage_config):
        """训练一个 episode"""
        # 选择对手
        opponent = self.opponent_pool.sample_opponent(stage_config)
        opponent_type = self._identify_opponent_type(opponent)
        
        # 重置环境
        target_ball = 'solid' if self.global_episode % 2 == 0 else 'stripe'
        self.env.reset(target_ball=target_ball)
        
        # Episode 追踪器
        tracker = EpisodeTracker()
        
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            current_player = self.env.get_curr_player()
            
            if current_player == 'A':  # SAC agent 回合
                # 编码状态
                state = self.state_encoder.encode_from_env(self.env, 'A')
                
                # 选择动作
                action = self.sac_agent.select_action(state, deterministic=False)
                
                # 记录击球前的状态
                my_type = self.env.player_targets['A'][0]
                my_ball_ids = get_ball_ids_by_type(my_type)
                enemy_type = 'stripe' if my_type == 'solid' else 'solid'
                enemy_ball_ids = get_ball_ids_by_type(enemy_type)
                
                my_balls_before = count_remaining_balls(self.env.balls, my_ball_ids)
                enemy_balls_before = count_remaining_balls(self.env.balls, enemy_ball_ids)
                
                # 执行动作
                from config import denormalize_action
                action_dict = denormalize_action(action)
                shot_result = self.env.take_shot(**action_dict)
                
                # 计算奖励
                game_done, info = self.env.get_done()
                i_won = None
                if game_done:
                    i_won = (info.get('winner') == 'A')
                
                reward = self.reward_shaper.calculate_immediate_reward(
                    shot_result, my_balls_before, enemy_balls_before,
                    game_done, i_won
                )
                
                next_state = self.state_encoder.encode_from_env(self.env, 'A')
                
                # 存储 transition
                buffer_idx = self.replay_buffer.push(
                    state, action, reward, next_state, game_done,
                    meta_info={'is_sac_turn': True}
                )
                tracker.add_transition(buffer_idx, is_sac_turn=True)
                
                episode_reward += reward
                episode_length += 1
                done = game_done
                
            else:  # 对手回合
                balls, my_type, table = self.env.get_observation()
                action_dict = opponent.decision(balls, my_type, table)
                shot_result = self.env.take_shot(**action_dict)
                
                # 检查对手是否失误，追溯防守奖励
                if shot_result.get('WHITE_BALL_INTO_POCKET') or \
                   shot_result.get('FOUL_FIRST_HIT') or \
                   shot_result.get('NO_POCKET_NO_RAIL') or \
                   shot_result.get('NO_HIT') or \
                   not shot_result.get('ME_INTO_POCKET'):
                    
                    last_sac_idx = tracker.get_last_sac_turn_idx()
                    if last_sac_idx is not None:
                        defense_reward = self.reward_shaper.calculate_defense_reward(shot_result)
                        self.replay_buffer.add_defense_reward(last_sac_idx, defense_reward)
                        episode_reward += defense_reward
                
                tracker.add_transition(-1, is_sac_turn=False)
                done = self.env.get_done()[0]
            
            # 训练更新
            if len(self.replay_buffer) >= SAC_CONFIG['warmup_steps']:
                for _ in range(SAC_CONFIG['gradient_steps']):
                    self.sac_agent.update(self.replay_buffer, SAC_CONFIG['batch_size'])
        
        # 记录胜负
        game_done, info = self.env.get_done()
        if game_done:
            winner = info.get('winner')
            self.game_counts[opponent_type] += 1
            if winner == 'A':
                self.win_counts[opponent_type] += 1
        
        return {
            'reward': episode_reward,
            'length': episode_length,
            'opponent_type': opponent_type,
            'won': (winner == 'A') if game_done else False
        }
    
    def _identify_opponent_type(self, opponent):
        """识别对手类型"""
        from agent import BasicAgent, NewAgent
        if isinstance(opponent, BasicAgent):
            return 'basic'
        elif isinstance(opponent, NewAgent):
            return 'physics'
        else:
            return 'self'
    
    def _evaluate(self):
        """评估当前模型"""
        print("\n" + "-" * 60)
        print(f"评估中... (Episode {self.global_episode})")
        
        # 切换到确定性策略
        self.sac_wrapper.set_deterministic(True)
        
        results = {}
        for opponent_type in ['basic', 'physics']:
            wins = 0
            games = EVAL_CONFIG['eval_games']
            
            for i in range(games):
                opponent = self.opponent_pool.get_opponent(opponent_type)
                target_ball = 'solid' if i % 2 == 0 else 'stripe'
                self.env.reset(target_ball=target_ball)
                
                done = False
                while not done:
                    current_player = self.env.get_curr_player()
                    
                    if current_player == 'A':
                        balls, my_type, table = self.env.get_observation()
                        action_dict = self.sac_wrapper.decision(balls, my_type, table)
                        self.env.take_shot(**action_dict)
                    else:
                        balls, my_type, table = self.env.get_observation()
                        action_dict = opponent.decision(balls, my_type, table)
                        self.env.take_shot(**action_dict)
                    
                    done = self.env.get_done()[0]
                
                game_done, info = self.env.get_done()
                if info.get('winner') == 'A':
                    wins += 1
            
            winrate = wins / games
            results[f'{opponent_type}_winrate'] = winrate
            print(f"  vs {opponent_type}: {wins}/{games} = {winrate:.1%}")
        
        # 恢复随机策略
        self.sac_wrapper.set_deterministic(False)
        
        return results
    
    def _check_stage_completion(self, stage_config, eval_results):
        """检查阶段是否提前完成"""
        target_metrics = stage_config.get('target_metrics', {})
        
        for metric_name, target_value in target_metrics.items():
            if eval_results.get(metric_name, 0) >= target_value:
                return True
        
        return False
    
    def _should_skip_stage(self, stage_name):
        """判断是否应该跳过某个阶段（用于恢复训练）"""
        # 简单实现：按顺序训练
        stages = list(TRAINING_STAGES.keys())
        current_idx = stages.index(self.current_stage)
        target_idx = stages.index(stage_name)
        return target_idx < current_idx
    
    def _log_episode(self, episode_info):
        """记录 episode 信息"""
        if self.global_episode % EVAL_CONFIG['log_frequency'] == 0:
            stats = self.sac_agent.get_statistics()
            print(f"\nEpisode {self.global_episode} (Stage: {self.current_stage}, {self.stage_episode})")
            print(f"  Reward: {episode_info['reward']:.2f}")
            print(f"  Length: {episode_info['length']}")
            print(f"  Opponent: {episode_info['opponent_type']}")
            print(f"  Alpha: {stats.get('alpha_mean', 0):.4f}")
            print(f"  Buffer: {len(self.replay_buffer)}")
    
    def _log_evaluation(self, eval_results, final=False):
        """记录评估结果"""
        prefix = "最终评估" if final else "评估结果"
        print(f"\n{prefix} (Episode {self.global_episode}):")
        for metric, value in eval_results.items():
            print(f"  {metric}: {value:.1%}")
    
    def _save_checkpoint(self, is_final=False):
        """保存 checkpoint"""
        filename = f"final_model.pth" if is_final else f"checkpoint_ep{self.global_episode}.pth"
        filepath = os.path.join(CHECKPOINT_CONFIG['save_dir'], filename)
        
        self.sac_agent.save(filepath)
        print(f"💾 Checkpoint 已保存: {filepath}")
    
    def _load_checkpoint(self, filepath):
        """加载 checkpoint"""
        self.sac_agent.load(filepath)
        print(f"📂 Checkpoint 已加载: {filepath}")


# ==================== 主函数 ====================
def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='训练 SAC Agent')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的 checkpoint 路径')
    parser.add_argument('--test', action='store_true', help='使用测试配置（少量 episodes）')
    args = parser.parse_args()
    
    # 如果是测试模式，修改配置
    if args.test:
        print("⚠️  测试模式：使用减少的 episodes")
        from config import get_quick_test_config
        TRAINING_STAGES.update(get_quick_test_config())
    
    # 创建训练器
    trainer = SACTrainer(resume_from=args.resume)
    
    # 开始训练
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        trainer._save_checkpoint()
        print("模型已保存")
    except Exception as e:
        print(f"\n\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
        trainer._save_checkpoint()
        print("模型已保存")


if __name__ == '__main__':
    main()
