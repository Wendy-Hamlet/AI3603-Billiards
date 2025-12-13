#!/usr/bin/env python3
"""
测试脚本：让训练好的SAC agent进行自对弈
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from train.sac.sac_agent import SACAgent
from train.environment.pool_wrapper import SelfPlayEnv
from train.environment.state_encoder import StateEncoder, ActionSpace

def load_agent(checkpoint_path: str, device: str = 'cuda'):
    """加载训练好的agent"""
    agent = SACAgent(
        state_dim=64,
        action_dim=5,
        hidden_dims=[512, 512, 512, 256],
        device=device
    )
    agent.load(checkpoint_path)
    agent.eval_mode()
    return agent

def test_selfplay(checkpoint_path: str, n_games: int = 10, verbose: bool = True):
    """
    让两个相同的agent进行自对弈
    
    Args:
        checkpoint_path: checkpoint路径
        n_games: 游戏局数
        verbose: 是否显示详细过程
    """
    print("=" * 70)
    print(f"加载模型: {checkpoint_path}")
    print("=" * 70)
    
    # 加载agent（使用CPU避免GPU内存问题）
    agent = load_agent(checkpoint_path, device='cpu')
    
    # 创建环境（完整游戏）
    env = SelfPlayEnv(own_balls=7, enemy_balls=7, enable_noise=False, verbose=verbose)
    
    # 统计
    stats = {
        'wins_a': 0,
        'wins_b': 0,
        'draws': 0,
        'total_steps': 0,
        'total_pockets_a': 0,
        'total_pockets_b': 0,
        'total_reward_a': 0.0,
        'total_reward_b': 0.0
    }
    
    encoder = StateEncoder()
    action_space = ActionSpace()
    
    print(f"\n开始 {n_games} 局自对弈测试...\n")
    
    for game_idx in range(n_games):
        state, current_player = env.reset()
        done = False
        steps = 0
        game_pockets_a = 0
        game_pockets_b = 0
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"游戏 {game_idx + 1}/{n_games}")
            print(f"{'='*70}")
        
        while not done:
            # 获取当前玩家
            current_player = env.current_player
            
            # 编码状态
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # 选择动作（确定性策略）
            with torch.no_grad():
                action_normalized = agent.actor.get_action(state_tensor, deterministic=True)
                action_normalized = action_normalized.squeeze(0).cpu().numpy()
            
            # 转换为环境动作
            action = action_space.from_normalized(action_normalized)
            
            # 执行动作
            next_state, reward, done, info, next_player = env.step(action_normalized)
            
            # 统计
            if current_player == 'A':
                stats['total_reward_a'] += reward
                if 'n_pocketed' in info:
                    game_pockets_a += info['n_pocketed']
            else:
                stats['total_reward_b'] += reward
                if 'n_pocketed' in info:
                    game_pockets_b += info['n_pocketed']
            
            steps += 1
            state = next_state
            current_player = next_player  # 更新当前玩家
            
            if verbose and steps % 5 == 0:
                print(f"  步骤 {steps}: Player {current_player} | 奖励: {reward:.2f}")
        
        # 游戏结束统计（从info中获取winner）
        winner = info.get('winner', 'SAME')
        if winner == 'A':
            stats['wins_a'] += 1
        elif winner == 'B':
            stats['wins_b'] += 1
        else:
            stats['draws'] += 1
        
        stats['total_steps'] += steps
        stats['total_pockets_a'] += game_pockets_a
        stats['total_pockets_b'] += game_pockets_b
        
        if verbose:
            print(f"\n游戏结束:")
            print(f"  胜者: {winner if winner != 'SAME' else '平局'}")
            print(f"  总步数: {steps}")
            print(f"  Player A 进球: {game_pockets_a}")
            print(f"  Player B 进球: {game_pockets_b}")
        else:
            print(f"游戏 {game_idx + 1}: {winner if winner != 'SAME' else '平局'} | "
                  f"步数: {steps} | A进球: {game_pockets_a} B进球: {game_pockets_b}")
    
    # 打印总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"总游戏数: {n_games}")
    print(f"Player A 胜利: {stats['wins_a']} ({stats['wins_a']/n_games*100:.1f}%)")
    print(f"Player B 胜利: {stats['wins_b']} ({stats['wins_b']/n_games*100:.1f}%)")
    print(f"平局: {stats['draws']} ({stats['draws']/n_games*100:.1f}%)")
    print(f"\n平均步数: {stats['total_steps']/n_games:.1f}")
    print(f"平均进球数 - A: {stats['total_pockets_a']/n_games:.2f}, B: {stats['total_pockets_b']/n_games:.2f}")
    print(f"平均奖励 - A: {stats['total_reward_a']/n_games:.2f}, B: {stats['total_reward_b']/n_games:.2f}")
    print("=" * 70)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='测试SAC agent自对弈')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/sac_selfplay/20251213_151113/sac_ep80000.pt',
                       help='Checkpoint路径')
    parser.add_argument('--games', type=int, default=10,
                       help='游戏局数')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细过程')
    
    args = parser.parse_args()
    
    test_selfplay(args.checkpoint, args.games, args.verbose)

