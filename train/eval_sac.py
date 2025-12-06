"""
Evaluate SAC Agent - 评估训练好的 SAC Agent
"""

import os
import sys
import argparse
import numpy as np
import torch

from config import CHECKPOINT_CONFIG, DEVICE
from state_encoder import StateEncoder
from sac_agent import SACAgent, SACAgentWrapper
from opponent_pool import OpponentPool
from poolenv import PoolEnv
from agent import BasicAgent, NewAgent


def evaluate_agent(sac_wrapper, opponent, opponent_name, n_games=40, verbose=False):
    """
    评估 SAC Agent 对阵某个对手的胜率
    
    Args:
        sac_wrapper: SACAgentWrapper
        opponent: opponent agent
        opponent_name: str
        n_games: int, 对局数
        verbose: bool
    
    Returns:
        dict: 评估结果
    """
    env = PoolEnv()
    wins = 0
    losses = 0
    ties = 0
    
    episode_lengths = []
    sac_scores = []  # SAC进球数
    opponent_scores = []
    
    print(f"\n{'='*60}")
    print(f"评估: SAC Agent vs {opponent_name}")
    print(f"对局数: {n_games}")
    print(f"{'='*60}")
    
    for game_idx in range(n_games):
        # 轮换先手和球型（保证公平性）
        target_ball = 'solid' if game_idx % 2 == 0 else 'stripe'
        sac_is_first = (game_idx % 4 < 2)
        
        env.reset(target_ball=target_ball)
        
        episode_length = 0
        done = False
        
        while not done:
            current_player = env.get_curr_player()
            
            # 判断当前是谁
            is_sac_turn = (current_player == 'A' and sac_is_first) or \
                         (current_player == 'B' and not sac_is_first)
            
            if is_sac_turn:
                balls, my_type, table = env.get_observation()
                action_dict = sac_wrapper.decision(balls, my_type, table)
                env.take_shot(**action_dict)
            else:
                balls, my_type, table = env.get_observation()
                action_dict = opponent.decision(balls, my_type, table)
                env.take_shot(**action_dict)
            
            episode_length += 1
            done = env.get_done()[0]
        
        # 统计结果
        game_done, info = env.get_done()
        winner = info.get('winner')
        
        # 判断 SAC 是 A 还是 B
        sac_player = 'A' if sac_is_first else 'B'
        
        if winner == sac_player:
            wins += 1
        elif winner == 'SAME':
            ties += 1
        else:
            losses += 1
        
        episode_lengths.append(episode_length)
        
        if verbose or (game_idx + 1) % 10 == 0:
            print(f"  Game {game_idx+1}/{n_games}: Winner={winner}, Length={episode_length}")
    
    # 计算统计
    winrate = wins / n_games
    
    results = {
        'opponent': opponent_name,
        'n_games': n_games,
        'wins': wins,
        'losses': losses,
        'ties': ties,
        'winrate': winrate,
        'avg_episode_length': np.mean(episode_lengths),
        'std_episode_length': np.std(episode_lengths),
    }
    
    print(f"\n{'='*60}")
    print(f"评估结果:")
    print(f"  胜: {wins}, 负: {losses}, 平: {ties}")
    print(f"  胜率: {winrate:.1%}")
    print(f"  平均回合数: {results['avg_episode_length']:.1f} ± {results['std_episode_length']:.1f}")
    print(f"{'='*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='评估 SAC Agent')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型 checkpoint 路径')
    parser.add_argument('--opponent', type=str, default='all',
                       choices=['basic', 'physics', 'all'],
                       help='对手类型')
    parser.add_argument('--games', type=int, default=40, help='每个对手的对局数')
    parser.add_argument('--verbose', action='store_true', help='显示详细信息')
    args = parser.parse_args()
    
    print("="*60)
    print("SAC Agent 评估程序")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {DEVICE}")
    print(f"Games per opponent: {args.games}")
    
    # 检查 checkpoint 是否存在
    if not os.path.exists(args.checkpoint):
        print(f"\n❌ 错误: Checkpoint 不存在: {args.checkpoint}")
        return
    
    # 初始化 SAC Agent
    print("\n加载 SAC Agent...")
    state_encoder = StateEncoder()
    sac_agent = SACAgent()
    sac_agent.load(args.checkpoint)
    sac_wrapper = SACAgentWrapper(sac_agent, state_encoder)
    sac_wrapper.set_deterministic(True)  # 评估时使用确定性策略
    print("✅ SAC Agent 加载成功")
    
    # 初始化对手池
    opponent_pool = OpponentPool()
    
    # 评估对手列表
    if args.opponent == 'all':
        opponents = [
            ('basic', opponent_pool.get_opponent('basic')),
            ('physics', opponent_pool.get_opponent('physics')),
        ]
    else:
        opponents = [
            (args.opponent, opponent_pool.get_opponent(args.opponent))
        ]
    
    # 评估所有对手
    all_results = []
    for opponent_name, opponent in opponents:
        results = evaluate_agent(
            sac_wrapper,
            opponent,
            opponent_name,
            n_games=args.games,
            verbose=args.verbose
        )
        all_results.append(results)
    
    # 汇总报告
    print("\n" + "="*60)
    print("评估汇总")
    print("="*60)
    for results in all_results:
        print(f"{results['opponent']:10s}: {results['wins']:2d}/{results['n_games']:2d} = {results['winrate']:.1%}")
    print("="*60)
    
    # 保存结果
    import json
    result_file = args.checkpoint.replace('.pth', '_eval_results.json')
    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 结果已保存到: {result_file}")


if __name__ == '__main__':
    main()
