"""
test_mcts.py - 测试MCTS Agent

运行MCTS智能体与自己或随机对手对弈
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from poolenv import PoolEnv
from mcts.mcts_agent import MCTSAgent


def random_action() -> dict:
    """生成随机动作"""
    return {
        'V0': np.random.uniform(2.0, 5.0),
        'phi': np.random.uniform(0, 360),
        'theta': np.random.choice([0, 5, 10]),
        'a': 0.0,
        'b': 0.0
    }


def play_game(mcts_agent: MCTSAgent,
              opponent: str = 'random',
              verbose: bool = True) -> dict:
    """
    进行一局游戏
    
    Args:
        mcts_agent: MCTS智能体
        opponent: 对手类型 ('random', 'mcts', 'self')
        verbose: 是否输出详细信息
    
    Returns:
        dict: 游戏结果
    """
    env = PoolEnv(verbose=verbose)
    env.reset(target_ball='solid')
    
    # MCTS 是 Player A
    mcts_player = 'A'
    
    step = 0
    max_steps = 60
    
    stats = {
        'steps': 0,
        'mcts_pockets': 0,
        'opponent_pockets': 0,
        'winner': None
    }
    
    while step < max_steps:
        current_player = env.get_curr_player()
        balls, my_targets, table = env.get_observation()
        
        if verbose:
            print(f"\n--- Step {step + 1}, Player {current_player} ---")
            remaining = sum(1 for bid in my_targets if bid != '8' and balls[bid].state.s != 4)
            print(f"  Targets: {my_targets}, Remaining: {remaining}")
        
        # 选择动作
        if current_player == mcts_player:
            action = mcts_agent.select_action(env, current_player, verbose=verbose)
            agent_type = "MCTS"
        else:
            if opponent == 'mcts':
                # 对手也用MCTS
                action = mcts_agent.select_action(env, current_player, verbose=False)
                agent_type = "MCTS"
            else:
                # 随机对手
                action = random_action()
                agent_type = "Random"
        
        if verbose:
            print(f"  {agent_type} action: V0={action['V0']:.1f}, phi={action['phi']:.1f}")
        
        # 执行击球
        result = env.take_shot(action)
        
        # 统计进球
        own_pocketed = result.get('ME_INTO_POCKET', [])
        if len(own_pocketed) > 0:
            if current_player == mcts_player:
                stats['mcts_pockets'] += len(own_pocketed)
            else:
                stats['opponent_pockets'] += len(own_pocketed)
            
            if verbose:
                print(f"  ✓ Pocketed: {own_pocketed}")
        
        # 检查游戏结束
        game_done, game_info = env.get_done()
        if game_done:
            stats['winner'] = game_info.get('winner')
            stats['steps'] = step + 1
            
            if verbose:
                print(f"\n=== Game Over ===")
                print(f"Winner: {stats['winner']}")
                print(f"MCTS pockets: {stats['mcts_pockets']}")
                print(f"Opponent pockets: {stats['opponent_pockets']}")
            
            return stats
        
        step += 1
    
    # 超时
    stats['steps'] = max_steps
    stats['winner'] = 'SAME'
    
    if verbose:
        print(f"\n=== Game Over (Timeout) ===")
        print(f"MCTS pockets: {stats['mcts_pockets']}")
        print(f"Opponent pockets: {stats['opponent_pockets']}")
    
    return stats


def run_benchmark(n_games: int = 10,
                  simulation_budget: int = 50,
                  time_limit: float = 3.0,
                  opponent: str = 'random',
                  verbose: bool = False):
    """
    运行基准测试
    
    Args:
        n_games: 游戏局数
        simulation_budget: MCTS模拟次数
        time_limit: 时间限制
        opponent: 对手类型
        verbose: 是否输出详细信息
    """
    print("=" * 60)
    print("MCTS Agent Benchmark")
    print("=" * 60)
    print(f"  Games: {n_games}")
    print(f"  Simulation budget: {simulation_budget}")
    print(f"  Time limit: {time_limit}s")
    print(f"  Opponent: {opponent}")
    print("=" * 60)
    
    # 创建MCTS Agent
    agent = MCTSAgent(
        simulation_budget=simulation_budget,
        time_limit=time_limit,
        n_action_samples=30
    )
    
    results = {
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'total_mcts_pockets': 0,
        'total_opponent_pockets': 0,
        'total_steps': 0
    }
    
    for game in range(n_games):
        print(f"\n--- Game {game + 1}/{n_games} ---")
        
        stats = play_game(agent, opponent=opponent, verbose=verbose)
        
        if stats['winner'] == 'A':  # MCTS是A
            results['wins'] += 1
            print(f"  Result: WIN")
        elif stats['winner'] == 'SAME':
            results['draws'] += 1
            print(f"  Result: DRAW")
        else:
            results['losses'] += 1
            print(f"  Result: LOSS")
        
        results['total_mcts_pockets'] += stats['mcts_pockets']
        results['total_opponent_pockets'] += stats['opponent_pockets']
        results['total_steps'] += stats['steps']
        
        print(f"  MCTS pockets: {stats['mcts_pockets']}, Opponent: {stats['opponent_pockets']}")
    
    # 汇总
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"  Wins: {results['wins']} ({100*results['wins']/n_games:.1f}%)")
    print(f"  Losses: {results['losses']} ({100*results['losses']/n_games:.1f}%)")
    print(f"  Draws: {results['draws']} ({100*results['draws']/n_games:.1f}%)")
    print(f"  Avg MCTS pockets: {results['total_mcts_pockets']/n_games:.2f}")
    print(f"  Avg Opponent pockets: {results['total_opponent_pockets']/n_games:.2f}")
    print(f"  Avg steps: {results['total_steps']/n_games:.1f}")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='测试MCTS Agent')
    parser.add_argument('--games', type=int, default=5, help='游戏局数')
    parser.add_argument('--budget', type=int, default=50, help='MCTS模拟次数')
    parser.add_argument('--time', type=float, default=3.0, help='时间限制（秒）')
    parser.add_argument('--opponent', type=str, default='random',
                       choices=['random', 'mcts'], help='对手类型')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    parser.add_argument('--single', action='store_true', help='只玩一局（详细输出）')
    
    args = parser.parse_args()
    
    if args.single:
        # 单局详细测试
        agent = MCTSAgent(
            simulation_budget=args.budget,
            time_limit=args.time,
            n_action_samples=30
        )
        play_game(agent, opponent=args.opponent, verbose=True)
    else:
        # 基准测试
        run_benchmark(
            n_games=args.games,
            simulation_budget=args.budget,
            time_limit=args.time,
            opponent=args.opponent,
            verbose=args.verbose
        )


if __name__ == '__main__':
    main()

