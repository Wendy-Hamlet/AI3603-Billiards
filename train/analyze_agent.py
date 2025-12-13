"""
analyze_agent.py - Agent决策分析工具

分析训练中的agent行为：
1. 动作分布分析
2. 状态-动作相关性
3. 决策可视化
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from train.sac.sac_agent import SACAgent
from train.environment.pool_wrapper import create_env


def analyze_action_distribution(agent, env, n_games=20, verbose=False):
    """分析agent的动作分布"""
    print("=" * 60)
    print("动作分布分析")
    print("=" * 60)
    
    all_actions = []
    all_states = []
    
    # 根据动作维度确定动作名称
    action_dim = env.action_dim
    if action_dim == 2:
        action_names = ['V0', 'phi']
    else:
        action_names = ['V0', 'phi', 'theta', 'a', 'b']
    
    for game in range(n_games):
        result = env.reset()
        state = result[0] if isinstance(result, tuple) else result
        done = False
        
        while not done:
            # 获取动作
            action = agent.select_action(state, deterministic=True)
            all_actions.append(action)
            all_states.append(state)
            
            # 执行动作
            step_result = env.step(action)
            next_state = step_result[0]
            reward = step_result[1]
            done = step_result[2]
            info = step_result[3]
            state = next_state
    
    actions = np.array(all_actions)
    states = np.array(all_states)
    
    print(f"\n收集了 {len(actions)} 个动作样本")
    print("\n动作统计 (归一化范围 [-1, 1]):")
    print("-" * 50)
    
    for i, name in enumerate(action_names):
        print(f"{name:8s}: 均值={actions[:, i].mean():7.3f}, "
              f"标准差={actions[:, i].std():6.3f}, "
              f"范围=[{actions[:, i].min():6.3f}, {actions[:, i].max():6.3f}]")
    
    # 转换为实际参数值
    print("\n实际参数值:")
    print("-" * 50)
    
    if action_dim == 2:
        ranges = {
            'V0': (1.0, 6.0),
            'phi': (0.0, 360.0),
        }
    else:
        ranges = {
            'V0': (0.5, 8.0),
            'phi': (0.0, 360.0),
            'theta': (0.0, 45.0),
            'a': (-0.4, 0.4),
            'b': (-0.4, 0.4)
        }
    
    for i, name in enumerate(action_names):
        low, high = ranges[name]
        actual = (actions[:, i] + 1) * 0.5 * (high - low) + low
        print(f"{name:8s}: 均值={actual.mean():7.2f}, "
              f"标准差={actual.std():6.2f}, "
              f"范围=[{actual.min():6.2f}, {actual.max():6.2f}]")
    
    return actions, states


def analyze_state_action_correlation(actions, states):
    """分析状态与动作的相关性"""
    print("\n" + "=" * 60)
    print("状态-动作相关性分析")
    print("=" * 60)
    
    action_names = ['V0', 'phi', 'theta', 'a', 'b']
    
    # 分析白球位置与phi角度的关系
    cue_x = states[:, 0]  # 白球x
    cue_y = states[:, 1]  # 白球y
    
    phi_values = (actions[:, 1] + 1) * 180  # 转换为角度
    
    # 计算白球位置与击球角度的相关系数
    corr_x_phi = np.corrcoef(cue_x, phi_values)[0, 1]
    corr_y_phi = np.corrcoef(cue_y, phi_values)[0, 1]
    
    print(f"\n白球X位置 vs Phi角度 相关系数: {corr_x_phi:.3f}")
    print(f"白球Y位置 vs Phi角度 相关系数: {corr_y_phi:.3f}")
    
    # 分析与最近目标球的关系
    # 己方球从 state[5:26] (7*3=21维)
    my_balls_start = 5
    nearest_ball_x = states[:, my_balls_start]  # 最近球的x
    nearest_ball_y = states[:, my_balls_start + 1]  # 最近球的y
    
    # 计算白球到最近目标球的方向
    dx = nearest_ball_x - cue_x
    dy = nearest_ball_y - cue_y
    target_angle = np.arctan2(dy, dx) * 180 / np.pi
    target_angle = (target_angle + 360) % 360
    
    corr_target_phi = np.corrcoef(target_angle, phi_values)[0, 1]
    print(f"目标球方向 vs Phi角度 相关系数: {corr_target_phi:.3f}")
    
    if abs(corr_target_phi) < 0.3:
        print("\n⚠️ 警告：Agent的击球方向与目标球位置相关性很低！")
        print("   这表明Agent可能没有学会将球打向目标方向")


def analyze_shot_effectiveness(agent, env, n_games=10, verbose=True):
    """分析击球效果"""
    print("\n" + "=" * 60)
    print("击球效果分析")
    print("=" * 60)
    
    stats = {
        'total_shots': 0,
        'balls_pocketed': 0,
        'own_pocketed': 0,
        'enemy_pocketed': 0,
        'fouls': 0,
        'cue_pocketed': 0,
        'no_hit': 0,
    }
    
    for game in range(n_games):
        if verbose:
            print(f"\n--- Game {game + 1} ---")
        
        result = env.reset()
        state = result[0] if isinstance(result, tuple) else result
        done = False
        
        while not done:
            action = agent.select_action(state, deterministic=True)
            step_result = env.step(action)
            next_state, reward, done, info = step_result[0], step_result[1], step_result[2], step_result[3]
            
            stats['total_shots'] += 1
            
            # 分析 shot_result
            shot_result = info.get('shot_result', {})
            
            own = shot_result.get('ME_INTO_POCKET', [])
            enemy = shot_result.get('ENEMY_INTO_POCKET', [])
            
            stats['own_pocketed'] += len(own)
            stats['enemy_pocketed'] += len(enemy)
            stats['balls_pocketed'] += len(own) + len(enemy)
            
            if shot_result.get('WHITE_BALL_INTO_POCKET', False):
                stats['cue_pocketed'] += 1
            if shot_result.get('NO_HIT', False):
                stats['no_hit'] += 1
            if shot_result.get('FOUL_FIRST_HIT', False) or shot_result.get('NO_POCKET_NO_RAIL', False):
                stats['fouls'] += 1
            
            if verbose and (len(own) > 0 or len(enemy) > 0):
                print(f"  Step: Own pocketed: {own}, Enemy pocketed: {enemy}")
            
            state = next_state
        
        if verbose:
            print(f"  Game result: Winner={info.get('winner')}, Reward={reward:.1f}")
    
    print(f"\n统计摘要 ({n_games} 场游戏):")
    print("-" * 50)
    print(f"总击球数: {stats['total_shots']}")
    print(f"己方进球: {stats['own_pocketed']} ({100*stats['own_pocketed']/stats['total_shots']:.1f}%)")
    print(f"对方进球: {stats['enemy_pocketed']} ({100*stats['enemy_pocketed']/stats['total_shots']:.1f}%)")
    print(f"白球进袋: {stats['cue_pocketed']} ({100*stats['cue_pocketed']/stats['total_shots']:.1f}%)")
    print(f"空杆: {stats['no_hit']} ({100*stats['no_hit']/stats['total_shots']:.1f}%)")
    print(f"其他犯规: {stats['fouls']} ({100*stats['fouls']/stats['total_shots']:.1f}%)")
    
    return stats


def analyze_action_entropy(agent, env, n_samples=500):
    """分析动作熵（探索程度）"""
    print("\n" + "=" * 60)
    print("动作熵分析")
    print("=" * 60)
    
    states = []
    result = env.reset()
    state = result[0] if isinstance(result, tuple) else result
    done = False
    
    while len(states) < n_samples:
        states.append(state)
        action = agent.select_action(state, deterministic=False)
        step_result = env.step(action)
        next_state, done = step_result[0], step_result[2]
        state = next_state
        if done:
            result = env.reset()
            state = result[0] if isinstance(result, tuple) else result
    
    states = np.array(states)
    
    # 对同一状态采样多次，看动作变化
    test_state = states[0:1]
    repeated_actions = []
    
    for _ in range(100):
        action = agent.select_action(test_state[0], deterministic=False)
        repeated_actions.append(action)
    
    repeated_actions = np.array(repeated_actions)
    
    print("\n同一状态下100次采样的动作分布:")
    print("-" * 50)
    action_dim = env.action_dim
    if action_dim == 2:
        action_names = ['V0', 'phi']
    else:
        action_names = ['V0', 'phi', 'theta', 'a', 'b']
    for i, name in enumerate(action_names):
        std = repeated_actions[:, i].std()
        print(f"{name:8s}: 标准差={std:.4f}")
    
    mean_std = repeated_actions.std(axis=0).mean()
    print(f"\n平均标准差: {mean_std:.4f}")
    
    if mean_std < 0.1:
        print("\n⚠️ 警告：动作熵很低，Agent可能陷入了确定性策略")
        print("   建议增加 alpha 或使用更高的探索噪声")


def visualize_typical_shots(agent, env, n_games=3):
    """可视化典型击球"""
    print("\n" + "=" * 60)
    print("典型击球分析")
    print("=" * 60)
    
    action_dim = env.action_dim
    if action_dim == 2:
        ranges = {
            'V0': (1.0, 6.0),
            'phi': (0.0, 360.0),
        }
    else:
        ranges = {
            'V0': (0.5, 8.0),
            'phi': (0.0, 360.0),
            'theta': (0.0, 45.0),
            'a': (-0.4, 0.4),
            'b': (-0.4, 0.4)
        }
    
    for game in range(n_games):
        print(f"\n--- Game {game + 1} 前5杆 ---")
        result = env.reset()
        state = result[0] if isinstance(result, tuple) else result
        
        for shot in range(5):
            # 获取球位置
            cue_x, cue_y = state[0] * 0.99, state[1] * 1.98
            eight_x, eight_y = state[2] * 0.99, state[3] * 1.98
            nearest_x, nearest_y = state[5] * 0.99, state[6] * 1.98
            
            # Agent选择的动作
            action = agent.select_action(state, deterministic=True)
            
            # 转换为实际值
            actual = {}
            for i, (name, (low, high)) in enumerate(ranges.items()):
                actual[name] = (action[i] + 1) * 0.5 * (high - low) + low
            
            # 计算理想角度（白球到最近目标球）
            dx = nearest_x - cue_x
            dy = nearest_y - cue_y
            ideal_phi = np.arctan2(dy, dx) * 180 / np.pi
            if ideal_phi < 0:
                ideal_phi += 360
            
            print(f"\n  Shot {shot + 1}:")
            print(f"    白球位置: ({cue_x:.2f}, {cue_y:.2f})")
            print(f"    最近目标: ({nearest_x:.2f}, {nearest_y:.2f})")
            print(f"    理想角度: {ideal_phi:.1f}°")
            print(f"    Agent角度: {actual['phi']:.1f}°")
            print(f"    角度误差: {min(abs(actual['phi'] - ideal_phi), 360 - abs(actual['phi'] - ideal_phi)):.1f}°")
            print(f"    力度: {actual['V0']:.1f} m/s")
            
            step_result = env.step(action)
            next_state, reward, done, info = step_result[0], step_result[1], step_result[2], step_result[3]
            
            shot_result = info.get('shot_result', {})
            own = shot_result.get('ME_INTO_POCKET', [])
            if len(own) > 0:
                print(f"    ✓ 进球: {own}")
            
            state = next_state
            if done:
                break


def check_state_encoding(env):
    """检查状态编码是否合理"""
    print("\n" + "=" * 60)
    print("状态编码检查")
    print("=" * 60)
    
    result = env.reset()
    state = result[0] if isinstance(result, tuple) else result
    
    print(f"\n状态维度: {len(state)}")
    print("\n状态分解:")
    print("-" * 50)
    
    idx = 0
    print(f"[{idx}:{idx+2}] 白球位置: ({state[0]:.3f}, {state[1]:.3f})")
    idx = 2
    print(f"[{idx}:{idx+3}] 黑8: pos=({state[2]:.3f}, {state[3]:.3f}), pocketed={state[4]:.0f}")
    idx = 5
    print(f"[{idx}:{idx+21}] 己方7球 (x, y, pocketed):")
    for i in range(7):
        x, y, p = state[idx + i*3], state[idx + i*3 + 1], state[idx + i*3 + 2]
        status = "进袋" if p > 0.5 else "在台"
        print(f"         球{i+1}: ({x:.3f}, {y:.3f}) [{status}]")
    
    idx = 26
    print(f"[{idx}:{idx+21}] 对方7球 (x, y, pocketed):")
    for i in range(7):
        x, y, p = state[idx + i*3], state[idx + i*3 + 1], state[idx + i*3 + 2]
        status = "进袋" if p > 0.5 else "在台"
        print(f"         球{i+1}: ({x:.3f}, {y:.3f}) [{status}]")
    
    idx = 47
    print(f"[{idx}:{idx+12}] 球袋位置 (6个):")
    pocket_names = ['lb', 'lc', 'lt', 'rb', 'rc', 'rt']
    for i, name in enumerate(pocket_names):
        x, y = state[idx + i*2], state[idx + i*2 + 1]
        print(f"         {name}: ({x:.3f}, {y:.3f})")
    
    idx = 59
    print(f"[{idx}:{idx+5}] 游戏信息:")
    print(f"         己方剩余: {state[59]*7:.0f}/7")
    print(f"         对方剩余: {state[60]*7:.0f}/7")
    print(f"         击球进度: {state[61]*60:.0f}/60")
    print(f"         可打黑8: {state[62]:.0f}")
    print(f"         预留: {state[63]:.0f}")
    
    # 问题分析
    print("\n状态编码问题分析:")
    print("-" * 50)
    print("1. ❌ 缺少球到袋口的角度/距离信息")
    print("2. ❌ 缺少击球可行性评估（是否有清晰路径）")
    print("3. ❌ 球位置是绝对坐标，没有相对白球的极坐标")
    print("4. ❌ 袋口位置是固定的，可以预计算不需要编码")


def main():
    parser = argparse.ArgumentParser(description='分析Agent决策')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--own_balls', type=int, default=1, help='己方球数')
    parser.add_argument('--enemy_balls', type=int, default=1, help='对方球数')
    parser.add_argument('--state_encoder', type=str, default=None, help='状态编码器版本 (v1/v2)')
    parser.add_argument('--action_space', type=str, default=None, help='动作空间类型 (full/simple)')
    args = parser.parse_args()
    
    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 从参数或config加载环境配置
    state_encoder_version = args.state_encoder or config['state'].get('encoder_version', 'v2')
    action_space_type = args.action_space or config['action'].get('action_type', 'simple')
    
    print(f"State encoder: {state_encoder_version}")
    print(f"Action space: {action_space_type}")
    
    # 创建环境
    env = create_env(
        own_balls=args.own_balls,
        enemy_balls=args.enemy_balls,
        enable_noise=False,
        verbose=False,
        state_encoder_version=state_encoder_version,
        action_space_type=action_space_type
    )
    
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    # 从config加载网络配置
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建 agent（使用配置中的网络架构）
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=config['network']['actor']['hidden_dims'],
        device=device
    )
    
    # 加载权重
    checkpoint = torch.load(args.checkpoint, map_location=device)
    agent.actor.load_state_dict(checkpoint['actor'])
    print(f"模型加载自: {args.checkpoint}")
    print(f"训练集数: {checkpoint.get('episode', 'Unknown')}")
    
    # 分析
    check_state_encoding(env)
    actions, states = analyze_action_distribution(agent, env, n_games=20)
    analyze_state_action_correlation(actions, states)
    analyze_action_entropy(agent, env)
    analyze_shot_effectiveness(agent, env, n_games=10, verbose=False)
    visualize_typical_shots(agent, env, n_games=2)


if __name__ == '__main__':
    main()

