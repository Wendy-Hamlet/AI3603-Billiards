# SAC Reinforcement Learning

This directory contains the Soft Actor-Critic (SAC) reinforcement learning implementation for billiards.

## Overview

The SAC implementation includes:
- **State Encoding**: 64-dimensional state representation (Section III-A-1)
- **Reward Engineering**: Multi-component reward function (Section III-C)
- **Curriculum Learning**: Ball count curriculum (4-ball, 6-ball stages)
- **Network Architecture**: Actor-Critic with Gaussian policy

## Key Components

### 1. Reward Function (`reward_large.py`)

The reward function implements the design described in Section III-C of the report:

#### Components:
- **R_pocket**: Proportional pocketing rewards
  - Own balls: `(n_pocketed / n_remaining) × C1`, where C1=100
  - Opponent balls: `-(n_opponent_pocketed / n_opponent_remaining) × C2`, where C2=100
- **R_turn**: Turn dynamics (+10 for keeping turn, -5 for losing turn)
- **R_foul**: Foul penalties
  - Scratch (cue ball pocketed): -20
  - Wrong first hit: -15
  - No rail contact: -10
  - No contact: -25
- **R_terminal**: Terminal rewards
  - Win: +1000
  - Loss: -1000
  - Draw: -1000 (strict draw penalty)

**Note**: Defense reward was designed but not used in training (as described in report).

### 2. State Encoding (`../env/state_encoder.py`)

64-dimensional state representation:
- Cue ball: 2 dims (x, y normalized)
- Own balls: 21 dims (7 balls × 3, sorted by distance)
- Opponent balls: 21 dims (7 balls × 3, sorted by distance)
- 8-ball: 3 dims (x, y, pocketed)
- Pocket positions: 12 dims (6 pockets × 2)
- Global info: 5 dims (counts, phase, turn)

### 3. Environment Wrapper (`../env/pool_wrapper.py`)

- Self-play mode
- Ball count curriculum support
- State encoding integration

## Training Configuration

### Ball Count Curriculum

- **4-Ball Stage**: 1 cue ball, 1 own ball, 1 opponent ball, 8-ball
- **6-Ball Stage**: 2 own balls, 2 opponent balls, cue ball, 8-ball
- **Goal**: Full 16-ball game

### Key Findings

1. **Failure against BasicAgent**: 0% win rate when environment noise removed
2. **Degenerate Strategy in Self-Play**: Agent learned to avoid pocketing to run out clock
3. **Strict Draw Penalty**: Reduced draw rate from 51% to 36% for 4-ball scenario
4. **Failure to Scale**: Regression to degenerate play when scaling to 6-ball scenario

## Usage

```python
from train.sac.reward_large import RewardCalculatorLarge
from train.env.state_encoder import StateEncoder
from train.env.pool_wrapper import PoolEnvWrapper

# Create reward calculator
reward_calc = RewardCalculatorLarge(
    C1=100.0,  # Own ball value
    C2=100.0,  # Opponent ball penalty
    C3=10.0,   # Turn keeping value
    win_reward=1000.0,
    loss_reward=-1000.0,
    draw_reward=-1000.0  # Strict draw penalty
)

# State encoder
state_encoder = StateEncoder()

# Environment wrapper
env = PoolEnvWrapper(own_balls=1, enemy_balls=1)  # 4-ball curriculum
```

## Files

- `reward_large.py`: Reward function implementation
- `README.md`: This file

**Note**: Training scripts and agent implementation are in parent directories:
- State encoding: `../env/state_encoder.py`
- Environment wrapper: `../env/pool_wrapper.py`

## References

- Section III of the project report (REINFORCEMENT LEARNING WITH SAC)
- Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning" (2018)
