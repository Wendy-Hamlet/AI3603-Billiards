# Physics-Based Baseline Agent

This directory contains the implementation of a physics-based baseline agent for billiards.

## Overview

The physics-based agent uses geometric shot calculation and pooltool physics simulation to select shots. It achieves approximately **60% win rate** against BasicAgent, establishing a baseline understanding of the problem requirements.

## Key Features

1. **Ghost Ball Calculation**: Computes the position where the cue ball must be placed to pocket the target ball along a straight line to the pocket
2. **Cut Angle Evaluation**: Evaluates all target-pocket combinations (up to 42: 7 target balls × 6 pockets), filtering shots with extreme cut angles (>55 degrees)
3. **Path Obstacle Detection**: Checks for obstructions along both the cue-to-target and target-to-pocket paths
4. **Physics Simulation Verification**: Uses pooltool physics engine to simulate shot outcomes and verify feasibility
5. **Danger Avoidance**: Identifies and avoids risky positions involving the 8-ball and opponent balls

## Implementation

### File Structure

```
train/physics/
├── agent.py       # NewAgent implementation
└── README.md      # This file
```

### Key Methods

- `decision(balls, my_targets, table)`: Main decision method
- `_evaluate_shot()`: Evaluates a target-pocket combination
- `_optimize_with_simulation()`: Uses pooltool to verify shot outcomes
- `_path_has_obstacle_safe()`: Checks for path obstructions
- `_calculate_danger_penalty()`: Evaluates risk from dangerous balls

## Performance

- **Win Rate**: ~60% against BasicAgent
- **Approach**: Greedy one-shot-ahead with physics verification
- **Limitations**: 
  - No look-ahead planning
  - No defensive positioning
  - Missed shots due to execution noise
  - Leaves cue ball in poor positions

## Usage

```python
from train.physics.agent import NewAgent

agent = NewAgent()
action = agent.decision(balls, my_targets, table)
```

## Dependencies

- `pooltool`: Physics engine for shot simulation
- `numpy`: Numerical operations
- `math`: Mathematical functions

## References

This implementation is described in Section II-B of the project report.
