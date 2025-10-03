# Reinforcement Learning: DQN and Expected SARSA

**School Assignment** - COMP579 (McGill University)  
Implementation of Deep Q-Network (DQN) and Expected SARSA algorithms on Gymnasium environments.

## Overview

This project implements and compares two value-based RL algorithms:
- **DQN (Deep Q-Network)**: Uses max Q-value for bootstrapping
- **Expected SARSA**: Uses expected Q-value under ε-greedy policy for bootstrapping

Both algorithms use a neural network (128→64→actions) as the Q-function approximator.

## Environments

1. **Acrobot-v1**: Classic control task (6D state space, 3 actions)
2. **ALE/Assault-ram-v5**: Atari game using RAM state (128D state space, 7 actions)

## Key Features

- PyTorch neural network Q-function
- Optional experience replay buffer
- Epsilon-greedy exploration
- RMSprop optimizer
- Hyperparameter grid search (learning rate, epsilon, buffer on/off)

## Usage

Train agents:
```python
from agents import DQN, ExpectedSarsa
import gymnasium as gym

env = gym.make('Acrobot-v1')
agent = ExpectedSarsa(env, lr=0.001, epsilon=0.1)
rewards = agent.train(num_episodes=1000, episode_len=1e10, use_buffer=True)
```

Run experiments:
```bash
python run_experiments.py
```

Generate plots from results:
```bash
python plot_runs.py
```

## Files

- `agents.py`: DQN and ExpectedSarsa classes
- `run_experiments.py`: Hyperparameter sweep across both environments
- `plot_runs.py`: Visualization of results
- `test.py`: Unit tests
- `*.json`: Experimental results (10 runs × hyperparameters)

## Results

The `plot_runs.py` script generates figures showing:
- Episode rewards over training
- Cumulative rewards
- Moving average (window=11)
- Average reward over final 100 episodes

Results are saved as `figure_[Model]_[Env]_results.json.png`.

## Note

This implementation includes substantial provided starter code as part of a course assignment. The main contributions are in the algorithm implementations and experimental analysis.
