import gymnasium as gym
from agents import DQN, ExpectedSarsa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# Save results dict as a json file
import json

# Hyperparams for DQN
lrs = [1e-2, 1e-3, 1e-4]
epsilons = [0.05, 0.1, 0.2]
numruns = 10
num_episodes = 1000

model = 'ExpectedSarsa'

# FOR TESTING ONLY, not full range
'''
lrs = [1/4, 1/8]
epsilons = [0.05, 0.1]
numruns = 2
num_episodes = 3
'''
envs = ['Acrobot-v1', 'ALE/Assault-ram-v5']
short_envs = ['Acrobot', 'Assault']

# Seeds, 10 seeds for 10 runs
seeds = [563133, 665248, 414684, 18048, 863607, 984126, 969211, 736317, 
 406635, 514876]

# ACROBOT

for env_name, short in zip(envs, short_envs):
    results = {}
    for use_buffer in [True, False]:
        for lr in lrs:
            for epsilon in tqdm(epsilons, desc = f"Epsilons for lr {lr} buffer {use_buffer} for {env_name}"):
                rewards = np.zeros((numruns, num_episodes))
                for i in tqdm(range(numruns), desc=f"Independent runs for {env_name}", leave = False):
                    env = gym.make(env_name)
                    # Mod in case of not enough seeds, better than crashing
                    if model == 'DQN':
                        agent = DQN(env, lr, epsilon=epsilon, seed=seeds[i%len(seeds)])
                    elif model == 'ExpectedSarsa':
                        agent = ExpectedSarsa(env, lr, epsilon=epsilon, seed=seeds[i%len(seeds)])
                    rewards[i] = agent.train(num_episodes, 1e10,render = False, plot_results=False, use_buffer=use_buffer)

                results[f"({lr},{epsilon},{use_buffer})"] = rewards

                # Convert numpy arrays to lists for JSON serialization
                # Do this after every combination of hyperparameters to avoid losing data
                results_serializable = {k: v.tolist() for k, v in results.items()}
                with open(f'{model}_{short}_results.json', 'w') as f:
                    json.dump(results_serializable, f)


