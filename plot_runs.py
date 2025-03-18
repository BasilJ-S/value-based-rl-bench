from turtle import pos
import matplotlib.pyplot as plt
import json
import numpy as np


files = ['ExpectedSarsa_Acrobot_results.json', 'DQN_Acrobot_results.json','ExpectedSarsa_Assault_results.json', 'DQN_Assault_results.json']


for file in files:
    try:
        with open(file) as f:
            rewards = json.load(f)
    except:
        print(f"{file} not found")
        continue

    fig, axs = plt.subplots(2,2, figsize=(10,7))


    for hyperparams in rewards:
        #Plot the rewards
        # Set label to the hyperparameters
        means = np.mean(rewards[hyperparams], axis=0)

        axs[0,0].plot(means, label = hyperparams)
        axs[0,0].set_title("Reward")
        axs[0,0].set_xlabel("Episode")
        axs[0,0].set_ylabel("Reward")
        #axs[0,0].legend(loc='upper left', bbox_to_anchor=(1.02, 1))


        #Plot the cumulative reward
        cumulative = np.cumsum(means)
        axs[1,0].plot(cumulative, label = hyperparams)
        axs[1,0].set_title("Cumulative Reward")
        axs[1,0].set_xlabel("Episode")
        axs[1,0].set_ylabel("Reward")
        #axs[1,0].legend(loc='upper left', bbox_to_anchor=(1.02, 1))

        #Plot the moving average
        moving_average = []
        for i in range(len(means)):
            moving_average.append(np.mean(means[max(0,i-5):min(len(means),i+5)]))
        axs[0,1].plot(moving_average, label = hyperparams)
        axs[0,1].set_title("Moving Average")
        axs[0,1].set_xlabel("Episode")
        axs[0,1].set_ylabel("Reward")
        #axs[0,1].legend(loc='upper left', bbox_to_anchor=(1.02, 1))

        #Plot the average reward over the last 100 episodes
        avg100 = np.mean(means[-100:])
        axs[1,1].bar(hyperparams, avg100)
        axs[1,1].set_title("Average Reward over Last 100 Episodes")
        axs[1,1].set_ylabel("Reward")
        axs[1,1].set_xlabel("Hyperparameters")  
    axs[1,1].set_xticks(axs[1,1].get_xticks())  # Ensure ticks are set
    axs[1,1].set_xticklabels(axs[1,1].get_xticklabels(), rotation=45, ha='right', fontsize=6)  # Rotate and align ticks
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.2, 1))

    fig.suptitle(f"{file[:-13]} Results")
    #Space out the subplots by a bit
    plt.tight_layout()
    plt.savefig(f'figure_{file}.png', bbox_inches='tight')