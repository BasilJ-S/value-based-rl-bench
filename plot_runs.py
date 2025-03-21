from turtle import pos
import matplotlib.pyplot as plt
import json
import numpy as np


files = ['ExpectedSarsa_Acrobot_results.json', 'DQN_Acrobot_results.json','ExpectedSarsa_Assault_results.json', 'DQN_Assault_results.json']

fig_acrobot, axs_acrobot = plt.subplots(2,2, figsize=(10,7))
fig_nobuffer_acrobot, axs_nobuffer_acrobot = plt.subplots(2,2, figsize=(10,7))

fig_assault, axs_assault = plt.subplots(2,2, figsize=(10,7))
fig_nobuffer_assault, axs_nobuffer_assault = plt.subplots(2,2, figsize=(10,7))

linestyles = ['solid', 'dashed', 'dashdot', 'dotted', (0,(1,1)),(5,(10,3)),(0, (5, 1)), (0,(3,1,1,1)), (0,(3,1,1,1,1,1))]
dqncount = 0
escount = 0


for file in files:
    try:
        with open(file) as f:
            rewards = json.load(f)
    except:
        print(f"{file} not found")
        continue

    for hyperparams in rewards:
        #Plot the rewards
        # Set label to the hyperparameters
        means = np.mean(rewards[hyperparams], axis=0)
        stds = np.std(rewards[hyperparams], axis=0)
        if hyperparams.split(',')[2] == 'True)':
            if 'Acrobot' in file:
                ax = axs_acrobot
            else:
                ax = axs_assault
        else:
            if 'Acrobot' in file:
                ax = axs_nobuffer_acrobot
            else:
                ax = axs_nobuffer_assault
        if 'ExpectedSarsa' in file:
            hyperparams = f"ExpectedSarsa {hyperparams}"
            colour = 'red'
            linestyle = linestyles[escount%len(linestyles)]
            escount += 1
        else:
            hyperparams = f"DQN {hyperparams}"
            colour = 'green'
            linestyle = linestyles[dqncount%len(linestyles)]
            dqncount += 1
    

        ax[0,0].plot(means, label = hyperparams, linestyle = linestyle, color = colour)
        ax[0,0].set_title("Reward")
        ax[0,0].set_xlabel("Episode")
        ax[0,0].set_ylabel("Reward")
        ax[0,0].fill_between(range(len(means)), means-stds, means+stds, alpha=0.1, color = colour)
        #ax[0,0].legend(loc='upper left', bbox_to_anchor=(1.02, 1))


        #Plot the cumulative reward
        cumulative = np.cumsum(means)
        std_devs = np.cumsum(stds)
        ax[1,0].plot(cumulative, label = hyperparams, linestyle = linestyle, color = colour)
        ax[1,0].set_title("Cumulative Reward")
        ax[1,0].set_xlabel("Episode")
        ax[1,0].set_ylabel("Reward")
        ax[1,0].fill_between(range(len(cumulative)), cumulative-std_devs, cumulative+std_devs, alpha=0.1, color = colour)
        #ax[1,0].legend(loc='upper left', bbox_to_anchor=(1.02, 1))

        #Plot the moving average
        moving_average = []
        avg_std_dev = []
        average_over = 20
        for i in range(len(means)):
            moving_average.append(np.mean(means[max(0,i-average_over):min(len(means),i+average_over)]))
            avg_std_dev.append(np.mean(stds[max(0,i-average_over):min(len(means),i+average_over)]))
        ax[0,1].plot(moving_average, label = hyperparams, linestyle = linestyle, color = colour)
        ax[0,1].set_title("Moving Average")
        ax[0,1].set_xlabel("Episode")
        ax[0,1].set_ylabel("Reward")
        ax[0,1].fill_between(range(len(moving_average)), np.array(moving_average)-np.array(avg_std_dev), np.array(moving_average)+np.array(avg_std_dev), alpha=0.1, color = colour)
        #ax[0,1].legend(loc='upper left', bbox_to_anchor=(1.02, 1))

        #Plot the average reward over the last 100 episodes
        avg100 = np.mean(means[-100:])
        ax[1,1].bar(hyperparams, avg100)
        ax[1,1].set_title("Average Reward over Last 100 Episodes")
        ax[1,1].set_ylabel("Reward")
        ax[1,1].set_xlabel("Hyperparameters")  
i = 0
figurenames = ['Acrobot - With Replay Buffer', 'Acrobot - Without Replay Buffer', 'Assault - With Replay Buffer', 'Assault - Without Replay Buffer']
for fig,axs in [(fig_acrobot, axs_acrobot), (fig_nobuffer_acrobot, axs_nobuffer_acrobot), (fig_assault, axs_assault), (fig_nobuffer_assault, axs_nobuffer_assault)]:

    axs[1,1].set_xticks(axs[1,1].get_xticks())  # Ensure ticks are set
    axs[1,1].set_xticklabels(axs[1,1].get_xticklabels(), rotation=45, ha='right', fontsize=6)  # Rotate and align ticks
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.3, 1))

    fig.suptitle(f"{figurenames[i]}")
    #Space out the subplots by a bit
    fig.tight_layout()
    fig.savefig(f'figure_{i}.png', bbox_inches='tight', dpi = 300)
    i += 1


for file in files:
    try:
        with open(file) as f:
            rewards = json.load(f)
    except:
        print(f"{file} not found")
        continue

    fig, axs = plt.subplots(1,2, figsize=(10,4))

    for hyperparams in rewards:
        #Plot the rewards
        # Set label to the hyperparameters
        means = np.mean(rewards[hyperparams], axis=0)
        stds = np.std(rewards[hyperparams], axis=0)
        if hyperparams.split(',')[2] == 'True)':
            plot = 0
            title = 'Moving Average, With Buffer'
        else:
            plot = 1
            title = 'Moving Average, Without Buffer'
    

        #Plot the moving average
        moving_average = []
        avg_std_dev = []
        average_over = 20
        for i in range(len(means)):
            moving_average.append(np.mean(means[max(0,i-average_over):min(len(means),i+average_over)]))
            avg_std_dev.append(np.mean(stds[max(0,i-average_over):min(len(means),i+average_over)]))
        axs[plot].plot(moving_average, label = hyperparams)
        axs[plot].set_title(title)
        axs[plot].set_xlabel("Episode")
        axs[plot].set_ylabel("Reward")
        axs[plot].fill_between(range(len(moving_average)), np.array(moving_average)-np.array(avg_std_dev), np.array(moving_average)+np.array(avg_std_dev), alpha=0.1)
        #axs[plot].legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    limits = axs[0].get_ylim()
    axs[1].set_ylim(limits)
    handles, labels = axs[0].get_legend_handles_labels()
    labels = [f"{l.split(',')[0]},{l.split(',')[1]})" for l in labels]
    # map to string
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.2, 1))
    #Space out the subplots by a bit
    fig.suptitle(file)
    fig.tight_layout()
    fig.savefig(f'{file}.png', bbox_inches='tight', dpi = 300)
    i += 1