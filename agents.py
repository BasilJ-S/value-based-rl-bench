
from hmac import new
from tokenize import Double
import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch.optim.rmsprop
from tqdm import tqdm


class DQN:
    def __init__(self, env: gym.Env, lr: float, gamma: float = 0.99, device = 'cpu', seed: int = 23):
        self.env = env
        self.lr = lr
        self.seed = seed
        self.device = device
        self.seed_model(seed)
        self.gamma = gamma

        # We have 1d observations for this assignment, but adding more for more general case
        self.observation_size = 1
        for x in env.observation_space.shape:
            self.observation_size = self.observation_size * x
        
        self.actions = list(range(env.action_space.n))
        self.Q = torch.nn.Sequential(
            torch.nn.Linear(self.observation_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, env.action_space.n)
        )

        self.reinit_weights()
        
        self.Q.to(device)
        self.optimizer = torch.optim.RMSprop(self.Q.parameters(), lr=lr)
    
    # Batch is a list of tuples from the replay buffer
    # All will be numpy arrays
    def update(self, batch) -> None:
        # Batch is a list of tuples from the replay buffer
        # Update the Q function
        # Every element 0 of the tuple is the observation. We need to stack them to get a tensor of observations
        # Format is (observation, action, reward, observation_prime, terminated or truncated)
        batch = list(zip(*batch))
        states = torch.stack([torch.from_numpy(s).float() for s in batch[0]])
        actions = torch.tensor(batch[1])
        rewards = torch.tensor(batch[2])
        next_states = torch.stack([torch.from_numpy(s).float() for s in batch[3]])
        terminated = torch.tensor(batch[4], dtype=torch.bool)
        #print(terminated)

        # Get the next values
        with torch.no_grad():
            next_values = torch.max(self.Q.forward(next_states), dim=1).values

        # Use terminated to mask next_values
        next_values[terminated] =0 
        y = rewards + self.gamma * next_values

        value_estimates = self.Q.forward(states)
        state_value_estimates = value_estimates.gather(1, actions.unsqueeze(1)).squeeze(1)

        #print(state_value_estimates.shape)
        #print(y.shape)
        
        self.optimizer.zero_grad()
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(state_value_estimates, y)
        #print("LOSS: ",loss)
        loss.backward()
        self.optimizer.step()

        
    def seed_model(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed

    def plot_reward(self, rewards: list) -> None:
        fig, axs = plt.subplots(2,2, figsize=(10,5))
        #Plot the rewards
        
        axs[0,0].plot(rewards)
        axs[0,0].set_title("Reward")
        axs[0,0].set_xlabel("Time")
        axs[0,0].set_ylabel("Reward")

        #Plot the cumulative reward
        cumulative = [0]
        for i in range(len(rewards)):
            cumulative.append(cumulative[i]+rewards[i])
        axs[1,0].plot(cumulative)
        axs[1,0].set_title("Cumulative Reward")
        axs[1,0].set_xlabel("Episode")
        axs[1,0].set_ylabel("Reward")

        #Plot the moving average
        moving_average = []
        for i in range(len(rewards)):
            moving_average.append(np.mean(rewards[max(0,i-5):min(len(rewards),i+5)]))
        axs[0,1].plot(moving_average)
        axs[0,1].set_title("Moving Average")
        axs[0,1].set_xlabel("Episode")
        axs[0,1].set_ylabel("Reward")


        #Space out the subplots by a bit
        plt.tight_layout()
        plt.show()

    def reinit_weights(self) -> None:
        for layer in self.Q:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.uniform_(layer.weight, -0.001, 0.001)
                torch.nn.init.uniform_(layer.bias, -0.001, 0.001)

    def select_action(self, observation: torch.Tensor) -> int:
        # Take in observation from env, return action
        with torch.no_grad():
            estimates = self.Q.forward(observation)
        return torch.argmax(estimates).item()


    def train(self, num_episodes: int, episode_len: int, epsilon: float = 0.3, use_buffer = False, 
              replay: int = 1e7, batch_size: int = 32, plot_results = False) -> None:
        # Collect episode 
        # update replay buffer if you have one
        # update the Neural network 
        # Replay and batch size only used if use_buffer is True

        self.seed_model(self.seed)

        D = []
        rewards = []

        for episode in tqdm(range(num_episodes)):
            if False and episode == num_episodes - 1:
                new_env = gym.make(self.env.spec.id, render_mode='human')
                self.env.close()
                self.env = new_env
            else:
                #Don't render the other episodes
                pass
            episode_rewards = []
            observation, _ = self.env.reset(seed=self.seed + episode) # ADD epsiode so the seed is different for each episode
            #print(type(observation))
            #print(observation.shape)
            for t in range(episode_len):
                rand = np.random.rand() 
                if rand < epsilon:
                    action = np.random.randint(0,self.env.action_space.n)
                else:
                    action = self.select_action(torch.from_numpy(observation).float().to(self.device))
                
                #print("Action off device: ", action)
                observation_prime, reward, terminated, truncated, info = self.env.step(action)                    

                episode_rewards.append(reward)
                
                # Replay buffer
                if use_buffer:
                    D.append((observation, action, reward, observation_prime, terminated or truncated))
                    if len(D) > replay:
                        D.pop(0)
                    if len(D) > batch_size:
                        try:
                            # I THINK IT's looking across dimensions. Look later
                            batch_indexes = np.random.choice(len(D), batch_size)
                            batch = [D[i] for i in batch_indexes]
                        except:
                            print("TIME: ",t)
                            print(D[0])
                            exit()
                    else:
                        batch = D
                # No replay buffer
                else:
                    batch = [(observation, action, reward, observation_prime, terminated or truncated)]
                    
                self.update(batch)
                
                observation = observation_prime
                if terminated or truncated:
                    break
            rewards.append(sum(episode_rewards))

        self.env.close()
        if plot_results:
            self.plot_reward(rewards)


                    
                    


    
