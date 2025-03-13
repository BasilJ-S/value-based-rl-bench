
import re
import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
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
        self.optimizer = torch.optim.AdamW(self.Q.parameters(), lr=lr)

    def select_action(self, obs) -> torch.Tensor:
        with torch.no_grad():
            obs = torch.Tensor(obs)
            return torch.argmax(self.Q.forward(obs))
        
    
    def update(self, batch) -> None:
        # Batch is a list of tuples from the replay buffer
        # Update the Q function
        # Every element 0 of the tuple is the observation. We need to stack them to get a tensor of observations
        # Format is (observation, action, reward, observation_prime, terminated or truncated)
        states = torch.stack([torch.Tensor(b[0]) for b in batch])
        actions = torch.tensor([b[1] for b in batch])
        rewards = torch.tensor([b[2] for b in batch])
        next_states = torch.stack([torch.Tensor(b[3]) for b in batch])
        terminated = torch.tensor([b[4] for b in batch], dtype=torch.bool)
        #print(terminated)

        # Get the next values
        next_values = torch.max(self.Q.forward(next_states), dim=1).values

        # Use terminated to mask next_values
        next_values[terminated] = 0

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
        plt.plot(rewards)
        plt.show()

        #Plot the cumulative reward
        cumulative = []
        for i in range(len(rewards)):
            cumulative.append(sum(rewards[:i+1]))
        plt.plot(cumulative)
        plt.show()

    def reinit_weights(self) -> None:
        for layer in self.Q:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.uniform_(layer.weight, -0.001, 0.001)
                torch.nn.init.uniform_(layer.bias, -0.001, 0.001)


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
            observation, _ = self.env.reset(seed=self.seed + episode) # ADD epsiode so the seed is different for each episode
            #print(type(observation))
            #print(observation.shape)
            for t in range(episode_len):
                rand = np.random.rand() 
                if rand < epsilon:
                    action = torch.randint(self.env.action_space.n, (1,))
                else:

                    estimates = self.Q.forward(torch.from_numpy(observation).float().to(self.device))
                    #print(f"TIME: {t}, Episode: {episode}, estimates: ", estimates)
                    action = torch.argmax(estimates).item() 
                
                #print("Action off device: ", action)
                observation_prime, reward, terminated, truncated, info = self.env.step(action)                    

                rewards.append(reward)
                
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

        self.env.close()
        if plot_results:
            self.plot_reward(rewards)


                    
                    


    
