from collections import deque
from multiprocessing import reduction
import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch.optim.rmsprop
from tqdm import tqdm

### INCOMPLETE AGENTS - TO BE COMPLETED SOME OTHER TIME ###


class REINFORCE:
    def __init__(self, env: gym.Env, lr: float, gamma: float = 0.99, temp = 1, device = 'cpu', seed: int = 23):
        self.env = env
        self.lr = lr
        self.seed = seed
        self.device = device
        self.seed_model(seed)
        self.gamma = gamma
        self.temp = temp

        # We have 1d observations for this assignment, but adding more for more general case
        self.observation_size = 1
        for x in env.observation_space.shape:
            self.observation_size = self.observation_size * x
        
        self.actions = list(range(env.action_space.n))
        self.z = torch.nn.Sequential(
            torch.nn.Linear(self.observation_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, env.action_space.n)
        )

        self.reinit_weights()
        
        self.z.to(device)
        self.optimizer = torch.optim.RMSprop(self.z.parameters(), lr=lr)
    
    #TODO - Implement this function
    def update(self, batch) -> None:
        # Batch is a list of tuples for each set of the episode
        # All will be numpy
        # Update the Q function
        # Every element 0 of the tuple is the observation. We need to stack them to get a tensor of observations
        # Format is (observation, action, reward, observation_prime, terminated or truncated)
        batch = list(zip(*batch))
        states = torch.stack([torch.from_numpy(s).float() for s in batch[0]])
        actions = torch.tensor(batch[1])
        rewards = torch.tensor(batch[2]).unsqueeze(1)
        next_states = torch.stack([torch.from_numpy(s).float() for s in batch[3]])
        terminated = torch.tensor(batch[4], dtype=torch.bool)
        #print(terminated)
     
        # Find returns from each action
        # NEEd to think about how this works with episodes
        tri = torch.triu(torch.ones(len(rewards),len(rewards)))
        tri = [[tri[j,i] * (0.99 ** (i-j)) for i in range(len(rewards))] for j in range(len(rewards))]
        tri = torch.tensor(tri, dtype=torch.float32)
        rewards = torch.matmul(tri, rewards).squeeze(1).type(torch.long)


        # Get the next values
        with torch.no_grad():
            action_probabilities = self.z.forward(states)
            action_probabilities = torch.nn.functional.softmax(action_probabilities, dim=1)
            log_probs = torch.log(action_probabilities)
            log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1).type(torch.long)


        #print(state_value_estimates.shape)
        #print(y.shape)
        
        self.optimizer.zero_grad()
        loss_fn = torch.nn.NLLLoss(reduction='sum')
        print(log_probs.shape, rewards.shape)
        
        
        loss = loss_fn(log_probs, rewards)
        loss.backward()
        self.optimizer.step()


    # Complete
    def seed_model(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed

    # Complete
    def reinit_weights(self) -> None:
        for layer in self.z:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.uniform_(layer.weight, -0.001, 0.001)
                torch.nn.init.uniform_(layer.bias, -0.001, 0.001)

    # Complete
    def softmax(self, x, temp):
        z = x - np.max(x)
        # write your solution here
        sum = np.sum(np.e ** (np.divide(z,temp)))
        softmax = (np.e ** (z/temp)) / (sum + 0.000000001)
        return softmax
    
    # Complete
    def select_action(self, observation: torch.Tensor) -> int:
        # Take in observation from env, return action
        z_vals = self.z.forward(observation).detach().numpy()
        pi_vals = self.softmax(z_vals,self.temp)
        action = np.random.choice(self.actions, p = pi_vals)
        return action      

    # Complete
    def train(self, num_episodes: int, episode_len: int, render = True) -> None:
        # Collect episode 
        # update replay buffer if you have one
        # update the Neural network 
        # Replay and batch size only used if use_buffer is True

        self.seed_model(self.seed)

        rewards = []

        for episode in tqdm(range(num_episodes), leave=False, desc="Episodes"):
            D = deque(maxlen=episode_len)
            if render and episode == num_episodes - 1:
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
            t = -1
            while t < episode_len:
                t += 1
                action = self.select_action(observation)
                
                #print("Action off device: ", action)
                observation_prime, reward, terminated, truncated, info = self.env.step(action)                    

                episode_rewards.append(reward)
                
                # Replay buffer
                D.append((observation, action, reward, observation_prime, terminated or truncated))
                                    
                observation = observation_prime
                if terminated or truncated:
                    break
            rewards.append(sum(episode_rewards))
            self.update(D)

        self.env.close()
        return rewards