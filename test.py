from fileinput import filename
import torch
import unittest
import os
import gymnasium as gym
from agents import DQN
import numpy as np

class TestDQN(unittest.TestCase):
    env_name = 'ALE/Assault-ram-v5'
    env = gym.make(env_name)
    dqn = DQN(env, 0.001)
    # Test if the DQN class is correctly initialized
    def test_init(self):
        self.assertEqual(self.dqn.env, self.env)
        self.assertEqual(self.dqn.lr, 0.001)
        self.assertEqual(self.dqn.observation_size, 128)
        self.assertEqual(self.dqn.actions, list(range(self.env.action_space.n)))
        self.assertEqual(len(self.dqn.Q), 5)
    
    def testSelectAction(self):
        obs = torch.rand(128)
        action = self.dqn.select_action(obs)
        self.assertTrue(action in self.dqn.actions)

    def testSeed(self):
        seed = 23
        self.dqn.seed_model(seed)
        self.assertEqual(torch.initial_seed(), seed)
        self.assertEqual(np.random.get_state()[1][0], seed)

    def testBatchUpdate(self):
        batch = [(torch.rand(self.dqn.observation_size), torch.randint(7, (1,)), 0.5, torch.rand(self.dqn.observation_size), torch.randint(0, 2, (1,)).item() < 1) for i in range(32)]
        self.dqn.update(batch)
        self.assertEqual(len(batch), 32)
        self.assertEqual(len(self.dqn.optimizer.param_groups), 1)

    def testSingleUpdate(self):
        obs = np.random.rand(self.dqn.observation_size)
        action = np.random.randint(0,7)
        reward = 0.5
        next_obs = np.random.rand(self.dqn.observation_size)
        terminated = np.random.randint(0, 2) < 1
        old = self.dqn.Q[0].weight.clone()
        self.dqn.update([(obs, action, reward, next_obs, terminated)])
        self.assertEqual(len(self.dqn.optimizer.param_groups), 1)
        # Assert that model weights change
        new = self.dqn.Q[0].weight.clone()
        self.assertFalse(torch.equal(old, new))


    def testTrain(self):
        self.dqn.train(50, 1000, plot_results=True, use_buffer=True)
        self.assertEqual(len(self.dqn.optimizer.param_groups), 1)
        

if __name__ == '__main__':
    unittest.main(verbosity=2)