from fileinput import filename
from pydoc import render_doc
import torch
import unittest
import os
import gymnasium as gym
from agents import DQN
import numpy as np

class TestDQN(unittest.TestCase):
    env_name = 'ALE/Assault-ram-v5'
    env = gym.make(env_name)
    dqn = DQN(env, 0.0001)
    # Test if the DQN class is correctly initialized
    def test_init(self):
        env = gym.make(self.env_name)
        dqn = DQN(env, 0.0001)
        self.assertEqual(dqn.env, env)
        self.assertEqual(dqn.observation_size, 128)
        self.assertEqual(dqn.actions, list(range(env.action_space.n)))
        self.assertEqual(len(dqn.Q), 5)
    
    def testSelectAction(self):
        obs = np.random.rand(128)
        action = self.dqn.select_action(obs)
        self.assertTrue(action in self.dqn.actions)

    def testSeed(self):
        seed = 23
        self.dqn.seed_model(seed)
        self.assertEqual(torch.initial_seed(), seed)
        self.assertEqual(np.random.get_state()[1][0], seed)

    def testBatchUpdate(self):
        batch = [(np.random.rand(self.dqn.observation_size), np.random.randint(0,7), 0.5, np.random.rand(self.dqn.observation_size), np.random.randint(0, 2) < 1) for i in range(32)]
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
        self.dqn.train(10, 10, plot_results=True, use_buffer=True)
        self.assertEqual(len(self.dqn.optimizer.param_groups), 1)

class TestDQNAcrobot(unittest.TestCase):
    env_name = 'Acrobot-v1'
    env = gym.make(env_name)
    learnRate = 0.0001
    dqn = DQN(env, learnRate)
    # Test if the DQN class is correctly initialized
    def test_init(self):
        env = gym.make(self.env_name)
        dqn = DQN(env, self.learnRate)
        self.assertEqual(dqn.env, env)
        self.assertEqual(dqn.observation_size, 6)
        self.assertEqual(dqn.lr, self.learnRate)
        self.assertEqual(dqn.actions, list(range(env.action_space.n)))
        self.assertEqual(len(dqn.Q), 5)
    
    def testSelectAction(self):
        obs = np.random.rand(6)
        action = self.dqn.select_action(obs)
        self.assertTrue(action in self.dqn.actions)

    def testSeed(self):
        seed = 23
        self.dqn.seed_model(seed)
        self.assertEqual(torch.initial_seed(), seed)
        self.assertEqual(np.random.get_state()[1][0], seed)

    def testBatchUpdate(self):
        batch = [(np.random.rand(self.dqn.observation_size), np.random.randint(0,3), 0.5, np.random.rand(self.dqn.observation_size), np.random.randint(0, 2) < 1) for i in range(32)]
        self.dqn.update(batch)
        self.assertEqual(len(batch), 32)
        self.assertEqual(len(self.dqn.optimizer.param_groups), 1)

    def testSingleUpdate(self):
        obs = np.random.rand(self.dqn.observation_size)
        action = np.random.randint(0,3)
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
        self.dqn.train(10, 10, plot_results=True, use_buffer=True)
        self.assertEqual(len(self.dqn.optimizer.param_groups), 1)
        

if __name__ == '__main__':
    unittest.main(verbosity=2)