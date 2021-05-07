import pfrl


import numpy as np
import torch
from abc import ABCMeta, abstractmethod

from tensortrade.core import Identifiable

class QFunction(torch.nn.Module):

    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.l1 = torch.nn.Linear(obs_size, 120)
        self.l2 = torch.nn.Linear(120, 120)
        self.l3 = torch.nn.Linear(120, n_actions)

    def forward(self, x):
        h = x
        h = torch.nn.functional.relu(self.l1(h))
        h = torch.nn.functional.relu(self.l2(h))
        h = self.l3(h)
        return pfrl.action_value.DiscreteActionValue(h)

class Agent(Identifiable, metaclass=ABCMeta):

    def __init__(self, env: 'TradingEnv'):
        self.env = env
        self.n_actions = env.action_space.n
        self.observation_shape = env.observation_space.shape


        self.target_network.trainable = False
        self.dqn_agent = self.create_agent()
        self.env.agent_id = self.id

    def create_agent(self):

        q_func = QFunction(self.obs_size, self.n_actions)
        # Use Adam to optimize q_func. eps=1e-2 is for stability.
        optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-2)

        # Set the discount factor that discounts future rewards.
        gamma = 0.9

        # Use epsilon-greedy for exploration
        explorer = pfrl.explorers.ConstantEpsilonGreedy(
            epsilon=0.3, random_action_func=self.env.action_space.sample)

        # DQN uses Experience Replay.
        # Specify a replay buffer and its capacity.
        replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)

        # Since observations from CartPole-v0 is numpy.float64 while
        # As PyTorch only accepts numpy.float32 by default, specify
        # a converter as a feature extractor function phi.
        phi = lambda x: x.astype(np.float32, copy=False)

        # Set the device id to use GPU. To use CPU only, set it to -1.
        gpu = -1

        # Now create an agent that will interact with the environment.
        agent = pfrl.agents.DoubleDQN(
            q_func,
            optimizer,
            replay_buffer,
            gamma,
            explorer,
            replay_start_size=500,
            update_interval=1,
            target_update_interval=100,
            phi=phi,
            gpu=gpu,
        )

        return agent

    @abstractmethod
    def restore(self, path: str, **kwargs):
        """Restore the agent from the file specified in `path`."""
        pass

    @abstractmethod
    def save(self, path: str, **kwargs):
        """Save the agent to the directory specified in `path`."""
        pass

    @abstractmethod
    def get_action(self, state: np.ndarray, **kwargs) -> int:
        """Get an action for a specific state in the environment."""
        action = self.dqn_agent.act(state)
        return action

    @abstractmethod
    def train(self,
              n_steps: int = None,
              n_episodes: int = 10000,
              save_every: int = None,
              save_path: str = None,
              callback: callable = None,
              **kwargs) -> float:
        """Train the agent in the environment and return the mean reward."""

        #n_episodes = 1000
        max_episode_len = 2**30
        for i in range(1, n_episodes + 1):
            obs = self.env.reset()
            t = 0  # time step
            while True:
                # Uncomment to watch the behavior in a GUI window
                # env.render()
                action = self.get_action(obs)
                obs, reward, done, _ = self.env.step(action)
                # print(action, reward, done)
                t += 1
                reset = t == max_episode_len
                self.agent.observe(obs, reward, done, reset)
                if done or reset:
                    break

        print('Finished.')

