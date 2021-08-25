import pfrl
import torch
import numpy


class QFunction(torch.nn.Module):

    def __init__(self, obs_size, n_actions):

        super().__init__()

        self.l1 = torch.nn.Linear(obs_size, 360)
        self.l2 = torch.nn.Linear(360, 360)
        self.l3 = torch.nn.Linear(360, 512)
        self.l4 = torch.nn.Linear(512, 360)
        self.l5 = torch.nn.Linear(360, 50)
        self.l6 = torch.nn.Linear(50, n_actions)

        self.dropout_02 = torch.nn.Dropout(p=0)

    def forward(self, input):
        h = torch.nn.functional.leaky_relu(self.l1(input))

        h = torch.nn.functional.leaky_relu(self.l2(h))
        h = self.dropout_02(h)

        h = torch.nn.functional.leaky_relu(self.l3(h))
        h = self.dropout_02(h)

        h = torch.nn.functional.leaky_relu(self.l4(h))
        h = self.dropout_02(h)

        h = torch.nn.functional.leaky_relu(self.l5(h))
        output = self.l6(h)

        return pfrl.action_value.DiscreteActionValue(output)


class DoubleDQN:

    def __init__(self, obs_size, n_actions):
        #obs_size = env.observation_space.low.size
        #n_actions = env.action_space.n
        q_func = QFunction(obs_size, n_actions)
        # Use Adam to optimize q_func. eps=1e-2 is for stability.
        optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-2)
        # Set the discount factor that discounts future rewards.
        gamma = 0.9

        # Use epsilon-greedy for exploration
        explorer = pfrl.explorers.ConstantEpsilonGreedy(
            epsilon=0.3, random_action_func=env.action_space.sample)
        explorer = pfrl.explorers.Boltzmann()
        # DQN uses Experience Replay.
        # Specify a replay buffer and its capacity.
        replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)

        # Since observations from CartPole-v0 is numpy.float64 while
        # As PyTorch only accepts numpy.float32 by default, specify
        # a converter as a feature extractor function phi.
        phi = lambda x: x.astype(numpy.float32, copy=False)

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