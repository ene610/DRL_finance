import sys
from typing import Dict, List, Tuple

import collections
import torch

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
import shutil
from torch.utils.tensorboard import SummaryWriter

class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims, n_neurons_layer=512, dropout=0.1, device="cpu"):
        super(DeepQNetwork, self).__init__()

        self.hidden_space = n_neurons_layer
        self.device = device
        self.fc1 = nn.Linear(input_dims, n_neurons_layer)
        self.fc2 = nn.Linear(n_neurons_layer, n_neurons_layer)
        self.fc3 = nn.Linear(n_neurons_layer, n_neurons_layer)
        self.fc4 = nn.Linear(n_neurons_layer, n_neurons_layer)
        self.lstm1 = nn.LSTM(n_neurons_layer, n_neurons_layer, batch_first=True)
        self.fc5 = nn.Linear(n_neurons_layer, n_actions)

        # Definition of some Batch Normalization layers
        # self.bn1 = nn.BatchNorm1d(numberOfNeurons)
        # self.bn2 = nn.BatchNorm1d(numberOfNeurons)
        # self.bn3 = nn.BatchNorm1d(numberOfNeurons)
        # self.bn4 = nn.BatchNorm1d(numberOfNeurons)

        # Definition of some Dropout layers.
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        # Xavier initialization for the entire neural network
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.xavier_uniform_(self.fc5.weight)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()

        self.to(device)

    def forward(self, state, h, c):

        h = h.to(self.device)
        c = c.to(self.device)

        x = self.dropout1(F.leaky_relu(self.fc1(state)))
        x = self.dropout2(F.leaky_relu(self.fc2(x)))
        x = self.dropout3(F.leaky_relu(self.fc3(x)))
        x = self.dropout4(F.leaky_relu(self.fc4(x)))
        x, (new_h, new_c) = self.lstm1(x, (h, c))

        new_h.to(self.device)
        new_c.to(self.device)
        actions = self.fc5(x)

        return actions, new_h, new_c

    def init_hidden_state(self, batch_size, training=None):

        assert training is not None, "training step parameter should be dtermined"
        if training is True:
            return T.zeros([1, batch_size, self.hidden_space]), T.zeros([1, batch_size, self.hidden_space]).to(
                self.device)
        else:
            return T.zeros([1, 1, self.hidden_space]), T.zeros([1, 1, self.hidden_space]).to(self.device)

    def save_checkpoint(self, path):
        T.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        self.load_state_dict(T.load(path))


class EpisodeBuffer:
    """A simple numpy replay buffer."""

    def __init__(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.next_obs = []
        self.done = []

    def put(self, transition):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.reward.append(transition[2])
        self.next_obs.append(transition[3])
        self.done.append(transition[4])

    def sample(self, random_update=False, lookup_step=None, idx=None) -> Dict[str, np.ndarray]:
        obs = np.array(self.obs)
        action = np.array(self.action)
        reward = np.array(self.reward)
        next_obs = np.array(self.next_obs)
        done = np.array(self.done)

        if random_update is True:
            obs = obs[idx:idx + lookup_step]
            action = action[idx:idx + lookup_step]
            reward = reward[idx:idx + lookup_step]
            next_obs = next_obs[idx:idx + lookup_step]
            done = done[idx:idx + lookup_step]

        return dict(obs=obs,
                    acts=action,
                    rews=reward,
                    next_obs=next_obs,
                    done=done)

    def __len__(self) -> int:
        return len(self.obs)


class EpisodeMemory():
    """Episode memory for recurrent agent"""

    def __init__(self, random_update=False,
                 max_epi_num=100, max_epi_len=500,
                 batch_size=1,
                 lookup_step=None, seed=1):
        self.random_update = random_update  # if False, sequential update
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.batch_size = batch_size
        self.lookup_step = lookup_step
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        if (random_update is False) and (self.batch_size > 1):
            sys.exit(
                'It is recommend to use 1 batch for sequential update, if you want, erase this code block and modify code')

        self.memory = collections.deque(maxlen=self.max_epi_num)

    def put(self, episode):
        self.memory.append(episode)

    def sample(self):
        sampled_buffer = []

        # self.rng.integers(low=0, high=1000, size=1)

        ##################### RANDOM UPDATE ############################

        if self.random_update:  # Random update
            sampled_episodes = []
            sampled_episodes = self.rng.choice(self.memory, size=self.batch_size)
            # sampled_episodes = random.sample(self.memory, self.batch_size)

            check_flag = True  # check if every sample data to train is larger than batch size
            min_step = self.max_epi_len

            for episode in sampled_episodes:
                min_step = min(min_step, len(episode))  # get minimum step from sampled episodes

            for episode in sampled_episodes:
                if min_step > self.lookup_step:  # sample buffer with lookup_step size
                    # idx = np.random.randint(0, len(episode) - self.lookup_step + 1)
                    idx = self.rng.integers(low=0, high=len(episode) - self.lookup_step + 1, size=1)[0]
                    sample = episode.sample(random_update=self.random_update, lookup_step=self.lookup_step, idx=idx)
                    sampled_buffer.append(sample)
                else:
                    # idx = np.random.randint(0, len(episode) - min_step + 1)  # sample buffer with minstep size
                    idx = self.rng.integers(low=0, high=len(episode) - min_step + 1, size=1)[0]
                    sample = episode.sample(random_update=self.random_update, lookup_step=min_step, idx=idx)
                    sampled_buffer.append(sample)

        ##################### SEQUENTIAL UPDATE ############################
        else:  # Sequential update
            idx = self.rng.integers(low=0, high=len(self.memory), size=1)[0]
            sampled_buffer.append(self.memory[idx].sample(random_update=self.random_update))

        return sampled_buffer, len(sampled_buffer[0]['obs'])  # buffers, sequence_length

    def __len__(self):
        return len(self.memory)


class DRQNAgent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, id_agent, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, chkpt_dir='tmp/dqn', random_update=True,
                 lookup_step=10, max_epi_len=3000, device="cpu", seed=1, n_neurons_layer=512, dropout=0.1):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace

        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.batch_size = batch_size
        # random_update = True

        # DRQN param #episode buffer paraeter
        self.random_update = random_update
        self.lookup_step = lookup_step
        self.max_epi_len = max_epi_len
        self.device = device
        self.seed = seed
        self.writer = SummaryWriter(f"Tensorboard plot/DRQN/{id_agent}")

        self.q_eval = DeepQNetwork(self.lr,
                                   self.n_actions,
                                   input_dims=self.input_dims,
                                   device=self.device,
                                   n_neurons_layer=n_neurons_layer,
                                   dropout=dropout)

        self.q_next = DeepQNetwork(self.lr,
                                   self.n_actions,
                                   input_dims=self.input_dims,
                                   device=self.device,
                                   n_neurons_layer=n_neurons_layer,
                                   dropout=dropout)

        self.q_next.load_state_dict(self.q_eval.state_dict(), )

        self.episode_memory = EpisodeMemory(random_update=self.random_update,
                                            max_epi_num=100,
                                            max_epi_len=max_epi_len,
                                            batch_size=batch_size,
                                            lookup_step=lookup_step,
                                            seed=self.seed)

        self.episode_buffer = EpisodeBuffer()

    def choose_action(self, observation, h, c):

        state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
        actions, h, c = self.q_eval.forward(state, h, c)
        if np.random.random() > self.epsilon:

            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action, h, c

    def store_episode(self):
        self.episode_memory.put(self.episode_buffer)

    def store_transition(self, transition):
        self.episode_buffer.put(transition)

    def reset_buffer(self):
        self.episode_buffer = EpisodeBuffer()

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def save_models(self, episode):
        self.q_eval.save_checkpoint(self.chkpt_dir + f"/episode{episode}/q_eval")
        self.q_next.save_checkpoint(self.chkpt_dir + f"/episode{episode}/q_next")

    def load_models(self, episode):
        self.q_eval.load_checkpoint(self.chkpt_dir + f"/episode{episode}/q_eval")
        self.q_next.load_checkpoint(self.chkpt_dir + f"/episode{episode}/q_next")

    def learn(self):

        q_net = self.q_eval
        target_q_net = self.q_next
        episode_memory = self.episode_memory
        optimizer = self.q_eval.optimizer
        device = self.device
        gamma = self.gamma
        batch_size = self.batch_size

        # Get batch from replay buffer
        samples, seq_len = episode_memory.sample()

        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []

        for i in range(batch_size):
            observations.append(samples[i]["obs"])
            actions.append(samples[i]["acts"])
            rewards.append(samples[i]["rews"])
            next_observations.append(samples[i]["next_obs"])
            dones.append(samples[i]["done"])

        observations = np.array(observations)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_observations = np.array(next_observations)
        dones = np.array(dones)

        observations = torch.FloatTensor(observations.reshape(batch_size, seq_len, -1)).to(device)
        actions = torch.LongTensor(actions.reshape(batch_size, seq_len, -1)).to(device)
        rewards = torch.FloatTensor(rewards.reshape(batch_size, seq_len, -1)).to(device)
        next_observations = torch.FloatTensor(next_observations.reshape(batch_size, seq_len, -1)).to(device)
        dones = torch.FloatTensor(dones.reshape(batch_size, seq_len, -1)).to(device)

        h_target, c_target = target_q_net.init_hidden_state(batch_size=batch_size, training=True)

        q_target, _, _ = target_q_net(next_observations, h_target.to(device), c_target.to(device))

        q_target_max = q_target.max(2)[0].view(batch_size, seq_len, -1).detach()
        targets = rewards + gamma * q_target_max * dones

        h, c = q_net.init_hidden_state(batch_size=batch_size, training=True)
        q_out, _, _ = q_net(observations, h.to(device), c.to(device))
        q_a = q_out.gather(2, actions)  # forse mettere 3

        # Multiply Importance Sampling weights to loss
        loss = F.smooth_l1_loss(q_a, targets)

        # Update Network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.decrement_epsilon()
        return loss.item()


    def convert_obs(self, obs, obs_size):
        return obs.reshape(1, obs_size)

    def evaluate(self, env):

        self.q_eval.eval()
        self.epsilon = 0

        done = False
        observation = env.reset()
        observation = self.convert_obs(observation, self.input_dims)
        h, c = self.q_eval.init_hidden_state(batch_size=self.batch_size, training=False)

        while not done:
            action, h, c = self.choose_action(observation, h, c)
            observation_, reward, done, info = env.step(action)
            observation_ = self.convert_obs(observation_, self.input_dims)
            observation = observation_

        return env

    def train(self, env, coin, n_episodes=100, checkpoint_freq=10):

        best_score = -np.inf
        n_steps = 0
        scores, eps_history, steps_array = [], [], []

        for episode in range(n_episodes):
            done = False

            observation = env.reset()
            h, c = self.q_eval.init_hidden_state(batch_size=self.batch_size, training=False)
            observation = self.convert_obs(observation, self.input_dims)
            steps = 0
            score = 0
            loss = 0

            while not done:
                action, h, c = self.choose_action(observation, h, c)
                observation_, reward, done, info = env.step(action)
                observation_ = self.convert_obs(observation_, self.input_dims)

                self.store_transition([observation, action, reward, observation_, done])

                score += reward
                observation = observation_
                n_steps += 1
                steps += 1

                if episode > self.batch_size:
                    loss_iteration = self.learn()

                if loss_iteration != None:
                    loss += loss_iteration

            self.store_episode()
            self.reset_buffer()

            self.writer.add_scalar(f"Loss/train/{coin}", loss, episode)
            self.writer.add_scalar(f"Reward/train/{coin}", score, episode)

            if episode % checkpoint_freq == 0:
                if os.path.exists(self.chkpt_dir + f"/episode{episode}"):
                    shutil.rmtree(self.chkpt_dir + f"/episode{episode}")
                os.makedirs(self.chkpt_dir + f"/episode{episode}")
                self.save_models(episode)

            eps_history.append(self.epsilon)

            scores.append(score)
            steps_array.append(n_steps)

            avg_scores = np.average(scores)
            print('episode: ', episode, 'score: ', score, ' average score %.1f' % avg_scores,
                  'best score %.2f' % best_score, 'epsilon %.2f' % self.epsilon, 'steps', n_steps)

            eps_history.append(self.epsilon)


# chkpt_dir = os.getcwd() + "/trained_agents/DQN/BTC"
# batch_size = 6
# obs_size = env.observation_space.shape[0] * env.observation_space.shape[1]
# agent = DRQNAgent(gamma=0.99,
#                   epsilon=0.5,
#                   lr=0.0001,
#                   input_dims = obs_size,
#                   n_actions=env.action_space.n,
#                   mem_size=50000,
#                   eps_min=0.01,
#                   batch_size=1,#batch_size,
#                   replace=1000,
#                   eps_dec=3e-5,
#                   random_update=False,
#                   lookup_step=10,
#                   max_epi_len=3000,
#                   device="cuda",
#                   seed=1,
#                   chkpt_dir=chkpt_dir)

# agent.train(env)