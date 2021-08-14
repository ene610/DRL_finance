import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
from gym import wrappers

import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import collections
import sys
from typing import Dict, List, Tuple

def convert_obs(obs,obs_size):
  return obs.reshape(1,obs_size)

class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        numberOfNeurons = 512
        self.hidden_space = numberOfNeurons
        dropout = 0.1

        self.fc1 = nn.Linear(input_dims, numberOfNeurons)
        self.fc2 = nn.Linear(numberOfNeurons, numberOfNeurons)
        self.fc3 = nn.Linear(numberOfNeurons, numberOfNeurons)
        self.fc4 = nn.Linear(numberOfNeurons, numberOfNeurons)
        self.lstm1 = nn.LSTM(numberOfNeurons, numberOfNeurons, batch_first=True)
        self.fc5 = nn.Linear(numberOfNeurons, n_actions)

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
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, h, c):

        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x, (new_h, new_c) = self.lstm1(x, (h, c))
        actions = self.fc5(x)
        return actions, new_h, new_c

    def init_hidden_state(self, batch_size, training=None):

        assert training is not None, "training step parameter should be dtermined"

        if training is True:
            return T.zeros([1, batch_size, self.hidden_space]), T.zeros([1, batch_size, self.hidden_space])
        else:
            # print("False")
            return T.zeros([1, 1, self.hidden_space]), T.zeros([1, 1, self.hidden_space])

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

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
            obs = obs[idx:idx+lookup_step]
            action = action[idx:idx+lookup_step]
            reward = reward[idx:idx+lookup_step]
            next_obs = next_obs[idx:idx+lookup_step]
            done = done[idx:idx+lookup_step]

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
                 lookup_step=None):
        self.random_update = random_update  # if False, sequential update
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.batch_size = batch_size
        self.lookup_step = lookup_step

        if (random_update is False) and (self.batch_size > 1):
            sys.exit(
                'It is recommend to use 1 batch for sequential update, if you want, erase this code block and modify code')

        self.memory = collections.deque(maxlen=self.max_epi_num)

    def put(self, episode):
        self.memory.append(episode)

    def sample(self):
        sampled_buffer = []

        ##################### RANDOM UPDATE ############################
        if self.random_update:  # Random upodate
            sampled_episodes = random.sample(self.memory, self.batch_size)

            check_flag = True  # check if every sample data to train is larger than batch size
            min_step = self.max_epi_len

            for episode in sampled_episodes:
                min_step = min(min_step, len(episode))  # get minimum step from sampled episodes

            for episode in sampled_episodes:
                if min_step > self.lookup_step:  # sample buffer with lookup_step size
                    idx = np.random.randint(0, len(episode) - self.lookup_step + 1)
                    sample = episode.sample(random_update=self.random_update, lookup_step=self.lookup_step, idx=idx)
                    sampled_buffer.append(sample)
                else:
                    idx = np.random.randint(0, len(episode) - min_step + 1)  # sample buffer with minstep size
                    sample = episode.sample(random_update=self.random_update, lookup_step=min_step, idx=idx)
                    sampled_buffer.append(sample)

        ##################### SEQUENTIAL UPDATE ############################
        else:  # Sequential update
            idx = np.random.randint(0, len(self.memory))
            sampled_buffer.append(self.memory[idx].sample(random_update=self.random_update))

        return sampled_buffer, len(sampled_buffer[0]['obs'])  # buffers, sequence_length

    def __len__(self):
        return len(self.memory)

    def train(self, env, id_str):

        if True:
            # env = make_env('CartPole-v1')
            # env = gym.make('CartPole-v1')
            best_score = -np.inf
            load_checkpoint = False
            n_episodes = 50
            batch_size = 6
            obs_size = env.observation_space.shape[0] * env.observation_space.shape[1]
            agent = DQNAgent(gamma=0.99, epsilon=0.5, lr=0.0001,
                             input_dims=obs_size,
                             n_actions=env.action_space.n, mem_size=50000, eps_min=0.01,
                             batch_size=batch_size, replace=1000, eps_dec=3e-5,
                             chkpt_dir='/content/models/', algo='DQNAgent',
                             env_name=id_str)

            # batch_size = 8
            learning_rate = 1e-3
            buffer_len = int(100000)
            min_epi_num = 16  # Start moment to train the Q network
            episodes = 650
            print_per_iter = 20
            target_update_period = 4
            eps_start = 0.1
            eps_end = 0.001
            eps_decay = 0.995
            tau = 1e-2
            max_step = 2000

            if load_checkpoint:
                agent.load_models()

            n_steps = 0
            scores, eps_history, steps_array = [], [], []

            for episode in range(n_episodes):
                done = False

                observation = env.reset()
                h, c = agent.q_eval.init_hidden_state(batch_size=batch_size, training=False)
                observation = convert_obs(observation, obs_size)
                # print(observation.shape)
                # input()
                steps = 0
                score = 0

                while not done:
                    # print("!")
                    action, h, c = agent.choose_action(observation, h, c)
                    observation_, reward, done, info = env.step(action)
                    observation_ = convert_obs(observation_, obs_size)
                    agent.store_transition([observation, action, reward, observation_, done])
                    # observation = convert_obs(observation_,obs_size)
                    score += reward
                    observation = observation_
                    n_steps += 1
                    steps += 1
                    if not load_checkpoint and episode > 13:
                        agent.learn()

                agent.store_episode()
                agent.reset_buffer()

                scores.append(score)
                steps_array.append(n_steps)

                avg_scores = np.average(scores)
                print('episode: ', episode, 'score: ', score, ' average score %.1f' % avg_scores,
                      'best score %.2f' % best_score, 'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

                eps_history.append(agent.epsilon)

        return env

    def eval(self, env):

        env.reset()
        load_checkpoint = True
        n_episodes = 1
        n_steps = 0
        agent = self.agent
        agent.epsilon = 0

        for i in range(1):
            done = False
            observation = env.reset()
            # print(observation.shape)
            obs_size = env.observation_space.shape[0] * env.observation_space.shape[1]
            observation = convert_obs(observation, obs_size)
            h, c = agent.q_eval.init_hidden_state(batch_size=batch_size, training=False)
            steps = 0
            score = 0
            while not done:
                action, h, c = agent.choose_action(observation, h, c)
                observation_, reward, done, info = env.step(action)
                observation_ = convert_obs(observation_, obs_size)

                score += reward

                observation = observation_
                n_steps += 1
                steps += 1

        env.render_all()
        return env