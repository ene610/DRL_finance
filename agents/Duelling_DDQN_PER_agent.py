import os

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
from gym import wrappers
import os
import shutil
from torch.utils.tensorboard import SummaryWriter

import random
from torch.autograd import Variable


class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.8
    beta = 0.3
    beta_increment_per_sampling = 0.0005

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims, n_neurons_layer=512, dropout=0.1, device="cpu"):
        super(DuelingDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dims, n_neurons_layer)
        self.fc2 = nn.Linear(n_neurons_layer, n_neurons_layer)
        self.fc3 = nn.Linear(n_neurons_layer, n_neurons_layer)
        self.fc4 = nn.Linear(n_neurons_layer, n_neurons_layer)
        self.fc5 = nn.Linear(n_neurons_layer, n_neurons_layer)

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

        self.V = nn.Linear(n_neurons_layer, 1)
        self.A = nn.Linear(n_neurons_layer, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = device
        self.to(self.device)

    def forward(self, state):
        # flat1 = F.relu(self.fc1(state))
        # flat2 = F.relu(self.fc2(flat1))

        x = self.dropout1(F.leaky_relu(self.fc1(state)))
        x = self.dropout2(F.leaky_relu(self.fc2(x)))
        x = self.dropout3(F.leaky_relu(self.fc3(x)))
        x = self.dropout4(F.leaky_relu(self.fc4(x)))

        flat = self.fc5(x)

        V = self.V(flat)
        A = self.A(flat)

        return V, A

    def save_checkpoint(self, path):
        T.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        self.load_state_dict(T.load(path))


class DuelingDDQNAgent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, id_agent, id_train_env, id_obs_type, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, chkpt_dir='tmp/dqn', seed=1, device="cpu", n_neurons_layer=512, dropout=0.1):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        # self.chkpt_dir = "//trained_agents//DuelingDDQNAgent//BTC//"
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.seed = seed
        self.writer = SummaryWriter(f"Tensorboard plot/Duelling_DDQN/{id_agent}/{id_train_env}/{id_obs_type}")

        self.memory = Memory(mem_size)

        self.q_eval = DuelingDeepQNetwork(self.lr,
                                          self.n_actions,
                                          input_dims=self.input_dims,
                                          device=device,
                                          n_neurons_layer=n_neurons_layer,
                                          dropout=dropout)

        self.q_next = DuelingDeepQNetwork(self.lr,
                                          self.n_actions,
                                          input_dims=self.input_dims,
                                          device=device,
                                          n_neurons_layer=n_neurons_layer,
                                          dropout=dropout)

    def store_transition(self, state, action, reward, next_state, done):

        # state = np.array([state], copy=False, dtype=np.float32)
        np_state = np.array(state, dtype=np.float32)
        np_action = np.array(action, dtype=np.int64)
        np_reward = np.array(reward, dtype=np.float32)
        np_new_state = np.array(next_state, dtype=np.float32)
        np_terminal = np.array(done, dtype=np.bool)

        state_tensor = T.tensor(np_state).to(self.q_eval.device)
        _, advantages = self.q_eval.forward(state_tensor)
        old_val = advantages[action]

        next_state_tensor = T.FloatTensor(np_new_state).to(self.q_eval.device)
        _, target_val = self.q_next(next_state_tensor)

        if done:
            next_val = reward
        else:
            next_val = reward + self.gamma * T.max(target_val)

        error = abs(old_val - next_val).cpu().detach().numpy()

        self.memory.add(error, (np_state, np_action, np_reward, np_new_state, np_terminal))

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = np.array([observation], copy=False, dtype=np.float32)
            state_tensor = T.tensor(state).to(self.q_eval.device)
            _, advantages = self.q_eval.forward(state_tensor)

            action = T.argmax(advantages).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def replace_target_network(self):
        if self.replace_target_cnt is not None and \
                self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def learn(self):
        if self.memory.tree.n_entries < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()
        mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)
        mini_batch = np.array(mini_batch).transpose()

        np_states = np.array([e for e in mini_batch[0]], dtype=np.float32)
        np_actions = np.array([e for e in mini_batch[1]], dtype=np.int64)
        np_rewards = np.array([e for e in mini_batch[2]], dtype=np.float32)
        np_new_states = np.array([e for e in mini_batch[3]], dtype=np.float32)
        np_terminals = np.array([e for e in mini_batch[4]], dtype=np.bool)

        states = T.from_numpy(np_states).to(self.q_eval.device)
        actions = T.from_numpy(np_actions).to(self.q_eval.device)
        rewards = T.from_numpy(np_rewards).to(self.q_eval.device)
        states_ = T.from_numpy(np_new_states).to(self.q_eval.device)
        dones = T.from_numpy(np_terminals).to(self.q_eval.device)

        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)

        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        q_pred = T.add(V_s,
                       (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]

        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))

        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next[indices, max_actions]

        errors = T.abs(q_pred - q_target).data.cpu().detach().numpy()

        # update priority
        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, errors[i])

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
        return loss.item()

    def save_models(self, episode):
        self.q_eval.save_checkpoint(self.chkpt_dir + f"/episode{episode}/q_eval")
        self.q_next.save_checkpoint(self.chkpt_dir + f"/episode{episode}/q_next")

    def load_models(self, episode):
        self.q_eval.load_checkpoint(self.chkpt_dir + f"/episode{episode}/q_eval")
        self.q_next.load_checkpoint(self.chkpt_dir + f"/episode{episode}/q_next")

    def convert_obs(self, obs, obs_size):
        return obs.reshape(obs_size, )

    def train(self, env, coin, n_episodes=100, checkpoint_freq=10):

        best_score = -np.inf
        load_checkpoint = False
        obs_size = self.input_dims

        n_steps = 0
        scores, eps_history, steps_array = [], [], []

        for i in range(n_episodes):

            done = False
            observation = env.reset()
            observation = self.convert_obs(observation, obs_size)
            steps = 0
            score = 0
            loss = 0
            while not done:
                action = self.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                observation_ = self.convert_obs(observation_, obs_size)
                score += reward
                self.store_transition(observation, action, reward, observation_, done)
                loss_iteration = self.learn()
                if loss_iteration != None:
                    loss += loss_iteration

                observation = observation_
                n_steps += 1
                steps += 1

            scores.append(score)
            steps_array.append(n_steps)

            self.writer.add_scalar(f"Train/Loss/{coin}", loss, i)
            self.writer.add_scalar(f"Train/Reward/{coin}", score, i)

            if i % checkpoint_freq == 0:
                # path = os.getcwd()
                # dir = f"{path}/trained_agents/DQNAgent/BTC/episode{i}"
                if os.path.exists(self.chkpt_dir + f"/episode{i}"):
                    shutil.rmtree(self.chkpt_dir + f"/episode{i}")
                os.makedirs(self.chkpt_dir + f"/episode{i}")
                self.save_models(i)

            avg_scores = np.average(scores)
            print('episode: ', i, 'score: ', score, ' average score %.1f' % avg_scores, 'best score %.2f' % best_score,
                  'epsilon %.2f' % self.epsilon, 'steps', n_steps)

            eps_history.append(self.epsilon)

        return env

    def evaluate(self, env, coin, episode, env_id=None):

        done = False
        observation = env.reset()
        obs_size = self.input_dims
        observation = self.convert_obs(observation, obs_size)
        self.epsilon = 0
        self.q_eval.eval()
        while not done:
            action = self.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            observation_ = self.convert_obs(observation_, obs_size)
            observation = observation_

        sharpe_ratio = env.sharpe_calculator_total_quantstats()
        sortino_ratio = env.sortino_calculator_total_quantstats()
        total_profit = list(env.returns_balance.values())[-1] - 10000
        total_reward = env._total_reward

        if env_id:
            tensorboard_path = f"Eval/{env_id}"
        else:
            tensorboard_path = "Eval"

        self.writer.add_scalar(f"{tensorboard_path}/Profit/{coin}", total_profit, episode)
        self.writer.add_scalar(f"{tensorboard_path}/Reward/{coin}", total_reward, episode)
        self.writer.add_scalar(f"{tensorboard_path}/Sharpe/{coin}", sharpe_ratio, episode)
        self.writer.add_scalar(f"{tensorboard_path}/Sortino/{coin}", sortino_ratio, episode)

        return env

# chkpt_dir = os.getcwd() + "/trained_agents/DuelingDDQNAgent/BTC"
# env = gym.make(id_str, df=df, frame_bound=(122,326), window_size=22)

# obs_size = env.observation_space.shape[0] * env.observation_space.shape[1]
# agent =DuelingDDQNAgent(gamma=0.99,
#                 epsilon=1.0,
#                 lr=0.0001,
#                 input_dims=(obs_size),
#                 n_actions=env.action_space.n,
#                 mem_size=50000,
#                 eps_min=0.1,
#                 batch_size=32,
#                 replace=10000,
#                 eps_dec=1e-5,
#                 chkpt_dir=chkpt_dir,
#                 seed = 1,
#                 device = device
#                 )

# agent.train(env)