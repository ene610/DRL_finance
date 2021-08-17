import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
import numpy as np
from gym import wrappers
import os
import shutil

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        numberOfNeurons = 512
        dropout = 0.1

        self.fc1 = nn.Linear(input_dims, numberOfNeurons)
        self.fc2 = nn.Linear(numberOfNeurons, numberOfNeurons)
        self.fc3 = nn.Linear(numberOfNeurons, numberOfNeurons)
        self.fc4 = nn.Linear(numberOfNeurons, numberOfNeurons)
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

    def forward(self, state):
        # x = self.dropout1(F.leaky_relu(self.bn1(self.fc1(state))))
        # x = self.dropout2(F.leaky_relu(self.bn2(self.fc2(x))))
        # x = self.dropout3(F.leaky_relu(self.bn3(self.fc3(x))))
        # x = self.dropout4(F.leaky_relu(self.bn4(self.fc4(x))))

        x = self.dropout1(F.leaky_relu(self.fc1(state)))
        x = self.dropout2(F.leaky_relu(self.fc2(x)))
        x = self.dropout3(F.leaky_relu(self.fc3(x)))
        x = self.dropout4(F.leaky_relu(self.fc4(x)))

        #x = F.leaky_relu(self.fc1(state))
        #x = F.leaky_relu(self.fc2(x))
        #x = F.leaky_relu(self.fc3(x))
        #x = F.leaky_relu(self.fc4(x))

        action = self.fc5(x)

        return action

    def save_checkpoint(self, path):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(path))


class DDQNAgent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, (input_dims,), n_actions)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                   input_dims=self.input_dims,
                                   name=self.env_name + '_' + self.algo + '_q_eval',
                                   chkpt_dir=self.chkpt_dir)
        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                   input_dims=self.input_dims,
                                   name=self.env_name + '_' + self.algo + '_q_next',
                                   chkpt_dir=self.chkpt_dir)

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
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
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()

        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_)
        q_eval = self.q_eval.forward(states_)

        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next[indices, max_actions]
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()

        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

    def save_models(self, path):
        self.q_eval.save_checkpoint(path + "/q_eval")
        self.q_next.save_checkpoint(path + "/q_next")

    def load_models(self, path):
        self.q_eval.load_checkpoint(path + "/q_eval")
        self.q_next.load_checkpoint(path + "/q_next")

    def convert_obs(self, obs, obs_size):
        return obs.reshape(obs_size, )

    def train(self, env):
        best_score = -np.inf
        load_checkpoint = False
        n_episodes = 100

        if load_checkpoint:
            self.agent.load_models()

        n_steps = 0
        scores, eps_history, steps_array = [], [], []
        obs_size = self.input_dims
        for i in range(n_episodes):

            done = False
            observation = env.reset()
            observation = self.convert_obs(observation, obs_size)
            steps = 0
            score = 0

            while not done:
                action = self.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                observation_ = self.convert_obs(observation_, obs_size)
                score += reward

                if not load_checkpoint:
                    self.store_transition(observation, action, reward, observation_, done)
                    self.learn()

                observation = observation_
                n_steps += 1
                steps += 1

            scores.append(score)
            steps_array.append(n_steps)

            avg_scores = np.average(scores)
            print('episode: ', i, 'score: ', score, ' average score %.1f' % avg_scores, 'best score %.2f' % best_score,
                  'epsilon %.2f' % self.epsilon, 'steps', n_steps)

            eps_history.append(self.epsilon)

            if i % 10 == 0:
                path = os.getcwd()
                dir = f"{path}/trained_agents/DDQNAgent/BTC/episode{i}"
                if os.path.exists(dir):
                    shutil.rmtree(dir)
                os.makedirs(dir)
                self.chkpt_dir = dir
                self.save_models(dir)
        return env

    def eval(self, env):

        self.epsilon = 0
        obs_size = self.input_dims
        self.q_eval.eval()
        done = False
        observation = env.reset()
        observation = self.convert_obs(observation, obs_size)

        while not done:
            action = self.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            observation_ = self.convert_obs(observation_, obs_size)
            observation = observation_

        return env

# obs_size = env.observation_space.shape[0] * env.observation_space.shape[1]
# agent =DDQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001,
#                                 input_dims=(obs_size),
#                                 n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
#                                 batch_size=32, replace=10000, eps_dec=1e-5,
#                                 chkpt_dir='/content/models/', algo='DuelingDDQNAgent',
#                                 env_name=id_str)

# agent.train(env)