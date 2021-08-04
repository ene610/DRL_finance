import numpy as np

from machin.env.utils.openai_gym import disable_view_window
from machin.frame.algorithms import DQNPer
from machin.utils.logging import default_logger as logger

import gym
import torch as t
import torch.nn as nn

inizio = 10022
fine = inizio + 60 #* 7
env = gym.make(id_str, df=df, frame_bound=(inizio, fine), window_size=22)
observe_dim = env.reset().shape[0] * env.reset().shape[1]
action_num = 3
max_episodes = 250
max_steps = 20000
solved_reward = 190
solved_repeat = 5

def convert(mem: np.ndarray):
    return t.tensor(mem.reshape(1, observe_dim).astype(np.float32))

class MachinDRQN():

    class History:
        def __init__(self, history_depth, state_shape):
            self.history = [t.zeros(state_shape) for _ in range(history_depth)]
            self.state_shape = state_shape

        def append(self, state):
            assert (
                    t.is_tensor(state)
                    and state.dtype == t.float32
                    and tuple(state.shape) == self.state_shape
            )
            self.history.append(state)
            self.history.pop(0)
            return self

        def get(self):
            # size: (1, history_depth, ...)
            return t.cat(self.history, dim=0).unsqueeze(0)


    # Q network model definition
    # for atari games
    class RecurrentQNet(nn.Module):
        def __init__(self, action_num):
            super().__init__()
            self.gru = nn.GRU(observe_dim, 256, batch_first=True)
            self.fc1 = nn.Linear(256, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, 256)
            self.fc4 = nn.Linear(256, action_num)

        def forward(self, mem=None, hidden=None, history_mem=None):

            if mem is not None:
                # in sampling
                a, h = self.gru(mem.unsqueeze(1), hidden)
                out = self.fc1(t.relu(a.flatten(start_dim=1)))
                out = self.fc2(t.relu(out))
                out = self.fc3(t.relu(out))
                out = self.fc4(t.relu(out))
                return out, h
            else:

                # in updating

                batch_size = history_mem.shape[0]
                seq_length = history_mem.shape[1]

                hidden = t.zeros([1, batch_size, 256], device=history_mem.device)
                for i in range(seq_length):
                    _, hidden = self.gru(history_mem[:, i].unsqueeze(1), hidden)
                # a[:, -1] = h
                out = self.fc1(t.relu(hidden.transpose(0, 1).flatten(start_dim=1)))
                out = self.fc2(t.relu(out))
                out = self.fc3(t.relu(out))
                out = self.fc4(t.relu(out))
                return out

    def __init__(self,action_num,observe_dim,history_depth = 10):
        self.history_depth = history_depth
        self.obs_dim = observe_dim
        r_q_net = self.RecurrentQNet(action_num)  # .to("cuda:0")
        r_q_net_t = self.RecurrentQNet(action_num)  # .to("cuda:0")
        history_depth = 10

        self.drqn = DQNPer(
            r_q_net,
            r_q_net_t,
            t.optim.Adam,
            nn.MSELoss(reduction="sum"),
            learning_rate=0.01
        )

    def train(self, env, max_episodes = 250):
        episode, step, reward_fulfilled = 0, 0, 0
        smoothed_total_reward = 0

        while episode < max_episodes:
            episode += 1
            total_reward = 0
            terminal = False
            step = 0

            hidden = t.zeros([1, 1, 256])
            state = convert(env.reset())
            history = self.History(self.history_depth, (1, self.obs_dim))

            tmp_observations = []
            while not terminal:
                step += 1
                with t.no_grad():
                    old_state = state
                    # print(state.shape)
                    history.append(state)
                    # agent model inference
                    action, hidden = self.drqn.act_discrete_with_noise(
                        {"mem": old_state, "hidden": hidden, }  # "history_mem" : history}
                    )
                    # print(reward)
                    # info is {"ale.lives": self.ale.lives()}, not used here
                    state, reward, terminal, info = env.step(action.item())
                    state = convert(state)
                    reward = float(reward)
                    total_reward += reward
                    # print(state.dtype)
                    # history mem includes current state
                    old_history = history.get()
                    new_history = history.append(state).get()
                    # print("nH",new_history.shape)
                    # if(terminal): env.render_all()
                    tmp_observations.append(
                        {
                            "state": {"history_mem": old_history},
                            "action": {"action": action},
                            "next_state": {"history_mem": new_history},
                            "reward": reward,
                            "terminal": terminal,
                        }
                    )

            self.drqn.store_episode(tmp_observations)

            # update, update more if episode is longer, else less
            if episode > 20:
                # drqn.update()
                for _ in range(step // self.history_depth):  # // history_depth
                    self.drqn.update()

            # show reward
            smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1

            logger.info(f"Episode {episode} smoothed_total_reward={smoothed_total_reward:.2f} total_reward={total_reward} ")

    def evaluate(self,env):
        episode, step, reward_fulfilled = 0, 0, 0
        smoothed_total_reward = 0
        #env = gym.make(id_str, df=df, frame_bound=(inizio, fine), window_size=22)

        total_reward = 0
        terminal = False
        step = 0
        hidden = t.zeros([1, 1, 256])
        state = convert(env.reset())
        history = self.History(self.history_depth, (1, observe_dim))
        tmp_observations = []

        while not terminal:
            step += 1
            with t.no_grad():
                old_state = state
                history.append(state)
                # agent model inference
                action, hidden = self.drqn.act_discrete(
                    {"mem": old_state, "hidden": hidden}
                )

                # info is {"ale.lives": self.ale.lives()}, not used here
                state, reward, terminal, _ = env.step(action.item())
                state = convert(state)
                total_reward += reward

                # history mem includes current state
                old_history = history.get()
                new_history = history.append(state).get()
                tmp_observations.append(
                    {
                        "state": {"history_mem": old_history},
                        "action": {"action": action},
                        "next_state": {"history_mem": new_history},
                        "reward": reward,
                        "terminal": terminal,
                    }
                )
        env.render_all()
