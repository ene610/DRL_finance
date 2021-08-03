from machin.frame.algorithms import RAINBOW
from machin.utils.logging import default_logger as logger
import torch as t
import torch.nn as nn
import gym

# configurations
# maximum and minimum of reward value
# since reward is 1 for every step, maximum q value should be
# below 20(reward_future_steps) * (1 + discount ** n_steps) < 40




class MachinRainbow():
# model definition
    class QNet(nn.Module):
        # this test setup lacks the noisy linear layer and dueling structure.
        def __init__(self, state_dim, action_num, atom_num=10):
            super().__init__()

            self.fc1 = nn.Linear(state_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, 512)
            self.fc4 = nn.Linear(512, 512)
            self.fc5 = nn.Linear(512, 512)
            self.fc6 = nn.Linear(512, action_num * atom_num)
            self.action_num = action_num
            self.atom_num = atom_num

        def forward(self, state):
            a = t.relu(self.fc1(state))
            a = t.relu(self.fc2(a))
            a = t.relu(self.fc3(a))
            a = t.relu(self.fc4(a))
            a = t.relu(self.fc5(a))
            return t.softmax(self.fc6(a).view(-1, self.action_num, self.atom_num), dim=-1)


    def __init__(self,observe_dim,action_num):
        value_max = 40
        value_min = 0
        reward_future_steps = 20
        q_net = self.QNet(observe_dim, action_num)
        q_net_t = self.QNet(observe_dim, action_num)

        rainbow = RAINBOW(
            q_net,
            q_net_t,
            t.optim.Adam,
            value_min,
            value_max,
            reward_future_steps=reward_future_steps,
        )

    def train(self, env, max_episodes = 250):
        episode, step, reward_fulfilled = 0, 0, 0
        smoothed_total_reward = 0
        max_steps = 10000
        while episode < max_episodes:
            episode += 1
            total_reward = 0
            terminal = False
            step = 0
            state = t.tensor(env.reset(), dtype=t.float32).reshape(1, observe_dim)

            tmp_observations = []
            while not terminal and step <= max_steps:
                step += 1
                with t.no_grad():
                    old_state = state
                    # agent model inference
                    action = self.rainbow.act_discrete_with_noise({"state": old_state})
                    reward = float(reward)
                    state, reward, terminal, _ = env.step(action.item())
                    state = t.tensor(state, dtype=t.float32).reshape(1, observe_dim)
                    total_reward += reward

                    tmp_observations.append(
                        {
                            "state": {"state": old_state},
                            "action": {"action": action},
                            "next_state": {"state": state},
                            "reward": reward,
                            "terminal": terminal or step == max_steps,
                        }
                    )

            self.rainbow.store_episode(tmp_observations)

            # update, update more if episode is longer, else less
            if episode > 100:
                for _ in range(step):
                    self.rainbow.update()

            # show reward
            smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
            logger.info(f"Episode {episode} smoothed_total_reward={smoothed_total_reward:.2f} total_reward={total_reward} ")

    def evaluate(self,env):
        total_reward = 0
        terminal = False
        step = 0
        max_steps = 200000
        state = t.tensor(env.reset(), dtype=t.float32).reshape(1, observe_dim)
        tmp_observations = []
        while not terminal and step <= max_steps:
                    step += 1
                    with t.no_grad():
                        old_state = state
                        # agent model inference
                        action = self.rainbow.act_discrete({"state": old_state},use_target = True)
                        state, reward, terminal, _ = env.step(action.item())
                        reward = float(reward)
                        state = t.tensor(state, dtype=t.float32).reshape(1, observe_dim)
                        total_reward += reward

                        tmp_observations.append({
                                "state": {"state": old_state},
                                "action": {"action": action},
                                "next_state": {"state": state},
                                "reward": reward,
                                "terminal": terminal or step == max_steps,
                            }
                        )
        env.render_all()