from machin.frame.algorithms import DQN
from machin.utils.logging import default_logger as logger
import torch as t
import torch.nn as nn
import gym

# configurations
# maximum and minimum of reward value
# since reward is 1 for every step, maximum q value should be
# below 20(reward_future_steps) * (1 + discount ** n_steps) < 40

class QNet(nn.Module):
    def __init__(self, state_dim, action_num):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_num)

    def forward(self, state):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        return self.fc3(a)

class MachinDDQN():
    # model definition

    def __init__(self, observe_dim, action_num):

        self.observe_dim = observe_dim
        q_net = QNet(observe_dim, action_num)
        q_net_t = QNet(observe_dim, action_num)

        #cambia nome
        self.ddqn = DQN(q_net, q_net_t, t.optim.Adam, nn.MSELoss(reduction="sum"),mode = "double")

    def train(self, env, max_episodes=250):
        episode, step, reward_fulfilled = 0, 0, 0
        smoothed_total_reward = 0
        max_steps = 10000
        while episode < max_episodes:
            episode += 1
            total_reward = 0
            terminal = False
            step = 0
            state = t.tensor(env.reset(), dtype=t.float32).reshape(1, self.observe_dim)

            tmp_observations = []
            while not terminal and step <= max_steps:
                step += 1
                with t.no_grad():
                    old_state = state
                    # agent model inference
                    action = self.ddqn.act_discrete_with_noise({"state": old_state})
                    reward = float(reward)
                    state, reward, terminal, _ = env.step(action.item())
                    state = t.tensor(state, dtype=t.float32).reshape(1, self.observe_dim)
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

            self.ddqn.store_episode(tmp_observations)

            # update, update more if episode is longer, else less
            if episode > 100:
                for _ in range(step):
                    self.ddqn.update()

            # show reward
            smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
            logger.info(
                f"Episode {episode} smoothed_total_reward={smoothed_total_reward:.2f} total_reward={total_reward} ")

    def evaluate(self, env):

        total_reward = 0
        terminal = False
        step = 0
        max_steps = 200000
        state = t.tensor(env.reset(), dtype=t.float32).reshape(1, self.observe_dim)
        tmp_observations = []

        while not terminal and step <= max_steps:
            step += 1
            with t.no_grad():
                old_state = state
                # agent model inference
                action = self.ddqn.act_discrete({"state": old_state}, use_target=True)
                state, reward, terminal, _ = env.step(action.item())
                reward = float(reward)
                state = t.tensor(state, dtype=t.float32).reshape(1, self.observe_dim)
                total_reward += reward

                tmp_observations.append({
                    "state": {"state": old_state},
                    "action": {"action": action},
                    "next_state": {"state": state},
                    "reward": reward,
                    "terminal": terminal or step == max_steps,
                })
        env.render_all()