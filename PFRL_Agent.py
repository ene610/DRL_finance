from ShortLongCTE import CryptoTradingEnv
import pfrl
import torch
import pandas as pd
import numpy
import numpy as np
import gym
import matplotlib.pyplot as plt
import os
import datetime
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from mega import Mega
import shutil
mega = Mega()

email = ""
password = ""

m = mega.login(email, password)


datetime_object = datetime.datetime.now()
year = str(datetime_object.year)
month = str(datetime_object.month)
day = str(datetime_object.day)
hour = str(datetime_object.hour)
minute = str(datetime_object.minute)
folder_name = "Long_short_" + day + "-" + month + "-" + year + "|" + hour + ":" + minute
abs_path = os.path.abspath(os.getcwd())
tensorboard_path = abs_path + "/tensorboard_" + folder_name


PATH = abs_path + "/" + folder_name
PATH_TO_MEGA_FOLDER = m.create_folder(folder_name)[folder_name]


def create_folders(PATH):
    os.mkdir(PATH)
    os.mkdir(PATH + "/agents" )
    os.mkdir(PATH + "/visualization" )
    os.mkdir(PATH + "/visualization/train" )
    os.mkdir(PATH + "/visualization/eval" )

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

def train_agent(env, agent, n_episodes = 100):

    writer = SummaryWriter(tensorboard_path)
    for episode in range(1, n_episodes + 1):

        print(f"training episode:{episode} of {n_episodes + 1}")
        obs = env.reset()
        obs = obs.flatten()
        R = 0  # return (sum of rewards)
        t = 0  # time step

        while True:

            # Uncomment to watch the behavior in a GUI window
            # env.render()
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            obs = obs.flatten()
            R += reward
            t += 1
            reset = False
            agent.observe(obs, reward, done, reset)

            if done:
                break

        #salva scalari
        writer.add_scalar('Reward/train', R, episode)
        #TODO inserisci anche reward e loss
        #salva agente
        os.mkdir(PATH + f"/agents/agent_train{episode}")
        agent.save(PATH + f"/agents/agent_train{episode}")
        #salva render
        info_training = f"episode:{episode}/{n_episodes}"
        env.render_all()
        plt.title(info_training)
        plt.savefig(PATH + f"/visualization/train/graph_train{episode}")

        #ogni tot salva su mega e pulisce la directory con render e agente
        if episode % 100 ==  0:
            #SALVA Agente
            agent.save(PATH + f"/agents/agent_train{episode}")
            #CREA ARCHIVIO
            name_archivie = folder_name + "_Train:" + str(episode)
            shutil.make_archive(name_archivie, 'zip', PATH)
            #Upload su mega
            m.upload(name_archivie + ".zip", PATH_TO_MEGA_FOLDER)
            #Rimuove  il folder PATH e lo ricrea
            shutil.rmtree(PATH, ignore_errors=True)
            create_folders(PATH)
            #rimuove l'archivuo inviato a MEga
            os.remove(name_archivie + ".zip")

def evalute_agent(agent, id_str, data, n_episodes = 100):

    writer = SummaryWriter(tensorboard_path)
    inizio = 100000

    with agent.eval_mode():
        for episode in range(1, n_episodes):
            R = 0
            fine = inizio + 1440
            env = gym.make(id_str, df=data, frame_bound= (inizio, fine), window_size=22)
            obs = env.reset()

            while True:
                # Uncomment to watch the behavior in a GUI window
                # env.render()
                action = agent.act(obs)

                obs, reward, done, _ = env.step(action)
                R += reward
                reset = False
                agent.observe(obs, reward, done, reset)
                if done:
                    break

            # TODO spedisci tutto a mega

            info_eval = f"episode:{episode}/{n_episodes}"
            writer.add_scalar('Reward/evaluation', R, episode)
            plt.figure(figsize=(0.00000001, 0.00000000001))
            plt.title(info_eval)
            env.render_all()
            plt.savefig(PATH + f"/visualization/eval/graph_eval{episode}")

            if episode > 0:
                # CREA ARCHIVIO
                name_archivie = folder_name + "_Eval:" + str(episode)
                shutil.make_archive(name_archivie, 'zip', PATH)
                # Upload su mega
                m.upload(name_archivie + ".zip", PATH_TO_MEGA_FOLDER)
                os.remove(name_archivie + ".zip")
                # Rimuove  il folder PATH e lo ricrea
                shutil.rmtree(PATH, ignore_errors=True)
                create_folders(PATH)
                # rimuove l'archivuo inviato a MEga
                os.remove(name_archivie + ".zip")

            inizio = fine

    print('Finished.')

def create_enviroment(id_str):

    if id_str in gym.envs.registry.env_specs:
        del gym.envs.registry.env_specs[id_str]

    from gym.envs.registration import register
    register(
        id=id_str,
        entry_point=CryptoTradingEnv,
    )

def load_data():
    df = pd.read_csv('https://github.com/ene610/DRL_finance/blob/main/data/Binance_BTCUSDT_minute.csv?raw=true',
                     skiprows=1)
    df = df.rename(columns={'Volume USDT': 'volume'})
    df = df.iloc[::-1]
    df = df.drop(columns=['symbol', 'Volume BTC'])
    df['date'] = pd.to_datetime(df['unix'], unit='ms')
    df = df.set_index("date")
    df = df.drop(columns=['unix'])
    return df

def save_model(q_func):
    PATH = "/content/sample_data/alfa"
    print(type(q_func))
    torch.save(q_func.state_dict(), PATH)

def create_agent(env):
    obs_size = env.observation_space.low.size
    n_actions = env.action_space.n
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

def directory_choiche_load_agent():
    abs_path = os.path.abspath(os.getcwd())
    trained_agent_dirname = abs_path + "/trained_agents"
    path_to_agent = None
    while True:
        i = 0
        list_of_agents = os.listdir(trained_agent_dirname)
        for agent in list_of_agents:
            print(f"[{i}]" + agent)
            i += 1
        print("agent to load (-1 for exit)")
        choiche = input()
        if choiche in range(len(os.listdir(trained_agent_dirname))):
            path_to_agent = list_of_agents[choiche]
            break
        if choiche == -1:
            break

    return path_to_agent

    return
def load_agent(agent):
    path_to_agent = directory_choiche_load_agent
    if path_to_agent:
        agent.load(path_to_agent)
    return agent


def main():
    create_folders(PATH)
    #crea tensorboard folder
    os.mkdir(tensorboard_path)

    data = load_data()
    id_str = "Long-Short-Crypto-env-v1"
    create_enviroment(id_str)
    #TODO cambia a 100k
    env = gym.make(id_str, df=data, frame_bound=(22, 1440), window_size = 22)
    agent = create_agent(env)

    if False:
        load_agent(agent)

    train_agent(env, agent, n_episodes=1000)
    evalute_agent(agent, id_str, data)

if __name__ == "__main__":
  main()
