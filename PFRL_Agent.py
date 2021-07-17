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
pd.options.mode.chained_assignment = None
from DoubleDQN import DoubleDQN

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
        statistics = agent.get_statistics()


        #salva scalari
        writer.add_scalar('Train/Reward', R, episode)
        writer.add_scalar('Train/average_q', statistics[0][1], episode)
        writer.add_scalar('Train/average_loss', statistics[1][1], episode)
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
        if episode % 2 ==  0:
            #SALVA Agente
            agent.save(PATH + f"/agents/agent_train{episode}")
            #CREA ARCHIVIO
            name_archivie = folder_name + "_Train:" + str(episode)
            shutil.make_archive(name_archivie, 'zip', PATH)
            #Upload su mega
            m.upload(name_archivie + ".zip", PATH_TO_MEGA_FOLDER)
            #Rimuove  il folder PATH e lo ricrea e l'archivuo inviato a MEga
            shutil.rmtree(PATH, ignore_errors=True)
            os.remove(name_archivie + ".zip")

            #
            create_folders(PATH)

        plt.close()

def evalute_agent(agent, id_str, data, n_episodes = 100):

    writer = SummaryWriter(tensorboard_path)
    inizio = 100000

    with agent.eval_mode():
        for episode in range(1, n_episodes):
            R = 0
            fine = inizio + 60
            env = gym.make(id_str, df=data, frame_bound= (inizio, fine), window_size=22)
            obs = env.reset()
            print(f"eval episode: {episode + 1} of {n_episodes}")
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
            writer.add_scalar('Evaluation/Reward', R, episode)
            #plt.figure(figsize=(0.00000001, 0.00000000001))
            #plt.title(info_eval)
            env.render_all()
            plt.savefig(PATH + f"/visualization/eval/graph_eval{episode}")

            if episode % 2 == 0:
                # CREA ARCHIVIO
                name_archivie = folder_name + "_Eval:" + str(episode)
                shutil.make_archive(name_archivie, 'zip', PATH)
                # Upload su mega
                m.upload(name_archivie + ".zip", PATH_TO_MEGA_FOLDER)
                # Rimuove  il folder PATH e lo ricrea
                shutil.rmtree(PATH, ignore_errors=True)
                create_folders(PATH)
                # rimuove l'archivuo inviato a MEga
                os.remove(name_archivie + ".zip")

            plt.close()
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



def directory_choice_load_agent():

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
        choice = int(input())
        if choice in range(len(list_of_agents)):
            path_to_agent = list_of_agents[choice]
            break
        if choice == -1:
            break

    return trained_agent_dirname + "/" + path_to_agent

def load_agent(agent):
    path_to_agent = directory_choice_load_agent()
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

    obs_size = env.observation_space.low.size
    n_actions = env.action_space.n
    agent = DoubleDQN(obs_size, n_actions)

    if False:
        load_agent(agent)

    train_agent(env, agent, n_episodes=10)
    evalute_agent(agent, id_str, data, n_episodes = 10)

if __name__ == "__main__":
  main()
