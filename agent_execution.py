import pandas as pd
import os
import ast
from pathlib import Path
from agents.Dqn_agent import DQNAgent
from agents.DDqn_agent import DDQNAgent
from agents.Duelling_DDqn_agent import DuelingDDQNAgent
from agents.RDQN_agent import DRQNAgent
import torch
from env.ShortLongCTE_no_invalid import CryptoTradingEnv

hyperparameter = {
    "agent_id" : 1,
    "hyperparameter" : {
        "agent_type": "DQN",
        "gamma": 0.99,
        'epsilon': 0.9,
        "lr": 0.001,
        "input_dims": 22,
        "mem_size": 1000,
        "batch_size": 64,
        "eps_min": 0.01,
        "eps_dec": 5e-7,
        "replace": 1000,
        "n_neurons_layer": 512,
        "dropout": 0.1
        }
}

env_parameter = {
    "env_id": 20,
    "parameter" : {
        "frame_bound" : (200, 250),
        #"frame_bound_low": 200,
        #"frame_bound_high": 250,
        "reward_option": "profit",
        "window_size": 22,
        "position_in_observation" : True,
        "indicators": ['diff_pct_1', 'diff_pct_5', 'diff_pct_15', 'diff_pct_22']
    }
}

row = {
    "agent_type": "DQN",
    "agent_id": "1",
    "env_id": "20"
    }

def load_agent(id_agent, path):
    agents_csv = "tuning/agents.csv"
    df_agent = pd.read_csv(path + "/" + agents_csv, sep=";")
    df_agent = df_agent.set_index("agent_id")
    agents = df_agent.to_dict(orient="index")
    agent = agents[id_agent]
    agent_hyperparameter = ast.literal_eval(agent["hyperparameter"])
    return agent_hyperparameter

def load_env(id_env, path):

    envs_csv = "tuning/envs.csv"

    df_env = pd.read_csv(path + "/" + envs_csv, sep=";")
    df_env = df_env.set_index("env_id")
    envs = df_env.to_dict(orient="index")
    env = envs[id_env]
    env_parameter = ast.literal_eval(env["parameter"])
    return env_parameter

def insert_agent_row(agent_hyperparameter, path):

    agents_csv = "tuning/agents.csv"
    agent_row = pd.DataFrame.from_dict(agent_hyperparameter, orient='index')
    agent_row = agent_row.transpose()
    agent_row = agent_row.set_index("agent_id")

    agent_row.to_csv(path + "/" + agents_csv, sep=";", mode='a', header=None)

def insert_env_row(env_parameter, path):
    envs_csv = "tuning/envs.csv"
    env_row = pd.DataFrame.from_dict(env_parameter, orient='index')
    env_row = env_row.transpose()
    env_row = env_row.set_index("env_id")

    env_row.to_csv(path + "/" + envs_csv, sep=";", mode='a', header=None)

def load_data(coin):
    # Load data
    path = os.getcwd()


    df = pd.read_csv(f"{path}/data/Binance_{coin}USDT_minute.csv", skiprows=1)
    df = df.rename(columns={'Volume USDT': 'volume'})
    df = df.iloc[::-1]
    df = df.drop(columns=['symbol', f"Volume {coin}"])
    df['date'] = pd.to_datetime(df['unix'], unit='ms')
    df = df.set_index("date")
    df = df.drop(columns=['unix'])
    return df

def create_env(env_paramenter, coin):
    dataframe = load_data(coin)
    env_paramenter["df"] = dataframe
    env = CryptoTradingEnv(**env_paramenter)
    return env

def create_agent(agent_hyperparameter):

    agent_type = agent_hyperparameter.pop("agent_type")

    if agent_type == "DQN":
        agent = DQNAgent(**agent_hyperparameter)

    elif agent_type == "DDQN":
        agent = DDQNAgent(**agent_hyperparameter)

    elif agent_type == "DDDQN":
        agent = DuelingDDQNAgent(**agent_hyperparameter)

    else:
        agent = DRQNAgent(**agent_hyperparameter)

    return agent


def iterative_train(coin, agent, env):
    agent.train(env, coin, n_episodes=100, checkpoint_freq=10)

def select_env(id_env,coin):
    path = os.getcwd()
    env_paramenter = load_env(id_env, path)
    env = create_env(env_paramenter, coin)
    return env

def select_agent(id_agent,env,coin):
    path = os.getcwd()
    agent_hyperparameter = load_agent(id_agent, path)
    agent_type = agent_hyperparameter["agent_type"]
    chkpt_dir = path + f"/trained_agents/{agent_type}/{id_agent}/{coin}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    n_actions = env.action_space.n


    agent_hyperparameter["n_actions"] = n_actions
    agent_hyperparameter["input_dims"] = obs_size
    agent_hyperparameter["chkpt_dir"] = chkpt_dir
    agent_hyperparameter["device"] = device
    agent_hyperparameter["id_agent"] = 10

    agent = create_agent(agent_hyperparameter)

    return agent

coin = "BTC"
env = select_env(20,coin)
agent = select_agent(1, env, coin)


