import pandas as pd
import os
import ast
from pathlib import Path
from agents.Dqn_agent import DQNAgent
from agents.DDqn_agent import DDQNAgent
from agents.Duelling_DDqn_agent import DuelingDDQNAgent
from agents.DRQN_agent import DRQNAgent
import torch
from env.ShortLongCTE_no_invalid import CryptoTradingEnv

hyperparameter_dummy = {
    "agent_id" : 1,

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
        "dropout": 0.1,
        "random_update": None,
        "lookup_step": None,
        "max_epi_len": None,

}

env_parameter_dummy = {
    "env_id": 20,
    "frame_bound" : (200, 250),
    "reward_option": "profit",
    "window_size": 22,
    "position_in_observation" : True,
    "indicators": ['diff_pct_1', 'diff_pct_5', 'diff_pct_15', 'diff_pct_22']
}

obs_type_dummy = {
    "obs_type_id" : 2,
    "position_in_observation": True,
    "indicators": ['diff_pct_1', 'diff_pct_5', 'diff_pct_15',"diff_pct_22"]
}


hyperparameter_DRQN_dummy = {
    "agent_id" : 100,

        "agent_type": "DRQN",
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
        "dropout": 0.1,
        #Cambiano solo questi 3
        "random_update":True,
        "lookup_step":10,
        "max_epi_len":3000,

}

def load_agent(id_agent, path):
    agents_csv = "tuning/agents.csv"
    df_agent = pd.read_csv(path + "/" + agents_csv, sep=";")
    df_agent = df_agent.set_index("agent_id")
    row_agent = df_agent.loc[id_agent, :].dropna(axis=0, inplace=False)
    agent_hyperparameter = row_agent.to_dict()

    return agent_hyperparameter

# def load_env(id_env, path):
#
#     envs_csv = "tuning/envs.csv"
#
#     df_env = pd.read_csv(path + "/" + envs_csv, sep=";")
#     df_env = df_env.set_index("env_id")
#     row_env = df_env.loc[id_env, :].dropna(axis=0, inplace=False)
#     env_parameter = row_env.to_dict()
#     #convert string into tuple
#     env_parameter["frame_bound"] = ast.literal_eval(env_parameter["frame_bound"])
#     env_parameter["indicators"] = ast.literal_eval(env_parameter["indicators"])
#     return env_parameter

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
    #project_folder_string = "DRL_finance"
    #project_path = path.rsplit(project_folder_string, 1)[0] + project_folder_string

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

def select_env(id_env,obs_type_id,coin):
    path = os.getcwd()
    env_parameter = load_env(id_env,obs_type_id, path)
    env = create_env(env_parameter, coin)
    return env

def select_agent(id_agent,env,coin,id_train_env,id_obs_type,):

    path = os.getcwd()
    agent_hyperparameter = load_agent(id_agent, path)
    agent_type = agent_hyperparameter["agent_type"]
    chkpt_dir = path + f"/trained_agents/{agent_type}/{id_agent}_{id_train_env}_{id_obs_type}/{coin}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    n_actions = env.action_space.n

    agent_hyperparameter["n_actions"] = n_actions
    agent_hyperparameter["input_dims"] = obs_size
    agent_hyperparameter["chkpt_dir"] = chkpt_dir
    agent_hyperparameter["device"] = device
    agent_hyperparameter["id_agent"] = id_agent
    agent_hyperparameter["id_train_env"] = id_train_env
    agent_hyperparameter["id_obs_type"] = id_obs_type
    agent = create_agent(agent_hyperparameter)

    return agent

def train_agent(coin, agent, env, n_episodes, checkpoint_freq):
    agent.train(env, coin, n_episodes = n_episodes, checkpoint_freq = checkpoint_freq)

def evaluate_agent(coin, agent, env_eval, id_agent, env_eval_id, obs_type_id, env_train_id, n_episodes, checkpoint_freq):    #crea cartella per il plot
    save_fig_path = os.getcwd() + f"/plot/{id_agent}_{env_train_id}_{obs_type_id}/{coin}/{env_eval_id}"
    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)
    #iterativamente esegue la valutazione per tutti i checkpoint creati in fase di train
    for episode in range(0, n_episodes, checkpoint_freq):
        agent.load_models(episode)
        agent.evaluate(env_eval, coin, episode, env_id=env_eval_id).render_all(episode, savepath=save_fig_path)

#RICORDA 1: train_and_eval sovrascrive
#               tutte le cartelle di train con agent_id e train_id uguali
#               tutte le cartelle di eval con agent_id e eval_id uguali

#RICORDA 2: se cancelli agents o env csv, ricreali mettendo l'header:
#               env_row.to_csv(path + "/" + envs_csv, sep=";", mode='a')

def insert_obs_type_row(obs_type, path):

    obs_type_csv = "tuning/obs_type.csv"
    agent_row = pd.DataFrame.from_dict(obs_type, orient='index')
    agent_row = agent_row.transpose()
    agent_row = agent_row.set_index("obs_type_id")

    agent_row.to_csv(path + "/" + obs_type_csv, sep=";", mode='a', header=None)

def load_env(id_env,obs_type_id, path):

    envs_csv = "tuning/envs.csv"

    df_env = pd.read_csv(path + "/" + envs_csv, sep=";")
    df_env = df_env.set_index("env_id")
    row_env = df_env.loc[id_env, :].dropna(axis=0, inplace=False)
    env_parameter = row_env.to_dict()
    #convert string into tuple
    env_parameter["frame_bound"] = ast.literal_eval(env_parameter["frame_bound"])
    obs_type_dict = select_obs_type(obs_type_id)
    env_parameter["indicators"] = ast.literal_eval(obs_type_dict["indicators"])
    env_parameter["position_in_observation"] = obs_type_dict["position_in_observation"]
    return env_parameter

def select_obs_type(obs_type_id):
    obs_type_csv = "tuning/obs_type.csv"
    path = os.getcwd()
    df_env = pd.read_csv(path + "/" + obs_type_csv, sep=";")
    df_env = df_env.set_index("obs_type_id")
    row_env = df_env.loc[obs_type_id, :].dropna(axis=0, inplace=False)
    obs_type_dict = row_env.to_dict()


    return obs_type_dict

def train_and_eval(agent_id, env_train_id,obs_type_id, env_eval_ids, coin, n_episodes=100, checkpoint_freq=10):
    #env_eval_ids array di id su cui l'agente verrà valutato
    #train e eval vengono svolti su un singolo coin
    train_agent_on_env(agent_id, env_train_id, obs_type_id, coin, n_episodes=n_episodes, checkpoint_freq=checkpoint_freq)
        # #Train
        # env_train = select_env(env_train_id, coin)
        # agent = select_agent(agent_id, env_train, coin)
        # train_agent(coin, agent, env_train, n_episodes, checkpoint_freq)
    eval_agent_on_env(agent_id, env_train_id, obs_type_id, env_eval_ids, coin, n_episodes=n_episodes, checkpoint_freq=checkpoint_freq)
        # #Eval su tutti gli ambienti scelti
        # for env_eval_id in env_eval_ids:
        #     env_eval = select_env(env_eval_id, coin)
        #     evaluate_agent(coin, agent, env_eval, agent_id, env_eval_id, n_episodes, checkpoint_freq)

def train_agent_on_env(agent_id, env_train_id, obs_type_id, coin, n_episodes=100, checkpoint_freq=10):

    #Train
    env_train = select_env(env_train_id, obs_type_id, coin)
    agent = select_agent(agent_id, env_train, coin,env_train_id, obs_type_id,)
    train_agent(coin, agent, env_train, n_episodes, checkpoint_freq)

def eval_agent_on_env(agent_id, env_train_id,obs_type_id, env_eval_ids, coin, n_episodes=100, checkpoint_freq=10):

    # Train
    env_train = select_env(env_train_id, obs_type_id, coin)
    agent = select_agent(agent_id, env_train, coin,env_train_id, obs_type_id)

    for env_eval_id in env_eval_ids:
        env_eval = select_env(env_eval_id, obs_type_id, coin)
        evaluate_agent(coin, agent, env_eval, agent_id, env_eval_id, obs_type_id, env_train_id, n_episodes, checkpoint_freq)


# hyperparameter_dummy = {
#     "agent_id" : 1,
#
#         "agent_type": "DQN",
#         "gamma": 0.99,
#         'epsilon': 0.9,
#         "lr": 0.001,
#         "input_dims": 22,
#         "mem_size": 1000,
#         "batch_size": 10,
#         "eps_min": 0.01,
#         "eps_dec": 30e-5,
#         "replace": 1000,
#         "n_neurons_layer": 512,
#         "dropout": 0.1,
#         "random_update": None,
#         "lookup_step": None,
#         "max_epi_len": None,
# }
# path = os.getcwd()
# N_id = 1
# for i in [64,128,256]:
#     for j in [0.1,0.2,0.3]:
#         for k in [0.01,0.001,0.0001]:
#             hyperparameter_dummy["agent_id"] = N_id
#             hyperparameter_dummy["dropout"] = j
#             hyperparameter_dummy["lr"] = k
#             hyperparameter_dummy["n_neurons_layer"] = i
#             N_id += 1
#             insert_agent_row(hyperparameter_dummy,path)


# path = os.getcwd()
# #insert_obs_type_row(obs_type_dummy, path)
# env = load_env(31,1,path)
# print(env)
# for coin in ["BTC", "ETH", "ADA"]:
#     start_btc = 302081 - 2
#     end_btc = start_btc + 1440
#     print(load_data(coin)[start_btc:end_btc].head())
