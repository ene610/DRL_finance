

import pandas as pd
from agents.Dqn_agent import DQNAgent
from env.ShortLongCTE_no_invalid import CryptoTradingEnv
import gym
import os
import torch as T

# Register enviroment
id_str = 'cryptostocks-v1'

if 'cryptostocks-v1' in list(gym.envs.registry.env_specs.keys()):
  del gym.envs.registry.env_specs['cryptostocks-v1']

from gym.envs.registration import register
register(
    id=id_str,
    entry_point=CryptoTradingEnv,
)

def load_data(coin):

        #Load data
        path = os.getcwd()
        df = pd.read_csv(f"{path}\\data\\Binance_{coin}USDT_minute.csv", skiprows=1)
        df = df.rename(columns={'Volume USDT': 'volume'})
        df = df.iloc[::-1]
        df = df.drop(columns=['symbol', f"Volume {coin}"])
        df['date'] = pd.to_datetime(df['unix'], unit='ms')
        df = df.set_index("date")
        df = df.drop(columns=['unix'])
        return df

def iterative_train(coin):

      data = load_data(coin)
      env = gym.make(id_str, df=data, frame_bound=(122,500), window_size=22,reward_option = "profit")
      chkpt_dir = os.getcwd() + f"/trained_agents/DQN/{coin}"
      device = T.device("cuda" if T.cuda.is_available() else "cpu")
      obs_size = env.observation_space.shape[0] * env.observation_space.shape[1]
      agent =DQNAgent(gamma = 0.99,
                      epsilon = 1.0,
                      lr = 0.0001,
                      input_dims = (obs_size),
                      n_actions = env.action_space.n,
                      mem_size = 50000,
                      eps_min = 0.1,
                      batch_size = 32,
                      replace=10000,
                      eps_dec=1e-5,
                      chkpt_dir=chkpt_dir,
                      seed = 1,
                      device = device,
                      n_neurons_layer=512,
                      dropout=0.1
                      )

      agent.train(env,coin,n_episodes=10, checkpoint_freq=10)




