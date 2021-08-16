

import pandas as pd
from agents.Dqn_agent import DQNAgent
from env.ShortLongCTE_no_invalid import CryptoTradingEnv
import gym

# Register enviroment
id_str = 'cryptostocks-v1'

if 'cryptostocks-v1' in list(gym.envs.registry.env_specs.keys()):
  del gym.envs.registry.env_specs['cryptostocks-v1']

from gym.envs.registration import register
register(
    id=id_str,
    entry_point=CryptoTradingEnv,
)

#Load data
df = pd.read_csv('\\Users\\Ene\\PycharmProjects\\DRL_finance\\data\\Binance_BTCUSDT_minute.csv', skiprows=1)
df = df.rename(columns={'Volume USDT': 'volume'})
df = df.iloc[::-1]
df = df.drop(columns=['symbol', 'Volume BTC'])
df['date'] = pd.to_datetime(df['unix'],unit='ms')
df = df.set_index("date")
df = df.drop(columns=['unix'])

env = gym.make(id_str, df=df, frame_bound=(122,326), window_size=22)
obs_size = env.observation_space.shape[0] * env.observation_space.shape[1]

agent = DQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001,
                                  input_dims=(obs_size),
                                  n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
                                  batch_size=32, replace=10000, eps_dec=1e-5,
                                  chkpt_dir='/content/models/', algo='DuelingDDQNAgent',
                                  env_name=id_str)

agent.train(env)