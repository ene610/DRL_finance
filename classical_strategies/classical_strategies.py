import numpy as np
import pandas as pd
from env.ShortLongCTE_no_invalid import CryptoTradingEnv
from agent_execution import load_data

# Utilizzare uno schema fisso
# Quando cominci: compra
# Holda fino alla fine
class BuynHoldAgent():

    def __init__(self):
        return

    # return action
    def act(self, obs):
        # 1 = OpenLongPos
        return 1

class SellnHoldAgent():

    def __init__(self):
        return

    # return action
    def act(self, obs):
        # 2 = OpenShortPos
        return 2

class MomentumAgent():

    def __init__(self):
        return

    def act(self, obs):
        #TODO cambiare nomi colonne
        #obs deve contenere solo le chiusure
        obs = pd.DataFrame(obs,columns=["close"])
        #print(obs)
        obs['9-day'] = obs['close'].rolling(9).mean()
        obs['20-day'] = obs['close'].rolling(20).mean()

        obs['signal'] = np.where(obs['9-day'] > obs['20-day'], 1, 0)
        obs['signal'] = np.where(obs['9-day'] < obs['20-day'], 2, obs['signal'])

        action = int(obs["signal"].iloc[-1])
        return action


class MovingAverageCrossoverAgent():

    # The second thing of importance is coming to understand the
    # trigger for trading with moving average crossovers.
    # A buy or sell signal is triggered once the smaller moving average crosses above or
    # below the larger moving average, respectively.

    def __init__(self):
        return

    # obs deve contentere ['closes', 'sma']
    # obs dovrebbe essere un numpy
    # TODO controllare che il valore di default di sma in Indicators
    def act(self, obs, short_sma=1, long_sma=24):
        obs = pd.DataFrame(obs, columns=["close", "sma"])
        obs["ssma"] = obs["close"].rolling(window=short_sma).mean()
        obs["lsma"] = obs["close"].rolling(window=long_sma).mean()

        pre_ssma, ssma = obs.ssma.values[-2], obs.ssma.values[-1]
        pre_lsma, lsma = obs.lsma.values[-2], obs.lsma.values[-1]

        # BUY SIGNAL = se sma5 incrocia (cioè supera da sotto verso sopra) sma10 compro
        if pre_ssma < pre_lsma and ssma > lsma:
            return 1

        if pre_ssma > pre_lsma and ssma < lsma:
            return 2

        if ssma < lsma:
            return 2
        if ssma > lsma:
            return 1

        return 0

from datetime import datetime


#Statements



def find_best_MAC_agent():
    start = datetime.now()
    print("MOVING AVERAGE CROSSOVER agent")
    moving_average_crossover = MovingAverageCrossoverAgent()
    max_profit = -np.inf
    best_l = 1
    best_h = 1
    #questi in hyperparametri
    frame_bound_lower = 200
    frame_bound_upper = 400
    df = load_data("BTC")
    env = CryptoTradingEnv(df=df, frame_bound=(frame_bound_lower, frame_bound_upper), window_size=50,
                           indicators=['close', 'SMA'], position_in_observation=False)

    for l in range(0, 48):

      print("l =", l)
      print("Max profit = ", max_profit, " --- (l, h) = ", best_l, best_h)

      for h in range(l+1, 49):
        #env = gym.make(id_str, df=df, frame_bound=(2500, 5000), window_size=50, indicators=['close', 'SMA'], position_in_observation=False)
        obs = env.reset()

        for i in range(frame_bound_upper - (frame_bound_lower + 1)):
          action = moving_average_crossover.act(obs, l, h)
          obs, step_reward, done, info = env.step(action)

        if action == 1:
          action = 2
        else:
          action = 1

        obs, step_reward, done, info = env.step(action)

        if info.get("total_profit") > max_profit:
          max_profit = info.get("total_profit")
          best_l = l
          best_h = h
          print("Update max profit =", max_profit, " --- (l, h) =", best_l, best_h)

    print("Max profit = ", max_profit, " --- (l, h) = ", best_l, best_h)
    print(datetime.now() - start)
find_best_MAC_agent()


# ['one', 'two', 'three', 'four', 'five']