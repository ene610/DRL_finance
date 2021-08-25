from enum import Enum
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from junk.indicators import Indicators

class Actions(Enum):
    # Mappo sono le azioni legittime nella forma:
    # StatoAzione
    Hold = 0
    Buy = 1
    Sell = 2

    # Potrei anche mappare solo le quattro azioni e poi dare il reward negativo sull'azione in base alla posizione corrente (che comunque devo tenermi salvata)


class Positions(Enum):
    # La posizione è lo stato attuale in cui si trova l'algoritmo.
    # In Free è permesso solo di fare DoNothing e OpenPosition
    # in Long è permesso di fare solo HoldPosition e ClosePosition
    Free = 0
    Long = 1

    def opposite(self):
        return Positions.Free if self == Positions.Long else Positions.Long


class OneTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, frame_bound, window_size, indicators=None):
        assert df.ndim == 2
        assert len(frame_bound) == 2
        self.indicators_to_use = indicators
        self.seed()
        self.df = df
        self.frame_bound = frame_bound
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        # TODO metti nel init il flag

        self.shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        self._balance = 10000.
        self.USDT = 10000.
        self.BTC = 0
        # episode

        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1

        self._current_tick = self._start_tick
        self._open_position_tick = None
        self._holding_price_difference = np.zeros(0)
        self._last_trade_tick = None

        self._position_history = None
        self._total_reward = 0.
        self._total_profit = 0.
        self._balance = 10000
        self._first_rendering = None
        self.history = None
        self._done = None
        self.info = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        self.buy_ticks = []
        self.hold_ticks = []
        self.sell_ticks = []

        self._open_position_tick = 0
        self._last_trade_tick = self._current_tick - 1

        self._total_reward = 0.
        self._total_profit = 0.  # unit
        self._first_rendering = True
        self.history = {}
        self._holding_price_difference = np.zeros(0)
        self.start_local_tick = self._current_tick
        self.USDT = 10000.
        self.BTC = 0
        self.valore_precedente = self.BTC * self.prices[self._current_tick] + self.USDT
        self.valore_successivo = 0
        self._done = False

        return self._get_observation()

    def step(self, action):
        '''
        Responsabilità di step():
            1. calcolare il profit attraverso la funzione di update_profit
            2. calcolare step_reward, position, open_position_tick tramite la funzione calculate_reward()
            4. aggiornare la history delle posizioni
        '''

        # 0 : hold
        # 1 : buy BTC
        # 2 : sell BTC
        self._current_tick += 1
        prezzo_BTC = self.prices[self._current_tick]
        final_reward = None
        if action == 1:
            # compra tutti i BTC in USDT al valore attuale
            if self.USDT > 0:
                self.BTC += self.USDT / prezzo_BTC
                self.USDT = 0
                self.buy_ticks.append(self._current_tick)
            else:
                self.hold_ticks.append(self._current_tick)

        elif action == 2:
            # vende tutti i BTC in USDT al valore attuale
            if self.BTC > 0:
                self.USDT += prezzo_BTC * self.BTC
                self.BTC = 0
                self._done = True
                self.sell_ticks.append(self._current_tick)
                if self.USDT > 10000:
                    final_reward = 1
                else:
                    final_reward = -1
            else:
                self.hold_ticks.append(self._current_tick)

        else:
            self.hold_ticks.append(self._current_tick)

        self.valore_successivo = self.BTC * prezzo_BTC + self.USDT

        if final_reward == None:
            step_reward = self._calculate_reward()
        else:
            step_reward = final_reward
            # step_reward = self._calculate_reward()
        self._total_reward += step_reward
        self.valore_precedente = self.valore_successivo
        observation = self._get_observation()
        # TODO cambiala come se fosse uno step
        info = dict(
            action=action,
            step_reward=step_reward,
            BTC=self.BTC,
            USDT=self.USDT,
            tick=self._current_tick
        )
        self.info = info
        self._update_history(info)

        return observation, step_reward, self._done, info

    def _calculate_reward(self):

        step_reward = self.valore_successivo / self.valore_precedente - 1

        return step_reward

    def _get_observation(self):
        signal_obs = self.signal_features[(self._current_tick - self.window_size):self._current_tick]

        obs = signal_obs

        return obs

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def render(self, mode='human'):
        pass

    def render_all(self, mode='human'):

        # plt.plot(xpoints, ypoints)

        plt.plot(self.buy_ticks, self.prices[self.buy_ticks], 'gs')
        plt.plot(self.hold_ticks, self.prices[self.hold_ticks], 'ys')
        plt.plot(self.sell_ticks, self.prices[self.sell_ticks], 'rs')

        total_tick = self.buy_ticks + self.hold_ticks + self.sell_ticks
        total_tick.sort()
        plt.plot(total_tick, self.prices[total_tick], 'blue')

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward)

    def close(self):
        plt.close()
        # print

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def _process_data(self):
        prices = self.df['close'].to_numpy()

        # prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0] - self.window_size:self.frame_bound[1]]

        # diff = np.insert(np.diff(prices), 0, 0)
        # signal_features = np.column_stack((prices, diff))

        signal_features = Indicators.sma_indicator(self.df)
        signal_features = Indicators.macd_indicator(signal_features)
        signal_features = Indicators.atr_indicator(signal_features)
        signal_features = Indicators.ema_indicator(signal_features)
        signal_features = Indicators.kc_indicator(signal_features)
        signal_features = Indicators.rsi_indicator(signal_features)
        signal_features = Indicators.mom_indicator(signal_features)
        signal_features = Indicators.vhf_indicator(signal_features)
        signal_features = Indicators.trix_indicator(signal_features)
        signal_features = Indicators.rocv_indicator(signal_features)
        signal_features = signal_features.drop(columns=['open', 'high', 'low', 'close', 'volume', 'tradecount'])
        signal_features = signal_features.dropna()

        signal_features.reset_index(drop=True, inplace=True)

        signal_features = self.filter_indicators(signal_features)
        signal_features = signal_features.round(decimals=3)

        return prices, signal_features.to_numpy()

    #############################################################################################################

    def max_possible_profit(self):  # trade fees are ignored
        raise NotImplementedError

    # ipoteticamente questa funzione può esser sostituita da una serie di if in _process_data
    # solo che mi pareva una pecionata
    def filter_indicators(self, signal_features):
        '''
                        Responsabilità di filter_indicators():
                            1. Filtrare le colonne che si vogliono utilizzare come osservazioni
                                attraverso la differenza tra l'insieme  degli indicatori calcolati
                                in _process_data e quelli indicati nel init nella variabile self.indicators_to_use
                        :param signal_features: Dataframe con tutti gli indicatori
                        :return filtered_signal_feature: Dataframe contente gli indicatori selezionati in init
        '''
        # self.indicators_to_use = ['SMA', 'MACD', 'MACD_signal_line', 'EMA', 'kc_middle', 'kc_upper', 'kc_lower', 'RSI', 'close_prev', 'MOM', 'VHF', 'TRIX', 'ROCV']
        self.indicators_to_use = ['MACD', 'MACD_signal_line', 'MOM', 'VHF', 'TRIX']
        indicators_column_to_filter = np.setdiff1d(signal_features.columns, self.indicators_to_use, assume_unique=True)
        filtered_signal_feature = signal_features.drop(columns=indicators_column_to_filter)

        return filtered_signal_feature
