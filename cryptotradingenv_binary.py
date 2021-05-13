from enum import Enum
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from indicators import Indicators

class Actions(Enum):
    '''
    Se in Free:
        - ToFree: rimango in Free
        - ToLongo: vado in Long
    Se in Long:
        - ToFree: vado in Free
        - ToLong: vado in Long
    '''
    ToFree = 0
    ToLong = 1


class Positions(Enum):
    # La posizione è lo stato attuale in cui si trova l'algoritmo.
    Free = 0
    Long = 1

    def opposite(self):
        return Positions.Free if self == Positions.Long else Positions.Long

class CryptoTradingEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, df, frame_bound, window_size):
        assert df.ndim == 2
        assert len(frame_bound) == 2

        self.seed()
        self.df = df
        self.frame_bound = frame_bound
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        # episode
        self._balance = 10000.
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._done = None
        self._current_tick = None
        self._open_position_tick = None
        self._holding_price_difference = np.zeros(0)
        self._last_trade_tick = None
        self._position = Positions.Free
        self._position_history = None
        self._total_reward = 0.
        self._total_profit = 0.
        self._balance = 10000
        self._first_rendering = None
        self.history = None

        # stats
        self._n_trades = 0
        self._n_neg_trades = 0
        self._n_pos_trades = 0
        self._done_nothing = 0


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._balance = 10000.
        self._done = False
        self._current_tick = self._start_tick
        self._open_position_tick = 0
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Free
        self._position_history = (self.window_size * [0]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 0.  # unit
        self._first_rendering = True
        self.history = {}
        self._holding_price_difference = np.zeros(0)

        # reset stats
        self._n_trades = 0
        self._n_neg_trades = 0
        self._n_pos_trades = 0
        self._done_nothing = 0

        return self._get_observation()

    def step(self, action):
        '''
        Responsabilità di step():
            1. calcolare il profit attraverso la funzione di update_profit
            2. calcolare step_reward, position, open_position_tick tramite la funzione calculate_reward()
            4. aggiornare la history delle posizioni
        '''
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        self._update_profit(action)
        # Attenzione! self._position viene cambiata in step_reward quindi update_profit() deve essere chiamato prima
        step_reward = self._calculate_reward(action)

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = dict(
            total_reward = self._total_reward,
            total_profit = self._total_profit,
            position = self._position.value,
            trades = self._n_trades,
            positive_trades = self._n_pos_trades,
            negative_trades = self._n_neg_trades,
            done_nothing = self._done_nothing
        )
        self._update_history(info)

        return observation, step_reward, self._done, info

    def _calculate_reward(self, action):
        '''
        Free:
            - ToFree = 0
            - ToLong = reward di incentivo
        Long:
            - ToFree = reward di profit
            - ToLong = reward di holding

        :param action:
        :return:
        '''
        step_reward = 0
        new_position = self._position

        # (Free, ToLong) -> Free
        if self._position == Positions.Free and action == Actions.ToLong.value:
            new_position = self._position.opposite()
            self._open_position_tick = self._current_tick
            step_reward = 0.15

        # (Long, ToLong) -> Long
        elif self._position == Positions.Long and action == Actions.ToLong.value:
            current_price = self.prices[self._current_tick]
            open_position_price = self.prices[self._open_position_tick]
            price_diff = (current_price - open_position_price) / open_position_price
            # reward = media dei reward ottenuti dall'hodlding?

            if self._holding_price_difference.size > 1:
                if np.sign(price_diff) != np.sign(self._holding_price_difference[0]):
                    self._holding_price_difference = np.zeros(0)

            np.append(self._holding_price_difference, price_diff)
            step_reward = np.mean(self._holding_price_difference)
            if np.isnan(step_reward):
                step_reward = 0

        # (Long, ToFree) -> Free
        elif self._position == Positions.Long and action == Actions.ToFree.value:
            new_position = self._position.opposite()
            current_price = self.prices[self._current_tick]
            open_position_price = self.prices[self._open_position_tick]
            price_diff = current_price - open_position_price
            if price_diff >= 0:
                step_reward = 1
                self._n_pos_trades += 1
            else:
                step_reward = -1
                self._n_neg_trades += 1
            self._holding_price_difference = np.zeros(0)
            self._n_trades += 1

        # (Free, ToFree) -> Free
        elif self._position == Positions.Free and action == Actions.ToFree.value:
            self._done_nothing += 1

        self._total_reward += step_reward
        self._position = new_position
        return step_reward

    def _update_profit(self, action):
        temp_balance = self._balance
        if self._position == Positions.Long and action == Actions.ToFree.value:
            current_price = self.prices[self._current_tick]
            open_position_price = self.prices[self._open_position_tick]
            self._balance = (self._balance / open_position_price) * current_price
            #se total profit inizializzato a 0
            self._total_profit += self._balance - temp_balance

    def _get_observation(self):
        # TODO aggiungere una colonna con la posizione mantenuta dall'agente nel dato timestep
        return self.signal_features[(self._current_tick - self.window_size):self._current_tick]

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def render_all(self, mode='human'):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        for i, tick in enumerate(window_ticks):
            if i != 0:
                current_tick = self._position_history[i]
                prev_tick = self._position_history[i - 1]
                if current_tick != prev_tick:
                    if current_tick == Positions.Free:
                        short_ticks.append(tick)
                    elif current_tick == Positions.Long:
                        long_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], 'ro')
        plt.plot(long_ticks, self.prices[long_ticks], 'go')

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit + ' ~ ' +
            "Balance: %.6f" % self._balance
        )

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def _process_data(self):
        prices = self.df['close'].to_numpy()

        #prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        #diff = np.insert(np.diff(prices), 0, 0)
        #signal_features = np.column_stack((prices, diff))

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
        print(signal_features.columns)
        return prices, signal_features.to_numpy()
