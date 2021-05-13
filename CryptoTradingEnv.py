from enum import Enum
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from indicators import Indicators

class Actions(Enum):
    # Mappo sono le azioni legittime nella forma:
    # StatoAzione
    DoNothing = 0
    OpenPosition = 1
    HoldPosition = 2
    ClosePosition = 3

    # Potrei anche mappare solo le quattro azioni e poi dare il reward negativo sull'azione in base alla posizione corrente (che comunque devo tenermi salvata)


class Positions(Enum):
    # La posizione è lo stato attuale in cui si trova l'algoritmo.
    # In Free è permesso solo di fare DoNothing e OpenPosition
    # in Long è permesso di fare solo HoldPosition e ClosePosition
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
            balance = self._balance,
            position = self._position.value
        )
        self._update_history(info)

        return observation, step_reward, self._done, info

    def _calculate_reward(self, action):
        '''
        Responsabilità di calculate_reward:
            1. Calcolare il reward
            2. Aggiornare self._total_reward
            3. Aggiornare self._position
            4. Aggiornare self._open_position_tick

        Logica di reward per azioni LEGITTIME:
                    - ClosePosition (da Long a Free) = profit di quella posizione
                    - OpenPosition (da Free a Long) = reward di incentivo per aprire posizione
                    - HoldPosition (da Long a Long) = profit che verrebbe fatta da quando si è aperta la posizione se si vendesse in quel momento
                    - DoNothing (da Free a Free) = 0 di reward, non punisco e lo incentivo in nessun modo su questa posizione

        Logica di reward per azioni ILLEGITTIME (tutte le coppie non listate sopra):
                    - Do un piccolo reward negativo di disincentivo per tutte le azioni illeggitime
                        ma non faccio cambiare il sistema

        Funzione di transizione: (Stato, Azione) -> Stato

        :param action:
        :return:
        '''
        # è negativo quando fa un'azione invalida?
        step_reward = -0.01
        new_position = self._position

        # (Free, DoNothing) -> Free
        if action == Actions.DoNothing.value and self._position == Positions.Free:
            step_reward = 0

        # (Free, OpenPosition) -> Long
        elif action == Actions.OpenPosition.value and self._position == Positions.Free:
            new_position = self._position.opposite()
            self._open_position_tick = self._current_tick
            # 1 è troppo alto? Ma se metto a 0.95 fa un trade ogni ora
            # così ne fa uno ogni due tre minuti
            step_reward = 0.15

        # 0 aperta posizione a 10
        # holding_rewards = []
        # 1 prezzo a 12 -> hold 12-10 / 10 = 0.2 = 0.2
        # 2 prezzo a 15 -> hold 15-10 / 10 => (0.5 +0.2) / 2 = 0.35
        # 3 prezzo a 13 -> hold 0.3 + 0.5 + 0.2 / 3 = 0.33

        # scendi invece di salire
        # holding_rewards = []
        # 4 prezzo a 8 -> hold 8-10 / 10 = -0.2
        # 5 prezzo a 5 -> hold 5-10/10 = -0.5 -0.2 + 0.3 + 0.5 + 0.2 / 5 = 0.06
        # 6 prezzoa a 15 -> hold

        # (Long, HoldPosition) -> Long
        elif action == Actions.HoldPosition.value and self._position == Positions.Long:
            current_price = self.prices[self._current_tick]
            open_position_price = self.prices[self._open_position_tick]
            price_diff = (current_price - open_position_price ) / open_position_price
            # reward = media dei reward ottenuti dall'hodlding?

            if self._holding_price_difference.size > 1:
                if price_diff / price_diff != self._holding_price_difference[0] / self._holding_price_difference[0]:
                    self._holding_price_difference = np.zeros(0)

            np.append(self._holding_price_difference, price_diff)
            step_reward = np.mean(self._holding_price_difference)

        # (Long, ClosePosition) -> Free
        elif action == Actions.ClosePosition.value and self._position == Positions.Long:
            new_position = self._position.opposite()
            current_price = self.prices[self._current_tick]
            open_position_price = self.prices[self._open_position_tick]
            price_diff = current_price - open_position_price
            # current_price - open_position_price > 0 ==> step_reawrd = 1
            # < 0 ==> step_reward = -1
            # step_reward += price_diff
            if price_diff >= 0:
                step_reward = 1
            else:
                step_reward = -1
            self._holding_price_difference = np.zeros(0)


        self._total_reward += step_reward
        self._position = new_position
        return step_reward

    def _update_profit(self, action):
        '''
                Responsabilità di update_profit():
                    1. Aggiornare self._total_profit con il profit reale
                    2. Aggiornare self._balance con il balance dopo la chiusura della transazione
                :param action:
                :return:
        '''
        temp_balance = self._balance
        if action == Actions.ClosePosition.value and self._position == Positions.Long:
            current_price = self.prices[self._current_tick]
            open_position_price = self.prices[self._open_position_tick]
            self._balance = (self._balance / open_position_price) * current_price
            #se total profit inizializzato a 0
            self._total_profit += self._balance - temp_balance


    def _get_observation(self):
        return self.signal_features[(self._current_tick - self.window_size):self._current_tick]

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def render(self, mode='human'):
        def _plot_position(position, tick):
            color = None
            if position == Positions.Free:
                color = 'red'
            elif position == Positions.Long:
                color = 'green'
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit + ' ~ ' +
            "Total Balance: %.6f" % self._balance
        )

        plt.pause(0.01)

    def render_all(self, mode='human'):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.Free:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.Long:
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

    #############################################################################################################

    def max_possible_profit(self):  # trade fees are ignored
        raise NotImplementedError