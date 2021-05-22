from gym.utils import seeding
import gym
from gym import spaces
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from indicators import Indicators

class Actions(Enum):
    # Mappo sono le azioni legittime nella forma:
    # StatoAzione
    DoNothing = 0
    OpenLongPos = 1
    HoldLongPos = 2
    CloseLongPos = 3
    OpenShortPos = 4
    HoldShortPos = 5
    CloseShortPos = 6

    # Potrei anche mappare solo le quattro azioni e poi dare il reward negativo sull'azione in base alla posizione corrente (che comunque devo tenermi salvata)


class Positions(Enum):
    # La posizione è lo stato attuale in cui si trova l'algoritmo.
    # In Free è permesso solo di fare DoNothing e OpenPosition
    # in Long è permesso di fare solo HoldPosition e ClosePosition
    Free = 0
    Long = 1
    Short = 2

    def switch_position(self, action):

        # Switch per la parte long
        if self == Positions.Free and action == Actions.OpenLongPos.value:
            return Positions.Long
        if self == Positions.Long and action == Actions.CloseLongPos.value:
            return Positions.Free

        # Switch per la parte short
        if self == Positions.Free and action == Actions.OpenShortPos.value:
            return Positions.Short
        if self == Positions.Short and action == Actions.CloseShortPos.value:
            return Positions.Free

class CryptoTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, frame_bound, window_size, indicators=None , position_in_observation : bool = True):
        assert df.ndim == 2
        assert len(frame_bound) == 2
        self.indicators_to_use = indicators
        self.seed()
        self.df = df
        self.frame_bound = frame_bound
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()

        self.position_in_observation = position_in_observation
        if self.position_in_observation:
            self.shape = (window_size + 1, self.signal_features.shape[1])
        else:
            self.shape = (window_size, self.signal_features.shape[1])
        self.invalid_action_replay = True
        self._invalid_action = False
        self._count_invalid_action = 0
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
        self._first_rendering = None
        self.history = None
        #TODO prevedi un flag per attivare questa funzione
        #self._max_profit_possible = self.max_possible_profit()
        self._max_profit_possible = 0
        self._profit_in_step = np.zeros(0)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._invalid_action = False
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
        self._count_invalid_action = 0
        self._profit_in_step = [0] * self.windows_size
        return self._get_observation()

    def step(self, action):
        '''
        Responsabilità di step():
            1. calcolare il profit attraverso la funzione di update_profit
            2. calcolare step_reward, position, open_position_tick tramite la funzione calculate_reward()
            4. aggiornare la history delle posizioni
        '''
        self._invalid_action = False
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        self._update_profit(action)
        # Attenzione! self._position viene cambiata in step_reward quindi update_profit() deve essere chiamato prima
        step_reward = self._calculate_reward(action)

        if self._invalid_action:
            self._current_tick -= 1
        else:
            self._position_history.append(self._position)
        observation = self._get_observation()
        info = dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            balance=self._balance,
            position=self._position.value
        )

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
        step_reward = -10
        new_position = self._position

        # (Free, DoNothing) -> Free
        if action == Actions.DoNothing.value and self._position == Positions.Free:
            step_reward = 0

        # (Free, OpenPosition) -> Long
        elif action == Actions.OpenLongPos.value and self._position == Positions.Free:
            new_position = self._position.switch_position(action)
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
        elif action == Actions.HoldLongPos.value and self._position == Positions.Long:
            current_price = self.prices[self._current_tick]
            open_position_price = self.prices[self._open_position_tick]
            price_diff = (current_price - open_position_price) / open_position_price
            # reward = media dei reward ottenuti dall'hodlding?

            if self._holding_price_difference.size > 1:
                if np.sign(price_diff) != np.sign(self._holding_price_difference[0]):
                    self._holding_price_difference = np.zeros(0)

            self._holding_price_difference = np.append(self._holding_price_difference, price_diff)
            #TODO esplora Nan values
            step_reward = np.mean(self._holding_price_difference)

        # (Long, ClosePosition) -> Free
        elif action == Actions.CloseLongPos.value and self._position == Positions.Long:
            new_position = self._position.switch_position(action)
            current_price = self.prices[self._current_tick]
            open_position_price = self.prices[self._open_position_tick]
            profit = current_price - open_position_price

            if profit >= 0:
                step_reward = 1
            else:
                step_reward = -1
            self._holding_price_difference = np.zeros(0)

        elif action == Actions.OpenShortPos.value and self._position == Positions.Free:
            new_position = self._position.switch_position(action)
            self._open_position_tick = self._current_tick
            step_reward = 0.15

        elif action == Actions.HoldShortPos.value and self._position == Positions.Short:
            current_price = self.prices[self._current_tick]
            open_position_price = self.prices[self._open_position_tick]
            #TODO verifica l'utilità e se fonte di NAN
            price_diff = -1 * (current_price - open_position_price) / open_position_price
            # 1: 10 , 2: 11 , 3: 8
            # -0.1
            # +0.2

            if self._holding_price_difference.size > 1:
                if np.sign(price_diff) != np.sign(self._holding_price_difference[0]):
                    self._holding_price_difference = np.zeros(0)

            self._holding_price_difference = np.append(self._holding_price_difference, price_diff)
            # TODO esplora Nan values
            step_reward = np.mean(self._holding_price_difference)

        elif action == Actions.CloseShortPos.value and self._position == Positions.Short:
            new_position = self._position.switch_position(action)
            current_price = self.prices[self._current_tick]
            open_position_price = self.prices[self._open_position_tick]

            profit = open_position_price - current_price

            if profit >= 0:
                step_reward = 1
            else:
                step_reward = -1
            self._holding_price_difference = np.zeros(0)

        else:
            if self.invalid_action_replay:
                self.invalid_action = True
                self._count_invalid_action += 1

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
        if action == Actions.CloseLongPos.value and self._position == Positions.Long:
            current_price = self.prices[self._current_tick]
            open_position_price = self.prices[self._open_position_tick]
            self._balance = (self._balance / open_position_price) * current_price
            # se total profit inizializzato a 0
            self._total_profit += self._balance - temp_balance

        # 10525 quando compriamo
        # 10200
        # vendo 1 BTC * 10525
        # Sul conto ho 10000
        # se il prezzo va giù la posizione rimane aperta all'infinito
        # viene liquidata se il prezzo SALE
        # 20k 1 BTC a 10525 -> 20k - 10525 = liquidazione
        # 19475 = questo è il prezzo di liquidazione
        # 0.5 BTC
        # compro 1 BTC * 10200

        # Quantità di BTC "venduta" quando è aperta la posizione short
        # balance = USDT + quantità moneta comprata * il prezzo
        # 1 Apro la posizione short: mi entra = 0.5 * prezzo_btc --> 100 * 0.5 = 50
        # 2 Chiudo la posizione short: mi esce = base_asset_quantity * prezzo_btc -->  50.. 0.5 * 200 = 100
        base_asset_quantity = 0.5

        if action == Actions.CloseShortPos.value and self._position == Positions.Short:
            current_price = self.prices[self._current_tick]
            open_position_price = self.prices[self._open_position_tick]
            profit = (base_asset_quantity * open_position_price) - (base_asset_quantity * current_price)
            self._balance += profit
            self._total_profit += profit

        self._profit_in_step = np.append(self._profit_in_step, self._total_profit)

    def _get_observation(self):
        signal_obs = self.signal_features[(self._current_tick - self.window_size) : self._current_tick]
        obs = []

        if self.position_in_observation:
            position_obs = [self._position.value] * self.signal_features.shape[1]

            obs = np.append(signal_obs, position_obs)
        else:
            obs = signal_obs

        return obs

    def _get_observation_price_variation_percentage(self):
        prev_price = None
        signal_obs = np.zeros(0)

        for price in self.prices[self._current_tick, self._current_tick + self.self.window_size]:
            if prev_price:
                price_variation_percentage = (price - prev_price) / prev_price
                signal_obs = np.append(signal_obs,price_variation_percentage)

            prev_price = price

        if self.position_in_observation:
            position_obs = [self._position.value] * self.signal_features.shape[1]

            obs = np.append(signal_obs, position_obs)
        else:
            obs = signal_obs

        return obs

    def render_all(self, mode='human'):
        fig, axs = plt.subplots(2, figsize=(15, 6))
        window_ticks = np.arange(len(self._position_history))

        axs[0].plot(self.prices)
        axs[1].plot(self._profit_in_step)

        free_ticks = []
        long_ticks = []
        short_ticks = []

        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.Free:
                free_ticks.append(tick)
            elif self._position_history[i] == Positions.Long:
                long_ticks.append(tick)
            elif self._position_history[i] == Positions.Short:
                short_ticks.append(tick)

        axs[0].plot(free_ticks, self.prices[free_ticks], 'yo')
        axs[0].plot(long_ticks, self.prices[long_ticks], 'go')
        axs[0].plot(short_ticks, self.prices[short_ticks], 'ro')

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit + ' ~ ' +
            "Balance: %.6f" % self._balance + ' ~ ' +
            "Invalid Action: %.6f" % self._count_invalid_action + ' ~ ' +
            "Max profit: %.6f" % self._max_profit_possible
        )



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

    def max_possible_profit(self):

        price = self.prices[self.window_size:]
        k = int(len(price) / 2)
        n = len(price)

        # Bottom-up DP approach
        profit = [[0 for i in range(k + 1)] for j in range(n)]

        # Profit is zero for the first
        # day and for zero transactions
        # TODO non dovrebbe esser moltiplicato per balance?
        for i in range(1, n):

            for j in range(1, k + 1):
                max_so_far = 0

                for l in range(i):
                    max_so_far = max(max_so_far, price[i] -
                                     price[l] + profit[l][j - 1])

                profit[i][j] = max(profit[i - 1][j], max_so_far)

        return profit[n - 1][k]
