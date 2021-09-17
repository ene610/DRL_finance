from gym.utils import seeding
import gym
from gym import spaces
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import pandas as pd
import math
from indicators import Indicators

pd.options.mode.chained_assignment = None
import quantstats as qs


class Actions(Enum):
    # Mappo sono le azioni legittime nella forma:
    # StatoAzione
    DoNothing = 0
    OpenLongPos = 1
    OpenShortPos = 2

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
        if self == Positions.Long and action == Actions.OpenShortPos.value:
            return Positions.Free

        # Switch per la parte short
        if self == Positions.Free and action == Actions.OpenShortPos.value:
            return Positions.Short
        if self == Positions.Short and action == Actions.OpenLongPos.value:
            return Positions.Free


class CryptoTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, frame_bound, window_size: int = 22,
                 indicators=['diff_pct_1', 'diff_pct_5', 'diff_pct_15', 'diff_pct_22'],
                 position_in_observation: bool = True, reward_option="sharpe"):
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
            self.shape = (window_size, self.signal_features.shape[1] + 1)
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
        # TODO prevedi un flag per attivare questa funzione
        # self._max_profit_possible = self.max_possible_profit()
        self._max_profit_possible = 0
        self._profit_in_step = np.zeros(0)
        self.reward_option = reward_option

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
        self._position_history = ([Positions.Free.value] * (self.window_size - 1)) + [self._position.value]
        self._total_reward = 0.
        self._total_profit = 0.  # unit
        self._first_rendering = True
        self.history = {}
        self._holding_price_difference = np.zeros(0)
        self._count_invalid_action = 0
        self._profit_in_step = [0] * self.window_size
        self.balance_array = [10000]

        self.returns_balance = dict()
        self.returns_balance[self._current_tick] = 10000
        obs = self._get_observation()
        #init obs

        return obs

    # change in sharpe_ratio_reward
    def sharpe_calculator(self, current_balance, rf=0):
        key_list = list(self.returns_balance.keys())
        returns_list = list(self.returns_balance.values())[key_list.index(self._open_position_tick):]
        roi = (returns_list[-1] - returns_list[0]) / returns_list[0] * 100
        std = np.array(returns_list).std()
        sharpe_ratio = (roi - rf) / std
        if math.isnan(sharpe_ratio):
            sharpe_ratio = 0
        return sharpe_ratio

    def sharpe_calculator_total(self, rf=0):

        returns_list = list(self.returns_balance.values())
        roi = (returns_list[-1] - returns_list[0]) / returns_list[0] * 100
        std = np.array(returns_list).std()
        sharpe_ratio = (roi - rf) / std
        return sharpe_ratio

    def sharpe_calculator_total_quantstats(self):
        returns_list = list(self.returns_balance.values())
        pd_returns_list = pd.DataFrame(returns_list)
        sharpe = qs.stats.sharpe(pd_returns_list, rf=0., periods=252, annualize=False, trading_year_days=252)[0]
        if math.isnan(sharpe):
            sharpe = 0
        return sharpe

    def sharpe_calculator_quantstats(self):
        key_list = list(self.returns_balance.keys())
        returns_list = list(self.returns_balance.values())[key_list.index(self._open_position_tick):]
        pd_returns_list = pd.DataFrame(returns_list)
        sharpe = qs.stats.sharpe(pd_returns_list, rf=0., periods=252, annualize=False, trading_year_days=252)[0]
        if math.isnan(sharpe):
            if ((returns_list[0] - returns_list[-1]) > 0):
                sharpe = 0.1
            else:
                sharpe = -0.1
        return sharpe

    def sortino_calculator_quantstats(self):
        key_list = list(self.returns_balance.keys())
        returns_list = list(self.returns_balance.values())[key_list.index(self._open_position_tick):]
        pd_returns_list = pd.DataFrame(returns_list)
        sortino = qs.stats.sortino(pd_returns_list, rf=0., periods=252, annualize=False, trading_year_days=252)[0]

        if math.isnan(sortino) or math.isinf(sortino):
            if((returns_list[0] - returns_list[-1]) > 0):
                sortino = 0.1
            else:
                sortino = -0.1

        return sortino / math.sqrt(2)

    def sortino_calculator_total_quantstats(self):
        returns_list = list(self.returns_balance.values())
        pd_returns_list = pd.DataFrame(returns_list)
        sortino = qs.stats.sortino(pd_returns_list, rf=0., periods=252, annualize=False, trading_year_days=252)[0]
        if math.isnan(sortino):
            sortino = 0
        return sortino / math.sqrt(2)

    def profit_calculator(self):
        if self.trade_type == "HoldingLong":
            return self.profit_holding_long()

        elif self.trade_type == "HoldingShort":
            return self.profit_holding_short()

        elif self.trade_type == "CloseLong":
            return self.profit_close_long()

        elif self.trade_type == "CloseShort":
            return self.profit_close_short()
        else:
            return 0

    def profit_holding_long(self):
        current_price = self.prices[self._current_tick]
        open_position_price = self.prices[self._open_position_tick]
        price_diff = (current_price - open_position_price) / open_position_price
        # reward = media dei reward ottenuti dall'holding?

        if self._holding_price_difference.size > 1:
            if np.sign(price_diff) != np.sign(self._holding_price_difference[0]):
                self._holding_price_difference = np.zeros(0)

        self._holding_price_difference = np.append(self._holding_price_difference, price_diff)
        # TODO esplora Nan values
        step_reward = np.mean(self._holding_price_difference)
        return step_reward

    def profit_holding_short(self):
        current_price = self.prices[self._current_tick]
        open_position_price = self.prices[self._open_position_tick]
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

        return step_reward

    def profit_close_long(self):

        current_price = self.prices[self._current_tick]
        open_position_price = self.prices[self._open_position_tick]
        profit = current_price - open_position_price

        if profit >= 0:
            step_reward = 1
        else:
            step_reward = -1

        return step_reward

    def profit_close_short(self):

        current_price = self.prices[self._current_tick]
        open_position_price = self.prices[self._open_position_tick]

        profit = open_position_price - current_price

        if profit >= 0:
            step_reward = 1
        else:
            step_reward = -1

        return step_reward

    def step(self, action):
        '''
        Responsabilità di step():
            1. calcolare il profit attraverso la funzione di update_profit
            2. calcolare step_reward, position, open_position_tick tramite la funzione calculate_reward()
            4. aggiornare la history delle posizioni
        '''
        self._invalid_action = False
        self._done = False

        self._update_profit(action)
        # Attenzione! self._position viene cambiata in step_reward quindi update_profit() deve essere chiamato prima
        step_reward = self._calculate_reward(action)

        info = dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            balance=self._balance,
            position=self._position.value
        )

        if not self._invalid_action:
            self._current_tick += 1
            self._position_history.append(self._position.value)
            self._profit_in_step = np.append(self._profit_in_step, self._total_profit)

        if self._current_tick == self._end_tick:
            self._done = True

        observation = self._get_observation()

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
        # reward option
        if self.reward_option == "sharpe":
            reward_function = self.sharpe_calculator_quantstats

        elif self.reward_option == "sortino":
            reward_function = self.sortino_calculator_quantstats

        else:
            reward_function = self.profit_calculator

        # è negativo quando fa un'azione invalida?
        step_reward = 0
        new_position = self._position

        # (Free, DoNothing) -> Free
        if action == Actions.DoNothing.value and self._position == Positions.Free:
            step_reward = 0
            # reinserisce l'ultima posizione
            # self.returns_balance = {self._current_tick : 10000}
            last_key = list(self.returns_balance.keys())[-1]
            self.returns_balance[self._current_tick] = self.returns_balance[last_key]
            # self.balance_array.append(self.balance_array[-1])


        # (Free, OpenPosition) -> Long
        elif action == Actions.OpenLongPos.value and self._position == Positions.Free:
            new_position = self._position.switch_position(action)
            self._open_position_tick = self._current_tick
            # 1 è troppo alto? Ma se metto a 0.95 fa un trade ogni ora
            # così ne fa uno ogni due tre minuti

            last_key = list(self.returns_balance.keys())[-1]
            self.returns_balance[self._current_tick] = self.returns_balance[last_key]

        # (Long, HoldPosition) -> Long
        elif (action == Actions.OpenLongPos.value or action == Actions.DoNothing.value) \
                and self._position == Positions.Long:

            self.trade_type = "HoldingLong"


            current_price = self.prices[self._current_tick]
            open_position_price = self.prices[self._open_position_tick]

            # inserisce il balance provvisorio ottenuto come
            # balance all'acquisto / prezzo di acquisto * prezzo corrente

            holding_in_long = self.returns_balance[self._open_position_tick] / open_position_price * current_price
            self.returns_balance[self._current_tick] = holding_in_long
            # step_reward = self.sharpe_calculator(holding_in_long)
            step_reward = reward_function()

            # (Long, ClosePosition) -> Free
        elif action == Actions.OpenShortPos.value and self._position == Positions.Long:

            self.trade_type = "CloseLong"


            new_position = self._position.switch_position(action)
            current_price = self.prices[self._current_tick]
            open_position_price = self.prices[self._open_position_tick]

            self._holding_price_difference = np.zeros(0)
            close_in_long = self.returns_balance[self._open_position_tick] / open_position_price * current_price
            self.returns_balance[self._current_tick] = close_in_long

            # step_reward = self.sharpe_calculator(close_in_long)
            step_reward = reward_function()

        # (Short, Free) -> Short
        elif action == Actions.OpenShortPos.value and self._position == Positions.Free:
            new_position = self._position.switch_position(action)
            self._open_position_tick = self._current_tick
            step_reward = 0
            # reinserisce l'ultima posizione
            last_key = list(self.returns_balance.keys())[-1]
            self.returns_balance[self._current_tick] = self.returns_balance[last_key]


        elif (action == Actions.OpenShortPos.value or action == Actions.DoNothing.value)\
                and self._position == Positions.Short:

            current_price = self.prices[self._current_tick]
            open_position_price = self.prices[self._open_position_tick]


            self.trade_type = "HoldingShort"

            short_pay = self.returns_balance[self._open_position_tick] / open_position_price * current_price
            holding_in_short = self.returns_balance[self._open_position_tick] * 2 - short_pay

            self.returns_balance[self._current_tick] = holding_in_short
            # step_reward = self.sharpe_calculator(holding_in_short)
            step_reward = reward_function()


        elif action == Actions.OpenLongPos.value and self._position == Positions.Short:

            new_position = self._position.switch_position(action)
            current_price = self.prices[self._current_tick]
            open_position_price = self.prices[self._open_position_tick]

            self.trade_type = "CloseShort"

            short_pay = self.returns_balance[self._open_position_tick] / open_position_price * current_price
            close_short = self.returns_balance[self._open_position_tick] * 2 - short_pay

            # self.balance_array.append(short_holding)
            self.returns_balance[self._current_tick] = close_short

            # step_reward = self.sharpe_calculator(close_short)
            step_reward = reward_function()

        else:
            if self.invalid_action_replay:
                self._invalid_action = True
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
        if action == Actions.OpenShortPos.value and self._position == Positions.Long:
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

        if action == Actions.OpenLongPos.value and self._position == Positions.Short:
            current_price = self.prices[self._current_tick]
            open_position_price = self.prices[self._open_position_tick]

            # sono i soldi guadagnati o persi ricomprando le monete che hai venduto
            transaction_short = (self._balance / open_position_price) * current_price
            profit = self._balance - transaction_short
            self._balance += profit
            self._total_profit += profit

    def _get_observation(self):

        signal_obs = self.signal_features[(self._current_tick - self.window_size): self._current_tick]

        if self.position_in_observation:
            # 1. Prendere le ultime window_size righe di history_position
            last_window_size_positions_history = np.array(self._position_history[-self.window_size:])

            # 2. Trasformarle in una colonna
            last_window_size_positions_history = last_window_size_positions_history[:, np.newaxis]

            # 3. Aggiungerle a obs
            obs = np.c_[signal_obs, last_window_size_positions_history]

        else:
            obs = signal_obs

        return obs

    def render_all(self, episode = 0, savepath = None):

        fig, axs = plt.subplots(2, figsize=(15, 6))
        window_ticks = np.arange(len(self._position_history))
        axs[0].plot(self.prices)

        returns = list(self.returns_balance.values())
        returns = [ret - 10000 for ret in returns]
        returns = [0] * self.window_size + returns
        axs[1].plot(returns)
        axs[1].plot(self._profit_in_step)

        free_ticks = []
        long_ticks = []
        short_ticks = []

        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.Free.value:
                free_ticks.append(tick)
            elif self._position_history[i] == Positions.Long.value:
                long_ticks.append(tick)
            elif self._position_history[i] == Positions.Short.value:
                short_ticks.append(tick)

        axs[0].plot(free_ticks, self.prices[free_ticks], 'yo')
        axs[0].plot(long_ticks, self.prices[long_ticks], 'go')
        axs[0].plot(short_ticks, self.prices[short_ticks], 'ro')

        sharpe_ratio = self.sharpe_calculator_total_quantstats()
        sortino_ratio = self.sortino_calculator_total_quantstats()

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit + ' ~ ' +
            f"Episode: {episode}" + ' ~ ' +
            f"Sortino: {sortino_ratio}"  + ' ~ ' +
            f"Sharpe Ratio: {sharpe_ratio}")

        if savepath != None:
            plt.savefig(f"{savepath}/{episode}")
            plt.close()



    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def _process_data(self):

        prices = self.df['close']
        prices = prices[self.frame_bound[0] - self.window_size: self.frame_bound[1]]

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
        signal_features['diff_pct_1'] = ((self.df['close'] / np.roll(self.df['close'], shift=(int(1)))) * 100) - 100
        signal_features['diff_pct_5'] = ((self.df['close'] / np.roll(self.df['close'], shift=(int(5)))) * 100) - 100
        signal_features['diff_pct_15'] = ((self.df['close'] / np.roll(self.df['close'], shift=(int(15)))) * 100) - 100
        signal_features['diff_pct_22'] = ((self.df['close'] / np.roll(self.df['close'], shift=(int(22)))) * 100) - 100

        # signal_features = signal_features.drop(columns=['open', 'high', 'low', 'close', 'volume', 'tradecount'])
        # signal_features = signal_features.dropna()
        signal_features = signal_features[self.frame_bound[0] - self.window_size: self.frame_bound[1]]
        signal_features.reset_index(drop=True, inplace=True)

        signal_features = self.filter_indicators(signal_features)
        signal_features = signal_features.round(decimals=3)

        np_signal_feature = signal_features.to_numpy()
        np_prices = prices.to_numpy()

        return np_prices, np_signal_feature

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
                        :return filtered_signal_feature: Dataframe contenente gli indicatori selezionati in init
        '''
        # self.indicators_to_use = ['SMA', 'MACD', 'MACD_signal_line', 'EMA', 'kc_middle', 'kc_upper', 'kc_lower', 'RSI', 'close_prev', 'MOM', 'VHF', 'TRIX', 'ROCV']
        # self.indicators_to_use = ['MACD', 'MACD_signal_line', 'MOM', 'VHF', 'TRIX']
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

