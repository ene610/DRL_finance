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
        if self == Positions.Free and action == Actions.OpenLongPos:
            return Positions.Long
        if self == Positions.Long and action == Actions.CloseLongPos:
            return Positions.Free
        # Switch per la parte short
        if self == Positions.Free and action == Actions.OpenShortPos:
            return Positions.Short
        if self == Positions.Short and action == Actions.CloseShortPos:
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
        self._max_profit_possible = self.maxProfit()

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
        step_reward = -10
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
            price_diff = (current_price - open_position_price) / open_position_price
            # reward = media dei reward ottenuti dall'hodlding?

            if self._holding_price_difference.size > 1:
                if price_diff / price_diff != self._holding_price_difference[0] / self._holding_price_difference[0]:
                    self._holding_price_difference = np.zeros(0)

            self._holding_price_difference = np.append(self._holding_price_difference, price_diff)
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
        else:
            if self.invalid_action_stop:
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
        ########### QUI ##############
        quanto_ho_comprato_BTC =

        if action == Actions.CloseShortPos.value and self._position == Positions.Short:
            current_price = self.prices[self._current_tick]
            open_position_price = self.prices[self._open_position_tick]
            self._balance = 0.5 * current_price
            self._total_profit += open_position_price - current_price

    def _get_observation(self):
        signal_obs = self.signal_features[(self._current_tick - self.window_size):self._current_tick]
        obs = []

        if self.position_in_observation:
            position_obs = [self._position.value] * self.signal_features.shape[1]

            obs = np.append(signal_obs, position_obs)
        else:
            obs = signal_obs

        return obs
