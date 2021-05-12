from enum import Enum
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt

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
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._done = None
        self._current_tick = None
        self._open_position_tick = None
        self._last_trade_tick = None
        self._position = Positions.Free
        self._position_history = None
        self._total_reward = 0.
        self._total_profit = 1.
        self._first_rendering = None
        self.history = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._open_position_tick = 0
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Free
        self._position_history = (self.window_size * [0]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}
        return self._get_observation()

    def step(self, action):
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        valid, step_reward = self._calculate_reward(action)
        self._total_reward += step_reward
        if valid:
          self._update_profit(action)
        else:
          # Se l'azione non è valida lo costringo ad imparare che non è valida, gli do un reward negativo e fermo l'episodio
          self._total_reward = -1000
          self._done = True

        if action == Actions.ClosePosition.value and self._position == Positions.Long:
            self._position = self._position.opposite()

        if action == Actions.OpenPosition.value and self._position == Positions.Free:
            self._position = self._position.opposite()
            self._open_position_tick = self._current_tick

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = dict(
            total_reward = self._total_reward,
            total_profit = self._total_profit,
            position = self._position.value
        )
        self._update_history(info)

        return observation, step_reward, self._done, info

    def _action_is_invalid(self, action):
        if self._position == Positions.Free.value and (action == Actions.ClosePosition.value or action == Actions.HoldPosition):
            return True
        if self._position == Positions.Long.value and (action == Actions.OpenPosition or action == Actions.DoNothing):
            return True
        return False

    def _calculate_reward(self, action):
        step_reward = 0

        # Controllo di avere una azione valida
        # Se è una azione invalida, il reward tornato sarà settato negativo dal metodo che richiama questo
        if self._action_is_invalid(action):
            return False, -1

        # Da qui in poi ho solo azioni valide:

        if action == Actions.HoldPosition and self._position == Positions.Long:
            current_price = self.prices[self._current_tick]
            open_position_price = self.prices[self._open_position_tick]
            price_diff = current_price - open_position_price
            # step_reward = price_diff / current_price
            step_reward += price_diff

        trade = False
        if ((action == Actions.OpenPosition.value and self._position == Positions.Free) or
            (action == Actions.ClosePosition.value and self._position == Positions.Long)):
            trade = True

        if trade:
            if action == Actions.OpenPosition.value:
                # Lo incentivo ad acquistare
                step_reward = 10
            elif action == Actions.ClosePosition.value:
                # Qui ci dovrebbe entrare per forza se è entrato nel primo if ma non nel secondo
                # Qui sta chiudendo la posizione Long!
                current_price = self.prices[self._current_tick]
                last_trade_price = self.prices[self._open_position_tick]
                price_diff = current_price - last_trade_price
                step_reward = price_diff

        return True, step_reward

    def _update_profit(self, action):
        if action == Actions.ClosePosition.value and self._position == Positions.Long:
            current_price = self.prices[self._current_tick]
            open_position_price = self.prices[self._open_position_tick]
            self._total_profit += current_price - open_position_price

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
                color = 'gray'
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
            "Total Profit: %.6f" % self._total_profit
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
            "Total Profit: %.6f" % self._total_profit
        )

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def _process_data(self):
        prices = self.df['close'].to_numpy()

        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        return prices, signal_features

    #############################################################################################################

    def max_possible_profit(self):  # trade fees are ignored
        raise NotImplementedError