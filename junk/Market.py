import pandas
import pandas as pd
import numpy as np
from indicators import Indicators


class MarketData():
    # per ora utilizzato BTC/USDT

    def __init__(self):
        # open_position = [quantity_base, quantity_quote, price]
        # self.open_position = [-1, -1, -1]

        # TUTTE le candele
        self.obs_size = 28
        self.oclhv = self.init_upload()

        #print(self.oclhv.loc[1599858480000])

        self.oclhv_indicators = Indicators.sma_indicator(self.oclhv)
        self.oclhv_indicators = Indicators.macd_indicator(self.oclhv_indicators)
        self.oclhv_indicators = Indicators.atr_indicator(self.oclhv_indicators)
        self.oclhv_indicators = Indicators.ema_indicator(self.oclhv_indicators)
        self.oclhv_indicators = Indicators.kc_indicator(self.oclhv_indicators)

        #SettingWithCopyWarning
        #decomenta per risolvere questo errore (https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas)
        pd.options.mode.chained_assignment = None
        self.oclhv_indicators = Indicators.rsi_indicator(self.oclhv_indicators)
        self.oclhv_indicators = Indicators.mom_indicator(self.oclhv_indicators)
        self.oclhv_indicators = Indicators.vhf_indicator(self.oclhv_indicators)
        self.oclhv_indicators = Indicators.trix_indicator(self.oclhv_indicators)
        self.oclhv_indicators = Indicators.rocv_indicator(self.oclhv_indicators)
        self.close = self.oclhv_indicators['close']
        self.oclhv_indicators = self.oclhv_indicators.drop(columns=['high', 'close', 'low', 'volume', 'open'])
        self.oclhv_indicators = self.oclhv_indicators.dropna()

        #print(self.oclhv_indicators.head(5))

        # Solo le candele interessanti per l'osservazione
        # self.market = self.oclhv_indicators[:28]
        self.market = pandas.DataFrame.empty
        # Se vendessi adesso avresti questa percentuale di guadagno
        # self.indicators['if_sell'] = -1
        # numero di contanti disponibili ma potrebbe tranquillamente esser la percentuale di monete in suo possesso

        #self.index = 1599858480000
        self.index = 0
        self.countdown = 60
        self.invalid_action = False
        self.last_action_taken = 0
        self.open_position = False
        self.double_open_position = False
        self.open_position_price = 0
        self.holding_rewards = np.empty(0)

    def action(self, action):
        # action : 0 = hold , 1 = buy , 2 = sell tutto
        # controlla azione valida con flag

        self.last_action_taken = action
        if action == 0:
            pass

        if action == 1:
            if self.double_open_position == False:
                if self.open_position == False:
                    #print(self.close.index)
                    self.open_position_price = self.close.iloc[self.index]
                    self.open_position = True
                else:
                    self.double_open_position = True

        if action == 2:
            # if not self.already_sold and self.already_bought:
            #     self.balance = self.market.iloc[19]['close'] * self.open_position[0]
            #     self.already_sold = True
            pass

        # self.steps_rewards.append(imemdiate_reward)
        #self.index += 60000
        self.index += 1
        self.countdown -= 1

    def observe(self):
        #TODO index cambioto
        # max 28
        #print(self.market.shape)
        #self.market = self.oclhv_indicators.iloc[self.index: (self.obs_size * 60000) + self.index]
        self.market = self.oclhv_indicators.iloc[self.index:self.index + self.obs_size]
        #print(self.market.shape)

        flatten_indicators = self.market.to_numpy().flatten()
        return np.append(flatten_indicators, int(self.open_position))

    def view(self):
        print("-" * 10)
        print(self.market)
        print("-" * 10)

    def is_done(self):
        # pensavo di procedere aspettando il tempo di chiusura di un azione e forzarlo o dare un parziale dopo tot step
        if self.countdown == 0:
            return True
        return self.invalid_action

    def init_upload(self):
        # C:\Users\Ene\PycharmProjects\pythonProject\Binance_BTCUSDT_minute.csv
        path = "Binance_BTCUSDT_minute.csv"
        df = pd.read_csv(path, skiprows=1)
        df = df.rename(columns={'Volume USDT': 'volume'})
        df = df.iloc[::-1]
        df = df.set_index("unix")
        df = df.drop(columns=['date', 'symbol', 'Volume BTC'])

        # df['pct_5'] = df['close'].rolling(int(5))/df['close']
        # print(np.roll(df['close'], shift=-(int(5))) / df['close'])
        # df['R5'] = np.roll(df['close'], shift=(int(5))) / df['close']

        # diff_close_pct_5 = differenza percentuale tra il close alla candela attuale e quello della candela di 5 timestamps fa?
        # df['diff_close_pct_5'] = ((df['close'] / np.roll(df['close'], shift=(int(5)))) * 100) - 100
        # df['diff_close_pct_10'] = ((df['close'] / np.roll(df['close'], shift=(int(10)))) * 100) - 100
        # df['diff_close_pct_15'] = ((df['close'] / np.roll(df['close'], shift=(int(15)))) * 100) - 100
        # Delete the first 15 rows
        df = df.iloc[self.obs_size:]
        # return df

        return df

    def evaluete(self):
        # sharpe-ratio = (return of portfolio - risk-free rate ) / standard deviation of the portfolio’s excess return

        # 100% è il migliore reward
        # non ha ancora comprato = 0
        # sta cercando di vendere senza aver comprato = reward minore del totale migliore -150%
        # compra = piccolo reward +-q
        # holda = profitto se hai comrpato +-1.5, sennò 0
        # vendere = profitto totale 100% -100%

        #immediate_reward = self.open_position_price - self.market.iloc[-1, :]['close']

        immediate_reward = self.open_position_price - self.close.iloc[self.index]

        # countdown 2
        # countdown 1
        # countdown 0 action = 1
        # caso compra

        if self.last_action_taken == 1 and self.countdown > 0:
            # non si può comprare se ho gia una transizione
            if self.double_open_position:
                self.invalid_action = True
                return -100
            else:
                self.open_position = True
                return immediate_reward

        # Se stai cercando di comprare all'ultimo step dell'episodio sei punito
        if self.last_action_taken == 1 and self.countdown == 0:
            return -100

        # Se non hai fatto trade nell'episodio sei punito
        if self.countdown == 0 and self.holding_rewards.size == 0:
            return -100

        if self.last_action_taken == 2 or self.countdown == 0:
            if not self.open_position:
                self.invalid_action = True
                reward = -100
            else:
                self.open_position = False
                self.double_open_position = False
                self.holding_rewards = self.holding_rewards[:-1]
                reward = np.sum(self.holding_rewards)
            self.done = True
            return reward

        if self.last_action_taken == 0:
            # se non ho una posizione aperta
            if not self.open_position:
                return 0
            else:
                # calcola il reward
                self.holding_rewards = np.append(self.holding_rewards, immediate_reward)
                return np.mean(self.holding_rewards)

    def reset(self):
        if self.index >= (self.oclhv.shape[0] - 60):
            self.index = 0
        self.countdown = 60
        self.invalid_action = False
        self.last_action_taken = 0
        self.open_position = False
        self.double_open_position = False
        self.open_position_price = 0
        self.holding_rewards = np.empty(0)
