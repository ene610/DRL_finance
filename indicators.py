import pandas as pd
import numpy as np

class Indicators:
    @staticmethod
    def sma_indicator(df: pd.DataFrame, time_period: int = 21):
        """
        Get The expected indicator in a pandas dataframe.
        Args:
            df(pandas.DataFrame): pandas Dataframe with close values\n
            time_period(int): look back time period to calculate moving average \n
        Returns:
            pandas.DataFrame: new pandas dataframe adding SMA as new column, preserving the columns which already exists\n
        """

        df["SMA"] = df["close"].rolling(window=time_period).mean()
        return df

    @staticmethod
    def macd_indicator(df: pd.DataFrame, short_time_period: int = 12, long_time_period: int = 26, need_signal: bool = True, signal_time_period: int = 9):
        """
        Get The expected indicator in a pandas dataframe.
        Args:
            df(pandas.DataFrame): pandas Dataframe with close values\n
            short_time_period(int): short term look back time period.\n
            long_time_period(int): long term look back time period.\n
            need_signal(bool): if True MACD signal line is added as a new column to the returning pandas dataframe.\n
            signal_time_period(int): look back period to calculate signal line\n
        Returns:
            pandas.DataFrame: new pandas dataframe adding MACD and MACD_signal_line(if required) as new column/s, preserving the columns which already exists\n
        """

        df["LONG_EMA"] = df["close"].ewm(span=long_time_period).mean()
        df["SHORT_EMA"] = df["close"].ewm(span=short_time_period).mean()

        df["MACD"] = df["SHORT_EMA"] - df["LONG_EMA"]
        if need_signal:
            df["MACD_signal_line"] = df["MACD"].ewm(span=signal_time_period).mean()

        df = df.drop(["LONG_EMA", "SHORT_EMA"], axis=1)
        return df

    @staticmethod
    def atr_indicator(df: pd.DataFrame, time_period: int = 14):
        """
        Get The expected indicator in a pandas dataframe.
        Args:
            df(pandas.DataFrame): pandas Dataframe with high, low and close values\n
            time_period(int): look back period to calculate ATR
        Returns:
            pandas.DataFrame: new pandas dataframe adding ATR as a new column, preserving the columns which already exists
        """

        df["close_prev"] = df["close"].shift(1)
        df["TR"] = df[["high", "close_prev"]].max(
            axis=1) - df[["low", "close_prev"]].min(axis=1)
        df["ATR"] = df["TR"].rolling(window=time_period).mean()

        df = df.drop(["close_prev", "TR"], axis=1)
        return df

    @staticmethod
    def ema_indicator(df: pd.DataFrame, time_period: int = 21):
        """
        Get The expected indicator in a pandas dataframe.
        Args:
            df(pandas.DataFrame): pandas Dataframe with close values\n
            time_period(int): look back time period\n
        Returns:
            pandas.DataFrame: new pandas dataframe adding EMA as a new column, preserving the columns which already exists\n
        """
        df["EMA"] = df["close"].ewm(span=time_period).mean()
        return df


    @staticmethod
    def kc_indicator(df: pd.DataFrame, time_period: int = 20, atr_time_period: int = 14, atr_multiplier: int = 2):
        """
        Get The expected indicator in a pandas dataframe.
        Args:
            df(pandas.DataFrame): pandas Dataframe with close values\n
            time_period(int): look back time period to calculate moving average
            atr_time_period(int): time period to calculate average true range
            atr_multiplier(int): constant value which will be multiplied by average true range
        Returns:
            pandas.DataFrame: new pandas dataframe adding kc_upper, kc_middle and kc_lower as three columns, preserving the columns which already exists\n
        """

        df["kc_middle"] = df["close"].ewm(span=time_period).mean()
        df = Indicators.atr_indicator(df, atr_time_period)

        df["kc_upper"] = df["kc_middle"] + atr_multiplier * df["ATR"]
        df["kc_lower"] = df["kc_middle"] - atr_multiplier * df["ATR"]

        df = df.drop(["ATR"], axis=1)

        return df


    @staticmethod
    def rsi_indicator(df, time_period=14):
        """
        Get The expected indicator in a pandas dataframe.
        Args:
            df(pandas.DataFrame): pandas Dataframe with close values\n
            time_period(int): look back time period to calculate moving average \n
        Returns:
            pandas.DataFrame: new pandas dataframe adding RSI as new column, preserving the columns which already exists\n
        """

        df["close_prev"] = df["close"].shift(1)

        df["GAIN"] = 0.0
        df["LOSS"] = 0.0

        df.loc[df["close"] > df["close_prev"],
               "GAIN"] = df["close"] - df["close_prev"]
        df.loc[df["close_prev"] > df["close"],
               "LOSS"] = df["close_prev"] - df["close"]
        df["AVG_GAIN"] = df["GAIN"].ewm(span=time_period).mean()
        df["AVG_LOSS"] = df["LOSS"].ewm(span=time_period).mean()
        df["AVG_GAIN"].iloc[:time_period] = np.nan
        df["AVG_LOSS"].iloc[:time_period] = np.nan
        df["RS"] = df["AVG_GAIN"] / \
                   (df["AVG_LOSS"] + 0.00000001)  # to avoid divide by zero

        df["RSI"] = 100 - ((100 / (1 + df["RS"])))

        df = df.drop(["close_prev", "GAIN", "LOSS",
                      "AVG_GAIN", "AVG_LOSS", "RS"], axis=1)
        return df


    @staticmethod
    def mom_indicator(df: pd.DataFrame, time_period=1):
        """
        MOM -> Momentum
        Momentum helps to determine the price changes from one period to another. \n
        \n Links:
            http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/Momentum.php5\n
        """
        """
                Get The expected indicator in a pandas dataframe.
                Args:
                    df(pandas.DataFrame): pandas Dataframe with close values\n
                    time_period(int): look back time period.\n
                Returns:
                    pandas.DataFrame: new pandas dataframe adding MOM as a new column, 
                    preserving the columns which already exists\n
                """
        df["close_prev"] = df["close"].shift(time_period)
        df["MOM"] = df["close"] - df["close_prev"]
        df.drop(columns="close_prev")
        return df

    @staticmethod
    def vhf_indicator(df: pd.DataFrame, time_period: int = 28):
        """
                Get The expected indicator in a pandas dataframe.
                Args:
                    df(pandas.DataFrame): pandas Dataframe with close values\n
                    time_period(int): look back time period\n
                Returns:
                    pandas.DataFrame: new pandas dataframe adding VHF as new column, preserving the columns which already exists\n
                """
        df["PC"] = df["close"].shift(1)
        df["DIF"] = abs(df["close"] - df["PC"])

        df["HC"] = df["close"].rolling(window=time_period).max()
        df["LC"] = df["close"].rolling(window=time_period).min()

        df["HC-LC"] = abs(df["HC"] - df["LC"])

        df["DIF"] = df["DIF"].rolling(window=time_period).sum()

        df["VHF"] = df["HC-LC"] / df["DIF"]

        df = df.drop(["PC", "DIF", "HC", "LC", "HC-LC"], axis=1)

        return df


    @staticmethod
    def trix_indicator(df: pd.DataFrame, time_period: int = 14):
            """
            Get The expected indicator in a pandas dataframe.
            Args:
                df(pandas.DataFrame): pandas Dataframe with close values\n
                time_period(int): look back time period \n
            Returns:
                pandas.DataFrame: new pandas dataframe adding Trix as new column, preserving the columns which already exists\n
            """

            df["EMA1"] = df["close"].ewm(span=time_period).mean()
            df["EMA2"] = df["EMA1"].ewm(span=time_period).mean()
            df["EMA3"] = df["EMA2"].ewm(span=time_period).mean()
            df["EMA_prev"] = df["EMA3"].shift(1)

            df["TRIX"] = (df["EMA3"] - df["EMA_prev"]) / \
                         df["EMA_prev"] * 100
            df = df.drop(["EMA1", "EMA2", "EMA3", "EMA_prev"], axis=1)
            return df

    @staticmethod
    def rocv_indicator(df: pd.DataFrame, time_period: int = 12):
        """
           ROCV -> Rate of Change Volume
           ROCV indicator is used to identify whether the price movement is confirmed by trading volume.
            Links:
               http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/VolumeRateOfChange.php5
               https://www.investopedia.com/articles/technical/02/091002.asp
           """

        """
        Get The expected indicator in a pandas dataframe.
        Args:
            df(pandas.DataFrame): pandas Dataframe with volume values\n
            time_period(int): look back time period\n
        Returns:
            pandas.DataFrame: new pandas dataframe adding ROCV as new column, preserving the columns which already exists\n
        """
        print(df.columns)
        df["prev_volume"] = df["volume"].shift(time_period)
        df["ROCV"] = (df["volume"] - df["prev_volume"]
                      ) / df["prev_volume"] * 100
        df = df.drop(["prev_volume"], axis=1)


        return df