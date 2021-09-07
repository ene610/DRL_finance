import pandas as pd

df = pd.read_csv("data/Binance_BTCUSDT_minute.csv", skiprows=1)

df = df.rename(columns={'Volume USDT': 'volume'})
df = df.iloc[::-1]
df = df.drop(columns=['symbol', f"Volume BTC"])
df['date'] = pd.to_datetime(df['unix'], unit='ms')
df = df.set_index('date')
#print(df)

data_index = pd.date_range(start='2020-09-11 20:40:00', end="2021-04-30 00:35:00", freq='min')
#print(data_index)

df_reindexed = df
df_reindexed = df_reindexed.reindex(data_index)
#print(df_reindexed)

pd.set_option('display.max_rows', 1000)

print(df_reindexed[df_reindexed['close'].isna()])