# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from BtcHistoricalData import MarketData
from FinanceEnv import FinanceBtcUsdtEnv
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env = FinanceBtcUsdtEnv()
    obs = env.reset()
    print(obs.size)
    #env.step(1)
    #print(env.observe())
    #print(env.step(1))



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
