import pandas as pd
import numpy as np
class GodAgent():

    def __init__(self, prices: np.ndarray = None, fees=0.02, slippage=0.1, balance=0):

        #dataframe
        # candele
        # dove stanno i minimi locali e i massimi locali
        # giu su
        self._prices = prices.flatten()
        self._max_profit = self.maxProfit(k=len(self._prices)/2, n=len(self._prices))
        print("Massimo profitto: ", self._max_profit)





    # def local_mins():
        # si calcola i timestep dei minimi locali
        # Return lista dei minimi locali

    # def local_maxs():
        # si calcola i timestep dei massimi locali
        # Return lista dei massimi locali

    #knapsack???
    # min_max_profit = [{min1, max}, {min2, max}, {min, max}, {min, max}, {min, max5}]
    # for i in min_max_profit :

        # min_max = [-prezzo_del_min (giu su), +prezzo_del_max(su giu), -prezzo_del_min (giu su)]
        # balance / prezzo_del_min * prezzo_del_max (- fees + fslippage) > 0?

    # def run():
        # (i)
        # se timestep == min_loc -> compra
        # se timestep == max loc -> vendi
        # se il profit  del trade (timestep_min, timestep_max) > 0 allora fai il trade
        # (balance / open_position_price ) - (balance / open_position_price * (fees + slippage) == base_asset_quantity che hai comprati
        #
    # k = max trading
    # n = numero di timestep
    def maxProfit(self, n, k):

        # Bottom-up DP approach
        profit = [[0 for i in range(k + 1)]
                  for j in range(n)]

        # Profit is zero for the first
        # day and for zero transactions
        for i in range(1, n):

            for j in range(1, k + 1):
                max_so_far = 0

                for l in range(i):
                    max_so_far = max(max_so_far, self._prices[i] -
                                     self._prices[l] + profit[l][j - 1])

                profit[i][j] = max(profit[i - 1][j], max_so_far)

        return profit[n - 1][k]
