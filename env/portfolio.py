import numpy as np
import random


class Portfolio:
    def __init__(self,
                 # BASICS: INPUTS ANY PORTFOLIO NEEDS
                 positions_price,
                 positions_weights):

        self.stock_p = positions_price
        self.stock_w = positions_weights
        self.volatility = np.ones(positions_price.shape)

    def curr_gains(self):
        return self.stock_w * self.stock_p

    # weights' sum should be in [0,1] range
    def weight_constraints(self):
        return {'low': 0, 'high': 1.0001}

    # total quantity of shares held
    def shares_held(self):
        return np.sum(self.stock_w[:-1])

    def cash_held(self):
        return self.stock_w[-1]

    # update stock prices
    def update_p(self, stock_p):
        self.stock_p = stock_p

    # update volatility prices
    def update_v(self, volatility):
        self.volatility = volatility

    # this changes the portfolio holdings based on newly introduced weights
    def purchase(self, weights):
        constraints = self.weight_constraints()

        # if 1 < np.sum(weights) < 1.001:
        #     weights -= 0.001

        if not constraints['low'] <= np.sum(weights) <= constraints['high']:
            # sanity check that weights sum to 1. Should not happen since we normalize
            print(weights)
            print(np.sum(weights))
            raise Exception("This allocation is beyond the range permitted")

        self.stock_w = weights






