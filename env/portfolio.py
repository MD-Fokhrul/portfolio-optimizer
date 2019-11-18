import numpy as np
import random


class Portfolio:
    def __init__(self,
                 # BASICS: INPUTS ANY PORTFOLIO NEEDS
                 init_cash,
                 positions_price,
                 positions_quantity,
                 weight_norm_epsilon=0.1
                 ):

        self.init_cash = init_cash
        self.cash = init_cash
        self.stock_p = positions_price
        self.stock_q = positions_quantity
        self.weight_norm_epsilon = weight_norm_epsilon # this makes sure we don't hit floating point equality errors for constraints (1.0 == 1.00...02 and thus violates constraints)

    # get the current weights of the portfolio's holdings.
    # a stock's weight is the holding's portion of the total net worth: (stock price * quantity held) / net worth
    def curr_weights(self):
        purchase_power = self.purchase_power()

        if purchase_power == 0:
            return 0

        return self.equity_val() / purchase_power

    # weights' sum should be in [0,1] range
    def weight_constraints(self):
        return {'low': 0, 'high': 1}

    # portfolio total net worth: value of equity + cash
    def purchase_power(self):
        return np.sum(self.equity_val())

    # value of equity in protfolio: stock prices * number of stocks held
    def equity_val(self):
        # equity value is price*quantity per stock.
        return np.append(self.stock_p * self.stock_q, self.cash)

    # total quantity of shares held
    def shares_held(self):
        return sum(self.stock_q)

    # avg stock price
    def cost_basis(self):
        shares_held = self.shares_held()
        if shares_held == 0:
            return 0
        return sum(self.equity_val()) / shares_held

    # update stock prices
    def update_p(self, stock_p):
        self.stock_p = stock_p

    # this changes the portfolio holdings based on newly introduced weights
    def purchase(self, weights):
        constraints = self.weight_constraints()
        orig_weights = weights

        # weights should already be normalized.
        # but we normalize them with an epsilon to ensure no floating point rounding errors occur
        weights = weights / (np.sum(weights) + self.weight_norm_epsilon)

        if not constraints['low'] <= np.sum(weights) <= constraints['high']:
            # sanity check that weights sum to 1. Should not happen since we normalize
            print(orig_weights)
            print(np.sum(orig_weights))
            print(weights)
            print(np.sum(weights))
            raise Exception("This allocation is beyond the range permitted")

        # exclude cash weight since we want to update actual cash as we buy/sell
        weights = weights[:-1]
        curr_weights = self.curr_weights()[:-1]

        delta_weights = weights - curr_weights
        pp = max(0, self.purchase_power())

        # assuming first stocks iterated over take precedence until we run out of money
        # we want to first sell and only then buy with newly liquid cash
        # also shuffling stock order every time so we don't get stuck on buying/selling the same stocks each time
        stock_indexes = list(range(len(weights)))
        random.shuffle(stock_indexes)

        for i in stock_indexes:
            # 3 options right now: buy, sell or hold (plan to add sell short, margin buy)
            curr_w = curr_weights[i]  # current state
            new_w = weights[i] # new desired state
            dw = delta_weights[i] # difference in weights to determine quantity to buy/sell

            if dw == 0:
                # no change in stock weight - hold
                continue
            elif dw < 0:
                # negative change in stock weight - sell
                if curr_w == 0 or new_w < 0:  # check the case of possible short selling
                    # hold
                    continue

                # cash_change += self._sell(dw, i, pp)  # this is regular sell that will happen either way
                self._sell(dw, i, pp)
            elif dw > 0:
                # positive change in stock weight - buy
                # cash_change -= self._buy(dw, i, pp)
                self._buy(dw, i, pp)

        assert self.cash >= 0 # sanity check. Should not happen

    # buy action. can change held share quantity
    # input: delta_weight - change in stock weight to determine quantity to buy
    # input: stock_idx - index of stock to buy in arrays
    # input: purchase power - total purchase power
    def _buy(self, delta_weight, stock_idx, pp):
        # get actual quantity to buy from desired change in stock weight.
        # this can be 0 if weight change is too small
        delta_quantity = self._quantity_from_weight(delta_weight, pp, stock_idx)

        # determine cost of buying this amount of the stock
        cash_adjustment = delta_quantity * self.stock_p[stock_idx]

        if self.cash - cash_adjustment < 0:
            # if we can't afford this purchase we hold
            return 0
        else:
            self.stock_q[stock_idx] += delta_quantity # update stock quantity in portfolio after purchase
            self.cash -= cash_adjustment # update cash amount after purchase
            return cash_adjustment

    def _sell(self, delta_weight, stock_idx, pp):
        # get actual quantity to buy from desired change in stock weight.
        # this can be 0 if weight change is too small
        delta_quantity = self._quantity_from_weight(delta_weight, pp, stock_idx)

        # sanity check to make sure we don't sell more than we have
        # TODO: remove when introduce short sell
        delta_quantity = min(delta_quantity, self.stock_q[stock_idx])

        # determine cost of selling this amount of the stock
        cash_adjustment = delta_quantity * self.stock_p[stock_idx]

        self.stock_q[stock_idx] -= delta_quantity # update stock quantity in portfolio after sale
        self.cash += cash_adjustment # update cash amount after sale
        return cash_adjustment

    def _quantity_from_weight(self, weight, purchase_power, stock_idx):
        # get stock quantity based on a weight which represents a proportion of the portfolio value.
        return abs(int(weight * purchase_power / self.stock_p[stock_idx])) # TODO: maybe should be floor/ceiling?






