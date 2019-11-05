import numpy as np


class Portfolio:
    def __init__(self,
                 # BASICS: INPUTS ANY PORTFOLIO NEEDS
                 init_cash,
                 positions_price,
                 positions_quantity,

                 # other stuff
                 minimum_init_margin = .5, minimum_req_margin = .25, risk_param_leverage = 1,
                 weight_norm_epsilon=0.1
                 ):

        self.cash = init_cash
        self.stock_p = positions_price
        self.stock_q = positions_quantity
        self.weight_norm_epsilon = weight_norm_epsilon # this makes sure we don't hit floating point equality errors for constraints (1.0 == 1.00...02 and thus violates constraints)

        ##self.param_init_margin = np.array(minimum_init_margin) #model parameter -- the required minimum %initial margin for transaction
        #self.param_req_margin = np.array(minimum_req_margin) #model parameter -- the minimum %required margin for maintenance
        #self.param_leverage_penalty = np.array(risk_param_leverage) #model parameter -- risk apetite for leverage

    """for finding the current weight allocation of the portfolio w.r.t purchase power"""
    def curr_weights(self):
        purchase_power = self.purchase_power()

        if purchase_power == 0:
            return 0

        return self.equity_val() / purchase_power

    def weight_constraints(self):
        return {'low': 0, 'high': 1}

    """for getting the current purchase power value; consider all your net worth minus prior obligations"""
    def purchase_power(self):
        return self.cash + np.sum(self.equity_val())

    """for getting the equity value of the portfolio"""
    def equity_val(self):
        # equity value is price*quantity per stock.
        return self.stock_p * self.stock_q

    def shares_held(self):
        return sum(self.stock_q)

    def cost_basis(self):
        shares_held = self.shares_held()
        if shares_held == 0:
            return 0
        return sum(self.equity_val()) / shares_held

    """for updating p in the database; will also automatically liquidate faulty margins"""
    def update_p(self, stock_p):
        self.stock_p = stock_p

    """this is 'a' in our RL model -- take a new w* for t+1 as argument, and adjust accordingly"""

    def purchase(self, weights):
        constraints = self.weight_constraints()
        orig_weights = weights
        weights = weights / (np.sum(weights) + self.weight_norm_epsilon)
        if not constraints['low'] <= np.sum(weights) <= constraints['high']:
            print(orig_weights)
            print(np.sum(orig_weights))
            print(weights)
            print(np.sum(weights))
            raise Exception("This allocation is beyond the range permitted")

        """First, check for feasibility of the transaction. THIS ERROR SHOULD NOT OCCUR!"""
        curr_weights = self.curr_weights()
        delta_weights = weights - curr_weights
        pp = max(0, self.purchase_power())

        """5 possible cases (of which 4 are written here directly, 1 is nested in self.buy()):"""
        """Sell, Sell short, Buy, Buy to cover a short position, buy on margin -- adjusted here"""
        for i in range(len(weights)):
            new_w = weights[i]
            dw = delta_weights[i]
            curr_w = curr_weights[i]

            if dw == 0:
                continue
            elif dw < 0:  # implies a selling action
                if curr_w == 0 or new_w <= 0:  # check the case of possible short selling
                    # hold?
                    continue

                # cash_change += self._sell(dw, i, pp)  # this is regular sell that will happen either way
                self._sell(dw, i, pp)
            elif dw > 0:  # implies a buying action
                # cash_change -= self._buy(dw, i, pp)
                self._buy(dw, i, pp)

        # self.cash = cash_change + self.cash  # temp: have we bought more than we can in terms of cash?

        # TODO: check we don't exceed cash limit? for now assert
        assert self.cash >= 0

    def _buy(self, delta_weight, stock_idx, pp):
        cash_adjustment = 0
        delta_quantity = self._quantity_from_weight(delta_weight, pp, stock_idx)

        """Regular purchase transaction"""
        cash_adjustment += delta_quantity * self.stock_p[stock_idx]
        if self.cash - cash_adjustment < 0:
            return 0
        else:
            self.stock_q[stock_idx] += delta_quantity
            self.cash -= cash_adjustment
            return cash_adjustment

    def _sell(self, delta_weight, stock_idx, pp):
        cash_adjustment = 0
        delta_quantity = abs(self._quantity_from_weight(delta_weight, pp, stock_idx))

        """First adjust the debt that might exist; then adjust cash."""
        cash_adjustment += delta_quantity * self.stock_p[stock_idx]

        self.stock_q[stock_idx] -= delta_quantity  # same goes in here. the portfolio holds quantity between [0,+inf)
        self.cash += cash_adjustment
        return cash_adjustment

    def _quantity_from_weight(self, weight, purchase_power, stock_idx):
        return int(weight * purchase_power / self.stock_p[stock_idx]) # TODO: maybe should be floor/ceiling?






