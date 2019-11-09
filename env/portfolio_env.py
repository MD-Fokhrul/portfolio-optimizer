import gym
import numpy as np
from env.portfolio import Portfolio


# OpenAI gym wrapper for environment class for portfolio
class PortfolioEnv(gym.Env):
    def __init__(self, data, init_cash):
        super(PortfolioEnv, self).__init__()

        # init data
        self.data = data # daily stock prices
        self.init_cash = init_cash # cash we start with
        self.max_steps = data.shape[0]
        self.num_shares = data.shape[1]

        # init spaces
        # actions are n sized vectors of weights in the range [0,1]
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.num_shares,), dtype=np.float16)
        # states are n sized vectors of weights in the range [0,1] as well
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.num_shares,), dtype=np.float16)

        # init fields
        self.current_step = None
        self.reward = None
        self.portfolio = None
        self.current_purchase_power = None
        self.max_purchase_power = None
        # init portfolio
        self.reset()

    # reset environment to inital state i.e. between episodes.
    # returns starting state
    def reset(self):
        self.current_step = 0
        self.reward = 0
        self.portfolio = Portfolio(
            init_cash=self.init_cash,
            positions_price=self._init_prices(),
            positions_quantity=self._init_positions()
        )
        self.current_purchase_power = 0
        self.max_purchase_power = 0

        return self._next_observation()

    # inner method to return current state
    def _next_observation(self):
        new_stock_p = self._get_step_prices()
        self.portfolio.update_p(new_stock_p)
        return self.portfolio.curr_weights()

    # step and take an action.
    # input: action - an n sized vector of weights
    # output: obs - new resulting state after action taken
    # output: reward - reward for action
    # output: done - did we finish all the steps
    # output: empty info - not needed
    def step(self, action):
        # Execute one time step within the environment
        prev_purchase_power = self.portfolio.purchase_power()

        self._take_action(action)
        self.current_step += 1

        # TODO: do we also need this: done = self.portfolio.cash <= 0?
        done = (self.current_step + 1) == self.max_steps
        obs = self._next_observation()

        # calculate reward: new net worth - old net worth
        new_purchase_power = self.portfolio.purchase_power()
        self.max_purchase_power = max(self.max_purchase_power, new_purchase_power)
        self.current_purchase_power = new_purchase_power
        reward = new_purchase_power - prev_purchase_power

        return obs, reward, done, {}

    def _take_action(self, action):
        self.portfolio.purchase(action)

    def render(self, close=False):
        # Render the environment to the screen
        current_purchase_power = self.portfolio.purchase_power()
        profit = current_purchase_power - self.init_cash
        max_profit = self.max_purchase_power - self.init_cash
        print('Step: %d' % self.current_step)
        print('Cash: %2f, total equity value: %2f' % (self.portfolio.cash, np.sum(self.portfolio.equity_val())))
        print('Shares held: %d, avg cost for held shares: %2f' % (self.portfolio.shares_held(), self.portfolio.cost_basis()))
        print('Net worth: %2f (Max net worth: %2f)' % (current_purchase_power, self.max_purchase_power))
        print('Profit: %2f (Max profit: %2f)' % (profit, max_profit))

    # return stock prices for current step
    def _get_step_prices(self):
        return self.data[self.current_step]

    # initialize price vector
    def _init_prices(self):
        return self._init_arr(dtype=float)

    # initialize quantity vector
    def _init_positions(self):
        return self._init_arr(dtype=int)

    # initialize np array of type dtype
    def _init_arr(self, dtype):
        return np.zeros((self.num_shares), dtype=dtype)

