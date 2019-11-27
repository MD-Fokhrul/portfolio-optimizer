import gym
import numpy as np
from env.portfolio import Portfolio
from env.util import calculate_volatility


# OpenAI gym wrapper for environment class for portfolio
class PortfolioEnv(gym.Env):
    def __init__(self, data, volatiltiy_lookback=30):
        super(PortfolioEnv, self).__init__()

        # init data
        self.data = data # daily stock prices
        self.volatility_lookback = volatiltiy_lookback # days lookback window for volatility
        self.max_steps = data.shape[0]
        self.num_stocks = data.shape[1]

        # init spaces
        # actions are n sized vectors of weights in the range [0,1]
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.num_stocks,), dtype=np.float16)
        # states are n sized vectors of weights in the range [0,1] as well
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.num_stocks,), dtype=np.float16)

        # init fields
        self.current_step = None
        self.reward = None
        self.portfolio = None
        self.current_gains = None
        self.current_volatility = None
        self.total_gains = None
        # init portfolio
        self.reset()

    # reset environment to inital state i.e. between episodes.
    # returns starting state
    def reset(self):
        self.current_step = 0
        self.reward = 0
        self.portfolio = Portfolio(
            positions_price=self._init_prices(),
            positions_quantity=self._init_positions()
        )
        self.current_gains = 0
        self.total_gains = 0
        self.current_volatility = 0

        return self._next_observation()

    # inner method to return current state
    def _next_observation(self):
        new_stock_p = self._get_step_prices()
        new_volatility = self._get_step_volatility()
        self.portfolio.update_v(new_volatility)
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

        self._take_action(action)
        self.current_step += 1

        done = (self.current_step + 1) == self.max_steps
        obs = self._next_observation()

        self.current_gains = np.sum(self.portfolio.curr_gains())
        self.total_gains += self.current_gains
        self.current_volatility = np.sum(self.portfolio.volatility)

        # calculate reward: new net worth - old net worth
        reward = self.current_gains / self.current_volatility # gain / volatility is our reward

        return obs, reward, done, {}

    def _take_action(self, action):
        self.portfolio.purchase(action)

    def render(self, close=False):
        # Render the environment to the screen
        print('Step: {}'.format(self.current_step))
        print('Total holdings: {:.2f} | cash: {:.2f} | total: {:.2f}'.format(self.portfolio.shares_held(), self.portfolio.cash_held(), np.sum(self.portfolio.stock_w)))
        print('Shares held: {}'.format(self.portfolio.shares_held()))
        print('Current step gains: {:.2f} | volatility: {:.2f}'.format(self.current_gains, self.current_volatility))
        print('Total ongoing gains: {:.2f}'.format(self.total_gains))

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
        return np.zeros(self.num_stocks, dtype=dtype)

    def _get_step_volatility(self):
        return calculate_volatility(self.data, self.current_step, self.volatility_lookback)

