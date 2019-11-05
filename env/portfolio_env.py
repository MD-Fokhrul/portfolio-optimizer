import gym
import numpy as np
from env.portfolio import Portfolio


class PortfolioEnv(gym.Env):
    def __init__(self, data, init_cash):
        super(PortfolioEnv, self).__init__()

        self.data = data
        self.init_cash = init_cash
        self.max_steps = data.shape[0]
        self.num_shares = data.shape[1]

        # init spaces
        # Actions of the format Buy x%, Sell x%, Hold, etc. Can act 0 to 1 as weight (portion of purchase power)
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.num_shares,), dtype=np.float16)
        # observation space is nx1 share prices where n is number of shares
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.num_shares,), dtype=np.float16)

        self.current_step = None
        self.reward = None
        self.portfolio = None
        self.max_purchase_power = None
        # init portfolio
        self.reset()

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.reward = 0
        self.portfolio = Portfolio(
            init_cash=self.init_cash,
            positions_price=self._init_prices(),
            positions_quantity=self._init_positions()
        )
        self.max_purchase_power = 0

        return self._next_observation()

    def _next_observation(self):
        new_stock_p =self._get_step_prices()
        self.portfolio.update_p(new_stock_p)
        return self.portfolio.curr_weights()

    def step(self, action):
        # Execute one time step within the environment
        prev_purchase_power = self.portfolio.purchase_power()

        self._take_action(action)
        self.current_step += 1
        if self.current_step > self.max_steps:
            self.current_step = 0

        # done = self.portfolio.cash <= 0
        done = (self.current_step + 1) == self.max_steps
        obs = self._next_observation()

        new_purchase_power = self.portfolio.purchase_power()
        self.max_purchase_power = max(self.max_purchase_power, new_purchase_power)
        reward = new_purchase_power - prev_purchase_power

        return obs, reward, done, {}

    def _take_action(self, action):
        self.portfolio.purchase(action)

    def render(self, close=False):
        # Render the environment to the screen
        current_purchase_power = self.portfolio.purchase_power()
        profit = current_purchase_power - self.init_cash
        print('Step: %d' % self.current_step)
        print('Cash: %2f, total equity value: %2f' % (self.portfolio.cash, np.sum(self.portfolio.equity_val())))
        print('Shares held: %d, avg cost for held shares: %2f' % (self.portfolio.shares_held(), self.portfolio.cost_basis()))
        print('Net worth: %2f (Max net worth: %2f)' % (current_purchase_power, self.max_purchase_power))
        print('Profit: %2f' % profit)

    def _get_step_prices(self):
        # return prices for current step
        return self.data[self.current_step]

    def _init_prices(self):
        return self._init_arr(dtype=float)

    def _init_positions(self):
        return self._init_arr(dtype=int)

    def _init_arr(self, dtype):
        return np.zeros((self.num_shares), dtype=dtype)



# TODO:
# - changed since copy:
#       - data_df to np array