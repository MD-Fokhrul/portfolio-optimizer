from collections import defaultdict
from env.portfolio_env import PortfolioEnv
import util
from env.util import plot_portfolio
import imageio


def test(data, agent, init_cash, log_interval_steps, log_comet, experiment, visualize_portfolio=False):
    num_days = data.shape[0]

    env = PortfolioEnv(data, init_cash)
    agent.is_training = False

    current_state = env.reset()
    results = defaultdict(list)  # for logging

    if visualize_portfolio:
        holdings_imgs = []

    for t in range(num_days - 1):
        # regular training. Let agent select action based on observation
        current_action = agent.select_action(current_state)

        # execute action on environment, observe new state and reward
        next_state, current_reward, done, _ = env.step(current_action)

        if visualize_portfolio:
            portfolio_img = plot_portfolio(env.portfolio, title='day-{}'.format(t + 1))
            holdings_imgs.append(portfolio_img)

        # logging
        results['reward'].append(current_reward)
        results['purchase_power'].append(env.current_purchase_power)
        results['profit'].append(env.current_purchase_power - env.init_cash)
        if t % log_interval_steps == 0:
            avg_reward = util.avg_results(results, 'reward', lookback=log_interval_steps)
            avg_ppwr = util.avg_results(results, 'purchase_power', lookback=log_interval_steps)
            avg_profit = util.avg_results(results, 'profit', lookback=log_interval_steps)

            print('Test: step: %d | reward: %2f' % (t, avg_reward))
            env.render()
            if log_comet:
                experiment.log_metric('test_interval_reward', avg_reward, step=t)
                experiment.log_metric('test_interval_ppwr', avg_ppwr, step=t)
                experiment.log_metric('test_interval_profit', avg_profit, step=t)

        current_state = next_state

    if visualize_portfolio:
        imageio.mimwrite('test_holdings.gif', holdings_imgs)

    print('Test: final results:')
    if log_comet:
        avg_reward = util.avg_results(results, 'reward')

        experiment.log_metric('test_avg_episode_reward', avg_reward)
        experiment.log_metric('test_max_episode_ppwr', env.max_purchase_power)
        experiment.log_metric('test_max_episode_profit', env.max_purchase_power - env.init_cash)
        experiment.log_metric('test_final_episode_ppwr', env.current_purchase_power)
        experiment.log_metric('test_final_episode_profit', env.current_purchase_power - env.init_cash)

        if visualize_portfolio:
            experiment.log_image('test_holdings.gif', 'test_holdings')

    env.render()








