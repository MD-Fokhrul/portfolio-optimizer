from collections import defaultdict
from env.portfolio_env import PortfolioEnv
import util
from env.util import plot_portfolio
import imageio


def test(data, agent, log_interval_steps, log_comet, experiment, visualize_portfolio=False):
    # TODO: keep learning during test?
    num_days = data.shape[0]

    env = PortfolioEnv(data)
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
            portfolio_img = plot_portfolio(env.portfolio, env.total_gains, title= 'day-{}'.format(t + 1))
            holdings_imgs.append(portfolio_img)

        # logging
        results['reward'].append(current_reward)
        results['current_volatility'].append(env.current_volatility)
        results['current_gains'].append(env.current_gains)
        if t % log_interval_steps == 0:
            avg_reward = util.avg_results(results, 'reward', lookback=log_interval_steps)
            avg_vol = util.avg_results(results, 'current_volatility', lookback=log_interval_steps)
            avg_gains = util.avg_results(results, 'current_gains', lookback=log_interval_steps)
            total_gains = env.total_gains

            print('Test: step: %d | reward: {:.2f} | avg vol: {:.2f} | avg_step_gains: {:.2f} | total_gains: {:.2f}'
                  .format(t, avg_reward, avg_vol, avg_gains, total_gains))

            env.render()
            if log_comet:
                experiment.log_metric('test_interval_reward', avg_reward, step=t)
                experiment.log_metric('test_interval_avg_vol', avg_vol, step=t)
                experiment.log_metric('test_interval_avg_gains', avg_gains, step=t)
                experiment.log_metric('test_interval_total_gains', total_gains, step=t)

        current_state = next_state

    if visualize_portfolio:
        imageio.mimwrite('test_holdings.gif', holdings_imgs)

    # logging
    avg_reward = util.avg_results(results, 'reward')
    avg_vol = util.avg_results(results, 'current_volatility')
    avg_gains = util.avg_results(results, 'current_gains')
    total_gains = env.total_gains

    print('Test final results - reward: {:.2f} | avg vol: {:.2f} |avg_gains: {:.2f} | total_gains: {:.2f}'
          .format(avg_reward, avg_vol, avg_gains, total_gains))

    if log_comet:

        experiment.log_metric('test_final_avg_reward', avg_reward)
        experiment.log_metric('test_final_avg_vol', avg_vol)
        experiment.log_metric('test_final_avg_gains', avg_gains)
        experiment.log_metric('test_final_total_gains', total_gains)

        if visualize_portfolio:
            experiment.log_image('test_holdings.gif', 'test_holdings')

    env.render()








