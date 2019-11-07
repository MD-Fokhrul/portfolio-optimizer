# PRELIMINARY IMPORTS #
import util
from collections import defaultdict
# END PRELIMINARY IMPORTS #

# CLI ARG PARSE #
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='sp500', help='dataset name')
parser.add_argument('--data_dir', type=str, default='data', help='data directory')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--init_cash', type=int, default=10000, help='initial cash')
parser.add_argument('--episodes', type=int, default=20, help='number of training episodes')
parser.add_argument('--limit_days', type=int, help='limit days (steps per episode)')
parser.add_argument('--limit_iters', type=int, help='limit total iterations - for debugging')
parser.add_argument('--num_sample_stocks', type=int, help='number of stocks to sample')
parser.add_argument('--discount_factor', type=float, default=0.9, help='ddpg discount factor')
parser.add_argument('--minibatch_size', type=int, default=8, help='ddpg minibatch size')
parser.add_argument('--warmup_iters', type=int, default=10, help='number of ddpg steps to warm up with a random action')
parser.add_argument('--random_process_theta', type=float, default=0.5, help='Random process theta')
parser.add_argument('--log_interval', type=int, default=20, help='steps interval for print and comet logging')
parser.add_argument('--log_comet', type=util.str2bool, nargs='?', const=True, default=False, help='should log to comet')
parser.add_argument('--comet_log_level', type=str, default='episode', help='[interval, episode]')
parser.add_argument('--comet_tags', nargs='+', default=[], help='tags for comet logging')
parser.add_argument('--force_cpu', type=util.str2bool, nargs='?', const=True, default=False, help='should force cpu even if cuda is available')
args = parser.parse_args()
# END CLI ARG PARSE #

# SET VARS #
log_comet = args.log_comet
init_cash = args.init_cash
num_episodes = args.episodes
num_sample_stocks = args.num_sample_stocks
num_warmup_iterations = args.warmup_iters
minibatch_size = args.minibatch_size
learning_rate = args.lr
discount_factor = args.discount_factor
data_dir = args.data_dir
dataset_name = args.dataset_name
random_process_args = {
    'theta': args.random_process_theta
}
force_cpu = args.force_cpu
limit_iterations = args.limit_iters
limit_days = args.limit_days
log_interval_steps = args.log_interval
comet_tags = args.comet_tags + [dataset_name]
comet_log_level = args.comet_log_level
# END SET VARS #

# OPTIONAL COMET DATA LOGGING SETUP #
experiment = None
if log_comet:
    from comet_ml import Experiment
    config = util.load_config()
    experiment = Experiment(api_key=config['comet']['api_key'],
                            project_name=config['comet']['project_name'],
                            workspace=config['comet']['workspace'])
# END OPTIONAL COMET DATA LOGGING SETUP #

# ADDITIONAL IMPORTS # - imports are split because comet_ml requires being imported before torch
from dataset.dataset_loader import DatasetLoader
from model.agent import DDPG
from env.portfolio_env import PortfolioEnv
from model.util import determine_device
# END ADDITIONAL IMPORTS #

# cuda/cpu
device_type = determine_device(force_cpu=force_cpu)

# load data
dataloader = DatasetLoader(data_dir, dataset_name)
data, stocks_plot_fig = dataloader.get_data(num_cols_sample=num_sample_stocks,
                                       limit_days=limit_days,
                                       as_numpy=True,
                                       plot=True)

# save plot of stock prices in selected stock sample and day range
stocks_plot_fig.savefig('stocks_plot.png')

num_days = data.shape[0]
num_stocks = data.shape[1]

params = {
    'init_cash': init_cash,
    'num_episodes': num_episodes,
    'num_warmup_iterations': num_warmup_iterations,
    'minibatch_size': minibatch_size,
    'lr': learning_rate,
    'discount_factor': discount_factor,
    'random_process_theta': random_process_args['theta'],
    'log_interval_steps': log_interval_steps,
    'data_shape': data.shape,
    'num_days': num_days,
    'num_stocks': num_stocks,
    'dataset_name': dataset_name,
    'device_type': device_type
}

print('Running with params: %s' % str(params))

if log_comet:
    experiment.log_parameters(params)
    experiment.log_image('stocks_plot.png', 'stocks')
    experiment.add_tags(comet_tags)

# init custom OpenAI gym env for stocks portfolio
env = PortfolioEnv(data, init_cash)

# init DDPG agent
agent = DDPG(num_stocks, num_stocks, minibatch_size, random_process_args,
             learning_rate=learning_rate, discount_factor=discount_factor,
             device_type=device_type)

# training
total_iterations_counter = 0 # counter for total iterations. num_episodes * num_days
for episode in range(num_episodes):
    agent.reset_action_noise_process()  # init random process for new episode
    current_state = env.reset()  # get initial state s(t)

    results = defaultdict(list) # for logging
    for t in range(num_days - 1):
        if limit_iterations is not None and total_iterations_counter >= limit_iterations:
            # option for hard limit on iterations for debugging
            break

        if total_iterations_counter < num_warmup_iterations:
            # warmup to fill up the buffer with random actions
            current_action = agent.select_random_action()
        else:
            # regular training. Let agent select action based on observation
            current_action = agent.select_action(current_state)

        # execute action on environment, observe new state and reward
        next_state, current_reward, done, _ = env.step(current_action)

        # logging
        results['reward'].append(current_reward)
        results['purchase_power'].append(env.current_purchase_power)
        results['profit'].append(env.current_purchase_power - env.init_cash)
        if t % log_interval_steps == 0:
            avg_reward = util.avg_results(results, 'reward', lookback=log_interval_steps)
            avg_ppwr = util.avg_results(results, 'purchase_power', lookback=log_interval_steps)
            avg_profit = util.avg_results(results, 'profit', lookback=log_interval_steps)

            print('Episode: %d | step: %d | reward: %2f' % (episode, t, avg_reward))
            env.render()
            if log_comet and comet_log_level in ['interval']:
                experiment.log_metric('interval_reward', avg_reward, step=total_iterations_counter)
                experiment.log_metric('interval_ppwr', avg_ppwr, step=total_iterations_counter)
                experiment.log_metric('interval_profit', avg_profit, step=total_iterations_counter)

        # TODO: might need to add episode done states to limit batches not to cross over episodes

        if total_iterations_counter >= num_warmup_iterations:
            # we only want to update the policy after the random state warmup

            # store transition in R (s(t), a(t), r(t), s(t+1))
            agent.append_observation(current_state, current_action, current_reward, next_state)

            # update policy
            critic_loss_val, actor_loss_val = agent.update_policy()

            # logging
            results['critic'].append(critic_loss_val)
            results['actor'].append(actor_loss_val)
            if log_comet and comet_log_level in ['interval']:
                avg_critic_loss = util.avg_results(results, 'critic', lookback=log_interval_steps)
                avg_actor_loss = util.avg_results(results, 'actor', lookback=log_interval_steps)
                experiment.log_metric('interval_critic_loss', avg_critic_loss, step=total_iterations_counter)
                experiment.log_metric('interval_actor_loss', avg_actor_loss, step=total_iterations_counter)

        current_state = next_state
        total_iterations_counter += 1

    if limit_iterations is not None and total_iterations_counter >= limit_iterations:
        # option for hard limit on iterations for debugging
        break

    # logging
    print('Episode: %d final results:' % episode)
    if log_comet and comet_log_level in ['episode', 'interval']:
        avg_reward = util.avg_results(results, 'reward')
        avg_critic_loss = util.avg_results(results, 'critic')
        avg_actor_loss = util.avg_results(results, 'actor')
        experiment.log_metric('avg_episode_reward', avg_reward, step=episode)
        experiment.log_metric('avg_episode_critic_loss', avg_critic_loss, step=episode)
        experiment.log_metric('avg_episode_actor_loss', avg_actor_loss, step=episode)
        experiment.log_metric('max_episode_ppwr', env.max_purchase_power, step=episode)
        experiment.log_metric('max_episode_profit', env.max_purchase_power - env.init_cash, step=episode)
        experiment.log_metric('final_episode_ppwr', env.current_purchase_power, step=episode)
        experiment.log_metric('final_episode_profit', env.current_purchase_power - env.init_cash, step=episode)

    env.render()

# logging
if log_comet:
    experiment.end()
