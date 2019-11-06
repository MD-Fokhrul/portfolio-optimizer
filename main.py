import util

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
parser.add_argument('--num_warmup_iterations', type=int, default=10, help='number of ddpg steps to warm up with a random action')
parser.add_argument('--random_process_theta', type=float, default=0.5, help='Random process theta')
parser.add_argument('--log_interval', type=float, default=20, help='steps interval for print and comet logging')
parser.add_argument('--log_comet', type=util.str2bool, nargs='?', const=True, default=False, help='should log to comet')
parser.add_argument('--force_cpu', type=util.str2bool, nargs='?', const=True, default=False, help='should force cpu even if cuda is available')
args = parser.parse_args()

log_comet = args.log_comet

init_cash = args.init_cash
num_episodes = args.episodes
num_sample_stocks = args.num_sample_stocks
num_warmup_iterations = args.num_warmup_iterations
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

if log_comet:
    from comet_ml import Experiment
    config = util.load_config()
    experiment = Experiment(api_key=config['comet']['api_key'],
                            project_name=config['comet']['project_name'],
                            workspace=config['comet']['workspace'])

from dataset.dataset_loader import DatasetLoader
from model.agent import DDPG
from env.portfolio_env import PortfolioEnv
from model.util import determine_device

device_type = determine_device(force_cpu=force_cpu)

log_interval_steps = args.log_interval
dataloader = DatasetLoader(data_dir, dataset_name)
data = dataloader.get_data(num_cols_sample=num_sample_stocks, limit_days=limit_days)
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
    'dataset_name': dataset_name
}

print('Running with params: %s' % str(params))

if log_comet:
    experiment.log_parameters(params)

env = PortfolioEnv(data, init_cash)
agent = DDPG(num_stocks, num_stocks, minibatch_size, random_process_args,
             learning_rate=learning_rate, discount_factor=discount_factor,
             device_type=device_type)

total_iterations_counter = 0
for episode in range(num_episodes):
    agent.reset_action_noise_process()  # init random process for new episode
    current_state = env.reset()  # get initial state s(t)
    rewards = []
    for t in range(num_days - 1):
        if limit_iterations is not None and total_iterations_counter >= limit_iterations:
            break

        if total_iterations_counter < num_warmup_iterations:
            # warmup to fill up the buffer with random actions
            current_action = agent.select_random_action()
        else:
            current_action = agent.select_action(current_state)

        next_state, current_reward, done, _ = env.step(current_action)  # TODO: execute action to get r(t) and s(t+1)

        rewards.append(current_reward)

        if t % log_interval_steps == 0:
            avg_reward = sum(rewards[-log_interval_steps:]) / len(rewards[-log_interval_steps:])

            print('Episode: %d | step: %d | reward: %2f' % (episode, t, avg_reward))
            env.render()
            if log_comet:
                experiment.log_metric('avg_step_interval_reward', current_reward, step=total_iterations_counter)
                purchase_power = env.portfolio.purchase_power()
                experiment.log_metric('purchase_power', purchase_power, step=total_iterations_counter)
                experiment.log_metric('profit', purchase_power - init_cash, step=total_iterations_counter)

        # TODO: might need to add episode done states to limit batches not to cross over episodes

        agent.append_observation(current_state, current_action, current_reward, next_state)  # store transition in R (s(t), a(t), r(t), s(t+1))
        agent.update_policy()

        current_state = next_state
        total_iterations_counter += 1

    if limit_iterations is not None and total_iterations_counter >= limit_iterations:
        break

    print('Episode: %d final results:' % episode)
    if log_comet:
        avg_episode_reward = sum(rewards) / len(rewards)
        experiment.log_metric('avg_episode_reward', avg_episode_reward, step=episode)
        experiment.log_metric('max_episode_purchase_power', env.max_purchase_power, step=episode)
    env.render()

if log_comet:
    experiment.end()
