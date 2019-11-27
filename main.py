# PRELIMINARY IMPORTS #
import util
import time
from collections import defaultdict
# END PRELIMINARY IMPORTS #

# CLI ARG PARSE #
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='sp500', help='dataset name')
parser.add_argument('--data_dir', type=str, default='data', help='data directory')
parser.add_argument('--test_split_days', type=int, default=152, help='number of days to set as test data')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
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
parser.add_argument('--visualize_portfolio', type=util.str2bool, nargs='?', const=True, default=True, help='should create portfolio visualization gif?')
parser.add_argument('--checkpoints_interval', type=int, default=50, help='episodes interval for saving model checkpoint')
parser.add_argument('--checkpoints_root_dir', type=str, default='checkpoints', help='checkpoint root directory')
parser.add_argument('--results_root_dir', type=str, default='results', help='results root directory')
parser.add_argument('--save_checkpoints', type=util.str2bool, nargs='?', const=True, default=False, help='should save checkpoints?')
parser.add_argument('--load_model', type=str, default=None, help='checkpoint dir path to load from')
parser.add_argument('--modes', nargs='+', default=['train'], help='train and/or test')
parser.add_argument('--plot_stocks', type=util.str2bool, nargs='?', const=True, default=False, help='should plot stocks?')
args = parser.parse_args()
# END CLI ARG PARSE #

# SET VARS #
log_comet = args.log_comet
num_episodes = args.episodes
num_sample_stocks = args.num_sample_stocks
num_warmup_iterations = args.warmup_iters
minibatch_size = args.minibatch_size
learning_rate = args.lr
discount_factor = args.discount_factor
data_dir = args.data_dir
test_split_days = args.test_split_days
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
visualize_portfolio = args.visualize_portfolio
checkpoints_interval = args.checkpoints_interval
checkpoints_root_dir = args.checkpoints_root_dir
results_root_dir = args.results_root_dir
save_checkpoints = args.save_checkpoints
load_model = args.load_model
modes = args.modes
plot_stocks = args.plot_stocks
# END SET VARS #

if len(modes) == 0 or len([x for x in modes if x not in ['train', 'test']]):
    print('please provide train or test modes')
    exit(1)

# OPTIONAL COMET DATA LOGGING SETUP #
experiment = None
if log_comet:
    from comet_ml import Experiment
    config = util.load_config()
    experiment = Experiment(api_key=config['comet']['api_key'],
                            project_name=config['comet']['project_name'],
                            workspace=config['comet']['workspace'])
# END OPTIONAL COMET DATA LOGGING SETUP #

dir_name = experiment.get_key() if experiment is not None else str(int(time.time()))

checkpoints_dir = None
if save_checkpoints:
    checkpoints_dir = '{}/{}'.format(checkpoints_root_dir, dir_name)

if 'test' in modes:
    results_dir = '{}/{}'.format(results_root_dir, dir_name)

# ADDITIONAL IMPORTS # - imports are split because comet_ml requires being imported before torch
from dataset.dataset_loader import DatasetLoader
from model.agent import DDPG
from model.util import determine_device
from train import train
from test import test
# END ADDITIONAL IMPORTS #

# cuda/cpu
device_type = determine_device(force_cpu=force_cpu)

# load data
dataloader = DatasetLoader(data_dir, dataset_name)
train_data_df, test_data_df, train_stocks_plot_fig, test_stocks_plot_fig = dataloader.get_data(
                                       num_cols_sample=num_sample_stocks,
                                       limit_days=limit_days,
                                       test_split_days=test_split_days,
                                       as_numpy=False,
                                       plot=plot_stocks)

if plot_stocks: # works with absolute stock prices not percent change
    # save plot of stock prices in selected stock sample and day range
    train_stocks_plot_fig.savefig('train_stocks_plot.png')
    if test_stocks_plot_fig is not None:
        test_stocks_plot_fig.savefig('test_stocks_plot.png')

params = {
    'num_episodes': num_episodes,
    'num_warmup_iterations': num_warmup_iterations,
    'minibatch_size': minibatch_size,
    'lr': learning_rate,
    'discount_factor': discount_factor,
    'random_process_theta': random_process_args['theta'],
    'log_interval_steps': log_interval_steps,
    'train_data_shape': train_data_df.shape,
    'test_data_shape': test_data_df.shape,
    'dataset_name': dataset_name,
    'device_type': device_type
}

print('Running with params: %s' % str(params))

if log_comet:
    experiment.log_parameters(params)
    experiment.add_tags(comet_tags)
    if plot_stocks:
        experiment.log_image('train_stocks_plot.png', 'train_window_stocks')
        if test_stocks_plot_fig is not None:
            experiment.log_image('test_stocks_plot.png', 'test_window_stocks')

num_stocks = train_data_df.shape[1]
num_states_and_actions = num_stocks

# init DDPG agent
agent = DDPG(num_states_and_actions, num_states_and_actions, minibatch_size, random_process_args,
             learning_rate=learning_rate, discount_factor=discount_factor,
             device_type=device_type, is_training=True)

if load_model is not None:
    agent.load_model(load_model)

if 'train' in modes:
    train(train_data_df, agent, num_episodes, limit_iterations, num_warmup_iterations,
          log_interval_steps, log_comet, comet_log_level, experiment, checkpoints_interval, checkpoints_dir, save_checkpoints)

if 'test' in modes:
    # test(test_data, agent, log_interval_steps, log_comet, experiment, visualize_portfolio=visualize_portfolio)

    # we still want to train
    train(test_data_df, agent, 1, limit_iterations, num_warmup_iterations,
          log_interval_steps, log_comet, comet_log_level, experiment, checkpoints_interval, checkpoints_dir,
          save_checkpoints=False, is_test=True, results_dir=results_dir)

# logging
if log_comet:
    experiment.end()
