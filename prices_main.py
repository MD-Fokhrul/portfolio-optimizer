import util
import time
import os

# CLI ARG PARSE #
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--force_cpu', type=util.str2bool, nargs='?', const=True, default=False, help='should force cpu even if cuda is available')
parser.add_argument('--data_dir', type=str, default='data', help='data directory')
parser.add_argument('--dataset_name', type=str, default='sp500', help='dataset name')
parser.add_argument('--batch_size', type=int, default=8, help='lstm batch size')
parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--limit_days', type=int, default=None, help='total set days limit')
parser.add_argument('--val_days', type=int, default=None, help='validation set days')
parser.add_argument('--num_sample_stocks', type=int, help='number of stocks to sample')
parser.add_argument('--target_size', type=int, help='target size')
parser.add_argument('--days_lookback_window', type=int, default=30, help='number of days to consider in an "image"')
parser.add_argument('--save_checkpoints', type=util.str2bool, nargs='?', const=True, default=True, help='should save checkpoints')
parser.add_argument('--checkpoints_interval', type=int, default=10, help='episodes interval for saving model checkpoint')
parser.add_argument('--checkpoints_root_dir', type=str, default='prices_checkpoints', help='checkpoint root directory')
parser.add_argument('--results_root_dir', type=str, default='prices_results', help='results directory')
parser.add_argument('--load_model', type=str, default=None, help='checkpoint dir path to load from')
parser.add_argument('--modes', nargs='+', default=['train'], help='train and/or test')
parser.add_argument('--test_predict_days', type=int, default=30, help='how many days to generate for test')
parser.add_argument('--log_interval', type=int, default=20, help='batch interval for print and comet logging')
parser.add_argument('--log_comet', type=util.str2bool, nargs='?', const=True, default=False, help='should log to comet')
parser.add_argument('--log_batches', type=util.str2bool, nargs='?', const=True, default=False, help='should log for batches')
parser.add_argument('--log_epochs', type=util.str2bool, nargs='?', const=True, default=True, help='should log for epochs')
parser.add_argument('--comet_tags', nargs='+', default=[], help='tags for comet logging')
args = parser.parse_args()
# END CLI ARG PARSE #

force_cpu = args.force_cpu
data_dir = args.data_dir
dataset_name = args.dataset_name
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
limit_days = args.limit_days
val_days = args.val_days
days_lookback_window = args.days_lookback_window
num_sample_stocks = args.num_sample_stocks
log_comet = args.log_comet
comet_tags = args.comet_tags
log_interval = args.log_interval
log_batches = args.log_batches
log_epochs = args.log_epochs
save_checkpoints = args.save_checkpoints
checkpoints_interval = args.checkpoints_interval
checkpoints_root_dir = args.checkpoints_root_dir
load_model = args.load_model
modes = args.modes
test_predict_days = args.test_predict_days
results_root_dir = args.results_root_dir
target_size = args.target_size

start = time.time()

# OPTIONAL COMET DATA LOGGING SETUP #
experiment = None

if log_comet:
    from comet_ml import Experiment

    config = util.load_config()
    experiment = Experiment(api_key=config['comet']['api_key'],
                            project_name=config['comet']['project_name'],
                            workspace=config['comet']['workspace'])
# END OPTIONAL COMET DATA LOGGING SETUP #

# SETUP CHECKPOINTS DIR #
if save_checkpoints:
    checkpoints_dir_name = experiment.get_key() if experiment is not None else str(int(start))
    checkpoints_dir = '{}/{}'.format(checkpoints_root_dir, checkpoints_dir_name)
    os.makedirs(checkpoints_dir, exist_ok=True)
else:
    checkpoints_dir = None
# END SETUP CHECKPOINTS DIR #

# SETUP RESULTS DIR #
if 'test' in modes:
    results_dir_name = experiment.get_key() if experiment is not None else str(int(start))

    if 'train' in modes and save_checkpoints:
        results_dir_name = checkpoints_dir_name
    elif load_model:
        results_dir_name = '{}_{}'.format(load_model.split('/')[-2 if load_model[-1] == '/' else -1], results_dir_name)

    results_dir = '{}/{}'.format(results_root_dir, results_dir_name)
    os.makedirs(results_dir, exist_ok=True)
# END SETUP RESULTS DIR #

# SETUP TORCH IMPORTS #  <--- must come after comet_ml import
import torch
from future_prices.models import PricePredictionModel
from future_prices.lstm_dataloader import FuturePricesLoader
from future_prices.train import train
from future_prices.test import test
# END SETUP TORCH IMPORTS #

# SETUP DEVICE #
device_type = 'cuda' if not force_cpu and torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)
print('device is: ', device)
# END SETUP DEVICE #

# SETUP DATALOADERS #

val_data_dim = None
train_data_dim = None
test_data_dim = None
train_loader = None
validation_loader = None
test_loader = None

data_loader_config = util.load_config('future_prices/lstm_config.json')

if 'train' in modes:
    train_loader = FuturePricesLoader(data_loader_config, 'train', batch_size, data_dir, dataset_name,
                                          days_lookback_window, num_sample_stocks,
                                          target_size=target_size,
                                          limit_days=limit_days,
                                          exclude_days=val_days)

    train_data_dim = train_loader.data_dim

    if val_days and val_days > 0:
        validation_loader = FuturePricesLoader(data_loader_config, 'validation', batch_size, data_dir, dataset_name,
                                              days_lookback_window, num_sample_stocks,
                                              target_size=target_size,
                                              limit_days=val_days)
        val_data_dim = validation_loader.data_dim


if 'test' in modes:
    test_loader = FuturePricesLoader(data_loader_config, 'test', batch_size, data_dir, dataset_name,
                                          days_lookback_window, num_sample_stocks,
                                          # +1 for window offset, +2 for temporal frequency offset (see indices in dataloader)
                                          target_size=target_size,
                                          limit_days=days_lookback_window)

    test_data_dim = test_loader.data_dim


output_size = train_loader.futureprices.target_size


validation_dataloader = None
# END SETUP DATALOADERS #

params = {
    'lr': lr,
    'batch_size': batch_size,
    'epochs': epochs,
    'log_interval': log_interval,
    'device': device_type,
    'train_data_shape': train_data_dim,
    'validation_data_shape': val_data_dim,
    'test_data_shape': test_data_dim,
    'test_predict_days': test_predict_days,
    'target_size': target_size
}

print('running with params: {}'.format(params))

if log_comet:
    comet_tags.append(dataset_name)

    experiment.log_parameters(params)
    experiment.add_tags(comet_tags)

# SETUP MODEL #
assert (train_data_dim or test_data_dim)
input_size = train_data_dim[1] if train_data_dim else test_data_dim[1]

model = PricePredictionModel(input_size=input_size,
                             output_size=output_size,
                             hidden_size=int(input_size / 4))

if load_model is not None:
    model.load(load_model)

if device_type == 'cuda':
    model.cuda()
# END SETUP MODEL #


if 'train' in modes:
    print('--Started training--')
    train(model, lr, train_loader, validation_loader, epochs, device,
          save_checkpoints, checkpoints_dir, checkpoints_interval,
          log_interval, log_batches, log_epochs, log_comet, experiment)
    print('--Finished training--')

if 'test' in modes:
    print('--Started testing--')
    test(model, test_predict_days, test_loader, device, results_dir)
    print('--Finished testing--')

