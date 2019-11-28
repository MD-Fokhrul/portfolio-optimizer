import argparse
from estimator.model import EstimationModel
import torch.nn as nn
import torch
import torch.optim as optim
from dataset.dataset_loader import DatasetLoader
import util
from model.util import to_tensor, determine_device
import pandas as pd
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='sp500', help='dataset name')
parser.add_argument('--data_dir', type=str, default='data', help='data directory')
parser.add_argument('--test_split_days', type=int, default=152, help='number of days to set as test data')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
parser.add_argument('--limit_days', type=int, help='limit days (steps per episode)')
parser.add_argument('--num_sample_stocks', type=int, help='number of stocks to sample')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--target_size', type=int, default=506, help='number of first k columns for target')
parser.add_argument('--log_interval', type=int, default=20, help='steps interval for print and comet logging')
parser.add_argument('--log_comet', type=util.str2bool, nargs='?', const=True, default=False, help='should log to comet')
parser.add_argument('--comet_tags', nargs='+', default=[], help='tags for comet logging')
parser.add_argument('--force_cpu', type=util.str2bool, nargs='?', const=True, default=False, help='should force cpu even if cuda is available')
parser.add_argument('--checkpoints_interval', type=int, default=10, help='epochs interval for saving model checkpoint')
parser.add_argument('--checkpoints_root_dir', type=str, default='estimation_checkpoints', help='checkpoint root directory')
parser.add_argument('--results_root_dir', type=str, default='estimation_results', help='results root directory')
parser.add_argument('--save_checkpoints', type=util.str2bool, nargs='?', const=True, default=False, help='should save checkpoints?')
parser.add_argument('--load_model', type=str, default=None, help='checkpoint dir path to load from')
parser.add_argument('--modes', nargs='+', default=['train'], help='train and/or test')
args = parser.parse_args()

log_interval = args.log_interval
log_comet = args.log_comet
num_epochs = args.epochs
batch_size = args.batch_size
target_size = args.target_size
learning_rate = args.lr
data_dir = args.data_dir
test_split_days = args.test_split_days
num_sample_stocks = args.num_sample_stocks
dataset_name = args.dataset_name
force_cpu = args.force_cpu
limit_days = args.limit_days
comet_tags = args.comet_tags + [dataset_name]
checkpoints_interval = args.checkpoints_interval
checkpoints_root_dir = args.checkpoints_root_dir
results_root_dir = args.results_root_dir
save_checkpoints = args.save_checkpoints
load_model = args.load_model
modes = args.modes

# cuda/cpu
device_type = determine_device(force_cpu=force_cpu)
device = torch.device(device_type)

experiment = None
start = time.time()

dataloader = DatasetLoader(data_dir, dataset_name)
train_data, test_data, _, _ = dataloader.get_data(limit_days=limit_days,
                                            test_split_days=test_split_days+1,
                                            as_numpy=True,
                                            drop_test=True)

print('train data: {} | test data: {} | batch_size: {}'.format(train_data.shape, test_data.shape, batch_size))


model = EstimationModel(input_size=train_data.shape[1], output_size=target_size)
if load_model:
    model.load(load_model)
if device_type == 'cuda':
    model.cuda()

# SETUP CHECKPOINTS DIR #
if save_checkpoints:
    checkpoints_dir_name = experiment.get_key() if experiment else str(int(start))
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

if 'train' in modes:
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    model.train()
    for epoch in range(num_epochs):
        running_losses = []
        batch_idx = 0
        for i in range(0, train_data.shape[0]-1, batch_size):
            data = to_tensor(train_data[i:i+batch_size], device=device)
            target = to_tensor(train_data[i+1:i+1+batch_size, :target_size], device=device)

            # for last batch off by one errors, otherwise could have just used i+batch_size for data
            data = data[:target.shape[0]]

            optimizer.zero_grad()

            prediction = model(data)

            loss = criterion(prediction, target)

            loss.backward()
            optimizer.step()
            # print statistics
            running_losses.append(loss.item())
            if (batch_idx + 1) % log_interval == 0:
                interval_losses = running_losses[-log_interval:]
                print('[epoch: {}, step:  {}] loss: {:.5f}'.format
                      (epoch + 1, i + 1, sum(interval_losses) / len(interval_losses)))
            batch_idx += 1

        if save_checkpoints and (epoch + 1) % checkpoints_interval == 0:
            model_path = '{}/model_{}.pth'.format(checkpoints_dir, epoch+1)
            torch.save(model.state_dict(), model_path)

        print('[epoch: {}, final] loss: {:.5f}'.format
              (epoch + 1, sum(running_losses) / len(running_losses)))

if 'test' in modes:
    output = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_data[:-1]): # we need to predict t+1 including May 31st and excluding last test day
            data = to_tensor(data, device=device)

            prediction = model(data)
            print('Predicted test day {}/{}'.format(i+1, test_split_days))
            output.append(prediction.cpu().numpy())

    output_path = '{}/results.csv'.format(results_dir)
    pd.DataFrame(output).to_csv(output_path)


