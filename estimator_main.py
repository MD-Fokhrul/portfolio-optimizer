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

###
#
# This NN regressor is used for two parts of the ensemble:
# 1. Learn real price change from predicted price change and predicted market variables
# 2. Learn optimal w* from DDPG predicted w* and conic optimizer predicted w*
# In both cases the two inputs need to be concatenated on the wide axis before using this script.
###

# CLI ARG PARSE #
parser = argparse.ArgumentParser()
parser.add_argument('--input_dataset_name', type=str, help='predicted input dataset name')
parser.add_argument('--target_dataset_name', type=str, help='target ground truth dataset name')
parser.add_argument('--data_dir', type=str, default='data', help='data directory')
parser.add_argument('--normalize', type=util.str2bool, nargs='?', const=True, default=False, help='should normalize data?')
parser.add_argument('--test_split_days', type=int, default=152, help='number of days to set as test data')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
parser.add_argument('--limit_days', type=int, help='limit days (steps per episode)')
parser.add_argument('--num_sample_stocks', type=int, help='number of stocks to sample')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--target_size', type=int, default=506, help='number of first k columns for target')
parser.add_argument('--log_interval', type=int, default=20, help='steps interval for printing')
parser.add_argument('--force_cpu', type=util.str2bool, nargs='?', const=True, default=False, help='should force cpu even if cuda is available')
parser.add_argument('--checkpoints_interval', type=int, default=10, help='epochs interval for saving model checkpoint')
parser.add_argument('--checkpoints_root_dir', type=str, default='estimation_checkpoints', help='checkpoint root directory')
parser.add_argument('--results_root_dir', type=str, default='estimation_results', help='results root directory')
parser.add_argument('--save_checkpoints', type=util.str2bool, nargs='?', const=True, default=False, help='should save checkpoints?')
parser.add_argument('--load_model', type=str, default=None, help='checkpoint dir path to load from')
parser.add_argument('--modes', nargs='+', default=['train'], help='train and/or test')
args = parser.parse_args()
# END CLI ARG PARSE #

# SET VARS #
log_interval = args.log_interval
num_epochs = args.epochs
batch_size = args.batch_size
target_size = args.target_size
learning_rate = args.lr
data_dir = args.data_dir
normalize = args.normalize
test_split_days = args.test_split_days
num_sample_stocks = args.num_sample_stocks
input_dataset_name = args.input_dataset_name
target_dataset_name = args.target_dataset_name
force_cpu = args.force_cpu
limit_days = args.limit_days
checkpoints_interval = args.checkpoints_interval
checkpoints_root_dir = args.checkpoints_root_dir
results_root_dir = args.results_root_dir
save_checkpoints = args.save_checkpoints
load_model = args.load_model
modes = args.modes
# END SET VARS #

# cuda/cpu
device_type = determine_device(force_cpu=force_cpu)
device = torch.device(device_type)

experiment = None
start = time.time()

# load data

# training input data. We exclude the predicted test days for training
input_dataloader = DatasetLoader(data_dir, input_dataset_name)
train_data, _, _, _ = input_dataloader.get_data(limit_days=limit_days+test_split_days,
                                                exclude_days=test_split_days,
                                                as_numpy=True,
                                                normalize=normalize)

# training target labels (should not have test days)
target_dataloader = DatasetLoader(data_dir, target_dataset_name)
train_labels, _, _, _ = target_dataloader.get_data(limit_days=limit_days,
                                                   as_numpy=True,
                                                   normalize=normalize)

# test input data. Does not have corresponding labels
test_dataloader = DatasetLoader(data_dir, input_dataset_name)
test_data_df, _, _, _ = test_dataloader.get_data(limit_days=test_split_days,
                                                 as_numpy=False,
                                                 normalize=normalize)

print('train data: {} | batch_size: {}'.format(train_data.shape, batch_size))


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
results_dir_name = experiment.get_key() if experiment is not None else str(int(start))

if 'train' in modes and save_checkpoints:
    results_dir_name = checkpoints_dir_name
elif load_model:
    results_dir_name = '{}_{}'.format(load_model.split('/')[-2 if load_model[-1] == '/' else -1], results_dir_name)

results_dir = '{}/{}'.format(results_root_dir, results_dir_name)
os.makedirs(results_dir, exist_ok=True)
# END SETUP RESULTS DIR #

## TRAINING ##
if 'train' in modes:
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    model.train()
    for epoch in range(num_epochs):
        running_losses = []
        batch_idx = 0
        for i in range(0, train_data.shape[0], batch_size):
            data = to_tensor(train_data[i:i+batch_size], device=device)
            target = to_tensor(train_labels[i:i+batch_size, :target_size], device=device)

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
## END TRAINING ##

## TESTING ##
if 'test' in modes:
    test_data = test_data_df.to_numpy()
    columns = test_data_df.columns[:target_size]

    # setup output file
    output = []
    output_path = '{}/results.csv'.format(results_dir)
    pd.DataFrame([], columns=columns).to_csv(output_path)

    output_interval = 5
    last_output_index = 0

    model.eval()
    with torch.no_grad():
        # first we predict and save for existing training data days
        for i, data in enumerate(train_data):
            data = to_tensor(data, device=device)

            prediction = model(data)
            output.append(prediction.detach().cpu().numpy())

            if (i+1) % output_interval == 0 or (i+1) == len(train_data):
                print('predicting training day t+1 {}/{}...'.format(
                    i + 1, len(train_data)))

                # with market variables we will be dealing with very large amounts of data. better to save every few iterations
                pd.DataFrame(output, columns=columns, index=range(last_output_index, i+1)) \
                    .to_csv(output_path,
                            header=False,
                            mode='a')
                last_output_index = i + 1
                output = []

        # then we predict and save for new test data days
        for i, data in enumerate(test_data):
            data = to_tensor(data, device=device)

            prediction = model(data)
            print('Predicted test day {}/{}'.format(i+1, test_split_days))
            output.append(prediction.detach().cpu().numpy())

        output_df = pd.DataFrame(output)
        output_df.index = range(last_output_index, last_output_index+len(output_df))
        output_df.to_csv(output_path,
                            header=False,
                            mode='a')
        print('Saved results to "{}"...'.format(output_path))

## END TESTING ##
