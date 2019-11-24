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
parser.add_argument('--days_lookback_window', type=int, default=30, help='number of days to consider in an "image"')
parser.add_argument('--save_checkpoints', type=util.str2bool, nargs='?', const=True, default=True, help='should save checkpoints')
parser.add_argument('--checkpoints_interval', type=int, default=10, help='episodes interval for saving model checkpoint')
parser.add_argument('--checkpoints_root_dir', type=str, default='prices_checkpoints', help='checkpoint root directory')
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
    checkpoints_dir_name = experiment.get_key() if experiment is not None else str(start)
    checkpoints_dir = '{}/{}'.format(checkpoints_root_dir, checkpoints_dir_name)
    os.makedirs(checkpoints_dir, exist_ok=True)
# END SETUP CHECKPOINTS DIR #

# SETUP TORCH IMPORTS #  <--- must come after comet_ml import
import torch
import torch.optim as optim
from torch import nn
from future_prices.models import PricePredictionModel
from future_prices.lstm_dataloader import FuturePricesLoader
# END SETUP TORCH IMPORTS #

# SETUP DEVICE #
device_type = 'cuda' if not force_cpu and torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)
print('device is: ', device)
# END SETUP DEVICE #

# SETUP DATALOADERS #
data_loader_config = util.load_config('future_prices/lstm_config.json')
train_loader = FuturePricesLoader(data_loader_config, 'train', batch_size, data_dir, dataset_name,
                                      days_lookback_window, num_sample_stocks,
                                      limit_days=limit_days,
                                      exclude_days=val_days)
if val_days and val_days > 0:
    validation_loader = FuturePricesLoader(data_loader_config, 'validation', batch_size, data_dir, dataset_name,
                                          days_lookback_window, num_sample_stocks,
                                          limit_days=val_days)

validation_dataloader = None
# END SETUP DATALOADERS #

params = {
    'lr': lr,
    'batch_size': batch_size,
    'epochs': epochs,
    'log_interval': log_interval,
    'device': device_type,
    'train_data_shape': train_loader.data_dim,
    'validation_data_shape': validation_loader.data_dim
}

print('running with params: {}'.format(params))

if log_comet:
    experiment.log_parameters(params)
    experiment.tags(comet_tags)

# SETUP MODEL #
model = PricePredictionModel(input_output_size=train_loader.data_dim[1],
                             hidden_size=64) # todo: what about this hyperparam?
if device_type == 'cuda':
    model.cuda()
# END SETUP MODEL #

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
for epoch in range(0, epochs):
    start_epoch_train = time.time()
    epoch_prices_losses = []
    # epoch training
    model.train()
    running_prices_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        for k, v in data.items():
            for k2, v2 in v.items():
                data[k][k2] = v2.float().to(device)
        for k, v in target.items():
            target[k] = v.float().to(device)

        # for k,v in data.items():
        #     print('{}: {}'.format(k, v))


        optimizer.zero_grad()
        prediction = model(data)

        prices_loss = criterion(prediction['next_prices'], target['next_prices'].squeeze())
        combined_loss = prices_loss # if we add more factor other than prices, consider weighting
        combined_loss.backward()

        optimizer.step()

        # print avg batch statistics
        running_prices_loss += prices_loss.item()

        if batch_idx > 0 and batch_idx % log_interval == 0:
            avg_batch_prices_loss = running_prices_loss / log_interval
            epoch_prices_losses.append(avg_batch_prices_loss)
            if log_comet and log_batches:
                experiment.log_metric('batch_train_prices_loss', avg_batch_prices_loss, step=batch_idx)
            print('[epoch: %d, batch:  %5d] prices loss: %.5f' % (epoch + 1, batch_idx + 1, avg_batch_prices_loss))
            running_prices_loss = 0.0

        # Remove this when actually training.
        # Used to terminate early.
    #             if batch_idx >= 4:
    #                 break

    if log_comet and log_epochs:
        if len(epoch_prices_losses) > 0 and len(epoch_prices_losses) > 0:
            epoch_prices_loss = sum(epoch_prices_losses) / len(epoch_prices_losses)
            print('[avg train loss epoch %d] prices loss %.5f' % (epoch, epoch_prices_loss))
            experiment.log_metric('epoch_train_prices_loss', epoch_prices_loss, epoch)
        else:
            print('0 epoch losses for training')
    end_epoch_train = time.time()
    epoch_elapsed = end_epoch_train - start_epoch_train

    print('Saving interim model...')
    if save_checkpoints and (epoch+1) % checkpoints_interval == 0:
        model_path = '{}/model_{}.pth'.format(checkpoints_dir, epoch)
        torch.save(model.state_dict(), model_path)

    print('epoch %d: %f elapsed' % (epoch, epoch_elapsed))
    if log_comet:
        experiment.log_metric('epoch_train_time', end_epoch_train - start_epoch_train, step=epoch)

    # epoch validation
    model.eval()
    with torch.no_grad():
        epoch_validation_prices_losses = []
        for batch_idx, (data, target) in enumerate(validation_loader):
            for k, v in data.items():
                for k2, v2 in v.items():
                    data[k][k2] = v2.float().to(device)
            for k, v in target.items():
                target[k] = v.float().to(device)

            prediction = model(data)

            prices_loss = criterion(prediction['next_prices'], target['next_prices'].squeeze())

            epoch_validation_prices_losses.append(prices_loss.item())

        if len(epoch_validation_prices_losses) > 0:
            epoch_validation_prices_loss = sum(epoch_validation_prices_losses) / len(epoch_validation_prices_losses)
            print('[avg validation loss epoch %d] prices loss %.5f' % (epoch, epoch_validation_prices_loss))

            if log_comet:
                experiment.log_metric('epoch_val_prices_loss', epoch_validation_prices_loss, epoch)
        else:
            print('0 epoch losses for validation')


