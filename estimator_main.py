import argparse
from estimator.ExpValModel import ExpValModel
import torch.nn as nn
import torch
import torch.optim as optim
from dataset.dataset_loader import DatasetLoader
import util
from model.util import to_tensor

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='sp500', help='dataset name')
parser.add_argument('--data_dir', type=str, default='data', help='data directory')
parser.add_argument('--test_split_days', type=int, default=152, help='number of days to set as test data')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
parser.add_argument('--limit_days', type=int, help='limit days (steps per episode)')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--log_comet', type=util.str2bool, nargs='?', const=True, default=False, help='should log to comet')
parser.add_argument('--comet_tags', nargs='+', default=[], help='tags for comet logging')
parser.add_argument('--force_cpu', type=util.str2bool, nargs='?', const=True, default=False, help='should force cpu even if cuda is available')
parser.add_argument('--checkpoints_interval', type=int, default=50, help='epochs interval for saving model checkpoint')
parser.add_argument('--checkpoints_root_dir', type=str, default='estimation_checkpoints', help='checkpoint root directory')
parser.add_argument('--results_root_dir', type=str, default='estimation_results', help='results root directory')
parser.add_argument('--save_checkpoints', type=util.str2bool, nargs='?', const=True, default=False, help='should save checkpoints?')
parser.add_argument('--load_model', type=str, default=None, help='checkpoint dir path to load from')
parser.add_argument('--modes', nargs='+', default=['train'], help='train and/or test')
args = parser.parse_args()

log_comet = args.log_comet
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr
data_dir = args.data_dir
test_split_days = args.test_split_days
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

dataloader = DatasetLoader(data_dir, dataset_name)
train_data, test_data = dataloader.get_data(limit_days=limit_days,
                                            test_split_days=test_split_days,
                                            as_numpy=True,
                                            drop_test=True)


model = ExpValModel()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
model.train()
for epoch in range(1):
    running_loss = 0.0
    for i in range(train_data.shape[0]):
        data = 
        optimizer.zero_grad()
        prediction = model()

        loss = criterion(prediction, torch.from_numpy(target))

        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 10 == 1:
            print('[epoch: %d, batch:  %5d] loss: %.5f' %
                  (epoch + 1, i + 1, running_loss / 2.0))
            running_loss = 0.0

model.eval()
with torch.no_grad():
    for i, (data, target) in enumerate(zip(test_data,test_val)):
        prediction = model(torch.from_numpy(data))

        mse = criterion(prediction, torch.from_numpy(target))
        print(mse)
