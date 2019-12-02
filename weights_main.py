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
parser.add_argument('--input_dataset_name', type=str, help='predicted input dataset name')
parser.add_argument('--target_dataset_name', type=str, help='target ground truth dataset name')
parser.add_argument('--data_dir', type=str, default='data', help='data directory')
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

log_interval = args.log_interval
num_epochs = args.epochs
batch_size = args.batch_size
target_size = args.target_size
learning_rate = args.lr
data_dir = args.data_dir
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