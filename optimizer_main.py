import pandas as pd
import numpy as np
from scipy.optimize import Bounds
from optimizer.util import optimize
import argparse

###
# This conic optimizer is used for two cases in the architecture:
# 1. generating ground truth labels for the final fully connected layer. In this case only specify input_path_real and not predicted
# 2. generating part of the training data for the final fully connected layer. In this case specify input_path_real and predicted
# In case #1 we generate an optimal weight using a lookback window of k real price vectors + one day lookahead real price vector.
# In case #2 we generate an optimal weight using a lookback window of k real price vectors + one day lookahead predicted price vector.
###

parser = argparse.ArgumentParser()
parser.add_argument('--input_path_real', required=True, type=str, help='prices input data csv path')
parser.add_argument('--input_path_predicted', default=None, type=str, help='prices input data csv path')
parser.add_argument('--output_path', required=True, type=str, help='optimal weights output data csv path')
parser.add_argument('--limit_days', type=int, default=None, help='limit to final days from input data. Must be greater than lookback window')
parser.add_argument('--exclude_predicted_days', type=int, default=None, help='exclude final days from predicted input data')
parser.add_argument('--weight_min', type=float, default=-0.5, help='weight min value')
parser.add_argument('--weight_max', type=float, default=2.0, help='weight max value')
parser.add_argument('--lookback_window', type=int, default=90, help='lookback window days')
args = parser.parse_args()

input_path_real = args.input_path_real
input_path_predicted = args.input_path_predicted
output_path = args.output_path
weight_min = args.weight_min
weight_max = args.weight_max
lookback_window = args.lookback_window
limit_days = args.limit_days
exclude_predicted_days = args.exclude_predicted_days

prices_real = pd.read_csv(input_path_real)
# prediction mode or ground truth mode
prices_predicted = pd.read_csv(input_path_predicted) if input_path_predicted else prices_real

if limit_days:
    prices_real = prices_real.iloc[-limit_days:]

    # for predicted prices we will likely have an excess of newly generate test days at the end
    # we do not need those for generating the optimized data and labels
    predicted_limit_days = limit_days + exclude_predicted_days if exclude_predicted_days else limit_days
    prices_predicted = prices_predicted.iloc[-predicted_limit_days:]

if exclude_predicted_days:
    prices_predicted = prices_predicted[:-exclude_predicted_days]

# make sure real and predicted prices are aligned
try:
    assert prices_real.shape == prices_predicted.shape
except AssertionError:
    print(prices_real.shape)
    print(prices_predicted.shape)
    exit(1)

num_days = prices_real.shape[0]
num_stocks = prices_real.shape[1]

w_ret = np.zeros(prices_real.shape)
bounds = Bounds([weight_min]*num_stocks, [weight_max]*num_stocks)
for i in range ((lookback_window-1), num_days):
    # get lookback of k days from real prices and lookahead of 1 from predicted prices
    train_real = prices_real.iloc[i-(lookback_window-1):i]
    train_predicted = prices_predicted.iloc[i:i+1]
    train = pd.concat([train_real, train_predicted], axis=0).reset_index(drop=True)

    print("optimizing row {}/{}...".format(i-(lookback_window-1), num_days))
    cov = train.cov()
    expected_val = train.mean()
    w = np.random.rand(len(train.columns))
    w = w/np.sum(w)
    test = optimize(w, cov.values, expected_val, bounds)
    w_ret[i-(lookback_window-1), :] = test

pd.DataFrame(w_ret).to_csv(output_path)
