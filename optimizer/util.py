import numpy as np
from scipy.optimize import minimize
import pandas as pd


# conic optimization of price change covariance to weights using SLSQP method
def optimize(w, cov, expected_val, bounds):
    init_guess = np.ones(len(w)) * (1.0 / len(w))
    weights = minimize(lambda w: (np.matmul(w.T,np.matmul(cov,w)) - np.matmul(w.T,expected_val)), init_guess,
                       method='SLSQP', options={'disp': False}, constraints=({'type': 'eq', 'fun': lambda w: 1.0 - np.sum(w)}), bounds=bounds)
    return weights.x


# multiprocessing approach to optimization (each row takes 30 seconds to optimize, so multithreading is crucial)
# modifies w_ret
def handle_optimization(i, bounds, lookback_window, prices_real, prices_predicted, num_stocks, num_days):
    # get lookback of k days from real prices and lookahead of 1 from predicted prices
    train_real = prices_real.iloc[i-(lookback_window-1):i]
    train_predicted = prices_predicted.iloc[i:i+1]
    train = pd.concat([train_real, train_predicted], axis=0).reset_index(drop=True)

    print("optimizing row {}/{}...".format(i-(lookback_window-1), num_days))
    cov = train.cov()
    expected_val = train.mean()
    w = np.random.rand(num_stocks)
    w = w/np.sum(w)
    test = optimize(w, cov.values, expected_val, bounds)
    return (i-(lookback_window-1), test)
