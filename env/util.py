import numpy as np


# calculate stock price volatility before current day step, over lookback days
# input: data - nxm daily stock prices (m stocks over n days)
# input: current_step -  day's step index
# input: lookback - desired volatility lookback window
# output: 1xm volatility of stocks within lookback window
def calculate_volatility(data, current_step, lookback):
    if current_step in (0, 1):
        # at steps 0,1 we don't have any volatility to compute yet
        return np.ones((1, data.shape[1])) / data.shape[1]

    lookback_start = max(current_step - lookback, 0)
    lookback_end = current_step
    prev_window = data[lookback_start:lookback_end]  # P[t-1] - latest lookback prices excluding today
    current_window = data[lookback_start + 1:lookback_end + 1]  # P[t] - latest lookback prices including today
    perc_change = np.log(current_window / prev_window)  # ln(P[t]/P[t-1]) - day-wise percentage change in prices
    perc_change_mean = np.mean(perc_change, axis=0)
    variance = (1 / (lookback - 1)) * np.sum(np.power(perc_change - perc_change_mean, 2), axis=0)  # price variance
    return np.sqrt(variance)



