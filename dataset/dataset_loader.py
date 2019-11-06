import pandas as pd
from dataset.util import plot_stocks


class DatasetLoader():
    def __init__(self, data_dir, dataset_name):
        dataset_path = '%s/%s.csv' % (data_dir, dataset_name)
        self.data_df = pd.read_csv(dataset_path)

    # get dataframe or numpy array.
    # can sample number of stocks (columns) and limit number of days (rows).
    # can also return plot figure with stock prices over time
    def get_data(self, num_cols_sample=None, limit_days=None, random_state=1, as_numpy=True, plot=False):
        data_ret = self.data_df.drop(['Date'], axis=1) # we don't need date col

        if limit_days:
            # limit to latest n days
            data_ret = data_ret.tail(limit_days)

        data_ret = data_ret.dropna(axis=1, how='any') # drop cols/stocks with NA prices in selected day range

        if num_cols_sample:
            # sample columns/stocks
            data_ret = data_ret.sample(num_cols_sample, axis=1, random_state=random_state)

        # plot stocks timeseries
        fig = plot_stocks(data_ret) if plot else None

        if as_numpy:
            data_ret = data_ret.to_numpy()

        return data_ret, fig


