import pandas as pd
from dataset.util import plot_stocks


class DatasetLoader():
    def __init__(self, data_dir, dataset_name):
        dataset_path = '%s/%s.csv' % (data_dir, dataset_name)
        self.data_df = pd.read_csv(dataset_path)

    def get_data(self, num_cols_sample=None, limit_days=None, random_state=1, as_numpy=True, plot=False):
        data_ret = self.data_df.drop(['Date'], axis=1)

        if limit_days:
            data_ret = data_ret.tail(limit_days) # limit to last n days

        data_ret = data_ret.dropna(axis=1, how='any') # drop cols/stocks with NA prices

        # remove date before col/stock sampling but keep it for pandas representation

        if num_cols_sample:
            data_ret = data_ret.sample(num_cols_sample, axis=1, random_state=random_state)

        fig = plot_stocks(data_ret) if plot else None

        if as_numpy:
            data_ret = data_ret.to_numpy()

        return data_ret, fig


