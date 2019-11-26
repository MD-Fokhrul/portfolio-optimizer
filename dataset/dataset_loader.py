import pandas as pd
from dataset.util import plot_stocks


class DatasetLoader():
    def __init__(self, data_dir, dataset_name):
        dataset_path = '%s/%s.csv' % (data_dir, dataset_name)
        self.data_df = pd.read_csv(dataset_path, index_col=0)

    # get dataframe or numpy array.
    # can sample number of stocks (columns) and limit number of days (rows).
    # can also return plot figure with stock prices over time
    def get_data(self, num_cols_sample=None, limit_days=None, exclude_days=None,
                 test_split_days=0, random_state=1, as_numpy=True, plot=False, dropna=True):

        data_ret = self.data_df

        if 'Date' in data_ret:
            data_ret = self.data_df.drop(['Date'], axis=1) # we don't need date col

        if limit_days:
            # limit to latest n days
            data_ret = data_ret.tail(limit_days).reset_index(drop=True, inplace=False)

        if exclude_days:
            data_ret = data_ret.head(data_ret.shape[0] - exclude_days).reset_index(drop=True, inplace=False)

        if dropna:
            data_ret = data_ret.dropna(axis=1, how='any') # drop cols/stocks with NA prices in selected day range

        if num_cols_sample:
            # sample columns/stocks
            data_ret = data_ret.sample(num_cols_sample, axis=1, random_state=random_state)

        # we want the first (1-test_split) rows as training data and the next test_split rows as test data
        num_rows_data = data_ret.shape[0]
        num_rows_test = test_split_days

        train_data = data_ret[:num_rows_data-num_rows_test]
        test_data = data_ret[-num_rows_test:]

        # plot stocks timeseries
        train_fig = plot_stocks(train_data) if plot else None
        test_fig = plot_stocks(test_data) if plot and test_split_days > 0 else None

        if as_numpy:
            train_data = train_data.to_numpy()
            test_data = test_data.to_numpy()

        return train_data, test_data, train_fig, test_fig


