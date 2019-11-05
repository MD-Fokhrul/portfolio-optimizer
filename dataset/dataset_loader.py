import pandas as pd


class DatasetLoader():
    def __init__(self, data_dir, dataset_name):
        dataset_path = '%s/%s.csv' % (data_dir, dataset_name)
        self.data_df = pd.read_csv(dataset_path).dropna(axis=1, how='any') # drop cols/stocks with NA prices

    def get_data(self, num_cols_sample=None, random_state=1):
        data_ret = self.data_df.drop(['Date'], axis=1)

        if num_cols_sample:
            data_ret = data_ret.sample(num_cols_sample, axis=1, random_state=random_state)

        return data_ret.to_numpy()


