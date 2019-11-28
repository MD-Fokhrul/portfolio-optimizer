from datetime import datetime
import pandas as pd
import numpy as np
from random import shuffle
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from dataset.dataset_loader import DatasetLoader


class SubsetSampler(Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def update_indices(self, indices):
        self.indices = indices


class FuturePricesLoader(DataLoader):

    def __init__(self, config, phase, batch_size, data_dir, dataset_name, past_prices_lookback_window=30,
                 num_cols_sample=None, limit_days=None, exclude_days=None, random_state=1):

        self.futureprices = FuturePrices(config, phase,
                                         data_dir=data_dir,
                                         dataset_name=dataset_name,
                                         past_prices_lookback_window=past_prices_lookback_window,
                                         num_cols_sample=num_cols_sample,
                                         limit_days=limit_days,
                                         exclude_days=exclude_days,
                                         random_state=random_state)

        sampler = SubsetSampler(self.futureprices.indices)
        num_workers = config['data_loader'][phase]['num_workers']

        self.data_dim = self.futureprices.dataframe.shape

        super().__init__(dataset=self.futureprices,
                         batch_size=batch_size,
                         sampler=sampler,
                         num_workers=num_workers)

    def add_day(self, day_prices):
        self.futureprices.add_day(day_prices)
        self.sampler.update_indices(self.futureprices.indices)


class FuturePrices(object):
    ## takes a config json object that specifies training parameters and a
    ## phase (string) to specifiy either 'train', 'test', 'validation'
    def __init__(self, config, phase, data_dir, dataset_name, past_prices_lookback_window=30,
                 num_cols_sample=None, limit_days=None, exclude_days=None, random_state=1):

        self.past_prices_lookback_window = past_prices_lookback_window
        self.config = config
        self.shuffle = config['data_loader'][phase]['shuffle']
        self.history_number = config['data_loader']['historic']['number']
        self.history_frequency = config['data_loader']['historic']['frequency']
        self.normalize_targets = config['target']['normalize']
        self.normalize_images = config['image']['normalize']
        self.target_mean = {}
        target_mean = config['target']['mean']
        for k, v in target_mean.items():
            self.target_mean[k] = np.asarray(v, dtype=np.float32)
        self.target_std = {}
        target_std = config['target']['std']
        for k, v in target_std.items():
            self.target_std[k] = np.asarray(v, dtype=np.float32)

        if limit_days is not None:
            # for testing need to pad days to accommodate for historic number window
            limit_days = limit_days + self.history_number + 1

        #### reading in dataframe from csv #####
        base_dataset_loader = DatasetLoader(data_dir, dataset_name)
        self.dataframe, _, _, _ = base_dataset_loader.get_data(num_cols_sample=num_cols_sample,
                                                               limit_days=limit_days,
                                                               exclude_days=exclude_days,
                                                               random_state=random_state,
                                                               as_numpy=False)

        # Here we calculate the temporal offset for the starting indices of each chapter. As we cannot cross chapter
        # boundaries but would still like to obtain a temporal sequence of images, we cannot start at index 0 of each chapter
        # but rather at some index i such that the i-max_temporal_history = 0
        # To explain see the diagram below:
        #
        #             chapter 1    chapter 2     chapter 3
        #           |....-*****| |....-*****| |....-*****|
        # indices:   0123456789   0123456789   0123456789
        #
        # where . are ommitted indices and - is the index. This allows using [....] as temporal input.
        #
        # Thus the first sample will consist of images:     [....-]
        # Thus the second sample will consist of images:    [...-*]
        # Thus the third sample will consist of images:     [..-**]
        # Thus the fourth sample will consist of images:    [.-***]
        # Thus the fifth sample will consist of images:     [-****]
        # Thus the sixth sample will consist of images:     [*****]

        self.sequence_length = self.history_number*self.history_frequency
        max_temporal_history = self.sequence_length

        # we remove window+temporal history from start so we always have a full window
        # we remove one from the end so that target/next_prices doesn't index out of bounds
        self.indices = self.dataframe.iloc[self.past_prices_lookback_window+max_temporal_history:-2].index.tolist()
        self.phase = phase

        #### phase specific manipulation #####
        if phase == 'train':
            pass
            # TODO: might want to add range for prices, but that might only work for log normalized price change
            # self.dataframe['canSteering'] = np.clip(self.dataframe['canSteering'], a_max=360, a_min=-360)

            ##### If you want to use binning on angle #####
            ## START ##
            # self.dataframe['bin_canSteering'] = pd.cut(self.dataframe['canSteering'],
            #                                            bins=[-360, -20, 20, 360],
            #                                            labels=['left', 'straight', 'right'])
            # gp = self.dataframe.groupby('bin_canSteering')
            # min_group = min(gp.apply(lambda x: len(x)))
            # bin_indices = gp.apply(lambda x: x.sample(n=min_group)).index.droplevel(level=0).tolist()
            # self.indices = list(set(self.indices) & set(bin_indices))
            ## END ##

        elif phase == 'validation':
            pass
            # TODO: might want to add range for prices, but that might only work for log normalized price change
            # self.dataframe['canSteering'] = np.clip(self.dataframe['canSteering'], a_max=360, a_min=-360)

        elif phase == 'test':
            self.indices = self.dataframe.iloc[self.past_prices_lookback_window+max_temporal_history:].index.tolist()
            # IMPORTANT: for the test phase indices will start 10s (100 samples) into each chapter
            # this is to allow challenge participants to experiment with different temporal settings of data input.
            # If challenge participants have a greater temporal length than 10s for each training sample, then they
            # must write a custom function here.

            # if 'next_prices' not in self.dataframe.columns:
            #     self.dataframe['next_prices'] = [0.0 for _ in range(len(self.dataframe))]


        # if self.normalize_targets and not phase == 'test':
        #     self.dataframe['next_prices'] = (self.dataframe['next_prices'].values -
        #                                     self.target_mean['next_prices']) / self.target_std['next_prices']

        if self.shuffle:
            shuffle(self.indices)

        print('Phase:', phase, '# of data:', len(self.indices))

        self.past_prices_transform = {
            True: transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=config['image']['norm']['mean'],
                                     std=config['image']['norm']['std'])
            ]),
            False: transforms.Compose([
                transforms.ToTensor()
            ])
        }[self.normalize_images]

        self.next_prices_transform = {
            True: transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=config['image']['norm']['mean'],
                                     std=config['image']['norm']['std'])
            ]),
            False: transforms.Compose([
                transforms.ToTensor()
            ])
        }[self.normalize_targets]
        # TODO: might want to add different transforms for factor data

    def add_day(self, day_prices):
        # update dataframe with new day and new index
        new_row = dict(zip(self.dataframe.columns, day_prices))
        current_max_index = self.dataframe.index.max()
        # new_index = (pd.Timestamp(current_max_index) + pd.DateOffset(days=1)).strftime('%d/%m/%Y')
        new_index = current_max_index + 1
        new_row_series = pd.Series(new_row, name=new_index)

        self.dataframe = self.dataframe.append(new_row_series)

        # update indices
        # remove oldest index and append new one (we don't want to process first day again)
        self.indices = self.indices[1:]
        self.indices.append(new_index)

    def __getitem__(self, index):
        inputs = {}
        labels = {}
        window_start = index - self.past_prices_lookback_window

        for i in range(self.sequence_length):
            inputs[i] = {}

            past_prices_img = self.dataframe.iloc[window_start-i+1:index-i+1].reset_index(drop=True, inplace=False).to_numpy()
            past_prices_img = self.past_prices_transform(past_prices_img)
            inputs[i]['past_prices'] = past_prices_img

        if self.phase != 'test':
            next_prices = self.dataframe.iloc[index+1].to_numpy().reshape(1, -1)
            next_prices = self.next_prices_transform(next_prices) # might need to remove reshape if we predict more than one line

            labels['next_prices'] = next_prices
        else:
            labels['next_prices'] = np.empty((1, inputs[0]['past_prices'].shape[1]))
        
        return inputs, labels

