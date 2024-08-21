import os.path
import numpy as np
import pickle

import os
import pandas as pd
import torch.nn.parallel
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter

class Dataset(Dataset):

    def __init__(self, use_index_list=None, dataset_type='train', seed=0, config=None):
        self.eval_length = config['others']['eval_length']
        self.feature_num = config['others']['feature_num']
        np.random.seed(seed)  # seed for ground truth choice
        dir_dataset = config['others']['dir_dataset']

        # The dataset has been normalized.
        path_observed_values = os.path.join(dir_dataset, 'observed_values.pickle')
        path_observed_masks = os.path.join(dir_dataset, 'observed_masks.pickle')
        path_gt_masks = os.path.join(dir_dataset, 'gt_masks.pickle')

        with open(path_observed_values, "rb") as f:
            self.observed_values = pickle.load(f).astype(float)
        with open(path_observed_masks, "rb") as f:
            self.observed_masks = pickle.load(f).astype(float)
        with open(path_gt_masks, "rb") as f:
            self.gt_masks = pickle.load(f).astype(float)

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        return {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }

    def __len__(self):
        return len(self.use_index_list)

def get_dataloader(seed=1, nfold=None, batch_size=16, missing_ratio=0.1, config=None):
    dataset = Dataset(dataset_type=missing_ratio, seed=seed, config=config)

    indlist = np.arange(len(dataset))
    np.random.seed(seed)
    np.random.shuffle(indlist)

    train_end = (int)(0.8 * len(dataset))
    train_index = indlist[0:train_end]
    val_end = (int)(0.1 * len(dataset))
    valid_index = indlist[train_end:train_end+val_end]
    test_index = indlist[train_end+val_end:]

    # training set
    train_dataset = Dataset(
        use_index_list=train_index, dataset_type='train', seed=seed, config=config
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # validation set
    valid_dataset = Dataset(
        use_index_list=valid_index, dataset_type='valid', seed=seed, config=config
    )

    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # test set
    test_dataset = Dataset(
        use_index_list=test_index, dataset_type='test', seed=seed, config=config
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, valid_loader, test_loader
