import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from copy import deepcopy

class FordADataset(Dataset):
    def __init__(self):
        self.root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
        x, y = self.read_data(self.root_url + "FordA_TRAIN.tsv")
        self.train_data = deepcopy(x)
        self.train_labels = deepcopy(y)

        x, y = self.read_data(self.root_url + "FordA_TEST.tsv")
        self.test_data = deepcopy(x)
        self.test_labels = deepcopy(y)

    def read_data(self, filename):
        data = np.loadtxt(filename, delimiter="\t")
        y = data[:, 0]
        x = data[:, 1:]
        return x, y.astype(int)

    def visualize_samples(self):
        import ipdb; ipdb.set_trace()
