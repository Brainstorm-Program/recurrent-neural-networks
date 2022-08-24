import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from copy import deepcopy
import matplotlib.pyplot as plt
import torch

class FordADataset(Dataset):
    def __init__(self):
        self.root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
        x, y = self.read_data(self.root_url + "FordA_TRAIN.tsv")
        self.train_data = np.expand_dims(deepcopy(x), -1)
        self.train_labels = deepcopy(y)
        self.train_labels[self.train_labels == -1] = 0

        x, y = self.read_data(self.root_url + "FordA_TEST.tsv")
        self.test_data = np.expand_dims(deepcopy(x), -1)
        self.test_labels = deepcopy(y)
        self.test_labels[self.test_labels == -1] = 0
       
        self.num_classes = len(np.unique(self.train_labels))

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, idx):
        return torch.Tensor(self.train_data[idx]).float(), torch.Tensor([self.train_labels[idx]]).long()

    def read_data(self, filename):
        data = np.loadtxt(filename, delimiter="\t")
        y = data[:, 0]
        x = data[:, 1:]
        return x, y.astype(int)

    ''' Randomly pick n samples from each class to visualize
    This is a binary classification problem
    '''
    def visualize_samples(self, n, mode):
        fig = plt.figure(figsize=(12,4))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        if mode == 'train':
            X_data, X_labels = self.train_data, self.train_labels
        elif mode == 'test':
            X_data, X_labels = self.test_data, self.test_labels

        posIdx = np.where(X_labels == 1)[0]
        idx = np.random.randint(posIdx.shape[0], size=(n,))
        posSamples = X_data[idx, :, 0].T

        negIdx = np.where(X_labels == 0)[0]
        idx = np.random.randint(negIdx.shape[0], size=(n,))
        negSamples = X_data[idx, :, 0].T

        ax1.plot(posSamples)
        ax1.set_xlabel('Time', fontsize=16, fontweight='bold')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_title('Positive Samples', fontweight='bold', fontsize=16)

        ax2.plot(negSamples)
        ax2.set_xlabel('Time', fontsize=16, fontweight='bold')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_title('Negative Samples', fontweight='bold', fontsize=16)

        plt.show()
