import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from copy import deepcopy
import matplotlib.pyplot as plt
import torch
from scipy import integrate

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


class FitzhughNagumo(Dataset):
    def __init__(self, N, T, I=0.5, a=0.7, b=0.8):
        self.I = I
        self.a = a
        self.b = b
        self.N = N
        self.T = T

        data_x = []
        data_y = []
        for i in range(N):
            t = np.linspace(0,400,T+1)
            x0 = np.array([float(np.random.rand(1))*2.-1.,0.])
            sol = integrate.solve_ivp(self.FHN_rhs, [0,400], x0, t_eval=t)
            data_x.append(sol.y[0,:-1])
            data_y.append(sol.y[0,1:])

        self.data_x = np.array(data_x).reshape(N,T,1)
        self.data_y = np.array(data_y).reshape(N,T,1)

    def __len__(self):
        return self.data_x.shape[0]

    def __getitem__(self, idx):
        return torch.Tensor(self.data_x[idx]), torch.Tensor(self.data_y[idx])

    def FHN_rhs(self, t,x):
        I, a, b = self.I, self.a, self.b
        eps = 1./50.
        dim1 = x[0] - (x[0]**3)/3. - x[1] + I
        dim2 = eps*(x[0] + a - b*x[1])
        out = np.stack((dim1,dim2)).T

        return out

    def get_init(self):
        t = np.linspace(0,400,self.T+1)
        x0 = np.array([float(np.random.rand(1))*2.-1.,0.])
        sol = integrate.solve_ivp(self.FHN_rhs, [0,400], x0, t_eval=t)
        init_x = sol.y[0, :50]
        return init_x

class FitzhughNagumoClassification(Dataset):
    def __init__(self, N, T):

        # Let's sample data from two distinct dynamical systems
        classA = FitzhughNagumo(N=N, T=T, I=0.5, a=0.95, b=0.2)
        classB = FitzhughNagumo(N=N, T=T, I=0.5, a=0.7, b=0.2)
       
        # now let's create the dataset with appropriate class labels
        self.data = torch.Tensor(np.vstack([classA.data_x, classB.data_x])).float()
        self.labels = torch.Tensor(np.vstack([np.zeros((classA.data_x.shape[0])), np.zeros((classA.data_x.shape[0]))+1])).flatten().long()
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def visualize_samples(self): 
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(self.data[0, :, 0], linewidth=2, alpha=0.75, c='tab:orange', label='Class A')
        ax.plot(self.data[-1, :, 0], linewidth=2, alpha=0.75, c='tab:brown', label='Class B')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Time', fontsize=16, fontweight='bold')
        ax.set_ylabel('Firing rate (in a.u.)', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right')
        ax.set_xticks([0., self.data.shape[1]])
        ax.set_xticklabels(['0ms', '{}ms'.format(self.data.shape[1])])
        ax.set_yticks([])
        ax.set_ylim([-2.5, 2.5]) 

        plt.show()
