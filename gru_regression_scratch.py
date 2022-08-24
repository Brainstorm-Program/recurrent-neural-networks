import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from datasets import *
from models import *

def train_model(model, dataset, params):

    # create the data generator to iterate over mini batches
    trainDataGenerator = torch.utils.data.DataLoader(dataset, **params['train_params'])

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['init_lr'])

    for epoch in range(params['num_epochs']):

        for data, label in trainDataGenerator:
            # The inputs need to be of the form T x B x N_in
            # where T is the total "time" duration of the signal, B is the batch size
            # and N_in is the feature dimensionality of an observation
            data = data.transpose(0, 1) #.to('cuda:0')

            # forward pass
            latent_activities, readout = model(data)

            readout = torch.stack(readout).permute(1,0,-1)
            # compute the loss
            loss = criterion(readout, label) #.to('cuda:0'))

            # backpropagate through time!
            loss.backward()

            # update model parameters
            optimizer.step()
            optimizer.zero_grad()
            
            print('loss: {}'.format(loss.item()))


if __name__ == '__main__':
    
    fhDataset = FitzhughNagumo(N=128, T=1000)

    params = {
        'n_inputs': 1,
        'n_hidden': 32,
        'num_epochs': 1000,
        'init_lr': 1e-2,
        'n_outputs': 1,

        'train_params': {
                    'batch_size': 128,
                    'shuffle': True,
                    'num_workers': 1
                }
    }

    model = GRU(params['n_inputs'], params['n_hidden'], params['n_outputs'])

    train_model(model, fhDataset, params)
