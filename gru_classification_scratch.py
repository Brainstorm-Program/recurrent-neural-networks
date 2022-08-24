import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from datasets import FordADataset

class GRU(torch.nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, sigma=0.01):
        super().__init__()

        # Gaussian random init with standard deviation *sigma*
        init_weight = lambda *shape: nn.Parameter(torch.randn(*shape) * sigma)

        # It is easier to initialize it this way since we always need to worry about
        # (1) projections from the inputs, (2) projections from the latent state, and (3) the bias
        # Note that unlike biological RNNs, we **do not** introduce stochasticity in the activities
 
        triple = lambda: (init_weight(num_inputs, num_hiddens),
                          init_weight(num_hiddens, num_hiddens),
                          nn.Parameter(torch.zeros(num_hiddens)))

        # create the parameters for the update gate
        self.W_xz, self.W_hz, self.b_z = triple()

        # create the parameters for the reset gate
        self.W_xr, self.W_hr, self.b_r = triple()

        # hidden state parameters
        self.W_xh, self.W_hh, self.b_h = triple()

        # readout layer parameters
        self.fc = nn.Linear(num_hiddens, num_outputs)
        self.relu = nn.ReLU()

    ''' Given that our parent class is nn.Module, what we are doing here is essentially *overloading*
    This is the function that will be called when we pass a batch of inputs to the GRU
    '''
    def forward(self, inputs, H=None):
        matmul_H = lambda A, B: torch.matmul(A, B) if H is not None else 0
        outputs = []
        for X in inputs:
            Z = torch.sigmoid(torch.matmul(X, self.W_xz) + (
                torch.matmul(H, self.W_hz) if H is not None else 0) + self.b_z)
            if H is None: H = torch.zeros_like(Z)
            R = torch.sigmoid(torch.matmul(X, self.W_xr) +
                            torch.matmul(H, self.W_hr) + self.b_r)
            H_tilda = torch.tanh(torch.matmul(X, self.W_xh) +
                               torch.matmul(R * H, self.W_hh) + self.b_h)
            H = Z * H + (1 - Z) * H_tilda
            outputs.append(H)

        # readout layer
        readout = self.fc(self.relu(H))

        return outputs, readout

def train_model(model, dataset, params):

    # create the data generator to iterate over mini batches
    trainDataGenerator = torch.utils.data.DataLoader(dataset, **params['train_params'])

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['init_lr'])

    for epoch in range(params['num_epochs']):

        for data, label in trainDataGenerator:
            optimizer.zero_grad()

            # forward pass
            latent_activities, readout = model(data)

            # compute the loss
            loss = criterion(readout, label)

            # backpropagate through time!
            loss.backward()

            # update model parameters
            optimizer.step()

            if epoch%5 == 0:
                print('loss: {}'.format(loss.item()))


if __name__ == '__main__':
    fordDataset = FordADataset()
    #fordDataset.visualize_samples(n = 25, mode = 'train')

    params = {
        'n_inputs': 1,
        'n_hidden': 32,
        'num_epochs': 100,
        'init_lr': 1e-4,
        'n_classes': 2,

        'train_params': {
                    'batch_size': 1024,
                    'shuffle': True,
                    'num_workers': 0
                }
    }

    model = GRU(params['n_inputs'], params['n_hidden'], params['n_classes'])
    # if you want to port the mode to GPU
    # model = model.to('cuda:0')

    train_model(model, fordDataset, params)
