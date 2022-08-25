import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from datasets import *
from models import *
import tqdm

def train_model(model, dataset, params, visualize_train=True):

    # create the data generator to iterate over mini batches
    trainDataGenerator = torch.utils.data.DataLoader(dataset, **params['train_params'])

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['init_lr'])

    if visualize_train:
        fig = plt.figure()
        ax = fig.add_subplot(111)

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
            
            if visualize_train:
                ax.clear()
                # Let's pick index 0, since batch is shuffled anyway!
                ax.plot(data[:,0,0].detach().numpy(), linewidth=2, color='tab:gray', label='groundtruth')
                ax.plot(readout[0,:,0].detach().numpy(), '--', linewidth=2, color='r', label='prediction')

                # Just formatting options. This is my pet peeve so you can safely ignore!
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_title('Training epoch: {}'.format(epoch))
                ax.set_xlabel('Time', fontsize=16, fontweight='bold')
                ax.set_ylabel('Firing rate (in a.u.)', fontsize=16, fontweight='bold')
                ax.legend(loc='upper right')
                ax.set_xticks([0., data.shape[0]])
                ax.set_xticklabels(['0ms', '{}ms'.format(data.shape[0])])
                ax.set_yticks([])
                ax.set_ylim([-2.5, 2.5]) 
                
                plt.pause(0.1)

        print('Epoch: {} | Training Loss: {}'.format(epoch, loss.item()))

    return model

def evaluate_model(model):
    # First off, let's create a new dataset. Since the initializations are random, we can 
    # consider this a proper test! To make life harder for the model, lets change up T too
    N, T = 1024, 2000
    test_dataset = FitzhughNagumo(N=1024, T=2000)

    # We still need an evaluation criterion
    criterion = torch.nn.MSELoss()
 
    # Create the data tensors
    x = torch.Tensor(test_dataset.data_x.reshape(N, T, 1)).permute(1,0,-1)
    y = torch.Tensor(test_dataset.data_y.reshape(N, T, 1))

    # Compute the feedforward pass. 
    # But since we aren't training, we can do without the gradients
    with torch.no_grad():
        _, pred = model(x)
        pred = torch.stack(pred).permute(1, 0, -1)
        test_error = criterion(pred, y).item()

    # Q: How does this compare with the training loss?
    # What can you say about this?
    print('Test error: {}'.format(test_error))

if __name__ == '__main__':
    
    fhDataset = FitzhughNagumo(N=128, T=1000)

    params = {
        'n_inputs': 1,
        'n_hidden': 32,
        'num_epochs': 50,
        'init_lr': 1e-2,
        'n_outputs': 1,

        'train_params': {
                    'batch_size': 128,
                    'shuffle': True,
                    'num_workers': 1
                }
    }

    # initialize the model architecture and set it to train mode
    model = GRU(params['n_inputs'], params['n_hidden'], params['n_outputs'])
    model = model.train()

    # Now let's train the model. 
    # Pass visualize_train=False to suppress any display
    model = train_model(model, fhDataset, params)

    # Let's set the model to eval mode, and see its performance on a new random set
    model = model.eval()
    evaluate_model(model)
