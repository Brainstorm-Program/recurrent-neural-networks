import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from datasets import *
from models import *
import tqdm

def train_model(model, dataset, params, visualize_train=True, warm_up=50):

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

            # forward pass to warm-up
            latent_activities, readout = model(data[:warm_up])

            # now the autoregression begins
            autoreg_outputs = []
            latent = latent_activities[-1]
            X = readout[-1]

            for t in range(warm_up, data.shape[0]):
                latent, X = model.single_step(X, latent)
                autoreg_outputs.append(X)

            autoreg_outputs = torch.stack(autoreg_outputs)

            # compute the loss
            loss = criterion(autoreg_outputs, data[warm_up:]) #.to('cuda:0'))

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


def generate(model, init_x, future_T=1000):
    model = model.eval()
    gen_seq = init_x.clone()

    for t in tqdm.tqdm(range(future_T)):
        with torch.no_grad():
            _, output = model(gen_seq)
            gen_seq = torch.cat([gen_seq, output[-1].unsqueeze(0)], dim=0) 

    return gen_seq

if __name__ == '__main__':
    
    fhDataset = FitzhughNagumo(N=256, T=512)

    params = {
        'n_inputs': 1,
        'n_hidden': 128,
        'num_epochs': 1000,
        'init_lr': 1e-3,
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
    model = model.to('cuda:0')

    # Now let's train the model. 
    # Pass visualize_train=False to suppress any display
    model = train_model(model, fhDataset, params, visualize_train=False)
    torch.save(model.state_dict(), 'autoregressiveGRU.pth')

    # This is going to be cool. We can treat RNNs as "generative" models too :)
    # Let's "seed" the model with an initial sequence
    init_x = fhDataset.get_init()
    init_x = torch.Tensor(init_x[:, np.newaxis, np.newaxis])
    gen_seq = generate(model, init_x)

    plt.plot(gen_seq.squeeze().numpy())
    plt.show()
