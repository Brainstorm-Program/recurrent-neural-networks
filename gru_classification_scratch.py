import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from datasets import *
from models import *
import sklearn.metrics as M

def train_model(model, dataset, params, visualize_train=True):

    # create the data generator to iterate over mini batches
    trainDataGenerator = torch.utils.data.DataLoader(dataset, **params['train_params'])

    # We use the cross entropy (or the negative log likelihood loss) since for this
    # example we care about classification!
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['init_lr'])

    if visualize_train:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    for epoch in range(params['num_epochs']):

        for data, label in trainDataGenerator:
            # The inputs need to be of the form T x B x N_in
            # where T is the total "time" duration of the signal, B is the batch size
            # and N_in is the feature dimensionality of an observation
            data = data.transpose(0, 1)

            # forward pass
            latent_activities, readout = model(data)

            readout = readout[0]
            # compute the loss
            loss = criterion(readout, label)

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
    N, T = 128, 2000
    test_dataset = FitzhughNagumoClassification(N=N, T=T)
 
    # Create the data tensors
    x = test_dataset.data.permute(1,0,-1)
    y = test_dataset.labels

    # Compute the feedforward pass. 
    # But since we aren't training, we can do without the gradients
    with torch.no_grad():
        _, pred = model(x)
        pred = torch.argmax(pred[0], dim=-1)

    # Let's create the confusion matrix
    cm = M.confusion_matrix(y.numpy(), pred.numpy()).astype(np.float32)
    # Compute accuracy
    print('Overall accuracy: {}'.format((cm[0,0]+cm[1,1])/cm.sum()))

    # row normalize
    cm[0] = cm[0]/cm[0].sum()
    cm[1] = cm[1]/cm[1].sum()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(cm, cmap=plt.get_cmap('YlGn'), vmin=0., vmax=1.)

    for y in range(cm.shape[0]):
        for x in range(cm.shape[1]):
            ax.text(x, y, '%.2f' % cm[y, x],
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=12,
                 color='white',
                 fontweight='bold'
                 )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_ylabel('True label', fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicted label', fontsize=16, fontweight='bold')
    plt.savefig('thumbs/cfmat.png', bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    
    fhDataset = FitzhughNagumoClassification(N=128, T=1000)

    # If you are curious uncomment this!
    #fhDataset.visualize_samples()

    params = {
        'n_inputs': 1,
        'n_hidden': 32,
        'num_epochs': 50,
        'init_lr': 1e-2,
        'n_outputs': 2,

        'train_params': {
                    'batch_size': 128,
                    'shuffle': True,
                    'num_workers': 1
                }
    }

    # initialize the model architecture and set it to train mode
    model = GRU(params['n_inputs'], params['n_hidden'], params['n_outputs'], per_timestep_readout=False)
    model = model.train()

    # Now let's train the model. 
    # Pass visualize_train=False to suppress any display
    model = train_model(model, fhDataset, params, visualize_train=False)

    # Let's set the model to eval mode, and see its performance on a new random set
    model = model.eval()
    evaluate_model(model)
