import torch
import matplotlib.pyplot as plt
import numpy as np

class GRU(torch.nn.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
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
        return outputs, (H, )

