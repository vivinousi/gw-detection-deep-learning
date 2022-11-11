import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# code from https://github.com/passalis/dain
class DAIN_Layer(nn.Module):
    def __init__(self, mode='full', mean_lr=0.00001, gate_lr=0.001, scale_lr=0.00001, input_dim=144):
        super(DAIN_Layer, self).__init__()
        print("Mode = ", mode)

        self.mode = mode
        self.mean_lr = mean_lr
        self.gate_lr = gate_lr
        self.scale_lr = scale_lr

        # Parameters for adaptive average
        self.mean_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.mean_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))

        # Parameters for adaptive std
        self.scaling_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.scaling_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))

        # Parameters for adaptive scaling
        self.gating_layer = nn.Linear(input_dim, input_dim)

        self.eps = 1e-8

    def forward(self, x):
        # Expecting  (n_samples, dim,  n_feature_vectors)
        # Step 1:
        avg = torch.mean(x, 2)
        adaptive_avg = self.mean_layer(avg)
        adaptive_avg = adaptive_avg.resize(adaptive_avg.size(0), adaptive_avg.size(1), 1)
        x = x - adaptive_avg

        # # Step 2:
        std = torch.mean(x ** 2, 2)
        std = torch.sqrt(std + self.eps)
        adaptive_std = self.scaling_layer(std)
        adaptive_std[adaptive_std <= self.eps] = 1

        adaptive_std = adaptive_std.resize(adaptive_std.size(0), adaptive_std.size(1), 1)
        x = x / adaptive_std

        # Step 3:
        avg = torch.mean(x, 2)
        gate = torch.sigmoid(self.gating_layer(avg))
        gate = gate.resize(gate.size(0), gate.size(1), 1)
        x = x * gate
        return x
