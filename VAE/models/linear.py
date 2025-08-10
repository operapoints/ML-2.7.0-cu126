import torch
import typing
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self, dim_in : int, dim_out : int):
        super().__init__()
        self.A = nn.Linear(dim_in,dim_out)
        self.apply(self.init_weights)

    def init_weights(self, m : torch.nn.Module):
        if isinstance(m,nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)

    def forward(self, x : torch.Tensor):
        x = x.flatten(start_dim = 1)
        x = self.A(x)
        x = F.leaky_relu(x)

        return x


