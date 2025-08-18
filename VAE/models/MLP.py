import torch
import typing
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, n_hidden):
        super().__init__()
        dims_list = [dim_in,dim_out]
        self.layers : list = []
        for _ in range(n_hidden):
            dims_list.insert(-1,dim_hidden)
        for (i,o) in zip(dims_list[0:-1],dims_list[1:]):
            self.layers.append(torch.nn.Linear(i,o))
            self.layers.append(torch.nn.LeakyReLU())
        self.model = torch.nn.Sequential(*self.layers)
        self.apply(self.init_weights)

    def init_weights(self, m : torch.nn.Module):
        if isinstance(m,nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)

    def forward(self, x : torch.Tensor):
        x = x.flatten(start_dim=1)
        return self.model(x)