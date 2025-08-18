import torch
import typing
import torch.nn as nn
import torch.nn.functional as F

class DEQ(torch.nn.Module):
    def __init__(self, transfer, update):
        super().__init__()
        self.transfer = transfer
        self.update = update
        self.maxiters = 24

    # def forward(self, x0):
    #     #Collect info for gradients for x0
    #     u0 = self.update(x0)
    #     z = torch.zeros(u0.shape).detach()
    #     last_z = z.clone().detach()
    #     last_norm_z = torch.norm(last_z, dim=-1)
    #     with torch.no_grad():
    #         while (cosine<0.999).any():
    #             z = self.transfer(last_z) + u0
    #             norm_z = torch.norm(z , dim = -1)
    #             cosine = (z * last_z).sum(dim = -1) / (norm_z * last_norm_z)
    #             last_z = z
    #             last_norm_z = norm_z
    #     #Collect info for gradients for dz_{t}/dz_{t-1}
    #     z_grad = z.clone()
    #     z_grad.requires_grad_()
    #     z_grad_out = self.transfer(z_grad)+u0
    #     return z_grad_out


