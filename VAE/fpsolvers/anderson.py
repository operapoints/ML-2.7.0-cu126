import sys
import os

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir,'..'))

import torch
from line_profiler import profile
from typing import Callable


def anderson(f : Callable[[torch.Tensor],torch.Tensor],
            x0 : torch.Tensor,
            m=5, lam=1e-4, max_iter=50,
            tol=1e-6) -> torch.Tensor:
    """
    Uses Anderson acceleration to solve for fixed points.
    Convergence seems to be pretty slow.
    """
    iters = 2
    device = x0.device
    dtype = x0.dtype
    original_shape = x0.shape
    x0 = torch.flatten(x0,start_dim=1)
    B,n = x0.shape
    Gf = torch.zeros(B,m,n,dtype = dtype, device = device)
    Ff = torch.zeros(B,m,n,dtype = dtype, device = device)
    Xf = torch.zeros(B,m,n,dtype = dtype, device = device)
    Xf[:,0,:] = x0
    Ff[:,0,:] = Xf[:,1,:] = f(x0)
    Ff[:,1,:] = f(Xf[:,1,:].reshape(*original_shape))
    Gf[:,0,:] = Ff[:,0,:] - Xf[:,0,:]
    Gf[:,1,:] = Ff[:,1,:] - Xf[:,1,:]
    yf = torch.zeros(B,m+1,1,dtype = dtype, device = device)
    yf[:,0,:] = 1
    Hf = torch.zeros(B,m+1,m+1,dtype = dtype, device = device)
    Hf[:,0,1:] = Hf[:,1:,0] = 1
    for k in range(1,max_iter):
        mk = min(m,k)
        if(mk<m):
            G = Gf[:,:mk+1,:] #b,mk,n
            F = Ff[:,:mk+1,:] #b,mk,n
            H = Hf[:,:mk+2,:mk+2] #b,mk+1,mk+1
            y = yf[:,:mk+2,:] #b,mk+1,n
        else:
            G=Gf
            F=Ff
            H=Hf
            y=yf
        H[:,1:,1:] = torch.bmm(G,G.transpose(1,2)) + lam * torch.eye(min(mk+1,m),dtype = dtype, device = device).unsqueeze(0)
        a : torch.Tensor = torch.linalg.solve(H,y)[:,1:,:] #b,mk,1
        x : torch.Tensor = (F.transpose(1,2)@a)[:,:,0] #b,n
        Xf[:,(k+1)%m,:] = x
        Ff[:,(k+1)%m,:] = f(x.reshape(*original_shape))
        Gf[:,(k+1)%m,:] = Ff[:,(k+1)%m,:] - Xf[:,(k+1)%m,:]
        norm = torch.norm(Gf[:,(k+1)%m,:],dim=-1)/(1e-5+torch.norm(Ff[:,(k+1)%m,:],dim=-1))
        iters +=1
        if (norm<tol).all():
            return Ff[:,(k+1)%m,:].reshape(*original_shape), iters
    return Ff[:,max_iter%m,:].reshape(*original_shape), iters

# def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-6, beta = 1.0):
#     """ Anderson acceleration for fixed point iteration. """
#     bsz, d = x0.shape
#     X = torch.zeros(bsz, m, d, dtype=x0.dtype, device=x0.device)
#     F = torch.zeros(bsz, m, d, dtype=x0.dtype, device=x0.device)
#     X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
#     X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)
    
#     H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
#     H[:,0,1:] = H[:,1:,0] = 1
#     y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
#     y[:,0] = 1
#     iters = 2
    
#     res = []
#     for k in range(2, max_iter):
#         n = min(k, m)
#         G = F[:,:n]-X[:,:n]
#         H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
#         alpha = torch.linalg.solve(H[:,:n+1,:n+1],y[:,:n+1], )[:, 1:n+1, 0]   # (bsz x n)
        
#         X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
#         F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
#         iters+=1
#         res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
#         if (res[-1] < tol):
#             break
#     return X[:,k%m].view_as(x0), iters

