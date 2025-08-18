import sys
import os

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir,'..'))

import torch
from line_profiler import profile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def naive(f,x0,max_iter = 50, tol = 1e-6):
    """
    This is the naive solver for fixed point problems.
    It's roughly 3x as fast as anderson for fast converging problems (<10 iters),
    but may be slower for longer problems.
    """
    b,n = x0.shape
    x = torch.zeros(2,b,n).to(x0.device)
    x[0,:,:] = x0
    x[1,:,:] = f(x0)
    write_head = 0
    iters = 0
    while iters<max_iter:
        if x0.is_cuda:
            torch.cuda.synchronize()
        resnorm = torch.norm(x[0,:,:] - x[1,:,:],dim=-1)/(1e-5+torch.norm(x[write_head,:,:],dim=-1))
        if x0.is_cuda:
            torch.cuda.synchronize()
        if (resnorm < tol).all():
            return x[write_head,:,:], iters
        if x0.is_cuda:
            torch.cuda.synchronize()
        write_head = (write_head+1)%2
        x[write_head,:,:] = f(x[(write_head+1)%2,:,:])
        if x0.is_cuda:
            torch.cuda.synchronize()
        iters+=1
    return x[write_head,:,:], iters+1


