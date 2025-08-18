import sys
import os

import fpsolvers.naive

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir,'..'))

import fpsolvers.anderson
import models
import torch

import time

def benchmark(fn, *args, warmup=5, reps=20):
    # warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    
    # timing
    start = time.time()
    for _ in range(reps):
        fn(*args)
    torch.cuda.synchronize()
    end = time.time()
    fpsolvers.nai
    return (end - start) / reps


if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = models.MLP(16,16,2048,4)
    x0 = torch.rand(16,16) - 0.5
    m = m.to(device)
    x0 = x0.to(device)
    input = (m,x0)

    print("naive:", benchmark(fpsolvers.naive, *input))
    print("anderson:", benchmark(fpsolvers.anderson, *input))
