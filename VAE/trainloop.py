import torch.utils.data.dataloader
import models
import torch
import torch.nn as nn

import os



if __name__=='__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = models.Linear(3*32*32,10).to(device)
    crit = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)

    trainset = torch.load('./dataset/CIFAR10_train.pt',weights_only=False)
    trainloader = torch.utils.data.DataLoader(trainset,32,shuffle=True,num_workers=16)

    testset = torch.load('./dataset/CIFAR10_test.pt',weights_only=False)
    testloader = torch.utils.data.DataLoader(testset,32,shuffle = True,num_workers=16)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    print_period = 500
    for epoch in range(10):
        curr_loss = 0
        for (i, data) in enumerate(trainloader):
            img,y = data
            img = img.to(device)
            y = y.to(device)
            y_hat = model(img)
            loss = crit(y_hat,y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            curr_loss += loss.item()
            if i%print_period == print_period-1:
                print(f"Epoch: {epoch}    Batch: {i}    Current loss: {curr_loss/print_period}")
                curr_loss = 0
    print("Done")            


