# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 02:24:04 2020

@author: dandy
"""
import torch

# Define model architecture


def generateANN(D_in, D_out= 1):
    H = D_in

    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 35),
        torch.nn.ReLU(),
        torch.nn.Linear(35, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, D_out),
    )
    return model


# Define Optimizer Config
learning_rate = 1e-4
optimizer = torch.optim.Adam

# Define epochs parameters
num_epochs = 10000

# Define loss criterion
def loss_fn(ypred,y):
    return torch.mean((ypred-y)**2)

# loss_fn = torch.nn.MSELoss(reduction='sum')
