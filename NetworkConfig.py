# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 02:24:04 2020

@author: dandy
"""
import torch

#Define model architecture
def generateANN(D_in,D_out=1):
    H = D_in//2
    
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
        )
    return model

#Define Optimizer Config
LEARNING_RATE = 1e-1
optimizer = torch.optim.Adam

#Define epochs parameters
NUM_EPOCHS = 10000

#Define loss criterion
def loss_fn(ypred,y):
  return (ypred - y).pow(2).mean()

#Define early stopping's patience
Patience = 10