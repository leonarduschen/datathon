# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 02:24:04 2020

@author: dandy
"""
import yaml
import torch

with open('config.yml', 'r') as file:
    config = yaml.load(file)

# Define model architecture


def generateANN(D_in, D_out=1):
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


# Config
num_epochs = config['num_epochs']
learning_rate = float(config['lr'])
optimizer = getattr(torch.optim, config['optimizer'])


# Define loss criterion


def loss_fn(ypred, y):
    return torch.mean((ypred - y)**2)/1000

# loss_fn = torch.nn.MSELoss(reduction='sum')
