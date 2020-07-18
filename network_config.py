# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 02:24:04 2020

@author: dandy
"""
import yaml
import torch
from collections import namedtuple
from torch import nn


# with open('config.yml', 'r') as file:
#     config = yaml.safe_load(file)

# Define model architecture


class CustomNNSequential(nn.modules.container.Sequential):
    def __init__(self,layers):
        nn.modules.container.Sequential.__init__(self,*layers)
   
  
    def forward(self, input):
        for module in self: 
            isLSTM = str(type(module)) == "<class 'torch.nn.modules.rnn.LSTM'>"
            if isLSTM:
                #In this case batch size is 1
                input,_ = module(input.view(len(input),1,-1))
            else:
                input = module(input)
        return input
     


def generateANN(constructor, input_shape):
    layers = list()
    initial_layer = True
    for c in constructor:
        # If initial layer follow shape of input
        if initial_layer:
            combiner = getattr(torch.nn, c.combiner)(input_shape, c.shape_out)
            initial_layer = False
        else:
            combiner = getattr(torch.nn, c.combiner)(c.shape_in, c.shape_out)
        # Activation function
        if c.activation:
            activation = getattr(torch.nn, c.activation)()
            layers.extend([combiner, activation])
        else:
            layers.append(combiner)
    
        model = CustomNNSequential(layers)
    return model

# Define loss criterion
loss_fn = torch.nn.L1Loss()



# Config
num_epochs = 50000
learning_rate = 1e-5
weight_decay = 0.01
optimizer = getattr(torch.optim, 'Adam')

# NN constructor
Layer = namedtuple('Layer', ['combiner',
                             'shape_in',
                             'shape_out',
                             'activation'])
