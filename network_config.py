# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 02:24:04 2020

@author: dandy
"""
import yaml
import torch
from collections import namedtuple

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Define model architecture


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

    model = torch.nn.Sequential(*layers)
    return model

# Define loss criterion


def loss_fn(ypred, y):
    return torch.mean((ypred - y)**2)/1e3


# Config
num_epochs = config['num_epochs']
learning_rate = float(config['lr'])
optimizer = getattr(torch.optim, config['optimizer'])

# NN constructor
Layer = namedtuple('Layer', ['combiner',
                             'shape_in',
                             'shape_out',
                             'activation'])
