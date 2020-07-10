# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 02:57:56 2020

@author: dandy
"""
import torch
def generate_dummy():
    N, D_in, D_out = 1000, 8, 1
    # Create random Tensors to hold inputs and outputs
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)
    return x,y
