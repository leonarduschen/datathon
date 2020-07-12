# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 14:13:26 2020

@author: dandy
"""


def testModelLoss(model,data,criterion):
    inputs,labels = data['test']
    preds = model(inputs.float())
    loss = criterion(preds,labels).item()
    return loss