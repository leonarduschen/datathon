# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 13:34:53 2020

THIS FILE IS TO CALCULATE LOSS OF BASELINE MODEL
"""


def baseline_model_loss(data, loss_fn, lag=18):
    pred = data['test'][1][:-lag]
    actual = data['test'][1][lag:]
    loss = loss_fn(pred, actual).item()
    return loss
