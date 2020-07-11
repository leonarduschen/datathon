# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 02:24:24 2020

@author: dandy
"""
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import time
import torch

# Stolen from https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss
        
def plot_loss(trainLoss,valLoss):
    plt.plot(trainLoss,label='train_loss')
    plt.plot(valLoss,label='val_loss')
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.show()
    plt.pause(0.0001)

        
def train_model(model, dataloaders, criterion, optimizer, num_epochs=10000,earlystoppingPatience=10,device =torch.device("cpu")):
    START_TIME = time.time()

    loss_Dict=defaultdict(list)
    
    model = model.to(device)
    early_stopping = EarlyStopping(patience=earlystoppingPatience, verbose=True)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        phases = ['train', 'val']
        for phase in phases:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)   # Set model to evaluate mode

            # Load data
            inputs, labels = dataloaders[phase]
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # 
            # Feed forward and track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                # Get model prediction and calculate loss
                preds = model(inputs.float())
                loss = criterion(ypred = preds, y = labels)

                # Back propagation + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
        
            epoch_loss = loss.item()/preds.shape[0]           
            loss_Dict[phase].append(epoch_loss)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
           
            #Check for early stopping condition during validation
            if phase == 'val':
                early_stopping(epoch_loss, model) 
        plot_loss(trainLoss = loss_Dict['train'],valLoss = loss_Dict['val'])
        if early_stopping.early_stop:
            print("Early stopping")
            break

    time_elapsed = time.time() - START_TIME
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
  
    # load best model weights
    model.load_state_dict(torch.load('checkpoint.pt'))

    

    return model,loss_Dict