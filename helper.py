# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 01:12:15 2020

@author: dandy
"""
from datetime import datetime
import os
import pickle
from matplotlib import pyplot as plt



def save_result(folder,*args,**kwargs):
    modelID = datetime.today().strftime('%d%m%Y_%H%M%S')
    subfolder = modelID
    print(f"Saving model on ./{folder}/{subfolder}")
    subfolderPath = os.path.join(folder,subfolder)
    #Create folder to save all files
    if not os.path.exists(folder):
        os.makedirs(folder) 
    if not os.path.exists(subfolderPath):
        os.makedirs(subfolderPath)
      
    #Saving file for reloading   
    filename = 'Model_'+subfolder    
    parametersFilename = filename+'.pickle'
    parametersFilepath = os.path.join(subfolderPath,parametersFilename)
    with open(parametersFilepath, 'wb') as handle:
        pickle.dump(kwargs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #Saving file for view
    filename = 'Model_'+subfolder    
    parametersFilename = filename+'.txt'
    parametersFilepath = os.path.join(subfolderPath,parametersFilename)
    with open(parametersFilepath, 'w') as f:
        for k, v in kwargs.items():
            f.write(str(k) + ' >>> '+ str(v) + '\n\n')
        
        
    #Saving plot
    train_loss = 'train_loss'
    val_loss='val_loss'
    if train_loss in kwargs.keys() and val_loss in kwargs.keys():
        plotFilename = filename+'.png'
        plotFilepath = os.path.join(subfolderPath,plotFilename)
        plt.plot(kwargs[train_loss])
        plt.plot(kwargs[val_loss])
        plt.title('Training vs Validation Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch') 
        plt.legend([train_loss, val_loss])  
        plt.savefig(plotFilepath) 
        plt.show()
        plt.close()
