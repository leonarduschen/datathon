from Preprocess import preprocess
from NetworkConfig import generateANN,optimizer,LEARNING_RATE,NUM_EPOCHS,loss_fn
from TrainNetwork import train_model, plot_loss
from TestModel import testModelLoss
from baselineModel import baselineModelLoss
import pandas as pd
import torch

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU") 
    
    #PREPROCESS DATA
    df = pd.read_csv('./rawdata/consolidated_autocaffe_data.csv')
    data = preprocess(df)   
    
    #GENERATE NETWORK 
    features = data['train'][0].shape[1]
    Network = generateANN(features).to(device)
    newoptimizer = optimizer(Network.parameters(), lr=LEARNING_RATE)
    
    #TRAIN NETWORK
    Network,loss=train_model(Network,data,criterion=loss_fn,optimizer=newoptimizer,
                       num_epochs = NUM_EPOCHS,device=device)
    
    #TEST MODEL
    BASELINELOSS = baselineModelLoss(data,loss_fn)
    print("Base model loss on test dataset : ",BASELINELOSS)
    
    ANNLOSS = testModelLoss(Network,data,loss_fn)
    print("Network loss on test dataset : ",ANNLOSS)

    
    
