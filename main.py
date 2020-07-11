from Preprocess import preprocess
from NetworkConfig import generateANN,optimizer,LEARNING_RATE,NUM_EPOCHS,loss_fn
from TrainNetwork import train_model, plot_loss
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('./rawdata/consolidated_autocaffe_data.csv')
    
    data = preprocess(df)   
    
    features = data['train'][0].shape[1]
    Network = generateANN(features)    
    newoptimizer = optimizer(Network.parameters(), lr=LEARNING_RATE)
    
    _,loss=train_model(Network,data,criterion=loss_fn,optimizer=newoptimizer,
                       num_epochs = NUM_EPOCHS)
    
    
    plot_loss(loss['train'],loss['val'])
