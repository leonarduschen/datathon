from preprocess import preprocess
from network_config import generateANN, optimizer, learning_rate, num_epochs, loss_fn
from train_network import train_model, plot_loss
from test_model import test_model_loss
from baseline_model import baseline_model_loss
import pandas as pd
import torch

from preprocess_beta import Dataset

if __name__ == '__main__':
    if torch.cuda.is_available():
        # you can continue going on here, like cuda:1 cuda:2....etc.
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    # PREPROCESS DATA

    #Load and generate features
    df = pd.read_csv('./rawdata/consolidated_autocaffe_data.csv')
    dataset = Dataset(df, lags_period = [1,2,3,4,5,23], EMA_spans = [5,10,15])
    
    #Aggregate all features, split, clean
    dataset.generate_final_dataset()
    dataset.train_val_test_split(dataset.final_df, 0.8, 0.1, 0.1)
    dataset.clean_train_val_test()
    
    #Load to torch
    data = dataset.load_data(drop_timestamp = True)
    

    # GENERATE NETWORK
    features = data['train'][0].shape[1]
    network = generateANN(features).to(device)
    newoptimizer = optimizer(network.parameters(),
                             lr=learning_rate, weight_decay=1e-4)

    # TRAIN NETWORK
    network, loss = train_model(network, data, criterion=loss_fn, optimizer=newoptimizer,
                                num_epochs=num_epochs, device=device)

    # TEST MODEL
    baseline_loss = baseline_model_loss(data, loss_fn)
    print("Base model loss on test dataset : ", baseline_loss)

    ann_loss = test_model_loss(network, data, loss_fn, device)
    print("Network loss on test dataset : ", ann_loss)
