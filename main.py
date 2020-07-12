from preprocess import preprocess
from network_config import generateANN, optimizer, learning_rate, num_epochs, loss_fn
from train_network import train_model, plot_loss
from test_model import test_model_loss
from baseline_model import baseline_model_loss
import pandas as pd
import torch

if __name__ == '__main__':
    if torch.cuda.is_available():
        # you can continue going on here, like cuda:1 cuda:2....etc.
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    # PREPROCESS DATA
    df = pd.read_csv('./rawdata/consolidated_autocaffe_data.csv')
    data = preprocess(df)

    # GENERATE NETWORK
    features = data['train'][0].shape[1]
    network = generateANN(features).to(device)
    newoptimizer = optimizer(network.parameters(), lr=learning_rate)

    # TRAIN NETWORK
    network, loss = train_model(network, data, criterion=loss_fn, optimizer=newoptimizer,
                                num_epochs=num_epochs, device=device)

    # TEST MODEL
    baseline_loss = baseline_model_loss(data, loss_fn)
    print("Base model loss on test dataset : ", baseline_loss)

    ann_loss = test_model_loss(network, data, loss_fn)
    print("Network loss on test dataset : ", ann_loss)
