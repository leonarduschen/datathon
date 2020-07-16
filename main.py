import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from network_config import (
    generateANN,
    optimizer,
    learning_rate,
    num_epochs,
    loss_fn,
    Layer
)
from train_network import train_model
from test_model import test_model_loss
from baseline_model import baseline_model_loss
from preprocess import Dataset


cols = ['speed-guitrancourt', 'speed-lieusaint', 'speed-lvs-pussay',
        'speed-parc-du-gatinais', 'speed-arville', 'speed-boissy-la-riviere',
        'speed-angerville-1', 'speed-angerville-2', 'speed-guitrancourt-b',
        'speed-lieusaint-b', 'speed-lvs-pussay-b', 'speed-parc-du-gatinais-b',
        'speed-arville-b', 'speed-boissy-la-riviere-b', 'speed-angerville-1-b',
        'speed-angerville-2-b', 'Energy']

feature_kwargs = {'lags_period': [0, 1, 2],
                  'lags_columns': cols,
                  'SMA_windows': [3, 12, 48],
                  'SMA_columns': cols,
                  'SMSTD_windows': [6, 24, 72],
                  'SMSTD_columns': cols}

split_kwargs = {'train_pctg': 0.8,
                'val_pctg': 0.1,
                'test_pctg': 0.1}

constructor = (
    Layer('Linear', None, 128, 'ReLU'),
    Layer('Linear', 128, 64, 'ReLU'),
    Layer('Linear', 64, 32, 'ReLU'),
    Layer('Linear', 32, 1, None)
)

if __name__ == '__main__':
    if torch.cuda.is_available():
        # you can continue going on here, like cuda:1 cuda:2....etc.
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    # Load
    df = pd.read_csv('./rawdata/consolidated_autocaffe_data.csv')

    # Generate Features
    dataset = Dataset(df, **feature_kwargs)

    # Aggregate all features, split, clean
    dataset.generate_final_dataset()
    dataset.train_val_test_split(dataset.final_df, **split_kwargs)
    dataset.clean_train_val_test()
    dataset.scale_train_val_test(scaler=StandardScaler())
    print(dataset.train.columns)

    # Load to torch
    data = dataset.load_data(drop_timestamp=True)
    print('Load successful')

    # GENERATE NETWORK
    features = data['train'][0].shape[1]

    network = generateANN(constructor=constructor,
                          input_shape=features).to(device)
    newoptimizer = optimizer(network.parameters(),
                             lr=learning_rate, weight_decay=1)

    # TRAIN NETWORK
    network, loss = train_model(network, data, criterion=loss_fn,
                                optimizer=newoptimizer,
                                num_epochs=num_epochs, device=device)

    # TEST MODEL
    baseline_loss = baseline_model_loss(data, loss_fn)
    print("Base model loss on test dataset : ", baseline_loss)

    ann_loss = test_model_loss(network, data, loss_fn, device)
    print("Network loss on test dataset : ", ann_loss)
