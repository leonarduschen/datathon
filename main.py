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
from preprocess import Dataset
from train_network import train_model
from eval_model import (
    model_loss,
    baseline_model_loss
)


cols = ['speed-guitrancourt', 'speed-lieusaint', 'speed-lvs-pussay',
        'speed-parc-du-gatinais', 'speed-arville', 'speed-boissy-la-riviere',
        'speed-angerville-1', 'speed-angerville-2', 'speed-guitrancourt-b',
        'speed-lieusaint-b', 'speed-lvs-pussay-b', 'speed-parc-du-gatinais-b',
        'speed-arville-b', 'speed-boissy-la-riviere-b', 'speed-angerville-1-b',
        'speed-angerville-2-b']

feature_kwargs = {'lags_period': [1],
                  'lags_columns': cols,}

split_kwargs = {'train_pctg': 0.8,
                'val_pctg': 0.1,
                'test_pctg': 0.1}

constructor = (
    Layer('Linear', None, 64, 'ReLU'),
    Layer('Linear', 64, 32, 'ReLU'),
    Layer('Linear', 32, 16, 'ReLU'),
    Layer('Linear', 16, 1, None)
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
    print(dataset.train.columns)

    # Load to torch
    data = dataset.load_data(device = device, drop_timestamp=True)
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
    print('\nResults\n----------')
    for phase in ['train','val','test']:
        ann_loss = model_loss(network, data[phase], loss_fn, device)
        print(f"Network loss on {phase} dataset : {ann_loss:.4f}")
    
    print('\nResults\n----------')
    for phase in ['train', 'val', 'test']:
        baseline_loss = baseline_model_loss(data[phase], loss_fn)
        print(f"Base model loss on {phase} dataset : {baseline_loss:.4f}")


    

