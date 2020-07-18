import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from helper import save_result
from network_config import (
    generateANN,
    optimizer,
    learning_rate,
    num_epochs,
    loss_fn,
    weight_decay,
    Layer
)
from preprocess import Dataset
from train_network import train_model
from eval_model import (
    model_loss,
    baseline_model_loss
)

torch.cuda.empty_cache()
cols = ['speed-lvs-pussay', 'speed-parc-du-gatinais', 'speed-arville', 'speed-boissy-la-riviere', 'speed-angerville-1',
        'speed-lvs-pussay-b', 'speed-parc-du-gatinais-b', 'speed-arville-b', 'speed-boissy-la-riviere-b', 'speed-angerville-1-b']

feature_kwargs = {'lags_period': [18,20,23,47],
                'lags_columns': ['Energy']}

split_kwargs = {'train_pctg': 0.5,
                'val_pctg': 0.1,
                'test_pctg': 0.1,
                'buffer_pctg':0.3}

constructor = (
    Layer('Linear', 64, 64, 'ReLU'),
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
    df = pd.read_csv('./rawdata/consolidated_autocaffe_data_shifted.csv')
    cols_to_drop = [col for col in df.columns if (col not in cols) & (col not in ['Energy','Timestamp'])]
    df.drop(cols_to_drop, axis = 1, inplace = True)

    df['Energy'] = df['Energy']/1000

    # Generate Features
    dataset = Dataset(df, **feature_kwargs)

    # Aggregate all features, split, clean
    dataset.generate_final_dataset()
    dataset.train_val_test_split(dataset.final_df, **split_kwargs)
    dataset.clean_train_val_test()
    dataset.scale_train_val_test(StandardScaler())
    print(dataset.train.columns)

    # Load to torch
    data = dataset.load_data(device=device, drop_timestamp=True)
    print('Load successful')

    # GENERATE NETWORK
    features = data['train'][0].shape[1]

    network = generateANN(constructor=constructor,
                          input_shape=features).to(device)
    newoptimizer = optimizer(network.parameters(),
                             lr=learning_rate, weight_decay=weight_decay)

    # TRAIN NETWORK
    network, loss = train_model(network, data, criterion=loss_fn,
                                optimizer=newoptimizer,
                                num_epochs=num_epochs, device=device)

    # TEST MODEL
    print('\nResults\n----------')
    model_results = dict()
    for phase in ['train', 'val', 'test']:
        model_results[phase] = model_loss(network, data[phase], loss_fn)
        print(f"Network loss on {phase} dataset : {model_results[phase]:.4f}")

    baseline_results = dict()
    for phase in ['train', 'val', 'test']:
        baseline_results[phase] = baseline_model_loss(data[phase], loss_fn)
        print(f"Base model loss on {phase} dataset : {baseline_results[phase]:.4f}")
    
    # save_result(folder ='train_result',
    #             model = network,
    #             train_loss = loss['train'],
    #             val_loss = loss['val'],
    #             cols = cols,
    #             feature_kwargs = feature_kwargs,
    #             feature_splits = split_kwargs,
    #             optimizer =  newoptimizer,
    #             test_loss = model_results,
    #             baseline_test_loss = baseline_results)
