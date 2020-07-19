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
from Preprocess import Dataset
from train_network import train_model
from eval_model import (
    model_loss,
    baseline_model_loss
)
from collections import defaultdict
torch.cuda.empty_cache()

#For energy scaling
scaling_mean = 0
scaling_numerator = 10000
to_scale_energy = True

KFOLD = 5


cols = ['speed-lvs-pussay', 'speed-parc-du-gatinais', 'speed-arville', 'speed-boissy-la-riviere', 'speed-angerville-1',
        'speed-lvs-pussay-b', 'speed-parc-du-gatinais-b', 'speed-arville-b', 'speed-boissy-la-riviere-b', 'speed-angerville-1-b']

feature_kwargs = {'lags_period': [1,2,3,23,47,71],
                'lags_columns' : cols,
                'energy_lags_period': [18,19,20,23,47,71,95,119,143],
                'energy_lags_columns': ['Energy'],
                'month_encode' :False,
                'year_encode' : False}

# NOTE THAT whereby train data is every data excluding buffer_pctg & test pctg
# train_pctg_constant is train pctg of train data.
# similarly val_pctg_constant is val pctg of train data
split_kwargs = {'train_pctg_constant': 0.7,
                'val_pctg_constant': 0.3,
                'test_pctg': 0.1,
                'buffer_pctg' : 0}

constructor = (
    Layer('Linear', None, 32, 'ReLU'),
    Layer('Linear', 32, 4, 'ReLU'),
    Layer('Linear', 4, 1, None)
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
    
    #ENERGY SCALING
    if to_scale_energy:
        df['Energy'] = (df['Energy'] - scaling_mean)/scaling_numerator

    
    # Generate Features
    dataset = Dataset(df, **feature_kwargs)

    # Aggregate all features, split, clean
    dataset.generate_final_dataset()
  
    # USING ROLLING WINDOW K FOLD VALIDATION
    train_portion = (1-split_kwargs['test_pctg'])
    baseline_results = defaultdict(list)
    model_results = defaultdict(list)
    models = list()
    for k in range(KFOLD) :
        split_kwargs['buffer_pctg'] = (train_portion/KFOLD)*(k)
        split_kwargs['train_pctg'] = (train_portion/KFOLD) * split_kwargs['train_pctg_constant']
        split_kwargs['val_pctg'] = (train_portion/KFOLD) * split_kwargs['val_pctg_constant']
        dataset.train_val_test_split(dataset.final_df, **split_kwargs)
        dataset.clean_train_val_test()
        dataset.scale_train_val_test(StandardScaler())
    
        # Load to torch
        data = dataset.load_data(device=device, drop_timestamp=True)
        print('Load successful')
    
        # Generate Network
        features = data['train'][0].shape[1]
    
        network = generateANN(constructor=constructor,
                              input_shape=features).to(device)
        newoptimizer = optimizer(network.parameters(),
                                 lr=learning_rate, weight_decay=weight_decay)
    
        # Train Network
        network, loss = train_model(network, data, criterion=loss_fn,
                                    optimizer=newoptimizer, batch_size=64,
                                    num_epochs=num_epochs, device=device,K=k)
        models.append(network)
        # Test Model
        print('\nResults\n----------')
        if to_scale_energy:
            print(f'Error already multiplied by {scaling_numerator}')
            
            for phase in ['train', 'val', 'test']:
                angentot = model_loss(network, data[phase], loss_fn) * scaling_numerator
                model_results[phase].append(angentot)
                print(f"Network loss on {phase} dataset : {angentot :.4f}")
    
            
            for phase in ['train', 'val', 'test']:
                baseline_loss = baseline_model_loss(data[phase], loss_fn) * scaling_numerator
                baseline_results[phase].append(baseline_loss )
                print(f"Base model loss on {phase} dataset : {baseline_loss :.4f}")
        
        else:
            for phase in ['train', 'val', 'test']:
                angentot =model_loss(network, data[phase], loss_fn)
                model_results[phase].append(angentot)
                print(f"Network loss on {phase} dataset : {angentot:.4f}")
    
            baseline_results = dict()
            for phase in ['train', 'val', 'test']:
                baseline_loss = baseline_model_loss(data[phase], loss_fn)
                baseline_results[phase].append(baseline_loss)
                print(f"Base model loss on {phase} dataset : {baseline_loss:.4f}")
    
    
    #ENSEMBLE
    Ensembledata = dataset.load_ensemble_data(trainedmodels = models,device=device, drop_timestamp=True)
    print('Load successful')
    constructor = (
        Layer('Linear', len(predictions), 1,None),
    )   
    EnsembleModel = generateANN(constructor=constructor,
                              input_shape=len(predictions)).to(device)
    newoptimizer = optimizer(EnsembleModel.parameters(),
                             lr=learning_rate, weight_decay=weight_decay)

    # Train Network
    EnsembleModel, loss = train_model(EnsembleModel, Ensembledata, criterion=loss_fn,
                                optimizer=newoptimizer, batch_size=64,
                                num_epochs=num_epochs, device=device,K=k)
    print('\nEnsemble Results\n----------')
    if to_scale_energy:
        print(f'Error already multiplied by {scaling_numerator}')
        
        for phase in ['train', 'val', 'test']:
            angentot = model_loss(EnsembleModel, Ensembledata[phase], loss_fn) * scaling_numerator
            model_results[phase].append(angentot)
            print(f"Network loss on {phase} dataset : {angentot :.4f}")

        
        for phase in ['train', 'val', 'test']:
            baseline_loss = baseline_model_loss(Ensembledata[phase], loss_fn) * scaling_numerator
            baseline_results[phase].append(baseline_loss )
            print(f"Base model loss on {phase} dataset : {baseline_loss :.4f}")
    
    else:
        for phase in ['train', 'val', 'test']:
            angentot =model_loss(EnsembleModel, Ensembledata[phase], loss_fn)
            model_results[phase].append(angentot)
            print(f"Network loss on {phase} dataset : {angentot:.4f}")

        baseline_results = dict()
        for phase in ['train', 'val', 'test']:
            baseline_loss = baseline_model_loss(Ensembledata[phase], loss_fn)
            baseline_results[phase].append(baseline_loss)
            print(f"Base model loss on {phase} dataset : {baseline_loss:.4f}")
    
    save_result(folder ='train_result',
                model = network,
                last_fold_train_loss = loss['train'],
                last_fold_val_loss = loss['val'],
                cols = cols,
                feature_kwargs = feature_kwargs,
                feature_splits = split_kwargs,
                optimizer =  newoptimizer,
                test_loss = model_results,
                baseline_test_loss = baseline_results)
    
