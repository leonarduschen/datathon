# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import torch
import datetime
import pandas as pd


def preprocess(df, train_pctg=0.8, val_pctg=0.1, test_pctg=0.1):
    df_dict = dict()
    dataloader_dict = dict()

    df.drop(['Timestamp'], inplace = True, axis = 1)
    
    df_dict['train'], df_dict['val'], df_dict['test'] = train_val_test_split(
        df, train_pctg, val_pctg, test_pctg)

    
    
    for key in df_dict.keys():
        tensor_data = torch.from_numpy(df_dict[key].values)
        Y = tensor_data[:, 0]
        X = tensor_data[:, 1:]
        dataloader_dict[key] = (X, Y)

    return dataloader_dict


def train_val_test_split(df, train_pctg, val_pctg, test_pctg):
    rows = df.shape[0]
    max_train_idx = round(rows * train_pctg)
    max_validation_idx = round(rows * val_pctg) + max_train_idx

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.sort_values('Timestamp', inplace = True)

    train = df[:max_train_idx]
    val = df[max_train_idx:max_validation_idx]
    test = df[max_validation_idx:]

    print(f"""
          Completed train val test split 
          -------------------------------
          Total data : {rows} rows
          Train data : {train.shape} rows
          Val data : {val.shape} rows
          Test data {test.shape} rows""")

    return train, val, test
