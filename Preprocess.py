# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import torch
import datetime
def preprocess(df,train_pctg=0.5,val_pctg =0.2,test_pctg=0.3):
    df.Energy = df.Energy.apply(lambda x : df[df.Energy!=0].Energy.mean() if x==0 else x)
    #THIS IS FEATURE FOR BASELINE MODEL
    df.insert(2,'t-18',df.Energy.shift(18))
    data=dict()
    data['train'],data['val'],data['test'] = train_val_test_split(df)
    
    dataloader = dict()
    for key in data.keys():
        tensor_data = torch.from_numpy(data[key].values)
        Y = tensor_data[:,0]
        X = tensor_data[:,1:]
        dataloader[key] = (X,Y)
    
    return dataloader

def train_val_test_split(df,train_pctg=0.5,val_pctg =0.2,test_pctg=0.3):
    df['Timestamp'] = df['Timestamp'].apply(lambda x: datetime.datetime.strptime(x,'%d/%m/%Y %H:%S'))
    rows = df.shape[0]
    MaxTrainidx = round(rows * train_pctg)
    MaxValidationidx= round(rows * val_pctg)+MaxTrainidx

    
    data = df.sort_values('Timestamp').iloc[:,1:] 
    
    train = data[:MaxTrainidx]
    val = data[MaxTrainidx:MaxValidationidx]
    test = data[MaxValidationidx:]
    print(f"""
          Completed train val test split 
          -------------------------------
          Total data : {rows} rows
          Train data : {train.shape} rows
          Val data : {val.shape} rows
          Test data {test.shape} rows""")
    return train,val,test