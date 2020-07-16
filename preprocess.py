import torch
import datetime
import pandas as pd
import numpy as np

class Dataset:
    def __init__(self, df, sort = True,
        lags_period = None, lags_columns = ['Energy'],
        EMA_spans = None, EMA_columns = ['Energy'],
        SMA_windows = None, SMA_columns = ['Energy'],
        SMSTD_windows = None, SMSTD_columns = ['Energy']):
        
        self.df = df
        
        #For training
        self.train = None
        self.val = None
        self.test = None
        self.final_df = None

        #Features
        self.lags = None
        self.EMA = None
        self.SMA = None    
        self.SMSTD = None

        if sort:
            'Sorted dataset by timestamp'
            self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
            self.df.sort_values('Timestamp', inplace = True)
        
        if lags_period:
            self.lags = self.generate_lag(self.df, lags = lags_period, columns = lags_columns)
        
        if EMA_spans:
            self.EMA = self.generate_EMA(self.df, spans = EMA_spans, columns = EMA_columns)

        if SMA_windows:
            self.SMA = self.generate_SMA(self.df, windows = SMA_windows, columns = SMA_columns)

        if SMSTD_windows:
            self.SMSTD = self.generate_SMA(self.df, windows = SMSTD_windows, columns = SMSTD_columns)

    
    def generate_final_dataset(self):
        """Aggregate all features and store in self.final_df"""

        self.final_df = pd.concat([self.df, self.lags, self.EMA, self.SMA, self.SMSTD], axis = 1)
        print("All features combined")

    def train_val_test_split(self, df, train_pctg, val_pctg, test_pctg):
        """Split and store dataframe in self.train, self.val, self.test"""

        rows = df.shape[0]
        max_train_idx = round(rows * train_pctg)
        max_validation_idx = round(rows * val_pctg) + max_train_idx
        print(f"""
            Completed train val test split 
            -------------------------------
            Total data : {df.shape}
            Train data : {df[:max_train_idx].shape}
            Val data : {df[max_train_idx:max_validation_idx].shape}
            Test data {df[max_validation_idx:].shape}\n """)

        self.train, self.val, self.test = (
            df[:max_train_idx].copy(), df[max_train_idx:max_validation_idx].copy(), df[max_validation_idx:].copy()
            )

    def clean_train_val_test(self):
        """Drop all null rows"""
        self.train.dropna(inplace = True, axis = 0)
        self.val.dropna(inplace = True, axis = 0)
        self.test.dropna(inplace = True, axis = 0)
        print(f"""
            Train {self.train.shape},
            Val {self.val.shape},
            Test {self.test.shape},
            \n """)

    def load_data(self, drop_timestamp = True):
        """Data loader for NN"""
        dataloader_dict = dict()

        if drop_timestamp:
            self.train.drop(['Timestamp'], inplace = True, axis = 1)
            self.val.drop(['Timestamp'], inplace = True, axis = 1)
            self.test.drop(['Timestamp'], inplace = True, axis = 1)

        for key, value in zip(['train', 'val', 'test'], [self.train.values, self.val.values, self.test.values]):
            tensor_data = torch.from_numpy(value)
            Y = tensor_data[:, 0]
            X = tensor_data[:, 1:]
            dataloader_dict[key] = (X, Y)

        return dataloader_dict

    #Feature called on __init__
    def generate_lag(self, df, lags, columns):
        lag_df = pd.concat([self.df[columns].shift(lag) for lag in lags], axis=1)
        lag_df.columns = [f'lagged-{lag}-{col}' for col in columns for lag in lags]
        print(f"Lag {lags} generated, size: {lag_df.shape}")

        return lag_df

    #Feature called on __init__
    def generate_EMA(self, df, spans, columns):
        EMA_df = pd.concat([self.df[columns].ewm(span = span).mean() for span in spans], axis = 1)
        EMA_df.columns = [f'EMA-{span}-{col}' for col in columns for span in spans]
        print(f"EMA {spans} generated, size: {EMA_df.shape}")

        return EMA_df

    #Feature calle on __init__
    def generate_SMA(self, df, windows, columns):
        SMA_df = pd.concat([self.df[columns].rolling(window = window).mean() for window in windows], axis = 1)
        SMA_df.columns = [f'SMA-{window}-{col}' for col in columns for window in windows]
        print(f"SMA {windows} generated, size: {SMA_df.shape}")
        
        return SMA_df

    #Feature called on __init__
    def generate_SMSTD(self, df, windows, columns):
        SMSTD_df = pd.concat([self.df[columns].rolling(window = window).std() for window in windows], axis = 1)
        SMSTD_df.columns = [f'STD-{window}-{col}' for col in columns for window in windows]
        print(f"EMA {windows} generated, size: {SMSTD_df.shape}")

        return SMSTD_df
