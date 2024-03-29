import torch
import pandas as pd


class Dataset:
    def __init__(self, df, sort=True, dayfirst = True, fixwindspeed=False,
                 lags_period=None, lags_columns=['Energy'],
                 EMA_spans=None, EMA_columns=['Energy'],
                 SMA_windows=None, SMA_columns=['Energy'],
                 SMSTD_windows=None, SMSTD_columns=['Energy'],
                 diffs_period = None, diffs_columns = ['Energy'],
                 energy_lags_period = None, energy_lags_columns=['Energy'],
                 month_encode = None, year_encode = None):

        self.df = df

        # For training
        self.train = None
        self.val = None
        self.test = None
        self.final_df = None

        # Features
        self.lags = None
        self.EMA = None
        self.SMA = None
        self.SMSTD = None
        self.diffs = None
        self.month = None
        self.year = None
        
        #Corrections to speed and direction
        if fixwindspeed:
            self.df.loc[df.Timestamp<'2017-08-01','speed-boissy-la-riviere'] = 0
            self.df.loc[df.Timestamp<'2017-08-01','speed-boissy-la-riviere-b']=0
            self.df.loc[df.Timestamp<'2019-07-02','speed-angerville-1']=0
            self.df.loc[df.Timestamp<'2019-07-02','speed-angerville-2']=0
            self.df.loc[df.Timestamp<'2019-07-02','speed-angerville-1-b']=0
            self.df.loc[df.Timestamp<'2019-07-02','speed-angerville-2-b']=0

        if sort:
            'Sorted dataset by timestamp'
            self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'], dayfirst = dayfirst)
            self.df.sort_values('Timestamp', inplace=True)

        if lags_period:
            self.lags = self.generate_lag(self.df,
                                          lags=lags_period,
                                          columns=lags_columns)

        if EMA_spans:
            self.EMA = self.generate_EMA(self.df,
                                         spans=EMA_spans,
                                         columns=EMA_columns)

        if SMA_windows:
            self.SMA = self.generate_SMA(self.df,
                                         windows=SMA_windows,
                                         columns=SMA_columns)

        if SMSTD_windows:
            self.SMSTD = self.generate_SMA(self.df,
                                           windows=SMSTD_windows,
                                           columns=SMSTD_columns)
        
        if diffs_period:
            self.diffs = self.generate_diffs(self.df,
                                            diffs = diffs_period,
                                            columns = diffs_columns)

        if energy_lags_period:
            self.energy_lags = self.generate_lag(self.df,
                                                lags = energy_lags_period,
                                                columns = energy_lags_columns)
        if month_encode:
            self.month = self.generate_month_one_hot(df)
            
        if year_encode:
            self.year = self.generate_year_one_hot(df)
    def generate_final_dataset(self):
        """Aggregate all features and store in self.final_df"""

        self.final_df = pd.concat([self.df, self.lags, self.EMA,
                                   self.SMA, self.SMSTD, self.diffs,
                                   self.energy_lags,self.month,self.year], axis=1)
        print("All features combined")

    def train_val_test_split(self, df, train_pctg, val_pctg, test_pctg, buffer_pctg = 0,*args,**kwargs):  #ADDED BUFFER OPTION HERE!
        """Split and store dataframe in self.train, self.val, self.test"""        
        rows = df.shape[0]

        max_buffer_idx = round(rows * buffer_pctg)
        max_train_idx = round(rows * train_pctg) + max_buffer_idx
        max_validation_idx = round(rows * val_pctg) + max_train_idx
        max_test_idx = round(rows *test_pctg) + max_validation_idx
        print(f"""
            Completed buffer,train val test split ({buffer_pctg},{train_pctg},{val_pctg},{test_pctg})
            -------------------------------
            Total data : {df.shape}
            Buffer data : {df[:max_buffer_idx].shape}
            Train data : {df[max_buffer_idx : max_train_idx].shape}
            Val data : {df[max_train_idx : max_validation_idx].shape}
            Test data {df[max_validation_idx:].shape}\n """)

        self.train, self.val, self.test = (
            df[max_buffer_idx : max_train_idx].copy(),
            df[max_train_idx : max_validation_idx].copy(),
            df[max_validation_idx : max_test_idx].copy()
        )

    def clean_train_val_test(self):
        """Drop all null rows"""
        self.train.dropna(inplace=True, axis=0)
        self.val.dropna(inplace=True, axis=0)
        self.test.dropna(inplace=True, axis=0)
        print(f"""
            Train {self.train.shape},
            Val {self.val.shape},
            Test {self.test.shape},
            \n """)

    def scale_train_val_test(self, scaler, scale_prediction = False):
        """Scale datasets using transformer with fit_transform and
        transform method, avoid timestamp and target column"""

        if scale_prediction:
            col_position = 1
        else:
            col_position = 2

        self.train.iloc[:, col_position:] = scaler.fit_transform(self.train.iloc[:, col_position:])
        self.val.iloc[:, col_position:] = scaler.transform(self.val.iloc[:, col_position:])
        self.test.iloc[:, col_position:] = scaler.transform(self.test.iloc[:, col_position:])
        print('Scaling successful.')

    def load_data(self, device, drop_timestamp=True):
        """Data loader for NN"""
        dataloader_dict = dict()

        if drop_timestamp:
            self.train.drop(['Timestamp'], inplace=True, axis=1)
            self.val.drop(['Timestamp'], inplace=True, axis=1)
            self.test.drop(['Timestamp'], inplace=True, axis=1)

        for key, value in zip(['train', 'val', 'test'],
                              [self.train.values,
                               self.val.values,
                               self.test.values]):
            tensor_data = torch.from_numpy(value)
            Y = tensor_data[:, 0].view(-1, 1)
            X = tensor_data[:, 1:]
            dataloader_dict[key] = (X.to(device), Y.to(device))

        return dataloader_dict
    def load_ensemble_data(self, trainedmodels,device, drop_timestamp=True):
        """Data loader for NN"""
        dataloader_dict = dict()

        if drop_timestamp:
            self.train.drop(['Timestamp'], inplace=True, axis=1)
            self.val.drop(['Timestamp'], inplace=True, axis=1)
            self.test.drop(['Timestamp'], inplace=True, axis=1)

        for key, value in zip(['train', 'val', 'test'],
                              [self.train.values,
                               self.val.values,
                               self.test.values]):
            tensor_data = torch.from_numpy(value)
            Y = tensor_data[:, 0].view(-1, 1).to(device)
            X = tensor_data[:, 1:].to(device)
            predictions = list()
            for model in trainedmodels:
                predictions.append(model(x))
            newX = torch.cat(predictions,dim=1)
            dataloader_dict[key] = (newX, Y)

        return dataloader_dict

    # Method called on __init__
    def generate_lag(self, df, lags, columns):
        lag_df = pd.concat([self.df[columns].shift(lag)
                            for lag in lags],
                           axis=1)
        lag_df.columns = [f'lagged-{lag}-{col}' for col in columns
                          for lag in lags]
        print(f"Lag {lags} generated, size: {lag_df.shape}")

        return lag_df

    # Method called on __init__
    def generate_EMA(self, df, spans, columns):
        EMA_df = pd.concat([self.df[columns].ewm(span=span).mean()
                            for span in spans],
                           axis=1)
        EMA_df.columns = [f'EMA-{span}-{col}' for col in columns
                          for span in spans]
        print(f"EMA {spans} generated, size: {EMA_df.shape}")

        return EMA_df

    # Method called on __init__
    def generate_SMA(self, df, windows, columns):
        SMA_df = pd.concat([self.df[columns].rolling(window=window).mean()
                            for window in windows],
                           axis=1)
        SMA_df.columns = [f'SMA-{window}-{col}' for col in columns
                          for window in windows]
        print(f"SMA {windows} generated, size: {SMA_df.shape}")

        return SMA_df

    # Method called on __init__
    def generate_SMSTD(self, df, windows, columns):
        SMSTD_df = pd.concat([self.df[columns].rolling(window=window).std()
                              for window in windows],
                             axis=1)
        SMSTD_df.columns = [f'STD-{window}-{col}' for col in columns
                            for window in windows]
        print(f"EMA {windows} generated, size: {SMSTD_df.shape}")

        return SMSTD_df

    def generate_diffs(self, df, diffs, columns):
        diffs_df = pd.concat([self.df[columns].diff()
                            for diff in diffs],
                            axis = 1)
        diffs_df.columns = [f'diff-{diff}-{col}' for col in columns
                            for diff in diffs]
        print(f"Differencings {diffs} generated, size : {diffs_df.shape}")
    
    def generate_month_one_hot(self,df):
        month_df = pd.get_dummies(df.Timestamp.apply(lambda x: x.month))
        return month_df
    
    def generate_year_one_hot(self,df):
        year_df = pd.get_dummies(df.Timestamp.apply(lambda x: x.year))
        return year_df