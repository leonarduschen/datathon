import pandas as pd

df = pd.read_csv('.\\rawdata\\consolidated_autocaffe_data.csv')
cols_to_shift = [col for col in df.columns if col not in ['Timestamp','Energy']]
df[cols_to_shift] = df[cols_to_shift].shift(18)
df.dropna(axis = 0, inplace = True)

df.to_csv('.\\rawdata\\consolidated_autocaffe_data_shifted.csv')