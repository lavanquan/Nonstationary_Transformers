import torch 
from torch.utils.data import DataLoader, Dataset
import os 
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def windowing_data(time_series_data, window_size=24):
    '''
    Inputs:
        + time_series_data (numpy.array): input time series data with dimensions (num_time_series, time_steps)
        + window_size (int): window to slide over time series
    Outputs:
        + X_train (numpy.array): (n_samples, window_size - 1)
        + Y_train (numpy.array): (n_samples, 1)
    Example of usage:
        X_train, Y_train = windowing_data(time_series_data=train_data, window_size=24)
        X_test, Y_test = windowing_data(time_series_data=test_data, window_size=24)

        print(f"X_train.shape: {X_train.shape}, Y_train.shape: {Y_train.shape}")
        print(f"X_test.shape: {X_test.shape}, Y_test.shape: {Y_test.shape}")
    '''
    x_data = []
    y_data = []
    for i in range(time_series_data.shape[1] - window_size + 1):
        window_data = time_series_data[:, i:i+window_size]
        data_x = window_data[:, : window_size - 1]
        data_y = window_data[:, -1]
        x_data.append(data_x)
        y_data.append(data_y)
    '''
        x_data: (time_series_data.shape[1] - window_size + 1, n_time_series, window_size - 1)
        y_data: (time_series_data.shape[1] - window_size + 1, n_time_series)
    '''
    X_train = np.array(x_data)
    Y_train = np.array(y_data)
    X_train = X_train.reshape(-1, X_train.shape[2])
    Y_train = Y_train.flatten()
    '''
        X_train: (n_samples, window_size - 1)
        Y_train: (n_sample, 1)
    '''
    return X_train, Y_train

def windowing_muloutput(time_series_data, x_window_size, y_window_size):
    x_data = []
    y_data = []
    frame_length = x_window_size + y_window_size

    for i in range(time_series_data.shape[1] - frame_length + 1):
        window_data = time_series_data[:, i:i+frame_length]
        data_x = window_data[:, :x_window_size]
        data_y = window_data[:, -y_window_size:]
        x_data.append(data_x)
        y_data.append(data_y)
    X_train = np.array(x_data)
    Y_train = np.array(y_data)
    X_train = X_train.reshape(-1, X_train.shape[2])
    Y_train = Y_train.reshape(-1, Y_train.shape[2])
    
    return X_train, Y_train

class TimeDataset(Dataset):
    '''
    Create Dataset from:
        + features: (n_samples, window_size - 1)
        + targets: (n_sample, 1)
    '''
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def download_elec_dataset():
    file = "LD2011_2014_clean.txt"
    isExist = os.path.exists(file)
    if not isExist:
        os.system('wget https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip')
        os.system('unzip LD2011_2014.txt.zip')
        # change commas to dots
        os.system('sed \'s/,/./g\' LD2011_2014.txt > LD2011_2014_clean.txt')
        # remove unused files
        os.system('rm -rf LD2011_2014.txt.zip')
        os.system('rm -rf LD2011_2014.txt')
    else: 
        print(f"Files are ready")
    
def clean_elec():
    """Preprocess data"""
    data = pd.read_csv('LD2011_2014_clean.txt', delimiter = ';')
    #remove data before 2012
    data = data.iloc[8760*4:]
    print('Data loaded..')
    data_2 = data.copy()
    #pick the first 20 houses
    data_2 = data_2.iloc[:,:]
    # Aggregate
    data_2['time'] =pd.to_datetime(data_2['Unnamed: 0']).dt.ceil('1h') 
    data_2 = data_2.drop(['Unnamed: 0'], axis = 1)
    agg_dict = {}
    for col in data_2.columns[:-1]:
        agg_dict[col] ='mean'
    aggregated_data = data_2.groupby(['time']).agg(agg_dict)
    # aggregated_data = aggregated_data.to_numpy()
    print(f'Data aggregated by hour: {aggregated_data.shape}')
    return aggregated_data

def set_train_test(data, test_day=2, start_ts=0, stop_ts=370):
    num_train_samples = data.shape[0] - test_day*24
    data_train = data[:num_train_samples, start_ts : stop_ts]
    data_test = data[num_train_samples:, start_ts : stop_ts]
    data_train = data_train.T 
    data_test = data_test.T
    return data_train, data_test

def download_traffic_dataset():
    file = "traffic.txt"
    isExist = os.path.exists(file)

    if not isExist:
        print(f"Files not exist")
    else:
        print("File is ready")

def clean_traffic(num_ts=860):
    """Preprocess data"""
    data = pd.read_csv('traffic.txt', delimiter = ',', header=None)
    print('data loaded..')
    data_2 = data.copy()
    #pick the first 20 clients
    data_2 = data_2.iloc[:,:num_ts]
    #create time column: 2 years 1 hour
    data_2['time'] = pd.to_datetime(np.arange(datetime(2015,1,1), datetime(2017,1,1), timedelta(hours=1)))
    data_2.index = data_2['time']
    data_2 = data_2.drop(['time'], axis = 1)
    #create column names
    data_3 = data_2.copy()
    col_names = ['MT_{0:03}'.format(i+1) for i in range(data_3.shape[1])]
    data_3.columns = col_names
    aggregated_data = data_3.copy()
    return aggregated_data

from sklearn.preprocessing import StandardScaler, MinMaxScaler
def preprocess(data, std=True):
    '''
    Function to do preprocess for input data
    inputs:
        + data: pandas.DataFrame
    outputs:
        + preprocessed_data: pandas.DataFrame
    '''
    if std:
        scaler = StandardScaler()
        temp = scaler.fit_transform(data)
    else:
        scaler = MinMaxScaler()
        temp = scaler.fit_transform(data)
    processed_data = pd.DataFrame(temp, index=data.index, columns = data.columns)
    processed_data = processed_data.to_numpy()
    return processed_data

from datasets import load_dataset

def get_dataset(dataset_name, test_day=2):
    if dataset_name == "Elec":
        # 1. Download the dataset
        download_elec_dataset()
        elec_data = clean_elec()
        # 2. Preprocess dataset
        processed_data = preprocess(elec_data)
        # 3. Get train dataset
        data_train, _ = set_train_test(processed_data, test_day=test_day, start_ts=0, stop_ts=370)
    elif dataset_name == "Traff":
        # 1. Download the dataset
        download_traffic_dataset()
        traff_data = clean_traffic()
        # 2. Clean dataset
        processed_data = preprocess(traff_data)
        # 3. Get train dataset
        data_train, _ = set_train_test(processed_data, test_day=test_day, start_ts=0, stop_ts=860)
    elif dataset_name == "Kdd":
        dataset = load_dataset("LeoTungAnh/kdd210_hourly")
        train_dataset = dataset['validation']
        train_data_list = []
        for data_row in train_dataset:
            train_data_list.append(data_row['target'])
        data_train = np.array(train_data_list)
        return data_train
    else:
        print("The dataset is not supported")
        return 0
    return data_train

def divide_data_to_client(data_train, client_id, num_data_per_client):
    # Divide train dataset to user
    client_data = data_train[(client_id)*num_data_per_client: (client_id+1)*num_data_per_client, :]
    print(f"client {client_id}: {client_data.shape}")
    return client_data[:, 0:-1], client_data[:, 1:]

