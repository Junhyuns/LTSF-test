import os
import json
import pandas as pd

from datetime import datetime
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class Dataset_Custom(Dataset):
    def __init__(self, config, split_proportion = [0.7, 0.1, 0.2], flag = "train"):
        self.config = config
        
        self.data_path = self.config.data_path      # path for data
        self.only_test = self.config.only_test       # flag for model train  [True: train and test, False: only test]
        self.scale = self.config.scale              # data scaling [True: data with scale, False: data without scale]
        
        self.data_size = split_proportion           # size for data split
        self.flag = flag                            # flag of model task
        
        # Setting model input and output length
        self.len_seq = self.config.len_seq
        self.len_pred = self.config.len_pred
        self.len_label = self.config.len_label
        
        # Check flag for typo
        assert self.flag in ["train", "valid", "test"], "You have entered an invalid flag"

        self.__read_data__()
    
    def __read_data__(self):
        # Convert date column to timestamp format
        def convert_timestamp(x):
            x = datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
            x = datetime.timestamp(x)
            
            return x
        
        self.tmp = pd.read_csv(self.data_path)
        self.total_data = self.tmp.drop(["date"], axis = 1)
        self.total_data_size = len(self.total_data)
        
        self.total_timestamp = self.tmp["date"].apply(convert_timestamp)
        
        # Load proportion of train, valid, and test dataset
        if self.data_size == None:
            self.train_data_size = 0.7
            self.valid_data_size = 0.2
            self.test_data_size = 0.1
        else:
            self.train_data_size = self.data_size[0]
            self.valid_data_size = self.data_size[1]
            self.test_data_size = self.data_size[2]
        
        # Split data if current mode is not only training
        if self.only_test == False:
            self.train_data = self.total_data.loc[:int(self.total_data_size * self.train_data_size) - 1, :]
            self.train_timestamp = self.total_timestamp.loc[:int(self.total_data_size * self.train_data_size) - 1].values
            
            self.valid_data = self.total_data.loc[int(self.total_data_size * self.train_data_size):int(self.total_data_size * (self.train_data_size + self.valid_data_size)) - 1, :]
            self.valid_timestamp = self.total_timestamp.loc[int(self.total_data_size * self.train_data_size):int(self.total_data_size * (self.train_data_size + self.valid_data_size)) - 1].values
            
            self.test_data = self.total_data.loc[int(self.total_data_size * (self.train_data_size + self.valid_data_size)):, :]
            self.test_timestamp = self.total_timestamp.loc[int(self.total_data_size * (self.train_data_size + self.valid_data_size)):].values
            
        elif self.only_test == True:
            self.test_data = self.total_data
        
        # Scale dataset if self.scale is True
        if self.scale == True:
            scaler = StandardScaler()
            
            scaler.fit(self.train_data)
            self.train_data = scaler.transform(self.train_data)
            self.valid_data = scaler.transform(self.valid_data)
            self.test_data = scaler.transform(self.test_data)
        else:
            self.train_data = self.train_data.values
            self.valid_data = self.valid_data.values
            self.test_data = self.test_data.values
        
        # Return dataset
        if self.flag == "train":
            self.features = self.train_data
            self.timestamp = self.train_timestamp
        elif self.flag == "valid":
            self.features = self.valid_data
            self.timestamp = self.valid_timestamp
        elif self.flag == "test":
            self.features = self.test_data
            self.timestamp = self.test_timestamp
    
    def __getitem__(self, idx): #!# check
        idx_begin_seq = idx                             # Index of start sequence
        idx_end_seq = idx_begin_seq + self.len_seq      # Index of last sequence

        idx_begin_pred = idx_end_seq                    # Index of start prediction
        idx_end_pred = idx_end_seq + self.len_pred      # Index of last prediction
        
        seq_features = self.features[idx_begin_seq:idx_end_seq]
        seq_features_stamp = self.timestamp[idx_begin_seq:idx_end_seq]
        
        seq_label = self.features[idx_begin_pred:idx_end_pred]
        seq_label_stamp = self.timestamp[idx_begin_pred:idx_end_pred]
        
        return seq_features, seq_features_stamp, seq_label, seq_label_stamp
    
    def __len__(self):
        return len(self.features) - (self.len_seq + self.len_pred) + 1