import os
import time
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm

from models import DLinear
from utils.tools import plot_graph, EarlyStopping
from data_processing.data_process import data_provider

class exp_base():
    def __init__(self, args):
        self.args = args
        
        self.device = self._check_device()
        self.model = self._build_model().to(self.device)
    
    def _check_device(self):
        '''
        Function to check if GPU is available
        '''
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.devices) if self.args.use_multi_gpu else str(self.args.gpu)
            device = torch.device("cuda:{}".format(self.args.gpu))
            print("Use GPU: cuda:{}".format(self.args.gpu))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        
        return device
    
    def _build_model(self):
        '''
        Function to build model architecture
        '''
        model_dict = {
            "DLinear": DLinear
        }
        
        model = model_dict[self.args.model].Model(self.args).float()
        
        if self.args.use_gpu and self.args.use_multi_gpu:
            model = nn.DataParallel(model, device_ids = self.args.devices)
        
        return model
    
    def _get_data(self, flag):
        '''
        Function to load dataset and dataloader
        '''
        data_set, data_loader = data_provider(self.args, flag)
        
        return data_set, data_loader
    
    def _set_optimizer(self):
        '''
        Function to set optimizer for model training
        '''
        optimizer_dict = {
            "Adam": optim.Adam
        }
        
        optimizer = optimizer_dict[self.args.optimizer](self.model.parameters(), lr = self.args.learning_rate)
        
        return optimizer
    
    def _set_criterion(self):
        '''
        Function to set criterion for model training
        '''
        criterion_dict = {
            "mse": nn.MSELoss
        }
        
        criterion = criterion_dict[self.args.loss]()
        
        return criterion
    
    def train(self):
        '''
        Function to train model
        '''
        train_data, train_loader = self._get_data(flag = "train")
        if not self.args.only_train:
            valid_data, valid_loader = self._get_data(flag = "test")
            test_data, test_loader = self._get_data(flag = "test")
        
        path = os.path.join(self.args.checkpoints, self.args.test_id)
        if not os.path.exists(path):
            os.makedirs(path)
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience = self.args.patience, verbose = True)
        
        optimizer = self._set_optimizer()
        criterion = self._set_criterion()
        
        for epoch in range(self.args.epochs):
            # iter_count = 0
            train_loss = []
            
            self.model.train()

            epoch_time = time.time()
            for idx, (batch_x, batch_x_position, batch_y, batch_y_position) in enumerate(tqdm(train_loader, desc = "Train Process")):
                # iter_count += 1
                optimizer.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_x_position = batch_x_position.float().to(self.device)
                
                batch_y = batch_y.float().to(self.device)
                batch_y_position = batch_y_position.float().to(self.device)
                
                # decoder input
                dec_input = torch.zeros_like(batch_y[:, -self.args.len_pred:, :]).float()
                dec_input = torch.cat([batch_y[:, :self.args.len_label, :], dec_input], dim = 1).float().to(self.device)
                
                if "Linear" in self.args.model:
                    outputs = self.model(batch_x)
                
                outputs = outputs[:, -self.args.len_pred:, 0:]
                batch_y = batch_y[:, -self.args.len_pred:, 0:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                # if (idx + 1) % 100 == 0:
                #     print("iters: {0}, epoch: {1}, loss: {2:.7f}".format(idx + 1, epoch + 1, loss.item()))
                
                loss.backward()
                optimizer.step()
            
            avg_train_loss = np.average(train_loss)
            
            if not self.args.only_train:
                avg_valid_loss = self.valid(valid_data, valid_loader, criterion, process = "Valid Process")
                avg_test_loss = self.valid(test_data, test_loader, criterion, process = "Test Process")

                print()
                print("Epoch: {0}, Train Loss {1:.7f}, Validation Loss {2:.7f}, Test Loss {3:.7f}, Cost Time {4}".format(epoch + 1, avg_train_loss, avg_valid_loss, avg_test_loss, time.time() - epoch_time))
            else:
                print()
                print("Epoch: {0}, Train Loss {1:.7f}, Cost Time {2}".format(epoch + 1, avg_train_loss, time.time() - epoch_time))
                
            early_stopping(avg_valid_loss, self.model, path)
            
            print()
            
            if early_stopping.early_stop:
                print("Activate Early Stopping Epoch {0} ".format(epoch + 1))
                break

    def valid(self, valid_data, valid_loader, criterion, process): #!#
        valid_loss = []
        
        self.model.eval()
        
        with torch.no_grad():
            for idx, (batch_x, batch_x_position, batch_y, batch_y_position) in enumerate(tqdm(valid_loader, desc = process)):
                batch_x = batch_x.float().to(self.device)
                batch_x_position = batch_x_position.float().to(self.device)
                
                batch_y = batch_y.float().to(self.device)
                batch_y_position = batch_y_position.float().to(self.device)
                
                # decoder input
                dec_input = torch.zeros_like(batch_y[:, -self.args.len_pred:, :]).float()
                dec_input = torch.cat([batch_y[:, :self.args.len_label, :], dec_input], dim = 1).float().to(self.device)
                
                if "Linear" in self.args.model:
                    outputs = self.model(batch_x)
                
                outputs = outputs[:, -self.args.len_pred:, 0:]
                batch_y = batch_y[:, -self.args.len_pred:, 0:].to(self.device)
                
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                
                loss = criterion(pred, true)
                
                valid_loss.append(loss)
        
        avg_valid_loss = np.average(valid_loss)
        self.model.train()
                
        return avg_valid_loss