#! /usr/bin/env python
#coding: utf-8

import numpy as np
import pandas as pd
from tqdm import tqdm
import gc

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.autograd import Variable

class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, output_dim):
        super(Predictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_dim = output_dim
        
        self.lstm = nn.LSTM(input_size=input_dim,
                           hidden_size=hidden_dim,
                           num_layers = num_layers,
                           dropout=dropout,
                           batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, input_feature, h0=None):
        output, (h_n, c_n) = self.lstm(input_feature, h0)
        output = self.fc_out(output[:,  -1, :])
        
        return output


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def batch_data(X, y, idx, batch_size, n_prev=60*10):
    # return X: (batch_size, time_window, feature_dim)
    # return y: (batch_size)
    start_idx = idx*batch_size
    end_idx = start_idx + batch_size
    X_lst = []
    y_lst = []
    for i in range(start_idx, end_idx):
        X_lst.append(X[i:i+n_prev])
        y_lst.append(y[i+n_prev])
    return X_lst, y_lst


def main():
    # load dataset
    train_data_path = "/home/sakka/cnn_anomaly_detection/data/datasets/gaussian/20170416.csv"
    train_df = pd.read_csv(train_data_path)
    X_train = train_df.as_matrix()
    y_train = train_df["label"].as_matrix()
    print("X shape: {}".format(X_train.shape))
    print("y shape: {}".format(y_train.shape))

    val_data_path = "/home/sakka/cnn_anomaly_detection/data/datasets/gaussian/20170418.csv"
    val_df = pd.read_csv(val_data_path)
    X_val = val_df.as_matrix()
    y_val = val_df["label"].as_matrix()
    print("X_val shape: {}".format(X_val.shape))
    print("y_val shape: {}".format(y_val.shape))

    # define model
    num_epochs = 100
    input_dim = 65
    hidden_dim = 1024
    num_layers = 2
    dropout = 0.5
    output_dim = 1
    lr = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE: {}".format(device))

    model = Predictor(input_dim, hidden_dim, num_layers, dropout, output_dim)
    #model = nn.DataParallel(model).to(device)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    batch_size = 32
    not_improved_count = 0
    best_epoch = 0
    min_epoch = 5
    stop_count = 3
    n_pred = 60*10
    save_model_path = "/home/sakka/cnn_anomaly_detection/data/model/model.pth"

    train_n_batches = int((X_train.shape[0]-n_pred)/batch_size)
    val_n_batches = int((X_val.shape[0]-n_pred)/batch_size)

    train_loss_lst = []
    val_loss_lst = []
    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0
        model.train()
        for train_idx in tqdm(range(train_n_batches)):
            optimizer.zero_grad()
            X, y = batch_data(X_train, y_train, train_idx, batch_size, n_prev=n_pred)
            X = to_variable(torch.Tensor(X))
            y = to_variable(torch.Tensor(y))
            output = model(X)[:, 0]
            
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()  
            train_loss += loss.item()
            del X, y, loss
            gc.collect()
        
        train_loss_lst.append(train_loss)
            
        model.eval()
        for val_idx in tqdm(range(val_n_batches)):
            X, y = batch_data(X_val, y_val, val_idx, batch_size)
            X = to_variable(torch.Tensor(X))
            y = to_variable(torch.Tensor(y))
                
            output = model(X)[:, 0]
            
            loss = criterion(output, y)
            val_loss += loss.item()
            del X, y, loss
            gc.collect()
            
        val_loss_lst.append(val_loss)
        
        # early stopping
        if (epoch > min_epoch) and (val_loss_lst[-1] > val_loss_lst[-2]):
            not_improved_count += 1
        else:
            # learning is going well
            not_improved_count = 0
            # save best params model
            best_epoch = epoch+1
            torch.save(model.state_dict(), save_model_path)

        print("EPOCH: {}, TRAIN LOSS: {}, VAL LOSS: {}, EARLY STOPPING: {}/{}".format(
                    epoch + 1, 
                    train_loss_lst[-1], 
                    val_loss_lst[-1],
                    not_improved_count,
                    stop_count))

        if not_improved_count == stop_count:
            print("Early Stopping")
            break

if __name__ == "__main__":
    main()
