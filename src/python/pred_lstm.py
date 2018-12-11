#! /usr/bin/env python
#coding: utf-8

import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import argparse
import gc
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.autograd import Variable

logger = logging.getLogger(__name__)
logs_path = "/home/sakka/cnn_anomaly_detection/logs/lstm.log"
logging.basicConfig(filename=logs_path,
                    level=logging.DEBUG,
                    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")

class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_ratio=0.5):
        super(Predictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio
        self.output_dim = output_dim
        
        self.lstm = nn.LSTM(input_size=input_dim,
                           hidden_size=hidden_dim,
                           num_layers = num_layers,
                           batch_first=True)

        if dropout_ratio < 1:
            self.fc_dropout = nn.Dropout(dropout_ratio)
        else:
            self.fc_dropout = None

        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, input_feature, h0=None):
        lstm = self.lstm
        fc_dropout = self.fc_dropout
        fc_out = self.fc_out

        output, (h_n, c_n) = lstm(input_feature, h0)
        if fc_dropout is not None:
            output = fc_out(fc_dropout(output[:, -1, :]))
        else:
            output = fc_out(output[:,  -1, :])
        
        return output


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def scale(X_train, X_val, y_train, y_val):
    # scale inputs
    X_sclr = MinMaxScaler()
    X_train = X_sclr.fit_transform(X_train)
    X_val = X_sclr.transform(X_val)

    # scale labels
    y_sclr = MinMaxScaler()
    y_train = y_sclr.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
    y_val = y_sclr.transform(y_val.reshape(-1, 1)).reshape(-1)

    return X_train, X_val, y_train, y_val


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


def main(args):
    # starting info
    date = datetime.now()
    learning_date = "{0}_{1}_{2}_{3}_{4}".format(
        date.year, date.month, date.day, date.hour, date.minute)
    save_model_path = "{0}/model_{1}.pth".format(args.save_model_dirc, learning_date)

    # load dataset
    train_df = pd.read_csv(args.train_path)
    y_train = train_df["label"].as_matrix()
    X_train = train_df.drop("label", axis=1).as_matrix()

    val_df = pd.read_csv(args.val_path)
    y_val = val_df["label"].as_matrix()
    X_val = val_df.drop("label", axis=1).as_matrix()

    # train and val data applied MinMaxScaler 
    X_train, X_val, y_train, y_val = scale(X_train, X_val, y_train, y_val)
    logger.debug("X_train shape: {}".format(X_train.shape))
    logger.debug("y_train shape: {}".format(y_train.shape))
    logger.debug("X_val shape: {}".format(X_val.shape))
    logger.debug("y_val shape: {}".format(y_val.shape))

    # define model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug("DEVICE: {}".format(device))

    model = Predictor(args.input_dim, args.hidden_dim, args.num_layers, args.output_dim, args.dropout_ratio)
    model = nn.DataParallel(model).to(device)
    #model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    not_improved_count = 0
    best_epoch = 0

    train_n_batches = int((X_train.shape[0]-args.n_pred)/args.batch_size)
    val_n_batches = int((X_val.shape[0]-args.n_pred)/args.batch_size)

    train_loss_lst = []
    val_loss_lst = []
    for epoch in range(args.num_epochs):
        train_loss = 0.0
        val_loss = 0.0
        model.train()
        for train_idx in tqdm(range(train_n_batches)):
            optimizer.zero_grad()
            X, y = batch_data(X_train, y_train, train_idx, args.batch_size, n_prev=args.n_pred)
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
            X, y = batch_data(X_val, y_val, val_idx, args.batch_size)
            X = to_variable(torch.Tensor(X))
            y = to_variable(torch.Tensor(y))
                
            output = model(X)[:, 0]
            
            loss = criterion(output, y)
            val_loss += loss.item()
            del X, y, loss
            gc.collect()
            
        val_loss_lst.append(val_loss)
        
        # early stopping
        if (epoch > args.min_epoch) and (val_loss_lst[-1] > val_loss_lst[-2]):
            not_improved_count += 1
        else:
            # learning is going well
            not_improved_count = 0
            # save best params model
            best_epoch = epoch+1
            torch.save(model.state_dict(), save_model_path)

        logger.debug("EPOCH: {}, TRAIN LOSS: {}, VAL LOSS: {}, EARLY STOPPING: {}/{}".format(
                    epoch + 1, 
                    train_loss_lst[-1], 
                    val_loss_lst[-1],
                    not_improved_count,
                    args.stop_count))

        if not_improved_count == args.stop_count:
            logger.debug("Early Stopping")
            break

    logger.debug("Model saved in \"{0}\"".format(save_model_path))


def make_lstm_parse():
    parser = argparse.ArgumentParser(
        prog="pred_lstm.py",
        usage="train lstm",
        description="description",
        epilog="end",
        add_help=True
    )

    # Data Argument
    parser.add_argument("--train_path", type=str,
                        default="/home/sakka/cnn_anomaly_detection/data/datasets/gaussian/train.csv")
    parser.add_argument("--val_path", type=str,
                        default="/home/sakka/cnn_anomaly_detection/data/datasets/gaussian/val.csv")
    parser.add_argument("--save_model_dirc", type=str,
                        default="/home/sakka/cnn_anomaly_detection/data/model")


    # Parameter Argument
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--input_dim", type=int, default=67)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout_ratio", type=int, default=0.5)
    parser.add_argument("--output_dim", type=float, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--min_epoch", type=int, default=5)
    parser.add_argument("--stop_count", type=int, default=3)
    parser.add_argument("--n_pred", type=int, default=60*30)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = make_lstm_parse()
    logger.debug("Running with args: {0}".format(args))
    main(args)
