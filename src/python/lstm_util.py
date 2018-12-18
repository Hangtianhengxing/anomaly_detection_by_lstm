#! /usr/bin/env python
#coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler


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
                            num_layers=num_layers,
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

    return X_train, X_val, y_train, y_val, X_sclr, y_sclr


def batch_data(X, y, idx, batch_size, n_prev=60 * 10, pred_point=60 * 10):
    # return X: (batch_size, time_window, feature_dim)
    # return y: (batch_size)
    start_idx = idx * batch_size
    end_idx = start_idx + batch_size
    X_lst = []
    y_lst = []
    for i in range(start_idx, end_idx):
        X_lst.append(X[i:i + n_prev])
        # label of ahead pred_point
        y_lst.append(y[i + n_prev + pred_point - 1])

    return X_lst, y_lst


def pred_batch_data(X, idx, batch_size, n_prev=60*30):
    # return X: (batch_size, time_window, feature_dim)
    start_idx = idx*batch_size
    end_idx = start_idx+batch_size
    if end_idx+n_prev > X.shape[0]:
        end_idx = X.shape[0]-n_prev
    X_lst = []
    for i in range(start_idx, end_idx):
        X_lst.append(X[i:i+n_prev])

    return X_lst


def plot_pred(pred_arr, answer_arr, n_prev, pred_point, save_path):
    plt.figure(figsize=(12, 4))
    plt.rcParams["font.size"] = 14

    # plot data
    plt.plot(pred_arr, label="prediction")
    plt.plot(answer_arr, label="answer")
    plt.fill([0, n_prev+pred_point, n_prev+pred_point, 0], [0, 0, 5, 5], color="k", alpha=0.3, label="initial data")
    plt.xticks([i * 107900 / 30 for i in range(9)],
        ["9:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00"])

    # graph setting
    plt.xlim(0, len(pred_arr))
    plt.ylim(0.0, 1.0)
    plt.grid()
    plt.title("Predcition of anomaly level by using LSTM")
    plt.xlabel("time")
    plt.ylabel("anomaly level")
    plt.legend(loc="upper right")

    plt.savefig(save_path)


def plot_corr(pred_arr, answer_arr, n_prev, pred_point, save_path):
    plt.figure(figsize=(8, 8))
    plt.rcParams["font.size"]=14

    plt_pred = pred_arr[n_prev+pred_point:-pred_point+1]
    plt_ans = answer_arr[n_prev+pred_point:]
    coef = np.corrcoef(np.array([plt_ans, plt_pred]))[0][1]

    plt.scatter(plt_ans, plt_pred, marker="o",
                s=1, alpha=0.5, edgecolors="b")
    plt.title("Correlation coefficients = {:.4f}".format(coef))
    plt.xlabel("answer")
    plt.ylabel("prediction")
    plt.grid()

    plt.savefig(save_path)
