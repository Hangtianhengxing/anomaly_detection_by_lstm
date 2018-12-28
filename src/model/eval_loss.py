#! /usr/bin/env python
#coding: utf-8

import numpy as np
import pandas as pd

from lstm_util import scale


def pred_loss(train_path, ans_path, pred_path, invalid_time=2400):
    train_df = pd.read_csv(train_path)
    y_train = train_df["label"].as_matrix()
    X_train = train_df.as_matrix()

    ans_df = pd.read_csv(ans_path)
    y_ans = ans_df["label"].as_matrix()
    X_ans = ans_df.as_matrix()
    
    # applied MinMaxScaler
    _, _, _, y_ans, _, _ = scale(X_train, X_ans, y_train, y_ans)

    # pred date is already scaled
    pred_arr = np.loadtxt(pred_path)

    # remove invalid time
    data_num = min(len(pred_arr), len(y_ans))
    pred_arr = pred_arr[invalid_time:data_num]
    y_ans = y_ans[invalid_time:data_num]
    assert len(pred_arr) == len(y_ans)

    # Mean Squared Error
    loss = np.sum(np.abs(pred_arr - y_ans)**2)

    print("Loss is {0}".format(loss))


if __name__ == "__main__":
    # train data using to scale dataset
    train_path = "/Users/sakka/cnn_anomaly_detection/data/datasets/datasets/train.csv"
    ans_path = "/Users/sakka/cnn_anomaly_detection/data/datasets/datasets/test.csv"
    pred_path = "/Users/sakka/cnn_anomaly_detection/data/prediction/default/value_20181220_1207.csv"
    invalid_time = 2400
    pred_loss(train_path, ans_path, pred_path, invalid_time)
