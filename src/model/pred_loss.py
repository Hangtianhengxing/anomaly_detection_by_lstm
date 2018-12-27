#! /usr/bin/env python
#coding: utf-8

import numpy as np


def pred_loss(pred_path, ans_path, invalid_time=2400):
    pred_arr = np.loadtxt(pred_path)
    ans_arr = np.loadtxt(ans_path)

    # remove invalid time
    data_num = min(len(pred_arr), len(ans_arr))
    pred_arr = pred_arr[invalid_time:data_num]
    ans_arr = ans_arr[invalid_time:data_num]
    assert len(pred_arr) == len(ans_arr)

    # Mean Squared Error
    loss = np.sum(np.abs(pred_arr - ans_arr)**2)

    print("Loss is {0}".format(loss))


if __name__ == "__main__":
    pred_path = "/Users/sakka/cnn_anomaly_detection/data/prediction/default/value_20181220_1207.csv"
    ans_path = "/Users/sakka/cnn_anomaly_detection/data/prediction/test_label.csv"
    invalid_time = 2400
    pred_loss(pred_path, ans_path, invalid_time)
