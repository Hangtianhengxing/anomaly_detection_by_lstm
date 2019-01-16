#! /usr/bin/env python
#coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging


def eval_metrics(tp_cnt, fp_cnt, tn_cnt, fn_cnt):
    if (tp_cnt + tn_cnt + fp_cnt + fn_cnt) != 0:
        accuracy = (tp_cnt + tn_cnt) / (tp_cnt + tn_cnt + fp_cnt + fn_cnt)
    else:
        accuracy = 0

    if (tp_cnt + fp_cnt) != 0:
        precision = (tp_cnt) / (tp_cnt + fp_cnt)
    else:
        precision = 0

    if (tp_cnt + fn_cnt) != 0:
        recall = (tp_cnt) / (tp_cnt + fn_cnt)
    else:
        recall = 0

    if (recall + precision) != 0:
        f_measure = (2 * precision * recall) / (recall + precision)
    else:
        f_measure = 0

    logger.debug("Accuracy  : {0}".format(accuracy))
    logger.debug("Precision : {0}".format(precision))
    logger.debug("Recall    : {0}".format(recall))
    logger.debug("F-measure : {0}".format(f_measure))

    return accuracy, precision, recall, f_measure


def evaluate(pred_arr, bin_ans_arr, pred_point, invalid_time, thresh):
    assert len(pred_arr) == (len(bin_ans_arr))
    valid_pred_arr = pred_arr.copy()[invalid_time:]
    valid_bin_arr = bin_ans_arr.copy()[invalid_time:]

    pred_idx = np.where(valid_pred_arr > thresh)[0]
    logger.debug("Number of alert: {}".format(len(pred_idx)))

    tp_cnt = 0
    fp_cnt = 0
    tn_cnt = 0
    fn_cnt = 0
    for i in range(len(valid_pred_arr)):
        if (valid_pred_arr[i] >= thresh) & (sum(valid_bin_arr[i:i + pred_point]) > 0):
            tp_cnt += 1
        elif (valid_pred_arr[i] >= thresh) & (sum(valid_bin_arr[i:i + pred_point]) == 0):
            fp_cnt += 1
        elif (valid_pred_arr[i] <= thresh) & (sum(valid_bin_arr[i:i + pred_point]) == 0):
            tn_cnt += 1
        elif (valid_pred_arr[i] <= thresh) & (sum(valid_bin_arr[i:i + pred_point]) > 0):
            fn_cnt += 1

    logger.debug("TP : {0}".format(tp_cnt))
    logger.debug("FP : {0}".format(fp_cnt))
    logger.debug("TN : {0}".format(tn_cnt))
    logger.debug("FN : {0}".format(fn_cnt))

    accuracy, precision, recall, f_measure = eval_metrics(
        tp_cnt, fp_cnt, tn_cnt, fn_cnt)

    return accuracy, precision, recall, f_measure


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logs_path = "../../logs/evaluate_control.log"
    logging.basicConfig(filename=logs_path,
                        level=logging.DEBUG,
                        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")

    # prediction
    pred_path = "../../data/prediction/default/value_20190107_244.csv"
    #pred_path = "../../data/prediction/default/control_evening.csv"
    logger.debug("Prediction data path: \"{0}\"".format(pred_path))
    pred_arr = np.loadtxt(pred_path)

    # binary answer label
    ans_path = "../../data/datasets/test_label.csv"
    logger.debug("Answer data path: \"{0}\"".format(ans_path))
    bin_arr = np.loadtxt(ans_path)

    pred_arr = pred_arr[:len(bin_arr)]

    # evaluate
    n_prev = 60*30
    pred_point = 60*10
    invalid_time = n_prev+pred_point
    thresh = 0.5
    logger.debug("n_prev: {0}, pred_point: {1}, invalid_time: {2}, thresh: {3}".format(\
            n_prev, pred_point, invalid_time, thresh))
    accuracy, precision, recall, f_measure = evaluate(
        pred_arr, bin_arr, pred_point, invalid_time, thresh)
