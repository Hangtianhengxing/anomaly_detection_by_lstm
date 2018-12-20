#! /usr/bin/env python
#coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging


def eval_metrics(tp_cnt, fp_cnt, fn_cnt):
    if (tp_cnt+fp_cnt+fn_cnt) != 0:
        accuracy = (tp_cnt)/(tp_cnt+fp_cnt+fn_cnt)
    else:
        accuracy = 0

    if (tp_cnt+fp_cnt) != 0:
        precision = (tp_cnt)/(tp_cnt+fp_cnt)
    else:
        precision = 0

    if (tp_cnt+fn_cnt) != 0:
        recall = (tp_cnt)/(tp_cnt+fn_cnt)
    else:
        recall = 0

    if (recall+precision) != 0:
        f_measure = (2*precision*recall)/(recall+precision)
    else:
        f_measure = 0

    # logger.debug("Accuracy  : {0}".format(accuracy))
    # logger.debug("Precision : {0}".format(precision))
    # logger.debug("Recall    : {0}".format(recall))
    # logger.debug("F-measure : {0}".format(f_measure))

    return accuracy, precision, recall, f_measure


def evaluate(pred_arr, bin_ans_arr, pred_point, invalid_time, thresh):
    assert len(pred_arr) == (len(bin_ans_arr) + pred_point)
    valid_pred_arr = pred_arr.copy()[invalid_time:]
    valid_bin_arr = bin_ans_arr.copy()[invalid_time:]

    pred_idx = np.where(valid_pred_arr > thresh)[0]
    tp_cnt, fp_cnt = 0, 0
    for end_idx in pred_idx:
        start_idx = max(0, end_idx-pred_point)
        if np.sum(valid_bin_arr[start_idx:end_idx]) > 0:
            tp_cnt += 1
        else:
            fp_cnt += 1

    ans_idx = np.where(valid_bin_arr == 1)[0]
    fn_cnt = 0
    for end_idx in ans_idx:
        start_idx = max(0, end_idx-pred_point)
        detect = np.where((start_idx <= pred_idx) & (pred_idx <= end_idx))[0]
        if len(detect) == 0:
            fn_cnt += 1

    # logger.debug("TP : {0}".format(tp_cnt))
    # logger.debug("FP : {0}".format(fp_cnt))
    # logger.debug("FN : {0}".format(fn_cnt))

    accuracy, precision, recall, f_measure = eval_metrics(
        tp_cnt, fp_cnt, fn_cnt)

    return accuracy, precision, recall, f_measure


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logs_path = "../../logs/evaluate.log"
    logging.basicConfig(filename=logs_path,
                        level=logging.DEBUG,
                        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")

    # prediction
    pred_path = "../../data/prediction/value_20181220_1207.csv"
    logger.debug("Prediction data path: \"{0}\"".format(pred_path))
    pred_arr = np.loadtxt(pred_path)

    # binary answer label
    ans_path = "../../data/datasets/test_label.csv"
    logger.debug("Answer data path: \"{0}\"".format(ans_path))
    bin_arr = np.loadtxt(ans_path)

    # evaluate
    n_prev = 60*30
    pred_point = 60*10
    invalid_time = n_prev+pred_point
    thresh = 0.5
    logger.debug("n_prev: {0}, pred_point: {1}, invalid_time: {2}, thresh: {3}".format(\
            n_prev, pred_point, invalid_time, thresh))
    accuracy, precision, recall, f_measure = evaluate(
        pred_arr, bin_arr, pred_point, invalid_time, thresh)
