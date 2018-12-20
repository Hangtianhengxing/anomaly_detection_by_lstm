#! /usr/bin/env python
#coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

from evaluate import evaluate
from lstm_util import plot_metrics, plot_rec2prec


logger = logging.getLogger(__name__)
logs_path = "../../logs/search_thresh.log"
logging.basicConfig(filename=logs_path,
                    level=logging.DEBUG,
                    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")


def search_thresh(pred_arr, bin_arr, pred_point, invalid_time, save_dirc, sample_size=10000):
    thresh_lst = [i/sample_size for i in range(sample_size+1)]
    acc_lst = []
    pre_lst = []
    rec_lst = []
    f_lst = []

    for thresh in tqdm(thresh_lst):
        accuracy, precision, recall, f_measure = evaluate(
            pred_arr, bin_arr, pred_point, invalid_time, thresh)
        acc_lst.append(accuracy)
        pre_lst.append(precision)
        rec_lst.append(recall)
        f_lst.append(f_measure)

    fig_save_path = "{0}/metrics.png".format(save_dirc)
    plot_metrics(acc_lst, pre_lst, rec_lst, f_lst, sample_size, fig_save_path)
    logger.debug("Save figure of metrics result in \"{0}\"".format(fig_save_path))

    best_idx = np.argmax(f_lst)
    rec2prec_save_path = "{0}/rec2prec.png".format(save_dirc)
    plot_rec2prec(rec_lst, pre_lst, best_idx, rec2prec_save_path)
    logger.debug("Save figure of recall vs precision in \"{0}\"".format(rec2prec_save_path))

    logger.debug("**************************************************")
    logger.debug("Best Threshold : {0}".format(thresh_lst[best_idx]))
    logger.debug("Best Accuracy  : {0}".format(acc_lst[best_idx]))
    logger.debug("Best Precision : {0}".format(pre_lst[best_idx]))
    logger.debug("Best Recall    : {0}".format(rec_lst[best_idx]))
    logger.debug("Best F-measure : {0}".format(f_lst[best_idx]))
    logger.debug("**************************************************")


if __name__ == "__main__":
    # prediction
    pred_path="../../data/prediction/value_20181220_1207.csv"
    logger.debug("Prediction data path: \"{0}\"".format(pred_path))
    pred_arr=np.loadtxt(pred_path)

    # binary answer label
    ans_path="../../data/datasets/test_label.csv"
    logger.debug("Answer data path: \"{0}\"".format(ans_path))
    bin_arr=np.loadtxt(ans_path)

    # seach best threshold
    n_prev = 60*30
    pred_point = 60*10
    invalid_time = n_prev+pred_point
    save_dirc = "../../data/prediction"
    search_thresh(pred_arr, bin_arr, pred_point, invalid_time, save_dirc, sample_size=10000)

