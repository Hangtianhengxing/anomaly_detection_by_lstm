#! /usr/bin/env python
#coding: utf-8

import os
import logging
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

logger = logging.getLogger(__name__)
logs_path = "/Users/sakka/cnn_anomaly_detection/logs/acceleration_thresh.log"
logging.basicConfig(filename=logs_path,
                    level=logging.DEBUG,
                    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")


def calc_thresh(value_lst, weight, window, min_value, max_value, prev_thresh):
    """
    calculate the acceleration threshold.
    thresh = weight*(mean of value_lst)
    condition: min_value < thresh < max_value
    input:
        value_lst: value list of time series
        weight: weight multiplied mean value
        window: number of past frames to consider
        min_value: minimum of threshold
        max_value: maximum of threshold
    """

    # if past time series data does not exist, zero padding.
    pad_size = window - len(value_lst)
    if pad_size > 0:
        pad_lst = [0 for _ in range(pad_size)]
        value_lst = pad_lst + value_lst
    thresh = weight * np.mean(value_lst[-window:])

    # condition
    if (thresh < min_value) or (max_value < thresh):
        thresh = prev_thresh

    return thresh


def acceleration_thresh(args):
    """
    calculate threshold by statistics value (mean, val, max) of time series 
    under the directory (args.root_stats_dirc).
    threshold data of several directory is saved in following path (input directory path + acc_thresh.csv).
    """
    
    # get time series directory list under the root directory
    times_lst = os.listdir("{0}/".format(args.root_stats_dirc))
    times_lst = [time for time in times_lst if os.path.isdir("{0}/{1}".format(args.root_stats_dirc, time))]

    # when the condition is NOT satisfied, the threshold use before one step
    prev_thresh = args.min_value

    for time in tqdm(times_lst):
        stats_df = pd.read_csv("{0}/{1}/{2}.csv".format(args.root_stats_dirc, time, args.stats_format))
        stats_lst = list(stats_df[args.stats_format])
        thresh_dctlst = {"frame_num":list(stats_df["frame_num"]), "acc_thresh":[]}
        for i in range(int(len(stats_lst)/args.window)):
            start_index = i*args.window
            current_thresh = calc_thresh(stats_lst[:start_index], args.weight, args.window, args.min_value, args.max_value, prev_thresh)
            window_thresh_lst = [current_thresh for _ in range(args.window)]
            thresh_dctlst["acc_thresh"].extend(window_thresh_lst)
            prev_thresh = current_thresh

        # terminal element processing
        remaining_num = len(stats_lst)%args.window
        if remaining_num != 0:
            current_thresh = calc_thresh(stats_lst[-remaining_num:], args.weight, args.window, args.min_value, args.max_value, prev_thresh)
            window_thresh_lst = [current_thresh for _ in range(remaining_num)]
            thresh_dctlst["acc_thresh"].extend(window_thresh_lst)

        save_path = "{0}/{1}/acc_thresh.csv".format(args.root_stats_dirc, time)
        pd.DataFrame(thresh_dctlst).to_csv(save_path, index=False)
        logger.debug("saved in {0}".format(save_path))


def make_acceleration_parse():
    parser = argparse.ArgumentParser(
        prog="acceleration_thresh.py",
        usage="calculate the threshold of acceleration in each frame.",
        description="description",
        epilog="end",
        add_help=True
    )

    # Data Argument
    parser.add_argument("--root_stats_dirc", type=str,
                        default="/Users/sakka/cnn_anomaly_detection/data/statistics/20170416")
    parser.add_argument("--stats_format", type=str,
                        default="max", help="select from mean, val, max")
   
    # Parameter Argumant
    parser.add_argument("--weight", type=float, default=2.4)
    parser.add_argument("--window", type=int, default=30*60*5)
    parser.add_argument("--min_value", type=int, default=150)
    parser.add_argument("--max_value", type=int, default=400)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = make_acceleration_parse()
    logger.debug("Running with args: {0}".format(args))
    acceleration_thresh(args)