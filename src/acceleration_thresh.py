#! /usr/bin/env python
#coding: utf-8

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm


def calc_thresh(value_lst, weight, window, min_value, max_value):
    # if past time series data does not exist, zero padding.
    pad_size = window - len(value_lst)
    if pad_size > 0:
        pad_lst = [0 for _ in range(pad_size)]
        value_lst = pad_lst + value_lst
    window_mean = weight * np.mean(value_lst[-window:])

    # min_value < thresh < max_value
    thresh = max(window_mean, min_value)
    thresh = min(thresh, max_value)

    return thresh


def acceleration_thresh(args):
    # get time series directory list under the root directory
    times_lst = os.listdir(args.root_stats_dirc)
    times_lst = [time for time in times_lst if os.path.isdir(args.root_stats_dirc + time)]

    for time in tqdm(times_lst):
        stats_lst = list(np.loadtxt(args.root_stats_dirc+time+"/"+args.stats_format))
        thresh_lst = []
        for i in range(int(len(stats_lst)/args.window)):
            start_index = i*args.window
            current_thresh = calc_thresh(stats_lst[:start_index], args.weight, args.window, args.min_value, args.max_value)
            window_thresh_lst = [current_thresh for _ in range(args.window)]
            thresh_lst.extend(window_thresh_lst)

        # terminal element processing
        remaining_num = len(stats_lst)%args.window
        if remaining_num != 0:
            current_thresh = calc_thresh(stats_lst[-remaining_num:], args.weight, args.window, args.min_value, args.max_value)
            window_thresh_lst = [current_thresh for _ in range(remaining_num)]
            thresh_lst.extend(window_thresh_lst)

        np.savetxt(args.root_stats_dirc + time +"/acc_thresh.csv", thresh_lst, delimiter=",")


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
                        default="/Users/sakka/cnn_anomaly_detection/data/statistics/20170421/")
    parser.add_argument("--stats_format", type=str,
                        default="max.csv", help="select from mean.csv, val.csv, max.csv")
   
    # Parameter Argumant
    parser.add_argument("--weight", type=float, default=2.5)
    parser.add_argument("--window", type=int, default=30*60)
    parser.add_argument("--min_value", type=int, default=100)
    parser.add_argument("--max_value", type=int, default=500)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = make_acceleration_parse()
    acceleration_thresh(args)
