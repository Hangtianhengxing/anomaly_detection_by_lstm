#! /usr/bin/env python
#coding: utf-8

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm


def calc_thresh(value_lst, weight=3, window=5, min_value=200):
    # if past time series data does not exist, zero padding.
    pad_size = window - len(value_lst)
    if pad_size > 0:
        pad_lst = [0 for _ in range(pad_size)]
        value_lst = pad_lst + value_lst
    window_mean = weight * np.mean(value_lst[-window:])

    return min(window_mean, min_value)


def acceleration_thresh(args):
    # get time series directory list under the root directory
    times_lst = os.listdir(args.root_stats_dirc)
    times_lst = [time for time in times_lst if os.path.isdir(args.root_stats_dirc + time)]

    for time in tqdm(times_lst):
        stats_lst = list(np.loadtxt(args.root_stats_dirc+time+"/"+args.stats_format))
        thresh_lst = []
        for frame_index in range(len(stats_lst)):
            current_thresh = calc_thresh(stats_lst[:frame_index])
            thresh_lst.append(current_thresh)
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

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = make_acceleration_parse()
    acceleration_thresh(args)
