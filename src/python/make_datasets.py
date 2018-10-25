#! /usr/bin/env python
#coding: utf-8

import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

logger = logging.getLogger(__name__)
logs_path = "/Users/sakka/cnn_anomaly_detection/logs/make_datasets.log"
logging.basicConfig(filename=logs_path,
                    level=logging.DEBUG,
                    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")


def make_datasets(args):
    # NEED FIX
    # company logo
    remove_last_frame = 120

    # initialize
    mean_lst = []
    val_lst = []
    max_lst = []
    thresh_lst = []
    human_lst = []
    grid_dictlst = {}
    for grid_y in range(args.grid_num[1]):
        for grid_x in range(args.grid_num[0]):
            grid_dictlst["grid_{}_{}".format(grid_y, grid_x)] = []

    # get several time series data
    for time_idx in range(args.start_time, args.end_time):
        # load current time series data
        tmp_mean_lst = list(np.loadtxt(args.root_statistics_dirc + args.day + "/{}/mean.csv".format(time_idx), delimiter=","))
        tmp_val_lst = list(np.loadtxt(args.root_statistics_dirc + args.day + "/{}/var.csv".format(time_idx), delimiter=","))
        tmp_max_lst = list(np.loadtxt(args.root_statistics_dirc + args.day + "/{}/max.csv".format(time_idx), delimiter=","))
        tmp_thresh_lst = list(np.loadtxt(args.root_statistics_dirc + args.day + "/{}/acc_thresh.csv".format(time_idx), delimiter=","))
        tmp_human_lst = list(np.loadtxt(args.root_human_dirc + args.day + "/human_{}.csv".format(time_idx), delimiter=","))

        # update time series data
        mean_lst.extend(tmp_mean_lst[:-remove_last_frame])
        val_lst.extend(tmp_val_lst[:-remove_last_frame])
        max_lst.extend(tmp_max_lst[:-remove_last_frame])
        thresh_lst.extend(tmp_thresh_lst[:-remove_last_frame])
        human_lst.extend(tmp_human_lst[:-remove_last_frame])

        tmp_grid_df = pd.read_csv(args.root_grid_count_dirc + args.day + "/{}.csv".format(time_idx))
        for grid_y in range(args.grid_num[1]):
            for grid_x in range(args.grid_num[0]):
                grid_dictlst["grid_{}_{}".format(grid_y, grid_x)].extend(list(tmp_grid_df["grid_{}_{}".format(grid_y, grid_x)]/tmp_grid_df["sum"]))


    feed_lst = list(np.loadtxt(args.root_feed_dirc + args.day + "/feed.csv", dtype="uint8", delimiter=","))
    diver_lst = list(np.loadtxt(args.root_diver_dirc + args.day + "/diver.csv", dtype="uint8", delimiter=","))

    # NEED FIX
    # completion
    for i in range(1, len(human_lst)):
        if human_lst[i] > 0.95:
            human_lst[i] = human_lst[i-1]

    
    # initialized datasets
    datasets_dictlst = {"mean": [], "val": [], "max": [], "thresh": [], "human": [],
                        "feed": [], "diver": [], "label": []}
    for grid_y in range(args.grid_num[1]):
            for grid_x in range(args.grid_num[0]):
                datasets_dictlst["grid_{}_{}".format(grid_y, grid_x)] = []
    
    # recode several time series data
    for i in range(0, len(mean_lst) - args.pred_frame, 30):
        datasets_dictlst["mean"].append(mean_lst[i])
        datasets_dictlst["val"].append(val_lst[i])
        datasets_dictlst["max"].append(max_lst[i])
        datasets_dictlst["thresh"].append(thresh_lst[i])
        datasets_dictlst["human"].append(human_lst[i])
        datasets_dictlst["feed"].append(feed_lst[i])
        datasets_dictlst["diver"].append(diver_lst[i])

        for grid_y in range(args.grid_num[1]):
            for grid_x in range(args.grid_num[0]):
                # NEED FIX
                datasets_dictlst["grid_{}_{}".format(grid_y, grid_x)] = grid_dictlst["grid_{}_{}".format(grid_y, grid_x)][int(i/30)]

        if max_lst[i + args.pred_frame] >= thresh_lst[i + args.pred_frame]:
            # anormal
            datasets_dictlst["label"].append(1)
        else:
            # normal
            datasets_dictlst["label"].append(0)

    datasets_df = pd.DataFrame(datasets_dictlst)
    logger.debug("DATA: {}, NORMAL: {}, ANORMAL: {}".format(\
                args.day, len(datasets_df[datasets_df["label"] == 0]), len(datasets_df[datasets_df["label"] == 1])))

    save_path = args.save_datasets_dirc + "time_series_{}.csv".format(args.day)
    datasets_df.to_csv(save_path, index=False)
    logger.debug("SAVE datasets: {}".format(save_path))



def datasets_parse():
    parser = argparse.ArgumentParser(
        prog="make_datasets.py",
        usage="",
        description="description",
        epilog="end",
        add_help=True
    )

    # Data Argument
    parser.add_argument("--day", type=str, default="20170421")
    parser.add_argument("--root_statistics_dirc", type=str,
                        default="/Users/sakka/cnn_anomaly_detection/data/statistics/")
    parser.add_argument("--root_human_dirc", type=str,
                        default="/Users/sakka/cnn_anomaly_detection/data/human_area/")
    parser.add_argument("--root_feed_dirc", type=str,
                        default="/Users/sakka/cnn_anomaly_detection/data/feed/")
    parser.add_argument("--root_diver_dirc", type=str,
                        default="/Users/sakka/cnn_anomaly_detection/data/diver/")
    parser.add_argument("--root_grid_count_dirc", type=str,
                        default="/Users/sakka/cnn_anomaly_detection/data/grid_count/")
    parser.add_argument("--save_datasets_dirc", type=str,
                        default="/Users/sakka/cnn_anomaly_detection/data/datasets/")

    # Parameter Argument
    parser.add_argument("--start_time", type=int, default=9)
    parser.add_argument("--end_time", type=int, default=17)
    parser.add_argument("--grid_num", type=tuple, default=(8, 1), help="(number of x axis, number of y axis)")
    parser.add_argument("--pred_frame", type=int, default=0*30, help="how many frame after anormaly is detected (time*FPS)")

    args = parser.parse_args()

    return args



if __name__ == "__main__":
    args = datasets_parse()
    make_datasets(args)
