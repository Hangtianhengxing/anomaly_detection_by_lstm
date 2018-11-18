#! /usr/bin/env python
#coding: utf-8

import os
import logging
import numpy as np
import pandas as pd
import argparse

from tqdm import tqdm

logger = logging.getLogger(__name__)
logs_path = "/Users/sakka/cnn_anomaly_detection/logs/make_datasets.log"
logging.basicConfig(filename=logs_path,
                    level=logging.DEBUG,
                    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")


def normalize(df, col_lst):
    result_df = df.copy()
    for col in col_lst:
        max_value = df[col].max()
        min_value = df[col].min()
        result_df[col] = (df[col] - min_value) / (max_value - min_value)

    return result_df


def make_datasets(args):
    # get time series directory list under the root directory
    times_lst = os.listdir("{0}/{1}/".format(args.root_stats_dirc, args.date))
    times_lst = [time for time in times_lst if os.path.isdir("{0}/{1}/{2}/".format(args.root_stats_dirc, args.date, time))]

    day_lst = ["Sun", "Mon", "Tue", "Wed", "Thurs", "Fri", "Sat"]
    time_lst = [str(i) for i in range(9, 17)]

    for time_idx in tqdm(times_lst):
        mean_df = pd.read_csv("{0}/{1}/{2}/mean.csv".format(args.root_stats_dirc, args.date, time_idx))
        var_df = pd.read_csv("{0}/{1}/{2}/var.csv".format(args.root_stats_dirc, args.date, time_idx))
        max_df = pd.read_csv("{0}/{1}/{2}/max.csv".format(args.root_stats_dirc, args.date, time_idx))
        thresh_df = pd.read_csv("{0}/{1}/{2}/acc_thresh.csv".format(args.root_stats_dirc, args.date, time_idx))
        degree_df = pd.read_csv("{0}/{1}/{2}/prep_degree.csv".format(args.root_stats_dirc, args.date, time_idx))
        degree_df = pd.get_dummies(degree_df)
        grid_df = pd.read_csv("{0}/{1}/{2}.csv".format(args.root_grid_dirc, args.date, time_idx))
        grid_df = grid_df.drop(["max", "raw_data"], axis=1)
        grid_df = pd.get_dummies(grid_df)
        human_df = pd.read_csv("{0}/{1}/{2}.csv".format(args.root_human_dirc, args.date, time_idx))
        diver_df = pd.read_csv("{0}/{1}/diver_{2}.csv".format(args.root_diver_dirc, args.date, time_idx))
        feed_df = pd.read_csv("{0}/{1}/feed_{2}.csv".format(args.root_feed_dirc, args.date, time_idx))

        # concat ttime series data
        time_series_df = mean_df
        time_series_df = pd.merge(time_series_df, var_df,  on="frame_num")
        time_series_df = pd.merge(time_series_df, max_df,  on="frame_num")
        time_series_df = pd.merge(time_series_df, thresh_df,  on="frame_num")
        time_series_df = pd.merge(time_series_df, degree_df,  on="frame_num")
        time_series_df = pd.merge(time_series_df, grid_df,  on="frame_num")
        time_series_df = pd.merge(time_series_df, human_df,  on="frame_num")
        time_series_df = pd.merge(time_series_df, diver_df, on="frame_num")
        time_series_df = pd.merge(time_series_df, feed_df, on="frame_num")
        time_series_df = time_series_df.drop("frame_num", axis=1)

        # make answer label
        time_series_df["label"] = pd.Series(
            time_series_df["max"] > time_series_df["acc_thresh"], dtype=int)
        time_series_df = time_series_df.drop("acc_thresh", axis=1)
        for end_idx in list(time_series_df.query('label == 1').index):
            start_idx = end_idx - args.pred_time if end_idx > args.pred_time else 0
            time_series_df.loc[start_idx:end_idx, "label"] = 1

        logger.debug("Normal: {0}".format(len(time_series_df[time_series_df["label"] == 0])))
        logger.debug("Anormal: {0}".format(len(time_series_df[time_series_df["label"] == 1])))

        # time information
        for cur_time in time_lst:
            if cur_time == time_idx:
                time_series_df["hour_{0}".format(cur_time)] = 1
            else:
                time_series_df["hour_{0}".format(cur_time)] = 0

        # weekly infomation
        for cur_day in day_lst:
            if cur_day == args.day:
                time_series_df[cur_day] = 1
            else:
                time_series_df[cur_day] = 0

        # select row for every interval
        time_series_df = time_series_df.iloc[[i for i in range(0, len(time_series_df), args.interval)]]

        # shift feature
        shift_col_lst = ["mean", "var", "max", "degree_mean", "degree_std"]
        for shift_col in shift_col_lst:
            time_series_df["{0}_shift1".format(shift_col)] = time_series_df[shift_col] - time_series_df[shift_col].shift()

        time_series_df = time_series_df.dropna()

        # normalize dataset
        if args.normalize:
            col_lst = ["mean", "var", "max", "degree_mean","degree_std"]
            time_series_df = normalize(time_series_df, col_lst)

        # save dataset
        if args.normalize:
            save_path = "{0}/{1}/normalize/time_series_{2}.csv".format(args.save_datasets_dirc, args.date, time_idx)
        else:
            save_path = "{0}/{1}/default/time_series_{2}.csv".format(args.save_datasets_dirc, args.date, time_idx)
        time_series_df.to_csv(save_path, index=False)
        logger.debug("save dataset: {0}".format(save_path))


def datasets_parse():
    parser = argparse.ArgumentParser(
        prog="make_datasets.py",
        usage="",
        description="description",
        epilog="end",
        add_help=True
    )

    # Data Argument
    parser.add_argument("--date", type=str, default="20170416")
    parser.add_argument("--day", type=str, default="Sun",
                        help="select from [Sun, Mon, Tue, Wed, Thurs, Fri, Sat]")
    parser.add_argument("--root_stats_dirc", type=str,
                        default="/Users/sakka/cnn_anomaly_detection/data/statistics")
    parser.add_argument("--root_grid_dirc", type=str,
                        default="/Users/sakka/cnn_anomaly_detection/data/grid_ratio")
    parser.add_argument("--root_human_dirc", type=str,
                        default="/Users/sakka/cnn_anomaly_detection/data/human_area")
    parser.add_argument("--root_feed_dirc", type=str,
                        default="/Users/sakka/cnn_anomaly_detection/data/feed")
    parser.add_argument("--root_diver_dirc", type=str,
                        default="/Users/sakka/cnn_anomaly_detection/data/diver")
    parser.add_argument("--save_datasets_dirc", type=str,
                        default="/Users/sakka/cnn_anomaly_detection/data/datasets")

    # Parameter Argument
    parser.add_argument("--pred_time", type=int, default=0, help="how many frame after anormaly is detected (sec*FPS)")
    parser.add_argument("--interval", type=int, default=30, help="interval of dataset row")
    parser.add_argument("--normalize", type=bool, default=False, help="whether normalize or not for dataset")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = datasets_parse()
    logger.debug("Running with args: {0}".format(args))
    make_datasets(args)
