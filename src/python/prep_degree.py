#! /usr/bin/env python
#coding: utf-8

import sys
import csv
import logging
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

logger = logging.getLogger(__name__)
logs_path = "/Users/sakka/cnn_anomaly_detection/logs/prep_degree.log"
logging.basicConfig(filename=logs_path,
                    level=logging.DEBUG,
                    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")


def load_degree(file_path, round_deg, deg_width):
    time_degree_lst = []
    degree_dictlst = {"frame_num":[], "right":[], "left":[], "up":[], "down":[], \
                            "right_down":[], "left_down":[], "left_up":[], "right_up":[], \
                            "overall_dir":[], "degree_mean":[], "degree_std":[],\
                            "horizontal":[], "vertical":[], "oblique":[]}

    with open(file_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            degree_dictlst["frame_num"].append(row[0])
            # skip index 0 (the value is frame number)
            if round_deg > 0:
                # NEED FIX last value is ""
                time_degree_lst = round_degree(row[1:-1], args.round_deg)
            else:
                time_degree_lst = np.array(row[1:-1], dtype="int32")
            # calc mean, val of degree for each time
            degree_dictlst["degree_mean"].append(np.mean(time_degree_lst))
            degree_dictlst["degree_std"].append(np.std(time_degree_lst))
            # calc ratio of diraction for each time
            dirc2ratio = direction_ratio(time_degree_lst, args.deg_width)
            for k, v in dirc2ratio.items():
                degree_dictlst[k].append(v)

    # information of rough direction
    degree_dictlst["horizontal"] = np.array(degree_dictlst["right"]) + np.array(degree_dictlst["left"])
    degree_dictlst["vertical"] = np.array(degree_dictlst["up"]) + np.array(degree_dictlst["down"])
    degree_dictlst["oblique"] = np.ones(len(degree_dictlst["horizontal"])) - (np.array(degree_dictlst["horizontal"]) + np.array(degree_dictlst["vertical"]))

    return pd.DataFrame(degree_dictlst)


def round_degree(degree_lst, round_deg):
    round_degree_lst = []
    for degree in degree_lst:
        # 360 is NOT moveing point
        if degree != 360:
            round_degree_lst.append(int(int(degree)/round_deg)*round_deg)
    return round_degree_lst


def direction_ratio(degree_lst, deg_width):
    right_count = 0
    down_count = 0
    left_count = 0
    up_count = 0
    right_down_count = 0
    left_down_count = 0
    left_up_count = 0
    right_up_count = 0
    
    for deg in degree_lst:
        if (deg >= deg_width) and (deg <= 90-deg_width):
            right_down_count += 1
        elif (deg >= 90-deg_width) and (deg <= 90+deg_width):
            down_count +=1
        elif (deg >= 90+deg_width) and (deg <= 180-deg_width):
            left_down_count += 1
        elif (deg >= 180-deg_width) and (deg <= 180+deg_width):
            left_count += 1
        elif (deg >= 180+deg_width) and (deg <= 270-deg_width):
            left_up_count += 1
        elif (deg >= 270-deg_width) and (deg <= 270+deg_width):
            up_count += 1
        elif (deg >= 270+deg_width) and (deg <= 360-deg_width):
            right_up_count += 1
        else:
            right_count += 1
    
    dir2ratio = {}
    
    dir2ratio["right"] = right_count/len(degree_lst)
    dir2ratio["down"] = down_count/len(degree_lst)
    dir2ratio["left"] = left_count/len(degree_lst)
    dir2ratio["up"] = up_count/len(degree_lst)
    dir2ratio["right_down"] = right_down_count/len(degree_lst)
    dir2ratio["left_down"] = left_down_count/len(degree_lst)
    dir2ratio["left_up"] = left_up_count/len(degree_lst)
    dir2ratio["right_up"] = right_up_count/len(degree_lst)
    dir2ratio["overall_dir"] = max(dir2ratio, key=dir2ratio.get)
    
    return dir2ratio


def prep_degree(args):
    for time_idx in tqdm(range(9, 17)):
        degree_df = load_degree(args.root_degree_dirc+"{}/degree.csv".format(time_idx), args.round_deg, args.deg_width)
        save_degree_path = args.root_degree_dirc + "{}/prep_degree.csv".format(time_idx)
        degree_df.to_csv(save_degree_path, index=False)
        logger.debug("save in {0}".format(save_degree_path))


def prep_degree_parse():
    parser = argparse.ArgumentParser(
        prog="prep_degree.py",
        usage="",
        description="description",
        epilog="end",
        add_help=True
    )

    # Data Argument
    parser.add_argument("--root_degree_dirc", type=str
                        default="/Users/sakka/cnn_anomaly_detection/data/statistics/20181101/")
   
    # Parameter Argument
    parser.add_argument("--round_deg", type=int, default=0)
    parser.add_argument("--deg_width", type=int, default=5)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = prep_degree_parse()
    logger.debug("Running with args: {0}".format(args))
    prep_degree(args)