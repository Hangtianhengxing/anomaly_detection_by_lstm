#! /usr/bin/env python
#coding: utf-8

import sys
import csv
import logging
import numpy as np
import pandas as pd
import argparse
from tqdm import trange

logger = logging.getLogger(__name__)
logs_path = "/Users/kenya/cnn_anomaly_detection/logs/prep_degree.log"
logging.basicConfig(filename=logs_path,
                    level=logging.DEBUG,
                    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")


def load_degree(file_path, round_deg, deg_width):
    time_degree_lst = []
    degree_dictlst = {"right":[], "left":[], "up":[], "down":[], \
                                      "others_1":[], "others_2":[], "others_3":[], "others_4":[], \
                                      "degree_mean":[], "degree_std":[]}
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            time_degree_lst = round_degree(row[:-1], args.round_deg)
            # calc mean, val of degree for each time
            degree_dictlst["degree_mean"].append(np.mean(time_degree_lst))
            degree_dictlst["degree_std"].append(np.std(time_degree_lst))
            # calc ratio of diraction for each time
            dirc2ratio = direction_ratio(time_degree_lst, args.deg_width)
            for k, v in dirc2ratio.items():
                degree_dictlst[k].append(v)

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
    others_1_count = 0
    others_2_count = 0
    others_3_count = 0
    others_4_count = 0
    
    for deg in degree_lst:
        if (deg >= deg_width) and (deg <= 90-deg_width):
            others_1_count += 1
        elif (deg >= 90-deg_width) and (deg <= 90+deg_width):
            down_count +=1
        elif (deg >= 90+deg_width) and (deg <= 180-deg_width):
            others_2_count += 1
        elif (deg >= 180-deg_width) and (deg <= 180+deg_width):
            left_count += 1
        elif (deg >= 180+deg_width) and (deg <= 270-deg_width):
            others_3_count += 1
        elif (deg >= 270-deg_width) and (deg <= 270+deg_width):
            up_count += 1
        elif (deg >= 270+deg_width) and (deg <= 360-deg_width):
            others_4_count += 1
        else:
            right_count += 1
    
    dir2ratio = {}
    dir2ratio["right"] = right_count/len(degree_lst)
    dir2ratio["down"] = down_count/len(degree_lst)
    dir2ratio["left"] = left_count/len(degree_lst)
    dir2ratio["up"] = up_count/len(degree_lst)
    dir2ratio["others_1"] = others_1_count/len(degree_lst)
    dir2ratio["others_2"] = others_2_count/len(degree_lst)
    dir2ratio["others_3"] = others_3_count/len(degree_lst)
    dir2ratio["others_4"] = others_4_count/len(degree_lst)
    
    return dir2ratio


def prep_degree(args):
    for time_idx in trange(9, 17):
        degree_df = load_degree(args.root_degree_dirc+"{}/degree.csv".format(time_idx), args.round_deg, args.deg_width)
        save_degree_path = args.root_degree_dirc + "{}/prep_degree.csv".format(time_idx)
        degree_df.to_csv(save_degree_path, index=False)
        logger.debug("save in {}".format(save_degree_path))


def prep_degree_parse():
    parser = argparse.ArgumentParser(
        prog="prep_degree.py",
        usage="",
        description="description",
        epilog="end",
        add_help=True
    )

    # Data Argument
    parser.add_argument("--root_degree_dirc", type=str,
                        default="/Users/kenya/cnn_anomaly_detection/data/statistics/20170421/")
   
    # Parameter Argumant
    parser.add_argument("--round_deg", type=int, default=10)
    parser.add_argument("--deg_width", type=int, default=10)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = prep_degree_parse()
    prep_degree(args)