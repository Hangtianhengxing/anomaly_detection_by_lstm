#! /usr/bin/env python
#coding: utf-8

import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)
logs_path = "/Users/sakka/cnn_anomaly_detection/logs/diver.log"
logging.basicConfig(filename=logs_path,
                    level=logging.DEBUG,
                    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")

def diver_dataset(args):
    """
    diver_day:
        whether the diver in tank the that day
    diver_cur:
        wheter the diver in tank the current time
    diver_day and diver_cur columns has 1 or 0 value. that mean below.
        1: diver in tank  
        0: diver NOT in tank
    """

    # initialize diver arr to zero
    for time_idx in tqdm(range(9, 17)):
        diver_dictlst = {"frame_num": [], "diver_day": [], "diver_cur": []}
        time_length = len(pd.read_csv("{0}/{1}/{2}/mean.csv".format(args.root_stats_dirc, args.date, time_idx)))
        diver_dictlst["frame_num"] = [args.start_frame_num + i for i in range(time_length)]
        if args.diver:
            diver_dictlst["diver_day"] = list(np.ones(time_length, dtype=np.int8))
        else:
            diver_dictlst["diver_day"] = list(np.zeros(time_length, dtype=np.int8))

        # whether the diver is in the tank in each frame 
        diver_arr = np.zeros(time_length, dtype=int)
        if args.diver:
            if (time_idx == args.start_diver_h) and (time_idx == args.end_diver_h):
                start_idx = args.start_diver_m*60*args.fps
                end_idx = args.end_diver_m*60*args.fps
                diver_arr[start_idx:end_idx] = 1
            elif(time_idx == args.start_diver_h) and (time_idx < args.end_diver_h):
                start_idx = args.start_diver_m*60*args.fps
                diver_arr[start_idx:] = 1
            elif (time_idx > args.start_diver_h) and (time_idx < args.end_diver_h):
                start_idx = 0
                diver_arr[start_idx:] = 1
            elif (time_idx > args.start_diver_h) and (time_idx == args.end_diver_h):
                start_idx = 0
                end_idx = args.end_diver_m*60*args.fps
                diver_arr[start_idx:end_idx] = 1

        diver_dictlst["diver_cur"] = list(diver_arr)

        assert len(diver_dictlst["frame_num"]) == len(diver_dictlst["diver_day"]) == len(diver_dictlst["diver_cur"])

        pd.DataFrame(diver_dictlst).to_csv("{0}/{1}/diver_{2}.csv".format(args.root_diver_dirc, args.date, time_idx), index=False)


def diver_parse():
    parser = argparse.ArgumentParser(
        prog="diver.py",
        usage="whether or not diver in tank",
        description="description",
        epilog="end",
        add_help=True
    )

    # Data Argument
    parser.add_argument("--date", type=str, default="20170416")
    parser.add_argument("--root_stats_dirc", type=str,
                        default="/Users/sakka/cnn_anomaly_detection/data/statistics")
    parser.add_argument("--root_diver_dirc", type=str,
                        default="/Users/sakka/cnn_anomaly_detection/data/diver")

    # Parameter Argumant
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--start_frame_num", type=int, default=121)
    parser.add_argument("--diver", type=bool, default=False)
    parser.add_argument("--start_diver_h", type=int, default=9)
    parser.add_argument("--start_diver_m", type=int, default=22)
    parser.add_argument("--end_diver_h", type=int, default=12)
    parser.add_argument("--end_diver_m", type=int, default=4)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = diver_parse()
    logger.debug("Running with args: {0}".format(args))
    diver_dataset(args)
