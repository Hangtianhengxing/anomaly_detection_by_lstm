#! /usr/bin/env python
#coding: utf-8

import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)
logs_path = "/Users/sakka/cnn_anomaly_detection/logs/feed.log"
logging.basicConfig(filename=logs_path,
                    level=logging.DEBUG,
                    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")


def feed_dataset(args):
    """
    feed_day:
        whether feed is given the that day
    feed_cur:
        wheter feed is given the current time
    feed_day and feed_cur columns has 1 or 0 value. that mean below.
        1: feed is given  
        0: feed is NOT given
    """

    # initialize feed arr to zero
    for time_idx in tqdm(range(9, 17)):
        feed_dictlst = {"frame_num": [], "feed_day": [], "feed_cur": []}
        time_length = len(pd.read_csv(
            "{0}/{1}/{2}/mean.csv".format(args.root_stats_dirc, args.date, time_idx)))
        feed_dictlst["frame_num"] = [
            args.start_frame_num + i for i in range(time_length)]
        if args.feed:
            feed_dictlst["feed_day"] = list(
                np.ones(time_length, dtype=np.int8))
        else:
            feed_dictlst["feed_day"] = list(
                np.zeros(time_length, dtype=np.int8))

        # whether the feed is in the tank in each frame
        feed_arr = np.zeros(time_length, dtype=int)
        if args.feed:
            if (time_idx == args.start_feed_h) and (time_idx == args.end_feed_h):
                start_idx = args.start_feed_m * 60 * args.fps
                end_idx = args.end_feed_m * 60 * args.fps
                feed_arr[start_idx:end_idx] = 1
            elif(time_idx == args.start_feed_h) and (time_idx < args.end_feed_h):
                start_idx = args.start_feed_m * 60 * args.fps
                feed_arr[start_idx:] = 1
            elif (time_idx > args.start_feed_h) and (time_idx < args.end_feed_h):
                start_idx = 0
                feed_arr[start_idx:] = 1
            elif (time_idx > args.start_feed_h) and (time_idx == args.end_feed_h):
                start_idx = 0
                end_idx = args.end_feed_m * 60 * args.fps
                feed_arr[start_idx:end_idx] = 1

        feed_dictlst["feed_cur"] = list(feed_arr)

        assert len(feed_dictlst["frame_num"]) == len(feed_dictlst["feed_day"]) == len(feed_dictlst["feed_cur"])

        pd.DataFrame(feed_dictlst).to_csv(
            "{0}/{1}/feed_{2}.csv".format(args.root_feed_dirc, args.date, time_idx), index=False)


def feed_parse():
    parser = argparse.ArgumentParser(
        prog="feed.py",
        usage="whether or not feed is given",
        description="description",
        epilog="end",
        add_help=True
    )

    # Data Argument
    parser.add_argument("--date", type=str, default="20170416")
    parser.add_argument("--root_stats_dirc", type=str,
                        default="/Users/sakka/cnn_anomaly_detection/data/statistics")
    parser.add_argument("--root_feed_dirc", type=str,
                        default="/Users/sakka/cnn_anomaly_detection/data/feed")

    # Parameter Argumant
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--start_frame_num", type=int, default=121)
    parser.add_argument("--feed", type=bool, default=True)
    parser.add_argument("--start_feed_h", type=int, default=14)
    parser.add_argument("--start_feed_m", type=int, default=38)
    parser.add_argument("--end_feed_h", type=int, default=14)
    parser.add_argument("--end_feed_m", type=int, default=41)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = feed_parse()
    logger.debug("Running with args: {0}".format(args))
    feed_dataset(args)
