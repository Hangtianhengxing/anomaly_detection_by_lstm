#! /usr/bin/env python
# coding: utf-8

import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)
logs_path = "/Users/sakka/cnn_anomaly_detection/logs/plot_human_area.log"
logging.basicConfig(filename=logs_path,
                    level=logging.DEBUG,
                    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")


def plot_timeseries(file_path, output_path, property_dict):
    timeseries_df = pd.read_csv(args.input_path, header=None, names=[args.target_col])
    timeseries_lst = list(timeseries_df[args.target_col])
    plt.figure(figsize=(20, 6))
    plt.title(property_dict[args.title])
    plt.xlabel(property_dict[args.xlabel])
    plt.ylabel(property_dict[args.ylabel])
    plt.ylim(property_dict[args.ylim])
    plt.tick_params(labelsize=8)
    plt.grid(True)
    plt.plot(timeseries_lst)
    plt.savefig(args.save_path)
    logger.debug("SAVE GRAPH: {}".format(args.save_path))


def human_parse():
    parser = argparse.ArgumentParser(
        prog="plot_human_area.py",
        usage="",
        description="description",
        epilog="end",
        add_help=True
    )

    # Data Argument
    parser.add_argument("--input_path", type=str,
                        default="/Users/sakka/cnn_anomaly_detection/data/cord/201704210900.csv")
    parser.add_argument("--save_path", type=str,
                        default="/Users/sakka/cnn_anomaly_detection/image/output/human_area/201704210900.png")

    # Parameter Argument
    parser.add_argument("--target_col", type=str, default="area_ratio")
    parser.add_argument("--title", type=str, default="area ratio of front human")
    parser.add_argument("--xlable", type=str, default="frame number")
    parser.add_argument("--yabel", type=str, default="ratio")
    parser.add_argument("--ylim", type=tuple, default=(0.0, 1.0))

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    plot_timeseries(file_path, output_path, property_dict)
