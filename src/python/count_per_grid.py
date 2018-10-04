#! /usr/bin/env python
#coding: utf-8

import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

logger = logging.getLogger(__name__)
logs_path = "/Users/sakka/cnn_anomaly_detection/logs/count_per_grid.log"
logging.basicConfig(filename=logs_path,
                    level=logging.DEBUG,
                    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")


def count_per_grid(cord_dirc, grid_num=(5,1)):
    """
    save data that counted oubject per grid

    input:
        cord_dirc: directory path (codinate of target (.npy))
        grid_num:    (col, row), default value is (5, 4)

    index of grid:
        ex) grid_num=(3, 2)
                           col
                -------------------------
                |   0   |   1   |   2   |
           row  -------------------------
                |   3   |   4   |   5   |
                -------------------------
    """

    def count_feature(cordinate_df, grid_num):
        """
        count the number of targets for each grid_dictlst

        input:
            cordinate_df: dataframe of target cordinate (columns: [x, y])
            grid_num:    (col, row)
        """

        img_size = (1280, 720)  # cordinate: (x, y)
        grid_size = (int(img_size[0]/grid_num[0]), int(img_size[1]/grid_num[1]))  # cordinate: (x, y)
        cordinate_df["x_g"] = (cordinate_df["x"]/grid_size[0]).astype(np.int32)
        cordinate_df["y_g"] = (cordinate_df["y"]/grid_size[1]).astype(np.int32)

        count_dict = {}
        grid_index = 0
        for y in range(grid_num[1]):
            for x in range(grid_num[0]):
                count = len(cordinate_df[(cordinate_df["x_g"] == x) & (cordinate_df["y_g"] == y)])
                count_dict[grid_index] = count
                grid_index += 1

        return count_dict

    file_lst = glob.glob(cord_dirc + "*.npy")
    if len(file_lst) == 0:
        sys.stderr.write("Error: not found cordinate file(.npy)")
        sys.exit(1)
    else:
        pass

    grid_dictlst = {}
    for i in range(grid_num[0]*grid_num[1]):
        grid_dictlst[i] = []

    for i in tqdm(range(1, len(file_lst)+1)):
        cordinate = np.load(cord_dirc + "{}.npy".format(i))
        cordinate_df = pd.DataFrame(cordinate, columns=["x", "y"]).astype(np.int32)
        count_dict = count_feature(cordinate_df, grid_num)
        for key, value in count_dict.items():
            grid_dictlst[key].append(value)

    grid_df = pd.DataFrame(grid_dictlst)
    sum_series = grid_df.sum(axis=1)
    max_series = grid_df.max(axis=1)
    idxmax_series = grid_df.idxmax(axis=1)
    grid_df["sum"] = sum_series
    grid_df["max"] = max_series
    grid_df["max_index"] = idxmax_series
    grid_df.to_csv(args.save_grid_path, index=False)

    return grid_df


def plot(value_lst, args):
    """
    input:
        value_lst: list of target value
        info_dict: dict of graph info  ex) key=title, xlabel, ylabel
        output_path: path of output graph
    """
    def set_graph(value_lst):
        valueX = []
        for i in range(len(value_lst)):
            valueX.append(i)
        valueXLimMin = valueX[0]
        valueXLimMax = valueX[-1]
        valueYLimMin = 0
        valueYLimMax = 1.0
        plt.xlim(valueXLimMin, valueXLimMax)
        plt.ylim(valueYLimMin, valueYLimMax)
        plt.xticks(np.arange(0,valueXLimMax+1, 30), np.arange(0,int(valueXLimMax/30)+1,1))  #30 is fps
        plt.yticks(np.arange(0,1.1,0.1))
        plt.rcParams["font.size"] = 50
        plt.grid(True)
        plt.plot(valueX, value_lst)

    plt.figure()
    plt.figure(figsize=(20, 6))
    plt.title(info_dict[args.title])
    plt.xlabel(info_dict[args.xlabel])
    plt.ylabel(info_dict[args.ylabel])
    set_graph(value_lst)
    plt.savefig(args.save_img_path)

    logger.debug("SAVE: graph({})".format(args.save_img_path))


def main(args):
    grid_df = count_per_grid(args.cord_dirc, args.grid_num)
    plot(list(grid_df["max"]/grid_df["sum"]), args)


def grid_parse():
    parser = argparse.ArgumentParser(
        prog="count_per_grid.py",
        usage="",
        description="description",
        epilog="end",
        add_help=True
    )

    # Data Argument
    parser.add_argument("--cord_dirc", type=str,
                        default="/Users/sakka/cnn_anomaly_detection/data/cord/20170421/9/")
    parser.add_argument("--save_grid_path", type=str,
                        default="/Users/sakka/cnn_anomaly_detection/data/output/grid_count/20170421/9.csv")
    parser.add_argument("--save_img_path", type=str,
                        default="/Users/sakka/cnn_anomaly_detection/image/output/grid_count/20170421/9.png")

    # Parameter Argument
    parser.add_argument("--grid_num", type=tuple, default=(5, 1))
    parser.add_argument("--sigma_pow", type=int, default=25)
    parser.add_argument("--titel", type=str, default="occupancy of feature points")
    parser.add_argument("--xlabel", type=str, default="frame number")
    parser.add_argument("--ylabel", type=str, default="occupancy rate [%]")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = grid_parse()
    main(args)
