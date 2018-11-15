#! /usr/bin/env python
#coding: utf-8

import sys
import re
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import glob
from tqdm import tqdm

logger = logging.getLogger(__name__)
logs_path = "/Users/sakka/cnn_anomaly_detection/logs/ratio_per_grid.log"
logging.basicConfig(filename=logs_path,
                    level=logging.DEBUG,
                    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")


def ratio_per_grid(cord_dirc, extention, skip, grid_num=(8,1)):
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

        cnt_dict = {}
        grid_y_index = 0
        grid_x_index = 0
        for y in range(grid_num[1]):
            for x in range(grid_num[0]):
                count = len(cordinate_df[(cordinate_df["x_g"] == x) & (cordinate_df["y_g"] == y)])
                cnt_dict["grid_{0}_{1}".format(grid_y_index, grid_x_index)] = count
                grid_x_index += 1
            grid_y_index += 1
            grid_x_index = 0

        return cnt_dict

    def get_raw_info(file_lst, skip):
        """
        Extract the frame number from file path and extend it to the size of the raw data.
        """
        raw_data_lst = []
        frame_num_lst = []
        for path in file_lst:
            raw_data = path.split("/")[-1]
            raw_data_lst.extend([raw_data for _ in range(skip)])
            # get frame nuber from file name of input cordinate
            frame_num = int(raw_data[:-4])
            frame_num_lst.extend([frame_num+i for i in range(skip)])
        return raw_data_lst, frame_num_lst

    def numerical_sort(value):
        numbers = re.compile(r"(\d+)")
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    file_lst = sorted(glob.glob("{0}*{1}".format(cord_dirc, extention)), key=numerical_sort)
    if len(file_lst) == 0:
        sys.stderr.write("Error: not found cordinate file(.npy)")
        sys.exit(1)
    else:
        pass

    grid_dictlst = {}
    for grid_y in range(grid_num[1]):
        for grid_x in range(grid_num[0]):
            grid_dictlst["grid_{0}_{1}".format(grid_y, grid_x)] = []

    for i in tqdm(range(len(file_lst))):
        cordinate = np.loadtxt(file_lst[i], delimiter=",")
        cordinate_df = pd.DataFrame(cordinate, columns=["x", "y"]).astype(np.int32)
        cnt_dict = count_feature(cordinate_df, grid_num)
        # conver count to ratio and complement to interval frame
        total_cnt = sum(cnt_dict.values())
        for key, value in cnt_dict.items():
            grid_dictlst[key].extend([value/total_cnt for _ in range(skip)])

    grid_df = pd.DataFrame(grid_dictlst)
    max_series = grid_df.max(axis=1)
    idxmax_series = grid_df.idxmax(axis=1)
    grid_df["max"] = max_series
    grid_df["max_index"] = idxmax_series

    raw_data_lst, frame_num_lst = get_raw_info(file_lst, skip)
    assert len(grid_df) == len(raw_data_lst)
    grid_df["raw_data"] = raw_data_lst
    assert len(grid_df) == len(frame_num_lst)
    grid_df["frame_num"] = frame_num_lst

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
        plt.xticks(np.arange(0,valueXLimMax+1, args.skip), np.arange(0,int(valueXLimMax/args.skip)+1,1))
        plt.yticks(np.arange(0,1.1,0.1))
        plt.rcParams["font.size"] = 50
        plt.grid(True)
        plt.plot(valueX, value_lst)

    plt.figure()
    plt.figure(figsize=(20, 6))
    plt.title(args.titel)
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    set_graph(value_lst)
    plt.savefig(args.save_img_path)

    logger.debug("SAVE: graph({})".format(args.save_img_path))


def main(args):
    grid_df = ratio_per_grid(args.cord_dirc, args.extention, args.skip, args.grid_num)
    #plot(list(grid_df["max"]), args)


def grid_parse():
    parser = argparse.ArgumentParser(
        prog="ratio_per_grid.py",
        usage="",
        description="description",
        epilog="end",
        add_help=True
    )

    # Data Argument
    parser.add_argument("--cord_dirc", type=str,
                        default="/Users/sakka/cnn_anomaly_detection/data/cord/20170416/9/")
    parser.add_argument("--extention", type=str, default=".csv")
    parser.add_argument("--save_grid_path", type=str,
                        default="/Users/sakka/cnn_anomaly_detection/data/grid_ratio/20170416/9.csv")
    parser.add_argument("--save_img_path", type=str,
                        default="/Users/sakka/cnn_anomaly_detection/image/grid_ratio/20170416/9.png")

    # Parameter Argument
    parser.add_argument("--grid_num", type=tuple, default=(8, 1))
    parser.add_argument("--sigma_pow", type=int, default=25)
    parser.add_argument("--skip", type=int, default=30, help="skip interval of frame")
    parser.add_argument("--titel", type=str, default="occupancy of feature points")
    parser.add_argument("--xlabel", type=str, default="frame number")
    parser.add_argument("--ylabel", type=str, default="occupancy rate [%]")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = grid_parse()
    logger.debug("Running with args: {0}".format(args))
    main(args)
