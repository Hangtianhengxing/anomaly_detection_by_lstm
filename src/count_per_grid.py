#! /usr/bin/env python
#coding: utf-8

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm


def count_per_grid(cordinate_dp, grid_num=(5,1)):
    """
    save data that counted oubject per grid

    input:
        cordinate_dp: directory path (codinate of target (.npy))
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

    file_lst = glob.glob(cordinate_dp + "*.npy")
    if len(file_lst) == 0:
        sys.stderr.write("Error: not found cordinate file(.npy)")
        sys.exit(1)
    else:
        pass

    grid_dictlst = {}
    for i in range(grid_num[0]*grid_num[1]):
        grid_dictlst[i] = []

    for i in tqdm(range(1, len(file_lst)+1)):
        cordinate = np.load(cordinate_dp + "{}.npy".format(i))
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

    return grid_df


def plot(value_lst, info_dict, output_path):
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
        plt.xticks(np.arange(0,valueXLimMax+1,30), np.arange(0,int(valueXLimMax+1/30),1)) #30 is fps
        plt.yticks(np.arange(0,1.1,0.1))
        plt.rcParams["font.size"] = 50
        plt.grid(True)
        plt.plot(valueX, value_lst)

    plt.figure()
    plt.figure(figsize=(20, 6))
    plt.title(info_dict["title"])
    plt.xlabel(info_dict["xlabel"])
    plt.ylabel(info_dict["ylabel"])
    set_graph(value_lst)
    plt.savefig(output_path)

    print("SAVE: graph({})".format(output_path))


def main(cordinate_dp, grid_num):
    grid_df = count_per_grid(cordinate_dp, grid_num)
    grid_df.to_csv("../data/output/{}_grid_count.csv".format(cordinate_dp.split("/")[-2]), index=False)
    info_dict = {"title":"occupancy of feature points", "xlabel":"frame number", "ylabel":"occupancy [%]"}
    plot(list(grid_df["max"]/grid_df["sum"]), info_dict, "../data/output/1min_grid.png")

if __name__ == "__main__":
    args = sys.argv
    main(args[1], grid_num=(5,1))
