#! /usr/bin/env python
#coding: utf-8

import sys
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm

def count_gird(cordinate_df, grid_num):
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



def main(cordinate_dp, grid_num=(5,4)):
    """
    save data that counted oubject per grid

    input:
        cordinate_dp: directory path (codinate of target (.npy))
        grid_num:    (col, row), default value is (5, 4)

    index of grid:
        ex) grid_num=(3, 2)
                        row
                -------------------------
                |   0   |   1   |   2   |
         col    -------------------------
                |   3   |   4   |   5   |
                -------------------------
    """

    file_lst = glob.glob(cordinate_dp + "*.npy")
    if len(file_lst) == 0:
        sys.stderr.write("Error: not found cordinate file(.npy)")
    else:
        pass

    grid_dictlst = {}
    for i in range(grid_num[0]*grid_num[1]):
        grid_dictlst[i] = []

    for i in tqdm(range(1, len(file_lst)+1)):
        cordinate = np.load(cordinate_dp + "{}.npy".format(i))
        cordinate_df = pd.DataFrame(cordinate, columns=["x", "y"]).astype(np.int32)
        count_dict = count_gird(cordinate_df, grid_num)
        for key, value in count_dict.items():
            grid_dictlst[key].append(value)

    pd.DataFrame(grid_dictlst).to_csv("../data/output/{}_grid_count.csv".format(cordinate_dp.split("/")[-2]))


if __name__ == "__main__":
    args = sys.argv
    main(args[1], grid_num=(5, 4))
