#! /usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_timeseries(file_path, output_path, property_dict):
    timeseries_df = pd.read_csv(file_path, header=None, names=["area_ratio"])
    timeseries_lst = list(timeseries_df["area_ratio"])
    plt.figure(figsize=(20, 6))
    plt.title(property_dict["title"])
    plt.xlabel(property_dict["xlabel"])
    plt.ylabel(property_dict["ylabel"])
    plt.ylim(property_dict["ylim"])
    plt.tick_params(labelsize=8)
    plt.grid(True)
    plt.plot(timeseries_lst)
    plt.savefig(output_path)
    print("SAVE GRAPH: {}".format(output_path))



if __name__ == "__main__":
    file_path = "../output/data/human_area/201704211300.csv"
    output_path = "../output/image/human_area/201704211300.png"
    property_dict = {"title":"area ratio of front human",
                     "xlabel":"frame number",
                     "ylabel":"ratio",
                     "ylim": (0.0, 1.0)
                     }

    plot_timeseries(file_path, output_path, property_dict)
