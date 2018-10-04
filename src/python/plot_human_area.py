#! /usr/bin/env python
# coding: utf-8

import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)
logs_path = "/Users/sakka/cnn_anomaly_detection/logs/plot__human_area.log"
logging.basicConfig(filename=logs_path,
                    level=logging.DEBUG,
                    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")

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
    file_path = input("input file path: ")
    file_name = file_path.split("/")[-1]
    file_name = file_name.replace(".csv", ".png")
    output_path = "../output/image/human_area/" + file_name
    property_dict = {"title":"area ratio of front human",
                     "xlabel":"frame number",
                     "ylabel":"ratio",
                     "ylim": (0.0, 1.0)
                     }

    plot_timeseries(file_path, output_path, property_dict)
