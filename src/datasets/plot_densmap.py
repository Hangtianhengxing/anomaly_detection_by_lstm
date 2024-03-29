#! /usr/bin/env python
#coding: utf-8

import logging
import numpy as np 
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from tqdm import tqdm
from tqdm import trange


logger = logging.getLogger(__name__)
logs_path = "/Users/sakka/cnn_anomaly_detection/logs/plot_densmap.log"
logging.basicConfig(filename=logs_path,
                    level=logging.DEBUG,
                    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")


def total_cord(cord_dircpath):
    cord_file_lst = glob.glob(args.cord_dirc+"*.csv")

    cord_dictlst = {"x":[], "y":[]}
    for path in tqdm(cord_file_lst):
        cord_arr = np.loadtxt(path, delimiter=",", dtype="int32")
        cord_arr = cord_arr[~((cord_arr[:, 0] < 400) * (cord_arr[:, 1] < 100))]
        cord_dictlst["x"].extend(cord_arr[:,0])
        cord_dictlst["y"].extend(cord_arr[:, 1])
        
    assert len(cord_dictlst["x"]) == len(cord_dictlst["y"]) 

    return cord_dictlst


def plot_kde(args):
    cord_dictlst = total_cord(args.cord_dirc)

    logger.debug("NOW: plot density map")
    ax = sns.kdeplot(cord_dictlst["x"], cord_dictlst["y"], bw=args.band_width)
    ax.set_ylim([720, 0])
    ax.set_xlim([0, 1280])
    plt.savefig(args.save_path)
    logger.debug("SAVE FIG: {0}".format(args.save_path))


def gausian_kernel(args):
    cord_dictlst = total_cord(args.cord_dirc)

    # init distance matrix
    cordinate_matrix = np.zeros((720, 1280, 2), dtype="int32")
    for y in range(720):
        for x in range(1280):
            cordinate_matrix[y][x] = [y, x]

    # NEEDFIX: remove range
    remove_x = np.linspace(0,349,350)
    remove_y = np.linspace(0,99,100)

    kernel = np.zeros((720, 1280))
    for i in trange(len(cord_dictlst["x"])):
        tmp_cord_matrix = np.array(cordinate_matrix)
        x = cord_dictlst["x"][i]
        y = cord_dictlst["y"][i]
        if (x not in remove_x) and (y not in remove_y):
            point_matrix = np.full((720, 1280, 2), [y, x])
            diff_matrix = tmp_cord_matrix - point_matrix
            pow_matrix = diff_matrix * diff_matrix
            norm = pow_matrix[:, :, 0] + pow_matrix[:, :, 1]
            kernel += np.exp(-norm/ (2 * args.sigma_pow))

    plt.imshow(kernel)
    plt.savefig(args.save_path)
    logger.debug("SAVE FIG: {0}".format(args.save_path))


def densmap_parse():
    parser = argparse.ArgumentParser(
        prog="plot_densmap.py",
        usage="",
        description="description",
        epilog="end",
        add_help=True
    )

    # Data Argument
    parser.add_argument("--cord_dirc", type=str,
                        default="/Users/sakka/cnn_anomaly_detection/data/cord/1min/")
    parser.add_argument("--save_path", type=str,
                        default="/Users/sakka/cnn_anomaly_detection/data/20170421/map/11.png")

    # Parameter Argument
    parser.add_argument("--band_width", type=int, default=15)
    parser.add_argument("--sigma_pow", type=int, default=25)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = densmap_parse()
    logger.debug("Running with args: {0}".format(args))
    plot_kde(args)
    #gausian_kernel(args)
