#! /usr/bin/env python
#coding: utf-8


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from tqdm import tqdm


def plot_densmap(cord_dircpath, save_filepath, bw=20):
    cord_file_lst = glob.glob(cord_dircpath+"*.csv")

    cord_dictlst = {"x":[], "y":[]}
    for path in tqdm(cord_file_lst):
        cord_arr = np.loadtxt(path, delimiter=",")
        cord_dictlst["x"].extend(cord_arr[:,0])
        cord_dictlst["y"].extend(cord_arr[:, 1])
        
    assert len(cord_dictlst["x"]) == len(cord_dictlst["y"])

    print("NOW: plot density map")
    ax = sns.kdeplot(cord_dictlst["x"], cord_dictlst["y"], bw=bw)
    ax.set_ylim([720, 0])
    ax.set_xlim([0, 1280])
    plt.savefig(save_filepath)
    print("SAVE FIG: {}".format(save_filepath))

if __name__ == "__main__":
    cord_dircpath = "../data/20170421/11/"
    save_filepath = "../data/20170421/map/11.png"
    plot_densmap(cord_dircpath, save_filepath, bw=20)