#! /usr/bin/env python
#coding: utf-8

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import argparse
import gc
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

from lstm_util import Predictor, to_variable, scale, pred_batch_data, plot_pred, plot_corr

date = datetime.now()
learning_date = "{0:04d}{1:02d}{2:02d}_{3:2d}{4:02d}".format(
    date.year, date.month, date.day, date.hour, date.minute)

logger = logging.getLogger(__name__)
logs_path = "/home/sakka/cnn_anomaly_detection/logs/lstm_pred_{}.log".format(
    learning_date)
logging.basicConfig(filename=logs_path,
                    level=logging.DEBUG,
                    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")



def predict(args):
    # load dataset
    train_df = pd.read_csv(args.train_path)
    y_train = train_df["label"].as_matrix()
    X_train = train_df.as_matrix()

    test_df = pd.read_csv(args.test_path)
    y_test = test_df["label"].as_matrix()
    X_test = test_df.as_matrix()

    # train and val data applied MinMaxScaler
    _, X_test, _, y_test, X_sclr, y_sclr = scale(X_train, X_test, y_train, y_test)
    logger.debug("X_test shape: {0}".format(X_test.shape))
    logger.debug("y_test shape: {0}".format(y_test.shape))

    # device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug("DEVICE: {0}".format(device))

    # model setting
    model = Predictor(args.input_dim, args.hidden_dim, args.num_layers,
                   args.output_dim, dropout_ratio=0)
    #model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    model.eval()

    # initialize
    test_n_batches = int(X_test.shape[0]/args.batch_size)
    #pred_lst = [0 for _ in range(args.batch_size+args.pred_point-1)]
    pred_lst = [0 for _ in range(args.batch_size-1)]

    # prediction
    for test_idx in tqdm(range(test_n_batches)):
        X = pred_batch_data(X_test, test_idx, args.batch_size, args.n_prev)
        X = to_variable(torch.Tensor(X))
        y_pred = model(X)[:, 0]
        y_pred = y_pred.cpu().data.numpy()
        pred_lst.extend(list(y_pred))
    logger.debug("Length of pred data: {0}".format(len(pred_lst)))
    pred_lst.extend([0 for _ in range(args.pred_point)])

    if args.save_output_dirc is not None:
        data_save_path = "{0}/value_{1}.csv".format(args.save_output_dirc, learning_date)
        np.savetxt(data_save_path, np.array(pred_lst))
        logger.debug("Save pred data in {0}".format(data_save_path))
        fig_save_path = "{0}/fig_{1}.png".format(args.save_output_dirc, learning_date)
        plot_pred(np.array(pred_lst), y_test, args.n_prev, args.pred_point, fig_save_path, title_info="test")
        logger.debug("Save figure in {0}".format(fig_save_path))
        # NEEDFIX: separate evaluation from prediction 
        skip = args.batch_size+args.pred_point-1
        pred_lst = pred_lst[skip:-args.pred_point]
        y_test = y_test[-len(pred_lst):]
        print(len(pred_lst))
        print(len(y_test))
        corr_save_path = "{0}/corr_{1}.png".format(args.save_output_dirc, learning_date)
        plot_corr(np.array(pred_lst), y_test, args.n_prev, args.pred_point, corr_save_path, title_info="test")
        logger.debug("Save Corr in {0}".format(corr_save_path))


def make_pred_parse():
    parser = argparse.ArgumentParser(
        prog="lstm_pred.py",
        usage="train lstm",
        description="description",
        epilog="end",
        add_help=True
    )

    # Data Argument
    parser.add_argument("--train_path", type=str,
                        default="/home/sakka/cnn_anomaly_detection/data/datasets/train_gaus.csv")
    parser.add_argument("--test_path", type=str,
                        default="/home/sakka/cnn_anomaly_detection/data/datasets/test_gaus.csv")
    parser.add_argument("--model_path", type=str,
                        default="/home/sakka/cnn_anomaly_detection/data/model/model_20181217_1835.pth")

    # Parameter Argument
    parser.add_argument("--input_dim", type=int, default=67)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--output_dim", type=float, default=1)
    parser.add_argument("--n_prev", type=int, default=60*30)
    parser.add_argument("--batch_size", type=int, default=60*30)
    parser.add_argument("--pred_point", type=int, default=60*10)
    parser.add_argument("--save_output_dirc", type=str,
                        default="/home/sakka/cnn_anomaly_detection/data/prediction")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = make_pred_parse()
    logger.debug("Running with args: {0}".format(args))
    predict(args)
