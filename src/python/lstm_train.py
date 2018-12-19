#! /usr/bin/env python
#coding: utf-8

import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import argparse
import gc
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.autograd import Variable

from lstm_util import Predictor, to_variable, scale, batch_data

date = datetime.now()
learning_date = "{0:04d}{1:02d}{2:02d}_{3:02d}{4:02d}".format(
    date.year, date.month, date.day, date.hour, date.minute)

logger = logging.getLogger(__name__)
logs_path = "/home/sakka/cnn_anomaly_detection/logs/lstm_train_{}.log".format(learning_date)
logging.basicConfig(filename=logs_path,
                    level=logging.DEBUG,
                    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")


def train(args):
    # load dataset
    train_df = pd.read_csv(args.train_path)
    y_train = train_df["label"].as_matrix()
    X_train = train_df.as_matrix()

    val_df = pd.read_csv(args.val_path)
    y_val = val_df["label"].as_matrix()
    X_val = val_df.as_matrix()

    # train and val data applied MinMaxScaler 
    X_train, X_val, y_train, y_val, _, _ = scale(X_train, X_val, y_train, y_val)
    logger.debug("X_train shape: {}".format(X_train.shape))
    logger.debug("y_train shape: {}".format(y_train.shape))
    logger.debug("X_val shape: {}".format(X_val.shape))
    logger.debug("y_val shape: {}".format(y_val.shape))

    # define model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug("DEVICE: {}".format(device))
    model = Predictor(args.input_dim, args.hidden_dim, args.num_layers, args.output_dim, args.dropout_ratio)
    #model = nn.DataParallel(model).to(device)
    model = model.to(device)

    # learning condition
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    not_improved_count = 0
    best_epoch = 0

    # number of each batch
    train_n_batches = int((X_train.shape[0]-args.n_pred-args.pred_point+1)/args.batch_size)
    val_n_batches = int((X_val.shape[0]-args.n_pred-args.pred_point+1)/args.batch_size)

    # starting info
    save_model_path = "{0}/model_{1}.pth".format(
        args.save_model_dirc, learning_date)

    train_loss_lst = []
    val_loss_lst = []
    for epoch in range(args.num_epochs):
        train_loss = 0.0
        val_loss = 0.0
        model.train()
        for train_idx in tqdm(range(train_n_batches)):
            optimizer.zero_grad()
            X, y = batch_data(X_train, y_train, train_idx, args.batch_size, n_prev=args.n_pred, pred_point=args.pred_point)
            X = to_variable(torch.Tensor(X))
            y = to_variable(torch.Tensor(y))
            output = model(X)[:, 0]
            
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()  
            train_loss += loss.item()
            del X, y, loss
            gc.collect()
        
        train_loss_lst.append(train_loss)
            
        model.eval()
        for val_idx in tqdm(range(val_n_batches)):
            X, y = batch_data(X_val, y_val, val_idx, args.batch_size, n_prev=args.n_pred, pred_point=args.pred_point)
            X = to_variable(torch.Tensor(X))
            y = to_variable(torch.Tensor(y))
                
            output = model(X)[:, 0]
            
            loss = criterion(output, y)
            val_loss += loss.item()
            del X, y, loss
            gc.collect()
            
        val_loss_lst.append(val_loss)
        
        # early stopping
        if (epoch > args.min_epoch) and (val_loss_lst[-1] > val_loss_lst[-2]):
            not_improved_count += 1
        else:
            # learning is going well
            not_improved_count = 0
            # save best params model
            best_epoch = epoch+1
            torch.save(model.state_dict(), save_model_path)

        logger.debug("EPOCH: {}, TRAIN LOSS: {}, VAL LOSS: {}, EARLY STOPPING: {}/{}".format(
                    epoch + 1, 
                    train_loss_lst[-1], 
                    val_loss_lst[-1],
                    not_improved_count,
                    args.stop_count))

        if not_improved_count == args.stop_count:
            logger.debug("Early Stopping")
            break

    logger.debug("Model saved in \"{0}\"".format(save_model_path))


def make_train_parse():
    parser = argparse.ArgumentParser(
        prog="lstm_train.py",
        usage="train lstm",
        description="description",
        epilog="end",
        add_help=True
    )

    # Data Argument
    parser.add_argument("--train_path", type=str,
                        default="/home/sakka/cnn_anomaly_detection/data/datasets/train_gaus.csv")
    parser.add_argument("--val_path", type=str,
                        default="/home/sakka/cnn_anomaly_detection/data/datasets/val_gaus.csv")
    parser.add_argument("--save_model_dirc", type=str,
                        default="/home/sakka/cnn_anomaly_detection/data/model")

    # Parameter Argument
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--input_dim", type=int, default=67)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout_ratio", type=int, default=0.5)
    parser.add_argument("--output_dim", type=float, default=1)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--min_epoch", type=int, default=5)
    parser.add_argument("--stop_count", type=int, default=3)
    parser.add_argument("--n_pred", type=int, default=60*30)
    parser.add_argument("--pred_point", type=int, default=60*10)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = make_train_parse()
    logger.debug("Running with args: {0}".format(args))
    train(args)
