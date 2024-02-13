#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument(
        "--epochs", type=int, default=500, help="number of rounds of training"
    )
    parser.add_argument(
        "--num_users", type=int, default=3, help="number of users: K"
    )
    parser.add_argument(
        "--frac", type=float, default=1.0, help="the fraction of clients: C"
    )
    parser.add_argument(
        "--local_ep",
        "-ep",
        type=int,
        default=1,
        help="the number of local epochs: E",
    )
    parser.add_argument(
        "--local_bs", "-bs", type=int, default=64, help="local batch size: B"
    )

    parser.add_argument(
        "--local_lr",
        type=float,
        default=0.01,
        help="learning rate for local update",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="learning rate for global update",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.0,
        help="SGD momentum (default: 0.0)",
    )
    parser.add_argument(
        "--beta1", type=float, default=0.9, help="beta1 for adam"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.99, help="beta2 for adam"
    )
    parser.add_argument("--eps", type=float, default=0, help="eps for adam")
    parser.add_argument(
        "--max_init",
        type=float,
        default=1e-3,
        help="initialize max_v for adam",
    )

    # model arguments
    parser.add_argument(
        "--model", "-m", type=str, default="lenet", help="model name"
    )

    # other arguments
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="mnist",
        help="name \
                        of dataset",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
        help="number \
                        of classes",
    )

    parser.add_argument(
        "--gpu",
        "-g",
        default=0,
        help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.",
    )
    parser.add_argument(
        "--optimizer",
        "-o",
        type=str,
        default="fedadam",
        help="type \
                        of optimizer",
    )
    parser.add_argument(
        "--iid",
        type=int,
        default=1,
        help="Default set to IID. Set to 0 for non-IID.",
    )
    parser.add_argument(
        "--unequal",
        type=int,
        default=0,
        help="whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)",
    )
    parser.add_argument(
        "--stopping_rounds",
        type=int,
        default=10,
        help="rounds of early stopping",
    )
    parser.add_argument("--verbose", type=int, default=0, help="verbose")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--save", type=int, default=1, help="whether to save results"
    )
    parser.add_argument(
        "--outfolder",
        type=str,
        default="/home/hans/WorkSpace/Data/Models/FL-adaptive/FedCAMS/",
        help="save folder",
    )

    parser.add_argument(
        "--compressor", type=str, default="sign", help="compressor strategy"
    )
    args = parser.parse_args()
    if args.dataset == "ilsvrc2012":
        args.epochs = 100
    return args
