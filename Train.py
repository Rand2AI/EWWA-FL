# -*-coding:utf-8-*-
import argparse
import os

import torch

from utils import get_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        default="FedAdp",
        help="Normal, FedAvg, EWWA",
    )
    parser.add_argument(
        "--optimizer",
        "-o",
        type=str,
        default="adagrad",
        help="adam, adagrad, yogi",
    )
    parser.add_argument(
        "--network",
        "-n",
        type=str,
        default="lenet",
        help="lenet, resnet, vgg16",
    )
    parser.add_argument(
        "--layer",
        "-l",
        type=int,
        default=0,
        help="18, 34, 20, 32 for resnet only",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="mnist",
        help="mnist, cifar10, cifar100, ilsvrc2012",
    )
    parser.add_argument("--gpu", "-g", type=int, default=3, help="gpu id")
    parser.add_argument(
        "--batchsize", "-b", type=int, default=64, help="batchsize"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--rounds", "-r", type=int, default=500, help="rounds")
    parser.add_argument(
        "--iid", type=int, default=0, help="1 for iid, 0 for non-iid"
    )
    parser.add_argument("--debug", type=int, default=0, help="debug mode")
    args = parser.parse_args()

    config = get_config(args, os.path.dirname(os.path.realpath(__file__)))
    if config["DEVICE"]["DEVICE_TOUSE"] == "GPU":
        seed = 0
        torch.manual_seed(seed)  # sets the seed for generating random numbers.
        torch.cuda.manual_seed(seed)
        # Sets the seed for generating random numbers for the current GPU.
        # It’s safe to call this functionif CUDA is not available;
        # in that case, it is silently ignored.
        torch.cuda.manual_seed_all(seed)
        # Sets the seed for generating random numbers on all GPUs.
        # It’s safe to call this function if CUDA is not available;
        # in that case, it is silently ignored.

        if seed == 0:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch.multiprocessing.set_start_method("spawn")
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in config["DEVICE"]["DEVICE_GPUID"]]
        )
    else:
        raise Exception("Current version does not support CPU yet.")
    eval(config["METHODS"])(config)
