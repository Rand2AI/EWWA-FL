# -*-coding:utf-8-*-
import json
import os
import sys

__CONFIG__ = None


def get_config(args, file: str = None) -> dict:
    global __CONFIG__
    if file is None:
        file = os.path.dirname(os.path.realpath(__file__)) + "/config.json"
    else:
        file += "/config.json"
    if __CONFIG__ is None:
        try:
            with open(file, "r") as fid:
                __CONFIG__ = json.load(fid)
        except Exception:
            print("Unexpected Error:", sys.exc_info())
    __CONFIG__["METHODS"] = args.method
    __CONFIG__["NETWORK"]["BACKBONE"] = args.network
    __CONFIG__["NETWORK"]["LAYER_NUMBER"] = args.layer
    __CONFIG__["DATA"]["TRAIN_DATA"] = args.dataset
    __CONFIG__["DEVICE"]["DEVICE_GPUID"] = [args.gpu]
    __CONFIG__["TRAIN"]["BATCH_SIZE"] = args.batchsize
    __CONFIG__["OPTIMIZER"]["LEARNING_RATE"] = args.lr
    __CONFIG__["TRAIN"]["ROUNDS"] = args.rounds
    __CONFIG__["DEBUG"] = args.debug
    __CONFIG__["FED"]["OPTIMIZER"] = args.optimizer
    __CONFIG__["DATA"]["IS_IID"] = args.iid
    if args.debug:
        print("\n>>>>>>Debug mode is on.\n")
    if args.network == "lenet":
        __CONFIG__["DATA"]["IMG_SIZE"] = [32, 32]
    elif args.network == "resnet" and args.layer == 20:
        __CONFIG__["DATA"]["IMG_SIZE"] = [32, 32]
    elif args.network == "resnet" and args.layer == 32:
        __CONFIG__["DATA"]["IMG_SIZE"] = [32, 32]
    else:
        __CONFIG__["DATA"]["IMG_SIZE"] = [224, 224]
    if args.dataset == "ilsvrc2012":
        __CONFIG__["TRAIN"]["ROUNDS"] = 100
    return __CONFIG__
