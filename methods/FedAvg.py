# -*- coding: utf-8 -*-
import copy
import datetime
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from backbone.Model import build_model
from utils import (
    evaluator,
    gen_dataset,
    local_update,
    nn,
    save_args_as_json,
    shutil,
    split_iid_data,
    split_noniid_data,
)


def FedAvg(config):
    rounds = config["TRAIN"]["ROUNDS"]
    batchsize = config["TRAIN"]["BATCH_SIZE"]

    train_dataset, test_dataset, img_size, num_classes = gen_dataset(
        config["DATA"]["TRAIN_DATA"],
        config["DATA"]["IMG_SIZE"],
        config["DATA"]["DATA_ROOT"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=4 * len(config["DEVICE"]["DEVICE_GPUID"]),
        pin_memory=True,
    )
    # select client
    selected_client_num = max(
        int(config["FED"]["FRACTION"] * config["FED"]["CLIENTS_NUM"]), 1
    )
    print(
        f"{selected_client_num} of {config['FED']['CLIENTS_NUM']} clients are selected."
    )
    idxs_client = np.random.choice(
        range(config["FED"]["CLIENTS_NUM"]), selected_client_num, replace=False
    )

    # IID or Non-IID
    if config["DATA"]["IS_IID"]:
        print("IID data")
        dict_users = split_iid_data(
            train_dataset, config["FED"]["CLIENTS_NUM"]
        )
    else:
        print("Non-IID data")
        dict_users = split_noniid_data(
            train_dataset, config["FED"]["CLIENTS_NUM"]
        )

    model_global = build_model(num_classes, config)
    if config["DEVICE"]["DEVICE_TOUSE"] == "GPU":
        model_global.cuda()
        if len(config["DEVICE"]["DEVICE_GPUID"]) > 1:
            model_global = torch.nn.DataParallel(
                model_global,
                device_ids=list(range(len(config["DEVICE"]["DEVICE_GPUID"]))),
            )
    if config["TRAIN"]["FINETUNE"]:
        checkpoint = torch.load(config["TRAIN"]["WEIGHT_TOLOAD"])
        model_global.load_state_dict(checkpoint)

    modelID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if config["NETWORK"]["BACKBONE"] == "resnet":
        model_name = f"{config['NETWORK']['BACKBONE']}{config['NETWORK']['LAYER_NUMBER']}"
    else:
        model_name = config["NETWORK"]["BACKBONE"]
    if "lenet" in model_name:
        model_name = "lenet"
    if config["DATA"]["IS_IID"]:
        save_path = (
            f"{config['TRAIN']['SAVE_ROOT']}/{config['NAME']}/{config['METHODS']}/{config['METHODS']}-"
            f"{model_name}-{config['DATA']['TRAIN_DATA']}-iid-B{str(batchsize).zfill(3)}-{modelID}"
        )
    else:
        save_path = (
            f"{config['TRAIN']['SAVE_ROOT']}/{config['NAME']}/{config['METHODS']}/{config['METHODS']}-"
            f"{model_name}-{config['DATA']['TRAIN_DATA']}-noniid-B{str(batchsize).zfill(3)}-{modelID}"
        )
    print(f"\n>>>>>>>>>>>>> {save_path}\n")
    if not config["DEBUG"]:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # save arguments to local
        args_json_path = save_path + "/args.json"
        save_args_as_json(config, args_json_path)

        # client model path
        model_path = {}
        for idx in idxs_client:
            model_path[idx] = save_path + f"/Model/Client_{idx}/"
            if not os.path.exists(model_path[idx]):
                os.makedirs(model_path[idx])

    locals = {
        idx: local_update(
            config=config,
            client_idx=idx,
            dataset=train_dataset,
            data_idxs=dict_users[idx],
            model=copy.deepcopy(model_global),
            test_loader=test_loader,
        )
        for idx in idxs_client
    }
    weight_global = model_global.state_dict()
    best = 0
    local_best = [0, 0, 0]
    for rd in range(rounds):
        print("\n")
        print("-" * 100)
        print(f"[Round: {rd}/{rounds}]")
        # train
        loss_locals = []
        acc_locals = []
        weight_locals = []
        for idx in idxs_client:
            weight_local, loss_local, acc_local = locals[idx].train(
                weight_global
            )
            if np.mean(acc_local) > local_best[idx] and not config["DEBUG"]:
                local_best[idx] = np.mean(acc_local)
                torch.save(
                    weight_local,
                    model_path[idx]
                    + f"/client:{idx}-epoch:{str(rd).zfill(3)}-"
                      f"trn_loss:{np.round(loss_local, 4)}-"
                      f"trn_acc:{np.round(acc_local, 4)}-{modelID}.pth",
                )
            weight_locals.append(weight_local)
            loss_locals.append(loss_local)
            acc_locals.append(acc_local)
        weight_global = fedavg_weight(weight_locals)
        model_global.load_state_dict(weight_global)

        # test
        test_loss_avg, test_acc_avg = evaluator(
            model_global, test_loader, nn.CrossEntropyLoss(), batchsize
        )
        print(save_path)
        print(
            f"Round {rd}\nLocal loss: {np.mean(loss_locals)}, "
            f"Local Acc: {np.mean(acc_locals)}\n"
            f"Test  Loss: {test_loss_avg}, "
            f"Test  Acc: {test_acc_avg}"
        )
        if np.mean(test_acc_avg) > best and not config["DEBUG"]:
            best = np.mean(test_acc_avg)
            torch.save(
                model_global.state_dict(),
                f'{save_path}/{modelID}-{config["METHODS"]}-round:{str(rd).zfill(3)}-'
                f'tst_loss:{np.round(np.mean(test_loss_avg), 4)}-'
                f'tst_acc:{np.round(np.mean(test_acc_avg), 4)}-best.pth',
            )
    if not config["DEBUG"]:
        log_name = (f'{config["METHODS"]}-{config["NETWORK"]["BACKBONE"]}{config["NETWORK"]["LAYER_NUMBER"]}-'
                    f'{config["DATA"]["TRAIN_DATA"]}.txt')
        shutil.move(
            f"/mnt/4tssd/hans/WorkSpace/FL-adaptive/{log_name}",
            f"{save_path}/{log_name}",
        )


def fedavg_weight(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
