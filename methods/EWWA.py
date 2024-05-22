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
    weight_zero_init,
)


def EWWA(config):
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
    print("model builded.")

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

    model_zero_weihgt = copy.deepcopy(model_global)

    EWWA_obj = EWWAObj(
        model_zero_weihgt.apply(weight_zero_init).state_dict(),
        config["FED"]["OPTIMIZER"],
    )
    print("Aggregation method loaded.")
    modelID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if config["NETWORK"]["BACKBONE"] == "resnet":
        model_name = f"{config['NETWORK']['BACKBONE']}{config['NETWORK']['LAYER_NUMBER']}"
    else:
        model_name = config["NETWORK"]["BACKBONE"]
    if "lenet" in model_name:
        model_name = "lenet"
    if config["DATA"]["IS_IID"]:
        save_path = (
            f"{config['TRAIN']['SAVE_ROOT']}/{config['NAME']}/{config['METHODS']}/"
            f"{config['METHODS']}-{config['FED']['OPTIMIZER']}-{model_name}-{config['DATA']['TRAIN_DATA']}-"
            f"iid-B{str(batchsize).zfill(3)}-{modelID}"
        )
    else:
        save_path = (
            f"{config['TRAIN']['SAVE_ROOT']}/{config['NAME']}/{config['METHODS']}/"
            f"{config['METHODS']}-{config['FED']['OPTIMIZER']}-{model_name}-{config['DATA']['TRAIN_DATA']}-"
            f"noniid-B{str(batchsize).zfill(3)}-{modelID}"
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
    print("local client set up.")
    weight_global = model_global.state_dict()
    test_best = 0
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

        # weight_global_ = fedavg_weight(weight_locals)
        EWWA_obj.t = rd
        weight_global, percent_layer_wise = EWWA_obj.EWWA_weight(
            weight_global, weight_locals
        )
        model_global.load_state_dict(weight_global)

        if not config["DEBUG"]:
            # save percentage to local
            percentage_path = save_path + "/percentage/"
            if not os.path.exists(percentage_path):
                os.makedirs(percentage_path)
            percentage_json_path = (
                percentage_path + f"/rd_{str(rd).zfill(5)}.json"
            )
            save_args_as_json(percent_layer_wise, percentage_json_path)

        # test
        test_loss_avg, test_acc_avg = evaluator(
            model_global, test_loader, nn.CrossEntropyLoss(), batchsize
        )
        print(save_path)
        print(
            f"Local loss: {np.mean(loss_locals)}, "
            f"Local Acc: {np.mean(acc_locals)}\nTest  Loss: {test_loss_avg}, Test  Acc: {test_acc_avg}"
        )
        if np.mean(test_acc_avg) > test_best and not config["DEBUG"]:
            test_best = np.mean(test_acc_avg)
            torch.save(
                model_global.state_dict(),
                f'{save_path}/{modelID}-{config["METHODS"]}-'
                f'round:{str(rd).zfill(3)}-tst_loss:{np.round(np.mean(test_loss_avg), 4)}-'
                f'tst_acc:{np.round(np.mean(test_acc_avg), 4)}-test_best.pth',
            )
    if not config["DEBUG"]:
        log_name = (f'{config["METHODS"]}-{config["NETWORK"]["BACKBONE"]}{config["NETWORK"]["LAYER_NUMBER"]}-'
                    f'{config["DATA"]["TRAIN_DATA"]}.txt')
        shutil.move(
            f"/mnt/4tssd/hans/WorkSpace/FL-adaptive/{log_name}",
            f"{save_path}/{log_name}",
        )


class EWWAObj(object):
    def __init__(
        self,
        m_zero,
        optimiser,
        scale=1.0,
        beta1=0.9,
        beta2=0.999,
        epislon=1e-8,
    ):
        self.optimiser = optimiser
        self.m_zero = m_zero
        self.scale = scale
        self.beta1 = beta1
        self.beta2 = beta2
        self.epislon = epislon
        self.m = copy.deepcopy(self.m_zero)
        self.v = copy.deepcopy(self.m_zero)
        self.t = 0

    def adam(self, g):
        scale = (
            self.scale
            * (1 - self.beta2 ** (self.t + 1)) ** 0.5
            / (1 - self.beta1 ** (self.t + 1))
        )
        # print(f"scale: {scale}")
        result = copy.deepcopy(self.m_zero)
        for k in self.m.keys():
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (
                g[k] * g[k]
            )
            result[k] = (
                scale * self.m[k] / (torch.sqrt(self.v[k]) + self.epislon)
            )
        return result

    def adagrad(self, g):
        scale = (
            self.scale
            * (1 - self.beta2 ** (self.t + 1)) ** 0.5
            / (1 - self.beta1 ** (self.t + 1))
        )
        # print(f"scale: {scale}")
        result = copy.deepcopy(self.m_zero)
        for k in self.m.keys():
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g[k]
            self.v[k] = self.v[k] + (g[k] * g[k])
            result[k] = (
                scale * self.m[k] / (torch.sqrt(self.v[k]) + self.epislon)
            )
        return result

    def yogi(self, g):
        scale = (
            self.scale
            * (1 - self.beta2 ** (self.t + 1)) ** 0.5
            / (1 - self.beta1 ** (self.t + 1))
        )
        # print(f"scale: {scale}")
        result = copy.deepcopy(self.m_zero)
        for k in self.m.keys():
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g[k]
            self.v[k] = self.v[k] - (1 - self.beta2) * (
                g[k] * g[k]
            ) * torch.sign(self.v[k] - (g[k] * g[k]))
            result[k] = (
                scale * self.m[k] / (torch.sqrt(self.v[k]) + self.epislon)
            )
        return result

    def optimise_(self, g):
        result = copy.deepcopy(self.m_zero)
        bias_correction1 = 1 - self.beta1 ** (self.t + 1)
        bias_correction2 = 1 - self.beta2 ** (self.t + 1)
        for k in g.keys():
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g[k]
            self.v[k] = (
                self.beta2 * self.v[k] + (1 - self.beta2) * g[k] * g[k].conj()
            )
            result[k] = (
                self.scale
                * (self.m[k] / bias_correction1)
                / (torch.sqrt(self.v[k] / bias_correction2) + self.epislon)
            )
        return result

    def EWWA_weight(self, w, w_list):
        g_hat = []
        for wi in w_list:
            g = copy.deepcopy(wi)
            for k in w.keys():
                g[k] = w[k] - wi[k]
            if self.optimiser == "adam":
                new_g = self.adam(g)
            elif self.optimiser == "adagrad":
                new_g = self.adagrad(g)
            elif self.optimiser == "yogi":
                new_g = self.yogi(g)
            else:
                raise ValueError("optimiser not supported")
            g_hat.append(new_g)
        # element-wise sum
        s = copy.deepcopy(g_hat[0])
        # small_dict = {}
        temp_small = []
        for k in g_hat[0].keys():
            # find the smallest value in each layer
            for i in range(len(g_hat)):
                temp_small.append(torch.min(g_hat[i][k]).item())
        small_dict = min(temp_small)
        for k in g_hat[0].keys():
            for i in range(len(g_hat)):
                if i == 0:
                    s[k] = torch.log(g_hat[i][k] - small_dict + 1) + 1e-8
                else:
                    s[k] += torch.log(g_hat[i][k] - small_dict + 1) + 1e-8

        w_adp = copy.deepcopy(w_list[0])
        percent_layer_wise = {}
        for k in w_adp.keys():
            temp_percent = []
            for i in range(len(w_list)):
                percent = (torch.log(g_hat[i][k] - small_dict + 1) + 1e-8) / s[
                    k
                ]
                if i == 0:
                    w_adp[k] = percent * w_list[i][k]
                else:
                    w_adp[k] += percent * w_list[i][k]
                temp_percent.append(torch.mean(percent).item())
            percent_layer_wise[k] = temp_percent
        # print(percent_layer_wise)
        return w_adp, percent_layer_wise
