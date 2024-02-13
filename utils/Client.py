# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils import set_optimizer, trainer


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class local_update(object):
    def __init__(
        self, config, client_idx, dataset, data_idxs, model, test_loader
    ):
        self.test_loader = test_loader
        self.model = model
        self.client_idx = client_idx
        self.data_idxs = data_idxs
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = DataLoader(
            DatasetSplit(dataset, data_idxs),
            batch_size=self.config["TRAIN"]["BATCH_SIZE"],
            shuffle=True,
            num_workers=4 * len(self.config["DEVICE"]["DEVICE_GPUID"]),
            pin_memory=True,
        )
        self.optimizer = set_optimizer(self.model.parameters(), self.config)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[
                config["TRAIN"]["ROUNDS"] // 3,
                2 * (config["TRAIN"]["ROUNDS"] // 3),
            ],
            gamma=0.1,
        )
        # self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
        #                                                          milestones=[60, 120, 160],
        #                                                          gamma=0.2)

    def train(self, weight_global):
        self.model.load_state_dict(weight_global)

        epoch_loss = []
        epoch_acc = []
        for epoch in range(self.config["FED"]["CLIENT_EPOCH"]):
            print(
                f"[Client {self.client_idx} Epoch: {epoch + 1}/{self.config['FED']['CLIENT_EPOCH']}]"
            )
            train_epoch_loss_avg, train_epoch_acc_avg = trainer(
                self.model,
                self.train_loader,
                self.optimizer,
                self.criterion,
                self.config["TRAIN"]["BATCH_SIZE"],
            )
            epoch_loss.append(train_epoch_loss_avg)
            epoch_acc.append(train_epoch_acc_avg)
            print("-" * 10)
        self.lr_scheduler.step()
        local_weights = self.model.state_dict()
        return local_weights, np.mean(epoch_loss), np.mean(epoch_acc)
