# -*-coding:utf-8-*-

import torch
from torchvision.models.resnet import resnet18, resnet34
from torchvision.models.vgg import vgg16

from backbone.LeNet import lenet
from backbone.ResNet_cifar import resnet20, resnet32


def build_model(num_classes, config, act="relu"):
    if config["NETWORK"]["BACKBONE"] == "lenet":
        net = lenet(channel=3, hideen=768, num_classes=num_classes)
    elif config["NETWORK"]["BACKBONE"] == "vgg16":
        net = vgg16(num_classes=num_classes)
    elif config["NETWORK"]["BACKBONE"] == "resnet":
        if config["NETWORK"]["LAYER_NUMBER"] == 18:
            net = resnet18(pretrained=False)
            num_ftrs = net.fc.in_features
            net.fc = torch.nn.Linear(num_ftrs, num_classes)
        elif config["NETWORK"]["LAYER_NUMBER"] == 34:
            net = resnet34(pretrained=False)
            num_ftrs = net.fc.in_features
            net.fc = torch.nn.Linear(num_ftrs, num_classes)
        elif config["NETWORK"]["LAYER_NUMBER"] == 20:
            net = resnet20(num_classes=num_classes)
        elif config["NETWORK"]["LAYER_NUMBER"] == 32:
            net = resnet32(num_classes=num_classes)
        else:
            raise Exception("Wrong ResNet Layer Number.")
    else:
        raise Exception("Wrong Backbone Name.")
    return net
