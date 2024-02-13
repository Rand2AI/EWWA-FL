# -*- coding: utf-8 -*-
import json

import torch.nn as nn


def save_args_as_json(FLconfig, path):
    with open(str(path), "w") as f:
        json.dump(FLconfig, f, indent=4)


def weight_zero_init(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.constant_(m.weight, 0)
        try:
            nn.init.constant_(m.bias, 0)
        except AttributeError:
            pass
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 0)
        nn.init.constant_(m.bias, 0)
