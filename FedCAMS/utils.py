#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from sampling import (
    cifar_iid,
    dataset_noniid,
    mnist_iid,
    mnist_noniid,
    mnist_noniid_unequal,
)
from torchvision import datasets, transforms

# from models.randaug import RandAugment


def get_model(model_name, dataset, img_size, nclass):
    if model_name == "vggnet":
        from models import vgg

        model = vgg.VGG("VGG11", num_classes=nclass)

    elif model_name == "resnet18":
        from models import resnet

        model = resnet.ResNet18(num_classes=nclass)
    elif model_name == "resnet34":
        from models import resnet

        model = resnet.ResNet34(num_classes=nclass)
    elif model_name == "resnet20":
        model = resnet20(num_classes=nclass)
    elif model_name == "resnet32":
        model = resnet32(num_classes=nclass)
    elif model_name == "lenet":
        model = lenet(channel=3, hideen=768, num_classes=nclass)

    elif model_name == "wideresnet":
        from models import wideresnet

        model = wideresnet.WResNet_cifar10(
            num_classes=nclass, depth=16, multiplier=4
        )

    elif model_name == "cnnlarge":
        from models import simple

        model = simple.CNNLarge()

    elif model_name == "convmixer":
        from models import convmixer

        model = convmixer.ConvMixer(n_classes=nclass)

    elif model_name == "cnn":
        from models import simple

        if dataset == "mnist":
            model = simple.CNNMnist(num_classes=nclass, num_channels=1)
        elif dataset == "fmnist":
            model = simple.CNNFashion_Mnist(num_classes=nclass)
        elif dataset == "cifar":
            model = simple.CNNCifar(num_classes=nclass)
    elif model_name == "ae":
        from models import simple

        if dataset == "mnist" or dataset == "fmnist":
            model = simple.Autoencoder()

    elif model_name == "mlp":
        from models import simple

        len_in = 1
        for x in img_size:
            len_in *= x
            model = simple.MLP(dim_in=len_in, dim_hidden=64, dim_out=nclass)
    else:
        exit("Error: unrecognized model")

    return model


class lenet(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super(lenet, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            # nn.BatchNorm2d(12),
            nn.Sigmoid(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            # nn.BatchNorm2d(12),
            nn.Sigmoid(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            # nn.BatchNorm2d(12),
            nn.Sigmoid(),
        )
        self.fc = nn.Sequential(nn.Linear(hideen, num_classes))

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def get_dataset(args):
    """Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    print(args.dataset)
    if args.dataset == "cifar10" or args.dataset == "cifar100":
        if args.dataset == "cifar10":
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
            std = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]
        if args.model == "resnet20" or "resnet32":
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )

            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            transform_train = transforms.Compose(
                [
                    transforms.Resize([224, 224]),
                    transforms.RandomCrop(224, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )

            transform_test = transforms.Compose(
                [
                    transforms.Resize([224, 224]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )

        if args.dataset == "cifar10":
            data_dir = "/home/hans/WorkSpace/Data//cifar10/"

            train_dataset = datasets.CIFAR10(
                data_dir, train=True, download=True, transform=transform_train
            )

            test_dataset = datasets.CIFAR10(
                data_dir, train=False, download=True, transform=transform_test
            )

            num_classes = 10
        elif args.dataset == "cifar100":
            data_dir = "/home/hans/WorkSpace/Data/cifar100/"

            train_dataset = datasets.CIFAR100(
                data_dir, train=True, download=True, transform=transform_train
            )

            test_dataset = datasets.CIFAR100(
                data_dir, train=False, download=True, transform=transform_test
            )

            num_classes = 100
        else:
            print(f"wrong dataset name {args.dataset}")
            raise NotImplementedError()
        # sample training data amongst users
        if args.iid:
            print("IID")
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            print("non-IID")
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = dataset_noniid(train_dataset, args.num_users)

    elif args.dataset == "mnist":
        apply_transform = transforms.Compose(
            [
                transforms.Resize([32, 32]),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )
        train_dataset = datasets.MNIST(
            "/home/hans/WorkSpace/Data/mnist",
            train=True,
            download=True,
            transform=apply_transform,
        )

        test_dataset = datasets.MNIST(
            "/home/hans/WorkSpace/Data/mnist",
            train=False,
            download=True,
            transform=apply_transform,
        )
        num_classes = 10

        # sample training data amongst users
        if args.iid:
            print("IID")
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            print("non-IID")
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(
                    train_dataset, args.num_users
                )
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    elif args.dataset == "ilsvrc2012":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        num_classes = 1000

        root = "/home/hans/WorkSpace/Data//Object/ILSVRC/2012/"
        tt_train = transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.RandomCrop(224, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        tt_tst = transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        train_dataset = datasets.ImageNet(
            root=root, split="train", transform=tt_train
        )
        test_dataset = datasets.ImageNet(
            root=root, split="val", transform=tt_tst
        )
        # sample training data amongst users
        if args.iid:
            print("IID")
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            print("non-IID")
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = dataset_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, num_classes, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def average_parameter_delta(ws, w0):
    w_avg = copy.deepcopy(ws[0])
    for key in range(len(w_avg)):
        w_avg[key] = torch.zeros_like(w_avg[key])
        for i in range(0, len(ws)):
            w_avg[key] += ws[i][key] - w0[key]
        w_avg[key] = torch.div(w_avg[key], len(ws))
    return w_avg


def exp_details(args):
    print("\nExperimental details:")
    print(f"    Model     : {args.model}")
    print(f"    Dataset     : {args.dataset}")
    print(f"    Optimizer : {args.optimizer}")
    print(f"    Learning  : {args.lr}")
    print(f"    Global Rounds   : {args.epochs}\n")

    print("    Federated parameters:")
    if args.iid:
        print("    IID")
    else:
        print("    Non-IID")
    print(f"    Fraction of users  : {args.frac}")
    print(f"    Local Batch size   : {args.local_bs}")
    print(f"    Local Epochs       : {args.local_ep}\n")
    return


def add_params(x, y):
    z = []
    for i in range(len(x)):
        z.append(x[i] + y[i])
    return z


def sub_params(x, y):
    z = []
    for i in range(len(x)):
        z.append(x[i] - y[i])
    return z


def mult_param(alpha, x):
    z = []
    for i in range(len(x)):
        z.append(alpha * x[i])
    return z


def norm_of_param(x):
    z = 0
    for i in range(len(x)):
        z += torch.norm(x[i].flatten(0))
    return z


def _weights_init(m):
    m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A", act="relu"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.act = act
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        if self.act == "relu":
            out = torch.relu(self.bn1(self.conv1(x)))
        else:
            out = torch.sigmoid(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if self.act == "relu":
            out = torch.relu(out)
        else:
            out = torch.sigmoid(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, act="relu"):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.act = act
        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes, planes, stride=stride, act=self.act)
            )
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.act == "relu":
            out = torch.relu(self.bn1(self.conv1(x)))
        else:
            out = torch.sigmoid(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(num_classes, act="relu"):
    return ResNet(BasicBlock, [3, 3, 3], num_classes, act)


def resnet32(num_classes, act="relu"):
    return ResNet(BasicBlock, [5, 5, 5], num_classes, act)
