#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import os
import pickle
import time

import numpy as np
import torch
from options import args_parser
from tqdm import tqdm
from update import LocalUpdate, test_inference, update_model_inplace

from utils import average_parameter_delta, average_weights, exp_details, get_dataset

if __name__ == "__main__":
    start_time = time.time()

    args = args_parser()
    exp_details(args)
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(x) for x in [args.gpu]])
    # define paths
    file_name = "/{}_{}_{}_llr[{}]_glr[{}]_eps[{}]_le[{}]_bs[{}]_iid[{}]_mi[{}]_frac[{}].pkl".format(
        args.dataset,
        args.model,
        args.optimizer,
        args.local_lr,
        args.lr,
        args.eps,
        args.local_ep,
        args.local_bs,
        args.iid,
        args.max_init,
        args.frac,
    )

    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    # torch.set_num_threads(1) # limit cpu use
    print("-- pytorch version: ", torch.__version__)

    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # if device != 'cpu':
    #     torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.outfolder):
        os.mkdir(args.outfolder)

    # load dataset and user groups
    train_dataset, test_dataset, num_classes, user_groups = get_dataset(args)

    # Set the model to train and send it to device.
    if args.model == "resnet18":
        from torchvision.models.resnet import resnet18

        global_model = resnet18(pretrained=False)
        num_ftrs = global_model.fc.in_features
        global_model.fc = torch.nn.Linear(num_ftrs, num_classes)
    elif args.model == "resnet34":
        from torchvision.models.resnet import resnet34

        global_model = resnet34(pretrained=False)
        num_ftrs = global_model.fc.in_features
        global_model.fc = torch.nn.Linear(num_ftrs, num_classes)
    elif args.model == "resnet20":
        from models.ResNet_cifar import resnet20

        global_model = resnet20(num_classes=num_classes)
    elif args.model == "resnet32":
        from models.ResNet_cifar import resnet32

        global_model = resnet32(num_classes=num_classes)
    elif args.model == "lenet":
        from models.LeNet import lenet

        global_model = lenet(channel=3, hideen=768, num_classes=num_classes)
    else:
        raise Exception("Error: unrecognized model name ")
        # global_model = get_model(args.model, args.dataset, train_dataset[0][0].shape, num_classes)
    global_model.to(device)
    global_model.train()

    momentum_buffer_list = []
    exp_avgs = []
    exp_avg_sqs = []
    max_exp_avg_sqs = []
    for i, p in enumerate(global_model.parameters()):
        momentum_buffer_list.append(
            torch.zeros_like(
                p.data.detach().clone(), dtype=torch.float, requires_grad=False
            )
        )
        exp_avgs.append(
            torch.zeros_like(
                p.data.detach().clone(), dtype=torch.float, requires_grad=False
            )
        )
        exp_avg_sqs.append(
            torch.zeros_like(
                p.data.detach().clone(), dtype=torch.float, requires_grad=False
            )
        )
        max_exp_avg_sqs.append(
            torch.zeros_like(
                p.data.detach().clone(), dtype=torch.float, requires_grad=False
            )
            + args.max_init
        )  # 1e-2

    # Training
    train_loss_sampled, train_loss, train_accuracy = [], [], []
    test_loss, test_accuracy = [], []
    start_time = time.time()
    for epoch in tqdm(range(args.epochs)):
        ep_time = time.time()

        local_weights, local_params, local_losses = [], [], []
        print(f"\n | Global Training Round : {epoch+1} |\n")

        par_before = []
        for p in global_model.parameters():  # get trainable parameters
            par_before.append(p.data.detach().clone())
        # this is to store parameters before update
        w0 = (
            global_model.state_dict()
        )  # get all parameters, includeing batch normalization related ones

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(
                args=args, dataset=train_dataset, idxs=user_groups[idx]
            )

            w, p, loss = local_model.update_weights_local(
                model=copy.deepcopy(global_model), global_round=epoch
            )

            local_weights.append(copy.deepcopy(w))
            local_params.append(copy.deepcopy(p))
            local_losses.append(copy.deepcopy(loss))

        bn_weights = average_weights(local_weights)
        global_model.load_state_dict(bn_weights)

        # this is to update trainable parameters via different optimizers
        global_delta = average_parameter_delta(
            local_params, par_before
        )  # calculate compression in this function

        update_model_inplace(
            global_model,
            par_before,
            global_delta,
            args,
            epoch,
            momentum_buffer_list,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
        )

        # report and store loss and accuracy
        # this is local training loss on sampled users
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        print(
            "Epoch Run Time: {0:0.4f} of {1} global rounds".format(
                time.time() - ep_time, epoch + 1
            )
        )
        print(f"Training Loss : {train_loss[-1]}")

        global_model.eval()

        # Test inference after completion of training
        test_acc, test_ls = test_inference(args, global_model, test_dataset)
        test_accuracy.append(test_acc)
        test_loss.append(test_ls)

        # print global training loss after every rounds

        print(f"Test Loss : {test_loss[-1]}")
        print(f"Test Accuracy : {test_accuracy[-1]} \n")

        if args.save:
            # Saving the objects train_loss and train_accuracy:
            with open(args.outfolder + file_name, "wb") as f:
                pickle.dump([train_loss, test_loss, test_accuracy], f)

    print("\n Total Run Time: {0:0.4f}".format(time.time() - start_time))
