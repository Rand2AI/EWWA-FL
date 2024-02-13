import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing


def std_deviation_iid_ewwafl():
    a = [98, 97.99, 98]
    b = [89.73, 89.74, 89.43]
    c = [64.14, 64.16, 64.10]
    d = [90.17, 90.17, 90.10]
    e = [65.63, 65.84, 65.38]
    f = [90.77, 91.17, 90.88]
    g = [70.98, 70.34, 70.78]
    h = [65.64, 65.64, 65.54]
    i = [91.23, 91.13, 91.13]
    j = [70.15, 69.77, 70.34]
    k = [68.23, 68.33, 68.27]

    dataset = [a, b, c, d, e, f, g, h, i, j, k]
    std_list = [np.std(x) for x in dataset]
    for x in dataset:
        print(np.std(x))
    print(np.mean(std_list))


def std_deviation_iid_fedopt():
    a = [73.59, 64.63, 65.93]
    b = [74.84, 67.36, 71.90]
    c = [78.07, 73.88]
    d = [81.71, 71.8]
    dataset = [a, b, c, d]

    std_list = [np.std(x) for x in dataset]
    for x in dataset:
        print(np.std(x))
    print(np.mean(std_list))


def read_loss_acc_from_file(
    method, model, dataset, iid, optimizer=None, norm=False
):
    root_path = "/home/hans/WorkSpace/FL-adaptive/"
    train_loss = []
    test_acc = []
    # FedCAMS, FedAMS, FedOpt
    if method != "FedAdp" and method != "FedAvg":
        root_path += "FedCAMS/"
        if iid == 1:
            filename = f"{method}-{model}-{dataset}-iid.txt"
        else:
            filename = f"{method}-{model}-{dataset}-noniid.txt"
        with open(root_path + filename, "r") as f:
            lines = f.readlines()
        for line in lines:
            if "Training Loss : " in line:
                value = line.split("Training Loss : ")[-1]
                if value == "nan":
                    raise Exception(f"Training Loss is nan. {filename}")
                train_loss.append(float(value))
            if "Test Accuracy : " in line:
                value = line.split("Test Accuracy : ")[-1]
                if value == "nan":
                    raise Exception(f"Test Accuracy is nan. {filename}")
                test_acc.append(float(value))
    # FedAdp, FedAvg
    else:
        if iid == 1:
            filename = f"{method}-{optimizer}-{model}-{dataset}-iid.txt"
        else:
            filename = f"{method}-{optimizer}-{model}-{dataset}-noniid.txt"
        with open(root_path + filename, "r") as f:
            lines = f.readlines()
        for line in lines:
            if "Local loss: " in line:
                value = line.split(",")[0].split("Local loss: ")[-1]
                if value == "nan":
                    raise Exception(f"Training Loss is nan. {filename}")
                train_loss.append(float(value))
            if " Test  Acc: " in line:
                value = line.split(",")[-1].split(" Test  Acc: ")[-1]
                if value == "nan":
                    raise Exception(f"Test Accuracy is nan. {filename}")
                test_acc.append(float(value))
    if norm:
        scaler = preprocessing.MinMaxScaler()
        train_loss = list(
            scaler.fit_transform(np.array(train_loss).reshape(-1, 1)).reshape(
                -1
            )
        )
        test_acc = list(
            scaler.fit_transform(np.array(test_acc).reshape(-1, 1)).reshape(-1)
        )
    return train_loss, test_acc


def vis_32_10_iid():
    iid = 1
    model = "resnet32"
    dataset = "cifar10"
    train_loss_1, test_acc_1 = read_loss_acc_from_file(
        "FedAdp", model, dataset, iid, "adagrad"
    )
    train_loss_3, test_acc_3 = read_loss_acc_from_file(
        "fedams", model, dataset, iid, None
    )
    train_loss_4, test_acc_4 = read_loss_acc_from_file(
        "fedadam", model, dataset, iid, None
    )
    train_loss_4 = train_loss_4[10::]
    test_acc_4 = test_acc_4[10::]
    scaler = preprocessing.MinMaxScaler()
    train_loss_4 = list(
        scaler.fit_transform(np.array(train_loss_4).reshape(-1, 1)).reshape(-1)
    )
    test_acc_4 = list(
        scaler.fit_transform(np.array(test_acc_4).reshape(-1, 1)).reshape(-1)
    )

    plt.figure(figsize=(10, 8))
    plt.plot(train_loss_1, label="EWWA-FL", marker="o")
    plt.plot(train_loss_3, label="FedCAMS", marker="^")
    plt.plot(train_loss_4, label="FedOpt", marker="1")

    plt.legend()
    plt.xlabel("Round")
    plt.ylabel("Loss")
    if iid == 1:
        plt.title(f"{model}-{dataset}-iid")
    else:
        plt.title(f"{model}-{dataset}-noniid")
    plt.tight_layout(pad=1)
    plt.show()


def vis_32_10_non():
    iid = 0
    model = "resnet32"
    dataset = "cifar10"
    train_loss_1, test_acc_1 = read_loss_acc_from_file(
        "FedAdp", model, dataset, iid, "adam"
    )
    train_loss_2, test_acc_2 = read_loss_acc_from_file(
        "fedcams", model, dataset, iid, None
    )
    # train_loss_3, test_acc_3 = read_loss_acc_from_file('fedams', model, dataset, iid, None)
    train_loss_4, test_acc_4 = read_loss_acc_from_file(
        "fedadam", model, dataset, iid, None
    )
    train_loss_4 = train_loss_4[17::]
    test_acc_4 = test_acc_4[17::]
    scaler = preprocessing.MinMaxScaler()
    train_loss_4 = list(
        scaler.fit_transform(np.array(train_loss_4).reshape(-1, 1)).reshape(-1)
    )
    test_acc_4 = list(
        scaler.fit_transform(np.array(test_acc_4).reshape(-1, 1)).reshape(-1)
    )

    plt.figure(figsize=(10, 8))
    plt.plot(train_loss_1, label="EWWA-FL", marker="o")
    plt.plot(train_loss_2, label="FedCAMS", marker="*")
    # plt.plot(train_loss_3, label='FedAMS', marker='^')
    plt.plot(train_loss_4, label="FedOpt", marker="1")
    plt.legend()
    plt.xlabel("Round")
    plt.ylabel("Loss")
    if iid == 1:
        plt.title(f"{model}-{dataset}-iid")
    else:
        plt.title(f"{model}-{dataset}-noniid")
    plt.tight_layout(pad=1)
    plt.show()


def vis_18_10_iid():
    iid = 1
    model = "resnet18"
    dataset = "cifar10"
    train_loss_1, test_acc_1 = read_loss_acc_from_file(
        "FedAdp", model, dataset, iid, "adam"
    )
    train_loss_3, test_acc_3 = read_loss_acc_from_file(
        "fedams", model, dataset, iid, None
    )
    train_loss_4, test_acc_4 = read_loss_acc_from_file(
        "fedadam", model, dataset, iid, None
    )
    train_loss_4 = train_loss_4[17::]
    test_acc_4 = test_acc_4[17::]
    scaler = preprocessing.MinMaxScaler()
    train_loss_4 = list(
        scaler.fit_transform(np.array(train_loss_4).reshape(-1, 1)).reshape(-1)
    )
    test_acc_4 = list(
        scaler.fit_transform(np.array(test_acc_4).reshape(-1, 1)).reshape(-1)
    )

    plt.figure(figsize=(10, 8))
    plt.plot(train_loss_1, label="EWWA-FL", marker="o")
    plt.plot(train_loss_3, label="FedCAMS", marker="^")
    plt.plot(train_loss_4, label="FedOpt", marker="1")
    plt.legend()
    plt.xlabel("Round")
    plt.ylabel("Loss")
    if iid == 1:
        plt.title(f"{model}-{dataset}-iid")
    else:
        plt.title(f"{model}-{dataset}-noniid")
    plt.tight_layout(pad=1)
    plt.show()


def vis_18_10_non():
    iid = 0
    model = "resnet18"
    dataset = "cifar10"
    train_loss_1, test_acc_1 = read_loss_acc_from_file(
        "FedAdp", model, dataset, iid, "adam"
    )
    train_loss_2, test_acc_2 = read_loss_acc_from_file(
        "fedcams", model, dataset, iid, None
    )
    # train_loss_3, test_acc_3 = read_loss_acc_from_file('fedams', model, dataset, iid, None)

    plt.figure(figsize=(10, 8))
    plt.plot(train_loss_1, label="EWWA-FL", marker="o")
    plt.plot(train_loss_2, label="FedCAMS", marker="*")
    # plt.plot(train_loss_3, label='FedAMS', marker='^')
    plt.legend()
    plt.xlabel("Round")
    plt.ylabel("Loss")
    if iid == 1:
        plt.title(f"{model}-{dataset}-iid")
    else:
        plt.title(f"{model}-{dataset}-noniid")
    plt.tight_layout(pad=1)
    plt.show()


def vis_20_10_iid():
    iid = 1
    model = "resnet20"
    dataset = "cifar10"
    train_loss_1, test_acc_1 = read_loss_acc_from_file(
        "FedAdp", model, dataset, iid, "adam"
    )
    train_loss_3, test_acc_3 = read_loss_acc_from_file(
        "fedams", model, dataset, iid, None
    )
    train_loss_4, test_acc_4 = read_loss_acc_from_file(
        "fedadam", model, dataset, iid, None
    )
    train_loss_4 = train_loss_4[17::]
    test_acc_4 = test_acc_4[17::]
    scaler = preprocessing.MinMaxScaler()
    train_loss_4 = list(
        scaler.fit_transform(np.array(train_loss_4).reshape(-1, 1)).reshape(-1)
    )
    test_acc_4 = list(
        scaler.fit_transform(np.array(test_acc_4).reshape(-1, 1)).reshape(-1)
    )

    plt.figure(figsize=(10, 8))
    plt.plot(train_loss_1, label="EWWA-FL", marker="o")
    plt.plot(train_loss_3, label="FedCAMS", marker="^")
    plt.plot(train_loss_4, label="FedOpt", marker="1")

    plt.legend()
    plt.xlabel("Round")
    plt.ylabel("Loss")
    if iid == 1:
        plt.title(f"{model}-{dataset}-iid")
    else:
        plt.title(f"{model}-{dataset}-noniid")
    plt.tight_layout(pad=1)
    plt.show()


def vis_20_10_non():
    iid = 0
    model = "resnet20"
    dataset = "cifar10"
    train_loss_1, test_acc_1 = read_loss_acc_from_file(
        "FedAdp", model, dataset, iid, "adam"
    )
    train_loss_2, test_acc_2 = read_loss_acc_from_file(
        "fedcams", model, dataset, iid, None
    )
    train_loss_3, test_acc_3 = read_loss_acc_from_file(
        "fedams", model, dataset, iid, None
    )
    train_loss_4, test_acc_4 = read_loss_acc_from_file(
        "fedadam", model, dataset, iid, None
    )
    train_loss_4 = train_loss_4[17::]
    test_acc_4 = test_acc_4[17::]
    scaler = preprocessing.MinMaxScaler()
    train_loss_4 = list(
        scaler.fit_transform(np.array(train_loss_4).reshape(-1, 1)).reshape(-1)
    )
    test_acc_4 = list(
        scaler.fit_transform(np.array(test_acc_4).reshape(-1, 1)).reshape(-1)
    )

    plt.figure(figsize=(10, 8))
    plt.plot(train_loss_1, label="EWWA-FL", marker="o")
    plt.plot(train_loss_2, label="FedCAMS", marker="*")
    # plt.plot(train_loss_3, label='FedAMS', marker='^')
    plt.plot(train_loss_4, label="FedOpt", marker="1")

    plt.legend()
    plt.xlabel("Round")
    plt.ylabel("Loss")
    if iid == 1:
        plt.title(f"{model}-{dataset}-iid")
    else:
        plt.title(f"{model}-{dataset}-noniid")
    plt.tight_layout(pad=1)
    plt.show()


def vis_18_100_iid():
    iid = 1
    model = "resnet18"
    dataset = "cifar100"
    train_loss_1, test_acc_1 = read_loss_acc_from_file(
        "FedAdp", model, dataset, iid, "adam"
    )
    train_loss_3, test_acc_3 = read_loss_acc_from_file(
        "fedams", model, dataset, iid, None
    )

    plt.figure(figsize=(10, 8))
    plt.plot(train_loss_1, label="EWWA-FL", marker="o")
    plt.plot(train_loss_3, label="FedCAMS", marker="^")

    plt.legend()
    plt.xlabel("Round")
    plt.ylabel("Loss")
    if iid == 1:
        plt.title(f"{model}-{dataset}-iid")
    else:
        plt.title(f"{model}-{dataset}-noniid")
    plt.tight_layout(pad=1)
    plt.show()


def vis_18_100_non():
    iid = 0
    model = "resnet18"
    dataset = "cifar100"
    train_loss_1, test_acc_1 = read_loss_acc_from_file(
        "FedAdp", model, dataset, iid, "adam"
    )
    train_loss_2, test_acc_2 = read_loss_acc_from_file(
        "fedcams", model, dataset, iid, None
    )

    plt.figure(figsize=(10, 8))
    plt.plot(train_loss_1, label="EWWA-FL", marker="o")
    plt.plot(train_loss_2, label="FedCAMS", marker="*")

    plt.legend()
    plt.xlabel("Round")
    plt.ylabel("Loss")
    if iid == 1:
        plt.title(f"{model}-{dataset}-iid")
    else:
        plt.title(f"{model}-{dataset}-noniid")
    plt.tight_layout(pad=1)
    plt.show()


def visualize_loss_acc():
    _ = [
        "FedAdp",
        "FedAvg",
        "fedams",
        "fedcams",
        "fedadam",
        "fedadagrad",
        "fedyogi",
    ]
    # vis_18_10_iid()
    vis_18_10_non()
    # vis_18_100_iid()
    # vis_18_100_non()
    # vis_20_10_iid()
    # vis_20_10_non()
    # vis_32_10_iid()
    # vis_32_10_non()


if __name__ == "__main__":
    visualize_loss_acc()
