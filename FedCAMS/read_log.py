def highest_acc(file_name):
    try:
        with open(file_name, "r") as f:
            lines = f.readlines()
        highest_acc = 0
        for line in lines:
            if "Test Accuracy" in line:
                cur_acc = float(line.split(" ")[3])
                if cur_acc > highest_acc:
                    highest_acc = cur_acc
        print(f"{str(round(highest_acc * 100, 2)).zfill(2)}", end=" & ")
    except Exception:
        print("-", end=" & ")


def main():
    dataset = ["cifar10", "cifar100"]
    network = ["resnet20", "resnet32", "resnet18", "resnet34"]
    method = "fedadam"  # fedams, fedadam, fedadagrad, fedyogi, fedcams
    file_name = f"{method}-lenet-mnist.txt"
    highest_acc(file_name)
    for n in network:
        for d in dataset:
            file_name = f"{method}-{n}-{d}.txt"
            highest_acc(file_name)


if __name__ == "__main__":
    main()
