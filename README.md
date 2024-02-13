# EWWA-FL
This is the official repo for the paper "An Element-Wise Weights Aggregation Method for Federated Learning"

## Introduction

This is the implementation of the paper "An Element-Wise Weights Aggregation Method for Federated Learning". This paper introduces an innovative Element-Wise Weights Aggregation Method for Federated Learning (EWWA-FL) aimed at optimizing learning performance and accelerating convergence speed. Unlike traditional FL approaches, EWWA-FL aggregates local weights to the global model at the level of individual elements, thereby allowing each participating client to make element-wise contributions to the learning process. By taking into account the unique dataset characteristics of each client, EWWA-FL enhances the robustness of the global model to different datasets while also achieving rapid convergence. The method is flexible enough to employ various weighting strategies. Through comprehensive experiments, we demonstrate the advanced capabilities of EWWA-FL, showing significant improvements in both accuracy and convergence speed across a range of backbones and benchmarks.

<div align=center><img src="https://github.com/Rand2AI/EWWA-FL/blob/main/images/intro.png"/></div>

## Requirements

python==3.6.9

torch==1.4.0

torchvision==0.5.0

numpy==1.18.2

tqdm==4.45.0

...

## Performance

<div align=center><img src="https://github.com/Rand2AI/EWWA-FL/blob/main/images/performance"/></div>

<div align=center><img src="https://github.com/Rand2AI/EWWA-FL/blob/main/images/32-10.png"/></div>
<div align=center><img src="https://github.com/Rand2AI/EWWA-FL/blob/main/images/32-10-non.png"/></div>

## Citation

If you find this work helpful for your research, please cite the following paper:

```
@INPROCEEDINGS{10411551,
    author={Hu, Yi and Ren, Hanchi and Hu, Chen and Deng, Jingjing and Xie, Xianghua},
    booktitle={2023 IEEE International Conference on Data Mining Workshops (ICDMW)},
    title={An Element-Wise Weights Aggregation Method for Federated Learning},
    year={2023},
    volume={},
    number={},
    pages={188-196},
    keywords={Federated learning;Neural networks;Distributed databases;Robustness;Reproducibility of results;Optimization;Convergence;Federated Learning;Weights Aggregation;Adaptive Learning},
    doi={10.1109/ICDMW60847.2023.00031}}
```
