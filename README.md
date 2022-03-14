### AdaAug
AdaAug: Learning class- and instance-adaptive augmentation policies.

### Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Run Search](#run-adaaug-search)
4. [Run Training](#run-adaaug-training)
5. [Citation](#citation)
6. [References & Opensources](#references-&-opensources)

### Introduction

AdaAug is a framework that finds class- and instance-adaptive data augmentation policies to augment a given dataset.

This repository contains code for the work "AdaAug: Learning class- and instance-adaptive data augmentation policies" (https://openreview.net/forum?id=rWXfFogxRJN) implemented using the PyTorch library.

### Getting Started
Code supports Python 3.

####  Install requirements

```shell
pip install -r requirements.txt
```

### Run AdaAug search
Script to search for the augmentation policy for is located in `scripts/search.sh`. Pass the dataset name as the argument to call the script.

For example, to search for the augmentation policy for reduced_svhn dataset:

```shell
bash scripts/search.sh reduced_svhn
```

The training log and candidate policies of the search will be output to the `./search` directory.

### Run AdaAug training
To use the searched policy, paste the path of the g_model and h_model as the G and H variables respectively in `scripts/train.sh`. The path should look like this (./search/...). Then, pass the dataset name as the argument to call the script located in `scripts/train.sh`. The results will be output to the `./eval` directory

```shell
bash scripts/train.sh reduced_svhn
```

### Citation
If you use this code in your research, please cite our paper.
```
@inproceedings{cheung2022adaaug,
  title     =  {AdaAug: Learning class and instance-adaptive data augmentation policies},
  author    =  {Tsz-Him Cheung and Dit-Yan Yeung},
  booktitle =  {International Conference on Learning Representations},
  year      =  {2022},
  url       =  {https://openreview.net/forum?id=rWXfFogxRJN}
}
```

### References & Opensources
Part of our implementation is adopted from the Fast AutoAugment and DADA repositories.
- Fast AutoAugment (https://github.com/kakaobrain/fast-autoaugment)
- DADA (https://github.com/VDIGPKU/DADA)
