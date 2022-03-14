### AdaAug
AdaAug: Learning class- and instance-adaptive augmentation policies.

### Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Run Search](#run-adaaug-search)
4. [Run Training](#run-adaaug-training)

### Introduction

AdaAug is a framework that finds class- and instance-adpative augmentation policy to augment a given dataset.

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
