# Conditional Similarity Networks (CSNs-Tensorflow)

This repository contains a [Tensorflow](https://github.com/tensorflow/tensorflow) implementation of the paper [Conditional Similarity Networks](https://arxiv.org/abs/1603.07810) presented at CVPR 2017. 

The code is based on the [Tensorflow example for training cnn on Imagenet](https://github.com/MachineLP/train_arch) 


## Table of Contents
0. [Introduction](#introduction)
0. [Usage](#usage)
0. [Contact](#contact)

## Introduction
What makes images similar? To measure the similarity between images, they are typically embedded in a feature-vector space, in which their distance preserve the relative dissimilarity. However, when learning such similarity embeddings the simplifying assumption is commonly made that images are only compared to one unique measure of similarity.

[Conditional Similarity Networks](https://arxiv.org/abs/1603.07810) address this shortcoming by learning a nonlinear embeddings that gracefully deals with multiple notions of similarity within a shared embedding. Different aspects of similarity are incorporated by assigning responsibility weights to each embedding dimension with respect to each aspect of similarity.

<img src="https://github.com/MachineLP/conditional-similarity-networks-Tensorflow/blob/master/csn_overview.png?raw=true" width="600">

Images are passed through a convolutional network and projected into a nonlinear embedding such that different dimensions encode features for specific notions of similarity. Subsequent masks indicate which dimensions of the embedding are responsible for separate aspects of similarity. We can then compare objects according to various notions of similarity by selecting an appropriate masked subspace.

## Usage
The detault setting for this repo is a CSN with fixed masks, an embedding dimension 128 and four notions of similarity.

You can download the Zappos dataset as well as the training, validation and test triplets used in the paper with

```sh
python get_data.py
```

The network can be simply trained with `python train.py` or with optional arguments for different hyperparameters:
```sh
$ python train.py 
```

## Contact

Any discussions, suggestions and questions are welcome!
