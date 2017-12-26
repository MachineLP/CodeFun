# -*- coding: utf-8 -*-
"""
Created on 2017 10.17
@author: liupeng
wechat: lp9628
blog: http://blog.csdn.net/u014365862/article/details/78422372
"""

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import numpy as np
import argparse
import os
from PIL import Image
from datetime import datetime
import math
import time
try:
    from load_image import load_database_path, get_next_batch_from_path
except:
    from load_image.load_image import load_database_path, get_next_batch_from_path
try:
    from train import train
except:
    from train_net.train import train
import cv2
import os
from keras.utils import np_utils
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import config

if __name__ == '__main__':

    IMAGE_HEIGHT = config.IMAGE_HEIGHT
    IMAGE_WIDTH = config.IMAGE_WIDTH
    num_classes = config.num_classes
    # epoch
    epoch = config.epoch
    batch_size = config.batch_size
    # 模型的学习率
    learning_rate = config.learning_rate
    keep_prob = config.keep_prob

    ##----------------------------------------------------------------------------##
    # 设置训练样本的占总样本的比例：
    train_rate = config.train_rate
    # 每个类别保存到一个文件中，放在此目录下，只要是二级目录就可以。
    craterDir = config.craterDir

    # 选择需要的模型
    # arch_model="arch_inception_v4";  arch_model="arch_resnet_v2_50"; arch_model="vgg_16"
    arch_model=config.arch_model
    # 设置要更新的参数和加载的参数，目前是非此即彼，可以自己修改哦
    checkpoint_exclude_scopes = config.checkpoint_exclude_scopes
    # 迁移学习模型参数
    checkpoint_path=config.checkpoint_path
    
    ##----------------------------------------------------------------------------##
    print ("-----------------------------load_image.py start--------------------------")
    # 准备训练数据
    X_sample, Y_sample = load_database_path(craterDir)
    image_n = len(X_sample)
    # 样本的总数量
    print ("样本的总数量:")
    print (image_n)
    # 定义90%作为训练样本
    train_n = int(image_n*train_rate)
    valid_n = int(image_n*(1-train_rate))
    train_data, train_label = X_sample[0:train_n], Y_sample[0:train_n]
    # 定位10%作为测试样本
    valid_data, valid_label = X_sample[train_n:image_n], Y_sample[train_n:image_n]
    # ont-hot
    train_label = np_utils.to_categorical(train_label, num_classes)
    valid_label = np_utils.to_categorical(valid_label, num_classes)
    ##----------------------------------------------------------------------------##

    print ("-----------------------------train.py start--------------------------")
    train(train_data,train_label,valid_data,valid_label,train_n,valid_n,IMAGE_HEIGHT,IMAGE_WIDTH,num_classes,epoch,batch_size,keep_prob,
          arch_model, checkpoint_exclude_scopes, checkpoint_path)
