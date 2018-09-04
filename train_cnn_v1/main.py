# coding=utf-8
"""
Created on 2017 10.17
@author: liupeng
"""

import numpy as np
import os
import tensorflow as tf
slim = tf.contrib.slim
from lib.data_load.data_loader import read_inputs
from lib.model.build_model.build_net import net_arch
from lib.utils.utils import g_parameter
# from lib.train.train3 import train3 as train
from keras.utils import np_utils
import config
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sample_dir = config.sample_dir
num_classes = config.num_classes
batch_size = config.batch_size
arch_model = config.arch_model
checkpoint_exclude_scopes = config.checkpoint_exclude_scopes
dropout_prob = config.dropout_prob
train_rate = config.train_rate
epoch = config.epoch
# 是否使用提前终止训练
early_stop = config.early_stop
EARLY_STOP_PATIENCE = config.EARLY_STOP_PATIENCE
# 是否使用learning_rate
learning_r_decay = config.learning_r_decay
learning_rate_base = config.learning_rate_base
decay_rate = config.decay_rate
height, width = config.height, config.width
# 模型保存的路径
train_dir = config.train_dir
# 是否进行fine-tune。 选择fine-tune的的参数
fine_tune = config.fine_tune
# 训练所有层的参数
train_all_layers = config.train_all_layers
# 迁移学习的网络模型
checkpoint_path = config.checkpoint_path

if arch_model=='arch_multi_alexnet_v2' or arch_model=='arch_multi_vgg16' or arch_model=='arch_multi_vgg16_conv':
    from lib.train.train3 import train3 as train
else:
    from lib.train.train import train as train


train_data, train_label, valid_data, valid_label, train_n, valid_n = read_inputs(sample_dir, train_rate, batch_size, is_training=True, num_threads=20)

print (train_data, train_label, valid_data, valid_label)

if not os.path.isdir(train_dir):
    os.makedirs(train_dir)

train(train_data,train_label,valid_data,valid_label,train_n,valid_n,train_dir,num_classes,batch_size,arch_model,learning_r_decay,learning_rate_base,decay_rate,dropout_prob,epoch,height,width,checkpoint_exclude_scopes,early_stop,EARLY_STOP_PATIENCE,fine_tune,train_all_layers,checkpoint_path,g_parameter)

