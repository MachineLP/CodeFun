# coding=utf-8
"""
Created on 2017 10.17
@author: liupeng
"""

import numpy as np
import os
import tensorflow as tf
slim = tf.contrib.slim
from lib.data_load.data_load_from_txt_mullabel import data_load_from_txt_mullabel
from lib.model.build_model.build_net import net_arch
from lib.utils.multi_label_utils import g_parameter
from lib.utils.multi_label_utils import to_one_hot
# from lib.train.train3 import train3 as train
from keras.utils import np_utils
import config
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sample_dir = config.sample_dir
tfRecord = config.tfRecord
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


from lib.train.train_multi_label import train_multi_label as train
from lib.train.train_multi_label_tfRecords import train_multi_label_tfRecords as train_tfRecords
train_data, train_label, valid_data, valid_label, train_n, valid_n, note_label = data_load_from_txt_mullabel(sample_dir, train_rate).gen_train_valid()

print ('note_label', note_label)
print (train_data)
print (train_label)
if arch_model!='arch_seg_vgg16_conv' and arch_model!='arch_vgg16_ocr':
    # train_label = np_utils.to_categorical(train_label, num_classes)
    # valid_label = np_utils.to_categorical(valid_label, num_classes)
    train_label = to_one_hot(train_label, num_classes)
    valid_label = to_one_hot(valid_label, num_classes)
print (train_label)
if not os.path.isdir(train_dir):
    os.makedirs(train_dir)
if tfRecord:
    train_file_tfRecord = 'img_train.tfrecords'
    valid_file_tfRecord = 'img_test.tfrecords'
    train_tfRecords(train_file_tfRecord,valid_file_tfRecord,train_dir,num_classes,batch_size,arch_model,learning_r_decay,learning_rate_base,decay_rate,dropout_prob,epoch,height,width,checkpoint_exclude_scopes,early_stop,EARLY_STOP_PATIENCE,fine_tune,train_all_layers,checkpoint_path,train_n,valid_n,g_parameter)
else:
    train(train_data,train_label,valid_data,valid_label,train_dir,num_classes,batch_size,arch_model,learning_r_decay,learning_rate_base,decay_rate,dropout_prob,epoch,height,width,checkpoint_exclude_scopes,early_stop,EARLY_STOP_PATIENCE,fine_tune,train_all_layers,checkpoint_path,train_n,valid_n,g_parameter)

