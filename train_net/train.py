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
import cv2

from keras.utils import np_utils
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

try:
    from load_image import load_database_path, get_next_batch_from_path
except:
    from load_image.load_image import load_database_path, get_next_batch_from_path
# inception_v4
try:
    from inception_v4 import inception_v4_arg_scope, inception_v4
except:
    from net.inception_v4.inception_v4 import inception_v4_arg_scope, inception_v4
# resnet_v2_50, resnet_v2_101, resnet_v2_152
try:
    from resnet_v2 import resnet_arg_scope, resnet_v2_50
except:
    from net.resnet_v2.resnet_v2 import resnet_arg_scope, resnet_v2_50
# vgg16, vgg19
try:
    from vgg import vgg_arg_scope, vgg_16
except:
    from net.vgg.vgg import vgg_arg_scope, vgg_16


def arch_inception_v4(X, num_classes, dropout_keep_prob=0.8, is_train=False):
    arg_scope = inception_v4_arg_scope()
    with slim.arg_scope(arg_scope):
        net, end_points = inception_v4(X, is_training=is_train)
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
        with tf.variable_scope('Logits_out'):
            # 8 x 8 x 1536
            net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                      scope='AvgPool_1a_out')
            # 1 x 1 x 1536
            net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b_out')
            net = slim.flatten(net, scope='PreLogitsFlatten_out')
            # 1536
            net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu, scope='Logits_out0')
            net = slim.fully_connected(net, num_classes, activation_fn=None,scope='Logits_out1')
    return net

def arch_resnet_v2_50(X, num_classes, dropout_keep_prob=0.8, is_train=False):
    arg_scope = resnet_arg_scope()
    with slim.arg_scope(arg_scope):
        net, end_points = resnet_v2_50(X, is_training=is_train)
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
        with tf.variable_scope('Logits_out'):
            net = slim.conv2d(net, 1000, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_out0')
            net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b_out0')
            net = slim.conv2d(net, 200, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_out1')
            net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b_out1')
            net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_out2')
            net = tf.squeeze(net,[1,2], name='SpatialSqueeze')
    return net

def arch_vgg16(X, num_classes, dropout_keep_prob=0.8, is_train=False):
    arg_scope = vgg_arg_scope()
    with slim.arg_scope(arg_scope):
        net, end_points = vgg_16(X, is_training=is_train)
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
        with tf.variable_scope('Logits_out'):
            net = slim.conv2d(net, num_classes, [1, 1],activation_fn=None,normalizer_fn=None,scope='fc8')
            net = tf.squeeze(net,[1,2], name='fc8/squeezed')
    return net


def g_parameter(checkpoint_exclude_scopes):
    exclusions = []
    if checkpoint_exclude_scopes:
        exclusions = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
    print (exclusions)
    # 需要加载的参数。
    variables_to_restore = []
    # 需要训练的参数
    variables_to_train = []
    #for var in slim.get_model_variables():
    for var in tf.trainable_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                variables_to_train.append(var)
                print ("ok")
                print (var.op.name)
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore,variables_to_train


def train(train_data,train_label,valid_data,valid_label,train_n,valid_n,IMAGE_HEIGHT,IMAGE_WIDTH,num_classes,epoch,batch_size=64,keep_prob=0.8,
           arch_model="arch_inception_v4",checkpoint_exclude_scopes="Logits_out", checkpoint_path="pretrain/inception_v4/inception_v4.ckpt"):

    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    #Y = tf.placeholder(tf.float32, [None, 4])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    is_train = tf.placeholder(tf.bool, name='is_train')
    keep_prob = tf.placeholder(tf.float32) # dropout

    # 定义模型
    if arch_model == "arch_inception_v4":
        net = arch_inception_v4(X, num_classes, keep_prob, is_train)
    elif arch_model == "arch_resnet_v2_50":
        net = arch_resnet_v2_50(X, num_classes, keep_prob, is_train)
    elif arch_model == "vgg_16":
        net = arch_vgg16(X, num_classes, keep_prob, is_train)

    # 
    variables_to_restore,variables_to_train = g_parameter(checkpoint_exclude_scopes)

    # loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = net))
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = net))

    var_list = variables_to_train
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss, var_list=var_list)
    predict = tf.reshape(net, [-1, num_classes])
    max_idx_p = tf.argmax(predict, 1)
    max_idx_l = tf.argmax(Y, 1)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    #------------------------------------------------------------------------------------#
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    saver2 = tf.train.Saver(tf.global_variables())
    model_path = 'model/fine-tune'

    net_vars = variables_to_restore
    saver_net = tf.train.Saver(net_vars)
    # checkpoint_path = 'pretrain/inception_v4.ckpt'
    saver_net.restore(sess, checkpoint_path)

    # saver2.restore(sess, "model/fine-tune-1120")
    for epoch_i in range(epoch):
        for batch_i in range(int(train_n/batch_size)):

            images_train, labels_train = get_next_batch_from_path(train_data, train_label, batch_i, IMAGE_HEIGHT, IMAGE_WIDTH, batch_size=batch_size, is_train=True)

            los, _ = sess.run([loss,optimizer], feed_dict={X: images_train, Y: labels_train, keep_prob:0.8, is_train:True})
            # print (los)

            if batch_i % 20 == 0:
                loss_, acc_ = sess.run([loss, accuracy], feed_dict={X: images_train, Y: labels_train, keep_prob:1.0, is_train:False})
                print('Batch: {:>2}: Training loss: {:>3.5f}, Training accuracy: {:>3.5f}'.format(batch_i, loss_, acc_))

            elif batch_i % 100 == 0:
                images_valid, labels_valid = get_next_batch_from_path(valid_data, valid_label, batch_i%(valid_n/batch_size), IMAGE_HEIGHT, IMAGE_WIDTH, batch_size=batch_size, is_train=False)
                ls, acc = sess.run([loss, accuracy], feed_dict={X: images_valid, Y: labels_valid, keep_prob:1.0, is_train:False})
                print('Batch: {:>2}: Validation loss: {:>3.5f}, Validation accuracy: {:>3.5f}'.format(batch_i, ls, acc))
                if acc > 0.90:
                    saver2.save(sess, model_path, global_step=batch_i, write_meta_graph=False)
    sess.close()

if __name__ == '__main__':

    IMAGE_HEIGHT = 299
    IMAGE_WIDTH = 299
    num_classes = 4
    # epoch
    epoch = 100
    batch_size = 16
    # 模型的学习率
    learning_rate = 0.00001
    keep_prob = 0.8

    
    ##----------------------------------------------------------------------------##
    # 设置训练样本的占总样本的比例：
    train_rate = 0.9
    # 每个类别保存到一个文件中，放在此目录下，只要是二级目录就可以。
    craterDir = "train"
    # arch_model="arch_inception_v4";  arch_model="arch_resnet_v2_50"; arch_model="vgg_16"
    arch_model="arch_inception_v4"
    checkpoint_exclude_scopes = "Logits_out"
    checkpoint_path="pretrain/inception_v4/inception_v4.ckpt"
    
    ##----------------------------------------------------------------------------##
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
          arch_model,checkpoint_exclude_scopes, checkpoint_path)
