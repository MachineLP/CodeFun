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

# inception_v4
try:
    from inception_v4 import inception_v4_arg_scope, inception_v4
except:
    from lib.model.inception_v4.inception_v4 import inception_v4_arg_scope, inception_v4
# resnet_v2_50, resnet_v2_101, resnet_v2_152
try:
    from resnet_v2 import resnet_arg_scope, resnet_v2_50
except:
    from lib.model.resnet_v2.resnet_v2 import resnet_arg_scope, resnet_v2_50
# vgg16, vgg19
try:
    from vgg import vgg_arg_scope, vgg_16, vgg_16_conv
except:
    from lib.model.vgg.vgg import vgg_arg_scope, vgg_16, vgg_16_conv

try:
    from alexnet import alexnet_v2_arg_scope, alexnet_v2
except:
    from lib.model.alexnet.alexnet import alexnet_v2_arg_scope, alexnet_v2

try:
    from lp_net import lp_net, lp_net_arg_scope
except:
    from lib.model.lp_net.lp_net import lp_net, lp_net_arg_scope

try:
    from attention import attention
except:
    from lib.model.attention.attention import attention


class net_arch(object):
    
    # def __init__(self):

    def arch_inception_v4(self, X, num_classes, dropout_keep_prob=0.8, is_train=False):
        arg_scope = inception_v4_arg_scope()
        with slim.arg_scope(arg_scope):
            net_vis, end_points = inception_v4(X, is_training=is_train)
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope('Logits_out'):
                # 8 x 8 x 1536
                net = slim.avg_pool2d(net_vis, net_vis.get_shape()[1:3], padding='VALID',
                                      scope='AvgPool_1a_out')
                # 1 x 1 x 1536
                net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b_out')
                net = slim.flatten(net, scope='PreLogitsFlatten_out')
                # 1536
                net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu, scope='Logits_out0')
                net = slim.fully_connected(net, num_classes, activation_fn=None,scope='Logits_out1')
        return net, net_vis

    def arch_resnet_v2_50(self, X, num_classes, dropout_keep_prob=0.8, is_train=False):
        arg_scope = resnet_arg_scope()
        with slim.arg_scope(arg_scope):
            net, end_points = resnet_v2_50(X, is_training=is_train)
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope('Logits_out'):
                #net = slim.conv2d(net, 1000, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_out0')
                #net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b_out0')
                #net = slim.conv2d(net, 200, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_out1')
                #net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b_out1')
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_out2')
                net = tf.squeeze(net,[1,2], name='SpatialSqueeze')
        return net, net_vis

    def arch_vgg16(self, X, num_classes, dropout_keep_prob=0.8, is_train=False):
        arg_scope = vgg_arg_scope()
        with slim.arg_scope(arg_scope):
            net_vis, end_points = vgg_16(X, is_training=is_train)
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope('Logits_out'):
                net = slim.conv2d(net_vis, num_classes, [1, 1],activation_fn=None,normalizer_fn=None,scope='fc8')
                net = tf.squeeze(net,[1,2], name='fc8/squeezed')
        return net, net_vis

    def arch_lp_net(self, X, num_classes, dropout_keep_prob=0.8, is_train=False):
        arg_scope = lp_net_arg_scope()
        with slim.arg_scope(arg_scope):
            net_vis, end_points = lp_net(X, is_training=is_train)
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope('Logits_out'):
                # 8 x 8 x 1536
                net = slim.avg_pool2d(net_vis, net_vis.get_shape()[1:3], padding='VALID',
                                      scope='AvgPool_1a_out')
                # 1 x 1 x 1536
                net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b_out')
                net = slim.flatten(net, scope='PreLogitsFlatten_out')
                # 1536
                net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu, scope='Logits_out0')
                net = slim.fully_connected(net, num_classes, activation_fn=None,scope='Logits_out1')
        return net, net_vis

    def arch_inception_v4_rnn(self, X, num_classes, dropout_keep_prob=0.8, is_train=False):
        rnn_size = 256  
        num_layers = 2  
        arg_scope = inception_v4_arg_scope()
        with slim.arg_scope(arg_scope):
            net_vis, end_points = inception_v4(X, is_training=is_train)
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope('Logits_out'):
                # 8 x 8 x 1536
                orig_shape = net_vis.get_shape().as_list()
                net = tf.reshape(net_vis, [-1, orig_shape[1] * orig_shape[2], orig_shape[3]])
                
                def gru_cell():
                    return tf.contrib.rnn.GRUCell(rnn_size)
                def lstm_cell():  
                    return tf.contrib.rnn.LSTMCell(rnn_size)  
                def attn_cell():  
                    return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=dropout_keep_prob)  
                stack = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(0, num_layers)], state_is_tuple=True)  
                net, _ = tf.nn.dynamic_rnn(stack, net, dtype=tf.float32)  
                net = tf.transpose(net, (1, 0, 2))
                # 1536
                net = slim.fully_connected(net[-1], 256, activation_fn=tf.nn.relu, scope='Logits_out0')
                net = slim.fully_connected(net, num_classes, activation_fn=None,scope='Logits_out1')
        return net, net_vis
 
    def arch_resnet_v2_50_rnn(self, X, num_classes, dropout_keep_prob=0.8, is_train=False):
        rnn_size = 256
        num_layers = 2
        arg_scope = resnet_arg_scope()
        with slim.arg_scope(arg_scope):
            net_vis, end_points = resnet_v2_50(X, is_training=is_train)
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope('Logits_out'):
                orig_shape = net_vis.get_shape().as_list()
                net = tf.reshape(net_vis, [-1, orig_shape[1] * orig_shape[2], orig_shape[3]])
                
                def gru_cell():
                    return tf.contrib.rnn.GRUCell(run_size)
                def lstm_cell():
                    return tf.contrib.rnn.LSTMCell(rnn_size)
                def attn_cell():
                    return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=dropout_keep_prob)
                stack = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(0, num_layers)], state_is_tuple=True)
                net, _ = tf.nn.dynamic_rnn(stack, net, dtype=tf.float32)
                net = tf.transpose(net, (1, 0, 2))
                #
                net = slim.fully_connected(net[-1], 256, activation_fn=tf.nn.relu, scope='Logits_out0')
                net = slim.fully_connected(net, num_classes, activation_fn=None,scope='Logits_out1')
        return net, net_vis

    def arch_inception_v4_rnn_attention(self, X, num_classes, dropout_keep_prob=0.8, is_train=False):
        rnn_size = 256
        num_layers = 2
        attention_size = 64
        arg_scope = inception_v4_arg_scope()
        with slim.arg_scope(arg_scope):
            net_vis, end_points = inception_v4(X, is_training=is_train)
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope('Logits_out'):
                # 8 x 8 x 1536
                orig_shape = net_vis.get_shape().as_list()
                net = tf.reshape(net_vis, [-1, orig_shape[1] * orig_shape[2], orig_shape[3]])

                def gru_cell():
                    return tf.contrib.rnn.GRUCell(rnn_size)
                def lstm_cell():
                    return tf.contrib.rnn.LSTMCell(rnn_size)
                def attn_cell():
                    return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=dropout_keep_prob)
                stack = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(0, num_layers)], state_is_tuple=True)
                net, _ = tf.nn.dynamic_rnn(stack, net, dtype=tf.float32)
                # (B, T, D) => (T, B, D)
                net = tf.transpose(net, (1, 0, 2))
                
                net = attention(net, attention_size, True)
                
                #net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu, scope='Logits_out0')
                net = slim.fully_connected(net, num_classes, activation_fn=None,scope='Logits_out1')
        return net, net_vis


    def arch_resnet_v2_50_rnn_attention(self, X, num_classes, dropout_keep_prob=0.8, is_train=False):
        rnn_size = 256
        num_layers = 2
        attention_size = 64
        arg_scope = resnet_arg_scope()
        with slim.arg_scope(arg_scope):
            net_vis, end_points = resnet_v2_50(X, is_training=is_train)
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope('Logits_out'):
                orig_shape = net_vis.get_shape().as_list()
                net = tf.reshape(net_vis, [-1, orig_shape[1] * orig_shape[2], orig_shape[3]])

                def gru_cell():
                    return tf.contrib.rnn.GRUCell(run_size)
                def lstm_cell():
                    return tf.contrib.rnn.LSTMCell(rnn_size)
                def attn_cell():
                    return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=dropout_keep_prob)
                stack = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(0, num_layers)], state_is_tuple=True)
                net, _ = tf.nn.dynamic_rnn(stack, net, dtype=tf.float32)
                net = tf.transpose(net, (1, 0, 2))

                net = attention(net, attention_size, True)
                #
                #net = slim.fully_connected(net[-1], 256, activation_fn=tf.nn.relu, scope='Logits_out0')
                net = slim.fully_connected(net, num_classes, activation_fn=None,scope='Logits_out1')
        return net, net_vis

    def arch_alexnet_v2(self, X, num_classes, dropout_keep_prob=0.8, is_train=False):
        arg_scope = alexnet_v2_arg_scope()
        with slim.arg_scope(arg_scope):
            net_vis, end_points = alexnet_v2(X, is_training=is_train)
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope('Logits_out'):
                net = slim.conv2d(net_vis, num_classes, [1, 1],activation_fn=None,normalizer_fn=None,scope='fc8')
                net = tf.squeeze(net,[1,2], name='fc8/squeezed')
        return net, net_vis
    
    def arch_multi_alexnet_v2(self, X1, X2, X3, num_classes, dropout_keep_prob=0.8, is_train=False):
        arg_scope = alexnet_v2_arg_scope()
        with slim.arg_scope(arg_scope):
            with tf.variable_scope('arch_alexnet_v2_1'):
                net_vis1, end_points1 = alexnet_v2(X1, is_training=is_train)
            with tf.variable_scope('arch_alexnet_v2_2'):
                net_vis2, end_points2 = alexnet_v2(X2, is_training=is_train)
            with tf.variable_scope('arch_alexnet_v2_3'):
                net_vis3, end_points3 = alexnet_v2(X2, is_training=is_train)
            # net_vis3, end_points3 = alexnet_v2(X3, is_training=is_train)
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope('Logits_out'):
                net_vis = tf.concat([net_vis1, net_vis2, net_vis3],3)
                net = slim.conv2d(net_vis, num_classes, [1, 1],activation_fn=None,normalizer_fn=None,scope='fc8')
                net = tf.squeeze(net,[1,2], name='fc8/squeezed')
        return net, net_vis
    
    def arch_multi_vgg16(self, X1, X2, X3, num_classes, dropout_keep_prob=0.8, is_train=False):
        arg_scope = vgg_arg_scope()
        with slim.arg_scope(arg_scope):
            with tf.variable_scope('arch_multi_vgg16_1'):
                net_vis1, end_points1 = vgg_16(X1, is_training=is_train)
            with tf.variable_scope('arch_multi_vgg16_2'):
                net_vis2, end_points2 = vgg_16(X2, is_training=is_train)
            with tf.variable_scope('arch_multi_vgg16_3'):
                net_vis3, end_points3 = vgg_16(X2, is_training=is_train)
            # net_vis3, end_points3 = alexnet_v2(X3, is_training=is_train)
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope('Logits_out'):
                net_vis = tf.concat([net_vis1, net_vis2, net_vis3],3)
                net = slim.conv2d(net_vis, num_classes, [1, 1],activation_fn=None,normalizer_fn=None,scope='fc8')
                net = tf.squeeze(net,[1,2], name='fc8/squeezed')
        return net, net_vis

    
    def arch_multi_vgg16_conv(self, X1, X2, X3, num_classes, dropout_keep_prob=0.8, is_train=False):
        arg_scope = vgg_arg_scope()
        with slim.arg_scope(arg_scope):
            with tf.variable_scope('arch_multi_vgg16_conv_1'):
                net_vis1, end_points1, _ = vgg_16_conv(X1, is_training=is_train)
            with tf.variable_scope('arch_multi_vgg16_conv_2'):
                net_vis2, end_points2, _ = vgg_16_conv(X2, is_training=is_train)
            with tf.variable_scope('arch_multi_vgg16_conv_3'):
                net_vis3, end_points3, _ = vgg_16_conv(X3, is_training=is_train)
            # net_vis3, end_points3 = alexnet_v2(X3, is_training=is_train)
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope('Logits_out'):
                net_vis11 = slim.max_pool2d(net_vis1, net_vis1.get_shape()[1:3], padding='VALID',scope='AvgPool_11a_out')
                net_vis12 = slim.max_pool2d(net_vis1, net_vis1.get_shape()[1:3], padding='VALID',scope='AvgPool_12a_out')
                net_vis21 = slim.max_pool2d(net_vis2, net_vis2.get_shape()[1:3], padding='VALID',scope='AvgPool_21a_out')
                net_vis22 = slim.max_pool2d(net_vis2, net_vis2.get_shape()[1:3], padding='VALID',scope='AvgPool_22a_out')
                net_vis31 = slim.max_pool2d(net_vis3, net_vis3.get_shape()[1:3], padding='VALID',scope='AvgPool_31a_out')
                net_vis32 = slim.max_pool2d(net_vis3, net_vis3.get_shape()[1:3], padding='VALID',scope='AvgPool_32a_out')
                net_vis = tf.concat([net_vis11,net_vis12,net_vis21,net_vis22,net_vis31,net_vis32],3)
                # 加入一个全连接
                net_vis = slim.flatten(net_vis, scope='PreLogitsFlatten_out')
                net_vis = slim.fully_connected(net_vis, 256, activation_fn=None, scope='Logits_out0')
                net = tf.nn.relu(net_vis)
                net = slim.fully_connected(net, num_classes, activation_fn=None,scope='Logits_out1')
                #net = slim.conv2d(net_vis, num_classes, [1, 1],activation_fn=None,normalizer_fn=None,scope='fc8')
                #net = tf.squeeze(net,[1,2], name='fc8/squeezed')
        return net, net_vis


