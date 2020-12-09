# -*- coding: utf-8 -*-
"""
Created on 2018 06.18
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

try:
    from vgg import vgg_arg_scope, vgg_16, vgg_16_conv
except:
    from csn_lib.vgg import vgg_arg_scope, vgg_16, vgg_16_conv
    
try:
    from resnet_v2 import resnet_arg_scope, resnet_v2_50
except:
    from csn_lib.resnet_v2 import resnet_arg_scope, resnet_v2_50

try:
    from inception_v4 import inception_v4_arg_scope, inception_v4
except:
    from csn_lib.inception_v4 import inception_v4_arg_scope, inception_v4

def l2norm_embed(x):
    norm2 = tf.norm(x, ord=2, axis=1)
    norm2 = tf.reshape(norm2,[-1,1])
    l2norm = x/norm2
    return l2norm

def l2norm(x):
    norm2 = tf.norm(x, ord=2, axis=1)
    return norm2

def l1norm(x):
    norm1 = tf.norm(x, ord=1, axis=1)
    return norm1

class NetArch(object):

    def arch_vgg16(self, X, num_classes, dropout_keep_prob=0.8, is_train=False, embedding_size=128):
        arg_scope = vgg_arg_scope()
        with slim.arg_scope(arg_scope):
            net_vis, end_points, _ = vgg_16_conv(X, is_training=is_train)
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope('Logits_out'):
                net_vis = slim.avg_pool2d(net_vis, net_vis.get_shape()[1:3], padding='VALID',
                                      scope='AvgPool_1a_out')
                # 1 x 1 x 512
                net_vis = slim.dropout(net_vis, dropout_keep_prob, scope='Dropout_1b_out')
                net_vis = slim.flatten(net_vis, scope='PreLogitsFlatten_out')
                net_vis = slim.fully_connected(net_vis, embedding_size, activation_fn=None, scope='Logits_out0')
                net = slim.fully_connected(net_vis, num_classes, activation_fn=None,scope='Logits_out1')
        return net, net_vis
    
    def arch_vgg16_multi_conv(self, X, num_classes,dropout_keep_prob=0.8, is_train=False,embedding_size=64):
        arg_scope = vgg_arg_scope()
        with slim.arg_scope(arg_scope):
            _, end_points, net_c = vgg_16_conv(X, is_training=is_train)
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope('Logits_out'):
                #net_1 = slim.max_pool2d(net_c[-5], [32,32], stride=32, padding='VALID', scope='net_c_1')
                #net_1 = slim.conv2d(net_1, net_1.get_shape()[3], [1, 1], scope='net_1')
                #net_2 = slim.max_pool2d(net_c[-4], [16,16], stride=16, padding='VALID', scope='net_c_1')
                #net_2 = slim.conv2d(net_2, net_2.get_shape()[3], [1, 1], scope='net_2')
                #net_3 = slim.max_pool2d(net_c[-3], [8,8], stride=8, padding='VALID', scope='net_c_1')
                #net_3 = slim.conv2d(net_3, net_3.get_shape()[3], [1, 1], scope='net_3')
                net_4 = slim.max_pool2d(net_c[-2], [4,4], stride=4, padding='VALID', scope='net_c_1')
                net_4 = slim.conv2d(net_4, net_4.get_shape()[3], [1, 1], scope='net_4')
                net_5 = slim.max_pool2d(net_c[-1], [2,2], stride=2, padding='VALID', scope='net_c_1')
                net_5 = slim.conv2d(net_5, net_5.get_shape()[3], [1, 1], scope='net_5')
                # net_vis = tf.concat([net_1, net_2, net_3, net_4, net_5],3)
                net_vis = tf.concat([net_4, net_5],3)
                net_vis = slim.avg_pool2d(net_vis, net_vis.get_shape()[1:3], padding='VALID', scope='AvgPool_1a_out')
                # 1 x 1 x 512
                net_vis = slim.dropout(net_vis, dropout_keep_prob, scope='Dropout_1b_out')
                net_vis = slim.flatten(net_vis, scope='PreLogitsFlatten_out')
                net_vis = slim.fully_connected(net_vis, embedding_size, activation_fn=None, scope='Logits_out0')
                net = slim.fully_connected(net_vis, num_classes, activation_fn=None,scope='Logits_out1')
        return net, net_vis

    def arch_inception_v4(self, X, num_classes, dropout_keep_prob=0.8, is_train=False, embedding_size=128):
        arg_scope = inception_v4_arg_scope()
        with slim.arg_scope(arg_scope):
            net_vis, end_points = inception_v4(X, is_training=is_train)
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope('Logits_out'):
                # 8 x 8 x 1536
                net_vis = slim.avg_pool2d(net_vis, net_vis.get_shape()[1:3], padding='VALID',
                                      scope='AvgPool_1a_out')
                # 1 x 1 x 1536
                net_vis = slim.dropout(net_vis, dropout_keep_prob, scope='Dropout_1b_out')
                net_vis = slim.flatten(net_vis, scope='PreLogitsFlatten_out')
                # 1536
                net_vis = slim.fully_connected(net_vis, embedding_size, activation_fn=None, scope='Logits_out0')
                net = slim.fully_connected(net_vis, num_classes, activation_fn=None,scope='Logits_out1')
        return net, net_vis

    def arch_resnet_v2_50(self, X, num_classes, dropout_keep_prob=0.8, is_train=False, embedding_size=128):
        arg_scope = resnet_arg_scope()
        with slim.arg_scope(arg_scope):
            net_vis, end_points = resnet_v2_50(X, is_training=is_train)
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope('Logits_out'):
                # 8 x 8 x 1536
                net_vis = slim.avg_pool2d(net_vis, net_vis.get_shape()[1:3], padding='VALID',
                                      scope='AvgPool_1a_out')
                # 1 x 1 x 1536
                net_vis = slim.dropout(net_vis, dropout_keep_prob, scope='Dropout_1b_out')
                net_vis = slim.flatten(net_vis, scope='PreLogitsFlatten_out')
                # 1536
                net_vis = slim.fully_connected(net_vis, embedding_size, activation_fn=None, scope='Logits_out0')
                net = slim.fully_connected(net_vis, num_classes, activation_fn=None,scope='Logits_out1')
        return net, net_vis

class ConditionalSimNet(object):
    def __init__(self, n_conditions, embedding_size, learnedmask=True, prein=False):
        self.learnedmask = learnedmask
        self.prein = prein
        # create the mask
        # create the mask
        if learnedmask:
            if prein:
                # initialize masks
                mask_array = np.zeros([n_conditions, embedding_size])
                mask_array.fill(0.1)
                mask_len = int(embedding_size / n_conditions)
                for i in range(n_conditions):
                    mask_array[i, i*mask_len:(i+1)*mask_len] = 1
                mask_array = tf.cast(mask_array, tf.float32)
                self.W = tf.Variable(mask_array, name='W')
            else:
                self.W = tf.Variable(tf.random_uniform([n_conditions, embedding_size], 0.9, 0.7), name='W')
        else:
            # initialize masks
            mask_array = np.zeros([n_conditions, embedding_size])
            mask_len = int(embedding_size / n_conditions)
            for i in range(n_conditions):
                mask_array[i, i*mask_len:(i+1)*mask_len] = 1
            mask_array = tf.cast(mask_array, tf.float32)
            self.W = tf.Variable(mask_array, name='W')

    def forward(self, embeddings, c):
        if self.learnedmask:
            if self.prein:
                self.mask = tf.nn.embedding_lookup(self.W, c)
            else:
                self.mask = tf.nn.embedding_lookup(self.W, c)
        else:
            # no gradients for the masks
            self.mask = tf.stop_gradient(tf.nn.embedding_lookup(self.W, c))

        self.mask = tf.nn.relu(self.mask)
        masked_embedding = embeddings * self.mask[0]
        return masked_embedding, l1norm(self.mask[0])[0], l2norm(embeddings)[0], l2norm(masked_embedding)[0]



class VggCSN():
    def __init__(self, heght=224, width=224, n_class=10, n_conditions=4, embedding_size=128,learnedmask=True, prein=True, checkpoint_path='pretrain/vgg/vgg_16.ckpt'):
        # 定义输入
        self.X = tf.placeholder(tf.float32, [None, 224, 224, 3])
        # 选择检索的风格
        self.C = tf.placeholder(tf.int32, [None, None])
        # 总的类别数，这个没有用到
        self.Y = tf.placeholder(tf.float32, [None, n_class])
        # 定义是否训练
        self.is_train = tf.placeholder(tf.bool)
        # 定义dropout
        self.keep_prob_fc = tf.placeholder(tf.float32)
        # 
        self.variables_to_restore = []
        self.variables_to_train = []
        self.checkpoint_path = checkpoint_path

        _, net_vis = NetArch().arch_vgg16(self.X, n_class, self.keep_prob_fc, self.is_train)
        with tf.variable_scope('Logits_csn'):
            csn = ConditionalSimNet(n_conditions, embedding_size, learnedmask=True, prein=True)
            self.emds, self.mask, self.emds02, self.emds2 = csn.forward(net_vis, self.C)
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver_csn = tf.train.Saver(tf.global_variables())
        self._load_weight()
    
    def _load_weight(self):
        self._g_parameter()
        self.saver_net = tf.train.Saver(self.variables_to_restore)
        self.saver_net.restore(self.sess, self.checkpoint_path)

    def _g_parameter(self, checkpoint_exclude_scopes="Logits_out, Logits_csn"):
        exclusions = []
        if checkpoint_exclude_scopes:
            exclusions = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
        for var in tf.global_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    self.variables_to_train.append(var)
                    break
            if not excluded:
                self.variables_to_restore.append(var)

    def build_vgg_csn(self):
        return self.X,self.C,self.is_train,self.keep_prob_fc,self.emds, self.mask, self.emds02, self.emds2

    def test_vgg_csn(self, img, c):
        print (self.variables_to_restore)
        emds1, mask1, emds021, emds21 = self.sess.run([self.emds, self.mask, self.emds02, self.emds2], feed_dict={self.X: [img], self.C:[[c]],self.is_train:False, self.keep_prob_fc:1.0})
        return emds1, mask1, emds021, emds21



# 下面是调试模块， 
if __name__ == '__main__':
    import cv2
    import numpy as np
    img = np.random.uniform(0, 224*224*3, size=[224,224,3]).reshape([224,224,3])
    vgg_csn = VggCSN()
    emds, mask, emds02, emds2 = vgg_csn.test_vgg_csn(img, 2)
    print (emds)
    print (mask)


'''
# 下面是调试模块， 
if __name__ == '__main__':
    import cv2
    import numpy as np
    X = tf.placeholder(tf.float32, [None, 224, 224, 3])
    C = tf.placeholder(tf.int32, [None, None])
    Y = tf.placeholder(tf.float32, [None, 10])
    is_train = tf.placeholder(tf.bool)
    keep_prob_fc = tf.placeholder(tf.float32)

    _, net_vis = NetArch().arch_vgg16(X, 10, keep_prob_fc, is_train)
    csn = ConditionalSimNet(4, 128, learnedmask=True, prein=True)
    emds, mask, emds02, emds2 = csn.forward(net_vis, C)
    img = np.random.uniform(0, 224*224*3, size=[224,224,3]).reshape([224,224,3])
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    # 下面给值的时候要特别注意， 尤其是C:[[2]]。
    emds, mask, emds02, emds2 = sess.run([emds, mask, emds02, emds2], feed_dict={X: [img], C:[[2]],is_train:False, keep_prob_fc:1.0})
    print ( emds, mask, emds02, emds2 )
'''
