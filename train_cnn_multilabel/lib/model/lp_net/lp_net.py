# -*- coding: utf-8 -*-
"""
Created on 2017 10.17
@author: liupeng
wechat: lp9628
blog: http://blog.csdn.net/u014365862/article/details/78422372
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

def lp_net_arg_scope(weight_decay=0.00004,
                        use_batch_norm=True,
                        batch_norm_decay=0.9997,
                        batch_norm_epsilon=0.001):
  batch_norm_params = {'decay': batch_norm_decay,'epsilon': batch_norm_epsilon,'updates_collections': tf.GraphKeys.UPDATE_OPS,}
  if use_batch_norm:
    normalizer_fn = slim.batch_norm
    normalizer_params = batch_norm_params
  else:
    normalizer_fn = None
    normalizer_params = {}
  with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(weight_decay)):
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=slim.variance_scaling_initializer(),
        activation_fn=tf.nn.relu,
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params) as sc:
      return sc

def lp_net(inputs,
               num_classes=None,
               is_training=True,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               scope='lp_net'):
  with tf.variable_scope(scope, 'lp_net', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=[end_points_collection]):
      net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID',scope='conv1')
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
      net = slim.conv2d(net, 192, [5, 5], scope='conv2')
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
      net = slim.conv2d(net, 384, [3, 3], scope='conv3')
      net = slim.conv2d(net, 384, [3, 3], scope='conv4')
      net = slim.conv2d(net, 256, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if num_classes is not None:
          with slim.arg_scope([slim.conv2d],weights_initializer=trunc_normal(0.005),biases_initializer=tf.constant_initializer(0.1)):
            # net = slim.conv2d(net, 4096, [5, 5], padding='VALID',scope='fc6') 
            net = slim.conv2d(net, 4096, net.get_shape()[1:3], padding='VALID',scope='fc6')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,scope='dropout6')
            net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,scope='dropout7')
            net = slim.conv2d(net, num_classes, [1, 1],activation_fn=None,normalizer_fn=None,biases_initializer=tf.zeros_initializer(),scope='fc8')
            if spatial_squeeze:
              net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
              end_points[sc.name + '/fc8'] = net
      else:
          net = net
          end_points = end_points
      return net, end_points

