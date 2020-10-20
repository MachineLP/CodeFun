# coding = utf-8

"""
Created on 2017 10.17
@author: liupeng
wechat: lp9628
blog: http://blog.csdn.net/u014365862/article/details/78422372
"""

import tensorflow as tf
from tensorflow.python.framework import graph_util
from lib.data_load.data_load_from_txt_mullabel import data_load_from_txt_mullabel
from lib.utils.multi_label_utils import get_next_batch_from_path, shuffle_train_data
from lib.utils.multi_label_utils import input_placeholder, build_net_multi_label, cost, train_op, model_mAP
from keras.utils import np_utils
import cv2
import numpy as np
import os
import sys
import config

num_classes = config.num_classes
height, width = config.height, config.width
arch_model = config.arch_model
batch_size = config.batch_size
sample_dir = 'val.txt'
train_rate = 1.0
test_data, test_label, valid_data, valid_label, test_n, valid_n, note_label = data_load_from_txt_mullabel(sample_dir, train_rate).gen_train_valid()
print ('test_n', test_n)
print ('valid_n', valid_n)
# test_label = np_utils.to_categorical(test_label, num_classes)
# valid_label = np_utils.to_categorical(valid_label, num_classes)

X, Y, is_train, keep_prob_fc = input_placeholder(height, width, num_classes)
net, _ = build_net_multi_label(X, num_classes, keep_prob_fc, is_train,arch_model)
loss = cost(Y, net)
pre = tf.nn.sigmoid(net)
accuracy = model_mAP(pre, Y)
# predict = tf.reshape(net, [-1, num_classes], name='predictions')



if __name__ == '__main__':
    train_dir = 'model'
    latest = tf.train.latest_checkpoint(train_dir)
    if not latest:
        print ("No checkpoint to continue from in", train_dir)
        sys.exit(1)
    print ("resume", latest)
    sess = tf.Session()
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, latest)
    test_ls = 0
    test_acc = 0
    for batch_i in range(int(test_n/batch_size)):
            images_test, labels_test = get_next_batch_from_path(test_data, test_label, batch_i, height, width, batch_size=batch_size, training=False)
            epoch_ls, epoch_acc, pred = sess.run([loss, accuracy, pre], feed_dict={X: images_test, Y: labels_test, keep_prob_fc:1.0, is_train:False})
            print (epoch_acc)
            print (pred)
            test_ls = test_ls + epoch_ls
            test_acc = test_acc + epoch_acc
    print('Epoch: {:>2}: Validation loss: {:>3.5f}, Validation accuracy: {:>3.5f}'.format(0, test_ls/int(test_n/batch_size), test_acc/int(test_n/batch_size)))
