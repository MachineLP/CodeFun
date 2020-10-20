# coding=utf-8
"""
Created on 2017 10.17
@author: liupeng
wechat: lp9628
blog: http://blog.csdn.net/u014365862/article/details/78422372
"""

import tensorflow as tf
slim = tf.contrib.slim
import numpy as np
import sklearn.metrics
import cv2
import os
# from data_aug.data_aug import DataAugmenters
from lib.model.build_model.build_net import net_arch
import tensorflow as tf
from lib.data_aug.data_aug import DataAugmenters
from lib.loss.loss import softmax_loss, sigmoid_loss, squared_loss, add_l2
from lib.optimizer.optimizer import adam_optimizer,sgd_optimizer, rmsprop_optimizer
from lib.optimizer.optimizer_minimize import optimizer_minimize, optimizer_apply_gradients

def g_parameter(checkpoint_exclude_scopes):
    exclusions = []
    if checkpoint_exclude_scopes:
        exclusions = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
    variables_to_restore = []
    variables_to_train = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                variables_to_train.append(var)
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore,variables_to_train

def shuffle_train_data(train_imgs, train_labels):
    index = [i for i in range(len(train_imgs))]
    np.random.shuffle(index)
    train_imgs = np.asarray(train_imgs)
    train_labels = np.asarray(train_labels)
    train_imgs = train_imgs[index]
    train_labels = train_labels[index]
    return train_imgs, train_labels

def data_norm(img):
    img = img / 255.0
    img = img - 0.5
    img = img * 2
    return img
def data_aug(img):
    return DataAugmenters(img).run()

#------------------------------------------------#
# 功能：按照图像最小的边进行缩放
# 输入：img：图像，resize_size：需要的缩放大小
# 输出：缩放后的图像
#------------------------------------------------#
def img_crop_pre(img, resize_size=336):
    h, w, _ = img.shape
    deta = h if h < w else w
    alpha = resize_size / float(deta)
    # print (alpha)
    img = cv2.resize(img, (int(h*alpha), int(w*alpha)))
    return img

def get_next_batch_from_path(image_path, image_labels, pointer, height, width, batch_size=64, training=True):
    batch_x = np.zeros([batch_size, height, width,3])
    num_classes = len(image_labels[0])
    batch_y = np.zeros([batch_size, num_classes]) 
    for i in range(batch_size):  
        image = cv2.imread(image_path[i+pointer*batch_size])
        image = img_crop_pre(image, resize_size=336)
        # image = cv2.resize(image, (height, width)) 
        if training: 
            # image = data_aug([image])[0]
            image = data_aug(image)
        image = cv2.resize(image, (height, width)) 
        image = data_norm(image)
        batch_x[i,:,:,:] = image
        batch_y[i] = image_labels[i+pointer*batch_size]
    return batch_x, batch_y

def img_crop(img, box):
    # y1, x1, y2, x2 = box[1]-20, box[0]-20, box[1]+box[3]+40, box[0]+box[2]+40
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    img = img[y1:y2, x1:x2]
    return img


#######
def input_placeholder(height, width, num_classes):
    X = tf.placeholder(tf.float32, [None, height, width, 3])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    is_train = tf.placeholder(tf.bool, name='is_train')
    keep_prob_fc = tf.placeholder(tf.float32)
    return X, Y, is_train, keep_prob_fc


def build_net(X, num_classes, keep_prob_fc, is_train, arch_model):
    arch = net_arch()
    if arch_model == "arch_inception_v4":
        net, net_vis = arch.arch_inception_v4(X, num_classes, keep_prob_fc, is_train)
    elif arch_model == "arch_inception_v4_rnn":
        net, net_vis = arch.arch_inception_v4_rnn(X, num_classes, keep_prob_fc, is_train)
    elif arch_model == "arch_inception_v4_rnn_attention":
        net, net_vis = arch.arch_inception_v4_rnn_attention(X, num_classes, keep_prob_fc, is_train)
    elif arch_model == "arch_alexnet_v2":
        net, net_vis = arch.arch_alexnet_v2(X, num_classes, keep_prob_fc, is_train)
    elif arch_model == "arch_lp_net":
        net, net_vis = arch.arch_lp_net(X, num_classes, keep_prob_fc, is_train)
    elif arch_model == "arch_vgg16":
        net, net_vis = arch.arch_vgg16(X, num_classes, keep_prob_fc, is_train)
    else:
        print ('{} is error!', arch_model)
    return net, net_vis

def build_net_multi_label(X, num_classes, keep_prob_fc, is_train, arch_model):
    arch = net_arch()
    if arch_model == "arch_inception_v4_multi_label":
        net, net_vis = arch.arch_inception_v4(X, num_classes, keep_prob_fc, is_train)
    elif arch_model == "arch_inception_v4_rnn_multi_label":
        net, net_vis = arch.arch_inception_v4_rnn(X, num_classes, keep_prob_fc, is_train)
    elif arch_model == "arch_inception_v4_rnn_attention_multi_label":
        net, net_vis = arch.arch_inception_v4_rnn_attention(X, num_classes, keep_prob_fc, is_train)
    elif arch_model == "arch_alexnet_v2_multi_label":
        net, net_vis = arch.arch_alexnet_v2(X, num_classes, keep_prob_fc, is_train)
    elif arch_model == "arch_lp_net_multi_label":
        net, net_vis = arch.arch_lp_net(X, num_classes, keep_prob_fc, is_train)
    elif arch_model == "arch_vgg16_multi_label":
        net, net_vis = arch.arch_vgg16(X, num_classes, keep_prob_fc, is_train)
    else:
        print ('{} is error!', arch_model)
    return net, net_vis



def cost(label, logit):
    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = label, logits = logit))
    loss = sigmoid_loss(label = label, logit = logit)
    return loss


def train_op(learning_rate, loss, variables_to_train, global_step):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        if variables_to_train == []:
            opt_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
            # optimizer = adam_optimizer(learning_rate)
            # opt_op = optimizer_minimize(optimizer, loss, global_step)
        else:
            #opt_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, var_list = variables_to_train, global_step=global_step)
            optimizer = adam_optimizer(learning_rate)
            opt_op = optimizer_minimize(optimizer, loss, global_step, var_list = variables_to_train)
    return opt_op

def model_accuracy(net, Y, num_classes):
    predict = tf.reshape(net, [-1, num_classes])
    max_idx_p = tf.argmax(predict, 1)
    max_idx_l = tf.argmax(Y, 1)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

def model_accuracy_seg(net, Y, num_classes):
    predict = tf.reshape(net, [-1, num_classes])
    max_idx_p = tf.argmax(predict, 1)
    max_idx_l = tf.argmax(Y, 1)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

def compute_map(net, Y):

    nclasses = Y.shape[1]
    all_ap = []
    for cid in range(nclasses):
        gt_cls = Y[:, cid].astype('float32')
        pred_cls = net[:, cid].astype('float32')
        # 某个人标签没有属性值，continue；
        if np.sum(gt_cls) == 0:
            continue
        # As per PhilK. code:
        # https://github.com/philkr/voc-classification/blob/master/src/train_cls.py
        pred_cls -= 1e-5 * gt_cls
        ap = sklearn.metrics.average_precision_score(
                                                     gt_cls, pred_cls, average=None)
        all_ap.append(ap)
    mAP = np.mean(all_ap).astype('float32')
    return mAP

def model_mAP(net, Y):
    mAP = tf.py_func(compute_map,[net, Y], tf.float32)
    return mAP

# 同样也可用于多标签
def to_one_hot(labels, num_classes):
    labels_onehot = []
    for i in range(len(labels)):
        label = np.zeros([num_classes], np.float32)  # 标签容器，注意大小---------------------------change------------------------------------
        for j in list([labels[i]]):
            try:
                label[j] = 1.  # 若图像标签为j，label[j] = 1
            except():
                continue
        labels_onehot += [label]
    return labels_onehot



