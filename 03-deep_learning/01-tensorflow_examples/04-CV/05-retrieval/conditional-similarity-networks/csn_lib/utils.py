# -*- coding: utf-8 -*-
"""
    Created on 2018 06.18
    @author: liupeng
    wechat: lp9628
    blog: http://blog.csdn.net/u014365862/article/details/78422372
    """
import tensorflow as tf
slim = tf.contrib.slim
import numpy as np
import cv2
import os
import random

def input_placeholder3(height, width, num_classes):
    X1 = tf.placeholder(tf.float32, [None, height, width, 3])
    X2 = tf.placeholder(tf.float32, [None, height, width, 3])
    X3 = tf.placeholder(tf.float32, [None, height, width, 3])
    Y = tf.placeholder(tf.int64, [None, num_classes])
    is_train = tf.placeholder(tf.bool)
    keep_prob_fc = tf.placeholder(tf.float32)
    return X1, X2, X3, Y, is_train, keep_prob_fc

def hard_sample_loss(batch_size, ce_loss):
    num_examples = batch_size
    n_selected = num_examples/2   # 相当于选择一般进行权重的更新
    # find the most wrongly classified examples:
    n_selected = tf.cast(n_selected, tf.int32)
    vals, _ = tf.nn.top_k(ce_loss, k=n_selected)
    # 选择的topk的loss中的最小值，为了下面获得loss的mask，值为[1,1,0,0,1,1,1,1]
    th = vals[-1]
    selected_mask = ce_loss >= th # 得到类似的mask = [1,1,0,0,1,1,1,1]
    loss_weight = tf.cast(selected_mask, tf.float32)
    loss =  tf.reduce_sum(ce_loss*loss_weight) / tf.reduce_sum(loss_weight)
    return loss

def triplet_loss(anchor, positive, negative, batch_size, alpha=1., hard_sample=True):
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
        
        basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
        if hard_sample:
            loss = hard_sample_loss(batch_size, tf.maximum(basic_loss, 0.0))
        else:
            loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
    return loss

def g_parameter(checkpoint_exclude_scopes):
    exclusions = []
    if checkpoint_exclude_scopes:
        exclusions = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
    variables_to_restore = []
    variables_to_train = []
    for var in tf.global_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                variables_to_train.append(var)
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore,variables_to_train

def data_norm(img):
    # img = img / 255.0
    # img = img - 0.5
    # img = img * 2
    r, g, b = cv2.split(img)
    r = r - 127.5
    g = g - 127.5
    b = b - 127.5
    img = cv2.merge([r,g,b])
    return img

def add_img_padding(img):
    h, w, _ = img.shape
    width = np.max([h, w])
    # 按照长宽中大的初始化一个正方形
    img_padding = np.zeros([width, width,3])
    # 找出padding的中间位置
    h1 = int(width/2-h/2)
    h2 = int(width/2+h/2)
    w1 = int(width/2-w/2)
    w2 = int(width/2+w/2)
    # 进行padding， img为什么也要进行扣取？ 原因在于除以2取整会>造成尺寸不一致的情况。
    img_padding[h1:h2, w1:w2, :] = img[0:(h2-h1),0:(w2-w1),:]
    return img_padding

def get_next_batch_from_path(image_path, pointer, height, width, batch_size=64, training=True):
    batch_x0 = np.zeros([batch_size, height, width,3])
    batch_x1 = np.zeros([batch_size, height, width,3])
    batch_x2 = np.zeros([batch_size, height, width,3])
    C = np.zeros([batch_size,1])
    for i in range(batch_size):
        img_pairs = image_path[i+pointer*batch_size]
        image0 = cv2.imread(img_pairs[0])
        image1 = cv2.imread(img_pairs[1])
        image2 = cv2.imread(img_pairs[2])
        #image0 = add_img_padding(image0)
        #image1 = add_img_padding(image1)
        #image2 = add_img_padding(image2)
        image0 = cv2.resize(image0, (height, width))
        image1 = cv2.resize(image1, (height, width))
        image2 = cv2.resize(image2, (height, width))
        #if training: 
        #    # image = data_aug([image])[0]
        #    image0 = data_aug(image0)
        #    image1 = data_aug(image1)
        #    image2 = data_aug(image2)
        image0 = data_norm(image0)
        image1 = data_norm(image1)
        image2 = data_norm(image2)
        batch_x0[i,:,:,:] = image0
        batch_x1[i,:,:,:] = image1
        batch_x2[i,:,:,:] = image2
        C[i,:] = img_pairs[3]
    return batch_x0,batch_x1,batch_x2,C

def shuffle_data(train_imgs):
    index = [i for i in range(len(train_imgs))]
    np.random.shuffle(index)
    train_imgs = np.asarray(train_imgs)
    train_imgs = train_imgs[index]
    return train_imgs

def create_pairs(digit_indices, num_classes, condition_classes):
    pairs = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1  # 最小类别数
    for d in range(num_classes):
        for i in range(n):
            np.random.shuffle(digit_indices[d])
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z3 = digit_indices[dn][i]
            pairs += [[z1, z3, z2, condition_classes]]
    pairs = shuffle_data(pairs)
    return np.asarray(pairs)

