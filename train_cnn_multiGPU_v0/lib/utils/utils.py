#coding=utf-8
import tensorflow as tf
slim = tf.contrib.slim
import numpy as np
import cv2
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

def get_next_batch_from_path(image_path, image_labels, pointer, height, width, batch_size=64, training=True):
    batch_x = np.zeros([batch_size, height, width,3])
    num_classes = len(image_labels[0])
    batch_y = np.zeros([batch_size, num_classes]) 
    for i in range(batch_size):  
        image = cv2.imread(image_path[i+pointer*batch_size])
        image = cv2.resize(image, (height, width)) 
        if training: 
            # image = data_aug([image])[0]
            image = data_aug(image)
        image = data_norm(image)
        batch_x[i,:,:,:] = image
        batch_y[i] = image_labels[i+pointer*batch_size]
    return batch_x, batch_y

def img_crop(img, box):
    # y1, x1, y2, x2 = box[1]-20, box[0]-20, box[1]+box[3]+40, box[0]+box[2]+40
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    img = img[y1:y2, x1:x2]
    return img
def get_next_batch_from_path3(image_path, image_labels, pointer, height, width, batch_size=64, training=True):
    batch_x1 = np.zeros([batch_size, height, width,3])
    batch_x2 = np.zeros([batch_size, height, width,3])
    batch_x3 = np.zeros([batch_size, height, width,3])
    num_classes = len(image_labels[0])
    batch_y = np.zeros([batch_size, num_classes]) 
    for i in range(batch_size):  
        image = cv2.imread(image_path[i+pointer*batch_size])
        # image = cv2.resize(image, (height, width)) 
        if training: 
            # image = data_aug([image])[0]
            image = data_aug(image)
        h, w, _= image.shape
        image = data_norm(image)
        # 取人体的三部分， 15%、 35%、 50%
        # 上半身和下半身各占 50%
        # image1 = img_crop(image, [0,0,w,int(h*0.15)])
        image1 = image
        image1 = cv2.resize(image1, (height, width)) 
        # image2= img_crop(image, [0,int(h*0.15),w,int(h*0.5)])
        image2= img_crop(image, [0,0,w,int(h*0.5)])
        image2 = cv2.resize(image2, (height, width)) 
        image3 = img_crop(image, [0,int(h*0.5),w,h])
        image3 = cv2.resize(image3, (height, width)) 
        batch_x1[i,:,:,:] = image1
        batch_x2[i,:,:,:] = image2
        batch_x3[i,:,:,:] = image3
        batch_y[i] = image_labels[i+pointer*batch_size]
    return batch_x1,batch_x2,batch_x3, batch_y

#######
def input_placeholder(height, width, num_classes):
    X = tf.placeholder(tf.float32, [None, height, width, 3])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    is_train = tf.placeholder(tf.bool)
    keep_prob_fc = tf.placeholder(tf.float32)
    return X, Y, is_train, keep_prob_fc

def input_placeholder3(height, width, num_classes):
    X1 = tf.placeholder(tf.float32, [None, height, width, 3])
    X2 = tf.placeholder(tf.float32, [None, height, width, 3])
    X3 = tf.placeholder(tf.float32, [None, height, width, 3])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    is_train = tf.placeholder(tf.bool, name='is_train')
    keep_prob_fc = tf.placeholder(tf.float32)
    return X1, X2, X3, Y, is_train, keep_prob_fc

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
    else:
        print ('{} is error!', arch_model)
    return net, net_vis

def build_net3(X1,X2,X3, num_classes, keep_prob_fc, is_train, arch_model):
    arch = net_arch()
    if arch_model == "arch_multi_alexnet_v2":
        net, net_vis = arch.arch_multi_alexnet_v2(X1,X2,X3, num_classes, keep_prob_fc, is_train)
    elif arch_model == "arch_multi_vgg16":
        net, net_vis = arch.arch_multi_vgg16(X1,X2,X3, num_classes, keep_prob_fc, is_train)
    elif arch_model == "arch_multi_vgg16_conv":
        net, net_vis = arch.arch_multi_vgg16_conv(X1,X2,X3, num_classes, keep_prob_fc, is_train)
    else:
        print ('{} is error!', arch_model)
    return net, net_vis

def cost(label, logit):
    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = label, logits = logit))
    loss = softmax_loss(label = label, logit = logit)
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





