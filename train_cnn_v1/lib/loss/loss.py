# coding=utf-8

import tensorflow as tf 
import numpy as np 


# quared loss
def squared_loss(label, logit):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(label - logit), 1))
    return loss

# sigmoid loss
def sigmoid_loss(label, logit):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = label, logits = logit))
    return loss

# softmax loss
def softmax_loss(label, logit):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label, logits = logit))
    return loss

# triplet loss 
def triplet_loss(anchor, positive, negative, alpha):
    # 理论可以看这里： https://blog.csdn.net/tangwei2014/article/details/46788025
    # facenet理解： http://www.mamicode.com/info-detail-2096766.html
    """Calculate the triplet loss according to the FaceNet paper
    
    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
  
    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
        
        basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
      
    return loss


# center loss
def center_loss(features, label, alfa, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    """获取center loss及center的更新op 
    还可参考博客： https://blog.csdn.net/u014365862/article/details/79184966     
    Arguments: 
        features: Tensor,表征样本特征,一般使用某个fc层的输出,shape应该为[batch_size, feature_length]. 
        labels: Tensor,表征样本label,非one-hot编码,shape应为[batch_size]. 
        alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文. 
        num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少. 
     
    Return： 
        loss: Tensor,可与softmax loss相加作为总的loss进行优化. 
        centers: Tensor,存储样本中心值的Tensor，仅查看样本中心存储的具体数值时有用. 
        centers_update_op: op,用于更新样本中心的op，在训练时需要同时运行该op，否则样本中心不会更新 
    """  
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers


# 加入L2正则化的loss
def add_l2(loss, weight_decay):
    l2_losses = [weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)
    return reduced_loss

