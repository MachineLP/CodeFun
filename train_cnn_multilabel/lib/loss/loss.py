# coding=utf-8
"""
Created on 2017 10.17
@author: liupeng
wechat: lp9628
blog: http://blog.csdn.net/u014365862/article/details/78422372
"""

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


def get_center_loss(features, labels, alpha, num_classes):  
    """获取center loss及center的更新op 
     
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
    # 获取特征的维数，例如256维  
    len_features = features.get_shape()[1]  
    # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，  
    # 设置trainable=False是因为样本中心不是由梯度进行更新的  
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,  
        initializer=tf.constant_initializer(0), trainable=False)  
    # 将label展开为一维的，输入如果已经是一维的，则该动作其实无必要  
    # labels = tf.reshape(labels, [-1])
    labels = tf.argmax(labels, 1)
      
    # 根据样本label,获取mini-batch中每一个样本对应的中心值  
    centers_batch = tf.gather(centers, labels)  
    # 计算loss  
    loss = tf.nn.l2_loss(features - centers_batch)  
      
    # 当前mini-batch的特征值与它们对应的中心值之间的差  
    diff = centers_batch - features  
      
    # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)  
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)  
    appear_times = tf.gather(unique_count, unique_idx)  
    appear_times = tf.reshape(appear_times, [-1, 1])  
      
    diff = diff / tf.cast((1 + appear_times), tf.float32)  
    diff = alpha * diff  
      
    centers_update_op = tf.scatter_sub(centers, labels, diff)  
      
    return loss, centers, centers_update_op 


# 加入L2正则化的loss
def add_l2(loss, weight_decay):
    l2_losses = [weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)
    return reduced_loss


'''
  # 根据label及其阈值，得到positive的mask。
  pos_mask = tf.reduce_sum(
               tf.cast(
                 tf.greater_equal(
                   labels, tf.fill(tf.shape(labels), FLAGS.mask_thres)), 
                   tf.float32), 
             0)
  # 
  pos_curr_count = tf.cast(tf.greater(   pos_mask, 0), tf.float32)
  # negative的mask。
  neg_curr_count = tf.cast(tf.less_equal(pos_mask, 0), tf.float32)

  pos_count = tf.Variable(tf.zeros(shape=[FLAGS.class_num,]),  trainable=False)
  neg_count = tf.Variable(tf.zeros(shape=[FLAGS.class_num,]),  trainable=False)

  # how many class within only negtive samples in a batch select to learn， 每个batch只选择一部分的negative去学习。
  neg_select = tf.cast(
                 tf.less_equal(
                    tf.random_uniform(
                      shape=[FLAGS.class_num,], 
                      minval=0, maxval=1,
                      seed = FLAGS.random_seed),
                    FLAGS.neg_select), 
                 tf.float32)
  tf.summary.histogram('pos_curr_count', pos_curr_count)
  tf.summary.histogram('neg_curr_count', neg_curr_count)
  tf.summary.histogram('neg_select', neg_select)

  with tf.control_dependencies([pos_curr_count, neg_curr_count, neg_select]):

    pos_count = tf.assign_sub(
                   tf.assign_add(pos_count, pos_curr_count),
                   tf.multiply(pos_count, neg_curr_count))
    neg_count = tf.assign_sub(
                   tf.assign_add(neg_count, tf.multiply(neg_curr_count, neg_select)),
                   tf.multiply(neg_count, pos_curr_count))
    tf.summary.histogram('pos_count', pos_count)
    tf.summary.histogram('neg_count', neg_count)
  
  pos_loss_coef = -1 * (tf.log((0.01 + pos_count)/10)/tf.log(10.0))
  pos_loss_coef = tf.where(
                    tf.greater(pos_loss_coef, tf.fill(tf.shape(pos_loss_coef), 0.01)),
                    pos_loss_coef,
                    tf.fill(tf.shape(pos_loss_coef), 0.01))
  pos_loss_coef = tf.multiply(pos_loss_coef, pos_curr_count)
  
  tf.summary.histogram('pos_loss_coef', pos_loss_coef)
  neg_loss_coef = -1 * (tf.log((8 + neg_count)/10)/tf.log(10.0))
  neg_loss_coef = tf.where(
                   tf.greater(neg_loss_coef, tf.fill(tf.shape(neg_loss_coef), 0.01)),
                   neg_loss_coef,
                   tf.fill(tf.shape(neg_loss_coef), 0.001))
  neg_loss_coef = tf.multiply(neg_loss_coef, tf.multiply(neg_curr_count, neg_select))
  tf.summary.histogram('neg_loss_coef', neg_loss_coef)
  loss_coef = tf.add(pos_loss_coef, neg_loss_coef)
  tf.summary.histogram('loss_coef', loss_coef)

  # b. get non-negative mask
  non_neg_mask = tf.fill(tf.shape(labels), -1.0, name='non_neg')
  non_neg_mask = tf.cast(tf.not_equal(labels, non_neg_mask), tf.float32)
  tf.summary.histogram('non_neg', non_neg_mask)

  # cal loss
  cross_entropy = tf.nn.weighted_cross_entropy_with_logits(
       logits=logits, targets=labels, pos_weight=12, name='sigmod_cross_entropy')
  tf.summary.histogram('sigmod_ce', cross_entropy)
  cross_entropy_cost = tf.reduce_sum(tf.reduce_mean(cross_entropy * non_neg_mask, axis=0) * loss_coef)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy_cost, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy_cost)

  # Add weight decay to the loss. We exclude the batch norm variables because
  # doing so leads to a small improvement in accuracy.
  loss = cross_entropy_cost + FLAGS.weight_decay * tf.add_n(
    [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'batch_normalization' not in v.name])
'''
