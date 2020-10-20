# coding=utf-8
"""
Created on 2017 10.17
@author: liupeng
wechat: lp9628
blog: http://blog.csdn.net/u014365862/article/details/78422372
"""

import tensorflow as tf 
import numpy as np 


def optimizer_minimize(optimizer, loss, global_step=tf.train.get_global_step(), var_list=tf.all_variables()):
    opt_op = optimizer.minimize(loss, var_list=var_list)
    return opt_op


def optimizer_apply_gradients(optimizer, loss, global_step=tf.train.get_global_step(), var_list=tf.all_variables()):
    gradients = tf.gradients(loss, var_list)
    opt_op = optimizer.apply_gradients(zip(gradients, var_list), global_step=global_step)
    return opt_op
