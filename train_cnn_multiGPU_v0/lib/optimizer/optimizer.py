# coding=utf-8

import tensorflow as tf 
import numpy as np 

def adam_optimizer(learning_rate, adam_beta1=0.9, adam_beta2=0.999, opt_epsilon=1e-08):
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=adam_beta1, beta2=adam_beta2, epsilon=opt_epsilon)
    return optimizer

def adadelta_optimizer(learning_rate, adadelta_rho=0.95, opt_epsilon=1e-08):
    optimizer = tf.train.AdadeltaOptimizer(learning_rate, rho=adadelta_rho, epsilon=opt_epsilon)
    return optimizer

def adagrad_optimizer(learning_rate, adagrad_initial_accumulator_value=0.1):
    optimizer = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=adagrad_initial_accumulator_value)
    return optimizer

def ftrl_optimizer(learning_rate, ftrl_learning_rate_power=-0.5, ftrl_initial_accumulator_value=0.1, ftrl_l1=0.0, ftrl_l2=0.0):
    optimizer = tf.train.FtrlOptimizer(learning_rate, learning_rate_power=ftrl_learning_rate_power, initial_accumulator_value=ftrl_initial_accumulator_value, l1_regularization_strength=ftrl_l1, l2_regularization_strength=ftrl_l2)
    return optimizer

def momentum_optimizer(learning_rate, momentum):
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=momentum, name='Momentum')
    return optimizer

def rmsprop_optimizer(learning_rate, rmsprop_decay=0.9, rmsprop_momentum=0.0, opt_epsilon=1e-10):
    optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=rmsprop_decay, momentum=rmsprop_momentum, epsilon=opt_epsilon)
    return optimizer

def sgd_optimizer(learning_rate):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    return optimizer



def _configure_optimizer(learning_rate):
  """Configures the optimizer used for training.
  Args:
    learning_rate: A scalar or `Tensor` learning rate.
  Returns:
    An instance of an optimizer.
  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """
  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=FLAGS.adadelta_rho,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=FLAGS.ftrl_learning_rate_power,
        initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
        l1_regularization_strength=FLAGS.ftrl_l1,
        l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        name='Momentum')
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.rmsprop_momentum,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
  return optimizer



