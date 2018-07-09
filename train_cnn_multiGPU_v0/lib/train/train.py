# coding=utf-8
"""
Created on 2017 10.17
@author: liupeng
"""

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
from lib.utils.utils import get_next_batch_from_path, shuffle_train_data
from lib.utils.utils import input_placeholder, build_net, cost, train_op, model_accuracy
import os

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def assign_to_device(device, ps_device='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return device

    return _assign

def train(train_data,train_label,valid_data,valid_label,train_dir,num_classes,batch_size,arch_model,learning_r_decay,learning_rate_base,decay_rate,dropout_prob,epoch,height,width,checkpoint_exclude_scopes,early_stop,EARLY_STOP_PATIENCE,fine_tune,train_all_layers,checkpoint_path,train_n,valid_n,g_parameter):
    # ---------------------------------------------------------------------------------#
    X, Y, is_train, keep_prob_fc = input_placeholder(height, width, num_classes)

    global_step = tf.Variable(0, trainable=False)  
    if learning_r_decay:
        learning_rate = tf.train.exponential_decay(  
            learning_rate_base,                     
            global_step * batch_size,  
            train_n,                
            decay_rate,                       
            staircase=True)  
    else:
        learning_rate = learning_rate_base
    if train_all_layers:
        variables_to_train = []
    
    # Create an optimizer that performs gradient descent.
    opt = tf.train.AdamOptimizer(learning_rate)
    
    with tf.device('/cpu:0'):
        tower_grads = []
        #for i in range(num_gpus=2):
        #with tf.device(assign_to_device('/gpu:{}'.format(i), ps_device='/cpu:0')):
        with tf.device('/gpu:0'):
            with tf.variable_scope('', reuse=tf.AUTO_REUSE) as scope:
                net, _ = build_net(X[0:int(batch_size/2)], num_classes, keep_prob_fc, is_train,arch_model)
                loss_0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y[0:int(batch_size/2)], logits = net))
                accuracy_0 = model_accuracy(net, Y[0:int(batch_size/2)], num_classes)
                grads_0 = opt.compute_gradients(loss_0)
                tower_grads.append(grads_0)
        with tf.device('gpu:1'):
            with tf.variable_scope('', reuse=tf.AUTO_REUSE) as scope:
                net, _ = build_net(X[int(batch_size/2):batch_size], num_classes, keep_prob_fc, is_train,arch_model)
                loss_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y[int(batch_size/2):batch_size], logits = net))
                accuracy_1 = model_accuracy(net, Y[int(batch_size/2):batch_size], num_classes)
                grads_1 = opt.compute_gradients(loss_1)
                tower_grads.append(grads_1)
        variables_to_restore,variables_to_train = g_parameter(checkpoint_exclude_scopes)
        grads = average_gradients(tower_grads)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = opt.apply_gradients(grads, global_step=global_step)
        loss = tf.reduce_mean([loss_0,loss_1])
        accuracy = tf.reduce_mean([accuracy_0, accuracy_1])

        #------------------------------------------------------------------------------------#
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        saver2 = tf.train.Saver(tf.global_variables())
        #if not train_all_layers:
        saver_net = tf.train.Saver(variables_to_restore)
        saver_net.restore(sess, checkpoint_path)
    
        if fine_tune:
            # saver2.restore(sess, fine_tune_dir)
            latest = tf.train.latest_checkpoint(train_dir)
            if not latest:
                print ("No checkpoint to continue from in", train_dir)
                sys.exit(1)
            print ("resume", latest)
            saver2.restore(sess, latest)
    
        # early stopping
        best_valid = np.inf
        best_valid_epoch = 0

        for epoch_i in range(epoch):
            for batch_i in range(int(train_n/batch_size)):
                images, labels = get_next_batch_from_path(train_data, train_label, batch_i, height, width, batch_size=batch_size, training=True)
                los, _ = sess.run([loss,optimizer], feed_dict={X: images, Y: labels, is_train:True, keep_prob_fc:dropout_prob})
                print (los)
                if batch_i%20==0:
                    loss_, acc_ = sess.run([loss, accuracy], feed_dict={X: images, Y: labels,is_train:False, keep_prob_fc:1.0})
                    print('Batch: {:>2}: Training loss: {:>3.5f}, Training accuracy: {:>3.5f}'.format(batch_i, loss_, acc_))
    
                if batch_i%100==0:
                    images, labels = get_next_batch_from_path(valid_data, valid_label, batch_i%(int(valid_n/batch_size)), height, width, batch_size=batch_size, training=False)
                    ls, acc = sess.run([loss, accuracy], feed_dict={X: images, Y: labels, is_train:False, keep_prob_fc:1.0})
                    print('Batch: {:>2}: Validation loss: {:>3.5f}, Validation accuracy: {:>3.5f}'.format(batch_i, ls, acc))
            
        
            print('Epoch===================================>: {:>2}'.format(epoch_i))
            valid_ls = 0
            valid_acc = 0
            for batch_i in range(int(valid_n/batch_size)):
                images_valid, labels_valid = get_next_batch_from_path(valid_data, valid_label, batch_i, height, width, batch_size=batch_size, training=False)
                epoch_ls, epoch_acc = sess.run([loss, accuracy], feed_dict={X: images_valid, Y: labels_valid, keep_prob_fc:1.0, is_train:False})
                valid_ls = valid_ls + epoch_ls
                valid_acc = valid_acc + epoch_acc
            print('Epoch: {:>2}: Validation loss: {:>3.5f}, Validation accuracy: {:>3.5f}'.format(epoch_i, valid_ls/int(valid_n/batch_size), valid_acc/int(valid_n/batch_size)))

            if valid_acc/int(valid_n/batch_size) > 0.90:
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver2.save(sess, checkpoint_path, global_step=epoch_i, write_meta_graph=False)
            # ---------------------------------------------------------------------------------#
            if early_stop:
                loss_valid = valid_ls/int(valid_n/batch_size)
                if loss_valid < best_valid:
                    best_valid = loss_valid
                    best_valid_epoch = epoch_i
                elif best_valid_epoch + EARLY_STOP_PATIENCE < epoch_i:
                    print("Early stopping.")
                    print("Best valid loss was {:.6f} at epoch {}.".format(best_valid, best_valid_epoch))
                    break
            train_data, train_label = shuffle_train_data(train_data, train_label)
        sess.close()
