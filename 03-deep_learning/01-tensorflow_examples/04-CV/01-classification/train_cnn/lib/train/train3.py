# coding=utf-8
"""
Created on 2017 10.17
@author: liupeng
"""

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
from lib.utils.utils import get_next_batch_from_path3, shuffle_train_data
from lib.utils.utils import input_placeholder3, build_net3, cost, train_op, model_accuracy
import os

def train3(train_data,train_label,valid_data,valid_label,train_dir,num_classes,batch_size,arch_model,learning_r_decay,learning_rate_base,decay_rate,dropout_prob,epoch,height,width,checkpoint_exclude_scopes,early_stop,EARLY_STOP_PATIENCE,fine_tune,train_all_layers,checkpoint_path,train_n,valid_n,g_parameter):
    # ---------------------------------------------------------------------------------#
    X1,X2,X3, Y, is_train, keep_prob_fc = input_placeholder3(height, width, num_classes)
    net, _ = build_net3(X1,X2,X3, num_classes, keep_prob_fc, is_train,arch_model)
    variables_to_restore,variables_to_train = g_parameter(checkpoint_exclude_scopes)
    loss = cost(Y, net)
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
    optimizer = train_op(learning_rate, loss, variables_to_train, global_step)
    accuracy = model_accuracy(net, Y, num_classes)
    #------------------------------------------------------------------------------------#
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    saver2 = tf.train.Saver(tf.global_variables())
    if not train_all_layers:
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
            # images, labels = get_next_batch_from_path(train_data, train_label, batch_i, height, width, batch_size=batch_size, training=True)
            images1, images2, images3, labels = get_next_batch_from_path3(train_data, train_label, batch_i, height, width, batch_size=batch_size, training=True)
            los, _ = sess.run([loss,optimizer], feed_dict={X1: images1, X2: images2, X3: images3,Y: labels, is_train:True, keep_prob_fc:dropout_prob})
            print (los)
            if batch_i%20==0:
                loss_, acc_ = sess.run([loss, accuracy], feed_dict={X1: images1, X2: images2, X3: images3, Y: labels, is_train:False, keep_prob_fc:1.0})
                print('Batch: {:>2}: Training loss: {:>3.5f}, Training accuracy: {:>3.5f}'.format(batch_i, loss_, acc_))

            if batch_i%100==0:
                # images, labels = get_next_batch_from_path(valid_data, valid_label, batch_i%(int(valid_n/batch_size)), height, width, batch_size=batch_size, training=False)
                images1, images2, images3, labels = get_next_batch_from_path3(valid_data, valid_label, batch_i%(int(valid_n/batch_size)), height, width, batch_size=batch_size, training=False)
                ls, acc = sess.run([loss, accuracy], feed_dict={X1: images1, X2: images2, X3: images3, Y: labels, is_train:False, keep_prob_fc:1.0})
                print('Batch: {:>2}: Validation loss: {:>3.5f}, Validation accuracy: {:>3.5f}'.format(batch_i, ls, acc))
            
        
        print('Epoch===================================>: {:>2}'.format(epoch_i))
        valid_ls = 0
        valid_acc = 0
        for batch_i in range(int(valid_n/batch_size)):
            images_valid1, images_valid2, images_valid3, labels_valid = get_next_batch_from_path3(valid_data, valid_label, batch_i, height, width, batch_size=batch_size, training=False)
            epoch_ls, epoch_acc = sess.run([loss, accuracy], feed_dict={X1: images_valid1,X2: images_valid2,X3: images_valid3, Y: labels_valid, keep_prob_fc:1.0, is_train:False})
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
