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

def train(train_data,train_label,valid_data,valid_label,train_dir,num_classes,batch_size,arch_model,learning_r_decay,learning_rate_base,decay_rate,dropout_prob,epoch,height,width,checkpoint_exclude_scopes,early_stop,EARLY_STOP_PATIENCE,fine_tune,train_all_layers,checkpoint_path,g_parameter):
    # ---------------------------------------------------------------------------------#
    # X, Y, is_train, keep_prob_fc = input_placeholder(height, width, num_classes)
    # net, _ = build_net(X, num_classes, keep_prob_fc, is_train,arch_model)
    #---------------------------------------train---------------------------------------------#
    net, _ = build_net(train_data, num_classes, dropout_prob, True, arch_model)
    variables_to_restore,variables_to_train = g_parameter(checkpoint_exclude_scopes)
    loss = cost(train_label, net)
    global_step = tf.Variable(0, trainable=False)  
    if learning_r_decay:
        learning_rate = tf.train.exponential_decay(  
            learning_rate_base,                     
            global_step * batch_size,  
            1000,     # 多少次衰减一次           
            decay_rate,                       
            staircase=True)  
    else:
        learning_rate = learning_rate_base
    if train_all_layers:
        variables_to_train = []
    optimizer = train_op(learning_rate, loss, variables_to_train, global_step)
    accuracy = model_accuracy(net, train_label, num_classes)
    #---------------------------------------valid---------------------------------------------#
    with tf.variable_scope("", reuse=tf.AUTO_REUSE) as scope:
        valid_net, _ = build_net(valid_data, num_classes, dropout_prob=1.0, False, arch_model)
    valid_loss = cost(valid_label, valid_net)
    valid_accuracy = model_accuracy(valid_net, valid_label, num_classes)
    #------------------------------------------------------------------------------------#
    sess = tf.InteractiveSession()
    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()
    saver2 = tf.train.Saver(tf.global_variables())
    # if not train_all_layers:
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
    
    # Start the queue runners.
    tf.train.start_queue_runners(sess= sess)
    
    # early stopping
    best_valid = np.inf
    best_valid_epoch = 0

    for epoch_i in range(epoch):
        for batch_i in range(int(10000/batch_size)):
            los, _ = sess.run([loss,optimizer])
            # print (los)
            if batch_i%100==0:
                loss_, acc_ = sess.run([loss, accuracy])
                print('Batch: {:>2}: Training loss: {:>3.5f}, Training accuracy: {:>3.5f}'.format(batch_i, loss_, acc_))
            
            if batch_i%500==0:
                ls, acc = sess.run([valid_loss, valid_accuracy])
                print('Batch: {:>2}: Validation loss: {:>3.5f}, Validation accuracy: {:>3.5f}'.format(batch_i, ls, acc))
            if batch_i%500==0:
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver2.save(sess, checkpoint_path, global_step=epoch_i, write_meta_graph=False)
    sess.close()
