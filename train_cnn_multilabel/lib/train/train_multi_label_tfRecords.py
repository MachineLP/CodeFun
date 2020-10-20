# coding=utf-8
"""
Created on 2017 10.17
@author: liupeng
"""

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
from lib.utils.multi_label_utils import get_next_batch_from_path, shuffle_train_data
from lib.utils.multi_label_utils import input_placeholder, build_net_multi_label, cost, train_op, model_mAP
import os
import sys

def read_image(file_queue, num_classes, height, width):
    reader = tf.TFRecordReader()
    key, value = reader.read(file_queue)
    _, serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(
       serialized_example,
       features={
           'img': tf.FixedLenFeature([], tf.string),
           'label': tf.FixedLenFeature([num_classes], tf.float32),
           'height': tf.FixedLenFeature([], tf.int64),
           'width': tf.FixedLenFeature([], tf.int64),
           })
    image = tf.decode_raw(features['img'], tf.uint8)
    h = features['height']
    w = features['width']
    # print (h)
    image = tf.reshape(image, [height, width,3])
    
    label = features['label']
    label = tf.reshape(label, [num_classes])
    label = tf.cast(label, tf.float32)
    # print (image, label)
    return image, label

def read_image_batch(file_queue, num_classes, batchsize, height, width):
    img, label = read_image(file_queue, num_classes, height, width)
    min_after_dequeue = 2000
    capacity = min_after_dequeue + 3 * batchsize
    # image_batch, label_batch = tf.train.batch([img, label], batch_size=batchsize, capacity=capacity)
    image_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=batchsize, capacity=capacity, min_after_dequeue=min_after_dequeue)
    label_batch = tf.to_float(label_batch)
    return image_batch, label_batch

def train_multi_label_tfRecords(train_file_tfRecord,valid_file_tfRecord,train_dir,num_classes,batch_size,arch_model,learning_r_decay,learning_rate_base,decay_rate,dropout_prob,epoch,height,width,checkpoint_exclude_scopes,early_stop,EARLY_STOP_PATIENCE,fine_tune,train_all_layers,checkpoint_path,train_n,valid_n,g_parameter):
    # ---------------------------------------------------------------------------------#
    print (train_file_tfRecord)
    train_image_filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(train_file_tfRecord))
    train_images, train_labels = read_image_batch(train_image_filename_queue, num_classes, batch_size, height, width)
    valid_image_filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(valid_file_tfRecord))
    valid_images, valid_labels = read_image_batch(valid_image_filename_queue, num_classes, batch_size, height, width)
    # ---------------------------------------------------------------------------------#
    X, Y, is_train, keep_prob_fc = input_placeholder(height, width, num_classes)
    net, _ = build_net_multi_label(X, num_classes, keep_prob_fc, is_train,arch_model)
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
    pre = tf.nn.sigmoid(net)
    accuracy = model_mAP(pre, Y)
    #------------------------------------------------------------------------------------#
    sess = tf.InteractiveSession()
    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()
    #init = tf.global_variables_initializer()
    #sess.run(init)
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

    # start queue runner
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for epoch_i in range(epoch):
        for batch_i in range(int(train_n/batch_size)):
            train_image = tf.image.resize_images(train_images, (height, width), method=0)
            train_image = (tf.cast(train_image, tf.float32)/255 - 0.5)*2
            train_image, train_label = sess.run([train_image, train_labels])
            # print (train_image)
            # print (train_label)
            los, _ = sess.run([loss,optimizer], feed_dict={X: train_image, Y: train_label, is_train:True, keep_prob_fc:dropout_prob})
            # print (los)
            #checkpoint_path = os.path.join(train_dir, 'model.ckpt')
            #saver2.save(sess, checkpoint_path, global_step=batch_i, write_meta_graph=False)
            if batch_i%100==0:
                loss_, acc_ = sess.run([loss, accuracy], feed_dict={X: train_image, Y: train_label, is_train:False, keep_prob_fc:1.0})
                print('Batch: {:>2}: Training loss: {:>3.5f}, Training mAP: {:>3.5f}'.format(batch_i, loss_, acc_))

            if batch_i%500==0:
                valid_image = tf.image.resize_images(valid_images, (height, width), method=0)
                valid_image = (tf.cast(valid_image, tf.float32)/255 - 0.5)*2
                valid_image, valid_label = sess.run([valid_image, valid_labels])
                ls, acc = sess.run([loss, accuracy], feed_dict={X: valid_image, Y: valid_label, is_train:False, keep_prob_fc:1.0})
                print('Batch: {:>2}: Validation loss: {:>3.5f}, Validation mAP: {:>3.5f}'.format(batch_i, ls, acc))
            if batch_i%500==0:
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver2.save(sess, checkpoint_path, global_step=batch_i, write_meta_graph=False)
        
        print('Epoch===================================>: {:>2}'.format(epoch_i))
        valid_ls = 0
        valid_acc = 0
        for batch_i in range(int(valid_n/batch_size)):
            valid_image = tf.image.resize_images(valid_images, (height, width), method=0)
            valid_image = (tf.cast(valid_image, tf.float32)/255 - 0.5)*2
            valid_image, valid_label = sess.run([valid_image, valid_labels])
            # images_valid, labels_valid = get_next_batch_from_path(valid_image, valid_label, batch_i, height, width, batch_size=batch_size, training=False)
            epoch_ls, epoch_acc = sess.run([loss, accuracy], feed_dict={X: valid_image, Y: valid_label, keep_prob_fc:1.0, is_train:False})
            valid_ls = valid_ls + epoch_ls
            valid_acc = valid_acc + epoch_acc
        print('Epoch: {:>2}: Validation loss: {:>3.5f}, Validation mAP: {:>3.5f}'.format(epoch_i, valid_ls/int(valid_n/batch_size), valid_acc/int(valid_n/batch_size)))

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
        # train_data, train_label = shuffle_train_data(train_data, train_label)
    # stop queue runner
    coord.request_stop()
    coord.join(threads)
    sess.close()
