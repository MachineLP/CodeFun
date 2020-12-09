# coding=utf-8
"""
Created on 2018 06.18
@author: liupeng
"""

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import cv2
import os
try:
    from image_loader import TripletImageLoader
except:
    from csn_lib.image_loader import TripletImageLoader

try:
    from data_load import load_image
except:
    from csn_lib.data_load import load_image
    
try:
    from csn_net import ConditionalSimNet, NetArch
except:
    from csn_lib.csn_net import ConditionalSimNet, NetArch
try:
    import config
except:
    from csn_lib import config

try:
    from utils import input_placeholder3, triplet_loss, g_parameter, get_next_batch_from_path, create_pairs
except:
    from csn_lib.utils import input_placeholder3, triplet_loss, g_parameter, get_next_batch_from_path, create_pairs


height = config.height
width = config.width
epoch = config.epoch
num_classes = config.num_classes
n_conditions = config.n_conditions
embedding_size = config.embedding_size
conditions = config.conditions
margin = config.margin
batch_size = config.batch_size
dropout_prob = config.dropout_prob
checkpoint_exclude_scopes = config.checkpoint_exclude_scopes
learning_r_decay = config.learning_r_decay
learning_rate_base = config.learning_rate_base
decay_rate = config.decay_rate
checkpoint_path = config.checkpoint_path
early_stop = config.early_stop
EARLY_STOP_PATIENCE = config.EARLY_STOP_PATIENCE
embed_loss = config.embed_loss
mask_loss = config.mask_loss
hard_sample_train = config.hard_sample_train

def GPU_config(rate=0.5):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    gpuConfig = tf.ConfigProto()
    gpuConfig.allow_soft_placement = False
    gpuConfig.gpu_options.allow_growth = True
    gpuConfig.gpu_options.per_process_gpu_memory_fraction = rate

    return gpuConfig


def train(train_data_trip, val_data_trip, test_data_trip):
    
    np.random.shuffle(train_data_trip)
    np.random.shuffle(val_data_trip)
    np.random.shuffle(test_data_trip)
    
    train_n = len(train_data_trip)
    # print ('pairs_data', train_data)
    valid_n = len(val_data_trip)
    # print ('pairs_data', valid_data)
    test_n = len(test_data_trip)
    # ---------------------------------------------------------------------------------#
    X0,X1,X2, Y, is_train, keep_prob_fc = input_placeholder3(height, width, num_classes)
    C = tf.placeholder(tf.int32, [None, None])

    # arch_resnet_v2_50, arch_inception_v4, arch_vgg16
    with tf.variable_scope("", reuse=tf.AUTO_REUSE) as scope:
        _, net_vis_x = NetArch().arch_vgg16(X0, num_classes, keep_prob_fc, is_train, embedding_size)
        _, net_vis_y = NetArch().arch_vgg16(X1, num_classes, keep_prob_fc, is_train, embedding_size)
        _, net_vis_z = NetArch().arch_vgg16(X2, num_classes, keep_prob_fc, is_train, embedding_size)
        with tf.variable_scope("Logits_csn"):
            csn = ConditionalSimNet(n_conditions, embedding_size, learnedmask=True, prein=True)
            embedded_x, masknorm_norm_x, embed_norm_x, tot_embed_norm_x = csn.forward(net_vis_x, C)
            embedded_y, masknorm_norm_y, embed_norm_y, tot_embed_norm_y = csn.forward(net_vis_y, C)
            embedded_z, masknorm_norm_z, embed_norm_z, tot_embed_norm_z = csn.forward(net_vis_z, C)
            mask_norm = (masknorm_norm_x + masknorm_norm_y + masknorm_norm_z) / 3
            embed_norm = (embed_norm_x + embed_norm_y + embed_norm_z) / 3
            mask_embed_norm = (tot_embed_norm_x + tot_embed_norm_y + tot_embed_norm_z) / 3
    #x: Anchor image,
    #y: Distant (negative) image,
    #z: Close (positive) image
    # print (mask_norm)
    loss_triplet = triplet_loss(embedded_x,embedded_z,embedded_y,batch_size,alpha=margin,hard_sample=hard_sample_train)
    loss_embedd = embed_norm / np.sqrt(batch_size)
    loss_mask = mask_norm / batch_size
    loss = loss_triplet + embed_loss * loss_embedd + mask_loss * loss_mask

    variables_to_restore,variables_to_train = g_parameter(checkpoint_exclude_scopes)
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
    # l_v = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"Logits_csn")
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step, var_list=tf.global_variables())
    # print ('>>>>>>', tf.global_variables())
    # accuracy = tf.reduce_sum(loss)
    # accuracy = model_accuracy(net0, Y, num_classes)
    correct_pred = tf.greater(tf.reduce_sum(tf.square(tf.subtract(embedded_x, embedded_y)),1), tf.reduce_sum(tf.square(tf.subtract(embedded_x, embedded_z)),1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    #------------------------------------------------------------------------------------#
    sess = tf.Session(config=GPU_config())
    init = tf.global_variables_initializer()
    sess.run(init)
    saver2 = tf.train.Saver(tf.global_variables())
    #if not train_all_layers:
    try:
        saver_net = tf.train.Saver(variables_to_restore)
        saver_net.restore(sess, checkpoint_path)
    except:
        pass
    
    # early stopping
    best_valid = np.inf
    best_valid_epoch = 0


    for epoch_i in range(epoch):
        for batch_i in range(int(train_n/batch_size)):
            images0,images1,images2,cc = get_next_batch_from_path(train_data_trip, batch_i, height, width, batch_size=batch_size, training=True)
            # print (images0,images1,images2,cc)
            los, _ = sess.run([loss,optimizer], feed_dict={X0: images0,X1: images1,X2: images2,C:cc,is_train:True, keep_prob_fc:dropout_prob})
            print (los)
            ck_path = os.path.join('model', 'model.ckpt')
            saver2.save(sess, ck_path, global_step=batch_i, write_meta_graph=False)
            if batch_i%20==0:
                loss_, acc_ = sess.run([loss, accuracy], feed_dict={X0: images0,X1: images1,X2: images2, C:cc,is_train:False, keep_prob_fc:1.0})
                print('Batch: {:>2}: Training loss: {:>3.5f}, Training accuracy: {:>3.5f}'.format(batch_i, loss_, acc_))

            if batch_i%100==0:
                images0,images1,images2,cc = get_next_batch_from_path(val_data_trip, batch_i%(int(valid_n/batch_size)), height, width, batch_size=batch_size, training=False)
                ls, acc = sess.run([loss, accuracy], feed_dict={X0: images0,X1: images1,X2: images2,C:cc, is_train:False, keep_prob_fc:1.0})
                print('Batch: {:>2}: Validation loss: {:>3.5f}, Validation accuracy: {:>3.5f}'.format(batch_i, ls, acc))
            
        
        print('Epoch===================================>: {:>2}'.format(epoch_i))
        valid_ls = 0
        valid_acc = 0
        for batch_i in range(int(valid_n/batch_size)):
            images_valid0,images_valid1,images_valid2,cc = get_next_batch_from_path(val_data_trip, batch_i, height, width, batch_size=batch_size, training=False)
            epoch_ls, epoch_acc = sess.run([loss, accuracy], feed_dict={X0:images_valid0,X1:images_valid1,X2:images_valid2, C:cc,keep_prob_fc:1.0, is_train:False})
            valid_ls = valid_ls + epoch_ls
            valid_acc = valid_acc + epoch_acc
        print('Epoch: {:>2}: Validation loss: {:>3.5f}, Validation accuracy: {:>3.5f}'.format(epoch_i, valid_ls/int(valid_n/batch_size), valid_acc/int(valid_n/batch_size)))
        
        '''
        if valid_acc/int(valid_n/batch_size) > 0.90:
            ck_path = os.path.join('model', 'model.ckpt')
            saver2.save(sess, ck_path, global_step=epoch_i, write_meta_graph=False)'''
        
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
        
        np.random.shuffle(train_data_trip)
        np.random.shuffle(val_data_trip)
        np.random.shuffle(test_data_trip)

    sess.close()

def gen_train_val_data(sample_dir, train_rate, num_classes, condition_classes):
    train_data, train_label, valid_data, valid_label, train_n, valid_n, note_label = load_image(sample_dir, 1.0).gen_train_valid()
    train_data_trip = create_pairs(train_data, num_classes, condition_classes)
    train_num = int(train_n*train_rate)
    train_data_trip = train_data_trip[0:train_num]
    val_data_trip = train_data_trip[-train_num:-1]
    test_data_trip = val_data_trip
    return train_data_trip, val_data_trip, test_data_trip

if __name__ == '__main__':
    # 获取训练数据
    train_data_trip = TripletImageLoader('data', 'ut-zap50k-images', 'filenames.json', conditions, 'train', n_triplets=100000).get_trip_data()
    print (train_data_trip[:100])
    val_data_trip = TripletImageLoader('data', 'ut-zap50k-images', 'filenames.json', conditions, 'val', n_triplets=80000).get_trip_data()
    print (val_data_trip[:100])
    test_data_trip = TripletImageLoader('data', 'ut-zap50k-images', 'filenames.json', conditions, 'test', n_triplets=160000).get_trip_data()
    print (test_data_trip[:100])
    train(train_data_trip, val_data_trip, test_data_trip)
    
    '''
    sample_dir = 'gender'
    train_rate = 0.9
    condition_classes = 0
    num_classes = 4
    train_data_trip, val_data_trip, test_data_trip = gen_train_val_data(sample_dir, train_rate, num_classes, condition_classes)
    train(train_data_trip, val_data_trip, test_data_trip)
    '''
