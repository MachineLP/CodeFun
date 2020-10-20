# -*- coding: utf-8 -*-

import numpy as np  
import tensorflow as tf
import argparse
import os

from PIL import Image

FLAGS = None

batch_size = 64

def read_image(file_queue, num_classes):
    reader = tf.TFRecordReader()
    key, value = reader.read(file_queue)
    _, serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(
       serialized_example,
       features={
           'img': tf.FixedLenFeature([], tf.string),
           'label': tf.FixedLenFeature([20], tf.float32),
           #'height': tf.FixedLenFeature([], tf.string),
           #'width': tf.FixedLenFeature([], tf.string),
           })
    image = tf.decode_raw(features['img'], tf.uint8)
    #h = tf.decode_raw(features['height'], tf.int64)
    #w = tf.decode_raw(features['width'], tf.int64)
    # image_shape = tf.stack([h, w, 3])
    image = tf.reshape(image, [299,299,3])
    # image = tf.image.resize_images(image, (299, 299), method=0) 
    # image = tf.cast(image, tf.float32)
    
    label = features['label']
    print (label)
    # label = tf.cast(features['label'], tf.string)
    # label = tf.reshape(label, [num_classes])
    # label = tf.cast(label, tf.float32)
    return image, label

def read_image_batch(file_queue, num_classes, batchsize):
    img, label = read_image(file_queue, num_classes)
    min_after_dequeue = 2
    capacity = min_after_dequeue + 3 * batchsize
    # image_batch, label_batch = tf.train.batch([img, label], batch_size=batchsize, capacity=capacity)
    image_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=batchsize, capacity=capacity, min_after_dequeue=min_after_dequeue)
    label_batch = tf.to_float(label_batch)
    return image_batch, label_batch


def main(_):
    
    train_file_path = "img_train.tfrecords"
    test_file_path = "img_test.tfrecords"
    model_path = "cnn_jisuanshi_model"
    
    train_image_filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(train_file_path))
    img, label = read_image(train_image_filename_queue, 20)
    train_images, train_labels = read_image_batch(train_image_filename_queue, 20,batch_size)
    
    sess = tf.InteractiveSession()
    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()
    
    # start queue runner
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    for i in range(2):
        images, labels = sess.run([img, label])
        print (images, labels)
        images = tf.reshape(images, [299,299, 3])
        images = tf.image.convert_image_dtype(images, dtype=tf.uint8)
        images = tf.image.encode_jpeg(images)
        # print(sess.run(labels))
        with tf.gfile.GFile('pic_%d.jpg' % i, 'wb') as f:
            f.write(sess.run(images))
        #img=Image.fromarray(images, 'RGB')#这里Image是之前提到的
        #img.save(str(i)+'_''Label_'+str(i)+'.jpg')#存下图片

    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    tf.app.run(main=main)


