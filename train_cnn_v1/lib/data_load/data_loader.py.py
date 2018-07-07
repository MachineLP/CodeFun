# -*- coding: utf-8 -*-
"""
Created on 2017 10.17
@author: liupeng
"""
import os
import sys

from six.moves import xrange
import tensorflow as tf
from lib.data_load.data_load import load_image


train_data, train_label, valid_data, valid_label, train_n, valid_n, note_label = load_image(sample_dir, train_rate).gen_train_valid()


def read_inputs(sample_dir, train_rate, batch_size, is_training=True, num_threads=20):
  
  train_data, train_label, valid_data, valid_label, train_n, valid_n, note_label = load_image(sample_dir, train_rate).gen_train_valid()

  # Create a queue that produces the filenames to read.
  train_filename_queue = tf.train.slice_input_producer([train_data, train_label], shuffle= args.shuffle, capacity= 1024)
  valid_filename_queue = tf.train.slice_input_producer([valid_data, valid_label], shuffle= False,  capacity= 1024, num_epochs =1)

  # Read examples from files in the filename queue.
  train_file_content = tf.read_file(train_filename_queue[0])
  valid_file_content = tf.read_file(valid_filename_queue[0])
  # Read JPEG or PNG or GIF image from file
  train_reshaped_image = tf.to_float(tf.image.decode_jpeg(train_file_content, channels=3))
  valid_reshaped_image = tf.to_float(tf.image.decode_jpeg(valid_file_content, channels=3))
  # Resize image to 256*256
  train_reshaped_image = tf.image.resize_images(train_reshaped_image, [326,326])
  valid_reshaped_image = tf.image.resize_images(valid_reshaped_image, args.load_size)

  train_label = tf.cast(train_filename_queue[1], tf.int64)
  valid_label = tf.cast(valid_filename_queue[1], tf.int64)


  train_reshaped_image = _train_preprocess(train_reshaped_image)
  valid_reshaped_image = _test_preprocess(valid_reshaped_image)

   # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(5000 * min_fraction_of_examples_in_queue)
  #print(batch_size)
  print ('Filling queue with %d images before starting to train. '
         'This may take some times.' % min_queue_examples)

  train_images, train_label_batch = tf.train.batch(
        [train_reshaped_image, train_label],
        batch_size= batch_size,
        allow_smaller_final_batch= True if not is_training else False,
        num_threads=num_threads,
        capacity=min_queue_examples+3 * batch_size)
  valid_images, valid_label_batch = tf.train.batch(
        [train_reshaped_image, train_label],
        batch_size= batch_size,
        allow_smaller_final_batch= True if not is_training else False,
        num_threads=num_threads,
        capacity=min_queue_examples+3 * batch_size)
  return train_images, train_label_batch, valid_images, valid_label_batch


def _train_preprocess(reshaped_image,crop_size=299,num_channels=3):
  # Image processing for training the network. Note the many random
  # distortions applied to the image.

  # Randomly crop a [height, width] section of the image.
  reshaped_image = tf.random_crop(reshaped_image, [crop_size, crop_size, num_channels])

  # Randomly flip the image horizontally.
  reshaped_image = tf.image.random_flip_left_right(reshaped_image)

  # Because these operations are not commutative, consider randomizing
  # the order their operation.
  reshaped_image = tf.image.random_brightness(reshaped_image,
                                               max_delta=63)
  # Randomly changing contrast of the image
  reshaped_image = tf.image.random_contrast(reshaped_image,
                                             lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  reshaped_image = tf.image.per_image_standardization(reshaped_image)

  # Set the shapes of tensors.
  reshaped_image.set_shape([crop_size, crop_size, num_channels)
  #read_input.label.set_shape([1])
  return reshaped_image


def _test_preprocess(reshaped_image, crop_size=299,num_channels=3):

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, crop_size, crop_size)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(resized_image)

  # Set the shapes of tensors.
  float_image.set_shape([crop_size, crop_size, num_channels])

  return float_image
