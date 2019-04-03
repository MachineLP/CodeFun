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
import random


def _corrupt_brightness(image, label):
    """
    Radnomly applies a random brightness change.
    """
    cond_brightness = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32), tf.bool)
    image = tf.cond(cond_brightness, lambda: tf.image.random_hue(
        image, 0.1), lambda: tf.identity(image))
    return image, label


def _corrupt_contrast(image, label):
    """
    Randomly applies a random contrast change.
    """
    cond_contrast = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32), tf.bool)
    image = tf.cond(cond_contrast, lambda: tf.image.random_contrast(
        image, 0.2, 1.8), lambda: tf.identity(image))
    return image, label


def _corrupt_saturation(image, label):
    """
    Randomly applies a random saturation change.
    """
    cond_saturation = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32), tf.bool)
    image = tf.cond(cond_saturation, lambda: tf.image.random_saturation(
        image, 0.2, 1.8), lambda: tf.identity(image))
    return image, label

def _crop_random(image, label, crop_size=[400,400], channels=[3,3]):
    """
    Randomly crops image and mask in accord.
    """
    seed = random.random()
    cond_crop_image = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32, seed=seed), tf.bool)
    cond_crop_mask = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32, seed=seed), tf.bool)

    image = tf.cond(cond_crop_image, lambda: tf.random_crop(
        image, [crop_size[0], crop_size[1], channels[0]], seed=seed), lambda: tf.identity(image))
    # mask = tf.cond(cond_crop_mask, lambda: tf.random_crop(
    #    mask, [crop_size[0], crop_size[1], channels[1]], seed=seed), lambda: tf.identity(mask))
    return image, label


def _flip_left_right(image, label):
    """Randomly flips image and label left or right in accord."""
    seed = random.random()
    image = tf.image.random_flip_left_right(image, seed=seed)
    # label = tf.image.random_flip_left_right(label, seed=seed)

    return image, label

def _flip_up_down(image, label):
    """Randomly flips image and label left or right in accord."""
    seed = random.random()
    image = tf.image.random_flip_up_down(image, seed=seed)
    #label = tf.image.random_flip_up_down(label, seed=seed)

    return image, label

def _rotate90(image, label):
    '''Randomly rotate k*90 degree'''
    k = tf.random_uniform([1],minval=0,maxval=4,dtype=tf.int32)[0]
    image = tf.image.rot90(image,k)
    #label = tf.image.rot90(label,k)

    return image, label


def _normalize_data(image, label):
    """Normalize image and label within range 0-1."""
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = image - 0.5
    image = image * 2.0

    #label = tf.cast(label, tf.float32)
    #label = label / 255.0

    return image, label

def _parse_data(image_paths, label_paths):
    """Reads image and label files"""

    image_content = tf.read_file(image_paths)
    # label_content = tf.read_file(label_paths)

    images = tf.image.decode_png(image_content, channels=3)
    images = tf.image.resize_images(images, [299,299])
    # labels = tf.image.decode_png(label_content, channels=3)
    labels = label_paths

    return images, labels

def _data_aug(data, num_threads, num_prefetch, augment=['brightness', 'contrast', 'saturation', 'flip_ud', 'flip_lr', 'rot90'], normalize=True):

    if 'brightness' in augment:
        print ('brightness')
        data = data.map(_corrupt_brightness,
                        num_parallel_calls=num_threads).prefetch(num_prefetch)

    if 'contrast' in augment:
        print ('contrast')
        data = data.map(_corrupt_contrast,
                        num_parallel_calls=num_threads).prefetch(num_prefetch)

    if 'saturation' in augment:
        print ('saturation')
        data = data.map(_corrupt_saturation,
                        num_parallel_calls=num_threads).prefetch(num_prefetch)

    if 'crop_random' in augment:
        print ('crop_random')
        data = data.map(_crop_random,
                        num_parallel_calls=num_threads).prefetch(num_prefetch)
    

    if 'flip_ud' in augment:
        print ('flip_ud')
        data = data.map(_flip_up_down,
                        num_parallel_calls=num_threads).prefetch(num_prefetch)

    if 'flip_lr' in augment:
        print ('flip_lr')
        data = data.map(_flip_left_right,
                        num_parallel_calls=num_threads).prefetch(num_prefetch)

    if 'rot90' in augment:
        print ('rot90')
        data = data.map(_rotate90,
                        num_parallel_calls=num_threads).prefetch(num_prefetch)
    # Normalize
    if normalize:
        data = data.map(_normalize_data,
                        num_parallel_calls=num_threads).prefetch(num_prefetch)
    return data


def read_inputs(sample_dir, train_rate, batch_size, is_training=True, num_threads=20):

  train_data, train_label, valid_data, valid_label, train_n, valid_n, note_label = load_image(sample_dir, train_rate).gen_train_valid()

  num_threads = num_threads
  num_prefetch = 5*batch_size
  train_num_sample = len(train_data)
  valid_num_sample = len(valid_data)


  train_images = tf.convert_to_tensor(train_data, dtype=tf.string)
  train_labels = tf.convert_to_tensor(train_label, dtype=tf.int64)

  train_data = tf.data.Dataset.from_tensor_slices((train_images, train_labels))

  train_data = train_data.shuffle(buffer_size=train_num_sample)

  train_data = train_data.map(_parse_data, num_parallel_calls=num_threads).prefetch(num_prefetch)

  train_data = _data_aug(train_data, num_threads, num_prefetch)
  train_data = train_data.batch(batch_size)

  epoch = None
  # 无限重复数据集
  train_data = train_data.repeat(epoch)
  # train_iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
  train_iterator = train_data.make_one_shot_iterator()

  train_next_element = train_iterator.get_next()
  train_images, train_label_batch = train_next_element
  # ----------------------------------------------------------------------------------- # 

  valid_images = tf.convert_to_tensor(valid_data, dtype=tf.string)
  valid_labels = tf.convert_to_tensor(valid_label, dtype=tf.int64)

  valid_data = tf.data.Dataset.from_tensor_slices((valid_images, valid_labels))

  valid_data = valid_data.shuffle(buffer_size=valid_num_sample)

  valid_data = valid_data.map(_parse_data, num_parallel_calls=num_threads).prefetch(num_prefetch)

  valid_data = _data_aug(valid_data, num_threads, num_prefetch,  augment=[])  # 要做归一化啊
  valid_data = valid_data.batch(batch_size)

  epoch = None
  # 无限重复数据集
  valid_data = valid_data.repeat(epoch)
  # valid_iterator = tf.data.Iterator.from_structure(valid_data.output_types, valid_data.output_shapes)
  valid_iterator = valid_data.make_one_shot_iterator()

  valid_next_element = valid_iterator.get_next()

  valid_images, valid_label_batch = valid_next_element

  #with tf.Session() as sess:
  #    image, mask = sess.run([train_images, train_label_batch])
  #    print (mask)
  return train_images, train_label_batch, valid_images, valid_label_batch, train_n, valid_n


def read_inputs_(sample_dir, train_rate, batch_size, is_training=True, num_threads=20):
  
  train_data, train_label, valid_data, valid_label, train_n, valid_n, note_label = load_image(sample_dir, train_rate).gen_train_valid()

  # Create a queue that produces the filenames to read.
  train_filename_queue = tf.train.slice_input_producer([train_data, train_label], shuffle= True, capacity= 1024)
  valid_filename_queue = tf.train.slice_input_producer([valid_data, valid_label], shuffle= False,  capacity= 1024, num_epochs =1)

  # Read examples from files in the filename queue.
  train_file_content = tf.read_file(train_filename_queue[0])
  valid_file_content = tf.read_file(valid_filename_queue[0])
  # Read JPEG or PNG or GIF image from file
  train_reshaped_image = tf.to_float(tf.image.decode_jpeg(train_file_content, channels=3))
  valid_reshaped_image = tf.to_float(tf.image.decode_jpeg(valid_file_content, channels=3))
  # Resize image to 256*256
  train_reshaped_image = tf.image.resize_images(train_reshaped_image, [299,299])
  valid_reshaped_image = tf.image.resize_images(valid_reshaped_image, [299,299])

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
        [valid_reshaped_image, valid_label],
        batch_size= batch_size,
        allow_smaller_final_batch= True if not is_training else False,
        num_threads=num_threads,
        capacity=min_queue_examples+3 * batch_size)
  return train_images, train_label_batch, valid_images, valid_label_batch

def random_rotate_image(image):
    def random_rotate_image_func(img):
        if np.random.choice([True, False]):
            w,h = img.shape[1], img.shape[0]
            angle = np.random.randint(0,360)
            rotate_matrix = cv2.getRotationMatrix2D(center=(img.shape[1]/2, img.shape[0]/2), angle=angle, scale=0.7)
            img = cv2.warpAffine(img, rotate_matrix, (w,h))
        return img 
    image_rotate = tf.py_func(random_rotate_image_func, [image], tf.float32)
    return image_rotate

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

  #
  reshaped_image = random_rotate_image(reshaped_image)
  
  # Subtract off the mean and divide by the variance of the pixels.
  reshaped_image = tf.image.per_image_standardization(reshaped_image)

  # Set the shapes of tensors.
  reshaped_image.set_shape([crop_size, crop_size, num_channels])
  #read_input.label.set_shape([1])
  return reshaped_image

'''
def per_image_standardization(img):
    # stat = ImageStat.Stat(img)
    # mean = stat.mean
    # stddev = stat.stddev
    # img = (np.array(img) - stat.mean)/stat.stddev
    if img.mode == 'RGB':
        channel = 3
    num_compare = img.size[0] * img.size[1] * channel
    img_arr=np.array(img)
    #img_arr=np.flip(img_arr,2)
    img_t = (img_arr - np.mean(img_arr))/max(np.std(img_arr), 1/num_compare)
    return img_t
'''

def _test_preprocess(reshaped_image, crop_size=299,num_channels=3):

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, crop_size, crop_size)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(resized_image)

  # Set the shapes of tensors.
  float_image.set_shape([crop_size, crop_size, num_channels])

  return float_image
