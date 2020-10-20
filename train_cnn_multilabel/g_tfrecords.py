# coding=utf-8
"""
Created on 2017 10.17
@author: liupeng
"""

import numpy as np
import os
import tensorflow as tf
slim = tf.contrib.slim
from lib.data_load.data_load_from_txt_mullabel import data_load_from_txt_mullabel
from lib.model.build_model.build_net import net_arch
from lib.utils.multi_label_utils import g_parameter
from lib.utils.multi_label_utils import to_one_hot
from PIL import Image
# from lib.train.train3 import train3 as train
from keras.utils import np_utils
import config
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sample_dir = config.sample_dir
num_classes = config.num_classes
batch_size = config.batch_size
arch_model = config.arch_model
checkpoint_exclude_scopes = config.checkpoint_exclude_scopes
dropout_prob = config.dropout_prob
train_rate = config.train_rate
epoch = config.epoch
# 是否使用提前终止训练
early_stop = config.early_stop
EARLY_STOP_PATIENCE = config.EARLY_STOP_PATIENCE
# 是否使用learning_rate
learning_r_decay = config.learning_r_decay
learning_rate_base = config.learning_rate_base
decay_rate = config.decay_rate
height, width = config.height, config.width
# 模型保存的路径
train_dir = config.train_dir
# 是否进行fine-tune。 选择fine-tune的的参数
fine_tune = config.fine_tune
# 训练所有层的参数
train_all_layers = config.train_all_layers
# 迁移学习的网络模型
checkpoint_path = config.checkpoint_path


from lib.train.train_multi_label import train_multi_label as train
train_data, train_label, valid_data, valid_label, train_n, valid_n, note_label = data_load_from_txt_mullabel(sample_dir, train_rate).gen_train_valid()

def arr_to_list(train_label):
    train_labels = []
    for label_i in train_label:
        tmp_label = []
        for label_j in label_i:
            tmp_label.append(label_j)
        train_labels.append(tmp_label)
    return train_labels

print ('note_label', note_label)
print (train_data)
print (train_label)
if arch_model!='arch_seg_vgg16_conv' and arch_model!='arch_vgg16_ocr':
    # train_label = np_utils.to_categorical(train_label, num_classes)
    # valid_label = np_utils.to_categorical(valid_label, num_classes)
    train_label = to_one_hot(train_label, num_classes)
    valid_label = to_one_hot(valid_label, num_classes)

train_label = arr_to_list(train_label)
valid_label = arr_to_list(valid_label)
print (train_label)


if not os.path.isdir(train_dir):
    os.makedirs(train_dir)

class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image
def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto.
        """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Wrapper for inserting float features into Example proto.
        """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto.
        """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

TFwriter_train = tf.python_io.TFRecordWriter("img_train.tfrecords")
TFwriter_test = tf.python_io.TFRecordWriter("img_test.tfrecords")

image_reader = ImageReader()
sess = tf.Session()
for i in range(len(train_data)):
    #image_data = tf.gfile.FastGFile(img_path, 'rb').read()
    #height, width = image_reader.read_image_dims(sess, image_data)
    img = Image.open(train_data[i])
    img = img.resize((height, width))
    image_data = img.tobytes()
    height, width = height, width
    label = train_label[i]
    example = tf.train.Example(features=tf.train.Features(feature={
                                                          "label":float_feature(label),
                                                          "img":bytes_feature(image_data),
                                                          "height":int64_feature(height),
                                                          "width":int64_feature(width)
                                                          }) )
    TFwriter_train.write(example.SerializeToString())
for i in range(len(valid_data)):
    #image_data = tf.gfile.FastGFile(img_path, 'rb').read()
    #height, width = image_reader.read_image_dims(sess, image_data)
    img = Image.open(valid_data[i])
    img = img.resize((height, width))
    image_data = img.tobytes()
    height, width = height, width
    label = valid_label[i]
    example = tf.train.Example(features=tf.train.Features(feature={
                                                          "label":float_feature(label),
                                                          "img":bytes_feature(image_data),
                                                          "height":int64_feature(height),
                                                          "width":int64_feature(width)
                                                          }) )
    TFwriter_test.write(example.SerializeToString())
TFwriter_train.close()
TFwriter_test.close()
