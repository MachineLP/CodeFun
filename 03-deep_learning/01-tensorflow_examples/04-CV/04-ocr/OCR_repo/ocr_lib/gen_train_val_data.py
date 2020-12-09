#-*- coding:utf-8 -*-
"""
Created on 2018 08.20
@author: liupeng
wechat: lp9628
blog: http://blog.csdn.net/u014365862/article/details/78422372
github：https://github.com/MachineLP
"""

import numpy as np
import random
import os
from tqdm import tqdm
import pandas as pd
import cv2
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import l2
from keras.callbacks import *
from keras.optimizers import *
from keras.callbacks import *
from keras import backend as K
# 使用多GPU
# from keras.utils import multi_gpu_model
import string
from PIL import Image, ImageFont, ImageDraw
from skimage import exposure
import random

from ocr_lib.data_generator  import data_generator


class gen_train_val_data(object):
    
    def __init__(self, width, height, n_len, pooling_num):

        self._width = width #width
        self._height = height
        self._n_len = n_len
        self._conv_shape = [height, width//2**3] # conv_shape
        

    def gen_train(self, batch_size=128, num_workers=12, alphabet=''):
        X = np.zeros((batch_size, self._height, self._width, 3), dtype=np.uint8)
        y = np.zeros((batch_size, self._n_len), dtype=np.int32)
        label_length = np.ones(batch_size)
        while True:
            # XX = []
            for i in range(batch_size):
            
                img, img_label = data_generator().generate(alphabet, self._n_len)
                # cv2.imwrite('res.jpg', img)
                
                # 进行预处理  加一个规则， 如果图片小于我们设定的宽加padding， 如果大于则要进行resize。
                h, w, _ = img.shape

                # 加入光照和对比度的变化
                e_rate = np.random.uniform(0.5,1.5)
                img = exposure.adjust_gamma(img, e_rate)

                angle = np.random.randint(-2,2)
                RotateMatrix = cv2.getRotationMatrix2D(center=(w/2, h/2), angle=angle, scale=1.0)
                # image = cv2.warpAffine(image, RotateMatrix, (w,h), borderValue=(129,137,130))
                img = cv2.warpAffine(img, RotateMatrix, (w,h), borderMode=cv2.BORDER_REPLICATE)
    
                alpha = h * (1/60)
                img = cv2.resize(img, (int(w/alpha), int(h/alpha)) )
                h, w, _ = img.shape
                # print ('>>>>>>', h, w) 
                X[i] = np.array(cv2.resize(img, (self._width, self._height))) #.transpose(1, 0, 2))  # / 127.5 - 1.0
                ########

                label_lg = len(img_label)
                labels = []
                # 下面要注意哦， 在离线训练的时候要-1， 为了去除换行符。 在线训练的时候没必要。
                for label_id in range(label_lg):
                    # print ( img_label[label_id] )
                    labels.append( int (img_label[label_id]) )
                labels = np.array(labels)
                # print ('>>>>>>>', labels)
                #print (len(labels))
                #print (labels)
    
                #print ('len>>>>>>', n_len)
    
                # XX.append(img_crop_pre(X[i]))
                y[i,:len(labels)] = [x for x in labels]
                # print (y[i,:len(random_str)])
                y[i,len(labels):] = -1
                label_length[i] = len(labels)

            yield [X, y, np.ones(batch_size)*int(self._conv_shape[1]-2), label_length], np.ones(batch_size)


    def gen_val(self, batch_size=128, num_workers=12, alphabet=''):
        X = np.zeros((batch_size, self._height, self._width, 3), dtype=np.uint8)
        y = np.zeros((batch_size, self._n_len), dtype=np.int32)
        label_length = np.ones(batch_size)
        while True:
            # XX = []
            for i in range(batch_size):

                img, img_label = data_generator().generate(alphabet, self._n_len)
            
                # 进行预处理  加一个规则， 如果图片小于我们设定的宽加padding， 如果大于则要进行resize。
                h, w, _ = img.shape

                # 加入光照和对比度的变化
                e_rate = np.random.uniform(0.5,1.5)
                img = exposure.adjust_gamma(img, e_rate)

                angle = np.random.randint(-2,2)
                RotateMatrix = cv2.getRotationMatrix2D(center=(w/2, h/2), angle=angle, scale=1.0)
                # image = cv2.warpAffine(image, RotateMatrix, (w,h), borderValue=(129,137,130))
                img = cv2.warpAffine(img, RotateMatrix, (w,h), borderMode=cv2.BORDER_REPLICATE)


                alpha = h * (1/60)
                img = cv2.resize(img, (int(w/alpha), int(h/alpha)) )
                h, w, _ = img.shape
                # print ('>>>>>>', h, w)
                X[i] = np.array(cv2.resize(img, (self._width, self._height))) #.transpose(1, 0, 2))  # / 127.5 - 1.0
                ########

                label_lg = len(img_label)
                labels = []
                for label_id in range(label_lg):
                   # print ( img_label[label_id] )
                      labels.append( int (img_label[label_id]) )
                labels = np.array(labels)
                #print (len(labels))
                #print (labels)
    
                #print ('len>>>>>>', n_len)
    
                # XX.append(img_crop_pre(X[i]))
                y[i,:len(labels)] = [x for x in labels]
                # print (y[i,:len(random_str)])
                y[i,len(labels):] = -1
                label_length[i] = len(labels)

            yield [X, y, np.ones(batch_size)*int(self._conv_shape[1]-2), label_length], np.ones(batch_size)
