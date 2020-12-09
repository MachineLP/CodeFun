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
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Permute
from ocr_lib.vgg16 import VGG16
from ocr_lib.vgg19 import VGG19
from ocr_lib.Xception import Xception
# 使用多GPU
# from keras.utils import multi_gpu_model
import string
from PIL import Image, ImageFont, ImageDraw
from skimage import exposure
import random


class CRNN(object):

    def __init__(self, width=480, height=60, n_len=30, characters_length=3500, rnn_size=128, l2_rate=1e-5, train=True):
        '''
            width : 图像的宽度。
            height ：图像的高度。
            n_len ：定义的最长字符
            characters_length ： 字典的长度
        '''

        self._height = height
        self._width = width
        self._n_len = n_len
        self._n_class = characters_length
        self._rnn_size = rnn_size
        self._l2_rate = l2_rate

        self._labels = Input(name='the_labels', shape=[self._n_len], dtype='float32')
        self._input_length = Input(name='input_length', shape=[1], dtype='int64')
        self._label_length = Input(name='label_length', shape=[1], dtype='int64')

        # 初始化模型
        self.model_init()


    def model_init(self):
        # self.base_model, self.conv_shape = self.model(self._width)
        # self.base_model, self.conv_shape = self.model_cnn(self._width)
        # self.base_model, self.conv_shape = self.model_vgg(self._width)
        self.base_model, self.conv_shape = self.crnn_pytorch_version(self._width)
        
        self.loss_out = Lambda(self.ctc_loss, output_shape=(1,),  name='ctc')([self.base_model.output, self._labels, self._input_length, self._label_length])

        self.model = Model(inputs=[self.input_tensor, self._labels, self._input_length, self._label_length], outputs=[self.loss_out])

    
    def model(self, width):
        
        self.input_tensor = Input((self._height, width, 3))
        x = self.input_tensor
        for i, n_cnn in enumerate([3, 4, 6]):
            for j in range(n_cnn):
                x = Conv2D(32*2**i, (3, 3), padding='same', kernel_initializer='he_uniform', 
                           kernel_regularizer=l2(self._l2_rate))(x)
                x = BatchNormalization(gamma_regularizer=l2(self._l2_rate), beta_regularizer=l2(self._l2_rate))(x)
                x = Activation('relu')(x)
            x = MaxPooling2D((2, 2))(x)
        cnn_model = Model(self.input_tensor, x, name='cnn')

        x = cnn_model(self.input_tensor)

        conv_shape = x.get_shape().as_list()
        #rnn_length = conv_shape[1]
        #rnn_dimen = conv_shape[3]*conv_shape[2]

        #print (conv_shape, rnn_length, rnn_dimen)
        # x = Reshape(target_shape=(rnn_length, rnn_dimen))(x)
        x = Permute((2,1,3),name='permute')(x)
        x = TimeDistributed(Flatten(), name='flatten')(x)
        rnn_imp = 0

        x = Dense(self._rnn_size, kernel_initializer='he_uniform', kernel_regularizer=l2(self._l2_rate), bias_regularizer=l2(self._l2_rate))(x)
        x = BatchNormalization(gamma_regularizer=l2(self._l2_rate), beta_regularizer=l2(self._l2_rate))(x)
        x = Activation('relu')(x)
        # x = Dropout(0.2)(x)

        gru_1 = GRU(self._rnn_size, implementation=rnn_imp, return_sequences=True, name='gru1')(x)
        gru_1b = GRU(self._rnn_size, implementation=rnn_imp, return_sequences=True, go_backwards=True, name='gru1_b')(x)
        gru1_merged = add([gru_1, gru_1b])

        gru_2 = GRU(self._rnn_size, implementation=rnn_imp, return_sequences=True, name='gru2')(gru1_merged)
        gru_2b = GRU(self._rnn_size, implementation=rnn_imp, return_sequences=True, go_backwards=True, name='gru2_b')(gru1_merged)
        x = concatenate([gru_2, gru_2b])

        # x = Dropout(0.2)(x)
        x = Dense(self._n_class, activation='softmax', kernel_regularizer=l2(self._l2_rate), bias_regularizer=l2(self._l2_rate), name='output')(x)
        rnn_out = x
        base_model = Model(self.input_tensor, x)
        
        return base_model, conv_shape
    
    def crnn_pytorch_version(self, width):
        self.input_tensor = Input((self._height, width, 3))
        x = self.input_tensor
        for i, n_cnn in enumerate([2, 2, 2]):
            for j in range(n_cnn):
                x = Conv2D(32*2**i, (3, 3), padding='same', kernel_initializer='he_uniform',
                           kernel_regularizer=l2(self._l2_rate))(x)
                x = BatchNormalization(gamma_regularizer=l2(self._l2_rate), beta_regularizer=l2(self._l2_rate))(x)
                x = Activation('relu')(x)
            x = MaxPooling2D((2, 2))(x)
        
        for i, n_cnn in enumerate([2, 2]):
            for j in range(n_cnn):
                x = Conv2D(32*2**(i+2), (3, 3), padding='same', kernel_initializer='he_uniform',
                           kernel_regularizer=l2(self._l2_rate))(x)
                x = BatchNormalization(gamma_regularizer=l2(self._l2_rate), beta_regularizer=l2(self._l2_rate))(x)
                x = Activation('relu')(x)
            x = MaxPooling2D((2, 1))(x)

        # x = AveragePooling2D((1, 2))(x)
        cnn_model = Model(self.input_tensor, x, name='cnn')

        x = cnn_model(self.input_tensor)

        conv_shape = x.get_shape().as_list()
        #rnn_length = conv_shape[1]
        #rnn_dimen = conv_shape[3]*conv_shape[2]

        #print (conv_shape, rnn_length, rnn_dimen)
        # x = Reshape(target_shape=(rnn_length, rnn_dimen))(x)
        x = Permute((2,1,3),name='permute')(x)
        x = TimeDistributed(Flatten(), name='flatten')(x)
        rnn_imp = 0

        x = Dense(self._rnn_size, kernel_initializer='he_uniform', kernel_regularizer=l2(self._l2_rate), bias_regularizer=l2(self._l2_rate))(x)
        x = BatchNormalization(gamma_regularizer=l2(self._l2_rate), beta_regularizer=l2(self._l2_rate))(x)
        x = Activation('relu')(x)
        # x = Dropout(0.2)(x)

        gru_1 = GRU(self._rnn_size, implementation=rnn_imp, return_sequences=True, name='gru1')(x)
        gru_1b = GRU(self._rnn_size, implementation=rnn_imp, return_sequences=True, go_backwards=True, name='gru1_b')(x)
        gru1_merged = add([gru_1, gru_1b])

        gru_2 = GRU(self._rnn_size, implementation=rnn_imp, return_sequences=True, name='gru2')(gru1_merged)
        gru_2b = GRU(self._rnn_size, implementation=rnn_imp, return_sequences=True, go_backwards=True, name='gru2_b')(gru1_merged)
        x = concatenate([gru_2, gru_2b])

        # x = Dropout(0.2)(x)
        x = Dense(self._n_class, activation='softmax', kernel_regularizer=l2(self._l2_rate), bias_regularizer=l2(self._l2_rate), name='output')(x)
        rnn_out = x
        base_model = Model(self.input_tensor, x)
        
        return base_model, conv_shape
    
    def model_cnn(self, width):
        
        self.input_tensor = Input((self._height, width, 3))
        x = self.input_tensor
        for i, n_cnn in enumerate([3, 4, 6]):
            for j in range(n_cnn):
                x = Conv2D(32*2**i, (3, 3), padding='same', kernel_initializer='he_uniform', 
                           kernel_regularizer=l2(self._l2_rate))(x)
                x = BatchNormalization(gamma_regularizer=l2(self._l2_rate), beta_regularizer=l2(self._l2_rate))(x)
                x = Activation('relu')(x)
            x = MaxPooling2D((2, 2))(x)

        cnn_model = Model(self.input_tensor, x, name='cnn')

        x = cnn_model(self.input_tensor)

        conv_shape = x.get_shape().as_list()
        #rnn_length = conv_shape[1]
        #rnn_dimen = conv_shape[3]*conv_shape[2]

        #print (conv_shape, rnn_length, rnn_dimen)
        # x = Permute((2, 1, 3), name='permute')(x)
        x = Permute((2,1,3),name='permute')(x)
        x = TimeDistributed(Flatten(), name='flatten')(x)

        # x = Dropout(0.2)(x)
        x = Dense(self._n_class, activation='softmax', kernel_regularizer=l2(self._l2_rate), bias_regularizer=l2(self._l2_rate), name='output_cnn')(x)
        rnn_out = x
        base_model = Model(self.input_tensor, x)
        
        return base_model, conv_shape
    
    def model_vgg(self, width):

        self.input_tensor = Input((self._height, width, 3))

        vgg_model = VGG16(weights=None,input_tensor=self.input_tensor)
        # vgg_model = VGG19(weights=None,input_tensor=self.input_tensor)

        cnn_model = Model(input=vgg_model.input, output=vgg_model.get_layer('block3_pool').output, name='cnn')

        x = cnn_model(self.input_tensor)

        conv_shape = x.get_shape().as_list()
        #rnn_length = conv_shape[1]
        #rnn_dimen = conv_shape[3]*conv_shape[2]

        #print (conv_shape, rnn_length, rnn_dimen)
        # x = Reshape(target_shape=(rnn_length, rnn_dimen))(x)
        x = Permute((2,1,3),name='permute')(x)
        x = TimeDistributed(Flatten(), name='flatten')(x)
        rnn_imp = 0

        x = Dense(self._rnn_size, kernel_initializer='he_uniform', kernel_regularizer=l2(self._l2_rate), bias_regularizer=l2(self._l2_rate))(x)
        x = BatchNormalization(gamma_regularizer=l2(self._l2_rate), beta_regularizer=l2(self._l2_rate))(x)
        x = Activation('relu')(x)
        # x = Dropout(0.2)(x)

        gru_1 = GRU(self._rnn_size, implementation=rnn_imp, return_sequences=True, name='gru1')(x)
        gru_1b = GRU(self._rnn_size, implementation=rnn_imp, return_sequences=True, go_backwards=True, name='gru1_b')(x)
        gru1_merged = add([gru_1, gru_1b])

        gru_2 = GRU(self._rnn_size, implementation=rnn_imp, return_sequences=True, name='gru2')(gru1_merged)
        gru_2b = GRU(self._rnn_size, implementation=rnn_imp, return_sequences=True, go_backwards=True, name='gru2_b')(gru1_merged)
        x = concatenate([gru_2, gru_2b])

        # x = Dropout(0.2)(x)
        x = Dense(self._n_class, activation='softmax', kernel_regularizer=l2(self._l2_rate), bias_regularizer=l2(self._l2_rate), name='output_vgg')(x)
        rnn_out = x
        base_model = Model(self.input_tensor, x)

        return base_model, conv_shape
    
    def model_Xception(self, width):

        self.input_tensor = Input((self._height, width, 3))

        vgg_model = Xception(weights=None,input_tensor=self.input_tensor)
        # vgg_model = VGG19(weights=None,input_tensor=self.input_tensor)

        cnn_model = Model(input=vgg_model.input, output=vgg_model.get_layer('block3_pool').output, name='cnn')

        x = cnn_model(self.input_tensor)

        conv_shape = x.get_shape().as_list()
        #rnn_length = conv_shape[1]
        #rnn_dimen = conv_shape[3]*conv_shape[2]

        #print (conv_shape, rnn_length, rnn_dimen)
        # x = Reshape(target_shape=(rnn_length, rnn_dimen))(x)
        x = Permute((2,1,3),name='permute')(x)
        x = TimeDistributed(Flatten(), name='flatten')(x)
        rnn_imp = 0
        '''
        x = Dense(self._rnn_size, kernel_initializer='he_uniform', kernel_regularizer=l2(self._l2_rate), bias_regularizer=l2(self._l2_rate))(x)
        x = BatchNormalization(gamma_regularizer=l2(self._l2_rate), beta_regularizer=l2(self._l2_rate))(x)
        x = Activation('relu')(x)
        # x = Dropout(0.2)(x)

        gru_1 = GRU(self._rnn_size, implementation=rnn_imp, return_sequences=True, name='gru1')(x)
        gru_1b = GRU(self._rnn_size, implementation=rnn_imp, return_sequences=True, go_backwards=True, name='gru1_b')(x)
        gru1_merged = add([gru_1, gru_1b])

        gru_2 = GRU(self._rnn_size, implementation=rnn_imp, return_sequences=True, name='gru2')(gru1_merged)
        gru_2b = GRU(self._rnn_size, implementation=rnn_imp, return_sequences=True, go_backwards=True, name='gru2_b')(gru1_merged)
        x = concatenate([gru_2, gru_2b])
        '''
        # x = Dropout(0.2)(x)
        x = Dense(self._n_class, activation='softmax', kernel_regularizer=l2(self._l2_rate), bias_regularizer=l2(self._l2_rate), name='output_Xception')(x)
        rnn_out = x
        base_model = Model(self.input_tensor, x)

        return base_model, conv_shape



    def ctc_loss(self, args):
        y_pred, labels, input_length, label_length = args
        y_pred = y_pred[:, 2:, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
        
    
    def train_model(self, gen_train, gen_val, train_epochs=2, batch_size=64, learning_rate=1e-4, train_num=5000000, val_num=100000, num_workers = 12, finetune=False, model_path='model.h5'):
        
        # self.model_init()
        self.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(learning_rate))
        
        if finetune:
            self.load_model(model_path)
             
        h = self.model.fit_generator(gen_train, pickle_safe=True, workers=num_workers, 
                                #validation_data=([X_test, y_test, np.ones(n_test)*int(conv_shape[1]-2), label_length_test], np.ones(n_test)), 
                                validation_data=gen_val, validation_steps=int(val_num/batch_size),
                                steps_per_epoch=int(train_num/batch_size), epochs=train_epochs) 
        
        self.save_model(model_path)
    

    def test_model(self, gen, characters, n_len=200, model_path='model.h5'):
        
        # self.model_init()
        self.load_model(model_path)
        [X_test, y_test, _, _], _  = next(gen)
        y_pred = self.base_model.predict(X_test)
        shape = y_pred[:,2:,:].shape
        out = K.get_value(K.ctc_decode(y_pred[:,2:,:], input_length=np.ones(shape[0])*shape[1])[0][0])[:, :n_len]

        for i in range(16):
            y = ''.join([characters[x] for x in y_test[i] if x > -1])
            s = ''.join([characters[x] for x in out[i] if x > -1])
            # print (y_test)
            print ('gt>>>>>>>', y)
            print ('pred>>>>>', s)

    
    def load_model(self, model_path='model.h5'):
        
        self.model.load_weights(model_path, by_name=True)
    
    def save_model(self, model_path='model.h5'):
        
        self.model.save_weights(model_path)
        
