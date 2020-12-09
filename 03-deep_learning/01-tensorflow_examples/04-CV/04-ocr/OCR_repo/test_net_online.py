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
import config 
from ocr_lib.crnn import CRNN as crnn
from ocr_lib.gen_train_val_data import gen_train_val_data



alphabet = config.alphabet
width = config.width
height = config.height
train_sample_width = config.train_sample_width
pooling_num = config.pooling_num
n_len =  config.n_len
n_class = config.n_class

train_epochs = config.train_epochs
batch_size = config.batch_size
learning_rate = config.learning_rate
train_num = config.train_num
val_num = config.val_num
num_workers = config.num_workers
finetune = config.finetune
model_path = config.model_path
rnn_size = config.rnn_size
l2_rate = config.l2_rate


crnn_model = crnn(width=width, height=height, n_len=n_len, characters_length=n_class, rnn_size=rnn_size, l2_rate=l2_rate)

train_val_data = gen_train_val_data(width=train_sample_width, height=height, n_len=n_len, pooling_num=pooling_num)

#crnn_model.train_model(gen_train=train_val_data.gen_train(batch_size=batch_size, num_workers=num_workers, alphabet=alphabet), gen_val=train_val_data.gen_val(batch_size=batch_size, num_workers=num_workers, alphabet=alphabet), 
#                       train_epochs=2, batch_size=batch_size, learning_rate=1e-4, train_num=5000000, val_num=100000, num_workers = 12, finetune=False, model_path='model.h5')

crnn_model.test_model(gen=train_val_data.gen_val(batch_size=16, num_workers=num_workers, alphabet=alphabet), characters=alphabet)
