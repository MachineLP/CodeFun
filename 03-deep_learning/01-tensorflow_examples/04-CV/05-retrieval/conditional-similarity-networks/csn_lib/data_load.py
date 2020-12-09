# -*- coding: utf-8 -*-
"""
Created on 2017 10.17
@author: liupeng
"""

import sys
import tensorflow as tf
import numpy as np
import os
import cv2
from skimage import exposure
# from lib.utils.utils import shuffle_train_data

class load_image(object):

    def __init__(self, img_dir, train_rate):
        self.img_dir = img_dir
        self.train_rate = train_rate
        self.train_imgs = []
        self.train_labels = []
        self.note_label = []

    def _load_img_path(self, img_sub_dir, img_label):
        img_all_path = os.listdir(os.path.join(self.img_dir, img_sub_dir))
        img_num = len(img_all_path)
        data = []
        label = []
        for i in range (img_num):
            img_path = os.path.join(self.img_dir, img_sub_dir, img_all_path[i])
            # if cv2.imread(img_path) is not None:
            data.append(img_path)
            # print (img_path)
            label.append(int(img_label))
        return data, label

    def _load_database_path(self):
        file_path = os.listdir(self.img_dir)
        for i, path in enumerate(file_path):
            if os.path.isfile(os.path.join(self.img_dir, path)):
                continue
            data, label = self._load_img_path(path, i)
            #self.train_imgs.extend(data)
            self.train_labels.append(label)
            print (path, i)
            self.train_imgs.append(data)
            #self.train_labels.append(label)
            self.note_label.append([path, i])
        # self.train_imgs, self.train_labels = shuffle_train_data(self.train_imgs, self.train_labels)

    def gen_train_valid(self):
        self._load_database_path()
        image_n = len(self.train_labels)
        train_n = int(image_n*self.train_rate)
        valid_n = int(image_n*(1-self.train_rate))
        train_data, train_label = self.train_imgs[0:train_n], self.train_labels[0:train_n]
        valid_data, valid_label = self.train_imgs[train_n:image_n], self.train_labels[train_n:image_n]
        return train_data, train_label, valid_data, valid_label, train_n, valid_n, self.note_label
