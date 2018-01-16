# -*- coding: utf-8 -*-
"""
Created on 2017 10.17
@author: liupeng
wechat: lp9628
blog: http://blog.csdn.net/u014365862/article/details/78422372
"""

import numpy as np  
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import cv2
try:
    from data_aug import random_flip, random_exposure, random_rotation, random_crop
except:
    from data_aug.data_aug import random_flip, random_exposure, random_rotation, random_crop

# 适用于二级目录 。。。/图片类别文件/图片（.png ,jpg等）

def load_img_path(imgDir,imgFoldName, img_label):
    imgs = os.listdir(imgDir+imgFoldName)
    imgNum = len(imgs)
    data = []
    label = []
    for i in range (imgNum):
        img_path = imgDir+imgFoldName+"/"+imgs[i]
        # 用来检测图片是否有效，放在这里会太费时间。
        # img = cv2.imread(img_path)
        # if img is not None:
        data.append(img_path)
        label.append(int(img_label))
    return data,label

def shuffle_train_data(train_imgs, train_labels):
    index = [i for i in range(len(train_imgs))]
    np.random.shuffle(index)
    train_imgs = np.asarray(train_imgs)
    train_labels = np.asarray(train_labels)
    train_imgs = train_imgs[index]
    train_labels = train_labels[index]
    return train_imgs, train_labels

def load_database_path(imgDir):
    img_path = os.listdir(imgDir)
    train_imgs = []
    train_labels = []
    for i, path in enumerate(img_path):
        craterDir = imgDir + '/'
        foldName = path
        data, label = load_img_path(craterDir,foldName, i)
        train_imgs.extend(data)
        train_labels.extend(label)
        print ("文件名对应的label:")
        print (path, i)
    #打乱数据集
    train_imgs, train_labels = shuffle_train_data(train_imgs, train_labels)
    return train_imgs, train_labels




def get_next_batch_from_path(image_path, image_labels, pointer, IMAGE_HEIGHT=299, IMAGE_WIDTH=299, batch_size=64, is_train=True):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT,IMAGE_WIDTH,3])
    num_classes = len(image_labels[0])
    batch_y = np.zeros([batch_size, num_classes]) 
    for i in range(batch_size):  
        image = cv2.imread(image_path[i+pointer*batch_size])
        image = cv2.resize(image, ((int(IMAGE_HEIGHT*1.5), int(IMAGE_WIDTH*1.5)))  
        if is_train:
            image = random_flip(image)
            image = random_rotation(image)
            image = random_crop(image)
            image = random_exposure(image)
        image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))  
        # 选择自己预处理方式：
        '''
        m = image.mean()
        s = image.std()
        min_s = 1.0/(np.sqrt(image.shape[0]*image.shape[1]))
        std = max(min_s, s)
        image = (image-m)/std'''
        # image = (image-127.5)
        image = image / 255.0
        image = image - 0.5
        image = image * 2
        
        batch_x[i,:,:,:] = image
        # print labels[i+pointer*batch_size]
        batch_y[i] = image_labels[i+pointer*batch_size]
    return batch_x, batch_y


def test():

    craterDir = "train"
    data, label = load_database(craterDir)
    print (data.shape)
    print (len(data))
    print (data[0].shape)
    print (label[0])
    batch_x, batch_y = get_next_batch_from_path(data, label, 0, IMAGE_HEIGHT=299, IMAGE_WIDTH=299, batch_size=64, is_train=True)
    print (batch_x)
    print (batch_y)

if __name__ == '__main__':
    test()

