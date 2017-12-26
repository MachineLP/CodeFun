# -*- coding: utf-8 -*-
"""
Created on 2017 10.17
@author: liupeng
wechat: lp9628
blog: http://blog.csdn.net/u014365862/article/details/78422372
"""

import numpy as np  
import numpy as np
import os
from PIL import Image
import cv2
#from predict_cnn import *
#from predict import *
from predict import *

import csv
import argparse, json, textwrap
import sys
alg_core = TEAlg(pb_path_1="model/frozen_model.pb")

File = open("output.csv", "w")


def load_img(imgDir,imgFoldName, img_label):
    imgs = os.listdir(imgDir+imgFoldName)
    imgNum = len(imgs)
    data = []#np.empty((imgNum,224,224,3),dtype="float32")
    label = []#np.empty((imgNum,),dtype="uint8")
    for i in range (imgNum):
        image_path = imgDir+imgFoldName+"/"+imgs[i]
        img = cv2.imread(image_path)
        #for j in range(1):
        if img is not None:
            result_dict = ProjectInterface({image_path: image_path}, proxy=alg_core)
            print (result_dict)
            # print(result_dict.keys(), result_dict.values())
            #print ('Img_dir:{},label:{},prediction:{}'.format(result_dict.keys(), img_label, result_dict.values()))
            File.write(str(result_dict) + "\n")  
    return data,label
'''
craterDir = "train/"
foldName = "0male"
data, label = load_Img(craterDir,foldName, 0)

print (data[0].shape)
print (label[0])'''


def load_database(imgDir):
    img_path = os.listdir(imgDir)
    train_imgs = []
    train_labels = []
    for i, path in enumerate(img_path):
        craterDir = imgDir + '/'
        foldName = path
        data, label = load_img(craterDir,foldName, i)
        train_imgs.extend(data)
        train_labels.extend(label)
    #打乱数据集
    index = [i for i in range(len(train_imgs))]    
    np.random.shuffle(index)   
    train_imgs = np.asarray(train_imgs)
    train_labels = np.asarray(train_labels)
    train_imgs = train_imgs[index]  
    train_labels = train_labels[index] 
    return train_imgs, train_labels


def test():
    # train
    craterDir = "sample_test"
    global dir_path
    dir_path = "train_crop/"
    #dir_path = "train/"
    data, label = load_database(craterDir)
    #dir = "/1female"
    #data, label = load_img(craterDir,dir,0)
    File.close()
    print (data.shape)
    print (len(data))
    print (data[0].shape)
    print (label[0])


if __name__ == '__main__':
    test()
