# -*- coding: utf-8 -*-
"""
    Created on 2017 10.17
    @author: liupeng
    """

import sys
# import tensorflow as tf
import numpy as np
import os
import cv2
from skimage import exposure
from data_load import load_image

# 图片在二级目录
file_path = 'gender'
img_path, _, _, _, _, _, _ = load_image(file_path, 1.0).gen_train_valid()
print (img_path)


# 遍历所有的图片
dir = 'image'
num = 0
for path in img_path:
    print (path)
    # 对于每一张图片
    img = cv2.imread(path)
    if img is None:
        continue
    num = num + 1
    img_path = dir + str(num) + '.jpg'
    # 从现有的图片路径创建新的图片路径
    img_sub_dir = path.split('/')
    if not os.path.isdir(img_sub_dir[-3]+'_rename'):
        os.makedirs(img_sub_dir[-3]+'_rename')
    if not os.path.isdir(img_sub_dir[-3]+'_rename'+'/'+img_sub_dir[-2]):
        os.makedirs(img_sub_dir[-3]+'_rename'+'/'+img_sub_dir[-2])
    # 新的图片路径
    img_new_file = img_sub_dir[-3]+'_rename'+'/'+img_sub_dir[-2] + '/' + img_path
    cv2.imwrite(img_new_file, img)



