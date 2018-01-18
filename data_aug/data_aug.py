
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
from skimage import exposure


# 完成图像的左右镜像
def random_flip(image, random_flip=True):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image) # 左右
    if random_flip and np.random.choice([True, False]):
        image = np.flipud(image) # 上下
    return image

# 改变光照
# 光照调节也可以用log, 参数调节和gamma相反；
# img = exposure.adjust_log(img, 1.3)
def random_exposure(image, random_exposure=True):
    if random_exposure and np.random.choice([True, False]):
        image = exposure.adjust_gamma(image, 1.1) # 调暗
    if random_exposure and np.random.choice([True, False]):
        image = exposure.adjust_gamma(image, 1.3) # 调暗
    if random_exposure and np.random.choice([True, False]):
        image = exposure.adjust_gamma(image, 1.5) # 调暗
    if random_exposure and np.random.choice([True, False]):
        image = exposure.adjust_gamma(image, 0.9) # 调亮
    if random_exposure and np.random.choice([True, False]):
        image = exposure.adjust_gamma(image, 0.8) # 调亮
    if random_exposure and np.random.choice([True, False]):
        image = exposure.adjust_gamma(image, 0.7) # 调亮
    if random_exposure and np.random.choice([True, False]):
        image = exposure.adjust_gamma(image, 0.5) # 调亮
    return image

def random_rotation(image, random_rotation=True):
    if random_rotation and np.random.choice([True, False]):
        w,h = image.shape[1], image.shape[0]
        # 0-180随机产生旋转角度。
        angle = np.random.randint(0,10)
        RotateMatrix = cv2.getRotationMatrix2D(center=(image.shape[1]/2, image.shape[0]/2), angle=angle, scale=0.7)
        # image = cv2.warpAffine(image, RotateMatrix, (w,h), borderValue=(129,137,130))
        image = cv2.warpAffine(image, RotateMatrix, (w,h), borderMode=cv2.BORDER_REPLICATE)
    return image

def random_crop(image, crop_size=299, random_crop=True):
    if random_crop and np.random.choice([True, False]):
        if image.shape[1] > crop_size:
            sz1 = image.shape[1] // 2
            sz2 = crop_size // 2
            diff = sz1 - sz2
            (h, v) = (np.random.randint(0, diff + 1), np.random.randint(0, diff + 1))
            image = image[v:(v + crop_size), h:(h + crop_size), :]

    return image
