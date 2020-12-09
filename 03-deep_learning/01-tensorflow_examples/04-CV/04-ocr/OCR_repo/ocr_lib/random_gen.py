# -*- coding: utf-8 -*-
"""
Created on 2018 08.20
@author: liupeng
wechat: lp9628
blog: http://blog.csdn.net/u014365862/article/details/78422372
github：https://github.com/MachineLP
"""

import os
from PIL import Image, ImageFont, ImageDraw
import random
import numpy as np
import cv2

# 生成器

'''
用于OCR训练的图片生成
'''

from ocr_lib.data_load import load_image
# 读取的图片不要太大哦， 太费时间， 图片太大的话可以先预处理一下。
train_data, _,_,_,_,_,_ = load_image(img_dir='gender/', train_rate=1.0).gen_train_valid()
def random_gen_bg_img(img_dir='gender/', train_rate=1.0):
    img_bg = random.randint( 0, len(train_data)-1 )
    return train_data[img_bg]

def randmo_crop(img):
    crop_bg = random.randint( 4, 10)
    img.crop([img.size[0]/crop_bg,img.size[1]/crop_bg,img.size[0]*2/crop_bg,img.size[1]*2/crop_bg])
    return img

# 随机生成每一个文本的长度
def random_gen_per_text_length():
    text_length_list = [ 10, 11, 12 ]
    text_length = random.randint( 0, len(text_length_list)-1 )
    return text_length_list[text_length]

# 随机生成背景颜色
def randon_gen_bg_color():
    # 图片背景
    '''
    R = random.randrange(0,255,15)
    G = random.randrange(0,255,15)
    B = random.randrange(0,255,15)
    img_color_list = [ (R, G, B) ]'''
    img_color_list = [ (0,0,0), (255,255,255)] #, (255,0,0), (0,255,0), (0,0,255), (0,255,255), (255,0,255), (255,255,0),  (128,128,128), (128,0,0), (0,128,0), (0,0,128), (0,128,128), (128,0,128), (128,128,0) ]
    img_bg_color = random.randint( 0, len(img_color_list)-1 )
    return img_color_list[img_bg_color]

# 随机生成文本颜色
def random_gen_text_color():
    # 文字颜色
    '''
    R = random.randrange(0,255,15)
    G = random.randrange(0,255,15)
    B = random.randrange(0,255,15)
    text_color_list = [ (R, G, B) ]'''
    text_color_list = [ (0,0,0), (255,255,255)] #, (255,0,0), (0,255,0), (0,0,255), (0,255,255), (255,0,255), (255,255,0),  (128,128,128), (128,0,0), (0,128,0), (0,0,128), (0,128,128), (128,0,128), (128,128,0) ]
    text_color = random.randint( 0, len(text_color_list)-1 )
    return text_color_list[text_color]

def random_gen_font():
    font_list = [  'yahei_mono.ttf', 'simsun.ttc', 'simkai.ttf', 'simhei.ttf', 'simfang.ttf', 'msyhbd.ttf', 'msyh.ttf', 'huawenxihei.ttf', 'DroidSansFallback.ttf', '仿宋_GB2312.ttf', '楷体_GB2312.ttf', 'bb1550.ttf', '张海山锐线体2.0.TTF', '张海山锐谐体2.0.TTF', 'MISSYUAN-叶根友毛笔行书简体_0.TTF']
    font = random.randint( 0, len(font_list)-1 )
    return font_list[font]

