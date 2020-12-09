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
import string
from PIL import Image, ImageFont, ImageDraw
from skimage import exposure
import random

from ocr_lib.random_gen  import randmo_crop, random_gen_bg_img, random_gen_per_text_length, randon_gen_bg_color, random_gen_text_color, random_gen_font


class data_generator(object):
    
    def __init__(self):
        pass
    

    def generate(self, alphabet='', n_len=10):
        text_length = random_gen_per_text_length()
        alphabet_length = len(alphabet)
        text = ''
        t_id = []
        ran_alphabet_flag = np.random.choice([500, alphabet_length-1])
        for i in range(text_length):
            # 随机选择汉字表中的文字
            # ran_alphabet_flag = np.random.choice([50, 100, 200, 500, alphabet_length-1])
            # random_id = random.randint( 0, alphabet_length-1 )
            random_id = random.randint( 0, ran_alphabet_flag )
            text = text + str( alphabet[random_id] )
            t_id.append(int(random_id))
    
        # 根据随机的染色
        text_length = len(text)
        # print ('>>>>>>>>>>>>>>>', text_length)
        img_color_random = randon_gen_bg_color()

        im = Image.new("RGB", (10, 10), img_color_random)    
        # im = Image.new("RGB", (img_size_width, 60 + randon_img_height), img_color_random)
        # 
        dr = ImageDraw.Draw(im)  
        
        # 字体
        text_font = random_gen_font()
        # 随机的调整子体的大小
        randon_font_size = random.randint( -2, 2 )
        font = ImageFont.truetype(os.path.join("ocr_lib/fonts", text_font), 50 + randon_font_size)

        ### >>>>>>>>>>>>>>>>>>>>>>>>>>        
        text_shape = dr.textsize(text, font=font)
        # 宽高
        # print ('text_shape>>>>>>', text_shape)


        # 根据自己的多少 和 字体的大小 给定图片的大小， 计算公式： img_size_width = 字体大小 * 数量 + 40 （调试的结果）
        randon_img_width = random.randint( 0, 5 )
        randon_img_height = random.randint( 0, 2 )
        # 随机的调整宽高, 以最大长度作为宽
        img_size_width = 50 * n_len + 40 + randon_img_width
        flag = np.random.choice([False]) # True
        if flag:
            img_name = random_gen_bg_img()
            im = Image.open(img_name)
            im = randmo_crop(im)
            im = im.resize((text_shape[0]+randon_img_width, text_shape[1]+randon_img_height))
        else:
            im = Image.new("RGB", (text_shape[0]+randon_img_width, text_shape[1]+randon_img_height), img_color_random)
        # 
        dr = ImageDraw.Draw(im)

        
        # 字体颜色 
        fillColor = [0,0,0]
        while (1):
            fillColor = random_gen_text_color()
            # 必然背景 与 文字颜色相同
            temp = abs(fillColor[0] - img_color_random[0]) + abs(fillColor[1] - img_color_random[1])+ abs(fillColor[2] - img_color_random[2])
            if temp > 50:
                break
        randon_font_x = random.randint( 0, 5 )
        randon_font_y = random.randint( 0, 2 )
        dr.text((randon_font_x, randon_font_y), text, font=font, fill=fillColor)
        # im.save('lp.jpg')
        return np.array(im, dtype=np.uint8), t_id
