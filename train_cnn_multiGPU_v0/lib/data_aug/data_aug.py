# coding=utf-8

import numpy as np
import cv2
from skimage import exposure

class DataAugmenters():

    def __init__(self, image):
        self.img = image
    
    def _random_fliplr(self, random_fliplr=True):
        if random_fliplr and np.random.choice([True, False]):
            self.img = np.fliplr(self.img)
    
    def _random_flipud(self, random_flipud=True):
        if random_flipud and np.random.choice([True, False]):
            self.img = np.fliplr(self.img)
    
    def _random_rotation(self, random_rotation=True):
        if random_rotation and np.random.choice([True, False]):
            w,h = self.img.shape[1], self.img.shape[0]
            angle = np.random.randint(0,360)
            rotate_matrix = cv2.getRotationMatrix2D(center=(self.img.shape[1]/2, self.img.shape[0]/2), angle=angle, scale=0.7)
            self.img = cv2.warpAffine(self.img, rotate_matrix, (w,h))

    def _random_exposure(self, random_exposure=True):
        if random_exposure and np.random.choice([True, False]):
            e_rate = np.random.uniform(0.5,1.5)
            self.img = exposure.adjust_gamma(self.img, e_rate)
    
    # 裁剪
    def _random_crop(self, crop_size = 299, random_crop = True):
        if random_crop and np.random.choice([True, False]):
            if self.img.shape[1] > crop_size:
                sz1 = self.img.shape[1] // 2
                sz2 = crop_size // 2
                diff = sz1 - sz2
                (h, v) = (np.random.randint(0, diff + 1), np.random.randint(0, diff + 1))
                self.img = self.img[v:(v + crop_size), h:(h + crop_size), :]
    
    def run(self):
        data_aug_list = [self._random_fliplr, self._random_flipud, self._random_rotation, self._random_exposure, self._random_crop]
        data_aug_func = np.random.choice(data_aug_list, 2)
        for func in data_aug_func:
            func()
        return self.img
