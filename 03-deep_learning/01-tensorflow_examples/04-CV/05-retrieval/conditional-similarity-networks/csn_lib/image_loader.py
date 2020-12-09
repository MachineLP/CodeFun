# -*- coding: utf-8 -*-
"""
    Created on 2018 06.18
    @author: liupeng
    wechat: lp9628
    blog: http://blog.csdn.net/u014365862/article/details/78422372
    """
from PIL import Image
import os
import os.path
import numpy as np

filenames = {'train': ['class_tripletlist_train.txt', 'closure_tripletlist_train.txt', 
                'gender_tripletlist_train.txt', 'heel_tripletlist_train.txt'],
             'val': ['class_tripletlist_val.txt', 'closure_tripletlist_val.txt', 
                'gender_tripletlist_val.txt', 'heel_tripletlist_val.txt'],
             'test': ['class_tripletlist_test.txt', 'closure_tripletlist_test.txt', 
                'gender_tripletlist_test.txt', 'heel_tripletlist_test.txt']}

class TripletImageLoader(object):
    def __init__(self, root, base_path, filenames_filename, conditions, split, n_triplets):
        self.root = root
        self.base_path = base_path  
        self.filenamelist = []
        for line in open(os.path.join(self.root, filenames_filename)):
            self.filenamelist.append(line.rstrip('\n'))
        triplets = []
        if split == 'train':
            fnames = filenames['train']
        elif split == 'val':
            fnames = filenames['val']
        else:
            fnames = filenames['test']
        for condition in conditions:
            for line in open(os.path.join(self.root, 'tripletlists', fnames[condition])):
                triplets.append((line.split()[0], line.split()[1], line.split()[2], condition)) # anchor, far, close   
        # print(triplets[:100])   
        np.random.shuffle(triplets)
        # print(triplets[:100])  
        self.triplets = triplets[:int(n_triplets * 1.0 * len(conditions) / 4)]
        self.data_trip = []

    def get_trip_data(self):
        for path1, path2, path3, c in self.triplets:
            if os.path.exists(os.path.join(self.root, self.base_path, self.filenamelist[int(path1)])) and os.path.exists(os.path.join(self.root, self.base_path, self.filenamelist[int(path1)])) and os.path.exists(os.path.join(self.root, self.base_path, self.filenamelist[int(path1)])):
                self.data_trip.append([os.path.join(self.root, self.base_path, self.filenamelist[int(path1)]),os.path.join(self.root, self.base_path, self.filenamelist[int(path2)]),os.path.join(self.root, self.base_path, self.filenamelist[int(path3)]), c])
        return self.data_trip

    
# 以下测试用
if __name__ == '__main__':
    conditions = [0,1,2,3]
    train_data_trip = TripletImageLoader('data', 'ut-zap50k-images', 'filenames.json', conditions, 'train', n_triplets=100000).get_trip_data()
    print (train_data_trip[:100])
    val_data_trip = TripletImageLoader('data', 'ut-zap50k-images', 'filenames.json', conditions, 'val', n_triplets=80000).get_trip_data()
    print (val_data_trip[:100])
    test_data_trip = TripletImageLoader('data', 'ut-zap50k-images', 'filenames.json', conditions, 'test', n_triplets=160000).get_trip_data()
    print (test_data_trip[:100])


