#!/usr/bin/env python3
# coding: utf-8
 
# 数据链接:https://pan.baidu.com/s/1v-7aaAHWsx7NZ5d3IdWbiQ  密码:k5tx 

class DataLoader:
    def __init__(self):
        self.datafile = 'data/data.txt'
        self.dataset = self.load_data()
 
    '''加载数据集'''
    def load_data(self):
        dataset = []
        for line in open(self.datafile):
            line = line.strip().split(',')
            dataset.append([word for word in line[1].split(' ') if 'nbsp' not in word and len(word) < 11])
        return dataset


