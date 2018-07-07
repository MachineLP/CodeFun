#coding=utf-8

from lib.utils.utils import shuffle_train_data

class data_load_from_txt_nolabel(object):

    def __init__(self, img_dir, train_rate):
        self.img_dir = img_dir
        self.train_imgs = []
        self.train_rate = train_rate
    
    def _gen_img_path(self):
        data_lines = open(self.img_dir, 'r').readlines() 
        for line in data_lines:
            img_path = line.split(' ')[0]
            img_path = img_path.split('\n')[0]
            self.train_imgs.append(img_path)
    
    def gen_train_valid(self):
        self._gen_img_path()
        image_n = len(self.train_imgs)
        train_n = int(image_n*self.train_rate)
        valid_n = int(image_n*(1-self.train_rate))
        train_data = self.train_imgs[0:train_n]
        valid_data = self.train_imgs[train_n:image_n]
        return train_data, valid_data, train_n, valid_n

# 以下测试用
if __name__ == '__main__':
    data = data_load_from_txt_nolabel('train.txt', 0.9)
    train_data, valid_data, train_n, valid_n = data.gen_train_valid()
    print (train_data)
    print (valid_data)