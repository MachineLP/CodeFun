#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import misc
import tensorflow as tf
from threading import Lock
import os
import cv2
import sys

model_class = 1

def GPU_config(rate=0.99):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpuConfig = tf.ConfigProto()
    gpuConfig.allow_soft_placement = False
    gpuConfig.gpu_options.allow_growth = True
    gpuConfig.gpu_options.per_process_gpu_memory_fraction = rate

    return gpuConfig

def prewhiten(self, img):
    mean = np.mean(img)
    std = np.std(img)
    std_adj = np.maximum(std, 1.0/np.sqrt(img.size))
    ret = np.multiply(np.subtract(img, mean), 1/std_adj)
    return ret
def to_rgb(self,img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret
def img_crop(img, box):
    # y1, x1, y2, x2 = box[1]-20, box[0]-20, box[1]+box[3]+40, box[0]+box[2]+40
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    img = img[y1:y2, x1:x2]
    return img
def data_norm(img):
    img = img / 255.0
    img = img - 0.5
    img = img * 2
    return img
def add_img_padding(img):
    h, w, _ = img.shape
    width = np.max([h, w])
    # 按照长宽中大的初始化一个正方形
    img_padding = np.zeros([width, width,3])
    # 找出padding的中间位置
    h1 = int(width/2-h/2)
    h2 = int(width/2+h/2)
    w1 = int(width/2-w/2)
    w2 = int(width/2+w/2)
    # 进行padding， img为什么也要进行扣取？ 原因在于除以2取整会造成尺寸不一致的情况。
    img_padding[h1:h2, w1:w2, :] = img[0:(h2-h1),0:(w2-w1),:]
    return img_padding

# config=GPU_config()
class LPAlg(object):

    # default model path of .pb
    PB_PATH_1 = os.path.join(os.getcwd(), "model", "body_pose_model.pb")
    PB_PATH = [PB_PATH_1]

    CLASS_NUMBER = model_class

    def __init__(self, pb_path_1=None, gpu_config=GPU_config()):
        def get_path(path,default_path):
            return (path, default_path)[path is None]

        def load_graph(frozen_graph_filename):
            # We load the protobuf file from the disk and parse it to retrieve the
            # unserialized graph_def
            with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            # Then, we can use again a convenient built-in function to import a graph_def into the
            # current default Graph
            with tf.Graph().as_default() as graph:
                tf.import_graph_def(
                    graph_def,
                    input_map=None,
                    return_elements=None,
                    name="prefix",
                    op_dict=None,
                    producer_op_list=None
                )
            return graph

        # model
        def sess_def(pb_path):
            print (pb_path)
            graph = load_graph(pb_path)
            pred = graph.get_tensor_by_name('prefix/predictions:0')
            batch_size = tf.placeholder(tf.float32, [None, 1])
            label_indices = tf.placeholder(tf.float32, [None, 2])
            x1 = graph.get_tensor_by_name('prefix/inputs_placeholder:0')
            x2 = graph.get_tensor_by_name('prefix/C:0')
            # x2 = graph.get_tensor_by_name('prefix/inputs_placeholder2:0')
            # x3 = graph.get_tensor_by_name('prefix/inputs_placeholder3:0')
            sess = tf.Session(graph=graph,config=gpu_config)
            return [sess,x1,x2,batch_size,label_indices,pred]

        # multiple models
        def multi_model_def(pb_path_1):
            model_1 = sess_def(pb_path_1)
            return [model_1]

        path_1 = get_path(pb_path_1, LPAlg.PB_PATH[0])
        self._pb_path = [path_1]

        self.model = multi_model_def(self._pb_path[0])

    def _close(self):
        self.model[0][0].close()

    def _run(self, c, images_path=None):

        def img_pre_proc(filename, c):
            try:
                img = cv2.imread(filename)
                if img.ndim == 3 and img.shape[-1] == 3:
                    pass
                elif img.ndim == 2:
                    img = to_rgb(img)
                elif img.ndim == 3 and img.shape[-1] == 4:
                    img = img[:, :, :3]  # 4 channels convert to 3 channels
                else:
                    return 0, 0
                h, w, _= img.shape
                height, width = 112,112 #299, 299
                # img = add_img_padding(img)
                img = data_norm(img)
                img1 = cv2.resize(img, (height, width))
                # image2= img_crop(image, [0,int(h*0.15),w,int(h*0.5)])
                img2= img_crop(img, [0,0,w,int(h*0.5)])
                img2 = cv2.resize(img2, (height, width))
                img3 = img_crop(img, [0,int(h*0.5),w,h])
                img3 = cv2.resize(img3, (height, width))
                return img1,img2,img3, 1
            except:
                print ('img is error!')

        images_path = list(images_path)
        data_length = len(images_path)

        # no input data
        if data_length == 0:
            null_data = []
            for null_idx in range(data_length):
                null_data.append([])
            return null_data#,null_data

        # batch input processing
        imgs1 = []
        imgs2 = []
        imgs3 = []
        ncount = 0
        img_status = np.zeros(data_length)          # invalid data one-hot array: 0-valid, 1-invalid
        for i in range(data_length):

            img1,img2,img3, flag = img_pre_proc(images_path[i], c)
            #print ("image", img)
            if flag == 0:
                img_status[i] = 1
                continue
            img1 = np.array([img1])
            img2 = np.array([img2])
            img3 = np.array([img3])
            if ncount != 0:
                imgs1 = np.concatenate((imgs1, img), axis=0)
                imgs2 = np.concatenate((imgs2, img), axis=0)
                imgs3 = np.concatenate((imgs3, img), axis=0)
            else:
                imgs1 = img1
                imgs2 = img2
                imgs3 = img3
            ncount += 1

        # if all input data are invalid
        if ncount == 0:
            null_data = []
            for null_idx in range(data_length):
                null_data.append([])
            return null_data#, null_data

        label_indices_val = [LPAlg.CLASS_NUMBER for index in range(ncount)]

        idx = 0     # model index
        # print (imgs)
        predict = self.model[idx][0].run(
            self.model[idx][5],
            feed_dict={self.model[idx][1]: imgs1,
                       self.model[idx][2]: [[c]],
                       # self.model[idx][3]: imgs3
                                  }
                                  )
        print ('predict:', predict)

        if ncount == data_length:
            return predict.tolist()

        # merge if some of input data are invalid
        sub_idx = 0
        null_data = []
        merge_predict_1 = []

        for i in range(data_length):
            if img_status[i] == 1:
                sub_idx += 1
                merge_predict_1.append(null_data)
                continue

            index = i - sub_idx
            merge_predict_1.append(predict[index].tolist())

        return merge_predict_1

# 3->0, 0->1, 1->2, 2->3, 4->4
# note_label [['模特正面全身非特写图(9152)', 0], ['模特侧面全身非特写图(8987)', 1], ['模特背面全身非特写图(8950)', 2], ['模特正侧全身非特写图(10274)', 3], ['模特背侧全身非特写图(8098)', 4]]
keywords = [u'embedding']

def ProjectInterface(image_path_list, c, proxy=None):
    images_path = image_path_list.keys()
    predict = proxy._run(c, images_path)
    result_dict = {}
    image_number = len(images_path)
    for i in range(image_number):
        target = np.array(predict[i])
        print (target)
        top = {}
        for index, value in enumerate([target]):
            top[keywords[index]] = value
        result_dict[list(image_path_list.values())[i]] = top
    return result_dict


if __name__ == "__main__":
    # python predict.py 'lp.jpg' '2' (带标签输出逻辑)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='Assign the image path.', default="")
    parser.add_argument('c', type=int, help='Assign the c.', default="")
    args = parser.parse_args()
    alg_core = LPAlg(pb_path_1="model/csn_model.pb")
    result_dict = ProjectInterface({args.image: args.image}, args.c, proxy=alg_core)
    print(result_dict)
    queryVec = result_dict[args.image]['embedding']
    print ('>>>>', queryVec)
