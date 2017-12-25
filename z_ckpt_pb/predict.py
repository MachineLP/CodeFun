#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from scipy import misc
import tensorflow as tf
from threading import Lock
import os
import cv2
from inception_preprocessing import *
import sys


# classes of a sub-model
model_class = 3

# GPU 参数配置
def GPU_config(rate=0.99):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"      # 按照PCI_BUS_ID顺序从0开始排列GPU设备
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # 设置当前使用的GPU设备仅为0号设备
    gpuConfig = tf.ConfigProto()
    gpuConfig.allow_soft_placement = False      #设置为True，当GPU不存在或者程序中出现GPU不能运行的代码时，自动切换到CPU运行
    gpuConfig.gpu_options.allow_growth = True       #设置为True，程序运行时，会根据程序所需GPU显存情况，分配最小的资源
    gpuConfig.gpu_options.per_process_gpu_memory_fraction = rate    #程序运行的时，所需的GPU显存资源最大不允许超过rate的设定值

    return gpuConfig


def detect(image):
    im = cv2.GaussianBlur(image, (5, 5), 0)
    im = cv2.Canny(im, 1, 130)
    nonzero = np.nonzero(im)

    if len(nonzero[0]) <= 4:
        return None

    h_set = nonzero[0]
    w_set = nonzero[1]
    w_min = w_set[np.argmax(-w_set, axis=0)]
    w_max = w_set[np.argmax(w_set, axis=0)]
    h_min = h_set[np.argmax(-h_set, axis=0)]
    h_max = h_set[np.argmax(h_set, axis=0)]

    return [w_min,h_min,w_max-w_min+1,h_max-h_min+1]


tp = 100
def img_crop(img, box):
    # y1, x1, y2, x2 = box[1]-20, box[0]-20, box[1]+box[3]+40, box[0]+box[2]+40
    y1, x1, y2, x2 = box[1]-tp, box[0]-tp, box[1]+box[3]+tp, box[0]+box[2]+tp
    if y1 < 0:
        y1 = 0
    if x1 < 0:
        x1 = 0
    if y2 > 1080:
       y2 = 1080
    if x2 > 1920:
       x2 = 1920
    img = img[y1:y2, x1:x2]
    return img


class TEAlg(object):
    '''
    __instance = None
    _singleton_lock = Lock()

    def __new__(cls, *args, **kwargs):

        if cls.__instance is None:
            cls._singleton_lock.acquire()
            cls.__instance = (super(TEAlg, cls).__new__(cls, *args, **kwargs), cls.__instance)[
                cls.__instance is not None]
            cls._singleton_lock.release()

        return cls.__instance'''

    # default model path of .pb
    PB_PATH_1 = os.path.join(os.getcwd(), "model", "frozen_model.pb")
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
            #for op in graph.get_operations():
            #    print(op.name, op.values())
            #    print("name111111111111:",op.name)
            pred = graph.get_tensor_by_name('prefix/predictions:0') # cell_compute_4  4为sequence_length-2
            #batch_size = graph.get_tensor_by_name('prefix/batch_size:0')
            #label_indices = graph.get_tensor_by_name('prefix/train/LSTM/label_indices:0')
            batch_size = tf.placeholder(tf.float32, [None, 1])
            label_indices = tf.placeholder(tf.float32, [None, 2])
            x = graph.get_tensor_by_name('prefix/inputs_placeholder:0')
            sess = tf.Session(graph=graph,config=gpu_config)
            return [sess,x,batch_size,label_indices,pred]

        # multiple models
        def multi_model_def(pb_path_1):
            model_1 = sess_def(pb_path_1)
            return [model_1]

        path_1 = get_path(pb_path_1, TEAlg.PB_PATH[0])
        self._pb_path = [path_1]

        self.model = multi_model_def(self._pb_path[0])
        self.sess = tf.Session()
        #self.input = tf.placeholder(tf.float32, [None, None, 3])  
        #self.out = preprocess_image(self.input, 299,299, is_training=False)

    def _close(self):
        self.model[0][0].close()

    def _run(self, images_path=None):

        def img_pre_proc(filename):
            # return img, flag
            # @brief: flag-0: this file is invalid
            #         flag-1:this file is valid
            try:
                #img = misc.imread(filename)
                '''if img.ndim == 3 and img.shape[-1] == 3:
                    pass
                elif img.ndim == 2:
                    img = to_rgb(img)
                elif img.ndim == 3 and img.shape[-1] == 4:
                    img = img[:, :, :3]  # 4 channels convert to 3 channels
                else:
                    return 0, 0  '''
                #img = misc.imresize(img, [100, 100])
                
                #print (filename)
                #img = cv2.imread(filename)
                #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                #print (image)
                '''
                im = tf.read_file(filename)
                im = tf.image.decode_jpeg(im)
                img = self.sess.run(im)
                img = self.sess.run(self.out, feed_dict={self.input:img})
                img = tf.reshape(img, [299,299,3])
                img = tf.cast(img, tf.float32)
                img = self.sess.run(img)'''

                img = cv2.imread(filename)
                #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                box = detect(gray)
                print(filename, '=======>', box)
                img = img_crop(img, box)
                # cv2.imwrite(filename, img)
                print ('=======>',img.shape)
                img = cv2.resize(img, (299, 299))
                # img = (img-127.5)
                
                img = img / 255.0
                img = img - 0.5
                img = img * 2
                

                # print ("img", img)
                #img = TEAlg._prewhiten(img)
                return img,1
            except:
                return 0,0

        images_path = list(images_path)
        data_length = len(images_path)

        # no input data
        if data_length == 0:
            null_data = []
            for null_idx in range(data_length):
                null_data.append([])
            return null_data#,null_data

        # batch input processing
        imgs = []
        ncount = 0
        img_status = np.zeros(data_length)          # invalid data one-hot array: 0-valid, 1-invalid
        for i in range(data_length):
            
            img, flag = img_pre_proc(images_path[i])
            #print ("image", img)
            if flag == 0:
                img_status[i] = 1
                continue
            img = np.array([img])
            if ncount != 0:
                imgs = np.concatenate((imgs, img), axis=0)
            else:
                imgs = img
            ncount += 1

        # if all input data are invalid
        if ncount == 0:
            null_data = []
            for null_idx in range(data_length):
                null_data.append([])
            return null_data#, null_data

        label_indices_val = [TEAlg.CLASS_NUMBER for index in range(ncount)]

        # run
        idx = 0     # model index
        # print (imgs)
        predict_1 = self.model[idx][0].run(
            self.model[idx][4],
            feed_dict={self.model[idx][1]: imgs
                       #self.model[idx][2]: ncount,
                       #self.model[idx][3]: label_indices_val
                                  }
                                  )
        print (predict_1)
        # return if all input data are valid
        if ncount == data_length:
            return predict_1.tolist()#,predict_2.tolist()

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
            merge_predict_1.append(predict_1[index].tolist())

        return merge_predict_1

    @staticmethod
    def _prewhiten(x):
        mean = [119.22448922, 112.39633294, 110.69453342]  # yinian
        x = x[:, :, :-1] if x.shape[2] > 3 else x
        y = np.subtract(x, mean)
        return y

    @staticmethod
    def to_rgb(img):
        w, h = img.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
        return ret

keywords_1 = [u'NG2',u'NG1',u'OK']
keywords = keywords_1


def ProjectInterface(image_path_list, proxy=None):

    images_path = image_path_list.keys()
    predict = proxy._run(images_path)

    result_dict = {}
    image_number = len(images_path)

    for i in range(image_number):
        # hongyanquan 20 classes
        target = np.array(predict[i][:model_class])
        #print (target)
        top = {}
        for index, value in enumerate(target):
            #print (index)
            top[keywords[index]] = np.float(value) 
            #print (top)
        result_dict[list(image_path_list.values())[i]] = top

    return result_dict

# 下面是用于以上模块测试的。
'''
image_path_list = ["lp.jpg"]
alg_core = TEAlg(pb_path_1="model/frozen_model.pb")
out = ProjectInterface(image_path_list, alg_core)
print (out)'''


####################### 工具函数 ##########################
if __name__ == "__main__":
    # python predict.py lp.jpg (带标签输出逻辑)
    # python predict.py --mode alg lp.jpg(纯算法输出逻辑)
    # python predict.py -n lp.jpg -n "https://ss0.bdstatic.com/70cFuHSh_Q1YnxGkpoWK1HF6hhy/it/u=3669177687,479791909&fm=27&gp=0.jpg" 1(多张图片演示)
    # python predict.py -n "https://ss0.bdstatic.com/70cFuHSh_Q1YnxGkpoWK1HF6hhy/it/u=3669177687,479791909&fm=27&gp=0.jpg" 1(网络图片演示)
    import argparse, json, textwrap

    parser = argparse.ArgumentParser(prog='Detection Yinian',
                                     description=textwrap.dedent('''\
                                            Please read this!!
                                        -----------------------------
                                            TASK: Detection 
                                        -----------------------------
                                            Program by xxxxx
                                            Core by tensorflow  '''),
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="copyright: (c) 2017-2018 svaxm")

    parser.add_argument('image', type=str, help='Assign the image path.', default="")
    parser.add_argument("-m", '--mode', type=str, help='change mode: enum values[PRO, ALG]', default="PRO", dest="mode")
    parser.add_argument('-n', "--network", dest="url_from_network", action='append', default=[],
                        help='Add image url to a list')
    args = parser.parse_args()

    alg_core = TEAlg(pb_path_1="model/frozen_model.pb")
    result = None

    if args.url_from_network:
        import re
        try:
            import urllib.request as urllib2
        except:
            import urllib2
        from io import BytesIO

        def _string2url(url):
            try:
                if re.match(r'^(https?|ftp):/{2}\w.+$', url):
                    url = urllib2.urlopen(url)
                    url = BytesIO(url.read())
            except:
                url = url
            return url

        args.url_from_network = map(lambda x: x.strip("\"\'"), args.url_from_network)
        new_args_n = map(_string2url, args.url_from_network)
        print(args.url_from_network)

        if args.mode.upper() == "PRO":
            result_dict = ProjectInterface(dict(zip(new_args_n, args.url_from_network)), proxy=alg_core)
            if sys.version.split('.')[0] == '3':
                result = json.dumps(result_dict, ensure_ascii=False)
            else:
                result = json.dumps(result_dict, ensure_ascii=False, encoding='UTF-8')
        else:
            result = alg_core._run(new_args_n)
    else:
        if args.mode.upper() == "PRO":
            result_dict = ProjectInterface({args.image: args.image}, proxy=alg_core)
            if sys.version.split('.')[0] == '3':
                result = json.dumps(result_dict, ensure_ascii=False)
            else:
                result = json.dumps(result_dict, ensure_ascii=False, encoding='UTF-8')
        else:
            result = alg_core._run([args.image])

    alg_core._close()
    print ("第0位一个男人概率，第1位一个女人的概率，第2位多个人的概率，第3位其它")
    print(result)

