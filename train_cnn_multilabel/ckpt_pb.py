# coding = utf-8
"""
Created on 2017 10.17
@author: liupeng
wechat: lp9628
blog: http://blog.csdn.net/u014365862/article/details/78422372
"""

import tensorflow as tf
from tensorflow.python.framework import graph_util
from lib.utils.multi_label_utils import get_next_batch_from_path, shuffle_train_data
from lib.utils.multi_label_utils import input_placeholder, build_net_multi_label, cost, train_op, model_mAP
import cv2
import numpy as np
import os
import sys
import config
MODEL_DIR = "model/"
MODEL_NAME = "frozen_model.pb"
if not tf.gfile.Exists(MODEL_DIR): #创建目录
    tf.gfile.MakeDirs(MODEL_DIR)

height, width = config.height, config.width
num_classes = config.num_classes
arch_model = config.arch_model

X = tf.placeholder(tf.float32, [None, height, width, 3], name = "inputs_placeholder")
net, net_vis = build_net_multi_label(X, num_classes, 1.0, False, arch_model)
net = tf.nn.sigmoid(net)
predict = tf.reshape(net, [-1, num_classes], name='predictions')


def freeze_graph(model_folder):
    #checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    #input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径
    input_checkpoint = model_folder
    output_graph = os.path.join(MODEL_DIR, MODEL_NAME) #PB模型保存路径

    output_node_names = "predictions" #原模型输出操作节点的名字
    #saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True) #得到图、clear_devices ：Whether or not to clear the device field for an `Operation` or `Tensor` during import.
    saver = tf.train.Saver()

    graph = tf.get_default_graph() #获得默认的图
    input_graph_def = graph.as_graph_def()  #返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, input_checkpoint) #恢复图并得到数据

        #print "predictions : ", sess.run("predictions:0", feed_dict={"input_holder:0": [10.0]}) # 测试读出来的模型是否正确，注意这里传入的是输出 和输入 节点的 tensor的名字，不是操作节点的名字

        output_graph_def = graph_util.convert_variables_to_constants(  #模型持久化，将变量值固定
            sess,
            input_graph_def,
            output_node_names.split(",") #如果有多个输出节点，以逗号隔开
        )
        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点

        for op in graph.get_operations():
            #print(op.name, op.values())
            print("name:",op.name)
        print ("success!")


        #下面是用于测试， 读取pd模型，答应每个变量的名字。
        graph = load_graph("model/frozen_model.pb")
        for op in graph.get_operations():
            #print(op.name, op.values())
            print("name111111111111:",op.name)
        pred = graph.get_tensor_by_name('prefix/inputs_placeholder:0')
        print (pred)
        temp = graph.get_tensor_by_name('prefix/predictions:0')
        print (temp)

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


if __name__ == '__main__':
    train_dir = 'model'
    latest = tf.train.latest_checkpoint(train_dir)
    if not latest:
        print ("No checkpoint to continue from in", train_dir)
        sys.exit(1)
    print ("resume", latest)
    # saver2.restore(sess, latest)
    # model_folder = './model/model.ckpt-0'
    model_folder = latest
    freeze_graph(model_folder)

