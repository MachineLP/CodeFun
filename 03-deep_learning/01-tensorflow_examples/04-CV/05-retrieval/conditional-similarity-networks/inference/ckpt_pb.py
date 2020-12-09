# coding = utf-8

import cv2
import numpy as np
import os
import sys
import config
import tensorflow as tf
slim = tf.contrib.slim
from tensorflow.python.framework import graph_util

try:
    from csn_net import ConditionalSimNet, NetArch
except:
    from csn_lib.csn_net import ConditionalSimNet, NetArch
try:
    from utils import input_placeholder3, triplet_loss, g_parameter, get_next_batch_from_path, create_pairs
except:
    from csn_lib.utils import input_placeholder3, triplet_loss, g_parameter, get_next_batch_from_path, create_pairs


height = config.height
width = config.width
epoch = config.epoch
num_classes = config.num_classes
n_conditions = config.n_conditions
embedding_size = config.embedding_size
conditions = config.conditions
margin = config.margin
batch_size = config.batch_size
dropout_prob = config.dropout_prob
checkpoint_exclude_scopes = config.checkpoint_exclude_scopes
learning_r_decay = config.learning_r_decay
learning_rate_base = config.learning_rate_base
decay_rate = config.decay_rate
checkpoint_path = config.checkpoint_path
early_stop = config.early_stop
EARLY_STOP_PATIENCE = config.EARLY_STOP_PATIENCE
embed_loss = config.embed_loss
mask_loss = config.mask_loss
hard_sample_train = config.hard_sample_train


MODEL_DIR = "../model/"
MODEL_NAME = "csn_model.pb"
if not tf.gfile.Exists(MODEL_DIR): #创建目录
    tf.gfile.MakeDirs(MODEL_DIR)

def l2norm(x):
    norm2 = tf.norm(x, ord=2, axis=1)
    norm2 = tf.reshape(norm2,[-1,1])
    l2norm = x/norm2
    return l2norm


X = tf.placeholder(tf.float32, [None, height, width, 3], name = "inputs_placeholder")
C = tf.placeholder(tf.int32, [None, None], name = "C")

# arch_resnet_v2_50, arch_inception_v4, arch_vgg16
with tf.variable_scope("", reuse=tf.AUTO_REUSE) as scope:
    _, net_vis_x = NetArch().arch_vgg16(X, num_classes, 1.0, False, embedding_size)
    with tf.variable_scope("Logits_csn"):
        csn = ConditionalSimNet(n_conditions, embedding_size, learnedmask=True, prein=True)
        embedded_x, masknorm_norm_x, embed_norm_x, tot_embed_norm_x = csn.forward(net_vis_x, C)
embedded_x = l2norm(embedded_x)
# net = tf.nn.softmax(net)
# predict = tf.reshape(net, [-1, 1536], name='predictions')
predict = tf.reshape(embedded_x, [-1, embedding_size], name='predictions')


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
        graph = load_graph("model/csn_model.pb")
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
    model_folder = latest

    freeze_graph(model_folder)

