# coding = utf-8

import tensorflow as tf 
from lib.utils.utils import input_placeholder, g_parameter, build_net
from lib.grad_cam.grad_cam import vis
from skimage import io
from matplotlib import pyplot as plt
import cv2
import numpy as np
import config

arch_model = config.arch_model

def main(img, height, width, num_classes, mode_dir):
    X, Y, is_train, keep_prob_fc = input_placeholder(height, width, num_classes)
    net, net_vis = build_net(X, num_classes, keep_prob_fc, is_train, arch_model)

    net_ = tf.nn.softmax(net)
    predict = tf.reshape(net_, [-1, num_classes])
    model_output = tf.argmax(predict, 1)

    sess = tf.Session()
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, mode_dir)

    predicted_class = sess.run(model_output, feed_dict={X: [img], keep_prob_fc:1.0, is_train:False})

    cam3 = vis().grad_cam(img, X, keep_prob_fc, is_train, net_vis, net, sess, predicted_class[0], num_classes)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(float)
    img /= img.max()
    # Superimposing the visualization with the image.
    cam3 = cv2.applyColorMap(np.uint8(255*cam3), cv2.COLORMAP_JET)
    cam3 = cv2.cvtColor(cam3, cv2.COLOR_BGR2RGB)
    
    alpha = 0.0025
    new_img = img+alpha*cam3
    # new_img = img+3*cam3
    new_img /= new_img.max()

    # Display and save
    io.imshow(new_img)
    plt.show()
    io.imsave('vis.jpg', new_img)

def vis_conv(img, height, width, num_classes, mode_dir):
    X, Y, is_train, keep_prob_fc = input_placeholder(height, width, num_classes)
    net, net_vis = build_net(X, num_classes, keep_prob_fc, is_train, arch_model)

    net_ = tf.nn.softmax(net)
    predict = tf.reshape(net_, [-1, num_classes])
    model_output = tf.argmax(predict, 1)

    sess = tf.Session()
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, mode_dir)

    predicted_class = sess.run(model_output, feed_dict={X: [img], keep_prob_fc:1.0, is_train:False})

    cam3 = vis().grad_cam(img, X, keep_prob_fc, is_train, net_vis, net, sess, predicted_class[0], num_classes)
    
    img = img.astype(float)
    img /= img.max()
    cam3 = cv2.applyColorMap(np.uint8(255*cam3), cv2.COLORMAP_JET)
    # cam3 = cv2.cvtColor(cam3, cv2.COLOR_BGR2RGB)
    alpha = 0.0025
    new_img = img+alpha*cam3
    new_img /= new_img.max()
    return new_img



if __name__ == '__main__':
    img = cv2.imread('./lp.jpg')
    height, width = config.height, config.width
    num_classes = config.num_classes
    '''
    train_dir = './model'
    latest = tf.train.latest_checkpoint(train_dir)
    if not latest:
        print ("No checkpoint to continue from in", train_dir)
        sys.exit(1)
    print ("resume", latest)
    mode_dir = latest
    '''
    mode_dir = './model/model.ckpt-0'
    img = cv2.resize(img, (height, width)) 
    # main(img, height, width, num_classes, mode_dir)
    vis_image = vis_conv(img, height, width, num_classes, mode_dir)
    cv2.imshow('vis_image', vis_image)
    cv2.waitKey(0)

