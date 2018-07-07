# coding=utf-8

import tensorflow as tf
from skimage.transform import resize
import numpy as np

class vis:

    def __init__(self):
        pass

    def grad_cam(self, img_input, X_, keep_prob_fc, is_train, net_layers, net, sess, predicted_class, nb_classes):
        '''
        img_input       :  需要预测类别的图像。 (cv2.imread())
        X_              :  输入占位符 (模型的输入)
        keep_prob_fc    :  dropout
        is_train        :  False
        net_layers      :  卷积层的输出  (一般都是卷积最后一层的输出)
        net             :  模型的输出， softmax之前
        sess            :  
        predicted_class :  模型预测的类别
        nb_classes      :  总类别
        '''
        
        conv_layer =  net_layers
        one_hot = tf.sparse_to_dense(predicted_class, [nb_classes], 1.0)
        signal = tf.multiply(net, one_hot)

        loss = tf.reduce_mean(signal)
        grads = tf.gradients(loss, conv_layer)[0]

        norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))

        output, grads_val = sess.run([conv_layer, norm_grads], feed_dict={X_: [img_input], keep_prob_fc:1.0, is_train:False})

        output = output[0]           # [8,8,1536]
        grads_val = grads_val[0]	 

        weights = np.mean(grads_val, axis = (0, 1)) 			# [1536]
        cam = np.ones(output.shape[0 : 2], dtype = np.float32)	# [8,8]

        for i, w in enumerate(weights):
    	    cam += w * output[:, :, i]
        
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)
        cam = resize(cam, (299,299))

        cam3 = np.expand_dims(cam, axis=2)
        cam3 = np.tile(cam3,[1,1,3])
        
        return cam3