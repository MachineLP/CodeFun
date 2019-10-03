
> tensorflow搭建的一个训练框架，包含模型有：vgg(vgg16,vgg19), resnet(resnet_v2_50,resnet_v2_101,resnet_v2_152), inception_v4, inception_resnet_v2等。

> 此框架主要针对分类任务， 后面会陆续搭建多任务多标签、检测、以及rnn等框架，欢迎关注。
搭建时使用的环境为：Python3.5, tensorflow1.4

具体:

- [train_cnn_v0](https://github.com/MachineLP/train_arch/tree/master/train_cnn_v0)
> 实现基础cnn训练，数据读取方式慢。

- [train_cnn_v1](https://github.com/MachineLP/train_arch/tree/master/train_cnn_v1)
> 优化数据读取的方式，学习率加入衰减。

- [train_cnn-rnn](https://github.com/MachineLP/train_cnn-rnn)
> 在train_cnn_v0基础上加入rnn。

- [train_cnn-rnn-attention_v0](https://github.com/MachineLP/train_cnn-rnn-attention)
> 在train_cnn_v0基础上加入rnn、attention。

- [train_cnn_multiGPU_v0](https://github.com/MachineLP/train_arch/tree/master/train_cnn_multiGPU_v0)
> 使用多GPU训练(默认两块gpu)，以上其他框架使用多GPU，只需把train.py替换掉就可以了。

- [train_cnn_multilabel](https://github.com/MachineLP/train_cnn_multilabel)
> 多任务多标签训练及其总结。

- [train_cnn_GANs](https://github.com/MachineLP/train_cnn_GANs)
> GANs训练及其总结。

- [TensorFlow基础教程](https://github.com/MachineLP/Tensorflow-)
> 理论及其代码实践。

- [python实践教程](https://github.com/MachineLP/py_workSpace)
> MachineLP的日常代码。


对dl感兴趣，还可以关注我的博客，这是我的博客目录：（地址： http://blog.csdn.net/u014365862/article/details/78422372 ）
本文为博主原创文章，未经博主允许不得转载。有问题可以加微信：lp9628(注明CSDN)。

公众号MachineLN，邀请您扫码关注： 

![image](http://upload-images.jianshu.io/upload_images/4618424-3ef1722341ba72d2?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) 
