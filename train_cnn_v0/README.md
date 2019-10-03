
> 自己搭建的一个框架，包含模型有：vgg(vgg16,vgg19), resnet(resnet_v2_50,resnet_v2_101,resnet_v2_152), inception_v4, inception_resnet_v2等。

>
此框架主要针对分类任务， 后面会陆续搭建多任务多标签、检测、以及rnn等框架，欢迎关注。
使用说明：
搭建时使用的环境为：Python3.5, tensorflow1.4

变量设置参考config.py。
详细说明参见config.py。

( mkdir pretrain/inception_v4, 下载预训练模型, cp到pretrain/inception_v4/ ) 

运行代码： python main.py 

其中，z_ckpt_pb：ckpt转pb的代码，和测试接口。

另外如果想使用tensorboard，请使用train_net下面的train_tensorboard.py。将在工程目录下生成 xxx_log 的文件。
然后使用：tensorboard --logdir arch_inceion_v4_log查看。
后续有时间会把其它的功能加上，并且每个代码文件都会进行封装，用类的形式呈现。


对dl感兴趣，还可以关注我的博客，这是我的博客目录：（地址： http://blog.csdn.net/u014365862/article/details/78422372 ）
本文为博主原创文章，未经博主允许不得转载。有问题可以加微信：lp9628(注明CSDN)。

