
> 自己搭建的一个训练框架，包含模型有：cnn+rnn+attention: vgg(vgg16,vgg19)+rnn(LSTM, GRU)+attention, resnet(resnet_v2_50,resnet_v2_101,resnet_v2_152)+rnn(LSTM, GRU)+attention, inception_v4+rnn(LSTM, GRU)+attention, inception_resnet_v2+rnn(LSTM, GRU)+attention等。

>此框架主要针对分类任务， 后面会陆续搭建多任务多标签、检测等框架，欢迎关注。
使用说明：
搭建时使用的环境为：Python3.5, tensorflow1.4

变量设置参考config.py。
详细说明参见config.py。

( mkdir pretrain/inception_v4, 下载与训练模型, cp到pretrain/inception_v4/ ) 

运行代码： python main.py 

另外此代码加了tensorboard，将在工程目录下生成 xxx_log 的文件。 然后使用：tensorboard --logdir arch_inceion_v4_rnn_attention_train_log查看(tensorboard --logdir arch_inceion_v4_rnn_attention_valid_log)。 后续有时间会把其它的功能加上。

其中，z_ckpt_pb：ckpt转pb的代码，和测试接口。


对dl感兴趣，还可以关注我的博客，这是我的博客目录：（地址： http://blog.csdn.net/u014365862/article/details/78422372 ）
本文为博主原创文章，未经博主允许不得转载。有问题可以加微信：lp9628(注明CSDN)。

