
# 自己搭建的一个框架，包含模型有：vgg(vgg16,vgg19), resnet(resnet_v2_50,resnet_v2_101,resnet_v2_152), inception_v4, inception_resnet_v2等。
# 代码有点小乱，欢迎帮忙整理。
# 此框架主要针对分类任务， 后面会陆续搭建多任务多标签、检测、以后rnn等框架，欢迎关注。
使用说明：
搭建时使用的环境为：Python3.5, tensorflow1.2.1

变量设置参考config.py。
详细说明参见config.py。

其中，z_ckpt_pb：ckpt转pb的代码，和测试接口。


对dl感兴趣，还可以关注我的博客，这是我的博客目录：（地址： http://blog.csdn.net/u014365862/article/details/78422372 ） 本文为博主原创文章，未经博主允许不得转载。有问题可以加微信：lp9628(注明CSDN)。

人脸检测系列：

人脸检测——AFLW准备人脸
人脸检测——生成矫正人脸——cascade cnn的思想， 但是mtcnn的效果貌似更赞
人脸检测——准备非人脸
人脸检测——矫正人脸生成标签
人脸检测——mtcnn思想，生成negative、positive、part样本。
人脸检测——滑动窗口篇（训练和实现）
人脸检测——fcn
简单的人脸跟踪
Face Detection(OpenCV) Using Hadoop Streaming API
Face Recognition(face_recognition) Using Hadoop Streaming API
非极大值抑制（Non-Maximum-Suppression）
OCR系列：

tf20: CNN—识别字符验证码
身份证识别——生成身份证号和汉字
tf21: 身份证识别——识别身份证号
tf22: ocr识别——不定长数字串识别
baiduyun_deeplearning_competition
机器学习，深度学习系列:

反向传播与它的直观理解
卷积神经网络（CNN）：从原理到实现
机器学习算法应用中常用技巧-1
机器学习算法应用中常用技巧-2
一个隐马尔科夫模型的应用实例：中文分词
Pandas处理csv表格
sklearn查看数据分布
TensorFlow 聊天机器人
YOLO
感知机--模型与策略
从 0 到 1 走进 Kaggle
python调用Face++，玩坏了！
人脸识别keras实现教程
机器学习中的Bias(偏差)，Error(误差)，和Variance(方差)有什么区别和联系？
CNN—pooling层的作用
trick—Batch Normalization
tensorflow使用BN—Batch Normalization
trick—Data Augmentation
CNN图图图
为什么很多做人脸的Paper会最后加入一个Local Connected Conv？
Faster RCNN：RPN，anchor，sliding windows
深度学习这些坑你都遇到过吗？
image——Data Augmentation的代码
8种常见机器学习算法比较
几种常见的激活函数
Building powerful image classification models using very little data
机器学习模型训练时候tricks
OCR综述
一个有趣的周报
根据已给字符数据，训练逻辑回归、随机森林、SVM，生成ROC和箱线图
图像处理系列：

python下使用cv2.drawContours填充轮廓颜色
imge stitching图像拼接stitching
用python简单处理图片（1）：打开\显示\保存图像
用python简单处理图片（2）：图像通道\几何变换\裁剪
用python简单处理图片（3）：添加水印
用python简单处理图片（4）：图像中的像素访问
用python简单处理图片（5）：图像直方图
仿射变换，透视变换：二维坐标到二维坐标之间的线性变换，可用于landmark人脸矫正。
代码整合系列：

windows下C++如何调用matlab程序
ubuntu下C++如何调用matlab程序
matlab使用TCP/IP Server Sockets
ubuntu下C++如何调用python程序，gdb调试C++代码
How to pass an array from C++ to an embedded python
如何使用Python为Hadoop编写一个简单的MapReduce程序
图像的遍历
ubuntu下CMake编译生成动态库和静态库，以OpenTLD为例。
ubuntu下make编译生成动态库，然后python调用cpp。
数据结构和算法系列：

堆排序
red and black (深度优先搜索算法dfs)
深度优先搜索算法
qsort原理和实现
stack实现queue ; list实现stack
另一种斐波那契数列
堆和栈的区别(个人感觉挺不错的)
排序方法比较
漫画 ：什么是红黑树？
kinect 系列：

Kinect v2.0原理介绍之一：硬件结构
Kinect v2.0原理介绍之二：6种数据源
Kinect v2.0原理介绍之三：骨骼跟踪的原理
Kinect v2.0原理介绍之四：人脸跟踪探讨
Kinect v2.0原理介绍之五：只检测离kinect最近的人脸
Kinect v2.0原理介绍之六：Kinect深度图与彩色图的坐标校准
Kinect v2.0原理介绍之七：彩色帧获取
Kinect v2.0原理介绍之八：高清面部帧(1) FACS 介绍
Kinect v2.0原理介绍之九：高清面部帧(2) 面部特征对齐
Kinect v2.0原理介绍之十：获取高清面部帧的AU单元特征保存到文件
kinect v2.0原理介绍之十一：录制视频
Kinect v2.0原理介绍之十二：音频获取
Kinect v2.0原理介绍之十三：面部帧获取
Kinect for Windows V2和V1对比开发___彩色数据获取并用OpenCV2.4.10显示
Kinect for Windows V2和V1对比开发___骨骼数据获取并用OpenCV2.4.10显示
用kinect录视频库
tensorflow系列：

Ubuntu 16.04 安装 Tensorflow(GPU支持)
使用Python实现神经网络
tf1: nn实现评论分类
tf2: nn和cnn实现评论分类
tf3: RNN—mnist识别
tf4: CNN—mnist识别
tf5: Deep Q Network—AI游戏
tf6: autoencoder—WiFi指纹的室内定位
tf7: RNN—古诗词
tf8:RNN—生成音乐
tf9: PixelCNN
tf10: 谷歌Deep Dream
tf11: retrain谷歌Inception模型
tf12: 判断男声女声
tf13: 简单聊天机器人
tf14: 黑白图像上色
tf15: 中文语音识别
tf16: 脸部特征识别性别和年龄
tf17: “声音大挪移”
tf18: 根据姓名判断性别
tf19: 预测铁路客运量
tf20: CNN—识别字符验证码
tf21: 身份证识别——识别身份证号
tf22: ocr识别——不定长数字串识别
tf23: “恶作剧” --人脸检测
tf24: GANs—生成明星脸
tf25: 使用深度学习做阅读理解+完形填空
tf26: AI操盘手
tensorflow_cookbook--preface
01 TensorFlow入门（1）
01 TensorFlow入门（2）
02 The TensorFlow Way（1）
02 The TensorFlow Way（2）
02 The TensorFlow Way（3）
03 Linear Regression
04 Support Vector Machines
tf API 研读1：tf.nn，tf.layers， tf.contrib概述
tf API 研读2：math
tensorflow中的上采样(unpool)和反卷积(conv2d_transpose)
tf API 研读3：Building Graphs
tf API 研读4：Inputs and Readers
tf API 研读5：Data IO
tf API 研读6：Running Graphs
tf.contrib.rnn.static_rnn与tf.nn.dynamic_rnn区别
Tensorflow使用的预训练的resnet_v2_50，resnet_v2_101，resnet_v2_152等模型预测，训练
tensorflow下设置使用某一块GPU、多GPU、CPU的情况
工业器件检测和识别
将tf训练的权重保存为CKPT,PB ,CKPT 转换成 PB格式。并将权重固化到图里面,并使用该模型进行预测
tensorsor快速获取所有变量，和快速计算L2范数
cnn+rnn+attention
Tensorflow实战学习笔记
tf27: Deep Dream—应用到视频
tf28: 手写汉字识别
C++系列：

c++ primer之const限定符
c++primer之auto类型说明符
c++primer之预处理器
c++primer之string
c++primer之vector
c++primer之多维数组
c++primer之范围for循环
c++primer之运算符优先级表
c++primer之try语句块和异常处理
c++primer之函数(函数基础和参数传递)
c++primer之函数(返回类型和return语句)
c++primer之函数重载
c++重写卷积网络的前向计算过程，完美复现theano的测试结果
c++ primer之类
c++primer之类（构造函数再探）
c++primer之类（类的静态成员）
c++primer之顺序容器（容器库概览）
c++primer之顺序容器（添加元素）
c++primer之顺序容器（访问元素）
OpenCV系列：

自己训练SVM分类器，进行HOG行人检测。
opencv-haar-classifier-training
vehicleDectection with Haar Cascades
LaneDetection
OpenCV学习笔记大集锦
Why always OpenCV Error: Assertion failed (elements_read == 1) in unknown function ?
目标检测之训练opencv自带的分类器（opencv_haartraining 或 opencv_traincascade）
车牌识别 之 字符分割
仿射变换，透视变换：二维坐标到二维坐标之间的线性变换，可用于landmark人脸矫正。
python系列（web开发、多线程等）：

flask的web开发，用于机器学习（主要还是DL）模型的简单演示。
python多线程，获取多线程的返回值
其他:

MAC平台下Xcode配置使用OpenCV的具体方法 (2016最新)
python下如何安装.whl包？
给中国学生的第三封信：成功、自信、快乐
自己-社会-机器学习
不执著才叫看破，不完美才叫人生。
PCANet的C++代码——详细注释版
责任与担当
好走的都是下坡路
一些零碎的语言，却触动到内心深处。
用一个脚本学习 python
一个有趣的周报