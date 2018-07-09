
使用方法：
训  练： python main.py （通过config.PY设置参数）
可视化：python vis_cam.py



训练平台搭建代码说明：

1.  gender文件：
    存放样本，不同的类别已不同的文件夹存放。

2.  lib文件：
  （1）model文件：
      各网络。
  （2）data_aug文件：
      用于图像增强， 里边包含两种方法。
  （3）grad_cam文件：
      可视化模块。
  （4）data_load文件：
      加载训练数据。
  （5）train_文件：
      构建训练。
  （6）utils文件：
      。。。
   (7) loss文件：
      策略：损失函数。
  （8）optimizer文件：
      优化方法。

3.  model文件：
    存放训练过程中的保存的模型。

4.  pretrain文件：
    迁移学习中的预训练模型。

5.  config.py文件：
    调整训练过程的参数。

6.  main.py文件：
    启动训练: python main.py

7.  vis_cam.py文件：
    可视化： python vis_cam.py
8.  ckpt_pb.py文件：
    ckpt转pb的。
9.  test文件：
    用于模型的测试。