# coding=utf-8

# 训练文件夹
sample_dir = "train.txt" # 可以为txt， 但是需要该代码
tfRecord = False
# 需要分类的类别数量
num_classes = 20
# 最小批训练的大小
batch_size = 2
# 选择使用的模型
# arch_model="arch_resnet_v2_50_multi_label"
arch_model = "arch_inception_v4_multi_label"
# 选择训练的网络层
checkpoint_exclude_scopes = "Logits_out"
# dropout的大小
dropout_prob = 0.8
# 选择训练样本的比例
train_rate = 0.9
# 整个训练集上进行多少次迭代
epoch = 2000
# 是否使用提前终止训练
early_stop = True
EARLY_STOP_PATIENCE = 1000
# 是否使用learning_rate
learning_r_decay = True
learning_rate_base = 0.0001 
decay_rate = 0.95
height, width = 299,299 #224, 224
# 模型保存的路径
train_dir = 'model'
# 是否进行fine-tune。 选择fine-tune的的参数
fine_tune = False
# 是否训练所有层的参数
train_all_layers = False
# 迁移学习模型参数， 下载训练好模型：https://github.com/MachineLP/models/tree/master/research/slim
# checkpoint_path="pretrain/inception_v4/inception_v4.ckpt";
# checkpoint_path="pretrain/resnet_v2_50/resnet_v2_50.ckpt"
checkpoint_path = 'pretrain/inception_v4/inception_v4.ckpt'
