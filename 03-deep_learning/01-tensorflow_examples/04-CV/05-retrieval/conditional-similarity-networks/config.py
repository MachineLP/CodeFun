# coding=utf-8

# 定义风格， 检索的时候你想按照的多少类风格数。
n_conditions = 4
# tploss margin
margin = 0.5
# 定义映射的特征维度
embedding_size = 64
# 定义风格的选择形式
conditions = [0,1,2,3]
# 最小批训练的大小
batch_size = 128
# hard_sample_train
hard_sample_train = False
# 选择使用的模型， 此处没有用到，但是可以加上。
# arch_model="arch_vgg16"
# 选择训练的网络层
checkpoint_exclude_scopes = "Logits_out, Logits_csn"
# loss系数
embed_loss = 5e-3
mask_loss = 5e-4
# dropout的大小
dropout_prob = 0.8
# 整个训练集上进行多少次迭代
epoch = 2000
# 是否使用提前终止训练
early_stop = True
EARLY_STOP_PATIENCE = 1000
# 是否使用learning_rate
learning_r_decay = True
learning_rate_base = 0.0001
decay_rate = 0.95
height, width = 112, 112
# 迁移学习的网络模
checkpoint_path = 'pretrain/vgg/vgg_16.ckpt'
# checkpoint_path = 'pretrain/inception_v4/inception_v4.ckpt'
# checkpoint_path = 'pretrain/resnet_v2/resnet_v2_50.ckpt'
# 需要分类的类别数量,这个参数没用到
num_classes = 4
