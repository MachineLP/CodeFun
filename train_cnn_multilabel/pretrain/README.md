

# 预训练好的模型放在这里。

## arch_inception_v4 download inception_v4 model: http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
wget http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
tar zxvf inception_v4_2016_09_09.tar.gz

cd pretrain

mkdir inception_v4

mv .../inception_v4.ckpt inception_v4

## arch_vgg16 download vgg model: http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz

tar zxvf vgg_16_2016_08_28.tar.gz

cd pretrain

mkdir vgg

mv .../vgg_16.ckpt vgg

## arch_resnet_v2_50 download resnet_v2_50 model: http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz
wget http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz
tar zxvf resnet_v2_50_2017_04_14.tar.gz

cd pretrain

mkdir resnet_v2

mv .../resnet_v2_50.ckpt resnet_v2

