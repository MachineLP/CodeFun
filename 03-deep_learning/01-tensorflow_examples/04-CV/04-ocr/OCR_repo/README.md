## 代码目录
```
|---- config.py                              # 工程参数配置
|---- ocr_lib                                # 模型、数据生成
|    |---- crnn.py                           # 选用不同模型
|    |---- vgg16.py 
|    |---- vgg19.py 
|    |---- Xception.py 
|    |---- densenet.py
|    |---- data_generator.py
|    |---- data_load.py
|    |---- gen_train_val_data.py
|    |---- random_gen.py
|    |---- fonts                             # 字体
|---- train_net_online.py 
|---- test_net_online.py  
|---- requirements.txt   
```

- 代码是基于keras实现的ocr.


## 主要内容包含：
0. [相关介绍](#相关介绍)
0. [代码使用](#代码使用)
0. [联系](#联系)

## 相关介绍
- 如何进行ocr？ 主要方式有：

> (1) 文本检测：yolo v3、 ctpn、psenet等，主要是基于检测、分割的方案。

> (2) 文本识别：cnn+ctc、crnn+ctc等。


## 代码使用

1. OCR模型训练 (训练时间24小时以上,config.py的参数保持默认、或者修改)
```
$ python train_net_online.py 
```

2.OCR模型测试
```
$ python test_net_online.py
```

3. OCR模型优化

> (1) 选用不同的模型cnn、 crnn等。

> (2) 文字生成器选用[text_renderer](https://github.com/MachineLP/text_renderer)。

> (3) 还可可参考：[chineseocr](https://github.com/chineseocr/chineseocr)、[crnn.pytorch](https://github.com/MachineLP/crnn.pytorch)、[chinese_ocr](https://github.com/MachineLP/chinese_ocr)、[awesome-ocr](https://github.com/MachineLP/awesome-ocr)、[PSENet](https://github.com/MachineLP/tensorflow_PSENet)、[text-detection-ctpn](https://github.com/eragonruan/text-detection-ctpn)、[keras_ocr](https://github.com/MachineLP/keras_ocr)、[keras_PSENet](https://github.com/xiaomaxiao/PSENET)。

>（4）单字的检测和识别可以参考：[CPS-OCR-Engine](https://github.com/AstarLight/CPS-OCR-Engine)。

测试结果：
![image](https://github.com/MachineLP/Shows/blob/master/ocr.png)
```
gt>>>>>>> 戒敞古逊罪化综波早洼
pred>>>>> 戒敞古逊罪化综波早洼
gt>>>>>>> 枫矢浙o膝犁苹嫉轨徊
pred>>>>> 枫矢浙o膝犁苹嫉轨徊
gt>>>>>>> 壤驾橘荤僻境追闸择葱
pred>>>>> 壤驾橘荤僻境追闸择葱
gt>>>>>>> 光8旧亏入书丁寸乎}达正
pred>>>>> 光8旧亏入书丁寸乎}达正
gt>>>>>>> 年降姊雕吝奇癣盹印E彪帅
pred>>>>> 年降姊雕吝奇癣盹印E彪
gt>>>>>>> P鞍笼淆喝文姊籽举陨
pred>>>>> P鞍笼淆喝文姊籽举陨
gt>>>>>>> 惩沙掠巧博苗班扣答刽豌樱
pred>>>>> 惩沙掠巧博苗班扣答刽豌樱
gt>>>>>>> 突喜针垄麸汹亏废妒件扯燃
pred>>>>> 突喜针垄麸汹亏废妒件扯燃
gt>>>>>>> 指限蝌艺依铺蔓崩走胡
pred>>>>> 指限蝌艺依铺蔓崩走胡
gt>>>>>>> 支袱葬栖皮这庞法毫穴赫颇
pred>>>>> 支袱葬栖皮这庞法毫穴赫颇
gt>>>>>>> 敞换啊兽倔论涉究搓叔歧沫
pred>>>>> 敞换啊兽倔论涉究搓叔歧沫
gt>>>>>>> 去斥巨化勾]加g风矛召支
pred>>>>> 去斥巨化勾]加g风矛召支
gt>>>>>>> 宁S市扬当a边成计订
pred>>>>> 宁S市扬当a边成计订
gt>>>>>>> 古闪瓦丙迈见礼4旧女凤肉
pred>>>>> 古闪瓦丙迈见礼4旧女凤肉
gt>>>>>>> 捶儿丢覆这抬蔗抢沫肾
pred>>>>> 捶儿丢覆这抬蔗抢沫肾
```

## 联系

如何有问题或者建议请联系wechat: lp9628    (需要生成接口、.pb模型可联系)



### 为了更快的看到效果，只给了两个颜色：random_gen.py
### 随机生成背景颜色
```python
def randon_gen_bg_color():
    # 图片背景
    '''
    R = random.randrange(0,255,15)
    G = random.randrange(0,255,15)
    B = random.randrange(0,255,15)
    img_color_list = [ (R, G, B) ]'''
    img_color_list = [ (0,0,0), (255,255,255)] #, (255,0,0), (0,255,0), (0,0,255), (0,255,255), (255,0,255), (255,255,0),  (128,128,128), (128,0,0), (0,128,0), (0,0,128), (0,128,128), (128,0,128), (128,128,0) ]
    img_bg_color = random.randint( 0, len(img_color_list)-1 )
    return img_color_list[img_bg_color]

# 随机生成文本颜色
def random_gen_text_color():
    # 文字颜色
    '''
    R = random.randrange(0,255,15)
    G = random.randrange(0,255,15)
    B = random.randrange(0,255,15)
    text_color_list = [ (R, G, B) ]'''
    text_color_list = [ (0,0,0), (255,255,255)] #, (255,0,0), (0,255,0), (0,0,255), (0,255,255), (255,0,255), (255,255,0),  (128,128,128), (128,0,0), (0,128,0), (0,0,128), (0,128,128), (128,0,128), (128,128,0) ]
    text_color = random.randint( 0, len(text_color_list)-1 )
    return text_color_list[text_color]

# 在样本生成的时候没有选用图片作为背景： data_generator.py
flag = np.random.choice([False]) # True


# 随机的调整宽高, 以最大长度初始化
img_size_width = 50 * n_len + 40 + randon_img_width
```

