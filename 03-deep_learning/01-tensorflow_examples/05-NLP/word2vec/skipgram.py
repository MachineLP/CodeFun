#!/usr/bin/env python3
# coding: utf-8
 
 
import collections
import math
import random
import numpy as np
import tensorflow as tf
from load_data import *
 
class SkipGram:
    def __init__(self):
        self.data_index = 0
        self.trainpath = './data/data.txt'
        self.modelpath = './model/skipgram_wordvec.bin'
        self.min_count = 5#最低词频，保留模型中的词表
        self.batch_size = 200 # 每次迭代训练选取的样本数目
        self.embedding_size = 200  # 生成词向量的维度
        self.window_size = 5  # 考虑前后几个词，窗口大小, skipgram中的中心词-上下文pairs数目就是windowsize *2
        self.num_sampled = 100  # 负样本采样.
        self.num_steps = 1000000#定义最大迭代次数，创建并设置默认的session，开始实际训练
        self.dataset = DataLoader().dataset
        self.words = self.read_data(self.dataset)
    # 定义读取数据的函数，并把数据转成列表
    def read_data(self, dataset):
        words = []
        for data in dataset:
            words.extend(data)
        return words
    #创建数据集
    def build_dataset(self, words, min_count):
        # 创建词汇表，过滤低频次词语，这里使用的人是mincount>=5，其余单词认定为Unknown,编号为0,
        # 这一步在gensim提供的wordvector中，采用的是minicount的方法
        #对原words列表中的单词使用字典中的ID进行编号，即将单词转换成整数，储存在data列表中，同时对UNK进行计数
        count = [['UNK', -1]]
        count.extend([item for item in collections.Counter(words).most_common() if item[1] >= min_count])
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0
                unk_count += 1
            data.append(index)
        count[0][1] = unk_count
        # 将dictionary中的数据反转，即可以通过ID找到对应的单词，保存在reversed_dictionary中
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, count, dictionary, reverse_dictionary
 
    #生成训练样本，assert断言：申明其布尔值必须为真的判定，如果发生异常，就表示为假
    def generate_batch(self, batch_size, window_size, data):
        # 该函数根据训练样本中词的顺序抽取形成训练集
        # 这个函数的功能是对数据data中的每个单词，分别与前一个单词和后一个单词生成一个batch，
        # 即[data[1],data[0]]和[data[1],data[2]]，其中当前单词data[1]存在batch中，前后单词存在labels中
        # batch_size:每个批次训练多少样本
        # num_skips: 为每个单词生成多少样本（本次实验是2个），batch_size必须是num_skips的整数倍,这样可以确保由一个目标词汇生成的样本在同一个批次中。
        # window_size:单词最远可以联系的距离（本次实验设为1，即目标单词只能和相邻的两个单词生成样本），2*window_size>=num_skips
        '''
        eg:
        batch, labels = generate_batch(batch_size = 8, num_skips = 2, window_size = 1)
        #Sample data [0, 5241, 3082, 12, 6, 195, 2, 3137, 46, 59] ['UNK', 'anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first', 'used']
        #假设取num_steps为2, window_size为1, batchsize为8
        #batch:[5242, 3084, 12, 6]
        #labels[0, 3082, 5241, 12, 3082, 6, 12, 195]
        print(batch)  [5242 5242 3084 3084   12   12    6    6]，共8维
        print(labels) [[   0] [3082] [  12] [5242] [   6] [3082] [  12] [ 195]]，共8维
        '''
        batch = np.ndarray(shape = (batch_size), dtype = np.int32) #建一个batch大小的数组，保存任意单词
        labels = np.ndarray(shape = (batch_size, 1), dtype = np.int32)#建一个（batch，1）大小的二位数组，保存任意单词前一个或者后一个单词，从而形成一个pair
        span = 2 * window_size + 1 #窗口大小，为3，结构为[ window_size target window_size ][wn-i,wn,wn+i]
        buffer = collections.deque(maxlen = span) #建立一个结构为双向队列的缓冲区，大小不超过3，实际上是为了构造bath以及labels，采用队列的思想
        for _ in range(span):
            buffer.append(data[self.data_index])
            self.data_index = (self.data_index + 1) % len(data)
        #batch_size一定是Num_skips的倍数，保证每个batch_size都能够用完num_skips
        for i in range(batch_size // (window_size*2)):
            target = window_size
            targets_to_avoid = [window_size]
            for j in range(window_size*2):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * window_size*2 + j] = buffer[window_size]
                labels[i * window_size*2 + j, 0] = buffer[target]
            buffer.append(data[self.data_index])
            self.data_index = (self.data_index + 1)%len(data)
 
        return batch, labels
 
    def train_wordvec(self, vocabulary_size, batch_size, embedding_size, window_size, num_sampled, num_steps, data):
        #定义Skip-Gram Word2Vec模型的网络结构
        graph = tf.Graph()
        with graph.as_default():
            #输入数据， 大小为一个batch_size
            train_inputs = tf.placeholder(tf.int32, shape = [batch_size])
            #目标数据，大小为[batch_size]
            train_labels = tf.placeholder(tf.int32, shape = [batch_size, 1])
            #使用cpu进行训练
            with tf.device('/cpu:0'):
                #生成一个vocabulary_size×embedding_size的随机矩阵，为词表中的每个词，随机生成一个embedding size维度大小的向量，
                #词向量矩阵，初始时为均匀随机正态分布，tf.random_uniform((4, 4), minval=low,maxval=high,dtype=tf.float32)))
                #随机初始化一个值于介于-1和1之间的随机数，矩阵大小为词表大小乘以词向量维度
                embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
                #tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素。用于查找对应的wordembedding， ，将输入序列向量化
                #tf.nn.embedding_lookup(params, ids, partition_strategy='mod', name=None, validate_indices=True, max_norm=None)
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)
                #全连接层，Wx+b,设置W大小为，embedding_size×vocabulary_size的权重矩阵，模型内部参数矩阵，初始为截断正太分布
                nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev = 1.0 / math.sqrt(embedding_size)))
                # 全连接层，Wx+b,设置W大小为，vocabulary_size×1的偏置
                nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
            #定义loss，损失函数，tf.reduce_mean求平均值，# 得到NCE损失(负采样得到的损失)
            loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights,# 权重
                                                biases = nce_biases,# 偏差
                                                labels = train_labels,# 输入的标签
                                                inputs = embed, # 输入向量
                                                num_sampled = num_sampled,# 负采样的个数
                                                num_classes = vocabulary_size))# 类别数目
            #定义优化器，使用梯度下降优化算法
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
            #计算每个词向量的模，并进行单位归一化，保留词向量维度
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims = True))
            normalized_embeddings = embeddings / norm
            #初始化模型变量
            init = tf.global_variables_initializer()
 
        #基于构造网络进行训练
        with tf.Session(graph = graph) as session:
            #初始化运行
            init.run()
            #定义平均损失
            average_loss = 0
            #每步进行迭代
            for step in range(num_steps):
                batch_inputs, batch_labels = self.generate_batch(batch_size, window_size, data)
                #feed_dict是一个字典，在字典中需要给出每一个用到的占位符的取值。
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
                #计算每次迭代中的loss
                _, loss_val = session.run([optimizer, loss], feed_dict = feed_dict)
                #计算总loss
                average_loss += loss_val
                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    print("Average loss at step ", step, ":", average_loss)
                    average_loss = 0
            final_embeddings = normalized_embeddings.eval()
 
        return final_embeddings
    #保存embedding文件
    def save_embedding(self, final_embeddings, reverse_dictionary):
        f = open(self.modelpath, 'w+')
        for index, item in enumerate(final_embeddings):
            f.write(reverse_dictionary[index] + '\t' + ','.join([str(vec) for vec in item]) + '\n')
        f.close()
    #训练主函数
    def train(self):
        data, count, dictionary, reverse_dictionary = self.build_dataset(self.words, self.min_count)
        vocabulary_size = len(count)
        final_embeddings = self.train_wordvec(vocabulary_size, self.batch_size, self.embedding_size, self.window_size, self.num_sampled, self.num_steps, data)
        self.save_embedding(final_embeddings, reverse_dictionary)
 
def test():
 
    vector = SkipGram()
    vector.train()
 
test()


