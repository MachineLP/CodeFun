#!/usr/bin/env python3
# coding: utf-8
 
from load_data import *
import collections
from sklearn.decomposition import PCA
 
class WordVector:
    def __init__(self):
        self.dataset = DataLoader().dataset[:1000]
        self.min_count = 5
        self.window_size = 5
        self.word_demension = 200
        self.embedding_path = 'model/word2word_wordvec.bin'
    #统计总词数
    def build_word_dict(self):
        words = []
        for data in self.dataset:
            words.extend(data)
        reserved_words = [item for item in collections.Counter(words).most_common() if item[1] >= self.min_count]
        word_dict = {item[0]:item[1] for item in reserved_words}
        return word_dict
    #构造上下文窗口
    def build_word2word_dict(self):
        word2word_dict = {}
        for data_index, data in enumerate(self.dataset):
            contexts = []
            for index in range(len(data)):
                if index < self.window_size:
                    left = data[:index]
                else:
                    left = data[index - self.window_size: index]
                if index + self.window_size > len(data):
                    right = data[index + 1:]
                else:
                    right = data[index + 1: index + self.window_size + 1]
                context = left + [data[index]] + right
                for word in context:
                    if word not in word2word_dict:
                        word2word_dict[word] = {}
                    else:
                        for co_word in context:
                            if co_word != word:
                                if co_word not in word2word_dict[word]:
                                    word2word_dict[word][co_word] = 1
                                else:
                                    word2word_dict[word][co_word] += 1
            print(data_index)
        return word2word_dict
 
    #构造词词共现矩阵
    def build_word2word_matrix(self):
        word2word_dict = self.build_word2word_dict()
        word_dict = self.build_word_dict()
        word_list = list(word_dict)
        word2word_matrix = []
        words_all = len(word_list)
        count = 0
        for word1 in word_list:
            count += 1
            print(count,'/',words_all)
            tmp = []
            sum_tf = sum(word2word_dict[word1].values())
            for word2 in word_list:
                weight = word2word_dict[word1].get(word2, 0)/sum_tf
                tmp.append(weight)
            word2word_matrix.append(tmp)
 
        return word2word_matrix
 
    # 使用PCA进行降维
    def low_dimension(self):
        worddoc_matrix = self.build_word2word_matrix()
        pca = PCA(n_components=self.word_demension)
        low_embedding = pca.fit_transform(worddoc_matrix)
        return low_embedding
 
    # 保存模型
    def train_embedding(self):
        print('training.....')
        word_list = list(self.build_word_dict().keys())
        word_dict = {index: word for index, word in enumerate(word_list)}
        word_embedding_dict = {index: embedding for index, embedding in enumerate(self.low_dimension())}
        print('saving models.....')
        with open(self.embedding_path, 'w+') as f:
            for word_index, word_embedding in word_embedding_dict.items():
                word_word = word_dict[word_index]
                word_embedding = [str(item) for item in word_embedding]
                f.write(word_word + '\t' + ','.join(word_embedding) + '\n')
        f.close()
        print('done.....')
 
vec = WordVector()
vec.train_embedding()


