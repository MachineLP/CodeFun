#!/usr/bin/env python3
from load_data import *
import collections
import math
import numpy as np
from sklearn.decomposition import PCA
 
class WordVector:
    def __init__(self):
        self.dataset = DataLoader().dataset
        self.min_count = 5
        self.window_size = 5
        self.word_demension = 200
        self.embedding_path = 'model/word2doc_wordvec.bin'
    #统计总词数
    def build_word_dict(self):
        words = []
        for data in self.dataset:
            words.extend(data)
        reserved_words = [item for item in collections.Counter(words).most_common() if item[1] >= self.min_count]
        word_dict = {item[0]:item[1] for item in reserved_words}
        return word_dict
 
    #统计词语IDF
    def build_wordidf_dict(self):
        df_dict = {}
        sum_df = len(self.dataset)
        for data in self.dataset:
            for word in set(data):
                if word not in df_dict:
                    df_dict[word] = 1
                else:
                    df_dict[word] += 1
        idf_dict = {word:math.log(sum_df/word_df+1) for word, word_df in df_dict.items()}
        return idf_dict
 
    #统计词语-doc,tfidf
    def build_wordtfidf_dict(self):
        wordidf_dict = self.build_wordidf_dict()
        doctfidf_dict = {}
        for index, data in enumerate(self.dataset):
            doc_words = {item[0]:item[1] for item in [item for item in collections.Counter(data).most_common()]}
            sum_tf = sum(doc_words.values())
            doc_wordtf = {word: word_count/sum_tf for word, word_count in doc_words.items()}
            doc_wordtfidf = {word: word_tf*wordidf_dict[word] for word, word_tf in doc_wordtf.items()}
            doctfidf_dict[index] = doc_wordtfidf
        return doctfidf_dict
 
    #构造词语-文档共现矩阵
    def build_worddoc_matrix(self):
        worddoc_matrix = []
        doctfidf_dict = self.build_wordtfidf_dict()
        word_list = list(self.build_word_dict().keys())
        word_all = len(word_list)
        word_dict = {index : word for index, word in enumerate(word_list)}
        count = 0
        for word_id, word in word_dict.items():
            tmp = []
            for doc_index, word_dict in doctfidf_dict.items():
                weight = word_dict.get(word, 0)
                tmp.append(weight)
            count += 1
            print(count, '/', word_all)
            worddoc_matrix.append(tmp)
        worddoc_matrix = np.array(worddoc_matrix)
        return worddoc_matrix
 
    #使用PCA进行降维
    def low_dimension(self):
        worddoc_matrix = self.build_worddoc_matrix()
        pca = PCA(n_components=self.word_demension)
        low_embedding = pca.fit_transform(worddoc_matrix)
        return low_embedding
 
    #保存模型
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


