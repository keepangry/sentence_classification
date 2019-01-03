#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/30 3:49 PM
# @Author  : yangsen07@meituan.com
# @Site    : 
# @File    : prepare.py
# @Software: PyCharm

# 句子用onehot表示，然后进行dense

# rnn encoder decoder
import pandas as pd
import re
import jieba
from sklearn.feature_extraction.text import CountVectorizer
import keras
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras import regularizers
from keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np




comment_df = pd.read_csv('1024-1029.txt', delimiter='\t')
comment_df['comment_formated'] = comment_df['comment_content'].apply(lambda x: re.sub('\s+', ' ', x))
comment_content = comment_df['comment_content'].values

sample_n = 100000

comment_content = comment_content[:sample_n]

count = 0
sentence_vec = []
for content in comment_content:
    seg = list(jieba.cut(content))
    if len(seg) < 8:
        continue
    sentence_vec.append(' '.join(seg))
    count += 1
    if count % 1000 == 0:
        print(count)


# BOW
vectorizer = CountVectorizer(min_df=10, max_df=10000, binary=True)
X = vectorizer.fit_transform(sentence_vec)
print(len(vectorizer.get_feature_names()))

X = X.toarray()

dim = len(vectorizer.get_feature_names())


model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(dim,)))
model.add(Dense(128, activation='relu', name='sentence_vector'))
model.add(Dense(256, activation='relu'))
model.add(Dense(dim, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.fit(X, X, epochs=5, batch_size=128, verbose=1)

model.save('model.h5')


# 取某一层的输出为输出新建为model，采用函数模型
dense1_layer_model = Model(inputs=model.input, outputs=model.get_layer('sentence_vector').output)


# 测试数据
n = 1000
test = sentence_vec[:1000]
test_vec = vectorizer.transform(test).toarray()

result = dense1_layer_model.predict(test_vec)
print(result.shape)


def similarity(a, b):
    test_vec = vectorizer.transform([a, b]).toarray()
    result = dense1_layer_model.predict(test_vec)
    s = cosine_similarity(result[0].reshape(1, -1), result[1].reshape(1, -1))[0][0]
    return s, np.sum(test_vec, axis=1)


for i in range(n):
    for j in range(i+1, n, 1):
        s, cout = similarity(test[i], test[j])
        if s > 0.9 and cout[0] > 2 and cout[1] > 2:
            print(test[i])
            print(test[j])
            print(s, cout)
            print("==="*30)


