#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/7 12:13 PM
# @Author  : 0@keepangry.com
# @Site    : 
# @File    : textfile2tfrecords.py
# @Software: PyCharm
import tensorflow as tf
import os
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
# from gensim.models import Word2Vec
import re
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

"""
    定长写入tfrecords
"""

sequence_size=24
vecter_size=35
batch_size=128

X = []
Y = []
# fr = open('train_file_20181106.txt')
fr = ['0	652751989	20181105	21.8 36.8 0.592 0.0 0.0 7.0 0 5 1 145 303 1 0 0 1 1 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 19.8 32.8 0.604 0.0 0.0 7.0 388 6 4 507 2738 1 0 0 1 1 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 18.71 22.88 0.818 0.0 0.0 1.0 411 2 2 125 2334 1 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 14.5 15.5 0.935 0.0 0.0 0.0 411 4 4 392 26 1 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 19.8 32.8 0.604 0.0 0.0 7.0 411 1 4 63 1407 1 0 0 1 1 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 27.0 31.0 0.871 2.0 0.0 5.0 82 2 3 187 27 1 0 0 1 1 1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 19.8 32.8 0.604 0.0 0.0 7.0 322 1 4 46 14 1 0 0 1 1 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0']


for line in fr:
    line = line.strip().split('\t')
    seq_feature = np.array(line[3].split(' '), dtype='float32')
    seq_feature = seq_feature.reshape(int(seq_feature.shape[0] / vecter_size), vecter_size)
    X.append(seq_feature)
    Y.append([int(line[0])])

X = pad_sequences(X, maxlen=sequence_size, dtype='float32', padding='pre', truncating='pre', value=0.)
Y = to_categorical(np.array(Y), 2).astype('int64')

X, _, Y, _ = train_test_split(X, Y, test_size=0., random_state=0)
X = X.reshape((X.shape[0], vecter_size*sequence_size))


count = 0
def counter():
    global count
    count += 1
    if count % 1000 == 1:
        print(count)
### 写
tfrecord_writer = tf.python_io.TFRecordWriter('ord_seq.tfrecords')
for i in range(len(Y)):
    x = X[i]
    y = Y[i]
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'feature': tf.train.Feature(float_list=tf.train.FloatList(value=x)),
                # 'feature': tf.train.Feature(bytes_list=tf.train.BytesList(value=x)),
                     'label': tf.train.Feature(int64_list=tf.train.Int64List(value=y)),
                     }
        )
    )

    # frame_feature = list(map(lambda x_i: tf.train.Feature(float_list=tf.train.FloatList(value=x_i)),
    #                          x)
    #                      )
    #
    # example = tf.train.SequenceExample(
    #     context_features=tf.train.Features(feature={
    #                 'label': tf.train.Feature(int64_list=tf.train.Int64List(value=y))}),
    #     sequence_features=tf.train.FeatureLists(feature_list={
    #                 'sequence': tf.train.FeatureList(feature=frame_feature)
    #             })
    # )
    serialized = example.SerializeToString()
    tfrecord_writer.write(serialized)
    counter()
    if count > 20000:
        break
tfrecord_writer.close()


### 读
filename_queue = tf.train.string_input_producer(['ord_seq.tfrecords'], num_epochs=None)
# create a reader from file queue
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
# get feature from serialized example
features = tf.parse_single_example(serialized_example,
                                   features={
                                       # 'feature': tf.FixedLenSequenceFeature(shape=[sequence_size, vecter_size], dtype=tf.float32,
                                       #                                       allow_missing=True, default_value=.0),
                                        'feature': tf.FixedLenFeature(shape=[sequence_size, vecter_size], dtype=tf.float32),
                                       'label': tf.FixedLenFeature([2], tf.int64),
                                        }
                                   )

batch_data = tf.train.batch(tensors=features, batch_size=10, dynamic_pad=True)
result = tf.contrib.learn.run_n(batch_data)
result[0]['feature'].shape
print(result[0]['feature'])
print(result[0]['label'])

feature = features['feature']
# feature = tf.decode_raw(features['feature'], tf.float32)
feature.set_shape([sequence_size, vecter_size])

label = features['label']

