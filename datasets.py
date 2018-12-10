import os
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
BASE_PATH = "/home/yangsen/workspace/sentence_classification/"


class Dataset(object):
    def __init__(self, x_train, y_train, x_val=None, y_val=None, x_test=None, y_test=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test


def texts_to_sequences(texts, max_word_num):
    """
    文本转整数序列
    :param texts:
    :param max_word_num:
    :return:
    """
    tokenizer = Tokenizer(num_words=max_word_num)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    return sequences


def mr_load_data(max_word_num=5000):
    """
    数据集:
    MR: Movie reviews with one sentence per re-
view. Classification involves detecting posi-
tive/negative reviews (Pang and Lee, 2005).

    sentence-polarity-dataset-v1.0

    :param max_word_num:
    :return:
    """
    neg_file = os.path.join(BASE_PATH, 'small_datasets/mr/rt-polarity.neg')
    pos_file = os.path.join(BASE_PATH, 'small_datasets/mr/rt-polarity.pos')
    neg = list(map(str, open(neg_file, 'rb').readlines()))
    pos = list(map(str, open(pos_file, 'rb').readlines()))
    data = texts_to_sequences(neg+pos, max_word_num)
    labels = [0 for i in range(len(neg))] + [1 for i in range(len(pos))]
    return data, labels


def k_fold_split(x, y, k=5):
    assert x.shape[0] == y.shape[0]
    data_size = x.shape[0]
    fold_sample_num = data_size // k
    print(fold_sample_num)
    datasets = []
    for i in range(k):
        x_val = x[i * fold_sample_num: (i+1) * fold_sample_num]
        y_val = y[i * fold_sample_num: (i+1) * fold_sample_num]

        x_train = np.concatenate([
            x[: i * fold_sample_num],
            x[(i+1) * fold_sample_num:],
        ], axis=0)

        y_train = np.concatenate([
            y[: i * fold_sample_num],
            y[(i + 1) * fold_sample_num:],
        ], axis=0)
        datasets.append(Dataset(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val))
    return datasets


if __name__ == "__main__":
    # data, labels = mr_load_data(max_word_num=5000)

    x = np.array([1, 2, 3, 4, 5])
    y = np.array([6, 7, 8, 9, 10])
    datasets = k_fold_split(x, y, k=5)
    print(datasets[0].x_train)
