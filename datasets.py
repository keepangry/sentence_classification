import os
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
BASE_PATH = "/home/yangsen/workspace/sentence_classification/data/"


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
    neg_file = os.path.join(BASE_PATH, 'mr/rt-polarity.neg')
    pos_file = os.path.join(BASE_PATH, 'mr/rt-polarity.pos')
    neg = texts_to_sequences(list(map(str, open(neg_file, 'rb').readlines())), max_word_num)
    pos = texts_to_sequences(list(map(str, open(pos_file, 'rb').readlines())), max_word_num)
    data = neg+pos
    labels = [0 for i in range(len(neg))] + [1 for i in range(len(neg))]
    return data, labels


if __name__ == "__main__":
    data, labels = mr_load_data(max_word_num=5000)
