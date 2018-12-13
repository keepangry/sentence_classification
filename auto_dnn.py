"""

"""
from datasets import mr_load_data, k_fold_split
from keras.preprocessing.sequence import pad_sequences
from util import score
import numpy as np
import keras
from keras import layers, Input, regularizers
from keras.models import Sequential, Model
from util import gene_grid_search_candidates, time_string
import global_config


def random_search_generater():
    while True:
        vocabulary_size = int(np.random.uniform(0.5, 1) * 10000)
        time_steps = int(np.random.uniform(1.8, 3.3) * 10)
        embeding_units = int(np.random.uniform(64, 256))
        channels = int(np.random.uniform(3, 7))
        conv1d_filters = int(np.random.uniform(16, 64))
        dense_units = int(np.random.uniform(32, 128))
        dense_l2 = np.round(np.random.uniform(0.001, 0.01), 4)
        dense_dropout = np.round(np.random.uniform(0.05, 0.3), 4)

        params = Params(*[vocabulary_size, time_steps, embeding_units, channels, conv1d_filters,
                          dense_units, dense_l2, dense_dropout])
        yield params


def grid_search_candidates():
    vocabulary_size = [5000, 10000]
    time_steps = [20, 30]

    embeding_units = [64, 128]
    channels = [3, 4, 5]
    conv1d_filters = [16, 32, 64]
    dense_units = [64, 32]
    dense_l2 = [0.001]
    dense_dropout = [0.1]

    grid = [vocabulary_size, time_steps, embeding_units, channels, conv1d_filters,
            dense_units, dense_l2, dense_dropout]
    candis = gene_grid_search_candidates(*grid)
    candidates = []
    for candi in candis:
        candidates.append(Params(*candi))
    return candidates


def hyper_parameter_search(max_search_num=100, method="random"):
    """

    :param max_search_num:
    :param method: random / grid
    :return:
    """
    predict_result = []
    log_file_name = 'logs/auto_dnn_%s.txt' % time_string()

    def run(params):
        maxlen = params.time_steps
        vocabulary_size = params.vocabulary_size

        data, labels = mr_load_data(max_word_num=vocabulary_size)
        data = pad_sequences(data, padding="pre", maxlen=maxlen)
        labels = np.asarray(labels)
        indices = np.arange(data.shape[0])
        np.random.seed(2)
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
        k = 10
        datasets = k_fold_split(x=data, y=labels, k=k)

        print(params)
        accs = []
        for i in range(k):
            dataset = datasets[i]
            model = hyper_model(dataset=dataset, params=params)
            y_pred = model.predict(dataset.x_val)
            acc = score(y_pred, dataset.y_val)
            accs.append(acc)
            print("fold #%s, acc:%.4f" % (i, accs[i]))
        mean_acc = np.mean(accs)
        print("mean_acc: %s" % mean_acc)
        print("--"*20)
        with open(log_file_name, 'a') as fw:
            fw.write(str(params))
            fw.write("\nmean_acc: %.4f" % mean_acc)
            fw.write("%s\n" % time_string())
            fw.write("--"*20 + "\n")
            fw.write("\n")
        predict_result.append((mean_acc, params))

    if method == "grid":
        candidates = grid_search_candidates()
        list(map(run, candidates[:max_search_num]))
    elif method == "random":
        generater = random_search_generater()
        for i in range(max_search_num):
            params = next(generater)
            run(params)


class Params(object):
    def __init__(self, vocabulary_size, time_steps, embeding_units, channels,
                 conv1d_filters, dense_units, dense_l2, dense_dropout):
        self.vocabulary_size = vocabulary_size
        self.time_steps = time_steps

        self.embeding_units = embeding_units

        self.channels = channels
        self.conv1d_filters = conv1d_filters

        self.dense_units = dense_units
        self.dense_l2 = dense_l2
        self.dense_dropout = dense_dropout

    def __str__(self):
        return '\n'.join(['%-20s : %s' % item for item in self.__dict__.items()])


def hyper_model(dataset, params):
    """
    超参、部分结构搜索，基于multi-channel-conv方法，三层网络。
    大部分情况下三层网络能力已经足够


    :param dataset:
    :param params:
    :return:
    """
    text_input = Input(shape=(params.time_steps, ), dtype='int32', name='text')
    embedded_text = layers.Embedding(params.vocabulary_size, params.embeding_units)(text_input)

    channels = []
    for kernel_size in range(2, 2+params.channels):
        channel = layers.Conv1D(params.conv1d_filters, kernel_size, activation='relu')(embedded_text)
        channel = layers.MaxPool1D(params.time_steps-kernel_size+1)(channel)
        channels.append(channel)

    concatenated = layers.concatenate(channels, axis=-1)
    concatenated = layers.Flatten()(concatenated)
    output = layers.Dense(params.dense_units, activation='relu',
                          kernel_regularizer=regularizers.l2(params.dense_l2))(concatenated)
    output = layers.Dropout(params.dense_dropout)(output)
    output = layers.Dense(1, activation='sigmoid')(output)
    model = Model(text_input, output)

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            mode='auto'
        )
    ]
    history = model.fit(dataset.x_train, dataset.y_train,
                        epochs=30,
                        batch_size=128,
                        validation_data=(dataset.x_val, dataset.y_val),
                        callbacks=callbacks,
                        verbose=0)
    return model


if __name__ == "__main__":
    # grid_search_candidates()
    hyper_parameter_search()

    # generater = random_search_generater()
    # for i in range(10):
    #     params = next(generater)
    #     print(params)
