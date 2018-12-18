from datasets import mr_load_data, mr_read_files, Dataset, k_fold_split, texts_to_sequences
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from util import score
from models import mr_base_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import scorer
from sklearn.metrics import fbeta_score, make_scorer
import lightgbm as lgb
import global_config
from keras import layers, Input, regularizers
from keras.models import Sequential, Model
import keras


def shallow_classify_method():
    """
    tf-idf特征进行分类
    :return:
    """
    vocabulary_size = 5000
    data, labels = mr_read_files()

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=2)
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=vocabulary_size,
                                 min_df=2, stop_words='english',
                                 use_idf=True)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # svm 0.7018
    clf = LinearSVC(C=10, max_iter=2000, verbose=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(y_pred)
    print(scorer.accuracy_score(y_pred, y_test))

    # gbm 0.6929
    gbm = lgb.LGBMClassifier(objective='binary', n_estimators=200, learning_rate=0.3, max_depth=4)
    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='binary_logloss', early_stopping_rounds=10, verbose=0)
    y_pred = gbm.predict(X_test)
    print(y_pred)
    print(scorer.accuracy_score(y_pred, y_test))


def wide_deep_method():
    """

    :return:
    """

    data, labels = mr_read_files()
    # shuffle
    indices = np.arange(data.shape[0])
    np.random.seed(2)
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    train_size = int(data.shape[0] * 0.8)
    X_train, X_test, y_train, y_test = data[:train_size], data[train_size:], labels[:train_size], labels[train_size:]

    # wide feature
    wide_feature_size = 200
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=wide_feature_size,
                                 min_df=2, stop_words='english',
                                 use_idf=True)
    X_train_w = vectorizer.fit_transform(X_train)
    X_test_w = vectorizer.transform(X_test)

    # deep input data
    vocabulary_size = 10000
    time_steps = 30
    X_d = texts_to_sequences(data, vocabulary_size)
    X_d = pad_sequences(X_d, padding="pre", maxlen=time_steps)
    X_train_d, X_test_d = X_d[:train_size], X_d[train_size:]

    # wide & feature
    feature_input = Input(shape=(wide_feature_size,), name="wide")
    wide_dense = layers.Dense(32, activation='relu')(feature_input)

    # deep & text
    text_input = Input(shape=(time_steps, ), dtype='int32', name='deep')
    embedded_text = layers.Embedding(vocabulary_size, 128)(text_input)
    channels = []
    for kernel_size in range(3, 7):
        channel = layers.Conv1D(32, kernel_size, activation='relu')(embedded_text)
        channel = layers.MaxPool1D(time_steps-kernel_size+1)(channel)
        channels.append(channel)
    cnn_concatenated = layers.concatenate(channels, axis=-1)
    cnn_concatenated = layers.Flatten()(cnn_concatenated)

    # wide & deep
    concatenated = layers.concatenate([cnn_concatenated, wide_dense], axis=-1)
    output = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(concatenated)
    output = layers.Dense(1, activation='sigmoid')(output)

    model = Model([text_input, feature_input], output)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='auto'
        )
    ]
    model.summary()
    history = model.fit([X_train_d, X_train_w], y_train,
                        epochs=50,
                        batch_size=128,
                        validation_data=([X_test_d, X_test_w], y_test),
                        callbacks=callbacks,
                        verbose=2)
    return 0


def deep_learning_method():
    # MR base_line
    maxlen = 30
    vocabulary_size = 10000
    data, labels = mr_load_data(max_word_num=vocabulary_size)

    # avg length: 18.28
    # padding pre好于post
    data = pad_sequences(data, padding="pre", maxlen=maxlen)
    labels = np.asarray(labels)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # shuffle
    indices = np.arange(data.shape[0])
    np.random.seed(2)
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    k = 10
    datasets = k_fold_split(x=data, y=labels, k=k)

    # for i in range(k):
    #     dataset = datasets[i]
    #     print("fold #%s" % i)
    #     print("train pos rate: %.3f" % (dataset.y_train.sum() / dataset.y_train.shape[0],))
    #     print("val   pos rate: %.3f" % (dataset.y_val.sum() / dataset.y_val.shape[0],))
    #     print()

    result = []
    for i in range(k):
        dataset = datasets[i]
        print(dataset.x_train.shape)
        # exit()
        mr_model, history = mr_base_model(dataset=dataset, vocabulary_size=vocabulary_size, maxlen=maxlen,
                                          # method="base_multi_channel_net")
                                          method="base_attention")
        best_iter = np.argmax(history.history['val_acc'])
        print("fold #%s, best_iter: %s,  acc:%.4f" % (i, best_iter, history.history['val_acc'][best_iter]))
        # y_pred = mr_model.predict(dataset.x_val)
        # acc = score(y_pred, dataset.y_val)
        result.append(history.history['val_acc'][best_iter])

    print(np.mean(result))


if __name__ == "__main__":
    deep_learning_method()
    # wide_deep_method()
    # shallow_classify_method()
