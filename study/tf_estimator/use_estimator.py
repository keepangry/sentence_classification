import tensorflow as tf
# import keras
# from keras.models import Sequential, Model
# from keras import layers, Input, regularizers
# from keras.preprocessing.sequence import pad_sequences

from tensorflow.python.keras import layers, Input, regularizers
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras

from datasets import mr_load_data
import numpy as np


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def input_function(input_dict, labels=None, shuffle=False):
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x=input_dict,
        y=labels,
        shuffle=shuffle
    )
    return input_fn


def main():
    vocabulary_size = 10000
    maxlen = 24

    model = Sequential()
    model.add(layers.Embedding(vocabulary_size, 64, name="text"))
    model.add(layers.Conv1D(64, 4, padding='valid', activation='relu', strides=1))
    model.add(layers.MaxPooling1D(pool_size=3))
    model.add(layers.LSTM(64))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # # if use keras not tf.keras
    # model = tf.keras.models.Model(model)

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

    estimator_model = tf.keras.estimator.model_to_estimator(keras_model=model)

    data, labels = mr_load_data(max_word_num=vocabulary_size)
    data = pad_sequences(data, padding="pre", maxlen=maxlen)
    labels = np.asarray(labels).reshape(-1, 1)
    print(labels.shape)

    x_train, y_train = data, labels
    input_dict = {"text_input": x_train}
    input_fn = train_input_fn(input_dict, y_train, batch_size=32)
    print(input_fn)
    #
    # estimator_model.train(input_fn=input_fn, steps=10000)
    estimator_model.train(input_fn=input_function(input_dict, y_train), steps=10000)


if __name__ == "__main__":
    main()
