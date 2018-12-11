from keras import layers
from keras.models import Sequential


def base_embed_lstm_net(max_word_num):
    model = Sequential()
    model.add(layers.Embedding(max_word_num, 128))
    # 0.95
    # model.add(layers.LSTM(64))

    # 0.958
    # model.add(layers.Bidirectional(layers.LSTM(64)))

    # 0.9817
    model.add(layers.Bidirectional(layers.LSTM(64, dropout=0.1, recurrent_dropout=0.5, return_sequences=True)))
    model.add(layers.Bidirectional(layers.LSTM(64, dropout=0.1, recurrent_dropout=0.5)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def base_embed_cnn_lstm_net(max_word_num):
    model = Sequential()
    model.add(layers.Embedding(max_word_num, 128))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv1D(64, 5, padding='valid', activation='relu', strides=1))
    model.add(layers.MaxPooling1D(pool_size=4))
    model.add(layers.LSTM(64))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model
