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
    model.add(layers.Bidirectional(layers.LSTM(64, dropout=0.1, recurrent_dropout=0.5)))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


