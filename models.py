from datasets import mr_load_data
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# MR base_line
maxlen = 32
max_word_num = 5000
data, labels = mr_load_data(max_word_num=max_word_num)

# avg length: 18.28
data = pad_sequences(data, maxlen=maxlen)

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# shuffle
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

training_samples = 6000
validation_samples = 4000
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]


from keras import layers
from keras.models import Sequential


model = Sequential()
model.add(layers.Embedding(max_word_num, 128))
# 0.95
# model.add(layers.LSTM(64))

# 0.958
# model.add(layers.Bidirectional(layers.LSTM(64)))

# 0.9817
model.add(layers.Bidirectional(layers.LSTM(64, dropout=0.1, recurrent_dropout=0.5)))
model.add(layers.Dense(1, activation='sigmoid'))


from keras import metrics
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_data=(x_val, y_val))


y_pred = model.predict(x_val)
y_pred = np.array(list(map(lambda x: 0 if x[0] < 0.5 else 1, y_pred)))

1 - sum(list(map(lambda x: abs(x), (y_pred - y_val)))) / len(y_pred)