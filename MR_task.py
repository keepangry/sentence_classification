from datasets import mr_load_data, Dataset, k_fold_split
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from util import score
from models import mr_base_model

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
np.random.seed(1)
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

k = 5
datasets = k_fold_split(x=data, y=labels, k=k)

result = []
for i in range(k):
    dataset = datasets[i]
    mr_model = mr_base_model(dataset=dataset, max_word_num=max_word_num)
    y_pred = mr_model.predict(dataset.x_val)
    acc = score(y_pred, dataset.y_val)
    result.append(acc)
    print("fold #%s, acc:%.4f" % (i, result[i]))

print(np.mean(result))
