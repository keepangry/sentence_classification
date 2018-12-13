from datasets import mr_load_data, mr_read_files, Dataset, k_fold_split
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
    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='binary_logloss', early_stopping_rounds=10)
    y_pred = gbm.predict(X_test)
    print(y_pred)
    print(scorer.accuracy_score(y_pred, y_test))


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

    for i in range(k):
        dataset = datasets[i]
        print("fold #%s" % i)
        print("train pos rate: %.3f" % (dataset.y_train.sum() / dataset.y_train.shape[0],))
        print("val   pos rate: %.3f" % (dataset.y_val.sum() / dataset.y_val.shape[0],))
        print()

    result = []
    for i in range(k):
        dataset = datasets[i]
        mr_model = mr_base_model(dataset=dataset, vocabulary_size=vocabulary_size, maxlen=maxlen)
        y_pred = mr_model.predict(dataset.x_val)
        acc = score(y_pred, dataset.y_val)
        result.append(acc)
        print("fold #%s, acc:%.4f" % (i, result[i]))

    print(np.mean(result))


if __name__ == "__main__":
    deep_learning_method()
    # shallow_classify_method()
