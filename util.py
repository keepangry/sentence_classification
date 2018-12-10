import numpy as np


def score(y_pred, y_val):
    y_pred = np.array(list(map(lambda x: 0 if x[0] < 0.5 else 1, y_pred)))
    return 1 - sum(list(map(lambda x: abs(x), (y_pred - y_val)))) / len(y_pred)
