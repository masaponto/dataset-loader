#!/usr/bin/env python

import numpy as np


def cross_validation(estimator, data_set, cv=5):
    '''
    find cross_validation accuracy

    estimator must be implemented fit, predict() and Inheritance Baseestimator and ClassifierMixin

    '''

    from sklearn.utils import shuffle

    X, y = shuffle(data_set.data, data_set.target)

    n = data_set.data.shape[0]
    k = n // cv
    scores = []

    for index in range(0, n - (n % cv), k):
        x_test = X[index: index + k]
        y_test = y[index: index + k]

        x_train1 = X[:index]
        y_train1 = y[:index]

        x_train2 = X[index + k:]
        y_train2 = y[index + k:]

        x_train = np.r_[x_train1, x_train2]
        y_train = np.r_[y_train1, y_train2]

        estimator.fit(x_train, y_train)
        scores.append(estimator.score(x_test, y_test))

    return np.array(scores)


def main():
    from elm import ELM
    from sklearn.preprocessing import normalize
    from sklearn.datasets import fetch_mldata

    data_set = fetch_mldata('australian')
    data_set.data = normalize(data_set.data)

    print(cross_validation(ELM(100), data_set).mean())


if __name__ == "__main__":
    main()
