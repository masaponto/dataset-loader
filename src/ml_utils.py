#!/usr/bin/env python

import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler


def cross_validation(estimator, data_set, k=5, scaling=False):
    '''
    find cross_validation accuracy

    estimator must be implemented fit, predict and Inheritance Baseestimator and ClassifierMixin

    '''

    X, y = shuffle(data_set.data, data_set.target)

    n = data_set.data.shape[0]
    m = n // k

    scores = [validation(estimator, X, y, m, index, scaling) for index in range(0, n - (n % k), m)]

    return np.array(scores)


def split_data(X, y, m, index):

    x_test = X[index: index + m]
    y_test = y[index: index + m]

    x_train1 = X[:index]
    y_train1 = y[:index]

    x_train2 = X[index + m:]
    y_train2 = y[index + m:]

    x_train = np.r_[x_train1, x_train2]
    y_train = np.r_[y_train1, y_train2]

    return x_train, y_train, x_test, y_test


def validation(estimator, X, y, m, index, scaling=False):

    x_train, y_train, x_test, y_test = split_data(X, y, m, index)

    if scaling:
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    estimator.fit(x_train, y_train)
    test_score = estimator.score(x_test, y_test)

    return test_score


def validation_with_train(estimator, X, y, m, index):

    x_train, y_train, x_test, y_test = split_data(X, y, m, index)

    estimator.fit(x_train, y_train)
    test_score = estimator.score(x_test, y_test)
    train_score = estimator.score(x_train, y_train)

    return test_score, train_score


def argwrapper(args):
    return args[0](*args[1:])


def mp_cross_validation(estimator, data_set, k=5, p_num=4, scaling=False):
    '''
    Cross-validation with multi processing
    '''

    from multiprocessing import Pool
    from multiprocessing import Process

    assert(isinstance(k, int))
    assert(k > 0)
    assert(isinstance(p_num, int))
    assert(p_num > 0)

    X, y = shuffle(data_set.data, data_set.target)
    n = data_set.data.shape[0]
    m = n // k
    n = n - (n % k)

    p = Pool(p_num)
    func_args = [(validation, estimator, data_set.data, data_set.target, m, index, scaling)
                 for index in range(0, n, m)]
    scores = p.map(argwrapper, func_args)
    p.close()

    return np.array(scores)


def mp_cross_validation_with_train(estimator, data_set, k=5, p_num=4):

    from multiprocessing import Pool
    from multiprocessing import Process

    assert(isinstance(k, int))
    assert(k > 0)
    assert(isinstance(p_num, int))
    assert(p_num > 0)

    X, y = shuffle(data_set.data, data_set.target)
    n = data_set.data.shape[0]
    m = n // k
    n = n - (n % k)

    p = Pool(p_num)
    func_args = [(validation_with_train, estimator, data_set.data, data_set.target, m, index)
                 for index in range(0, n, m)]
    scores = p.map(argwrapper, func_args)
    p.close()

    test_scores = [s[0] for s in scores]
    train_scores = [s[1] for s in scores]

    return np.array(test_scores), np.array(train_scores)


def main():
    from elm import ELM
    from sklearn.preprocessing import normalize
    from sklearn.datasets import fetch_mldata

    data_set = fetch_mldata('australian')

    print(mp_cross_validation(ELM(100), data_set, scaling=True))
    print(cross_validation(ELM(100), data_set, scaling=True))

    data_set.data = normalize(data_set.data)

    print(mp_cross_validation_with_train(ELM(100), data_set))


if __name__ == "__main__":
    main()
