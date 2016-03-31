#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path
import csv
import numpy as np
from sklearn.datasets.base import Bunch


def load_qsar():
    path = os.path.expanduser('~/Data/')

    try:
        with open(path + 'qsar/biodeg.csv') as file:
            reader = csv.reader(file, delimiter=';',)
            xs, ys = [], []
            for row in reader:
                y = -1 if row[-1] == 'NRB' else 1
                ys.append(y)
                row = [float(x) for x in row[:-1]]
                xs.append(row)

        return Bunch(data=np.array(xs),
                    target=np.array(ys),
                    feature_names='qsar',
                    DESCR='no')

    except EnvironmentError:
        print('Data file not found in ' + path)


def main():
    data = load_qsar()

if __name__ == "__main__":
    main()
